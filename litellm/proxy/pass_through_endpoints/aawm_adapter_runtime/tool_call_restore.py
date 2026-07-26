"""Wave 6A extraction: tool_call_restore adapted-tool-call restoration.

Behavior-preserving extraction from llm_passthrough_endpoints.py.
Do not import llm_passthrough_endpoints at module scope.
"""

from __future__ import annotations

import codecs
import json
from typing import Any, Optional

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Host-global functions (bound via install())
    def _advertised_custom_tool_function_adapter_names(
        request_body: Optional[dict[str, Any]],
        *,
        adapter_model: str,
    ) -> set[str]: ...

    def _normalize_low_cardinality_tag_value(value: Any) -> Optional[str]: ...

    def _parse_adapted_custom_tool_function_arguments(
        arguments: Any,
    ) -> tuple[Optional[str], Optional[str]]: ...

    def _get_namespace_tool_function_adapter_names_for_model(
        model: Any,
    ) -> dict[str, set[str]]: ...

    def _adapt_codex_namespace_tool_definitions(
        tools: Any,
        *,
        adapter_names: dict[str, set[str]],
    ) -> tuple[Optional[list[Any]], list[dict[str, Any]], list[dict[str, Any]]]: ...

    def _build_failed_responses_diagnostic(
        *,
        response_body: dict[str, Any],
        adapter: str,
        adapter_model: str,
        stream_event_summaries: Optional[list[dict[str, Any]]] = None,
    ) -> dict[str, Any]: ...

    def _responses_repaired_output_item_id(
        item: dict[str, Any],
        index: int,
    ) -> str: ...

from types import FunctionType

from fastapi.responses import StreamingResponse

from litellm.proxy._types import ProxyException


_HOST_FUNCTION_NAMES = (
    "_restore_adapted_custom_tool_calls_in_response_body",
    "_advertised_namespace_tool_function_adapter_map",
    "_restore_adapted_namespace_tool_call_item",
    "_restore_adapted_namespace_tool_calls_in_response_body",
    "_adapted_custom_tool_stream_state_keys",
    "_remember_adapted_custom_tool_stream_state",
    "_get_adapted_custom_tool_stream_state",
    "_restore_adapted_custom_tool_calls_in_stream_event_payload",
    "_restore_adapted_custom_tool_calls_in_sse_event_block",
    "_restore_adapted_custom_tool_calls_in_streaming_response",
    "_restore_adapted_namespace_tool_calls_in_stream_event_payload",
    "_restore_adapted_namespace_tool_calls_in_sse_event_block",
    "_restore_adapted_namespace_tool_calls_in_streaming_response",
    "_raise_codex_auto_agent_malformed_adapted_custom_tool_call",
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


# ── Extracted functions ─────────────────────────────────────────────

def _restore_adapted_custom_tool_calls_in_response_body(
    response_body: dict[str, Any],
    *,
    request_body: Optional[dict[str, Any]],
    adapter_model: str,
) -> tuple[dict[str, Any], int, Optional[dict[str, Any]]]:
    adapted_names = _advertised_custom_tool_function_adapter_names(  # noqa: F821
        request_body,
        adapter_model=adapter_model,
    )
    if not adapted_names:
        return response_body, 0, None

    output = response_body.get("output")
    if not isinstance(output, list):
        return response_body, 0, None

    restored_output: list[Any] = []
    restored_count = 0
    for index, item in enumerate(output):
        if not isinstance(item, dict) or item.get("type") != "function_call":
            restored_output.append(item)
            continue

        item_name = _normalize_low_cardinality_tag_value(item.get("name"))  # noqa: F821
        if item_name not in adapted_names:
            restored_output.append(item)
            continue

        raw_input, error_reason = _parse_adapted_custom_tool_function_arguments(item.get("arguments"))  # noqa: F821
        if error_reason is not None or raw_input is None:
            return (
                response_body,
                0,
                {
                    "name": item_name,
                    "output_index": index,
                    "reason": error_reason or "missing_input",
                },
            )

        restored_item = dict(item)
        restored_item["type"] = "custom_tool_call"
        restored_item["input"] = raw_input
        restored_item.setdefault("status", "completed")
        restored_item.pop("arguments", None)
        restored_output.append(restored_item)
        restored_count += 1

    if restored_count == 0:
        return response_body, 0, None

    restored_body = dict(response_body)
    restored_body["output"] = restored_output
    return restored_body, restored_count, None


def _advertised_namespace_tool_function_adapter_map(
    request_body: Optional[dict[str, Any]],
    *,
    adapter_model: str,
) -> dict[str, str]:
    if not isinstance(request_body, dict):
        return {}

    adapter_names = _get_namespace_tool_function_adapter_names_for_model(adapter_model)  # noqa: F821
    if not adapter_names:
        return {}

    _, adapted_tools, _ = _adapt_codex_namespace_tool_definitions(  # noqa: F821
        request_body.get("tools"),
        adapter_names=adapter_names,
    )
    return {
        str(item["name"]): str(item["namespace"])
        for item in adapted_tools
        if isinstance(item.get("name"), str) and isinstance(item.get("namespace"), str)
    }


def _restore_adapted_namespace_tool_call_item(
    item: Any,
    *,
    namespace_by_name: dict[str, str],
) -> tuple[Any, int]:
    if not isinstance(item, dict) or item.get("type") != "function_call":
        return item, 0

    item_name = _normalize_low_cardinality_tag_value(item.get("name"))  # noqa: F821
    namespace = namespace_by_name.get(item_name or "")
    if namespace is None or item.get("namespace") is not None:
        return item, 0

    restored_item = dict(item)
    restored_item["namespace"] = namespace
    return restored_item, 1


def _restore_adapted_namespace_tool_calls_in_response_body(
    response_body: dict[str, Any],
    *,
    request_body: Optional[dict[str, Any]],
    adapter_model: str,
) -> tuple[dict[str, Any], int]:
    namespace_by_name = _advertised_namespace_tool_function_adapter_map(
        request_body,
        adapter_model=adapter_model,
    )
    if not namespace_by_name:
        return response_body, 0

    output = response_body.get("output")
    if not isinstance(output, list):
        return response_body, 0

    restored_output: list[Any] = []
    restored_count = 0
    for item in output:
        restored_item, item_restored_count = _restore_adapted_namespace_tool_call_item(
            item,
            namespace_by_name=namespace_by_name,
        )
        restored_output.append(restored_item)
        restored_count += item_restored_count

    if restored_count == 0:
        return response_body, 0

    restored_body = dict(response_body)
    restored_body["output"] = restored_output
    return restored_body, restored_count


def _adapted_custom_tool_stream_state_keys(
    event_payload: dict[str, Any],
    *,
    item: Any = None,
) -> list[str]:
    keys: list[str] = []
    for source in (item, event_payload):
        if not isinstance(source, dict):
            continue
        for field in ("call_id", "id", "item_id"):
            value = source.get(field)
            if isinstance(value, str) and value.strip():
                keys.append(f"id:{value.strip()}")
        output_index = source.get("output_index")
        if isinstance(output_index, int):
            keys.append(f"output:{output_index}")
    return list(dict.fromkeys(keys))


def _remember_adapted_custom_tool_stream_state(
    state_by_key: dict[str, dict[str, Any]],
    *,
    event_payload: dict[str, Any],
    item: dict[str, Any],
) -> dict[str, Any]:
    state = {
        "call_id": item.get("call_id") or item.get("id"),
        "name": item.get("name"),
        "arguments": "",
    }
    for key in _adapted_custom_tool_stream_state_keys(
        event_payload,
        item=item,
    ):
        state_by_key[key] = state
    return state


def _get_adapted_custom_tool_stream_state(
    state_by_key: dict[str, dict[str, Any]],
    event_payload: dict[str, Any],
) -> Optional[dict[str, Any]]:
    for key in _adapted_custom_tool_stream_state_keys(event_payload):
        state = state_by_key.get(key)
        if isinstance(state, dict):
            return state
    return None


def _restore_adapted_custom_tool_calls_in_stream_event_payload(
    event_payload: dict[str, Any],
    *,
    request_body: Optional[dict[str, Any]],
    adapter_model: str,
    adapted_names: set[str],
    state_by_key: dict[str, dict[str, Any]],
) -> tuple[Optional[dict[str, Any]], int]:
    event_type = event_payload.get("type")
    item = event_payload.get("item")

    if event_type == "response.output_item.added" and isinstance(item, dict) and item.get("type") == "function_call":
        item_name = _normalize_low_cardinality_tag_value(item.get("name"))  # noqa: F821
        if item_name in adapted_names:
            _remember_adapted_custom_tool_stream_state(
                state_by_key,
                event_payload=event_payload,
                item=item,
            )
            restored_item = dict(item)
            restored_item["type"] = "custom_tool_call"
            restored_item["input"] = ""
            restored_item.pop("arguments", None)
            restored_payload = dict(event_payload)
            restored_payload["item"] = restored_item
            return restored_payload, 1

    state = _get_adapted_custom_tool_stream_state(
        state_by_key,
        event_payload,
    )
    if event_type == "response.function_call_arguments.delta" and state is not None:
        delta = event_payload.get("delta")
        if isinstance(delta, str):
            state["arguments"] = f"{state.get('arguments') or ''}{delta}"
        return None, 1

    if event_type == "response.function_call_arguments.done" and state is not None:
        arguments = event_payload.get("arguments")
        if not isinstance(arguments, str):
            arguments = str(state.get("arguments") or "")
        raw_input, error_reason = _parse_adapted_custom_tool_function_arguments(arguments)  # noqa: F821
        if error_reason is None and raw_input is not None:
            restored_payload = dict(event_payload)
            restored_payload["type"] = "response.custom_tool_call_input.done"
            restored_payload["input"] = raw_input
            restored_payload.pop("arguments", None)
            return restored_payload, 1

    if event_type == "response.output_item.done" and isinstance(item, dict):
        restored_body, restored_count, adapter_error = _restore_adapted_custom_tool_calls_in_response_body(
            {"output": [item]},
            request_body=request_body,
            adapter_model=adapter_model,
        )
        if restored_count and adapter_error is None:
            restored_payload = dict(event_payload)
            restored_payload["item"] = restored_body["output"][0]
            return restored_payload, restored_count

    response_body = event_payload.get("response")
    if isinstance(response_body, dict):
        restored_body, restored_count, adapter_error = _restore_adapted_custom_tool_calls_in_response_body(
            response_body,
            request_body=request_body,
            adapter_model=adapter_model,
        )
        if restored_count and adapter_error is None:
            restored_payload = dict(event_payload)
            restored_payload["response"] = restored_body
            return restored_payload, restored_count

    return event_payload, 0


def _restore_adapted_custom_tool_calls_in_sse_event_block(
    event_block: str,
    *,
    request_body: Optional[dict[str, Any]],
    adapter_model: str,
    adapted_names: set[str],
    state_by_key: dict[str, dict[str, Any]],
) -> tuple[Optional[str], int]:
    lines = event_block.splitlines()
    data_line_indexes = [index for index, line in enumerate(lines) if line.startswith("data:")]
    if not data_line_indexes:
        return event_block, 0

    raw_data = "\n".join(lines[index].removeprefix("data:").lstrip(" ") for index in data_line_indexes)
    if not raw_data or raw_data == "[DONE]":
        return event_block, 0
    try:
        event_payload = json.loads(raw_data)
    except Exception:
        return event_block, 0
    if not isinstance(event_payload, dict):
        return event_block, 0

    restored_payload, restored_count = _restore_adapted_custom_tool_calls_in_stream_event_payload(
        event_payload,
        request_body=request_body,
        adapter_model=adapter_model,
        adapted_names=adapted_names,
        state_by_key=state_by_key,
    )
    if restored_payload is None:
        return None, restored_count
    if not restored_count:
        return event_block, 0

    rendered_data = json.dumps(restored_payload, ensure_ascii=False)
    restored_event_type = restored_payload.get("type")
    restored_lines: list[str] = []
    inserted_data = False
    data_line_index_set = set(data_line_indexes)
    for index, line in enumerate(lines):
        if line.startswith("event:") and isinstance(restored_event_type, str):
            restored_lines.append(f"event: {restored_event_type}")
            continue
        if index not in data_line_index_set:
            restored_lines.append(line)
            continue
        if not inserted_data:
            restored_lines.append(f"data: {rendered_data}")
            inserted_data = True
    return "\n".join(restored_lines), restored_count


def _restore_adapted_custom_tool_calls_in_streaming_response(
    response: StreamingResponse,
    *,
    request_body: Optional[dict[str, Any]],
    adapter_model: str,
) -> StreamingResponse:
    adapted_names = _advertised_custom_tool_function_adapter_names(  # noqa: F821
        request_body,
        adapter_model=adapter_model,
    )
    if not adapted_names:
        return response

    original_iterator = response.body_iterator

    async def _restoring_iterator() -> Any:
        buffer = ""
        decoder = codecs.getincrementaldecoder("utf-8")()
        state_by_key: dict[str, dict[str, Any]] = {}
        async for raw_chunk in original_iterator:
            if isinstance(raw_chunk, bytes):
                buffer += decoder.decode(raw_chunk)
            else:
                buffer += str(raw_chunk)

            while "\n\n" in buffer:
                event_block, buffer = buffer.split("\n\n", 1)
                restored_block, _ = _restore_adapted_custom_tool_calls_in_sse_event_block(
                    event_block,
                    request_body=request_body,
                    adapter_model=adapter_model,
                    adapted_names=adapted_names,
                    state_by_key=state_by_key,
                )
                if restored_block is not None:
                    yield f"{restored_block}\n\n"

        buffer += decoder.decode(b"", final=True)
        if buffer:
            restored_block, _ = _restore_adapted_custom_tool_calls_in_sse_event_block(
                buffer,
                request_body=request_body,
                adapter_model=adapter_model,
                adapted_names=adapted_names,
                state_by_key=state_by_key,
            )
            if restored_block is not None:
                yield restored_block

    return StreamingResponse(
        _restoring_iterator(),
        headers=dict(response.headers),
        status_code=response.status_code,
        media_type=response.media_type or "text/event-stream",
    )


def _restore_adapted_namespace_tool_calls_in_stream_event_payload(
    event_payload: dict[str, Any],
    *,
    namespace_by_name: dict[str, str],
) -> tuple[dict[str, Any], int]:
    restored_payload = event_payload
    restored_count = 0

    item, item_restored_count = _restore_adapted_namespace_tool_call_item(
        event_payload.get("item"),
        namespace_by_name=namespace_by_name,
    )
    if item_restored_count:
        restored_payload = dict(restored_payload)
        restored_payload["item"] = item
        restored_count += item_restored_count

    response_body = event_payload.get("response")
    if isinstance(response_body, dict):
        output = response_body.get("output")
        if isinstance(output, list):
            restored_output: list[Any] = []
            response_restored_count = 0
            for output_item in output:
                restored_item, output_item_restored_count = _restore_adapted_namespace_tool_call_item(
                    output_item,
                    namespace_by_name=namespace_by_name,
                )
                restored_output.append(restored_item)
                response_restored_count += output_item_restored_count
            if response_restored_count:
                restored_response_body = dict(response_body)
                restored_response_body["output"] = restored_output
                if restored_payload is event_payload:
                    restored_payload = dict(restored_payload)
                restored_payload["response"] = restored_response_body
                restored_count += response_restored_count

    return restored_payload, restored_count


def _restore_adapted_namespace_tool_calls_in_sse_event_block(
    event_block: str,
    *,
    namespace_by_name: dict[str, str],
) -> tuple[str, int]:
    lines = event_block.splitlines()
    data_line_indexes = [index for index, line in enumerate(lines) if line.startswith("data:")]
    if not data_line_indexes:
        return event_block, 0

    raw_data = "\n".join(lines[index].removeprefix("data:").lstrip(" ") for index in data_line_indexes)
    if not raw_data or raw_data == "[DONE]":
        return event_block, 0
    try:
        event_payload = json.loads(raw_data)
    except Exception:
        return event_block, 0
    if not isinstance(event_payload, dict):
        return event_block, 0

    restored_payload, restored_count = _restore_adapted_namespace_tool_calls_in_stream_event_payload(
        event_payload,
        namespace_by_name=namespace_by_name,
    )
    if not restored_count:
        return event_block, 0

    rendered_data = json.dumps(restored_payload, ensure_ascii=False)
    restored_lines: list[str] = []
    inserted_data = False
    data_line_index_set = set(data_line_indexes)
    for index, line in enumerate(lines):
        if index not in data_line_index_set:
            restored_lines.append(line)
            continue
        if not inserted_data:
            restored_lines.append(f"data: {rendered_data}")
            inserted_data = True
    return "\n".join(restored_lines), restored_count


def _restore_adapted_namespace_tool_calls_in_streaming_response(
    response: StreamingResponse,
    *,
    request_body: Optional[dict[str, Any]],
    adapter_model: str,
) -> StreamingResponse:
    namespace_by_name = _advertised_namespace_tool_function_adapter_map(
        request_body,
        adapter_model=adapter_model,
    )
    if not namespace_by_name:
        return response

    original_iterator = response.body_iterator

    async def _restoring_iterator() -> Any:
        buffer = ""
        decoder = codecs.getincrementaldecoder("utf-8")()
        async for raw_chunk in original_iterator:
            if isinstance(raw_chunk, bytes):
                buffer += decoder.decode(raw_chunk)
            else:
                buffer += str(raw_chunk)

            while "\n\n" in buffer:
                event_block, buffer = buffer.split("\n\n", 1)
                restored_block, _ = _restore_adapted_namespace_tool_calls_in_sse_event_block(
                    event_block,
                    namespace_by_name=namespace_by_name,
                )
                yield f"{restored_block}\n\n"

        buffer += decoder.decode(b"", final=True)
        if buffer:
            restored_block, _ = _restore_adapted_namespace_tool_calls_in_sse_event_block(
                buffer,
                namespace_by_name=namespace_by_name,
            )
            yield restored_block

    return StreamingResponse(
        _restoring_iterator(),
        headers=dict(response.headers),
        status_code=response.status_code,
        media_type=response.media_type or "text/event-stream",
    )


def _raise_codex_auto_agent_malformed_adapted_custom_tool_call(
    *,
    response_body: dict[str, Any],
    adapter_model: str,
    adapter: str,
    adapter_label: str,
    adapter_error: dict[str, Any],
    stream_event_summaries: Optional[list[dict[str, Any]]] = None,
) -> None:
    diagnostic = _build_failed_responses_diagnostic(  # noqa: F821
        response_body=response_body,
        adapter=adapter,
        adapter_model=adapter_model,
        stream_event_summaries=stream_event_summaries,
    )
    diagnostic["custom_tool_function_adapter_error"] = adapter_error
    exc = ProxyException(
        message=(
            f"Codex auto-agent {adapter_label} candidate returned invalid " "arguments for an adapted custom tool."
        ),
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
                "code": "aawm_auto_agent_malformed_tool_call_text",
                "status": "RESPONSES_MALFORMED_TOOL_CALL",
                "type": "invalid_request_error",
            },
            "diagnostic": diagnostic,
        },
    )
    raise exc
