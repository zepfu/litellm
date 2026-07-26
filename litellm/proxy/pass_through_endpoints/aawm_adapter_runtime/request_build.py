"""Wave 6A extraction: request_build response-body inspection and repair functions.

Behavior-preserving extraction from llm_passthrough_endpoints.py.
Do not import llm_passthrough_endpoints at module scope.
"""

from __future__ import annotations

import json
from functools import lru_cache
from typing import Any, Optional

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from starlette.requests import Request

    # Host-global modules (bound via install())
    _anthropic_grok_composer_repair: Any
    _aawm_provider_shaping: Any

    # Host-global functions
    def _extract_auto_agent_alias_metadata_value(request_body: dict, *keys: str) -> Optional[str]: ...
    def _extract_auto_agent_alias_agent_dispatch_fields(request: Request, body: dict) -> dict: ...
    def _extract_auto_agent_alias_incoming_endpoint(request: Request) -> str: ...
    def _extract_auto_agent_alias_session_id(request: Request, request_body: dict) -> Optional[str]: ...
    def _extract_passthrough_repository(request: Request, request_body: Optional[dict] = None) -> Optional[str]: ...
    def _get_request_header_or_passthrough_alias(request: Request, header_name: str) -> Optional[str]: ...
    def is_malformed_composer_call_literal_text(text: str) -> bool: ...
    def is_malformed_grok_literal_tool_label_transcript_text(text: str) -> bool: ...
    def _is_codex_auto_agent_malformed_tool_call_text_output(response_body: dict) -> bool: ...
    def _tool_definition_name(tool: dict) -> Optional[str]: ...
    def _tool_definition_parameters(tool: dict) -> Any: ...
    def _normalize_openai_function_tool_parameters(parameters: Any) -> dict: ...
    def _get_custom_tool_function_adapter_names_for_model(model: Any) -> set: ...
    def _normalize_low_cardinality_tag_value(value: Any) -> Optional[str]: ...
    def _get_openai_tool_type(tool: dict) -> Optional[str]: ...
    def _get_openai_tool_name(tool: dict) -> Optional[str]: ...

from types import FunctionType


_HOST_FUNCTION_NAMES = (
    "_responses_output_item_has_meaningful_content",
    "_is_empty_success_responses_body",
    "_is_failed_responses_body",
    "_build_malformed_tool_call_intake_context",
    "_get_anthropic_grok_composer_repair_runtime",
    "_grok_composer_literal_tool_block_strip_start",
    "_parse_grok_composer_literal_tool_label_blocks",
    "_parse_grok_composer_literal_tool_payload_json",
    "_sanitize_grok_composer_literal_tool_arguments",
    "_escape_unescaped_newlines_in_json_payload",
    "_strip_text_spans",
    "_build_advertised_openai_function_tools_index",
    "_json_schema_value_matches_type",
    "_validate_tool_arguments_against_openai_parameters",
    "_build_repaired_grok_composer_function_call_output_item",
    "_dedupe_repaired_grok_composer_call_id",
    "_repair_grok_composer_literal_tool_calls_in_text",
    "_response_body_has_grok_composer_literal_tool_label_blocks",
    "_repair_grok_composer_literal_tool_calls_in_message_item",
    "_try_repair_codex_auto_agent_grok_native_composer_literal_tool_call_response_body",
    "_advertised_custom_tool_function_adapter_names",
    "_parse_adapted_custom_tool_function_arguments",
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
        if not isinstance(_obj, FunctionType):
            # lru_cache wrappers and other callables: publish as-is.
            host_globals[_name] = _obj
            continue
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


def _responses_output_item_has_meaningful_content(item: Any) -> bool:
    if not isinstance(item, dict):
        return False
    item_type = item.get("type")
    if item_type in {"function_call", "mcp_call"}:
        return True
    if item_type == "message":
        content = item.get("content") or []
        if not isinstance(content, list):
            return False
        for part in content:
            if not isinstance(part, dict):
                continue
            if part.get("type") in {"output_text", "text"}:
                text = part.get("text")
                if isinstance(text, str) and text.strip():
                    return True
        return False
    if item_type == "reasoning":
        summary = item.get("summary") or []
        if not isinstance(summary, list):
            return False
        for part in summary:
            if isinstance(part, dict):
                text = part.get("text")
                if isinstance(text, str) and text.strip():
                    return True
        return False
    return False


def _is_empty_success_responses_body(response_body: dict[str, Any]) -> bool:
    status = response_body.get("status")
    if status not in {None, "completed"}:
        return False
    output = response_body.get("output") or []
    if not isinstance(output, list):
        return False
    if any(_responses_output_item_has_meaningful_content(item) for item in output):
        return False
    output_text = response_body.get("output_text")
    if isinstance(output_text, str) and output_text.strip():
        return False
    return True


def _is_failed_responses_body(response_body: dict[str, Any]) -> bool:
    return response_body.get("status") == "failed" or response_body.get("error") is not None


def _build_malformed_tool_call_intake_context(
    request: Optional[Request] = None,
    request_body: Optional[dict[str, Any]] = None,
    *,
    adapter: Optional[str] = None,
    upstream_url: Optional[str] = None,
    provider: Optional[str] = None,
    model_alias: Optional[str] = None,
) -> dict[str, Any]:
    body = request_body if isinstance(request_body, dict) else {}
    metadata = body.get("litellm_metadata")
    if not isinstance(metadata, dict):
        metadata = {}
    passthrough_metadata = body.get("metadata")
    if not isinstance(passthrough_metadata, dict):
        passthrough_metadata = {}

    def _meta(*keys: str) -> Optional[str]:
        for key in keys:
            for source in (metadata, passthrough_metadata):
                value = source.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
        return None

    context: dict[str, Any] = {
        "provider": provider,
        "model_alias": model_alias
        or _meta("model_alias", "alias_model", "requested_model")
        or (body.get("model") if isinstance(body.get("model"), str) else None),
        "route_family": _meta("passthrough_route_family", "route_family") or adapter,
        "endpoint": None,
        "upstream_url": upstream_url,
        "repository": _extract_auto_agent_alias_metadata_value(  # noqa: F821
            body,
            "repository",
            "repo",
            "repo_name",
            "repository_name",
        ),
        "agent_name": _meta("agent_name", "aawm_claude_agent_name"),
        "agent_id": _extract_auto_agent_alias_metadata_value(  # noqa: F821
            body,
            "agent_id",
            "aawm_agent_id",
            "codex_agent_id",
            "claude_agent_id",
        ),
        "session_id": None,
        "agent_role": _meta("agent_role", "aawm_agent_role", "codex_agent_role"),
        "agent_profile": _meta(
            "agent_profile",
            "aawm_agent_profile",
            "codex_agent_profile",
        ),
        "thread_source": _meta(
            "thread_source",
            "aawm_thread_source",
            "codex_thread_source",
        ),
        "thread_id": _meta("thread_id", "aawm_thread_id", "codex_thread_id"),
        "dispatch_id": _meta(
            "dispatch_id",
            "agent_dispatch_id",
            "aawm_dispatch_id",
            "codex_dispatch_id",
        ),
        "redispatch_ordinal": _meta(
            "redispatch_ordinal",
            "agent_redispatch_ordinal",
            "dispatch_ordinal",
        ),
        "terminal_outcome": "malformed_tool_call_rejected",
        "fallback_result": "none",
        "redispatch_required": False,
        "agent_session_killed": True,
        "trace_id": _meta("trace_id", "existing_trace_id"),
        "litellm_call_id": _meta("litellm_call_id"),
        "request_started_at": _meta("request_started_at", "start_time"),
    }
    if request is not None:
        context.update(_extract_auto_agent_alias_agent_dispatch_fields(request, body))  # noqa: F821
        context["endpoint"] = _extract_auto_agent_alias_incoming_endpoint(request)  # noqa: F821
        context["session_id"] = _extract_auto_agent_alias_session_id(request, body)  # noqa: F821
        if not context["repository"]:
            context["repository"] = _extract_passthrough_repository(request, body)  # noqa: F821
        if not context["trace_id"]:
            context["trace_id"] = _get_request_header_or_passthrough_alias(  # noqa: F821
                request, "langfuse_trace_id"
            ) or _get_request_header_or_passthrough_alias(request, "trace_id")  # noqa: F821
    return {key: value for key, value in context.items() if value is not None}


@lru_cache(maxsize=1)
def _get_anthropic_grok_composer_repair_runtime() -> _anthropic_grok_composer_repair.Runtime:  # noqa: F821
    return _anthropic_grok_composer_repair.Runtime(  # noqa: F821
        decode_json_prefix=_aawm_provider_shaping.decode_json_prefix,  # noqa: F821
        strip_text_spans=_strip_text_spans,
        build_advertised_function_tools_index=(_build_advertised_openai_function_tools_index),
        validate_tool_arguments=_validate_tool_arguments_against_openai_parameters,
        is_malformed_composer_literal_text=is_malformed_composer_call_literal_text,  # noqa: F821
        is_malformed_tool_call_text_output=(_is_codex_auto_agent_malformed_tool_call_text_output),  # noqa: F821
    )


def _grok_composer_literal_tool_block_strip_start(text: str, label_start: int) -> int:
    return _anthropic_grok_composer_repair.literal_tool_block_strip_start(  # noqa: F821
        text,
        label_start,
    )


def _parse_grok_composer_literal_tool_label_blocks(
    text: str,
) -> list[dict[str, Any]]:
    return _anthropic_grok_composer_repair.parse_literal_tool_label_blocks(  # noqa: F821
        _get_anthropic_grok_composer_repair_runtime(),
        text,
    )


def _parse_grok_composer_literal_tool_payload_json(payload: str) -> Any:
    return _anthropic_grok_composer_repair.parse_literal_tool_payload_json(payload)  # noqa: F821


def _sanitize_grok_composer_literal_tool_arguments(
    arguments: Any,
    parameters: dict[str, Any],
) -> Any:
    return _anthropic_grok_composer_repair.sanitize_literal_tool_arguments(  # noqa: F821
        arguments,
        parameters,
    )


def _escape_unescaped_newlines_in_json_payload(payload: str) -> str:
    if not isinstance(payload, str) or not payload:
        return payload

    output_chars: list[str] = []
    in_json_string = False
    escape_next = False
    for char in payload:
        if escape_next:
            output_chars.append(char)
            escape_next = False
            continue
        if char == "\\":
            output_chars.append(char)
            escape_next = True
            continue
        if char == '"':
            in_json_string = not in_json_string
            output_chars.append(char)
            continue
        if char == "\n" and in_json_string:
            output_chars.append("\\n")
            continue
        output_chars.append(char)
    return "".join(output_chars)


def _strip_text_spans(text: str, spans: list[tuple[int, int]]) -> str:
    if not spans:
        return text
    normalized_spans = [(start, end) for start, end in spans if isinstance(start, int) and isinstance(end, int)]
    if not normalized_spans:
        return text
    merged_spans: list[tuple[int, int]] = []
    for start, end in sorted(normalized_spans, key=lambda span: span[0]):
        if not merged_spans:
            merged_spans.append((start, end))
            continue
        previous_start, previous_end = merged_spans[-1]
        if start <= previous_end:
            merged_spans[-1] = (previous_start, max(previous_end, end))
            continue
        merged_spans.append((start, end))
    segments: list[str] = []
    cursor = 0
    for start, end in merged_spans:
        if start > cursor:
            segments.append(text[cursor:start])
        cursor = max(cursor, end)
    segments.append(text[cursor:])
    return "".join(segments)


def _build_advertised_openai_function_tools_index(
    request_body: Optional[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    if not isinstance(request_body, dict):
        return {}

    tools_index: dict[str, dict[str, Any]] = {}
    for source in (request_body.get("tools"), request_body.get("functions")):
        if not isinstance(source, list):
            continue
        for tool in source:
            if not isinstance(tool, dict):
                continue
            tool_name = _tool_definition_name(tool)  # noqa: F821
            if not isinstance(tool_name, str) or not tool_name.strip():
                continue
            parameters = _tool_definition_parameters(tool)  # noqa: F821
            tools_index[tool_name] = _normalize_openai_function_tool_parameters(parameters)  # noqa: F821
    return tools_index


def _json_schema_value_matches_type(value: Any, schema_type: str) -> bool:
    if schema_type == "string":
        return isinstance(value, str)
    if schema_type == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if schema_type == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if schema_type == "boolean":
        return isinstance(value, bool)
    if schema_type == "array":
        return isinstance(value, list)
    if schema_type == "object":
        return isinstance(value, dict)
    if schema_type == "null":
        return value is None
    return True


def _validate_tool_arguments_against_openai_parameters(
    *,
    tool_name: str,
    arguments: Any,
    parameters: dict[str, Any],
) -> Optional[str]:
    if not isinstance(arguments, dict):
        return "tool_arguments_not_object"

    if parameters.get("type") not in {None, "object"}:
        return "tool_schema_unsupported_root_type"

    properties = parameters.get("properties")
    if not isinstance(properties, dict):
        properties = {}

    required = parameters.get("required")
    required_fields: list[str] = []
    if isinstance(required, list):
        required_fields = [field for field in required if isinstance(field, str) and field.strip()]

    for field in required_fields:
        if field not in arguments:
            return f"missing_required_argument:{field}"

    additional_properties = parameters.get("additionalProperties", True)
    if additional_properties is False:
        unknown_keys = sorted(key for key in arguments.keys() if key not in properties)
        if unknown_keys:
            return f"unknown_argument:{unknown_keys[0]}"

    for key, value in arguments.items():
        property_schema = properties.get(key)
        if not isinstance(property_schema, dict):
            if additional_properties is False:
                return f"unknown_argument:{key}"
            continue
        schema_type = property_schema.get("type")
        if isinstance(schema_type, list):
            if not any(
                _json_schema_value_matches_type(value, candidate)
                for candidate in schema_type
                if isinstance(candidate, str)
            ):
                return f"argument_type_mismatch:{key}"
            continue
        if isinstance(schema_type, str) and not _json_schema_value_matches_type(value, schema_type):
            return f"argument_type_mismatch:{key}"

    return None


def _build_repaired_grok_composer_function_call_output_item(
    *,
    tool_name: str,
    call_id: Optional[str],
    arguments: dict[str, Any],
    block_index: int,
) -> dict[str, Any]:
    return _anthropic_grok_composer_repair.build_repaired_function_call_output_item(  # noqa: F821
        tool_name=tool_name,
        call_id=call_id,
        arguments=arguments,
        block_index=block_index,
    )


def _dedupe_repaired_grok_composer_call_id(
    call_id: Optional[str],
    *,
    block_index: int,
    used_call_ids: set[str],
) -> Optional[str]:
    return _anthropic_grok_composer_repair.dedupe_repaired_call_id(  # noqa: F821
        call_id,
        block_index=block_index,
        used_call_ids=used_call_ids,
    )


def _repair_grok_composer_literal_tool_calls_in_text(
    text: str,
    *,
    advertised_tools: dict[str, dict[str, Any]],
) -> tuple[Optional[str], list[dict[str, Any]]]:
    return _anthropic_grok_composer_repair.repair_literal_tool_calls_in_text(  # noqa: F821
        _get_anthropic_grok_composer_repair_runtime(),
        text,
        advertised_tools=advertised_tools,
    )


def _response_body_has_grok_composer_literal_tool_label_blocks(
    response_body: dict[str, Any],
) -> bool:
    return _anthropic_grok_composer_repair.response_body_has_literal_tool_label_blocks(  # noqa: F821
        _get_anthropic_grok_composer_repair_runtime(),
        response_body,
    )


def _repair_grok_composer_literal_tool_calls_in_message_item(
    item: dict[str, Any],
    *,
    advertised_tools: dict[str, dict[str, Any]],
) -> Optional[tuple[list[dict[str, Any]], bool]]:
    return _anthropic_grok_composer_repair.repair_literal_tool_calls_in_message_item(  # noqa: F821
        _get_anthropic_grok_composer_repair_runtime(),
        item,
        advertised_tools=advertised_tools,
    )


def _try_repair_codex_auto_agent_grok_native_composer_literal_tool_call_response_body(
    response_body: dict[str, Any],
    *,
    request_body: Optional[dict[str, Any]],
) -> Optional[dict[str, Any]]:
    return _anthropic_grok_composer_repair.try_repair_literal_tool_call_response_body(  # noqa: F821
        _get_anthropic_grok_composer_repair_runtime(),
        response_body,
        request_body=request_body,
    )


def _advertised_custom_tool_function_adapter_names(
    request_body: Optional[dict[str, Any]],
    *,
    adapter_model: str,
) -> set[str]:
    if not isinstance(request_body, dict):
        return set()

    configured_names = _get_custom_tool_function_adapter_names_for_model(adapter_model)  # noqa: F821
    if not configured_names:
        return set()

    tools = request_body.get("tools")
    if not isinstance(tools, list):
        return set()

    advertised_names: set[str] = set()
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        tool_type = _normalize_low_cardinality_tag_value(_get_openai_tool_type(tool))  # noqa: F821
        tool_name = _normalize_low_cardinality_tag_value(_get_openai_tool_name(tool))  # noqa: F821
        if (
            tool_type == "custom"
            and tool_name is not None
            and tool_name in configured_names
        ):
            advertised_names.add(tool_name)
    return advertised_names


def _parse_adapted_custom_tool_function_arguments(
    arguments: Any,
) -> tuple[Optional[str], Optional[str]]:
    if not isinstance(arguments, str):
        return None, "arguments_not_string"
    try:
        parsed_arguments = json.loads(arguments)
    except (TypeError, ValueError):
        return None, "arguments_not_json"
    if not isinstance(parsed_arguments, dict):
        return None, "arguments_not_object"
    if set(parsed_arguments) != {"input"}:
        return None, "arguments_not_exact_input_object"
    raw_input = parsed_arguments.get("input")
    if not isinstance(raw_input, str):
        return None, "input_not_string"
    return raw_input, None
