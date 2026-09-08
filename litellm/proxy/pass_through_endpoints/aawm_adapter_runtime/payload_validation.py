"""Wave 6A Author D extraction: payload_validation functions.

Behavior-preserving extraction from llm_passthrough_endpoints.py.
Do not import llm_passthrough_endpoints at module scope.

Provider-neutral SSE/build/decode helpers and ``_mapping_or_attr_get`` are
host-global integration dependencies, not locally owned functions.
"""

from __future__ import annotations

import codecs
import json
from typing import Any, AsyncIterator, Optional, cast

from fastapi import HTTPException, Request, Response
from fastapi.responses import StreamingResponse

from litellm._logging import verbose_proxy_logger
from litellm.integrations.aawm_agent_quality_rules import (
    is_malformed_composer_call_literal_text,
    is_malformed_grok_literal_tool_label_transcript_text,
)
from litellm.proxy._types import ProxyException
from litellm.proxy.aawm_runtime_error_logging import (
    schedule_persist_malformed_tool_call_detection,
)

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Host-global modules (bound via install())
    _aawm_alias_streaming: Any

    # Host-global constants
    _AAWM_VALIDATE_RESPONSES_STREAM_MAX_BUFFERED_CHUNKS: int
    _AAWM_VALIDATE_RESPONSES_STREAM_MAX_BUFFERED_BYTES: int

    # Host-global functions
    def _build_malformed_tool_call_intake_context(request: Any, request_body: Any, *, adapter: str, upstream_url: Any = None, provider: Any = None, model_alias: Any = None) -> dict: ...
    def _is_empty_success_responses_body(response_body: dict) -> bool: ...
    def _is_failed_responses_body(response_body: dict) -> bool: ...
    def _should_log_aawm_alias_routing_event(log_key: str) -> bool: ...
    def _mapping_or_attr_get(obj: Any, key: str, default: Any = None) -> Any: ...
    def _decode_http_response_body(body: Any) -> str: ...
    async def _collect_responses_response_from_stream(response: Any, event_summaries: Any = None) -> dict: ...
    def _restore_adapted_custom_tool_calls_in_streaming_response(response: Any, *, request_body: Any = None, adapter_model: str = "") -> Any: ...
    def _restore_adapted_namespace_tool_calls_in_streaming_response(response: Any, *, request_body: Any = None, adapter_model: str = "") -> Any: ...
    def _restore_adapted_custom_tool_calls_in_response_body(response_body: dict, *, request_body: Any = None, adapter_model: str = "") -> tuple: ...
    def _restore_adapted_namespace_tool_calls_in_response_body(response_body: dict, *, request_body: Any = None, adapter_model: str = "") -> tuple: ...
    def _try_repair_codex_auto_agent_grok_native_composer_literal_tool_call_response_body(response_body: dict, *, request_body: Any = None) -> Any: ...
    def _raise_codex_auto_agent_malformed_adapted_custom_tool_call(*, response_body: dict, adapter_model: str, adapter: str, adapter_label: str, adapter_error: Any, stream_event_summaries: Any = None) -> None: ...
    def _responses_sse_from_repaired_response_body(
        response_body: dict,
        *,
        request_body: Any = None,
    ) -> AsyncIterator[str]: ...
    def _build_empty_success_responses_diagnostic(*, response_body: dict, diagnostic_context: Any) -> dict: ...

from types import FunctionType


_HOST_FUNCTION_NAMES = (
    "_is_codex_auto_agent_malformed_tool_call_text_output",
    "_raise_codex_auto_agent_invalid_responses_shape",
    "_is_responses_shaped_body",
    "_responses_item_has_valid_content_part",
    "_responses_output_item_is_structurally_valid",
    "_validate_alias_candidate_responses_stream_if_needed",
    "_build_malformed_intake_context_for_anthropic_responses_adapter",
    "_is_codex_auto_agent_empty_success_responses_body",
    "_coerce_optional_int",
    "_usage_has_no_more_than_one_output_token",
    "_model_response_usage_dict",
    "_raise_codex_auto_agent_empty_success_response",
    "_build_failed_responses_diagnostic",
    "_raise_codex_auto_agent_malformed_tool_call_text_payload",
    "_raise_codex_auto_agent_failed_responses_payload",
    "_raise_responses_adapter_failed_response",
    "_preserve_distinct_function_call_identity_fields",
    "_validate_codex_auto_agent_responses_payload",
)

_RESPONSES_VALID_STATUSES = frozenset(
    {
        "cancelled",
        "completed",
        "failed",
        "in_progress",
        "incomplete",
        "queued",
    }
)

_RESPONSES_OUTPUT_ITEM_TYPES = frozenset(
    {
        "message",
        "file_search_call",
        "function_call",
        "function_call_output",
        "web_search_call",
        "computer_call",
        "computer_call_output",
        "reasoning",
        "compaction",
        "image_generation_call",
        "code_interpreter_call",
        "local_shell_call",
        "local_shell_call_output",
        "shell_call",
        "shell_call_output",
        "apply_patch_call",
        "apply_patch_call_output",
        "mcp_call",
        "mcp_call_output",
        "mcp_list_tools",
        "mcp_approval_request",
        "mcp_approval_response",
        "custom_tool_call",
        "custom_tool_call_output",
    }
)

_RESPONSES_VALID_ITEM_STATUSES = frozenset(
    {
        "in_progress",
        "completed",
        "incomplete",
        "failed",
        "cancelled",
        "queued",
        "calling",
        "searching",
        "generating",
        "interpreting",
    }
)

_RESPONSES_VALIDATION_STATE_ATTR = "_aawm_responses_validation_state"
_RESPONSES_VALIDATION_COMPLETE_ATTR = "_aawm_responses_validation_complete"
_RESPONSES_VALIDATION_VALID_ATTR = "_aawm_responses_validation_valid"
_RESPONSES_VALIDATION_CLEANUP_ATTR = "_aawm_responses_validation_cleanup"
_STREAM_CLEANUP_ATTR = "_aawm_streaming_response_cleanup"


def install(host_globals: dict) -> None:
    """Rebind moved functions to host_globals for live lookup.

    Each named function's __globals__ is replaced with the host module's
    live namespace dict, preserving monkeypatch compatibility.  The same
    rebound object is published to both this module and the host module.
    """
    _mod = globals()
    host_globals["_RESPONSES_VALID_STATUSES"] = _RESPONSES_VALID_STATUSES
    host_globals["_RESPONSES_OUTPUT_ITEM_TYPES"] = _RESPONSES_OUTPUT_ITEM_TYPES
    host_globals["_RESPONSES_VALID_ITEM_STATUSES"] = _RESPONSES_VALID_ITEM_STATUSES
    for _name in (
        "_responses_is_int",
        "_responses_optional_nonempty_string",
        "_responses_optional_string",
        "_responses_nonempty_string_list",
        "_responses_string_list",
        "_responses_optional_int",
        "_responses_status_is_valid",
        "_responses_output_text_annotation_is_valid",
    ):
        host_globals[_name] = _mod[_name]
    host_globals["_responses_output_value_is_structurally_valid"] = (
        _responses_output_value_is_structurally_valid
    )
    host_globals["_responses_required_nonempty_strings"] = (
        _responses_required_nonempty_strings
    )
    _body_status_helper = FunctionType(
        _responses_body_is_unsuccessful.__code__,
        host_globals,
        _responses_body_is_unsuccessful.__name__,
        _responses_body_is_unsuccessful.__defaults__,
        _responses_body_is_unsuccessful.__closure__,
    )
    _body_status_helper.__kwdefaults__ = _responses_body_is_unsuccessful.__kwdefaults__
    _body_status_helper.__annotations__ = _responses_body_is_unsuccessful.__annotations__
    _body_status_helper.__doc__ = _responses_body_is_unsuccessful.__doc__
    _body_status_helper.__module__ = _responses_body_is_unsuccessful.__module__
    _body_status_helper.__qualname__ = _responses_body_is_unsuccessful.__qualname__
    if _responses_body_is_unsuccessful.__dict__:
        _body_status_helper.__dict__.update(_responses_body_is_unsuccessful.__dict__)
    _mod["_responses_body_is_unsuccessful"] = _body_status_helper
    host_globals["_responses_body_is_unsuccessful"] = _body_status_helper
    host_globals["_RESPONSES_VALIDATION_STATE_ATTR"] = (
        _RESPONSES_VALIDATION_STATE_ATTR
    )
    host_globals["_RESPONSES_VALIDATION_COMPLETE_ATTR"] = (
        _RESPONSES_VALIDATION_COMPLETE_ATTR
    )
    host_globals["_RESPONSES_VALIDATION_VALID_ATTR"] = (
        _RESPONSES_VALIDATION_VALID_ATTR
    )
    host_globals["_RESPONSES_VALIDATION_CLEANUP_ATTR"] = (
        _RESPONSES_VALIDATION_CLEANUP_ATTR
    )
    host_globals["_STREAM_CLEANUP_ATTR"] = _STREAM_CLEANUP_ATTR
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


def _is_codex_auto_agent_malformed_tool_call_text_output(
    response_body: dict[str, Any],
) -> bool:
    output = response_body.get("output")
    if not isinstance(output, list):
        return False

    for item in output:
        if not isinstance(item, dict):
            continue
        if item.get("type") == "message":
            content = item.get("content")
            if isinstance(content, str):
                if is_malformed_composer_call_literal_text(content):
                    return True
                if is_malformed_grok_literal_tool_label_transcript_text(content):
                    return True
                continue
            if not isinstance(content, list):
                continue
            for part in content:
                if not isinstance(part, dict):
                    continue
                if part.get("type") not in {"text", "output_text"}:
                    continue
                part_text = part.get("text") or ""
                if is_malformed_composer_call_literal_text(part_text):
                    return True
                if is_malformed_grok_literal_tool_label_transcript_text(part_text):
                    return True
            continue

        if item.get("type") in {"function_call", "mcp_call"}:
            name = item.get("name")
            if isinstance(name, str) and name.strip().lower() == "composer_call":
                return True
            continue
    return False


async def _validate_alias_candidate_responses_stream_if_needed(
    response: StreamingResponse,
    *,
    enabled: bool,
    adapter_model: str,
    adapter: str,
    adapter_label: str,
    request: Optional[Request] = None,
    request_body: Optional[dict[str, Any]] = None,
    upstream_url: Optional[str] = None,
    provider: Optional[str] = None,
    model_alias: Optional[str] = None,
) -> StreamingResponse:
    if not enabled:
        return response
    intake_context = _build_malformed_tool_call_intake_context(  # noqa: F821
        request,
        request_body,
        adapter=adapter,
        upstream_url=upstream_url,
        provider=provider,
        model_alias=model_alias,
    )
    return cast(
        StreamingResponse,
        await _validate_codex_auto_agent_responses_payload(
            response,
            adapter_model=adapter_model,
            adapter=adapter,
            adapter_label=adapter_label,
            intake_context=intake_context,
            request_body=request_body,
        ),
    )


def _build_malformed_intake_context_for_anthropic_responses_adapter(
    *,
    request: Optional[Request],
    request_body: Optional[dict[str, Any]],
    adapter: str,
    adapter_model: str,
    upstream_url: Optional[str] = None,
    provider: Optional[str] = None,
) -> dict[str, Any]:
    return _build_malformed_tool_call_intake_context(  # noqa: F821
        request,
        request_body,
        adapter=adapter,
        upstream_url=upstream_url,
        provider=provider,
        model_alias=(
            request_body.get("model")
            if isinstance(request_body, dict) and isinstance(request_body.get("model"), str)
            else None
        ),
    )


def _is_codex_auto_agent_empty_success_responses_body(
    response_body: dict[str, Any],
) -> bool:
    if not _is_empty_success_responses_body(response_body):  # noqa: F821
        return False
    usage = response_body.get("usage") or {}
    if not isinstance(usage, dict):
        return False
    output_tokens = usage.get("output_tokens")
    if output_tokens is None:
        return False
    try:
        return int(output_tokens) <= 1
    except Exception:
        return False

def _coerce_optional_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except Exception:
        return None


def _usage_has_no_more_than_one_output_token(usage: Any) -> bool:
    if usage is None:
        return True
    saw_output_field = False
    for field in ("completion_tokens", "output_tokens", "output"):
        token_count = _coerce_optional_int(_mapping_or_attr_get(usage, field))
        if token_count is None:
            continue
        saw_output_field = True
        if token_count > 1:
            return False
    if saw_output_field:
        return True
    total_tokens = _coerce_optional_int(_mapping_or_attr_get(usage, "total_tokens"))
    if total_tokens == 0:
        return True
    return False


def _model_response_usage_dict(usage: Any) -> dict[str, Any]:
    if usage is None:
        return {}
    if isinstance(usage, dict):
        return dict(usage)
    model_dump = getattr(usage, "model_dump", None)
    if callable(model_dump):
        try:
            dumped = model_dump(exclude_none=True)
            if isinstance(dumped, dict):
                return dumped
        except Exception:
            pass
    result: dict[str, Any] = {}
    for field in (
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "output_tokens",
    ):
        value = getattr(usage, field, None)
        if value is not None:
            result[field] = value
    return result


def _raise_codex_auto_agent_empty_success_response(
    *,
    response_body: dict[str, Any],
    adapter_model: str,
    adapter: str = "codex_auto_agent_openrouter_responses",
    adapter_label: str = "OpenRouter",
    stream_event_summaries: Optional[list[dict[str, Any]]] = None,
) -> None:
    diagnostic = _build_empty_success_responses_diagnostic(  # noqa: F821
        response_body=response_body,
        diagnostic_context={
            "adapter": adapter,
            "adapter_model": adapter_model,
            **({"stream_events": stream_event_summaries} if stream_event_summaries is not None else {}),
        },
    )
    # RR-054 #23: empty successful payload is retryable upstream emptiness, not rate limit.
    exc = ProxyException(
        message=(f"Codex auto-agent {adapter_label} candidate returned an empty successful " "Responses payload."),
        type="upstream_error",
        param="model",
        code=502,
    )
    setattr(
        exc,
        "detail",
        {
            "error": {
                "message": exc.message,
                "code": "aawm_codex_auto_agent_empty_success",
                "status": "EMPTY_SUCCESS_RESPONSE",
                "type": "upstream_error",
            },
            "diagnostic": diagnostic,
        },
    )
    raise exc


def _build_failed_responses_diagnostic(
    *,
    response_body: dict[str, Any],
    adapter: str,
    adapter_model: str,
    stream_event_summaries: Optional[list[dict[str, Any]]] = None,
) -> dict[str, Any]:
    output = response_body.get("output") or []
    diagnostic: dict[str, Any] = {
        "adapter": adapter,
        "adapter_model": adapter_model,
        "response_id": response_body.get("id"),
        "status": response_body.get("status"),
        "model": response_body.get("model"),
        "error": response_body.get("error"),
        "incomplete_details": response_body.get("incomplete_details"),
        "output_count": len(output) if isinstance(output, list) else 0,
        "output_types": [item.get("type") for item in output[:20] if isinstance(item, dict)]
        if isinstance(output, list)
        else [],
    }
    if stream_event_summaries is not None:
        diagnostic["stream_events"] = stream_event_summaries
    return diagnostic


def _raise_codex_auto_agent_malformed_tool_call_text_payload(
    *,
    response_body: dict[str, Any],
    adapter_model: str,
    adapter: str,
    adapter_label: str,
    intake_context: Optional[dict[str, Any]] = None,
    stream_event_summaries: Optional[list[dict[str, Any]]] = None,
) -> None:
    try:
        # Offload synchronous JSONL intake when called under a running loop so
        # async request handlers do not block on disk I/O while rejecting
        # malformed tool-call text. Sync callers still persist inline.
        schedule_persist_malformed_tool_call_detection(
            response_body=response_body,
            adapter_model=adapter_model,
            adapter=adapter,
            adapter_label=adapter_label,
            intake_context=intake_context,
            stream_event_summaries=stream_event_summaries,
        )
    except Exception:
        # RR-054 #38: intake must stay best-effort, but never become silent.
        verbose_proxy_logger.exception("Failed to schedule malformed tool-call detection intake")
    diagnostic = _build_failed_responses_diagnostic(
        response_body=response_body,
        adapter=adapter,
        adapter_model=adapter_model,
        stream_event_summaries=stream_event_summaries,
    )
    # RR-054 #23: malformed tool-call text is not a rate limit.
    exc = ProxyException(
        message=(f"Codex auto-agent {adapter_label} candidate returned a malformed " "Responses marker payload."),
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


def _raise_codex_auto_agent_failed_responses_payload(
    *,
    response_body: dict[str, Any],
    adapter_model: str,
    adapter: str,
    adapter_label: str,
    stream_event_summaries: Optional[list[dict[str, Any]]] = None,
) -> None:
    diagnostic = _build_failed_responses_diagnostic(
        response_body=response_body,
        adapter=adapter,
        adapter_model=adapter_model,
        stream_event_summaries=stream_event_summaries,
    )
    # RR-054 #23: failed upstream Responses status is a bad gateway / upstream error.
    exc = ProxyException(
        message=(f"Auto-agent {adapter_label} candidate returned a failed Responses " "payload."),
        type="upstream_error",
        param="model",
        code=502,
    )
    setattr(
        exc,
        "detail",
        {
            "error": {
                "message": exc.message,
                "code": "aawm_auto_agent_failed_responses_payload",
                "status": "RESPONSES_STATUS_FAILED",
                "type": "upstream_error",
            },
            "diagnostic": diagnostic,
        },
    )
    raise exc


def _raise_codex_auto_agent_invalid_responses_shape(
    *,
    response_body: Any,
    adapter_model: str,
    adapter: str,
    adapter_label: str,
    stream_event_summaries: Optional[list[dict[str, Any]]] = None,
) -> None:
    diagnostic: dict[str, Any] = {
        "adapter": adapter,
        "adapter_model": adapter_model,
        "body_type": type(response_body).__name__,
    }
    if stream_event_summaries is not None:
        diagnostic["stream_event_summaries"] = stream_event_summaries
    exc = ProxyException(
        message=(
            f"Auto-agent {adapter_label} candidate returned a malformed "
            "Responses payload."
        ),
        type="upstream_error",
        param="model",
        code=502,
    )
    setattr(
        exc,
        "detail",
        {
            "error": {
                "message": exc.message,
                "code": "aawm_auto_agent_invalid_responses_shape",
                "status": "RESPONSES_INVALID_SHAPE",
                "type": "upstream_error",
            },
            "diagnostic": diagnostic,
        },
    )
    raise exc


def _responses_item_has_valid_content_part(part: Any) -> bool:
    if not isinstance(part, dict):
        return False
    part_type = part.get("type")
    if not isinstance(part_type, str) or not part_type.strip():
        return False
    if part_type in {"output_text", "text"}:
        if not isinstance(part.get("text"), str):
            return False
        annotations = part.get("annotations")
        return annotations is None or (
            isinstance(annotations, list)
            and all(_responses_output_text_annotation_is_valid(annotation) for annotation in annotations)
        )
    if part_type == "refusal":
        return isinstance(part.get("refusal"), str)
    return False


def _responses_output_text_annotation_is_valid(annotation: Any) -> bool:
    if not isinstance(annotation, dict):
        return False
    annotation_type = annotation.get("type")
    if annotation_type == "file_citation":
        return _responses_required_nonempty_strings(
            annotation,
            ("file_id", "filename"),
        ) and _responses_is_int(annotation.get("index"))
    if annotation_type == "url_citation":
        return (
            _responses_required_nonempty_strings(annotation, ("title", "url"))
            and _responses_is_int(annotation.get("start_index"))
            and _responses_is_int(annotation.get("end_index"))
        )
    if annotation_type == "container_file_citation":
        return (
            _responses_required_nonempty_strings(
                annotation,
                ("container_id", "file_id", "filename"),
            )
            and _responses_is_int(annotation.get("start_index"))
            and _responses_is_int(annotation.get("end_index"))
        )
    if annotation_type == "file_path":
        return _responses_required_nonempty_strings(
            annotation,
            ("file_id",),
        ) and _responses_is_int(annotation.get("index"))
    return False


def _responses_is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _responses_optional_nonempty_string(
    item: dict[str, Any],
    field: str,
) -> bool:
    value = item.get(field)
    return value is None or (isinstance(value, str) and bool(value.strip()))


def _responses_optional_string(
    item: dict[str, Any],
    field: str,
) -> bool:
    value = item.get(field)
    return value is None or isinstance(value, str)


def _responses_nonempty_string_list(value: Any) -> bool:
    return isinstance(value, list) and all(
        isinstance(entry, str) and bool(entry.strip()) for entry in value
    )


def _responses_string_list(value: Any) -> bool:
    return isinstance(value, list) and all(isinstance(entry, str) for entry in value)


def _responses_optional_int(item: dict[str, Any], field: str) -> bool:
    value = item.get(field)
    return value is None or _responses_is_int(value)


def _responses_status_is_valid(
    item: dict[str, Any],
    allowed: frozenset[str],
    *,
    required: bool = False,
) -> bool:
    status = item.get("status")
    if status is None:
        return not required
    return isinstance(status, str) and status in allowed


def _responses_output_value_is_structurally_valid(
    value: Any,
    *,
    allow_input_content: bool = True,
) -> bool:
    if isinstance(value, str):
        return True
    if not isinstance(value, list):
        return False
    for part in value:
        if not isinstance(part, dict):
            return False
        part_type = part.get("type")
        if part_type == "input_text":
            if not allow_input_content or not isinstance(part.get("text"), str):
                return False
        elif part_type == "input_image":
            if not allow_input_content:
                return False
            if part.get("detail") not in {"low", "high", "auto"}:
                return False
            if not _responses_optional_string(part, "file_id") or not _responses_optional_string(
                part,
                "image_url",
            ):
                return False
        elif part_type == "input_file":
            if not allow_input_content:
                return False
            if not all(
                _responses_optional_string(part, field)
                for field in ("file_data", "file_id", "file_url", "filename")
            ):
                return False
        else:
            return False
    return True


def _responses_required_nonempty_strings(
    item: dict[str, Any],
    fields: tuple[str, ...],
) -> bool:
    return all(
        isinstance(item.get(field), str) and bool(item[field].strip())
        for field in fields
    )


def _responses_output_item_is_structurally_valid(  # noqa: PLR0915
    item: Any,
) -> bool:
    if not isinstance(item, dict):
        return False
    item_type = item.get("type")
    if (
        not isinstance(item_type, str)
        or not item_type.strip()
        or item_type not in _RESPONSES_OUTPUT_ITEM_TYPES
    ):
        return False

    item_id_required = item_type not in {
        "function_call",
        "custom_tool_call",
        "custom_tool_call_output",
    }
    if item_id_required and not _responses_required_nonempty_strings(item, ("id",)):
        return False
    if not item_id_required and not _responses_optional_nonempty_string(item, "id"):
        return False

    if item_type == "message":
        if not _responses_status_is_valid(
            item,
            frozenset({"in_progress", "completed", "incomplete"}),
            required=True,
        ):
            return False
        if item.get("role") != "assistant":
            return False
        content = item.get("content")
        if not isinstance(content, list) or not all(
            _responses_item_has_valid_content_part(part) for part in content
        ):
            return False
    elif item_type == "file_search_call":
        if not _responses_status_is_valid(
            item,
            frozenset(
                {"in_progress", "searching", "completed", "incomplete", "failed"}
            ),
            required=True,
        ):
            return False
        if not _responses_nonempty_string_list(item.get("queries")):
            return False
        results = item.get("results")
        if results is not None:
            if not isinstance(results, list):
                return False
            for result in results:
                if not isinstance(result, dict):
                    return False
                if not all(
                    _responses_optional_string(result, field)
                    for field in ("file_id", "filename", "text")
                ):
                    return False
                if result.get("score") is not None and not isinstance(
                    result.get("score"),
                    (int, float),
                ):
                    return False
                attributes = result.get("attributes")
                if attributes is not None and (
                    not isinstance(attributes, dict)
                    or not all(
                        isinstance(key, str)
                        and isinstance(value, (str, int, float, bool))
                        for key, value in attributes.items()
                    )
                ):
                    return False
    elif item_type == "function_call":
        if not _responses_status_is_valid(
            item,
            frozenset({"in_progress", "completed", "incomplete"}),
        ):
            return False
        if not _responses_required_nonempty_strings(
            item,
            ("name", "arguments"),
        ):
            return False
        if not _responses_required_nonempty_strings(item, ("call_id",)):
            return False
    elif item_type == "function_call_output":
        if not _responses_status_is_valid(
            item,
            frozenset({"in_progress", "completed", "incomplete"}),
        ):
            return False
        if not _responses_required_nonempty_strings(item, ("call_id",)):
            return False
        if "output" not in item or not _responses_output_value_is_structurally_valid(
            item.get("output")
        ):
            return False
    elif item_type == "web_search_call":
        if not _responses_status_is_valid(
            item,
            frozenset({"in_progress", "searching", "completed", "failed"}),
            required=True,
        ):
            return False
        action = item.get("action")
        if not isinstance(action, dict):
            return False
        action_type = action.get("type")
        if action_type == "search":
            if not isinstance(action.get("query"), str):
                return False
            if action.get("queries") is not None and not _responses_string_list(
                action.get("queries")
            ):
                return False
            sources = action.get("sources")
            if sources is not None:
                if not isinstance(sources, list) or not all(
                    isinstance(source, dict)
                    and source.get("type") == "url"
                    and isinstance(source.get("url"), str)
                    for source in sources
                ):
                    return False
        elif action_type == "open_page":
            if not _responses_optional_string(action, "url"):
                return False
        elif action_type == "find_in_page":
            if not _responses_required_nonempty_strings(action, ("pattern", "url")):
                return False
        else:
            return False
    elif item_type == "computer_call":
        if not _responses_status_is_valid(
            item,
            frozenset({"in_progress", "completed", "incomplete"}),
            required=True,
        ):
            return False
        if not _responses_required_nonempty_strings(item, ("call_id",)):
            return False
        action = item.get("action")
        if not isinstance(action, dict) or not isinstance(action.get("type"), str):
            return False
        action_type = action["type"]
        if action_type in {"click", "double_click", "move"}:
            if not _responses_is_int(action.get("x")) or not _responses_is_int(
                action.get("y")
            ):
                return False
        elif action_type == "drag":
            path = action.get("path")
            if not isinstance(path, list) or not all(
                isinstance(point, dict)
                and _responses_is_int(point.get("x"))
                and _responses_is_int(point.get("y"))
                for point in path
            ):
                return False
        elif action_type == "keypress":
            if not _responses_string_list(action.get("keys")):
                return False
        elif action_type == "scroll":
            if not all(
                _responses_is_int(action.get(field))
                for field in ("scroll_x", "scroll_y", "x", "y")
            ):
                return False
        elif action_type == "type":
            if not isinstance(action.get("text"), str):
                return False
        elif action_type != "screenshot" and action_type != "wait":
            return False
        safety_checks = item.get("pending_safety_checks")
        if not isinstance(safety_checks, list) or not all(
            _responses_required_nonempty_strings(check, ("id",))
            and _responses_optional_string(check, "code")
            and _responses_optional_string(check, "message")
            for check in safety_checks
            if isinstance(check, dict)
        ):
            return False
        if not all(isinstance(check, dict) for check in safety_checks):
            return False
    elif item_type == "computer_call_output":
        if not _responses_status_is_valid(
            item,
            frozenset({"in_progress", "completed", "incomplete"}),
        ):
            return False
        if not _responses_required_nonempty_strings(item, ("call_id",)):
            return False
        output = item.get("output")
        if not isinstance(output, dict) or output.get("type") != "computer_screenshot":
            return False
        if not _responses_optional_string(output, "file_id") or not _responses_optional_string(
            output,
            "image_url",
        ):
            return False
    elif item_type == "mcp_call_output":
        if not _responses_status_is_valid(
            item,
            frozenset({"in_progress", "completed", "incomplete"}),
        ):
            return False
        if not _responses_required_nonempty_strings(item, ("call_id",)):
            return False
        if "output" not in item or not _responses_output_value_is_structurally_valid(
            item.get("output")
        ):
            return False
    elif item_type == "reasoning":
        if not _responses_status_is_valid(
            item,
            frozenset({"in_progress", "completed", "incomplete"}),
        ):
            return False
        summary = item.get("summary")
        if not isinstance(summary, list) or not all(
            isinstance(part, dict)
            and part.get("type") == "summary_text"
            and isinstance(part.get("text"), str)
            for part in summary
        ):
            return False
        content = item.get("content")
        if content is not None and (
            not isinstance(content, list)
            or not all(
                isinstance(part, dict)
                and part.get("type") == "reasoning_text"
                and isinstance(part.get("text"), str)
                for part in content
            )
        ):
            return False
        encrypted_content = item.get("encrypted_content")
        if encrypted_content is not None and not isinstance(encrypted_content, str):
            return False
    elif item_type == "compaction":
        if not _responses_required_nonempty_strings(item, ("encrypted_content",)):
            return False
        if not _responses_optional_string(item, "created_by"):
            return False
    elif item_type == "image_generation_call":
        if not _responses_status_is_valid(
            item,
            frozenset({"in_progress", "completed", "generating", "failed"}),
            required=True,
        ):
            return False
        if not _responses_optional_string(item, "result"):
            return False
    elif item_type == "code_interpreter_call":
        if not _responses_status_is_valid(
            item,
            frozenset(
                {"in_progress", "completed", "incomplete", "interpreting", "failed"}
            ),
            required=True,
        ):
            return False
        if not _responses_required_nonempty_strings(item, ("container_id",)):
            return False
        if not _responses_optional_string(item, "code"):
            return False
        outputs = item.get("outputs")
        if outputs is not None:
            if not isinstance(outputs, list):
                return False
            for output in outputs:
                if not isinstance(output, dict):
                    return False
                output_type = output.get("type")
                if output_type == "logs":
                    if not isinstance(output.get("logs"), str):
                        return False
                elif output_type == "image":
                    if not _responses_required_nonempty_strings(output, ("url",)):
                        return False
                else:
                    return False
    elif item_type == "local_shell_call":
        if not _responses_status_is_valid(
            item,
            frozenset({"in_progress", "completed", "incomplete"}),
            required=True,
        ):
            return False
        if not _responses_required_nonempty_strings(item, ("call_id",)):
            return False
        action = item.get("action")
        if (
            not isinstance(action, dict)
            or action.get("type") != "exec"
            or not _responses_string_list(action.get("command"))
            or not isinstance(action.get("env"), dict)
            or not all(
                isinstance(key, str) and isinstance(value, str)
                for key, value in action["env"].items()
            )
            or not _responses_optional_int(action, "timeout_ms")
            or not _responses_optional_string(action, "user")
            or not _responses_optional_string(action, "working_directory")
        ):
            return False
    elif item_type == "local_shell_call_output":
        if not _responses_status_is_valid(
            item,
            frozenset({"in_progress", "completed", "incomplete"}),
        ):
            return False
        if not _responses_required_nonempty_strings(item, ("output",)):
            return False
    elif item_type == "shell_call":
        if not _responses_status_is_valid(
            item,
            frozenset({"in_progress", "completed", "incomplete"}),
            required=True,
        ):
            return False
        if not _responses_required_nonempty_strings(item, ("call_id",)):
            return False
        action = item.get("action")
        if not isinstance(action, dict) or not _responses_string_list(
            action.get("commands")
        ):
            return False
        if not _responses_optional_int(action, "max_output_length") or not _responses_optional_int(
            action,
            "timeout_ms",
        ):
            return False
        environment = item.get("environment")
        if environment is not None:
            if not isinstance(environment, dict):
                return False
            environment_type = environment.get("type")
            if environment_type == "local":
                pass
            elif environment_type == "container_reference":
                if not _responses_required_nonempty_strings(
                    environment,
                    ("container_id",),
                ):
                    return False
            else:
                return False
    elif item_type == "shell_call_output":
        if not _responses_status_is_valid(
            item,
            frozenset({"in_progress", "completed", "incomplete"}),
            required=True,
        ):
            return False
        if not _responses_required_nonempty_strings(item, ("call_id",)):
            return False
        output = item.get("output")
        if not isinstance(output, list):
            return False
        for output_item in output:
            if not isinstance(output_item, dict):
                return False
            if not _responses_optional_string(output_item, "created_by"):
                return False
            if not isinstance(output_item.get("stdout"), str) or not isinstance(
                output_item.get("stderr"),
                str,
            ):
                return False
            outcome = output_item.get("outcome")
            if not isinstance(outcome, dict):
                return False
            if outcome.get("type") == "timeout":
                continue
            if outcome.get("type") == "exit" and _responses_is_int(
                outcome.get("exit_code")
            ):
                continue
            return False
    elif item_type == "apply_patch_call":
        if not _responses_status_is_valid(
            item,
            frozenset({"in_progress", "completed"}),
            required=True,
        ):
            return False
        if not _responses_required_nonempty_strings(item, ("call_id",)):
            return False
        operation = item.get("operation")
        if not isinstance(operation, dict):
            return False
        operation_type = operation.get("type")
        if operation_type not in {"create_file", "delete_file", "update_file"}:
            return False
        if not _responses_required_nonempty_strings(operation, ("path",)):
            return False
        if operation_type in {"create_file", "update_file"} and not isinstance(
            operation.get("diff"),
            str,
        ):
            return False
    elif item_type == "apply_patch_call_output":
        if not _responses_status_is_valid(
            item,
            frozenset({"completed", "failed"}),
            required=True,
        ):
            return False
        if not _responses_required_nonempty_strings(item, ("call_id",)):
            return False
        if not _responses_optional_string(item, "output"):
            return False
    elif item_type == "mcp_call":
        if not _responses_status_is_valid(
            item,
            frozenset({"in_progress", "completed", "incomplete", "calling", "failed"}),
        ):
            return False
        if not _responses_required_nonempty_strings(
            item,
            ("arguments", "name", "server_label"),
        ):
            return False
        if not _responses_optional_nonempty_string(item, "approval_request_id"):
            return False
        if not _responses_optional_string(item, "output") or not _responses_optional_string(
            item,
            "error",
        ):
            return False
    elif item_type == "mcp_call_output":
        if not _responses_required_nonempty_strings(item, ("call_id",)):
            return False
        if "output" not in item or not _responses_output_value_is_structurally_valid(
            item.get("output")
        ):
            return False
    elif item_type == "mcp_list_tools":
        if not _responses_required_nonempty_strings(item, ("server_label",)):
            return False
        tools = item.get("tools")
        if not isinstance(tools, list):
            return False
        for tool in tools:
            if not isinstance(tool, dict) or not _responses_required_nonempty_strings(
                tool,
                ("name",),
            ):
                return False
            if not _responses_optional_string(tool, "description"):
                return False
    elif item_type == "mcp_approval_request":
        if not _responses_required_nonempty_strings(
            item,
            ("arguments", "name", "server_label"),
        ):
            return False
    elif item_type == "mcp_approval_response":
        if not _responses_required_nonempty_strings(item, ("approval_request_id",)):
            return False
        if not isinstance(item.get("approve"), bool):
            return False
        if not _responses_optional_string(item, "reason"):
            return False
    elif item_type == "custom_tool_call":
        if not _responses_required_nonempty_strings(
            item,
            ("call_id", "name", "input"),
        ):
            return False
    elif item_type == "custom_tool_call_output":
        if not _responses_required_nonempty_strings(item, ("call_id",)):
            return False
        if "output" not in item or not _responses_output_value_is_structurally_valid(
            item.get("output")
        ):
            return False
    else:
        return False

    if "status" in item and item.get("status") is not None:
        if not isinstance(item.get("status"), str) or item.get(
            "status"
        ) not in _RESPONSES_VALID_ITEM_STATUSES:
            return False
    return True


def _is_responses_shaped_body(response_body: Any) -> bool:
    if not isinstance(response_body, dict):
        return False
    if response_body.get("object") != "response":
        return False
    response_id = response_body.get("id")
    if not isinstance(response_id, str) or not response_id.strip():
        return False
    status = response_body.get("status")
    if not isinstance(status, str) or status not in _RESPONSES_VALID_STATUSES:
        return False
    output = response_body.get("output")
    return isinstance(output, list) and all(
        _responses_output_item_is_structurally_valid(item) for item in output
    )


def _responses_body_is_unsuccessful(
    response_body: Any,
    *,
    terminal_event_type: Optional[str] = None,
) -> bool:
    if not isinstance(response_body, dict):
        return terminal_event_type in {"response.failed", "response.incomplete"}
    response_status = response_body.get("status")
    return (
        terminal_event_type in {"response.failed", "response.incomplete"}
        or (
            isinstance(response_status, str)
            and response_status in {"failed", "incomplete"}
        )
        or _is_failed_responses_body(response_body)
    )


def _raise_responses_adapter_failed_response(
    *,
    response_body: dict[str, Any],
    adapter_model: str,
    adapter: str,
    adapter_label: str,
    retryable_alias_candidate: bool = False,
    stream_event_summaries: Optional[list[dict[str, Any]]] = None,
) -> None:
    if retryable_alias_candidate:
        _raise_codex_auto_agent_failed_responses_payload(
            response_body=response_body,
            adapter_model=adapter_model,
            adapter=adapter,
            adapter_label=adapter_label,
            stream_event_summaries=stream_event_summaries,
        )

    diagnostic = _build_failed_responses_diagnostic(
        response_body=response_body,
        adapter=adapter,
        adapter_model=adapter_model,
        stream_event_summaries=stream_event_summaries,
    )
    raise HTTPException(
        status_code=502,
        detail={
            "error": f"{adapter_label} Responses adapter returned a failed response.",
            "diagnostic": diagnostic,
        },
    )



def _preserve_distinct_function_call_identity_fields(
    response_body: dict[str, Any],
) -> dict[str, Any]:
    """
    Preserve distinct Responses function_call ``id`` / ``call_id`` fields.

    OPENAI-007: ``call_id`` is exclusively the upstream provider tool id and
    ``id`` is the Responses item id (``fc_*``). For typed ``function_call``
    output only, repair malformed/non-native item ids deterministically through
    the shared identity helper using ``call_id``, while preserving ``call_id``
    byte-for-byte and leaving valid native ``fc_*`` ids untouched.
    """
    if not isinstance(response_body, dict):
        return response_body
    output = response_body.get("output")
    if not isinstance(output, list):
        return response_body

    preserved_output: list[Any] = []
    changed = False
    for item in output:
        if not isinstance(item, dict) or item.get("type") != "function_call":
            preserved_output.append(item)
            continue

        clean_item = dict(item)
        item_id = clean_item.get("id")
        call_id = clean_item.get("call_id")

        # Drop blank placeholders only. Never invent call_id from id.
        if "id" in clean_item and not (isinstance(item_id, str) and item_id.strip()):
            clean_item.pop("id", None)
            item_id = None
            changed = True
        if "call_id" in clean_item and not (
            isinstance(call_id, str) and call_id.strip()
        ):
            clean_item.pop("call_id", None)
            call_id = None
            changed = True

        # Repair malformed/non-native item ids from provider call_id only.
        # Preserve call_id byte-for-byte and leave valid native fc_* item ids
        # untouched. Never mutate function_call_output or nested/untyped fields.
        # Import inside the function so install()-rebound host globals still work.
        if isinstance(call_id, str) and call_id.strip():
            from litellm.responses.litellm_completion_transformation.function_call_identity import (
                is_native_responses_function_call_item_id,
                resolve_responses_function_call_identity,
            )

            resolved_item_id, _resolved_call_id = (
                resolve_responses_function_call_identity(call_id)
            )
            item_id_is_native = (
                isinstance(item_id, str)
                and bool(item_id.strip())
                and is_native_responses_function_call_item_id(item_id)
            )
            if not item_id_is_native and resolved_item_id:
                if clean_item.get("id") != resolved_item_id:
                    clean_item["id"] = resolved_item_id
                    changed = True

        preserved_output.append(clean_item)

    if not changed:
        return response_body
    updated = dict(response_body)
    updated["output"] = preserved_output
    return updated


async def _validate_codex_auto_agent_responses_payload(  # noqa: PLR0915
    response: Response,
    *,
    adapter_model: str,
    adapter: str,
    adapter_label: str,
    intake_context: Optional[dict[str, Any]] = None,
    request_body: Optional[dict[str, Any]] = None,
) -> Response:
    from litellm.proxy.pass_through_endpoints.aawm_adapter_runtime.repetitive_output import (
        inherit_or_wrap_passthrough_streaming_response,
        is_repetitive_output_loop_failure,
    )
    from litellm.proxy.pass_through_endpoints.aawm_adapter_runtime.deferred_success import (
        inherit_deferred_success_holder,
    )

    def _set_stream_validation_state(
        target: Any,
        state: dict[str, Any],
    ) -> None:
        setattr(target, _RESPONSES_VALIDATION_STATE_ATTR, state)
        setattr(
            target,
            _RESPONSES_VALIDATION_COMPLETE_ATTR,
            bool(state.get("complete")),
        )
        setattr(
            target,
            _RESPONSES_VALIDATION_VALID_ATTR,
            bool(state.get("valid")),
        )

    def _update_stream_validation_state(
        target: Any,
        state: dict[str, Any],
        **updates: Any,
    ) -> None:
        if state.get("invalid") and updates.get("valid") is True:
            updates["valid"] = False
            updates["reason"] = state.get("reason") or "invalid_stream"
        state.update(updates)
        _set_stream_validation_state(target, state)

    def _invalidate_stream(
        target: Any,
        state: dict[str, Any],
        reason: str,
    ) -> None:
        if not state.get("invalid"):
            state["invalid_reason"] = reason
        state["invalid"] = True
        state["valid"] = False
        state["reason"] = state.get("invalid_reason") or reason
        _set_stream_validation_state(target, state)

    def _validated_stream_state() -> dict[str, Any]:
        return {
            "complete": False,
            "valid": False,
            "terminal_seen": False,
            "terminal_status": None,
            "invalid": False,
            "invalid_reason": None,
            "reason": "awaiting_terminal_validation",
        }

    def _bind_incremental_stream_validation(  # noqa: PLR0915
        target: StreamingResponse,
        state: dict[str, Any],
        *,
        event_summaries: list[dict[str, Any]],
    ) -> StreamingResponse:
        original_iterator = target.body_iterator
        upstream_cleanup = getattr(target, _STREAM_CLEANUP_ATTR, None)
        decoder = codecs.getincrementaldecoder("utf-8")()
        sse_buffer = ""
        trailing_cr = False
        buffer_limit_exceeded = False
        decoder_failed = False
        terminal_response: Optional[dict[str, Any]] = None
        terminal_event_type: Optional[str] = None
        iterator_closed = False
        validation_cleanup_called = False
        max_buffered_bytes = max(
            0,
            int(_AAWM_VALIDATE_RESPONSES_STREAM_MAX_BUFFERED_BYTES),  # noqa: F821
        )
        # The stream peek bound limits retained upstream history, not the
        # size of one valid SSE event. Keep a separate bounded parser budget so
        # a large output-text delta does not turn an otherwise valid stream
        # into a malformed response merely because peeking stopped early.
        max_event_buffered_bytes = max(max_buffered_bytes, 8 * 1024 * 1024)
        terminal_event_types = frozenset(
            {
                "response.completed",
                "response.done",
                "response.failed",
                "response.incomplete",
            }
        )

        def _record_event_block(event_block: str) -> None:
            nonlocal terminal_response, terminal_event_type
            event_name: Optional[str] = None
            data_lines: list[str] = []
            has_meaningful_line = False
            for line in event_block.splitlines():
                if line.strip() and not line.lstrip().startswith(":"):
                    has_meaningful_line = True
                if line.startswith("event:"):
                    event_name = line.partition(":")[2].strip() or None
                elif line.startswith("data:"):
                    data_lines.append(line.partition(":")[2].lstrip())
            if not data_lines:
                if has_meaningful_line:
                    _invalidate_stream(target, state, "malformed_sse_event")
                return
            data_text = "\n".join(data_lines).strip()
            if data_text == "[DONE]":
                return
            if not data_text:
                _invalidate_stream(target, state, "malformed_sse_event")
                return
            try:
                payload = json.loads(data_text)
            except Exception:  # noqa: BLE001
                _invalidate_stream(target, state, "malformed_sse_event")
                return
            if not isinstance(payload, dict):
                _invalidate_stream(target, state, "malformed_sse_event")
                return
            payload_type = payload.get("type")
            if payload_type is not None and not isinstance(payload_type, str):
                _invalidate_stream(target, state, "malformed_sse_event")
                return
            event_type = payload_type or event_name
            if not isinstance(event_type, str) or not event_type.strip():
                _invalidate_stream(target, state, "malformed_sse_event")
                return
            if len(event_summaries) < 50:
                event_summaries.append({"type": event_type})
            if state.get("terminal_seen"):
                _invalidate_stream(target, state, "event_after_terminal")
                return
            if event_type not in terminal_event_types:
                return
            terminal_event_type = event_type
            state["terminal_seen"] = True
            response_payload = payload.get("response")
            if isinstance(response_payload, dict):
                terminal_response = response_payload
                state["terminal_status"] = response_payload.get("status")
            else:
                terminal_response = None
                state["terminal_status"] = None
                _invalidate_stream(target, state, "missing_terminal_response")

        def _consume_sse_text(text: str, *, final: bool = False) -> None:
            nonlocal buffer_limit_exceeded, sse_buffer, trailing_cr
            if buffer_limit_exceeded:
                return
            if trailing_cr:
                text = f"\r{text}"
                trailing_cr = False
            if text.endswith("\r"):
                text = text[:-1]
                trailing_cr = True
            normalized = text.replace("\r\n", "\n").replace("\r", "\n")
            while normalized:
                delimiter_index = normalized.find("\n\n")
                if delimiter_index < 0:
                    candidate = sse_buffer + normalized
                    if len(candidate.encode("utf-8")) > max_event_buffered_bytes:
                        buffer_limit_exceeded = True
                        sse_buffer = ""
                        _invalidate_stream(target, state, "byte_limit")
                        return
                    sse_buffer = candidate
                    normalized = ""
                    continue
                event_fragment = normalized[:delimiter_index]
                candidate = sse_buffer + event_fragment
                if len(candidate.encode("utf-8")) > max_event_buffered_bytes:
                    buffer_limit_exceeded = True
                    sse_buffer = ""
                    _invalidate_stream(target, state, "byte_limit")
                    return
                sse_buffer = candidate
                normalized = normalized[delimiter_index + 2 :]
                event_block, sse_buffer = sse_buffer, ""
                _record_event_block(event_block)
            if final:
                if trailing_cr:
                    trailing_cr = False
                    if (
                        len((sse_buffer + "\n").encode("utf-8"))
                        > max_event_buffered_bytes
                    ):
                        buffer_limit_exceeded = True
                        sse_buffer = ""
                        _invalidate_stream(target, state, "byte_limit")
                        return
                    sse_buffer += "\n"
                    trailing_cr = False
                if sse_buffer:
                    _record_event_block(sse_buffer)
                    sse_buffer = ""

        async def _close_bound_iterator() -> None:
            nonlocal iterator_closed
            if iterator_closed:
                return
            iterator_closed = True
            close = getattr(original_iterator, "aclose", None)
            if callable(close):
                try:
                    await close()
                except BaseException:
                    verbose_proxy_logger.debug(
                        "Failed to close Responses validation stream",
                        exc_info=True,
                    )

        async def _close_validation_resources() -> None:
            nonlocal validation_cleanup_called
            if validation_cleanup_called:
                return
            validation_cleanup_called = True
            if not state.get("complete"):
                if state.get("invalid"):
                    _update_stream_validation_state(
                        target,
                        state,
                        complete=True,
                        valid=False,
                        reason=state.get("reason") or "invalid_stream",
                    )
                else:
                    _update_stream_validation_state(
                        target,
                        state,
                        complete=False,
                        valid=False,
                        reason="stream_closed_before_validation",
                    )
            if callable(upstream_cleanup):
                try:
                    await upstream_cleanup()
                except BaseException:
                    verbose_proxy_logger.debug(
                        "Failed to close peeked Responses validation stream",
                        exc_info=True,
                    )
            await _close_bound_iterator()

        async def _validated_iterator() -> Any:  # noqa: PLR0915
            nonlocal decoder_failed
            try:
                async for raw_chunk in original_iterator:
                    yield raw_chunk
                    if decoder_failed:
                        continue
                    try:
                        if isinstance(raw_chunk, bytes):
                            chunk_text = decoder.decode(raw_chunk)
                        elif isinstance(raw_chunk, bytearray):
                            chunk_text = decoder.decode(bytes(raw_chunk))
                        else:
                            chunk_text = str(raw_chunk)
                    except UnicodeDecodeError:
                        decoder_failed = True
                        _invalidate_stream(target, state, "malformed_sse_event")
                        continue
                    _consume_sse_text(chunk_text)

                if not decoder_failed:
                    try:
                        _consume_sse_text(
                            decoder.decode(b"", final=True),
                            final=True,
                        )
                    except UnicodeDecodeError:
                        decoder_failed = True
                        _invalidate_stream(target, state, "malformed_sse_event")
                if state.get("invalid"):
                    _update_stream_validation_state(
                        target,
                        state,
                        complete=True,
                        valid=False,
                        reason=state.get("reason") or "invalid_stream",
                    )
                    _raise_codex_auto_agent_invalid_responses_shape(
                        response_body=terminal_response,
                        adapter_model=adapter_model,
                        adapter=adapter,
                        adapter_label=adapter_label,
                        stream_event_summaries=event_summaries,
                    )
                if terminal_response is None:
                    _update_stream_validation_state(
                        target,
                        state,
                        complete=True,
                        valid=False,
                        reason="missing_terminal_response",
                    )
                    _raise_codex_auto_agent_invalid_responses_shape(
                        response_body=None,
                        adapter_model=adapter_model,
                        adapter=adapter,
                        adapter_label=adapter_label,
                        stream_event_summaries=event_summaries,
                    )
                repetitive_failure = is_repetitive_output_loop_failure(
                    terminal_response
                )
                if repetitive_failure:
                    _update_stream_validation_state(
                        target,
                        state,
                        complete=True,
                        valid=False,
                        reason="repetitive_output_failure",
                    )
                    return
                if _responses_body_is_unsuccessful(
                    terminal_response,
                    terminal_event_type=terminal_event_type,
                ):
                    _update_stream_validation_state(
                        target,
                        state,
                        complete=True,
                        valid=False,
                        reason="failed_response",
                    )
                    _raise_codex_auto_agent_failed_responses_payload(
                        response_body=terminal_response,
                        adapter_model=adapter_model,
                        adapter=adapter,
                        adapter_label=adapter_label,
                        stream_event_summaries=event_summaries,
                    )
                if (
                    terminal_event_type not in {"response.completed", "response.done"}
                    or terminal_response.get("status") != "completed"
                ):
                    _update_stream_validation_state(
                        target,
                        state,
                        complete=True,
                        valid=False,
                        reason="unsuccessful_terminal_status",
                    )
                    _raise_codex_auto_agent_invalid_responses_shape(
                        response_body=terminal_response,
                        adapter_model=adapter_model,
                        adapter=adapter,
                        adapter_label=adapter_label,
                        stream_event_summaries=event_summaries,
                    )
                if not _is_responses_shaped_body(terminal_response):
                    _update_stream_validation_state(
                        target,
                        state,
                        complete=True,
                        valid=False,
                        reason="invalid_response_shape",
                    )
                    _raise_codex_auto_agent_invalid_responses_shape(
                        response_body=terminal_response,
                        adapter_model=adapter_model,
                        adapter=adapter,
                        adapter_label=adapter_label,
                        stream_event_summaries=event_summaries,
                    )
                _update_stream_validation_state(
                    target,
                    state,
                    complete=True,
                    valid=True,
                    reason="validated_terminal_response",
                )
            except BaseException:
                if not state.get("complete"):
                    if state.get("invalid"):
                        _update_stream_validation_state(
                            target,
                            state,
                            complete=True,
                            valid=False,
                            reason=state.get("reason") or "invalid_stream",
                        )
                    else:
                        _update_stream_validation_state(
                            target,
                            state,
                            complete=False,
                            valid=False,
                            reason=state.get("reason") or "stream_error",
                        )
                raise
            finally:
                await _close_validation_resources()

        target.body_iterator = _validated_iterator()
        setattr(
            target,
            _RESPONSES_VALIDATION_CLEANUP_ATTR,
            _close_validation_resources,
        )
        _set_stream_validation_state(target, state)
        return target

    if isinstance(response, StreamingResponse):
        event_summaries: list[dict[str, Any]] = []
        peek = await _aawm_alias_streaming.peek_streaming_response(  # noqa: F821
            response,
            max_chunks=_AAWM_VALIDATE_RESPONSES_STREAM_MAX_BUFFERED_CHUNKS,  # noqa: F821
            max_bytes=_AAWM_VALIDATE_RESPONSES_STREAM_MAX_BUFFERED_BYTES,  # noqa: F821
            terminalizer=_aawm_alias_streaming._get_stream_timeout_terminalizer(  # noqa: F821
                response
            ),
        )
        if not peek.exhausted:
            correlation = intake_context or {}
            model_alias = correlation.get("model_alias")
            if model_alias is None and isinstance(request_body, dict):
                model_alias = request_body.get("model")
            session_id = correlation.get("session_id")
            litellm_call_id = correlation.get("litellm_call_id")
            trace_id = correlation.get("trace_id")
            if peek.stop_reason == "pending_stream":
                verbose_proxy_logger.debug(
                    "Codex auto-agent responses validation continued lazily "
                    "(reason=%s chunks=%s bytes=%s adapter=%s "
                    "adapter_model=%s model_alias=%s session_id=%s "
                    "litellm_call_id=%s trace_id=%s); preserving the complete "
                    "upstream stream",
                    peek.stop_reason,
                    len(peek.buffered_chunks),
                    peek.buffered_bytes,
                    adapter,
                    adapter_model,
                    model_alias or "<missing>",
                    session_id or "<missing>",
                    litellm_call_id or "<missing>",
                    trace_id or "<missing>",
                )
            elif _should_log_aawm_alias_routing_event(f"validate-stream-limit:{adapter}:{peek.stop_reason}"):  # noqa: F821
                verbose_proxy_logger.warning(
                    "Codex auto-agent responses validation bypassed after bounded "
                    "peek limit (reason=%s chunks=%s bytes=%s max_chunks=%s "
                    "max_bytes=%s adapter=%s adapter_model=%s model_alias=%s "
                    "session_id=%s litellm_call_id=%s trace_id=%s); preserving "
                    "the complete upstream stream",
                    peek.stop_reason,
                    len(peek.buffered_chunks),
                    peek.buffered_bytes,
                    _AAWM_VALIDATE_RESPONSES_STREAM_MAX_BUFFERED_CHUNKS,  # noqa: F821
                    _AAWM_VALIDATE_RESPONSES_STREAM_MAX_BUFFERED_BYTES,  # noqa: F821
                    adapter,
                    adapter_model,
                    model_alias or "<missing>",
                    session_id or "<missing>",
                    litellm_call_id or "<missing>",
                    trace_id or "<missing>",
                )
            restored_response = _restore_adapted_custom_tool_calls_in_streaming_response(  # noqa: F821
                peek.response,
                request_body=request_body,
                adapter_model=adapter_model,
            )
            stream_cleanup = getattr(peek.response, _STREAM_CLEANUP_ATTR, None)
            if callable(stream_cleanup):
                setattr(restored_response, _STREAM_CLEANUP_ATTR, stream_cleanup)
            restored_response = _restore_adapted_namespace_tool_calls_in_streaming_response(  # noqa: F821
                restored_response,
                request_body=request_body,
                adapter_model=adapter_model,
            )
            if callable(stream_cleanup):
                setattr(restored_response, _STREAM_CLEANUP_ATTR, stream_cleanup)
            validated_response = inherit_or_wrap_passthrough_streaming_response(
                restored_response,
                source_response=response,
            )
            if callable(stream_cleanup):
                setattr(validated_response, _STREAM_CLEANUP_ATTR, stream_cleanup)
            validation_state = _validated_stream_state()
            _bind_incremental_stream_validation(
                validated_response,
                validation_state,
                event_summaries=event_summaries,
            )
            return validated_response
        validation_state = _validated_stream_state()
        validated_response = _bind_incremental_stream_validation(
            peek.response,
            validation_state,
            event_summaries=event_summaries,
        )
        response_body = await _collect_responses_response_from_stream(  # noqa: F821
            validated_response,
            event_summaries=event_summaries,
        )
        identity_changed = False
        if isinstance(response_body, dict):
            preserved_body = _preserve_distinct_function_call_identity_fields(
                response_body
            )
            identity_changed = preserved_body is not response_body
            response_body = preserved_body
        if _responses_body_is_unsuccessful(
            response_body
        ) and not is_repetitive_output_loop_failure(  # noqa: F821
            response_body
        ):
            # Exhausted failed SSE is the same failover signal as non-stream
            # failed JSON. Do not rewrite it to empty response.completed.
            _raise_codex_auto_agent_failed_responses_payload(
                response_body=response_body,
                adapter_model=adapter_model,
                adapter=adapter,
                adapter_label=adapter_label,
                stream_event_summaries=event_summaries,
            )
        if not isinstance(response_body, dict) or response_body.get("status") != "completed":
            _raise_codex_auto_agent_invalid_responses_shape(
                response_body=response_body,
                adapter_model=adapter_model,
                adapter=adapter,
                adapter_label=adapter_label,
                stream_event_summaries=event_summaries,
            )
        response_changed = identity_changed
        repaired_body = (
            _try_repair_codex_auto_agent_grok_native_composer_literal_tool_call_response_body(  # noqa: F821
                response_body,
                request_body=request_body,
            )
            if adapter in {
                "codex_auto_agent_grok_native_responses",
                "codex_auto_agent_xai_oauth_responses",
            }
            else None
        )
        if isinstance(repaired_body, dict):
            response_body = repaired_body
            response_changed = True
        (
            restored_body,
            restored_custom_tool_count,
            custom_tool_adapter_error,
        ) = _restore_adapted_custom_tool_calls_in_response_body(  # noqa: F821
            response_body,
            request_body=request_body,
            adapter_model=adapter_model,
        )
        if custom_tool_adapter_error is not None:
            _raise_codex_auto_agent_malformed_adapted_custom_tool_call(  # noqa: F821
                response_body=response_body,
                adapter_model=adapter_model,
                adapter=adapter,
                adapter_label=adapter_label,
                adapter_error=custom_tool_adapter_error,
                stream_event_summaries=event_summaries,
            )
        if restored_custom_tool_count:
            response_body = restored_body
            response_changed = True
        (
            restored_body,
            restored_namespace_tool_count,
        ) = _restore_adapted_namespace_tool_calls_in_response_body(  # noqa: F821
            response_body,
            request_body=request_body,
            adapter_model=adapter_model,
        )
        if restored_namespace_tool_count:
            response_body = restored_body
            response_changed = True
        if _is_codex_auto_agent_malformed_tool_call_text_output(response_body):
            _raise_codex_auto_agent_malformed_tool_call_text_payload(
                response_body=response_body,
                adapter_model=adapter_model,
                adapter=adapter,
                adapter_label=adapter_label,
                intake_context=intake_context,
                stream_event_summaries=event_summaries,
            )
        if not _is_responses_shaped_body(response_body):
            _raise_codex_auto_agent_invalid_responses_shape(
                response_body=response_body,
                adapter_model=adapter_model,
                adapter=adapter,
                adapter_label=adapter_label,
                stream_event_summaries=event_summaries,
            )
        if response_changed:
            reconstructed = StreamingResponse(
                _responses_sse_from_repaired_response_body(  # noqa: F821
                    response_body,
                    request_body=request_body if isinstance(request_body, dict) else None,
                ),
                headers={
                    key: value
                    for key, value in dict(response.headers).items()
                    if str(key).lower() != "content-length"
                },
                status_code=response.status_code,
                media_type=response.media_type or "text/event-stream",
            )
            validated_response = inherit_or_wrap_passthrough_streaming_response(
                reconstructed,
                source_response=response,
            )
            _set_stream_validation_state(
                validated_response,
                {
                    "complete": True,
                    "valid": True,
                    "terminal_seen": True,
                    "terminal_status": response_body.get("status"),
                    "reason": "validated_terminal_response",
                },
            )
            return validated_response

        async def _replay_iterator() -> Any:
            from litellm.proxy.pass_through_endpoints.aawm_adapter_runtime.encrypted_reasoning_provenance import (
                stamp_route_identity_in_sse_chunk,
            )

            identity_request_body = (
                request_body if isinstance(request_body, dict) else None
            )
            for raw_chunk in peek.buffered_chunks:
                yield stamp_route_identity_in_sse_chunk(
                    raw_chunk,
                    request_body=identity_request_body,
                )

        reconstructed = StreamingResponse(
            _replay_iterator(),
            headers={
                key: value
                for key, value in dict(response.headers).items()
                if str(key).lower() != "content-length"
            },
            status_code=response.status_code,
            media_type=response.media_type or "text/event-stream",
        )
        validated_response = inherit_or_wrap_passthrough_streaming_response(
            reconstructed,
            source_response=response,
        )
        _set_stream_validation_state(
            validated_response,
            {
                "complete": True,
                "valid": True,
                "terminal_seen": True,
                "terminal_status": response_body.get("status"),
                "reason": "validated_terminal_response",
            },
        )
        return validated_response

    if isinstance(response, Response) and not isinstance(response, StreamingResponse):
        try:
            response_body = json.loads(_decode_http_response_body(response.body))  # noqa: F821
        except Exception:
            _raise_codex_auto_agent_invalid_responses_shape(
                response_body=response.body,
                adapter_model=adapter_model,
                adapter=adapter,
                adapter_label=adapter_label,
            )
        if (
            isinstance(response_body, dict)
            and _responses_body_is_unsuccessful(response_body)  # noqa: F821
            and not is_repetitive_output_loop_failure(response_body)
        ):
            _raise_codex_auto_agent_failed_responses_payload(
                response_body=response_body,
                adapter_model=adapter_model,
                adapter=adapter,
                adapter_label=adapter_label,
            )
        identity_changed = False
        if isinstance(response_body, dict):
            preserved_body = _preserve_distinct_function_call_identity_fields(
                response_body
            )
            identity_changed = preserved_body is not response_body
            response_body = preserved_body
        if isinstance(response_body, dict):
            repaired_body = (
                _try_repair_codex_auto_agent_grok_native_composer_literal_tool_call_response_body(  # noqa: F821
                    response_body,
                    request_body=request_body,
                )
                if adapter in {
                    "codex_auto_agent_grok_native_responses",
                    "codex_auto_agent_xai_oauth_responses",
                }
                else None
            )
            if isinstance(repaired_body, dict):
                response_body = repaired_body
            (
                restored_body,
                restored_custom_tool_count,
                custom_tool_adapter_error,
            ) = _restore_adapted_custom_tool_calls_in_response_body(  # noqa: F821
                response_body,
                request_body=request_body,
                adapter_model=adapter_model,
            )
            if custom_tool_adapter_error is not None:
                _raise_codex_auto_agent_malformed_adapted_custom_tool_call(  # noqa: F821
                    response_body=response_body,
                    adapter_model=adapter_model,
                    adapter=adapter,
                    adapter_label=adapter_label,
                    adapter_error=custom_tool_adapter_error,
                )
            if restored_custom_tool_count:
                response_body = restored_body
            (
                restored_body,
                restored_namespace_tool_count,
            ) = _restore_adapted_namespace_tool_calls_in_response_body(  # noqa: F821
                response_body,
                request_body=request_body,
                adapter_model=adapter_model,
            )
            if restored_namespace_tool_count:
                response_body = restored_body
            if _is_codex_auto_agent_malformed_tool_call_text_output(response_body):
                _raise_codex_auto_agent_malformed_tool_call_text_payload(
                    response_body=response_body,
                    adapter_model=adapter_model,
                    adapter=adapter,
                    adapter_label=adapter_label,
                    intake_context=intake_context,
                )
            if (
                not isinstance(response_body, dict)
                or response_body.get("status") != "completed"
                or not _is_responses_shaped_body(response_body)
            ):
                _raise_codex_auto_agent_invalid_responses_shape(
                    response_body=response_body,
                    adapter_model=adapter_model,
                    adapter=adapter,
                    adapter_label=adapter_label,
                )
            # Serialize identity repairs even when no unrelated repair flag is set.
            if (
                identity_changed
                or isinstance(repaired_body, dict)
                or restored_custom_tool_count
                or restored_namespace_tool_count
            ):
                repaired_response = Response(
                    content=json.dumps(response_body),
                    media_type=response.media_type or "application/json",
                    status_code=response.status_code,
                    headers={
                        key: value
                        for key, value in dict(response.headers).items()
                        if str(key).lower() != "content-length"
                    },
                )
                return inherit_deferred_success_holder(
                    repaired_response,
                    source_response=response,
                )
        if not isinstance(response_body, dict):
            _raise_codex_auto_agent_invalid_responses_shape(
                response_body=response_body,
                adapter_model=adapter_model,
                adapter=adapter,
                adapter_label=adapter_label,
            )
    return response
