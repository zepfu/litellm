"""Bounded schema-rejection diagnostics for AAWM route telemetry."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from itertools import islice
from typing import Any, Optional

SCHEMA_REJECTION_KEY = "schema_rejection"
SCHEMA_REJECTION_FAILURE_CLASS = "schema_rejection"
SCHEMA_REJECTION_ERROR_CODE = "schema_rejection"
_GENERIC_FAILURE_IDENTITIES = frozenset(
    {
        "provider_terminal_error",
        "unclassified",
    }
)
_CURSOR_REJECTION_KEY = "cursor_replay_fresh_dispatch_reject"
_MAX_INDEX = 4096
_MAX_SAFE_KEYS = 32
_MAX_LOC_SEGMENTS = 16
_MAX_VALIDATION_ERRORS = 8
_MAX_IDENTIFIER_CHARS = 96
_MAX_TOKEN_CHARS = 64
_MAX_PATH_CHARS = 128
_IDENTIFIER_RE = re.compile(
    rf"[A-Za-z0-9][A-Za-z0-9_.:-]{{0,{_MAX_IDENTIFIER_CHARS - 1}}}\Z"
)
_TOKEN_RE = re.compile(rf"[A-Za-z][A-Za-z0-9_.:-]{{0,{_MAX_TOKEN_CHARS - 1}}}\Z")
_KEY_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_-]{0,63}\Z")
_PATH_RE = re.compile(
    r"[A-Za-z_][A-Za-z0-9_-]{0,63}"
    r"(?:\[(?:0|[1-9][0-9]{0,3})\])?"
    r"(?:\.[A-Za-z_][A-Za-z0-9_-]{0,63}"
    r"(?:\[(?:0|[1-9][0-9]{0,3})\])?)*\Z"
)
_PATH_SEGMENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_-]*")
_UUID_RE = re.compile(
    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\Z",
    re.IGNORECASE,
)
_HEX_ID_RE = re.compile(r"[0-9a-f]{16,}\Z", re.IGNORECASE)
_SENSITIVE_KEY_PARTS = (
    "api_key",
    "apikey",
    "authorization",
    "cookie",
    "credential",
    "password",
    "secret",
    "token",
)

# This is the contract-owned copy of the fixed Cursor replay rejection
# vocabulary. Unknown Cursor reasons are dropped rather than serialized.
_CURSOR_REASONS = frozenset(
    {
        "request_body_shape",
        "missing_replay_state",
        "replay_state_lookup",
        "replay_state_copy",
        "invalid_replay_state",
        "retained_session_present",
        "messages_container",
        "messages_empty",
        "message_role",
        "message_conversion",
        "replayed_input_container",
        "input_container",
        "input_item_count",
        "item_not_object",
        "item_key_set",
        "item_type",
        "id_shape",
        "metadata_shape",
        "metadata_key_set",
        "metadata_value_type",
        "content_container",
        "content_part_container",
        "content_part_keys",
        "content_part_type",
        "content_part_text_type",
        "empty_user_text",
        "function_call_position",
        "function_call_count",
        "function_call_output_position",
        "function_call_output_count",
        "function_call_fields",
        "call_id_shape",
        "call_id_alias_mismatch",
        "function_name",
        "arguments_not_object",
        "output_container",
        "output_not_string",
        "call_graph",
        "unresolved_call_id",
        "tool_container",
        "tool_item",
        "tool_type_validation",
        "tool_key_set",
        "tool_name",
        "tool_transform",
        "tool_validation",
        "tool_canonical_mismatch",
        "tool_copy_failure",
        "copy_failure",
        "replay_safety_rejected",
        "previous_response_id",
        "id_only_reasoning_reference",
        "explicit_item_reference",
        "invalid_body_shape",
    }
)
_UPSTREAM_SCHEMA_CODES = frozenset(
    {
        "invalid_function_parameters",
        "invalid_schema",
        "invalid_tool_schema",
        "schema_error",
    }
)
_UPSTREAM_SCHEMA_TYPES = frozenset(
    {"invalid_schema", "schema_error", "schema_validation_error"}
)
_UPSTREAM_ERROR_CLASSES = frozenset(
    {
        "bad_request",
        "invalid_request",
        "invalid_request_error",
        "schema_error",
        "unprocessable_entity",
        "validation_error",
    }
)
_LOCAL_VALIDATION_TYPES = frozenset(
    {
        "assertion_error",
        "bool_parsing",
        "bool_type",
        "date_parsing",
        "datetime_parsing",
        "dict_type",
        "enum",
        "extra_forbidden",
        "float_parsing",
        "float_type",
        "int_parsing",
        "int_type",
        "invalid_key",
        "json_type",
        "less_than",
        "list_type",
        "literal_error",
        "mapping_type",
        "missing",
        "model_attributes_type",
        "model_type",
        "none_required",
        "set_type",
        "string_type",
        "time_parsing",
        "too_long",
        "too_short",
        "tuple_type",
        "type_error",
        "value_error",
    }
)
_LOCAL_VALIDATION_TYPE_ALIASES = {
    "type_error.dict": "dict_type",
    "type_error.list": "list_type",
    "type_error.none.not_allowed": "none_required",
    "type_error.str": "string_type",
    "value_error.any_str.max_length": "too_long",
    "value_error.any_str.min_length": "too_short",
    "value_error.extra": "extra_forbidden",
    "value_error.missing": "missing",
}
_REASON_VALUES = _CURSOR_REASONS | _UPSTREAM_SCHEMA_CODES | {
    "schema_validation_error",
} | _LOCAL_VALIDATION_TYPES
_STAGES = frozenset(
    {
        "candidate_preflight",
        "request_preparation",
        "provider_dispatch",
        "upstream_4xx",
        "cursor_replay",
        "stock_full_history",
        "provider_neutral_tools",
        "fresh_body_copy",
        "rebuilt_body_replay_unsafe",
    }
)
_LOCAL_VALIDATION_PHASES = frozenset(
    {
        "candidate_preflight",
        "request_preparation",
        "provider_neutral_tools",
        "stock_full_history",
    }
)
_CATEGORIES = frozenset(
    {
        "input_schema",
        "request_shape",
        "schema_validation",
        "tool_schema",
        "cursor_replay",
    }
)
_SCHEMA_CATEGORIES = frozenset(
    {
        "input_schema",
        "request_shape",
        "schema_validation",
        "tool_schema",
    }
)
_STAGE_ALIASES = {
    "pre_egress": "candidate_preflight",
    "preflight": "candidate_preflight",
    "adapter": "request_preparation",
    "upstream": "upstream_4xx",
}
_CATEGORY_ALIASES = {
    "input": "input_schema",
    "schema": "schema_validation",
    "tool": "tool_schema",
    "validation": "schema_validation",
}
_OBJECT_TYPE_ALIASES = {
    "bool": "boolean",
    "boolean_type": "boolean",
    "dict_type": "dict",
    "float": "number",
    "float_type": "number",
    "int": "integer",
    "int_type": "integer",
    "list_type": "list",
    "str": "string",
    "string_type": "string",
    "tuple_type": "tuple",
}
_OBJECT_TYPES = frozenset(
    {
        "boolean",
        "content",
        "dict",
        "function",
        "function_call",
        "function_call_output",
        "input_text",
        "integer",
        "list",
        "mcp_approval_request",
        "mcp_approval_response",
        "mcp_call",
        "mcp_tool_call",
        "message",
        "model",
        "null",
        "number",
        "object",
        "output_text",
        "reasoning",
        "request",
        "response",
        "schema",
        "set",
        "string",
        "tool",
        "tool_result",
        "tool_use",
        "tuple",
        "unknown",
    }
)
_VALIDATION_TYPE_OBJECT_TYPES = {
    "bool_parsing": "boolean",
    "bool_type": "boolean",
    "dict_type": "dict",
    "float_parsing": "number",
    "float_type": "number",
    "int_parsing": "integer",
    "int_type": "integer",
    "list_type": "list",
    "mapping_type": "object",
    "model_attributes_type": "model",
    "model_type": "model",
    "set_type": "set",
    "string_type": "string",
    "tuple_type": "tuple",
}

# Only schema/request field names are eligible for a telemetry path. This
# keeps property values, IDs, and opaque user-controlled path components out.
_SAFE_PATH_SEGMENTS = frozenset(
    {
        "arguments",
        "body",
        "call_id",
        "choices",
        "content",
        "content_item_kinds",
        "content_part",
        "content_parts",
        "data",
        "description",
        "error",
        "errors",
        "field",
        "function",
        "function_call",
        "function_call_output",
        "id",
        "index",
        "input",
        "input_items",
        "input_text",
        "instructions",
        "internal_chat_message_metadata_passthrough",
        "items",
        "json_schema",
        "loc",
        "max_tokens",
        "message",
        "messages",
        "metadata",
        "mcp_approval_request",
        "mcp_approval_response",
        "mcp_call",
        "mcp_tool_call",
        "model",
        "name",
        "object",
        "object_type",
        "output",
        "output_items",
        "output_text",
        "parameters",
        "parallel_tool_calls",
        "previous_response_id",
        "properties",
        "prompt",
        "reasoning",
        "request",
        "required",
        "response",
        "response_format",
        "role",
        "root",
        "schema",
        "store",
        "stream",
        "text",
        "tool",
        "tool_call",
        "tool_calls",
        "tool_choice",
        "tool_result",
        "tool_use",
        "tools",
        "type",
        "value",
    }
)
_STRUCTURED_ERROR_KEYS = frozenset(
    {
        "category",
        "code",
        "error_class",
        "error_code",
        "error_type",
        "field_path",
        "keys",
        "loc",
        "object_type",
        "param",
        "path",
        "safe_keys",
        "schema_category",
        "schema_error",
        "type",
    }
)
_STRUCTURED_ERROR_COPY_KEYS = (
    "category",
    "code",
    "error_class",
    "error_code",
    "error_type",
    "field_path",
    "failure_phase",
    "failure_reason",
    "item_index",
    "item_keys",
    "item_type",
    "keys",
    "loc",
    "object_type",
    "param",
    "path",
    "phase",
    "reason",
    "safe_keys",
    "schema_category",
    "schema_error",
    "stage",
    "tool_index",
    "tool_keys",
    "tool_type",
    "type",
    "upstream_error_class",
    "upstream_error_code",
)


@dataclass(frozen=True)
class SchemaRejectionDiagnostic:
    """Sanitized schema-rejection evidence safe for telemetry serialization."""

    provider: str
    route_family: str
    stage: str
    reason: str
    category: str
    object_type: str = "unknown"
    safe_keys: tuple[str, ...] = ()
    item_index: Optional[int] = None
    tool_index: Optional[int] = None
    upstream_status: Optional[int] = None
    upstream_error_class: Optional[str] = None
    upstream_error_code: Optional[str] = None
    field_path: Optional[str] = None
    schema_category: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "provider": self.provider,
            "route_family": self.route_family,
            "stage": self.stage,
            "reason": self.reason,
            "category": self.category,
            "object_type": self.object_type,
            "safe_keys": list(self.safe_keys),
        }
        for field in (
            "item_index",
            "tool_index",
            "upstream_status",
            "upstream_error_class",
            "upstream_error_code",
            "field_path",
            "schema_category",
        ):
            if (value := getattr(self, field)) is not None:
                result[field] = value
        return result


def resolve_schema_rejection_failure_identity(
    *,
    failure_class: Any,
    error_code: Any,
) -> tuple[Any, Any]:
    """Preserve specific identities while replacing missing or generic ones."""

    def _resolved(value: Any, fallback: str) -> Any:
        if value is None:
            return fallback
        if isinstance(value, str):
            normalized = value.strip()
            if not normalized or normalized in _GENERIC_FAILURE_IDENTITIES:
                return fallback
        return value

    return (
        _resolved(failure_class, SCHEMA_REJECTION_FAILURE_CLASS),
        _resolved(error_code, SCHEMA_REJECTION_ERROR_CODE),
    )


def _safe_text(value: Any, pattern: re.Pattern[str]) -> Optional[str]:
    if not isinstance(value, str) or pattern.fullmatch(value) is None:
        return None
    if _UUID_RE.fullmatch(value) or _HEX_ID_RE.fullmatch(value):
        return None
    return value


def _safe_key(value: Any) -> Optional[str]:
    value = _safe_text(value, _KEY_RE)
    if value is None:
        return None
    normalized = value.casefold().replace("-", "_")
    return None if any(part in normalized for part in _SENSITIVE_KEY_PARTS) else value


def _safe_path_segment(value: Any) -> Optional[str]:
    value = _safe_text(value, _KEY_RE)
    if value is None:
        return None
    normalized = value.casefold().replace("-", "_")
    if any(part in normalized for part in _SENSITIVE_KEY_PARTS):
        return None
    if value not in _SAFE_PATH_SEGMENTS:
        return None
    return value


def _safe_path(value: Any) -> Optional[str]:
    if (
        not isinstance(value, str)
        or len(value) > _MAX_PATH_CHARS
        or _PATH_RE.fullmatch(value) is None
    ):
        return None
    segments = _PATH_SEGMENT_RE.findall(value)
    if not segments or any(_safe_path_segment(segment) is None for segment in segments):
        return None
    return value


def _path_from_loc(value: Any) -> Optional[str]:
    if (
        not isinstance(value, (list, tuple))
        or not value
        or len(value) > _MAX_LOC_SEGMENTS
    ):
        return None
    path = ""
    for segment in value:
        if isinstance(segment, int) and not isinstance(segment, bool):
            index = _bounded_index(segment)
            if index is None or not path:
                return None
            path += f"[{index}]"
            continue
        safe_segment = _safe_path_segment(segment)
        if safe_segment is None:
            return None
        path += f".{safe_segment}" if path else safe_segment
    return _safe_path(path)


def _safe_reason(value: Any) -> Optional[str]:
    if not isinstance(value, str):
        return None
    value = _LOCAL_VALIDATION_TYPE_ALIASES.get(value, value)
    return value if value in _REASON_VALUES else None


def _safe_upstream_error_class(value: Any) -> Optional[str]:
    return value if isinstance(value, str) and value in _UPSTREAM_ERROR_CLASSES else None


def _safe_upstream_error_code(value: Any) -> Optional[str]:
    return value if isinstance(value, str) and value in _UPSTREAM_SCHEMA_CODES else None


def _safe_schema_category(value: Any) -> Optional[str]:
    if not isinstance(value, str):
        return None
    value = _CATEGORY_ALIASES.get(value, value)
    return value if value in _SCHEMA_CATEGORIES else None


def _safe_object_type(value: Any) -> Optional[str]:
    if not isinstance(value, str):
        return None
    value = _OBJECT_TYPE_ALIASES.get(value, value)
    return value if value in _OBJECT_TYPES else None


def _safe_keys(*values: Any) -> tuple[str, ...]:
    result: set[str] = set()
    for value in values:
        if isinstance(value, Mapping):
            iterable = value.keys()
        elif isinstance(value, (list, tuple, set, frozenset)):
            iterable = value
        else:
            continue
        for raw_key in islice(iterable, _MAX_SAFE_KEYS):
            if (key := _safe_key(raw_key)) is not None:
                result.add(key)
    return tuple(sorted(result)[:_MAX_SAFE_KEYS])


def _bounded_index(value: Any) -> Optional[int]:
    if (
        isinstance(value, int)
        and not isinstance(value, bool)
        and 0 <= value <= _MAX_INDEX
    ):
        return value
    return None


def _enum(
    value: Any,
    *,
    aliases: Mapping[str, str],
    allowed: frozenset[str],
    default: str,
) -> Optional[str]:
    value = default if value is None else value
    if not isinstance(value, str):
        return None
    value = aliases.get(value, value)
    return value if value in allowed else None


def _status(value: Any) -> Optional[int]:
    if isinstance(value, bool) or value is None:
        return None
    try:
        value = int(value)
    except (TypeError, ValueError):
        return None
    return value if 400 <= value <= 599 else None


def _exception_status(exc: Any) -> Optional[int]:
    for source in (exc, getattr(exc, "response", None)):
        if (value := _status(getattr(source, "status_code", None))) is not None:
            return value
    return None


def sanitize_cursor_replay_rejection(value: Any) -> Optional[dict[str, Any]]:
    """Bridge the existing Cursor diagnostic into a bounded mapping."""
    if not isinstance(value, Mapping):
        return None
    stage = _enum(
        value.get("stage"),
        aliases=_STAGE_ALIASES,
        allowed=_STAGES,
        default="cursor_replay",
    )
    reason = _safe_reason(value.get("reason"))
    if stage is None or reason not in _CURSOR_REASONS:
        return None
    result: dict[str, Any] = {"stage": stage, "reason": reason}
    for field in ("item_index", "tool_index"):
        if (index := _bounded_index(value.get(field))) is not None:
            result[field] = index
    for field in ("item_type", "tool_type"):
        if (token := _safe_object_type(value.get(field))) is not None:
            result[field] = token
    for field in ("item_keys", "tool_keys"):
        if keys := _safe_keys(value.get(field)):
            result[field] = list(keys)
    return result


def normalize_schema_rejection(
    value: Any,
    *,
    provider: Optional[str],
    route_family: Optional[str],
    default_stage: str = "candidate_preflight",
    default_category: str = "schema_validation",
    upstream_status: Optional[int] = None,
    upstream_error_class: Optional[str] = None,
    upstream_error_code: Optional[str] = None,
) -> Optional[SchemaRejectionDiagnostic]:
    """Normalize one explicit structured diagnostic into the shared contract."""
    if not isinstance(value, Mapping):
        return None
    outer = value
    nested_payload = outer.get(SCHEMA_REJECTION_KEY)
    payload = nested_payload if isinstance(nested_payload, Mapping) else outer
    provider = (
        provider
        if provider is not None
        else payload.get("provider") or outer.get("provider")
    )
    route_family = (
        route_family
        if route_family is not None
        else payload.get("route_family") or outer.get("route_family")
    )
    legacy = payload.get(_CURSOR_REJECTION_KEY)
    legacy = legacy if isinstance(legacy, Mapping) else outer.get(_CURSOR_REJECTION_KEY)
    if isinstance(legacy, Mapping):
        legacy = sanitize_cursor_replay_rejection(legacy)
        if legacy is None:
            return None
        payload = {
            "stage": legacy["stage"],
            "reason": legacy["reason"],
            "category": "cursor_replay",
            "object_type": legacy.get("item_type") or legacy.get("tool_type"),
            "safe_keys": [
                *legacy.get("item_keys", []),
                *legacy.get("tool_keys", []),
            ],
            "item_index": legacy.get("item_index"),
            "tool_index": legacy.get("tool_index"),
        }
        default_stage = "cursor_replay"
        default_category = "cursor_replay"

    provider = _safe_text(provider, _IDENTIFIER_RE)
    route_family = _safe_text(route_family, _IDENTIFIER_RE)
    stage = _enum(
        payload.get("stage") or payload.get("failure_stage") or payload.get("phase"),
        aliases=_STAGE_ALIASES,
        allowed=_STAGES,
        default=default_stage,
    )
    category = _enum(
        payload.get("category") or payload.get("schema_category"),
        aliases=_CATEGORY_ALIASES,
        allowed=_CATEGORIES,
        default=default_category,
    )
    raw_reason = (
        payload.get("reason")
        or payload.get("failure_reason")
        or payload.get("upstream_error_code")
        or payload.get("code")
    )
    reason = _safe_reason(raw_reason)
    if reason is None and payload.get("schema_error") is True:
        reason = "schema_error"
    elif reason is None and category in _SCHEMA_CATEGORIES and (
        payload.get("field_path") is not None
        or payload.get("path") is not None
        or payload.get("param") is not None
        or payload.get("loc") is not None
    ):
        reason = "schema_validation_error"
    if None in (provider, route_family, stage, category, reason):
        return None

    field_path = _path_from_loc(payload.get("loc"))
    if field_path is None:
        field_path = _safe_path(
            payload.get("field_path") or payload.get("path") or payload.get("param")
        )
    object_type = _safe_object_type(
        payload.get("object_type")
        or payload.get("item_type")
        or payload.get("tool_type")
    )
    if object_type is None:
        object_type = "unknown"
    return SchemaRejectionDiagnostic(
        provider=provider,
        route_family=route_family,
        stage=stage,
        reason=reason,
        category=category,
        object_type=object_type,
        safe_keys=_safe_keys(
            payload.get("safe_keys"),
            payload.get("keys"),
            payload.get("item_keys"),
            payload.get("tool_keys"),
        ),
        item_index=_bounded_index(payload.get("item_index")),
        tool_index=_bounded_index(payload.get("tool_index")),
        upstream_status=_status(
            upstream_status
            if upstream_status is not None
            else payload.get("upstream_status") or payload.get("status_code")
        ),
        upstream_error_class=_safe_upstream_error_class(
            upstream_error_class
            if upstream_error_class is not None
            else payload.get("upstream_error_class")
            or payload.get("error_class")
            or payload.get("type")
        ),
        upstream_error_code=_safe_upstream_error_code(
            upstream_error_code
            if upstream_error_code is not None
            else payload.get("upstream_error_code")
            or payload.get("error_code")
            or payload.get("code")
        ),
        field_path=field_path,
        schema_category=_safe_schema_category(payload.get("schema_category")),
    )


def _structured_error(value: Any) -> Optional[Mapping[str, Any]]:
    if not isinstance(value, Mapping):
        return None
    error = value.get("error")
    if isinstance(error, Mapping):
        return error
    errors = value.get("errors")
    if isinstance(errors, (list, tuple)):
        for item in islice(errors, _MAX_VALIDATION_ERRORS):
            if isinstance(item, Mapping):
                return item
    return value if _STRUCTURED_ERROR_KEYS.intersection(value) else None


def _is_schema_error(error: Mapping[str, Any]) -> bool:
    if error.get("schema_error") is True:
        return True
    category = error.get("category") or error.get("schema_category")
    if category in _CATEGORIES or category in _CATEGORY_ALIASES:
        return True
    if (error.get("code") or error.get("error_code")) in _UPSTREAM_SCHEMA_CODES:
        return True
    if (error.get("type") or error.get("error_type")) in _UPSTREAM_SCHEMA_TYPES:
        return True
    code = error.get("code") or error.get("error_code")
    return code in {"invalid_request", "invalid_request_error"} and any(
        error.get(field) is not None
        for field in ("field_path", "loc", "path", "param", "safe_keys", "keys")
    )


def _copy_structured_error(error: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: error.get(key)
        for key in _STRUCTURED_ERROR_COPY_KEYS
        if key in error
    }


def _extract_structured(
    value: Any,
    *,
    provider: Optional[str],
    route_family: Optional[str],
    status: int,
) -> Optional[SchemaRejectionDiagnostic]:
    error = _structured_error(value)
    if error is None or not _is_schema_error(error):
        return None
    payload = _copy_structured_error(error)
    payload["stage"] = "upstream_4xx"
    return normalize_schema_rejection(
        payload,
        provider=provider,
        route_family=route_family,
        default_stage="upstream_4xx",
        upstream_status=status,
    )


def _optional_bool(value: Any) -> Optional[bool]:
    return value if isinstance(value, bool) else None


def _local_validation_phase(
    exc: Any,
    *,
    attempted_provider_call: Optional[bool],
    failure_phase: Optional[str],
) -> Optional[str]:
    attempted = (
        attempted_provider_call
        if attempted_provider_call is not None
        else _optional_bool(getattr(exc, "attempted_provider_call", None))
    )
    phase = (
        failure_phase
        if isinstance(failure_phase, str)
        else getattr(exc, "failure_phase", None)
    )
    if isinstance(phase, str) and phase in _LOCAL_VALIDATION_PHASES:
        return phase
    return "candidate_preflight" if attempted is False else None


def _structured_validation_list(exc: Any) -> Optional[list[Any] | tuple[Any, ...]]:
    for attribute in (
        "validation_errors",
        "structured_validation_errors",
        "_validation_errors",
    ):
        value = getattr(exc, attribute, None)
        if isinstance(value, (list, tuple)):
            return value

    errors = getattr(exc, "errors", None)
    if callable(errors):
        try:
            value = errors()
        except Exception:
            return None
        return value if isinstance(value, (list, tuple)) else None
    if isinstance(errors, (list, tuple)):
        return errors

    detail = getattr(exc, "detail", None)
    if isinstance(detail, Mapping):
        for key in ("validation_errors", "structured_validation_errors", "errors"):
            value = detail.get(key)
            if isinstance(value, (list, tuple)):
                return value
    return None


def _extract_local_validation(
    exc: Any,
    *,
    provider: Optional[str],
    route_family: Optional[str],
    attempted_provider_call: Optional[bool],
    failure_phase: Optional[str],
) -> Optional[SchemaRejectionDiagnostic]:
    stage = _local_validation_phase(
        exc,
        attempted_provider_call=attempted_provider_call,
        failure_phase=failure_phase,
    )
    if stage is None:
        return None
    errors = _structured_validation_list(exc)
    if errors is None:
        return None
    for error in islice(errors, _MAX_VALIDATION_ERRORS):
        if not isinstance(error, Mapping):
            continue
        validation_type = _LOCAL_VALIDATION_TYPE_ALIASES.get(error.get("type"))
        if validation_type is None and isinstance(error.get("type"), str):
            validation_type = error.get("type")
        if validation_type not in _LOCAL_VALIDATION_TYPES:
            continue
        object_type = _safe_object_type(error.get("object_type"))
        if object_type is None:
            object_type = _VALIDATION_TYPE_OBJECT_TYPES.get(validation_type, "unknown")
        path = _path_from_loc(error.get("loc"))
        loc_keys = (
            [
                segment
                for segment in error.get("loc", ())
                if isinstance(segment, str)
            ]
            if isinstance(error.get("loc"), (list, tuple))
            else []
        )
        payload: dict[str, Any] = {
            "stage": stage,
            "reason": validation_type,
            "category": "schema_validation",
            "object_type": object_type,
            "safe_keys": _safe_keys(
                error.get("safe_keys"),
                error.get("keys"),
                loc_keys,
            ),
        }
        if path is not None:
            payload["loc"] = error.get("loc")
        diagnostic = normalize_schema_rejection(
            payload,
            provider=provider,
            route_family=route_family,
            default_stage=stage,
            default_category="schema_validation",
        )
        if diagnostic is not None:
            return diagnostic
    return None


def extract_schema_rejection(
    exc: Any,
    *,
    provider: Optional[str],
    route_family: Optional[str],
    attempted_provider_call: Optional[bool] = None,
    failure_phase: Optional[str] = None,
) -> Optional[SchemaRejectionDiagnostic]:
    """Extract explicit metadata or bounded validation/provider evidence."""
    for attribute in (
        SCHEMA_REJECTION_KEY,
        "_aawm_schema_rejection",
        "schema_rejection_diagnostic",
    ):
        diagnostic = normalize_schema_rejection(
            getattr(exc, attribute, None),
            provider=provider,
            route_family=route_family,
        )
        if diagnostic is not None:
            return diagnostic

    legacy = getattr(exc, _CURSOR_REJECTION_KEY, None)
    if isinstance(legacy, Mapping):
        diagnostic = normalize_schema_rejection(
            {_CURSOR_REJECTION_KEY: legacy},
            provider=provider,
            route_family=route_family,
        )
        if diagnostic is not None:
            return diagnostic

    diagnostic = _extract_local_validation(
        exc,
        provider=provider,
        route_family=route_family,
        attempted_provider_call=attempted_provider_call,
        failure_phase=failure_phase,
    )
    if diagnostic is not None:
        return diagnostic

    status = _exception_status(exc)
    response = getattr(exc, "response", None)
    if (
        status is None
        or status >= 500
        or (
            getattr(exc, "_aawm_provider_returned", False) is not True
            and getattr(response, "status_code", None) is None
        )
    ):
        return None
    values = [getattr(exc, "detail", None), getattr(exc, "body", None)]
    json_loader = getattr(response, "json", None)
    if callable(json_loader):
        try:
            values.append(json_loader())
        except Exception:
            pass
    for value in values:
        diagnostic = _extract_structured(
            value,
            provider=provider,
            route_family=route_family,
            status=status,
        )
        if diagnostic is not None:
            return diagnostic
    return None


__all__ = [
    "SCHEMA_REJECTION_ERROR_CODE",
    "SCHEMA_REJECTION_FAILURE_CLASS",
    "SCHEMA_REJECTION_KEY",
    "SchemaRejectionDiagnostic",
    "extract_schema_rejection",
    "normalize_schema_rejection",
    "resolve_schema_rejection_failure_identity",
    "sanitize_cursor_replay_rejection",
]
