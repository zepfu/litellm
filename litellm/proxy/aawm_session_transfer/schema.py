"""Content-free session-transfer status contract (D1-617)."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple


SCHEMA_VERSION = "session-transfer-status.v1"
TRANSFER_ROUTE = "/internal/aawm/session-transfer-status"
TRANSFER_PERMISSION = "aawm_session_transfer_status"

ACTIVE_PHASES: Tuple[str, ...] = (
    "request_received",
    "request_preparing",
    "awaiting_upstream",
    "response_streaming",
    "finalizing",
)
TERMINAL_PHASES: Tuple[str, ...] = (
    "completed",
    "failed",
    "cancelled",
    "disconnected",
    "timed_out",
)
TRANSFER_PHASES: Tuple[str, ...] = ACTIVE_PHASES + TERMINAL_PHASES
TERMINAL_PHASE_SET = frozenset(TERMINAL_PHASES)
ACTIVE_PHASE_SET = frozenset(ACTIVE_PHASES)

STREAM_PATHS: Tuple[str, ...] = ("pass_through", "adapter", "unknown")
FRESHNESS_STATES: Tuple[str, ...] = ("live", "stale", "terminal", "unavailable")
REGISTRY_STATES: Tuple[str, ...] = ("ok", "degraded", "unavailable")
ERROR_CODES: Tuple[str, ...] = (
    "timeout",
    "disconnect",
    "cancelled",
    "upstream_error",
    "internal",
)

MAX_IDENTITY_CHARS = 240
MAX_LABEL_CHARS = 120
MAX_ERROR_CLASS_CHARS = 80
MAX_QUERY_RESULTS = 50
DEFAULT_QUERY_LIMIT = 20
MAX_INDEX_MEMBERS = 64

PROMPT_CATEGORY_FIELDS: Tuple[str, ...] = (
    "system",
    "tool_advertisement",
    "conversation",
    "other",
    "residual",
    "system_behavior",
    "system_safety",
    "system_instructional",
    "system_unclassified",
)

_SECRET_PREFIXES = ("bearer ", "sk-", "pk-", "xai-", "ya29.")
_BLOCKED_RESPONSE_KEYS = frozenset(
    {
        "prompt",
        "prompts",
        "messages",
        "content",
        "delta",
        "reasoning",
        "tool_calls",
        "tool_arguments",
        "tool_results",
        "arguments",
        "input",
        "output",
        "body",
        "payload",
        "headers",
        "raw_headers",
        "authorization",
        "api_key",
        "credential",
        "credentials",
        "redis_key",
        "redis_keys",
        "cache_key",
        "exception",
        "traceback",
        "error_message",
        "detail",
    }
)


def utc_now_iso(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, datetime):
        dt = value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)
        text = dt.astimezone(timezone.utc).isoformat()
        if text.endswith("+00:00"):
            text = text[:-6] + "Z"
        return text
    return sanitize_identity(value)


def sanitize_identity(value: Any, *, max_chars: int = MAX_IDENTITY_CHARS) -> Optional[str]:
    if value is None or isinstance(value, (dict, list, tuple, set)):
        return None
    if not isinstance(value, (str, int, float)):
        return None
    cleaned = "".join(
        char if char.isprintable() and char not in "\r\n\t" else " "
        for char in str(value).strip()
    )
    cleaned = " ".join(cleaned.split())
    if not cleaned:
        return None
    if cleaned.lower().startswith(_SECRET_PREFIXES):
        return None
    if len(cleaned) > max_chars:
        return cleaned[: max_chars - 3] + "..."
    return cleaned


def sanitize_label(value: Any) -> Optional[str]:
    return sanitize_identity(value, max_chars=MAX_LABEL_CHARS)


def sanitize_route_label(value: Any) -> Optional[str]:
    cleaned = sanitize_label(value)
    if cleaned is None:
        return None
    try:
        from urllib.parse import urlparse

        parsed = urlparse(cleaned)
        if parsed.scheme and parsed.hostname:
            host = parsed.hostname
            if parsed.port is not None:
                host = f"{host}:{parsed.port}"
            return f"{parsed.scheme}://{host}{parsed.path or '/'}"
    except Exception:
        return cleaned
    return cleaned


def coerce_non_negative_int(value: Any) -> Optional[int]:
    if value is None or isinstance(value, bool):
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    if parsed < 0:
        return None
    return parsed


def empty_prompt_category_tokens() -> Dict[str, int]:
    return {field: 0 for field in PROMPT_CATEGORY_FIELDS}


def sanitize_prompt_category_tokens(value: Any) -> Dict[str, int]:
    categories = empty_prompt_category_tokens()
    if not isinstance(value, Mapping):
        return categories
    for field in PROMPT_CATEGORY_FIELDS:
        parsed = coerce_non_negative_int(value.get(field))
        if parsed is not None:
            categories[field] = parsed
    return categories


def normalize_phase(value: Any) -> str:
    cleaned = sanitize_label(value)
    if cleaned in TRANSFER_PHASES:
        return cleaned
    return "request_received"


def normalize_stream_path(value: Any) -> str:
    cleaned = sanitize_label(value)
    if cleaned in STREAM_PATHS:
        return cleaned
    return "unknown"


def is_terminal_phase(phase: str) -> bool:
    return phase in TERMINAL_PHASE_SET


def new_transfer_record() -> Dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "litellm_call_id": None,
        "trace_id": None,
        "canonical_session_id": None,
        "session_id": None,
        "codex_session_id": None,
        "agent_id": None,
        "agent_name": None,
        "parent_agent_id": None,
        "parent_session_id": None,
        "provider": None,
        "model": None,
        "route": None,
        "stream_path": "unknown",
        "source_instance": None,
        "phase": "request_received",
        "active": True,
        "stale": False,
        "redis_degraded": False,
        "freshness": "live",
        "received_at": None,
        "preparing_at": None,
        "awaiting_upstream_at": None,
        "first_upstream_chunk_at": None,
        "first_downstream_chunk_at": None,
        "last_heartbeat_at": None,
        "finalized_at": None,
        "upstream_chunk_count": 0,
        "upstream_byte_count": 0,
        "downstream_chunk_count": 0,
        "downstream_byte_count": 0,
        "context_window": None,
        "estimated_input_tokens": None,
        "estimated_output_tokens": None,
        "provider_input_tokens": None,
        "provider_output_tokens": None,
        "remaining_tokens": None,
        "request_count": 1,
        "cumulative_input_tokens": None,
        "repeated_prefix_tokens": None,
        "prompt_category_tokens": empty_prompt_category_tokens(),
        "terminal_state": None,
        "disconnect_reason": None,
        "timeout_kind": None,
        "error_code": None,
        "error_class": None,
    }


def public_transfer_record(record: Mapping[str, Any]) -> Dict[str, Any]:
    """Return the consumer-visible subset. Never include Redis keys or content."""
    phase = normalize_phase(record.get("phase"))
    prompt_category_tokens = sanitize_prompt_category_tokens(
        record.get("prompt_category_tokens")
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "litellm_call_id": sanitize_identity(record.get("litellm_call_id")),
        "trace_id": sanitize_identity(record.get("trace_id")),
        "canonical_session_id": sanitize_identity(record.get("canonical_session_id")),
        "session_id": sanitize_identity(record.get("session_id")),
        "codex_session_id": sanitize_identity(record.get("codex_session_id")),
        "agent_id": sanitize_identity(record.get("agent_id")),
        "agent_name": sanitize_label(record.get("agent_name")),
        "parent_agent_id": sanitize_identity(record.get("parent_agent_id")),
        "parent_session_id": sanitize_identity(record.get("parent_session_id")),
        "provider": sanitize_label(record.get("provider")),
        "model": sanitize_label(record.get("model")),
        "route": sanitize_route_label(record.get("route")),
        "stream_path": normalize_stream_path(record.get("stream_path")),
        "source_instance": sanitize_label(record.get("source_instance")),
        "phase": phase,
        "active": bool(record.get("active")) and not is_terminal_phase(phase),
        "stale": bool(record.get("stale")),
        "redis_degraded": bool(record.get("redis_degraded")),
        "freshness": (
            record.get("freshness")
            if record.get("freshness") in FRESHNESS_STATES
            else "live"
        ),
        "received_at": sanitize_identity(record.get("received_at")),
        "preparing_at": sanitize_identity(record.get("preparing_at")),
        "awaiting_upstream_at": sanitize_identity(record.get("awaiting_upstream_at")),
        "first_upstream_chunk_at": sanitize_identity(
            record.get("first_upstream_chunk_at")
        ),
        "first_downstream_chunk_at": sanitize_identity(
            record.get("first_downstream_chunk_at")
        ),
        "last_heartbeat_at": sanitize_identity(record.get("last_heartbeat_at")),
        "finalized_at": sanitize_identity(record.get("finalized_at")),
        "upstream_chunk_count": coerce_non_negative_int(
            record.get("upstream_chunk_count")
        )
        or 0,
        "upstream_byte_count": coerce_non_negative_int(record.get("upstream_byte_count"))
        or 0,
        "downstream_chunk_count": coerce_non_negative_int(
            record.get("downstream_chunk_count")
        )
        or 0,
        "downstream_byte_count": coerce_non_negative_int(
            record.get("downstream_byte_count")
        )
        or 0,
        "context": {
            "context_window": coerce_non_negative_int(record.get("context_window")),
            "estimated_input_tokens": coerce_non_negative_int(
                record.get("estimated_input_tokens")
            ),
            "estimated_output_tokens": coerce_non_negative_int(
                record.get("estimated_output_tokens")
            ),
            "provider_input_tokens": coerce_non_negative_int(
                record.get("provider_input_tokens")
            ),
            "provider_output_tokens": coerce_non_negative_int(
                record.get("provider_output_tokens")
            ),
            "remaining_tokens": coerce_non_negative_int(record.get("remaining_tokens")),
            "request_count": coerce_non_negative_int(record.get("request_count")) or 1,
            "cumulative_input_tokens": coerce_non_negative_int(
                record.get("cumulative_input_tokens")
            ),
            "repeated_prefix_tokens": coerce_non_negative_int(
                record.get("repeated_prefix_tokens")
            ),
            "prompt_category_tokens": prompt_category_tokens,
        },
        "terminal_state": (
            sanitize_label(record.get("terminal_state"))
            if is_terminal_phase(phase)
            else None
        ),
        "disconnect_reason": sanitize_label(record.get("disconnect_reason")),
        "timeout_kind": sanitize_label(record.get("timeout_kind")),
        "error_code": (
            record.get("error_code")
            if record.get("error_code") in ERROR_CODES
            else None
        ),
        "error_class": sanitize_identity(
            record.get("error_class"), max_chars=MAX_ERROR_CLASS_CHARS
        ),
    }


def assert_content_free(payload: Any) -> None:
    """Raise ValueError if a public payload contains blocked content keys."""

    def _walk(value: Any, path: str) -> None:
        if isinstance(value, Mapping):
            for key, child in value.items():
                key_text = str(key)
                lowered = key_text.lower()
                if lowered in _BLOCKED_RESPONSE_KEYS or "redis_key" in lowered:
                    raise ValueError(f"blocked key at {path}.{key_text}")
                _walk(child, f"{path}.{key_text}")
        elif isinstance(value, (list, tuple)):
            for index, child in enumerate(value):
                _walk(child, f"{path}[{index}]")

    _walk(payload, "payload")


def merge_records(
    current: Mapping[str, Any], incoming: Mapping[str, Any]
) -> Dict[str, Any]:
    merged = dict(current)
    for key, value in incoming.items():
        if key == "prompt_category_tokens":
            merged[key] = sanitize_prompt_category_tokens(value)
            continue
        if key in {
            "upstream_chunk_count",
            "upstream_byte_count",
            "downstream_chunk_count",
            "downstream_byte_count",
        }:
            parsed = coerce_non_negative_int(value)
            if parsed is not None:
                merged[key] = parsed
            continue
        if value is None or value == "":
            continue
        if (
            key.endswith("_at")
            and key != "last_heartbeat_at"
            and merged.get(key)
        ):
            continue
        merged[key] = value
    return merged


def clamp_limit(value: Any) -> int:
    parsed = coerce_non_negative_int(value)
    if parsed is None or parsed == 0:
        return DEFAULT_QUERY_LIMIT
    return min(parsed, MAX_QUERY_RESULTS)


def iter_identity_values(record: Mapping[str, Any]) -> Iterable[Tuple[str, str]]:
    for field in (
        "litellm_call_id",
        "canonical_session_id",
        "session_id",
        "codex_session_id",
        "agent_id",
    ):
        cleaned = sanitize_identity(record.get(field))
        if cleaned:
            yield field, cleaned
