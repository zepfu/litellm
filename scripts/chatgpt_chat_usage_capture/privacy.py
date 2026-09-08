"""Metadata-only sanitization for ChatGPT Chat usage capture.

The privacy promise is persistence and reporting of allowlisted metadata, not a
claim that upstream responses never contain content. Credentials, cookies,
tokens, raw headers, browser storage, titles, and message bodies are stripped
before any ledger, log, fixture-export, or report write.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from typing import Any, Iterable, Mapping, Optional

ADAPTER_VERSION = "chatgpt-chat-history-v1"
SURFACE_CHAT = "chat"

SENSITIVE_KEY_RE = re.compile(
    r"(authorization|auth[_-]?token|bearer|cookie|csrf|credential|email|password|"
    r"refresh[_-]?token|secret|session[_-]?token|set-cookie|access[_-]?token|"
    r"id[_-]?token|api[_-]?key|x-auth|storage)",
    re.IGNORECASE,
)
CONTENT_KEY_RE = re.compile(
    r"^(content|text|title|parts|body|prompt|answer|message|html|markdown|"
    r"attachment|file_name|filename|tool_result|arguments|input_text|"
    r"output_text)$",
    re.IGNORECASE,
)
EMAIL_RE = re.compile(r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}", re.IGNORECASE)
REDACTED = "[redacted]"

OBSERVATION_ALLOWLIST = {
    "id",
    "conversation_id",
    "message_id",
    "node_id",
    "parent",
    "parent_id",
    "children",
    "author",
    "role",
    "channel",
    "create_time",
    "update_time",
    "created_at",
    "updated_at",
    "status",
    "end_turn",
    "weight",
    "recipient",
    "metadata",
    "model_slug",
    "requested_model",
    "requested_mode",
    "reasoning_effort",
    "default_model_slug",
    "generation_id",
    "request_id",
    "message_request_id",
    "parent_id_of_prompt",
    "is_archived",
    "is_starred",
    "gizmo_id",
    "workspace_id",
    "current_node",
    "mapping",
    "messages",
    "page_info",
    "has_previous_page",
    "start_cursor",
    "offset",
    "limit",
    "total",
    "items",
    "has_versions",
    "conversation_template_id",
    "surface",
    "origin",
    "shared",
    "imported",
    "copied",
    "error_type",
    "error_code",
}

METADATA_ALLOWLIST = {
    "model_slug",
    "requested_model",
    "requested_model_slug",
    "resolved_model",
    "resolved_model_slug",
    "requested_mode",
    "reasoning_effort",
    "default_model_slug",
    "generation_id",
    "request_id",
    "message_request_id",
    "parent_id",
    "is_complete",
    "is_visually_hidden_from_conversation",
    "timestamp_",
    "status",
    "gizmo_id",
    "conversation_id",
    "surface",
    "origin",
    "from_shared",
    "from_copy",
    "imported",
    "workspace_id",
}

IDENTITY_ALLOWLIST = {
    "provider_user_id",
    "workspace_id",
    "quota_owner_id",
    "collector_account_id",
    "surface",
    "auth_state",
    "plan_label",
    "identity_errors",
}

UNKNOWN_TYPE_NAMES = {
    dict: "object",
    list: "array",
    tuple: "array",
    str: "string",
    int: "number",
    float: "number",
    bool: "boolean",
    type(None): "null",
}

_METADATA_IDENTIFIER_KEYS = {
    "model_slug",
    "requested_model",
    "requested_model_slug",
    "resolved_model",
    "resolved_model_slug",
    "default_model_slug",
    "generation_id",
    "request_id",
    "message_request_id",
    "parent_id",
    "gizmo_id",
    "conversation_id",
    "workspace_id",
}
_METADATA_TOKEN_KEYS = {
    "requested_mode",
    "reasoning_effort",
    "status",
    "surface",
    "origin",
}
_METADATA_BOOLEAN_KEYS = {
    "is_complete",
    "is_visually_hidden_from_conversation",
    "from_shared",
    "from_copy",
    "imported",
}
_METADATA_NUMBER_KEYS = {"timestamp_"}
_SAFE_METADATA_TOKEN_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,127}$")
_DROP = object()


class PrivacyError(ValueError):
    """Raised when a payload cannot be sanitized into a safe projection."""


def classify_surface(raw: Any, *, default: Optional[str] = None) -> str:
    """Persist an explicit surface. Unknown is never automatically Chat."""
    if isinstance(raw, Mapping):
        candidates = [
            raw.get("surface"),
            raw.get("product"),
            (raw.get("metadata") or {}).get("surface") if isinstance(raw.get("metadata"), Mapping) else None,
            raw.get("conversation_template_id"),
        ]
    else:
        candidates = [raw]
    for candidate in candidates:
        if candidate is None:
            continue
        normalized = str(candidate).strip().lower()
        if not normalized:
            continue
        if normalized in {"chat", "chatgpt", "chatgpt-chat", "ordinary_chat"}:
            return SURFACE_CHAT
        if "codex" in normalized:
            return "codex"
        if normalized in {"work", "chatgpt-work"}:
            return "work"
        if "deep_research" in normalized or normalized == "deep-research":
            return "deep_research"
        if "agent" in normalized:
            return "agent_mode"
        if "voice" in normalized:
            return "voice"
        if "image" in normalized:
            return "image_generation"
        if normalized not in {"unknown", "none"}:
            return "unknown"
    if default == SURFACE_CHAT:
        return SURFACE_CHAT
    return "unknown"


def schema_fingerprint(value: Any) -> str:
    names = sorted(iter_unknown_fields(value))
    digest = hashlib.sha256(json.dumps(names, separators=(",", ":")).encode("utf-8"))
    return digest.hexdigest()[:32]


def iter_unknown_fields(value: Any, prefix: str = "") -> Iterable[str]:
    if isinstance(value, Mapping):
        for key, child in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            if str(key) not in OBSERVATION_ALLOWLIST and str(key) not in METADATA_ALLOWLIST:
                yield f"{path}:{_type_name(child)}"
            yield from iter_unknown_fields(child, path)
    elif isinstance(value, list):
        for index, child in enumerate(value[:8]):
            yield from iter_unknown_fields(child, f"{prefix}[]" if prefix else "[]")
            if index >= 7:
                break


def _type_name(value: Any) -> str:
    return UNKNOWN_TYPE_NAMES.get(type(value), type(value).__name__)


def redact_text(value: str) -> str:
    return EMAIL_RE.sub(REDACTED, value)


def sanitize_value(value: Any, *, key: str = "", allow_content: bool = False) -> Any:
    if isinstance(value, Mapping):
        return sanitize_mapping(value, allow_content=allow_content)
    if isinstance(value, list):
        return [sanitize_value(item, key=key, allow_content=allow_content) for item in value]
    if isinstance(value, str):
        if not allow_content and CONTENT_KEY_RE.search(key):
            return None
        if SENSITIVE_KEY_RE.search(key):
            return REDACTED
        return redact_text(value)
    return value


def sanitize_mapping(payload: Mapping[str, Any], *, allow_content: bool = False) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for raw_key, raw_value in payload.items():
        key = str(raw_key)
        if SENSITIVE_KEY_RE.search(key):
            continue
        if not allow_content and CONTENT_KEY_RE.fullmatch(key):
            continue
        if key == "metadata" and isinstance(raw_value, Mapping):
            out[key] = sanitize_metadata(raw_value)
            continue
        if key == "author" and isinstance(raw_value, Mapping):
            author_role = raw_value.get("role")
            out[key] = {
                "role": author_role,
                "name": "[redacted]",
            }
            continue
        out[key] = sanitize_value(raw_value, key=key, allow_content=allow_content)
    return out


def sanitize_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Project message metadata to typed, non-content provenance fields."""
    out: dict[str, Any] = {}
    for raw_key, raw_value in metadata.items():
        key = str(raw_key)
        if key not in METADATA_ALLOWLIST:
            continue
        projected = _project_metadata_value(key, raw_value)
        if projected is _DROP:
            continue
        out[key] = projected
    return out


def _project_metadata_value(key: str, value: Any) -> Any:
    if key in _METADATA_BOOLEAN_KEYS:
        return value if isinstance(value, bool) else _DROP
    if key in _METADATA_NUMBER_KEYS:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return _DROP
        return value if math.isfinite(float(value)) else _DROP
    if key not in _METADATA_IDENTIFIER_KEYS and key not in _METADATA_TOKEN_KEYS:
        return _DROP
    if not isinstance(value, str):
        return _DROP
    normalized = sanitize_token(value)
    if normalized is None:
        return _DROP
    return normalized


def sanitize_token(value: Any) -> Optional[str]:
    """Return a compact identifier/token or None for unsupported input."""
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    if not normalized or not _SAFE_METADATA_TOKEN_RE.fullmatch(normalized):
        return None
    return normalized


def sanitize_identity(payload: Mapping[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in IDENTITY_ALLOWLIST:
        if key in payload:
            value = payload[key]
            if SENSITIVE_KEY_RE.search(key):
                continue
            elif key == "identity_errors":
                if isinstance(value, (list, tuple)):
                    out[key] = [
                        str(item)
                        for item in value
                        if sanitize_token(item) is not None
                    ][:16]
            elif isinstance(value, str):
                normalized = sanitize_token(value)
                if normalized is not None:
                    out[key] = normalized
            else:
                continue
    out.setdefault("surface", classify_surface(payload, default=None))
    return out


def observation_projection(
    payload: Mapping[str, Any],
    *,
    source_kind: str,
    run_id: str,
    evidence_id: str,
) -> dict[str, Any]:
    sanitized = sanitize_mapping(payload)
    keep: dict[str, Any] = {}
    for key, value in sanitized.items():
        if key in OBSERVATION_ALLOWLIST:
            keep[key] = value
    keep["surface"] = classify_surface(payload, default=None)
    keep["provenance"] = {
        "adapter_version": ADAPTER_VERSION,
        "source_kind": source_kind,
        "run_id": run_id,
        "evidence_id": evidence_id,
        "schema_fingerprint": schema_fingerprint(payload),
        "unknown_fields": sorted(iter_unknown_fields(payload))[:32],
    }
    return keep


def assert_no_secrets(value: Any, *, path: str = "root") -> None:
    """Walk mapping keys and values looking for secret-like patterns."""
    _assert_no_secrets_walk(value, path=path)


def _assert_no_secrets_walk(value: Any, *, path: str) -> None:
    if isinstance(value, str):
        if _contains_secret_value(value):
            raise PrivacyError(f"secret-like value survived sanitization at {path}")
    elif isinstance(value, dict):
        for key, child in value.items():
            key_text = str(key)
            if SENSITIVE_KEY_RE.search(key_text) or _contains_secret_value(key_text):
                raise PrivacyError(f"secret-like mapping key survived sanitization at {path}.{key_text}")
            _assert_no_secrets_walk(child, path=f"{path}.{key_text}")
    elif isinstance(value, list):
        for idx, child in enumerate(value):
            _assert_no_secrets_walk(child, path=f"{path}[{idx}]")


def _contains_secret_value(value: str) -> bool:
    lowered = value.lower()
    return bool(
        any(needle in lowered for needle in ("bearer ", "set-cookie", "authorization", "eyj"))
        or EMAIL_RE.search(value)
    )


def evidence_identity(source_kind: str, source_id: str, revision: str) -> str:
    digest = hashlib.sha256(f"{source_kind}|{source_id}|{revision}".encode("utf-8"))
    return digest.hexdigest()
