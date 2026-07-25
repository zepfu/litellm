"""Wave 4 extraction: grok_side_channel pure-leaf functions.

Behavior-preserving extraction from llm_passthrough_endpoints.py.
Do not import llm_passthrough_endpoints at module scope.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Optional

import httpx
from fastapi import Request

from types import FunctionType


_HOST_FUNCTION_NAMES = (
    "_normalize_grok_endpoint_for_target",
    "_normalize_grok_endpoint_path",
    "_get_grok_side_channel_endpoint_type",
    "_get_grok_session_side_channel_endpoint_type",
    "_get_grok_side_channel_endpoint_path_template",
    "_get_grok_session_side_channel_endpoint_path_template",
    "_json_shape_type_name",
    "_extract_redacted_grok_json_request_shape",
    "_stable_grok_side_channel_body_digest",
    "_build_grok_side_channel_request_shape_metadata",
    "_merge_grok_side_channel_shape_into_passthrough_logging_metadata",
    "_get_grok_side_channel_retryable_status_codes",
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


# ── Restored constants ──────────────────────────────────────────────

_GROK_CLI_CHAT_PROXY_DEFAULT_BASE_URL = "https://cli-chat-proxy.grok.com"

_GROK_CLI_FORWARD_HEADER_ALLOWLIST = frozenset(
    {
        "accept",
        "accept-encoding",
        "authorization",
        "content-type",
        "grok-shell-timestamp",
        "user-agent",
        "x-email",
        "x-grok-agent-id",
        "x-grok-client-identifier",
        "x-grok-client-version",
        "x-grok-conv-id",
        "x-grok-model-override",
        "x-grok-req-id",
        "x-grok-session-id",
        "x-grok-turn-idx",
        "x-grok-user-id",
        "x-request-id",
        "x-teamid",
        "x-userid",
        "x-xai-token-auth",
    }
)

_GROK_CLI_FORWARD_HEADER_COMPARE_IGNORE = frozenset(
    {
        "content-length",
        "host",
        "traceparent",
        "tracestate",
        "x-litellm-api-key",
    }
)

_CODEX_AUTO_AGENT_GROK_ACCOUNT_QUOTA_DURABLE_COOLDOWN_SECONDS = 3 * 60 * 60.0

_CODEX_AUTO_AGENT_GROK_BUILD_USAGE_BALANCE_EXHAUSTED_TOKEN = "GROK_BUILD_USAGE_BALANCE_EXHAUSTED"

_CODEX_AUTO_AGENT_GROK_PERSONAL_TEAM_SPENDING_LIMIT_TOKEN = "GROK_PERSONAL_TEAM_SPENDING_LIMIT"

_CODEX_AUTO_AGENT_GROK_BUILD_USAGE_BALANCE_EXHAUSTED_UPSTREAM_URL = "https://cli-chat-proxy.grok.com/v1/responses"

# ── Extracted functions ─────────────────────────────────────────────

def _normalize_grok_endpoint_for_target(endpoint: str, base_target_url: str) -> str:
    normalized_endpoint = httpx.URL(endpoint).path
    if not normalized_endpoint.startswith("/"):
        normalized_endpoint = "/" + normalized_endpoint

    base_url = httpx.URL(base_target_url)
    base_path = base_url.path.rstrip("/")
    if base_path.endswith("/v1") and normalized_endpoint.startswith("/v1/"):
        normalized_endpoint = normalized_endpoint[len("/v1") :]
    return normalized_endpoint

def _normalize_grok_endpoint_path(endpoint: str) -> str:
    endpoint_path = httpx.URL(endpoint).path
    if not endpoint_path.startswith("/"):
        endpoint_path = "/" + endpoint_path
    if endpoint_path.startswith("/v1/"):
        endpoint_path = endpoint_path[len("/v1") :]
    return endpoint_path

def _get_grok_side_channel_endpoint_type(endpoint: str) -> Optional[str]:
    endpoint_path = _normalize_grok_endpoint_path(endpoint)
    if endpoint_path == "/sessions/register":
        return "sessions_register"
    if endpoint_path.startswith("/sessions/") and endpoint_path.endswith("/replicas/update"):
        return "sessions_replicas_update"
    if endpoint_path.startswith("/sessions/") and endpoint_path.endswith("/signals"):
        return "sessions_signals"
    if endpoint_path.startswith("/sessions/") and endpoint_path.endswith("/turn-deltas"):
        return "sessions_turn_deltas"
    if endpoint_path == "/traces":
        return "traces"
    return None

def _get_grok_session_side_channel_endpoint_type(endpoint: str) -> Optional[str]:
    return _get_grok_side_channel_endpoint_type(endpoint)

def _get_grok_side_channel_endpoint_path_template(
    endpoint_type: str,
) -> Optional[str]:
    if endpoint_type == "sessions_register":
        return "/sessions/register"
    if endpoint_type == "sessions_replicas_update":
        return "/sessions/{session_id}/replicas/update"
    if endpoint_type == "sessions_signals":
        return "/sessions/{session_id}/signals"
    if endpoint_type == "sessions_turn_deltas":
        return "/sessions/{session_id}/turn-deltas"
    if endpoint_type == "traces":
        return "/traces"
    return None

def _get_grok_session_side_channel_endpoint_path_template(
    endpoint_type: str,
) -> Optional[str]:
    return _get_grok_side_channel_endpoint_path_template(endpoint_type)

def _json_shape_type_name(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int) and not isinstance(value, bool):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, str):
        return "str"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return type(value).__name__

def _extract_redacted_grok_json_request_shape(parsed_body: Any) -> dict[str, Any]:
    if isinstance(parsed_body, dict):
        top_level_key_types = {
            str(key): _json_shape_type_name(parsed_body.get(key))
            for key in sorted(parsed_body.keys(), key=str)
            if str(key) != "litellm_metadata"
        }
        return {
            "json_container_type": "object",
            "top_level_key_types": top_level_key_types,
        }
    if isinstance(parsed_body, list):
        return {
            "json_container_type": "array",
            "array_length": len(parsed_body),
        }
    if parsed_body is None:
        return {"json_container_type": "null"}
    return {"json_container_type": _json_shape_type_name(parsed_body)}

def _stable_grok_side_channel_body_digest(
    *,
    parsed_body: Any = None,
    raw_body: Optional[bytes] = None,
) -> tuple[int, str, str]:
    if raw_body is not None:
        body_bytes = raw_body
        digest_source = "raw_body"
    elif isinstance(parsed_body, dict):
        upstream_body = {key: value for key, value in parsed_body.items() if str(key) != "litellm_metadata"}
        body_bytes = json.dumps(
            upstream_body,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
        digest_source = "canonical_json_without_litellm_metadata"
    elif isinstance(parsed_body, list):
        body_bytes = json.dumps(
            parsed_body,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
        digest_source = "canonical_json"
    else:
        body_bytes = b""
        digest_source = "empty_body"

    return len(body_bytes), hashlib.sha256(body_bytes).hexdigest(), digest_source

def _build_grok_side_channel_request_shape_metadata(
    *,
    endpoint: str,
    request: Request,
    parsed_body: Any = None,
    raw_body: Optional[bytes] = None,
) -> Optional[dict[str, Any]]:
    endpoint_type = _get_grok_side_channel_endpoint_type(endpoint)
    if endpoint_type is None:
        return None

    content_type = request.headers.get("content-type")
    (
        body_byte_length,
        body_sha256,
        digest_source,
    ) = _stable_grok_side_channel_body_digest(
        parsed_body=parsed_body,
        raw_body=raw_body,
    )
    json_shape = _extract_redacted_grok_json_request_shape(parsed_body)

    metadata: dict[str, Any] = {
        "grok_side_channel": True,
        "grok_side_channel_endpoint_type": endpoint_type,
        "grok_side_channel_endpoint_path_template": (_get_grok_side_channel_endpoint_path_template(endpoint_type)),
        "grok_side_channel_request_content_type": content_type,
        "grok_side_channel_request_body_byte_length": body_byte_length,
        "grok_side_channel_request_body_sha256": body_sha256,
        "grok_side_channel_request_body_digest_source": digest_source,
        "grok_side_channel_request_json_container_type": json_shape.get("json_container_type"),
    }
    if "top_level_key_types" in json_shape:
        metadata["grok_side_channel_request_top_level_key_types"] = json_shape["top_level_key_types"]
    if "array_length" in json_shape:
        metadata["grok_side_channel_request_array_length"] = json_shape["array_length"]

    return metadata

def _merge_grok_side_channel_shape_into_passthrough_logging_metadata(
    passthrough_logging_metadata: dict[str, Any],
    *,
    shape_metadata: Optional[dict[str, Any]],
) -> dict[str, Any]:
    if not shape_metadata:
        return passthrough_logging_metadata
    merged = dict(passthrough_logging_metadata)
    merged.update(shape_metadata)
    tags = list(merged.get("tags") or [])
    if "grok-side-channel" not in tags:
        tags.append("grok-side-channel")
    merged["tags"] = tags
    return merged

def _get_grok_side_channel_retryable_status_codes(endpoint: str) -> list[int]:
    is_session_side_channel = _get_grok_side_channel_endpoint_type(endpoint) is not None
    if not is_session_side_channel:
        return []

    return [500, 502, 503, 504]
