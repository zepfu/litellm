"""Sanitized pass-through response shape capture for AAWM investigations.

This is intentionally separate from ``aawm_payload_capture``. By default it
captures the shape of upstream provider responses as LiteLLM receives them,
while avoiding full prompt/body/content persistence.

Enable with ``AAWM_CAPTURE_PASSTHROUGH_SHAPES=1``. Artifacts are written under
``/tmp/captures/pass_through_shapes`` by default, which maps to ``./captures``
in the local dev compose stack.

For targeted investigations that need the complete provider payload, enable
``AAWM_CAPTURE_PASSTHROUGH_FULL_PAYLOADS=1`` or write a truthy value to
``/tmp/captures/pass_through_full_payloads.enabled``. Full payload artifacts are
written under ``/tmp/captures/pass_through_full_payloads`` by default and
intentionally persist request/response bodies without content redaction.
Request and response headers are persisted without redaction. The control file
is checked on each capture attempt, so it can be flipped without restarting the
proxy process.
"""

import base64
import hashlib
import json
import os
import re
import threading
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence
from urllib.parse import parse_qsl, urlparse

import httpx

from litellm._logging import verbose_proxy_logger


_ENV_FLAG = "AAWM_CAPTURE_PASSTHROUGH_SHAPES"
_DIR_ENV = "AAWM_CAPTURE_PASSTHROUGH_SHAPES_DIR"
_FULL_PAYLOAD_ENV_FLAG = "AAWM_CAPTURE_PASSTHROUGH_FULL_PAYLOADS"
_FULL_PAYLOAD_DIR_ENV = "AAWM_CAPTURE_PASSTHROUGH_FULL_PAYLOADS_DIR"
_FULL_PAYLOAD_CONTROL_FILE_ENV = "AAWM_CAPTURE_PASSTHROUGH_FULL_PAYLOADS_CONTROL_FILE"
_FULL_PAYLOAD_MAX_BYTES_ENV = "AAWM_CAPTURE_PASSTHROUGH_FULL_PAYLOADS_MAX_BYTES"
_DIAGNOSTIC_ENV_FLAG = "AAWM_DIAGNOSTIC_PAYLOAD_CAPTURE"
_DIAGNOSTIC_DIR_ENV = "AAWM_DIAGNOSTIC_PAYLOAD_CAPTURE_DIR"
_DIAGNOSTIC_ROUTE_FAMILIES_ENV = "AAWM_DIAGNOSTIC_PAYLOAD_CAPTURE_ROUTE_FAMILIES"
_DIAGNOSTIC_ENDPOINT_TEMPLATES_ENV = (
    "AAWM_DIAGNOSTIC_PAYLOAD_CAPTURE_ENDPOINT_TEMPLATES"
)
_DIAGNOSTIC_TRACE_IDS_ENV = "AAWM_DIAGNOSTIC_PAYLOAD_CAPTURE_TRACE_IDS"
_DIAGNOSTIC_LITELLM_CALL_IDS_ENV = "AAWM_DIAGNOSTIC_PAYLOAD_CAPTURE_LITELLM_CALL_IDS"
_DIAGNOSTIC_REDACTION_MODE_ENV = "AAWM_DIAGNOSTIC_PAYLOAD_CAPTURE_REDACTION_MODE"
_DIAGNOSTIC_ENV_NAME_ENV = "AAWM_DIAGNOSTIC_PAYLOAD_CAPTURE_ENVIRONMENT"
_DEFAULT_CAPTURE_DIR = Path("/tmp/captures/pass_through_shapes")
_DEFAULT_FULL_PAYLOAD_CAPTURE_DIR = Path("/tmp/captures/pass_through_full_payloads")
_DEFAULT_DIAGNOSTIC_CAPTURE_DIR = Path("/tmp/captures/diagnostic_payloads")
_DEFAULT_FULL_PAYLOAD_CONTROL_FILE = Path(
    "/tmp/captures/pass_through_full_payloads.enabled"
)
_MAX_KEY_PATHS = 240
_MAX_QUOTA_HITS = 80
_MAX_EVENT_SAMPLES = 40
_MAX_SHAPE_DEPTH = 5
_MAX_DICT_KEYS = 60
_MAX_LIST_ITEMS = 3

_counter_lock = threading.Lock()
_counter = 0

_QUOTA_TERMS = (
    "rate",
    "limit",
    "quota",
    "reset",
    "remaining",
    "used",
    "percent",
    "retry",
    "capacity",
    "exhaust",
    "window",
)
_LOW_CARDINALITY_VALUE_PATH_TERMS = (
    "modelid",
    "model_id",
    "modeltype",
    "model_type",
    "tokentype",
    "token_type",
    "tier",
    "claim",
)
_SENSITIVE_PATH_TERMS = (
    "authorization",
    "api_key",
    "apikey",
    "access_token",
    "refresh_token",
    "cookie",
    "secret",
    "password",
    "credential",
)
_CONTENT_PATH_TERMS = (
    "content",
    "text",
    "prompt",
    "delta",
    "output",
    "input",
    "instructions",
    "system",
    "messages",
)
_HEADER_VALUE_TERMS = (
    "rate",
    "limit",
    "quota",
    "retry",
    "remaining",
    "reset",
    "used",
    "percent",
    "window",
    "request-id",
    "trace",
)
_HEADER_DROP_TERMS = (
    "authorization",
    "api-key",
    "apikey",
    "cookie",
    "token",
    "secret",
)

_SECRET_PATTERNS = (
    re.compile(r"sk-[A-Za-z0-9_\-]{12,}"),
    re.compile(r"AIza[0-9A-Za-z_\-]{20,}"),
    re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._\-]+"),
    re.compile(r"(?i)(api[_-]?key|token|authorization|cookie)=([^&\s]+)"),
    re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}"),
)


def _is_truthy(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


def passthrough_shape_capture_enabled() -> bool:
    return _is_truthy(os.environ.get(_ENV_FLAG, ""))


def passthrough_full_payload_capture_enabled() -> bool:
    control_file = Path(
        os.environ.get(
            _FULL_PAYLOAD_CONTROL_FILE_ENV,
            str(_DEFAULT_FULL_PAYLOAD_CONTROL_FILE),
        )
    )
    try:
        if control_file.exists():
            return _is_truthy(control_file.read_text(encoding="utf-8"))
    except Exception:
        pass
    return _is_truthy(os.environ.get(_FULL_PAYLOAD_ENV_FLAG, ""))


def _split_scope_values(value: str) -> List[str]:
    values: List[str] = []
    for part in value.split(","):
        stripped = part.strip()
        if stripped:
            values.append(stripped)
    return values


def _diagnostic_capture_scope_values(env_name: str) -> List[str]:
    return _split_scope_values(os.environ.get(env_name, ""))


def diagnostic_payload_capture_enabled() -> bool:
    if not _is_truthy(os.environ.get(_DIAGNOSTIC_ENV_FLAG, "")):
        return False
    return any(
        _diagnostic_capture_scope_values(env_name)
        for env_name in (
            _DIAGNOSTIC_ROUTE_FAMILIES_ENV,
            _DIAGNOSTIC_ENDPOINT_TEMPLATES_ENV,
            _DIAGNOSTIC_TRACE_IDS_ENV,
            _DIAGNOSTIC_LITELLM_CALL_IDS_ENV,
        )
    )


def _capture_dir() -> Path:
    configured = os.environ.get(_DIR_ENV)
    if configured:
        return Path(configured)
    return _DEFAULT_CAPTURE_DIR


def _full_payload_capture_dir() -> Path:
    configured = os.environ.get(_FULL_PAYLOAD_DIR_ENV)
    if configured:
        return Path(configured)
    return _DEFAULT_FULL_PAYLOAD_CAPTURE_DIR


def _diagnostic_capture_dir() -> Path:
    configured = os.environ.get(_DIAGNOSTIC_DIR_ENV)
    if configured:
        return Path(configured)
    return _DEFAULT_DIAGNOSTIC_CAPTURE_DIR


def _diagnostic_environment_name() -> str:
    for env_name in (
        _DIAGNOSTIC_ENV_NAME_ENV,
        "AAWM_ENVIRONMENT",
        "LITELLM_ENVIRONMENT",
        "ENVIRONMENT",
    ):
        value = os.environ.get(env_name)
        if value and value.strip():
            return value.strip()
    return "unknown"


def _diagnostic_redaction_mode() -> str:
    configured = os.environ.get(_DIAGNOSTIC_REDACTION_MODE_ENV, "").strip()
    return configured or "shape_hash_manifest"


def _full_payload_max_bytes() -> Optional[int]:
    configured = os.environ.get(_FULL_PAYLOAD_MAX_BYTES_ENV)
    if not configured:
        return None
    try:
        value = int(configured)
    except ValueError:
        return None
    if value <= 0:
        return None
    return value


def _next_counter() -> int:
    global _counter
    with _counter_lock:
        _counter += 1
        return _counter


def _safe_enum_value(value: Any) -> Optional[str]:
    if value is None:
        return None
    enum_value = getattr(value, "value", None)
    if enum_value is not None:
        return str(enum_value)
    return str(value)


def _json_size_bytes(value: Any) -> int:
    try:
        serialized = json.dumps(
            value,
            default=str,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except Exception:
        serialized = f"<unserializable:{type(value).__name__}>"
    return len(serialized.encode("utf-8"))


def _canonical_bytes(value: Any) -> bytes:
    if value is None:
        return b""
    if isinstance(value, bytes):
        return value
    if isinstance(value, str):
        return value.encode("utf-8", errors="replace")
    try:
        return json.dumps(
            value,
            default=str,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except Exception:
        return str(value).encode("utf-8", errors="replace")


def _sha256_hexdigest(value: Any) -> Optional[str]:
    payload = _canonical_bytes(value)
    if not payload:
        return None
    return hashlib.sha256(payload).hexdigest()


def _sanitize_filename_part(value: Any) -> str:
    text = str(value or "unknown").lower()
    text = re.sub(r"[^a-z0-9_.-]+", "_", text)
    return text[:60] or "unknown"


def _metadata_mapping_from_request_body(request_body: Any) -> Mapping[str, Any]:
    if not isinstance(request_body, Mapping):
        return {}
    for key in ("litellm_metadata", "metadata"):
        value = request_body.get(key)
        if isinstance(value, Mapping):
            return value
    return {}


def _extract_context_value(
    key: str,
    *,
    request_body: Any,
    extra_metadata: Optional[Mapping[str, Any]],
) -> Optional[str]:
    if isinstance(extra_metadata, Mapping):
        value = extra_metadata.get(key)
        if value is not None:
            return str(value)
    metadata = _metadata_mapping_from_request_body(request_body)
    value = metadata.get(key)
    if value is not None:
        return str(value)
    return None


def _diagnostic_route_family(
    *,
    provider: Optional[str],
    endpoint_type: Any,
    request_body: Any,
    extra_metadata: Optional[Mapping[str, Any]],
) -> str:
    for key in ("passthrough_route_family", "route_family", "aawm_route_family"):
        value = _extract_context_value(
            key,
            request_body=request_body,
            extra_metadata=extra_metadata,
        )
        if value:
            return value
    if provider:
        return str(provider)
    endpoint_type_value = _safe_enum_value(endpoint_type)
    return endpoint_type_value or "unknown"


_UUIDISH_PATH_SEGMENT_RE = re.compile(
    r"^(?:[0-9a-f]{8,}(?:-[0-9a-f]{4,}){2,}|[0-9a-f]{16,}|[A-Za-z0-9_-]{24,})$",
    re.IGNORECASE,
)


def _endpoint_template_from_url(
    url_route: Optional[str],
    *,
    request_body: Any,
    extra_metadata: Optional[Mapping[str, Any]],
) -> str:
    for key in (
        "endpoint_template",
        "grok_side_channel_endpoint_path_template",
        "aawm_endpoint_template",
    ):
        value = _extract_context_value(
            key,
            request_body=request_body,
            extra_metadata=extra_metadata,
        )
        if value:
            return value

    parsed = urlparse(url_route or "")
    path = parsed.path or "/"
    templated_segments = []
    for segment in path.split("/"):
        if not segment:
            templated_segments.append(segment)
            continue
        if _UUIDISH_PATH_SEGMENT_RE.match(segment):
            templated_segments.append("{id}")
        else:
            templated_segments.append(segment)
    return "/".join(templated_segments) or "/"


def _matches_scope(actual: Optional[str], patterns: List[str], *, casefold: bool) -> bool:
    if not patterns:
        return True
    if actual is None:
        return False
    actual_value = actual.lower() if casefold else actual
    for pattern in patterns:
        pattern_value = pattern.lower() if casefold else pattern
        if actual_value == pattern_value:
            return True
    return False


def _diagnostic_scope_matches(
    *,
    route_family: str,
    endpoint_template: str,
    trace_id: Optional[str],
    litellm_call_id: Optional[str],
) -> bool:
    route_families = _diagnostic_capture_scope_values(_DIAGNOSTIC_ROUTE_FAMILIES_ENV)
    endpoint_templates = _diagnostic_capture_scope_values(
        _DIAGNOSTIC_ENDPOINT_TEMPLATES_ENV
    )
    trace_ids = _diagnostic_capture_scope_values(_DIAGNOSTIC_TRACE_IDS_ENV)
    litellm_call_ids = _diagnostic_capture_scope_values(_DIAGNOSTIC_LITELLM_CALL_IDS_ENV)
    return (
        _matches_scope(route_family, route_families, casefold=True)
        and _matches_scope(endpoint_template, endpoint_templates, casefold=False)
        and _matches_scope(trace_id, trace_ids, casefold=False)
        and _matches_scope(litellm_call_id, litellm_call_ids, casefold=False)
    )


def _redact_string(value: str) -> str:
    redacted = value
    for pattern in _SECRET_PATTERNS:
        redacted = pattern.sub("<redacted>", redacted)
    if len(redacted) > 360:
        redacted = f"{redacted[:360]}...<truncated>"
    return redacted


def _path_has_term(path: str, terms: Sequence[str]) -> bool:
    lower_path = path.lower()
    return any(term in lower_path for term in terms)


def _string_has_quota_terms(value: str) -> bool:
    lower_value = value.lower()
    return any(term in lower_value for term in _QUOTA_TERMS)


def _should_preserve_primitive(path: str, value: Any) -> bool:
    if _path_has_term(path, _SENSITIVE_PATH_TERMS):
        return False
    if _path_has_term(path, _CONTENT_PATH_TERMS):
        return False
    if _path_has_term(path, _LOW_CARDINALITY_VALUE_PATH_TERMS):
        return True
    if _path_has_term(path, _QUOTA_TERMS):
        return True
    if isinstance(value, str) and _string_has_quota_terms(value):
        return False
    return False


def _shape_primitive(value: Any, path: str) -> Any:
    if value is None:
        return "<null>"
    if isinstance(value, bool):
        return value if _should_preserve_primitive(path, value) else "<bool>"
    if isinstance(value, int):
        return value if _should_preserve_primitive(path, value) else "<int>"
    if isinstance(value, float):
        return value if _should_preserve_primitive(path, value) else "<float>"
    if isinstance(value, str):
        if _should_preserve_primitive(path, value):
            return _redact_string(value)
        return f"<str len={len(value)}>"
    return f"<{type(value).__name__}>"


def _shape_value(value: Any, *, path: str = "$", depth: int = 0) -> Any:
    if depth >= _MAX_SHAPE_DEPTH:
        return f"<{type(value).__name__}>"
    if isinstance(value, Mapping):
        shaped: Dict[str, Any] = {}
        for index, key in enumerate(sorted(value.keys(), key=str)):
            if index >= _MAX_DICT_KEYS:
                shaped["_truncated_keys"] = len(value) - _MAX_DICT_KEYS
                break
            key_text = str(key)
            child_path = f"{path}.{key_text}" if path else key_text
            if _path_has_term(child_path, _SENSITIVE_PATH_TERMS):
                shaped[key_text] = "<redacted>"
                continue
            shaped[key_text] = _shape_value(
                value[key], path=child_path, depth=depth + 1
            )
        return shaped
    if isinstance(value, list):
        shaped_items = [
            _shape_value(item, path=f"{path}[{idx}]", depth=depth + 1)
            for idx, item in enumerate(value[:_MAX_LIST_ITEMS])
        ]
        if len(value) > _MAX_LIST_ITEMS:
            shaped_items.append({"_truncated_items": len(value) - _MAX_LIST_ITEMS})
        return shaped_items
    return _shape_primitive(value, path)


def _collect_key_paths(
    value: Any,
    *,
    path: str = "$",
    depth: int = 0,
    paths: Optional[List[str]] = None,
) -> List[str]:
    if paths is None:
        paths = []
    if len(paths) >= _MAX_KEY_PATHS or depth >= _MAX_SHAPE_DEPTH:
        return paths
    if isinstance(value, Mapping):
        for key in sorted(value.keys(), key=str):
            if len(paths) >= _MAX_KEY_PATHS:
                break
            key_text = str(key)
            child_path = f"{path}.{key_text}"
            paths.append(child_path)
            _collect_key_paths(
                value[key], path=child_path, depth=depth + 1, paths=paths
            )
    elif isinstance(value, list):
        for index, item in enumerate(value[:_MAX_LIST_ITEMS]):
            child_path = f"{path}[{index}]"
            paths.append(child_path)
            _collect_key_paths(
                item, path=child_path, depth=depth + 1, paths=paths
            )
    return paths


def _collect_quota_hits(
    value: Any,
    *,
    path: str = "$",
    depth: int = 0,
    hits: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    if hits is None:
        hits = []
    if len(hits) >= _MAX_QUOTA_HITS or depth >= _MAX_SHAPE_DEPTH:
        return hits
    if isinstance(value, Mapping):
        for key in sorted(value.keys(), key=str):
            child_path = f"{path}.{key}"
            child_value = value[key]
            if _path_has_term(child_path, _SENSITIVE_PATH_TERMS):
                continue
            if not isinstance(child_value, (Mapping, list)) and _should_preserve_primitive(
                child_path, child_value
            ):
                hits.append(
                    {
                        "path": child_path,
                        "value": _shape_primitive(child_value, child_path),
                    }
                )
            _collect_quota_hits(
                child_value, path=child_path, depth=depth + 1, hits=hits
            )
    elif isinstance(value, list):
        for index, item in enumerate(value[:_MAX_LIST_ITEMS]):
            _collect_quota_hits(
                item, path=f"{path}[{index}]", depth=depth + 1, hits=hits
            )
    return hits


def _url_shape(url_route: Optional[str]) -> Dict[str, Any]:
    parsed = urlparse(url_route or "")
    query_keys = sorted({key for key, _ in parse_qsl(parsed.query, keep_blank_values=True)})
    return {
        "raw": url_route or None,
        "scheme": parsed.scheme or None,
        "host": parsed.hostname or None,
        "path": parsed.path or None,
        "query_keys": query_keys,
    }


def _diagnostic_url_shape(
    url_route: Optional[str],
    *,
    endpoint_template: str,
) -> Dict[str, Any]:
    parsed = urlparse(url_route or "")
    query_keys = sorted({key for key, _ in parse_qsl(parsed.query, keep_blank_values=True)})
    return {
        "scheme": parsed.scheme or None,
        "host": parsed.hostname or None,
        "endpoint_template": endpoint_template,
        "query_keys": query_keys,
    }


def _request_shape(request_body: Any) -> Dict[str, Any]:
    if not isinstance(request_body, Mapping):
        return {"kind": type(request_body).__name__}
    summary: Dict[str, Any] = {
        "top_level_keys": sorted(str(key) for key in request_body.keys()),
    }
    for key in ("model", "stream", "anthropic_version"):
        value = request_body.get(key)
        if isinstance(value, (str, int, float, bool)) or value is None:
            summary[key] = value
    if "metadata" in request_body and isinstance(request_body["metadata"], Mapping):
        summary["metadata_keys"] = sorted(str(key) for key in request_body["metadata"])
    return summary


def _sanitize_headers(headers: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    if not headers:
        return {"names": [], "selected_values": {}}
    names: List[str] = []
    selected_values: Dict[str, str] = {}
    for header_name, header_value in headers.items():
        normalized_name = str(header_name).lower()
        if any(term in normalized_name for term in _HEADER_DROP_TERMS):
            continue
        names.append(normalized_name)
        if any(term in normalized_name for term in _HEADER_VALUE_TERMS):
            selected_values[normalized_name] = _redact_string(str(header_value))
    return {
        "names": sorted(set(names)),
        "selected_values": dict(sorted(selected_values.items())),
    }


def _full_payload_headers(headers: Optional[Mapping[str, Any]]) -> Dict[str, str]:
    if not headers:
        return {}
    values: Dict[str, str] = {}
    for header_name, header_value in headers.items():
        values[str(header_name)] = str(header_value)
    return dict(sorted(values.items(), key=lambda item: item[0].lower()))


def _full_payload_header_items(
    headers: Optional[Mapping[str, Any]],
) -> List[Dict[str, str]]:
    if not headers:
        return []
    try:
        raw_items = list(headers.multi_items())  # type: ignore[attr-defined]
    except Exception:
        raw_items = list(headers.items())
    return [
        {"name": str(header_name), "value": str(header_value)}
        for header_name, header_value in raw_items
    ]


def _response_headers(response: Optional[httpx.Response]) -> Optional[Mapping[str, Any]]:
    if response is None:
        return None
    return response.headers


def _maybe_truncate_text(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    max_bytes = _full_payload_max_bytes()
    if max_bytes is None:
        return value
    encoded = value.encode("utf-8", errors="replace")
    if len(encoded) <= max_bytes:
        return value
    return encoded[:max_bytes].decode("utf-8", errors="replace")


def _maybe_truncate_bytes(value: bytes) -> bytes:
    max_bytes = _full_payload_max_bytes()
    if max_bytes is None or len(value) <= max_bytes:
        return value
    return value[:max_bytes]


def _was_truncated_bytes(value: bytes) -> bool:
    max_bytes = _full_payload_max_bytes()
    return max_bytes is not None and len(value) > max_bytes


def _decode_body_bytes(content: Optional[bytes]) -> Optional[str]:
    if content is None:
        return None
    try:
        return content.decode("utf-8", errors="replace")
    except Exception:
        return None


def _jsonable_full_payload(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, bytes):
        payload = _maybe_truncate_bytes(value)
        return {
            "encoding": "base64",
            "data": base64.b64encode(payload).decode("ascii"),
            "truncated": _was_truncated_bytes(value),
            "original_bytes": len(value),
            "stored_bytes": len(payload),
        }
    if isinstance(value, Mapping):
        return {str(key): _jsonable_full_payload(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable_full_payload(item) for item in value]
    return str(value)


def _full_response_body(
    response_body: Any,
    response_content: Optional[bytes],
) -> Dict[str, Any]:
    body: Dict[str, Any] = {}
    parsed_body = response_body
    if parsed_body is None:
        parsed_body = _parse_json_text(_decode_body_bytes(response_content))
    if parsed_body is not None:
        body["json"] = _jsonable_full_payload(parsed_body)
    if response_content is not None:
        stored_content = _maybe_truncate_bytes(response_content)
        body["content_base64"] = base64.b64encode(stored_content).decode("ascii")
        body["content_text"] = _maybe_truncate_text(_decode_body_bytes(stored_content))
        body["truncated"] = _was_truncated_bytes(response_content)
        body["original_bytes"] = len(response_content)
        body["stored_bytes"] = len(stored_content)
    return body


def _parse_json_text(text: Optional[str]) -> Any:
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        return None


def _httpx_request_content(upstream_request: Optional[httpx.Request]) -> Optional[bytes]:
    if upstream_request is None:
        return None
    try:
        return upstream_request.content
    except Exception:
        return None


def _httpx_request_payload(
    upstream_request: Optional[httpx.Request],
    *,
    fallback_url_route: Optional[str],
    fallback_request_body: Any,
) -> Dict[str, Any]:
    if upstream_request is None:
        return {
            "body": _jsonable_full_payload(fallback_request_body),
        }

    request_content = _httpx_request_content(upstream_request)
    payload: Dict[str, Any] = {
        "method": upstream_request.method,
        "url": str(upstream_request.url),
        "headers": _full_payload_headers(upstream_request.headers),
        "header_items": _full_payload_header_items(upstream_request.headers),
    }
    if request_content is not None:
        payload["body"] = _full_response_body(None, request_content)
    else:
        payload["body"] = _jsonable_full_payload(fallback_request_body)
        payload["body_source"] = "fallback_request_body"
    if fallback_url_route and str(upstream_request.url) != fallback_url_route:
        payload["logging_url"] = fallback_url_route
    return payload


def _body_shape(response_body: Any, response_content: Optional[bytes]) -> Dict[str, Any]:
    parsed_body = response_body
    if parsed_body is None:
        parsed_body = _parse_json_text(_decode_body_bytes(response_content))
    if isinstance(parsed_body, (Mapping, list)):
        return {
            "kind": "json",
            "shape": _shape_value(parsed_body),
            "key_paths": _collect_key_paths(parsed_body),
            "quota_hits": _collect_quota_hits(parsed_body),
        }
    text = _decode_body_bytes(response_content)
    if text is None:
        return {"kind": "empty_or_binary"}
    quota_terms = sorted({term for term in _QUOTA_TERMS if term in text.lower()})
    body_summary: Dict[str, Any] = {
        "kind": "text",
        "length": len(text),
        "quota_keyword_hits": quota_terms,
    }
    if quota_terms:
        body_summary["quota_text_sample"] = _redact_string(text)
    return body_summary


def _parse_stream_data_line(data_text: str) -> Any:
    stripped = data_text.strip()
    if stripped == "[DONE]":
        return {"done": True}
    try:
        return json.loads(stripped)
    except Exception:
        return stripped


def _event_name_from_payload(payload: Any) -> Optional[str]:
    if not isinstance(payload, Mapping):
        return None
    for key in ("type", "event", "kind"):
        value = payload.get(key)
        if isinstance(value, str) and value:
            return value
    nested_payload = payload.get("payload")
    if isinstance(nested_payload, Mapping):
        value = nested_payload.get("type")
        if isinstance(value, str) and value:
            return value
    return None


def _stream_shape(all_chunks: Sequence[str]) -> Dict[str, Any]:
    current_event: Optional[str] = None
    event_names: List[str] = []
    samples: List[Dict[str, Any]] = []
    line_prefix_counts: Counter[str] = Counter()
    non_json_text_hits: List[Dict[str, Any]] = []

    for line in all_chunks:
        stripped = str(line).strip()
        if not stripped:
            continue
        prefix = stripped.split(":", 1)[0] if ":" in stripped else "<raw>"
        line_prefix_counts[prefix] += 1
        if stripped.startswith("event:"):
            current_event = stripped.split(":", 1)[1].strip() or None
            continue
        if stripped.startswith("data:"):
            payload = _parse_stream_data_line(stripped.split(":", 1)[1])
            event_name = current_event or _event_name_from_payload(payload) or "data"
            current_event = None
        else:
            payload = _parse_stream_data_line(stripped)
            event_name = _event_name_from_payload(payload) or "raw"

        event_names.append(event_name)
        if len(samples) >= _MAX_EVENT_SAMPLES:
            continue
        sample: Dict[str, Any] = {"event": event_name}
        if isinstance(payload, (Mapping, list)):
            sample["data_shape"] = _shape_value(payload)
            sample["data_key_paths"] = _collect_key_paths(payload)
            sample["quota_hits"] = _collect_quota_hits(payload)
        elif isinstance(payload, str):
            sample["data_shape"] = f"<str len={len(payload)}>"
            quota_terms = sorted(
                {term for term in _QUOTA_TERMS if term in payload.lower()}
            )
            if quota_terms:
                text_hit = {
                    "event": event_name,
                    "quota_keyword_hits": quota_terms,
                    "quota_text_sample": _redact_string(payload),
                }
                non_json_text_hits.append(text_hit)
                sample.update(text_hit)
        else:
            sample["data_shape"] = _shape_primitive(payload, "$")
        samples.append(sample)

    event_counts = Counter(event_names)
    return {
        "line_count": len([line for line in all_chunks if str(line).strip()]),
        "line_prefix_counts": dict(sorted(line_prefix_counts.items())),
        "event_sequence_first": event_names[:100],
        "event_counts": dict(sorted(event_counts.items())),
        "sample_events": samples,
        "non_json_text_quota_hits": non_json_text_hits[:20],
    }


def _base_artifact(
    *,
    mode: str,
    provider: Optional[str],
    endpoint_type: Any,
    url_route: Optional[str],
    request_body: Any,
    response: Optional[httpx.Response],
    litellm_call_id: Optional[str],
    extra_metadata: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    artifact: Dict[str, Any] = {
        "captured_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "capture_kind": "aawm_passthrough_response_shape",
        "mode": mode,
        "provider": provider,
        "endpoint_type": _safe_enum_value(endpoint_type),
        "litellm_call_id": litellm_call_id,
        "url": _url_shape(url_route),
        "request": _request_shape(request_body),
        "response": {
            "status_code": response.status_code if response is not None else None,
            "headers": _sanitize_headers(_response_headers(response)),
        },
    }
    if extra_metadata:
        artifact["metadata"] = {
            str(key): value
            for key, value in extra_metadata.items()
            if isinstance(value, (str, int, float, bool)) or value is None
        }
    return artifact


def _base_full_payload_artifact(
    *,
    mode: str,
    provider: Optional[str],
    endpoint_type: Any,
    url_route: Optional[str],
    request_body: Any,
    response: Optional[httpx.Response],
    upstream_request: Optional[httpx.Request],
    litellm_call_id: Optional[str],
    extra_metadata: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    artifact: Dict[str, Any] = {
        "captured_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "capture_kind": "aawm_passthrough_full_payload",
        "capture_scope": (
            "upstream_http_transaction"
            if upstream_request is not None
            else "passthrough_logging_capture"
        ),
        "mode": mode,
        "provider": provider,
        "endpoint_type": _safe_enum_value(endpoint_type),
        "litellm_call_id": litellm_call_id,
        "url": _url_shape(url_route),
        "request": _httpx_request_payload(
            upstream_request,
            fallback_url_route=url_route,
            fallback_request_body=request_body,
        ),
        "response": {
            "status_code": response.status_code if response is not None else None,
            "headers": _full_payload_headers(_response_headers(response)),
            "header_items": _full_payload_header_items(_response_headers(response)),
        },
    }
    if extra_metadata:
        artifact["metadata"] = {
            str(key): value
            for key, value in extra_metadata.items()
            if isinstance(value, (str, int, float, bool)) or value is None
        }
    return artifact


def _diagnostic_payload_artifact(
    *,
    mode: str,
    provider: Optional[str],
    endpoint_type: Any,
    url_route: Optional[str],
    request_body: Any,
    response: Optional[httpx.Response],
    upstream_request: Optional[httpx.Request],
    response_body: Any,
    response_content: Optional[bytes],
    all_chunks: Optional[Sequence[str]],
    raw_bytes: Optional[Sequence[bytes]],
    litellm_call_id: Optional[str],
    extra_metadata: Optional[Mapping[str, Any]],
) -> Optional[Dict[str, Any]]:
    if not diagnostic_payload_capture_enabled():
        return None

    route_family = _diagnostic_route_family(
        provider=provider,
        endpoint_type=endpoint_type,
        request_body=request_body,
        extra_metadata=extra_metadata,
    )
    endpoint_template = _endpoint_template_from_url(
        url_route,
        request_body=request_body,
        extra_metadata=extra_metadata,
    )
    trace_id = _extract_context_value(
        "trace_id",
        request_body=request_body,
        extra_metadata=extra_metadata,
    )
    if trace_id is None:
        trace_id = _extract_context_value(
            "aawm_trace_id",
            request_body=request_body,
            extra_metadata=extra_metadata,
        )

    if not _diagnostic_scope_matches(
        route_family=route_family,
        endpoint_template=endpoint_template,
        trace_id=trace_id,
        litellm_call_id=litellm_call_id,
    ):
        return None

    response_content_bytes = response_content or b""
    stream_lines = list(all_chunks or [])
    raw_stream_chunks = list(raw_bytes or [])
    request_body_hash = _sha256_hexdigest(request_body)
    response_body_hash = _sha256_hexdigest(response_body)
    if response_body_hash is None and response_content_bytes:
        response_body_hash = _sha256_hexdigest(response_content_bytes)
    stream_lines_hash = _sha256_hexdigest(stream_lines)
    raw_stream_hash = _sha256_hexdigest(b"".join(raw_stream_chunks))
    upstream_request_content = _httpx_request_content(upstream_request)

    manifest: Dict[str, Any] = {
        "environment": _diagnostic_environment_name(),
        "route_family": route_family,
        "endpoint_template": endpoint_template,
        "trace_id": trace_id,
        "litellm_call_id": litellm_call_id,
        "redaction_mode": _diagnostic_redaction_mode(),
        "provider": provider,
        "endpoint_type": _safe_enum_value(endpoint_type),
        "mode": mode,
        "byte_counts": {
            "request_body_bytes": _json_size_bytes(request_body),
            "upstream_request_body_bytes": (
                len(upstream_request_content)
                if upstream_request_content is not None
                else None
            ),
            "response_body_bytes": _json_size_bytes(response_body),
            "response_content_bytes": len(response_content_bytes),
            "stream_line_count": len(stream_lines),
            "stream_text_bytes": sum(len(str(line).encode("utf-8")) for line in stream_lines),
            "raw_stream_chunk_count": len(raw_stream_chunks),
            "raw_stream_bytes": sum(len(chunk) for chunk in raw_stream_chunks),
        },
        "hashes": {
            "request_body_sha256": request_body_hash,
            "upstream_request_body_sha256": (
                _sha256_hexdigest(upstream_request_content)
                if upstream_request_content is not None
                else None
            ),
            "response_body_sha256": response_body_hash,
            "stream_lines_sha256": stream_lines_hash,
            "raw_stream_sha256": raw_stream_hash,
        },
        "omitted_fields": [
            "request.headers.values",
            "request.body.raw",
            "upstream_request.headers.values",
            "upstream_request.body.raw",
            "response.headers.values",
            "response.body.raw",
            "response.stream.raw_lines",
            "response.stream.raw_bytes",
        ],
    }

    artifact: Dict[str, Any] = {
        "captured_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "capture_kind": "aawm_diagnostic_payload_capture",
        "manifest": manifest,
        "url": _diagnostic_url_shape(
            url_route,
            endpoint_template=endpoint_template,
        ),
        "request": {
            "shape": _request_shape(request_body),
            "body_shape": _shape_value(request_body),
            "headers": _sanitize_headers(
                upstream_request.headers if upstream_request is not None else None
            ),
        },
        "response": {
            "status_code": response.status_code if response is not None else None,
            "headers": _sanitize_headers(_response_headers(response)),
        },
    }
    if response_body is not None or response_content is not None:
        artifact["response"]["body"] = _body_shape(response_body, response_content)
    if all_chunks is not None:
        artifact["response"]["stream"] = _stream_shape(stream_lines)
    if extra_metadata:
        artifact["metadata"] = {
            str(key): value
            for key, value in extra_metadata.items()
            if isinstance(value, (str, int, float, bool)) or value is None
        }
    return artifact


def _write_diagnostic_payload_artifact(artifact: Optional[Dict[str, Any]]) -> Optional[str]:
    if artifact is None:
        return None
    try:
        capture_dir = _diagnostic_capture_dir()
        capture_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        counter = _next_counter()
        manifest = artifact.get("manifest") or {}
        route_family = _sanitize_filename_part(manifest.get("route_family"))
        mode = _sanitize_filename_part(manifest.get("mode"))
        call_id = _sanitize_filename_part(manifest.get("litellm_call_id"))[:18]
        path = capture_dir / f"{ts}_{counter:04d}_{route_family}_{mode}_{call_id}.json"
        path.write_text(
            json.dumps(artifact, indent=2, sort_keys=True, default=str),
            encoding="utf-8",
        )
        try:
            path.chmod(0o600)
        except Exception:
            pass
        return str(path)
    except Exception as exc:
        verbose_proxy_logger.warning(
            "AawmDiagnosticPayloadCapture: capture failed: %s", exc
        )
        return None


def _write_artifact(artifact: Dict[str, Any]) -> Optional[str]:
    if not passthrough_shape_capture_enabled():
        return None
    try:
        capture_dir = _capture_dir()
        capture_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        counter = _next_counter()
        provider = _sanitize_filename_part(artifact.get("provider"))
        mode = _sanitize_filename_part(artifact.get("mode"))
        call_id = _sanitize_filename_part(artifact.get("litellm_call_id"))[:18]
        path = capture_dir / f"{ts}_{counter:04d}_{provider}_{mode}_{call_id}.json"
        path.write_text(
            json.dumps(artifact, indent=2, sort_keys=True, default=str),
            encoding="utf-8",
        )
        try:
            path.chmod(0o600)
        except Exception:
            pass
        return str(path)
    except Exception as exc:
        verbose_proxy_logger.warning(
            "AawmPassthroughShapeCapture: capture failed: %s", exc
        )
        return None


def _write_full_payload_artifact(artifact: Dict[str, Any]) -> Optional[str]:
    if not passthrough_full_payload_capture_enabled():
        return None
    try:
        capture_dir = _full_payload_capture_dir()
        capture_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        counter = _next_counter()
        provider = _sanitize_filename_part(artifact.get("provider"))
        mode = _sanitize_filename_part(artifact.get("mode"))
        call_id = _sanitize_filename_part(artifact.get("litellm_call_id"))[:18]
        path = capture_dir / f"{ts}_{counter:04d}_{provider}_{mode}_{call_id}.json"
        path.write_text(
            json.dumps(artifact, indent=2, sort_keys=True, default=str),
            encoding="utf-8",
        )
        try:
            path.chmod(0o600)
        except Exception:
            pass
        return str(path)
    except Exception as exc:
        verbose_proxy_logger.warning(
            "AawmPassthroughFullPayloadCapture: capture failed: %s", exc
        )
        return None


def capture_passthrough_shape(
    *,
    mode: str,
    provider: Optional[str],
    endpoint_type: Any = None,
    url_route: Optional[str] = None,
    request_body: Any = None,
    response: Optional[httpx.Response] = None,
    upstream_request: Optional[httpx.Request] = None,
    response_body: Any = None,
    response_content: Optional[bytes] = None,
    litellm_call_id: Optional[str] = None,
    extra_metadata: Optional[Mapping[str, Any]] = None,
) -> Optional[str]:
    diagnostic_path = _write_diagnostic_payload_artifact(
        _diagnostic_payload_artifact(
            mode=mode,
            provider=provider,
            endpoint_type=endpoint_type,
            url_route=url_route,
            request_body=request_body,
            response=response,
            upstream_request=upstream_request,
            response_body=response_body,
            response_content=response_content,
            all_chunks=None,
            raw_bytes=None,
            litellm_call_id=litellm_call_id,
            extra_metadata=extra_metadata,
        )
    )
    shape_enabled = passthrough_shape_capture_enabled()
    full_payload_enabled = passthrough_full_payload_capture_enabled()
    if not shape_enabled and not full_payload_enabled:
        return diagnostic_path

    full_payload_path: Optional[str] = None
    if full_payload_enabled:
        full_payload_artifact = _base_full_payload_artifact(
            mode=mode,
            provider=provider,
            endpoint_type=endpoint_type,
            url_route=url_route,
            request_body=request_body,
            response=response,
            upstream_request=upstream_request,
            litellm_call_id=litellm_call_id,
            extra_metadata=extra_metadata,
        )
        full_payload_artifact["response"]["body"] = _full_response_body(
            response_body,
            response_content,
        )
        full_payload_path = _write_full_payload_artifact(full_payload_artifact)

    if not shape_enabled:
        return full_payload_path or diagnostic_path

    artifact = _base_artifact(
        mode=mode,
        provider=provider,
        endpoint_type=endpoint_type,
        url_route=url_route,
        request_body=request_body,
        response=response,
        litellm_call_id=litellm_call_id,
        extra_metadata=extra_metadata,
    )
    artifact["response"]["body"] = _body_shape(response_body, response_content)
    return _write_artifact(artifact) or diagnostic_path


def capture_passthrough_stream_shape(
    *,
    provider: Optional[str],
    endpoint_type: Any = None,
    url_route: Optional[str] = None,
    request_body: Any = None,
    response: Optional[httpx.Response] = None,
    upstream_request: Optional[httpx.Request] = None,
    all_chunks: Sequence[str],
    raw_bytes: Optional[Sequence[bytes]] = None,
    litellm_call_id: Optional[str] = None,
    extra_metadata: Optional[Mapping[str, Any]] = None,
) -> Optional[str]:
    diagnostic_path = _write_diagnostic_payload_artifact(
        _diagnostic_payload_artifact(
            mode="stream",
            provider=provider,
            endpoint_type=endpoint_type,
            url_route=url_route,
            request_body=request_body,
            response=response,
            upstream_request=upstream_request,
            response_body=None,
            response_content=None,
            all_chunks=all_chunks,
            raw_bytes=raw_bytes,
            litellm_call_id=litellm_call_id,
            extra_metadata=extra_metadata,
        )
    )
    shape_enabled = passthrough_shape_capture_enabled()
    full_payload_enabled = passthrough_full_payload_capture_enabled()
    if not shape_enabled and not full_payload_enabled:
        return diagnostic_path

    full_payload_path: Optional[str] = None
    if full_payload_enabled:
        full_payload_artifact = _base_full_payload_artifact(
            mode="stream",
            provider=provider,
            endpoint_type=endpoint_type,
            url_route=url_route,
            request_body=request_body,
            response=response,
            upstream_request=upstream_request,
            litellm_call_id=litellm_call_id,
            extra_metadata=extra_metadata,
        )
        stream_payload: Dict[str, Any] = {
            "line_count": len(all_chunks),
            "lines": [_maybe_truncate_text(str(line)) for line in all_chunks],
        }
        if raw_bytes is not None:
            stored_raw_chunks = [_maybe_truncate_bytes(chunk) for chunk in raw_bytes]
            stream_payload["raw_chunk_count"] = len(raw_bytes)
            stream_payload["raw_total_bytes"] = sum(len(chunk) for chunk in raw_bytes)
            stream_payload["raw_chunks_base64"] = [
                base64.b64encode(chunk).decode("ascii") for chunk in stored_raw_chunks
            ]
            stream_payload["raw_chunks_truncated"] = [
                _was_truncated_bytes(chunk) for chunk in raw_bytes
            ]
        full_payload_artifact["response"]["stream"] = stream_payload
        full_payload_path = _write_full_payload_artifact(full_payload_artifact)

    if not shape_enabled:
        return full_payload_path or diagnostic_path

    artifact = _base_artifact(
        mode="stream",
        provider=provider,
        endpoint_type=endpoint_type,
        url_route=url_route,
        request_body=request_body,
        response=response,
        litellm_call_id=litellm_call_id,
        extra_metadata=extra_metadata,
    )
    artifact["response"]["stream"] = _stream_shape(all_chunks)
    return _write_artifact(artifact) or diagnostic_path
