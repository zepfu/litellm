"""Sanitized pass-through response shape capture for AAWM investigations.

This is intentionally separate from ``aawm_payload_capture``. By default it
captures the shape of upstream provider responses as LiteLLM receives them,
while avoiding full prompt/body/content persistence.

Enable with ``AAWM_CAPTURE_PASSTHROUGH_SHAPES=1``. Artifacts are written under
``/tmp/captures/pass_through_shapes`` by default, which maps to ``./captures``
in the local dev compose stack.

For targeted investigations that need the complete provider payload, enable
``AAWM_CAPTURE_PASSTHROUGH_FULL_PAYLOADS=1`` or a *trusted* process-owned control
file (default under world-writable ``/tmp`` is ignored). Full payload artifacts
persist bodies without content redaction but drop sensitive headers
(Authorization, cookies, tokens, secrets). Capture dirs use mode ``0700`` and
artifact files are written atomically mode ``0600``.
"""

import base64
import hashlib
import json
import math
import os
import re
import stat
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
_FULL_PAYLOAD_AGGREGATE_MAX_BYTES_ENV = (
    "AAWM_CAPTURE_PASSTHROUGH_FULL_PAYLOADS_AGGREGATE_MAX_BYTES"
)
_FULL_PAYLOAD_FIELD_MAX_BYTES_ENV = (
    "AAWM_CAPTURE_PASSTHROUGH_FULL_PAYLOADS_FIELD_MAX_BYTES"
)
_DEFAULT_FULL_PAYLOAD_FIELD_BYTES = 1024 * 1024
_DEFAULT_FULL_PAYLOAD_AGGREGATE_BYTES = 2 * 1024 * 1024
_MIN_FULL_PAYLOAD_AGGREGATE_BYTES = 256
_FULL_PAYLOAD_CONSTRUCTION_RESERVE_BYTES = 1536
_FULL_PAYLOAD_MAX_DEPTH = 12
_FULL_PAYLOAD_MAX_DICT_KEYS = 500
_FULL_PAYLOAD_MAX_LIST_ITEMS = 500
_FULL_PAYLOAD_MAX_HEADERS = 120
_FULL_PAYLOAD_HEADER_VALUE_MAX_BYTES = 8192
_FULL_PAYLOAD_OMITTED = object()
_FULL_PAYLOAD_CONTAINER_OVERHEAD_BYTES = 2
_FULL_PAYLOAD_KEY_OVERHEAD_BYTES = 2
_FULL_PAYLOAD_LIST_ITEM_OVERHEAD_BYTES = 1


class _FullPayloadBudget:
    def __init__(self, aggregate_limit: int):
        self.limit = aggregate_limit
        self.remaining = max(
            0,
            aggregate_limit - _FULL_PAYLOAD_CONSTRUCTION_RESERVE_BYTES,
        )
        self.truncations: List[Dict[str, Any]] = []

    def consume(self, byte_count: int) -> int:
        consumed = min(max(byte_count, 0), self.remaining)
        self.remaining -= consumed
        return consumed

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


def _control_file_is_trusted(control_file: Path) -> bool:
    """Only honor process-owned control files under non-world-writable parents.

    The default path under world-writable /tmp is intentionally untrusted so
    a co-located process cannot flip full-payload capture on.
    """
    try:
        file_stat = control_file.stat()
    except OSError:
        return False
    if not stat.S_ISREG(file_stat.st_mode):
        return False
    try:
        if file_stat.st_uid != os.getuid():
            return False
    except AttributeError:
        return False
    if file_stat.st_mode & (stat.S_IWGRP | stat.S_IWOTH):
        return False
    try:
        parent = control_file.resolve().parent
        parent_stat = parent.stat()
    except OSError:
        return False
    try:
        if parent_stat.st_uid != os.getuid():
            return False
    except AttributeError:
        return False
    if parent_stat.st_mode & (stat.S_IWGRP | stat.S_IWOTH):
        return False
    return True


def passthrough_full_payload_capture_enabled() -> bool:
    if _is_truthy(os.environ.get(_FULL_PAYLOAD_ENV_FLAG, "")):
        return True
    control_file = Path(
        os.environ.get(
            _FULL_PAYLOAD_CONTROL_FILE_ENV,
            str(_DEFAULT_FULL_PAYLOAD_CONTROL_FILE),
        )
    )
    try:
        if not control_file.exists():
            return False
        if not _control_file_is_trusted(control_file):
            return False
        return _is_truthy(control_file.read_text(encoding="utf-8"))
    except Exception:
        return False


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


def _full_payload_aggregate_max_bytes() -> int:
    configured = os.environ.get(_FULL_PAYLOAD_AGGREGATE_MAX_BYTES_ENV)
    if not configured:
        return _DEFAULT_FULL_PAYLOAD_AGGREGATE_BYTES
    try:
        value = int(configured)
    except ValueError:
        return _DEFAULT_FULL_PAYLOAD_AGGREGATE_BYTES
    if value <= 0:
        return _DEFAULT_FULL_PAYLOAD_AGGREGATE_BYTES
    return max(_MIN_FULL_PAYLOAD_AGGREGATE_BYTES, value)


def _field_byte_limit() -> int:
    aggregate_limit = _full_payload_aggregate_max_bytes()
    configured = os.environ.get(_FULL_PAYLOAD_FIELD_MAX_BYTES_ENV)
    if configured:
        try:
            value = int(configured)
            if value > 0:
                return min(value, aggregate_limit)
        except ValueError:
            pass
    return min(_DEFAULT_FULL_PAYLOAD_FIELD_BYTES, aggregate_limit)


def _new_full_payload_budget(
    aggregate_limit: Optional[int] = None,
) -> _FullPayloadBudget:
    effective_limit = (
        _full_payload_aggregate_max_bytes()
        if aggregate_limit is None
        else max(_MIN_FULL_PAYLOAD_AGGREGATE_BYTES, aggregate_limit)
    )
    return _FullPayloadBudget(effective_limit)


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


def _header_name_is_sensitive(header_name: str) -> bool:
    normalized_name = str(header_name).lower()
    return any(term in normalized_name for term in _HEADER_DROP_TERMS)


def _full_payload_headers(
    headers: Optional[Mapping[str, Any]],
    *,
    budget: Optional[_FullPayloadBudget] = None,
    path: str = "$.headers",
) -> Dict[str, Any]:
    """Copy headers for full-payload capture, dropping sensitive names."""
    if not headers:
        return {}
    values: Dict[str, str] = {}
    truncated_headers: List[Dict[str, Any]] = []
    dropped_count = 0
    aggregate_dropped_count = 0
    retained_count = 0
    for header_name in headers:
        name_text = str(header_name)
        if _header_name_is_sensitive(name_text):
            continue
        if retained_count >= _FULL_PAYLOAD_MAX_HEADERS:
            dropped_count += 1
            continue
        if budget is not None and budget.remaining <= 0:
            aggregate_dropped_count += 1
            continue
        header_value = headers[header_name]
        value_text = str(header_value)
        field_limited_value, field_bytes, original_bytes = (
            _truncate_to_byte_limit_with_counts(
                value_text,
                _header_value_byte_limit(),
            )
        )
        value_path = f"{path}.{name_text.lower()}"
        truncated_value = _budgeted_full_payload_text(
            field_limited_value,
            budget=budget,
            path=value_path,
            byte_limit=_header_value_byte_limit(),
            limit_reason="header_value_bytes",
        )
        if field_limited_value != value_text:
            truncated_headers.append(
                {
                    "path": value_path,
                    "original_bytes": original_bytes,
                    "stored_bytes": field_bytes,
                    "reason": "header_value_bytes",
                }
            )
        values[name_text] = truncated_value
        retained_count += 1
    result: Dict[str, Any] = dict(
        sorted(values.items(), key=lambda item: item[0].lower())
    )
    if truncated_headers:
        result["_truncated_headers"] = truncated_headers
    if dropped_count:
        result["_truncated_header_count"] = dropped_count
        _record_truncation(
            budget,
            path=path,
            reason="header_count",
            dropped_count=dropped_count,
        )
    if aggregate_dropped_count:
        result["_aggregate_truncated_header_count"] = aggregate_dropped_count
        _record_truncation(
            budget,
            path=path,
            reason="aggregate_limit",
            dropped_count=aggregate_dropped_count,
        )
    return result


def _full_payload_header_items(
    headers: Optional[Mapping[str, Any]],
    *,
    budget: Optional[_FullPayloadBudget] = None,
    path: str = "$.header_items",
) -> List[Dict[str, Any]]:
    if not headers:
        return []
    multi_items = getattr(headers, "multi_items", None)
    try:
        raw_items = multi_items() if callable(multi_items) else headers.items()
    except Exception:
        raw_items = headers.items()
    items: List[Dict[str, Any]] = []
    dropped_count = 0
    aggregate_dropped_count = 0
    for header_name, header_value in raw_items:
        name_text = str(header_name)
        if _header_name_is_sensitive(name_text):
            continue
        if len(items) >= _FULL_PAYLOAD_MAX_HEADERS:
            dropped_count += 1
            continue
        if budget is not None and budget.remaining <= 0:
            aggregate_dropped_count += 1
            continue
        value_text = str(header_value)
        field_limited_value, field_bytes, original_bytes = (
            _truncate_to_byte_limit_with_counts(
                value_text,
                _header_value_byte_limit(),
            )
        )
        truncated_value = _budgeted_full_payload_text(
            field_limited_value,
            budget=budget,
            path=f"{path}[{len(items)}].value",
            byte_limit=_header_value_byte_limit(),
            limit_reason="header_value_bytes",
        )
        item: Dict[str, Any] = {"name": name_text, "value": truncated_value}
        if field_limited_value != value_text:
            item["truncated"] = True
            item["original_bytes"] = original_bytes
            item["stored_bytes"] = field_bytes
        items.append(item)
    if dropped_count:
        items.append(
            {
                "path": path,
                "reason": "header_count",
                "dropped_count": dropped_count,
            }
        )
        _record_truncation(
            budget,
            path=path,
            reason="header_count",
            dropped_count=dropped_count,
        )
    if aggregate_dropped_count:
        items.append(
            {
                "path": path,
                "reason": "aggregate_limit",
                "dropped_count": aggregate_dropped_count,
            }
        )
        _record_truncation(
            budget,
            path=path,
            reason="aggregate_limit",
            dropped_count=aggregate_dropped_count,
        )
    return items


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
    return _truncate_to_byte_limit(value, max_bytes)


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


def _truncate_to_byte_limit_with_counts(
    value: str, byte_limit: int
) -> tuple[str, int, int]:
    """Return UTF-8 truncated value plus original and stored byte counts."""

    byte_limit = max(0, byte_limit)
    if not value:
        return value, 0, 0

    original_bytes = 0
    stored_bytes = 0
    prefix_index = 0
    truncated = False

    for index, char in enumerate(value):
        char_bytes = char.encode("utf-8", errors="replace")
        char_len = len(char_bytes)
        original_bytes += char_len
        if not truncated and stored_bytes + char_len <= byte_limit:
            stored_bytes += char_len
            prefix_index = index + 1
        else:
            truncated = True

    if not truncated:
        return value, stored_bytes, original_bytes
    return value[:prefix_index], stored_bytes, original_bytes


def _truncate_to_byte_limit(value: str, byte_limit: int) -> str:
    truncated, _, _ = _truncate_to_byte_limit_with_counts(value, byte_limit)
    return truncated


def _truncate_to_field_bytes(value: str) -> str:
    return _truncate_to_byte_limit(value, _field_byte_limit())


def _header_value_byte_limit() -> int:
    return min(_field_byte_limit(), _FULL_PAYLOAD_HEADER_VALUE_MAX_BYTES)


def _truncate_header_value(value: str) -> str:
    return _truncate_to_byte_limit(value, _header_value_byte_limit())


def _record_truncation(
    budget: Optional[_FullPayloadBudget],
    *,
    path: str,
    reason: str,
    original_bytes: Optional[int] = None,
    stored_bytes: Optional[int] = None,
    dropped_count: Optional[int] = None,
) -> None:
    if budget is None:
        return
    record: Dict[str, Any] = {"path": path, "reason": reason}
    if original_bytes is not None:
        record["original_bytes"] = original_bytes
    if stored_bytes is not None:
        record["stored_bytes"] = stored_bytes
    if dropped_count is not None:
        record["dropped_count"] = dropped_count
    budget.truncations.append(record)


def _budgeted_full_payload_text(
    value: str,
    *,
    budget: Optional[_FullPayloadBudget],
    path: str,
    byte_limit: int,
    limit_reason: str,
) -> str:
    field_limited, field_bytes, original_bytes = _truncate_to_byte_limit_with_counts(
        value,
        byte_limit,
    )
    if field_limited != value:
        _record_truncation(
            budget,
            path=path,
            reason=limit_reason,
            original_bytes=original_bytes,
            stored_bytes=field_bytes,
        )
    if budget is None:
        return field_limited

    allowed_bytes = min(field_bytes, budget.remaining)
    if field_bytes <= allowed_bytes:
        stored = field_limited
        stored_bytes = field_bytes
        budget.consume(field_bytes)
    else:
        stored, stored_bytes, _ = _truncate_to_byte_limit_with_counts(
            field_limited,
            allowed_bytes,
        )
        budget.consume(stored_bytes)

    if stored != field_limited:
        _record_truncation(
            budget,
            path=path,
            reason="aggregate_limit",
            original_bytes=original_bytes,
            stored_bytes=stored_bytes,
        )
    return stored


def _budgeted_full_payload_base64(
    value: bytes,
    *,
    budget: Optional[_FullPayloadBudget],
    path: str,
) -> tuple[str, int]:
    max_bytes = _full_payload_max_bytes()
    field_limit = len(value) if max_bytes is None else min(len(value), max_bytes)
    if budget is not None:
        aggregate_raw_limit = (budget.remaining // 4) * 3
        stored_limit = min(field_limit, aggregate_raw_limit)
    else:
        stored_limit = field_limit
    stored_value = value[:stored_limit]
    encoded = base64.b64encode(stored_value).decode("ascii")
    if budget is not None:
        budget.consume(len(encoded))
    if field_limit < len(value):
        _record_truncation(
            budget,
            path=path,
            reason="field_bytes",
            original_bytes=len(value),
            stored_bytes=field_limit,
        )
    if stored_limit < field_limit:
        _record_truncation(
            budget,
            path=path,
            reason="aggregate_limit",
            original_bytes=len(value),
            stored_bytes=stored_limit,
        )
    return encoded, stored_limit


def _consume_full_payload_scalar(
    value: Any,
    *,
    budget: Optional[_FullPayloadBudget],
    path: str,
) -> bool:
    if budget is None:
        return True
    serialized_bytes = len(
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    if serialized_bytes > budget.remaining:
        original_remaining = budget.remaining
        budget.consume(original_remaining)
        _record_truncation(
            budget,
            path=path,
            reason="aggregate_limit",
            original_bytes=serialized_bytes,
            stored_bytes=0,
        )
        return False
    budget.consume(serialized_bytes)
    return True


def _consume_full_payload_overhead(
    *,
    budget: Optional[_FullPayloadBudget],
    path: str,
    overhead_bytes: int,
) -> bool:
    if budget is None or overhead_bytes <= 0:
        return True
    if budget.remaining <= 0:
        budget.consume(0)
        _record_truncation(
            budget,
            path=path,
            reason="aggregate_limit",
            original_bytes=overhead_bytes,
            stored_bytes=0,
        )
        return False
    consumed = min(overhead_bytes, budget.remaining)
    budget.consume(consumed)
    if consumed < overhead_bytes:
        _record_truncation(
            budget,
            path=path,
            reason="aggregate_limit",
            original_bytes=overhead_bytes,
            stored_bytes=consumed,
        )
        return False
    return True


def _jsonable_full_payload_mapping(
    value: Mapping[Any, Any],
    *,
    budget: Optional[_FullPayloadBudget],
    path: str,
    depth: int,
) -> Dict[str, Any]:
    if not _consume_full_payload_overhead(
        budget=budget,
        path=path,
        overhead_bytes=_FULL_PAYLOAD_CONTAINER_OVERHEAD_BYTES,
    ):
        return _FULL_PAYLOAD_OMITTED
    result: Dict[str, Any] = {}
    cardinality_dropped_count = 0
    aggregate_dropped_count = 0
    for index, key in enumerate(value):
        if index >= _FULL_PAYLOAD_MAX_DICT_KEYS:
            cardinality_dropped_count += 1
            continue
        if budget is not None and budget.remaining <= 0:
            aggregate_dropped_count += 1
            continue
        key_text = _truncate_to_field_bytes(str(key))
        key_bytes = len(key_text.encode("utf-8", errors="replace"))
        if budget is not None and key_bytes > budget.remaining:
            budget.consume(budget.remaining)
            aggregate_dropped_count += 1
            continue
        if not _consume_full_payload_overhead(
            budget=budget,
            path=f"{path}.{key_text}",
            overhead_bytes=_FULL_PAYLOAD_KEY_OVERHEAD_BYTES,
        ):
            aggregate_dropped_count += 1
            continue
        if budget is not None:
            budget.consume(key_bytes)
        child_value = _jsonable_full_payload(
            value[key],
            budget=budget,
            path=f"{path}.{key_text}",
            depth=depth + 1,
        )
        if child_value is _FULL_PAYLOAD_OMITTED:
            aggregate_dropped_count += 1
            continue
        result[key_text] = child_value
    if cardinality_dropped_count:
        _record_truncation(
            budget,
            path=path,
            reason="dict_key_limit",
            dropped_count=cardinality_dropped_count,
        )
    if aggregate_dropped_count:
        _record_truncation(
            budget,
            path=path,
            reason="aggregate_limit",
            dropped_count=aggregate_dropped_count,
        )
    return result


def _jsonable_full_payload_sequence(
    value: Sequence[Any],
    *,
    budget: Optional[_FullPayloadBudget],
    path: str,
    depth: int,
) -> List[Any]:
    if not _consume_full_payload_overhead(
        budget=budget,
        path=path,
        overhead_bytes=_FULL_PAYLOAD_CONTAINER_OVERHEAD_BYTES,
    ):
        return _FULL_PAYLOAD_OMITTED
    total_items = len(value)
    retained_limit = min(total_items, _FULL_PAYLOAD_MAX_LIST_ITEMS)
    result: List[Any] = []
    aggregate_dropped_count = 0
    for index in range(retained_limit):
        if not _consume_full_payload_overhead(
            budget=budget,
            path=f"{path}[{index}]",
            overhead_bytes=_FULL_PAYLOAD_LIST_ITEM_OVERHEAD_BYTES,
        ):
            aggregate_dropped_count += retained_limit - index
            break
        if budget is not None and budget.remaining <= 0:
            aggregate_dropped_count += retained_limit - index
            break
        child_value = _jsonable_full_payload(
            value[index],
            budget=budget,
            path=f"{path}[{index}]",
            depth=depth + 1,
        )
        if child_value is _FULL_PAYLOAD_OMITTED:
            aggregate_dropped_count += retained_limit - index
            break
        result.append(child_value)
    if total_items > _FULL_PAYLOAD_MAX_LIST_ITEMS:
        _record_truncation(
            budget,
            path=path,
            reason="list_item_limit",
            dropped_count=total_items - _FULL_PAYLOAD_MAX_LIST_ITEMS,
        )
    if aggregate_dropped_count:
        _record_truncation(
            budget,
            path=path,
            reason="aggregate_limit",
            dropped_count=aggregate_dropped_count,
        )
    return result


def _jsonable_full_payload(
    value: Any,
    *,
    budget: Optional[_FullPayloadBudget] = None,
    path: str = "$",
    depth: int = 0,
) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        value = str(value)
    if value is None or isinstance(value, bool) or isinstance(value, (int, float)):
        if not _consume_full_payload_scalar(value, budget=budget, path=path):
            return _FULL_PAYLOAD_OMITTED
        return value
    if isinstance(value, str):
        return _budgeted_full_payload_text(
            value,
            budget=budget,
            path=path,
            byte_limit=_field_byte_limit(),
            limit_reason="field_bytes",
        )
    if isinstance(value, bytes):
        data, stored_bytes = _budgeted_full_payload_base64(
            value,
            budget=budget,
            path=path,
        )
        wrapper_overhead_bytes = len(
            json.dumps(
                {
                    "encoding": "base64",
                    "data": "",
                    "truncated": stored_bytes < len(value),
                    "original_bytes": len(value),
                    "stored_bytes": stored_bytes,
                },
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        if not _consume_full_payload_overhead(
            budget=budget,
            path=path,
            overhead_bytes=wrapper_overhead_bytes,
        ):
            return _FULL_PAYLOAD_OMITTED
        return {
            "encoding": "base64",
            "data": data,
            "truncated": stored_bytes < len(value),
            "original_bytes": len(value),
            "stored_bytes": stored_bytes,
        }
    if depth >= _FULL_PAYLOAD_MAX_DEPTH:
        _record_truncation(budget, path=path, reason="depth_limit")
        return {"_truncated": True, "reason": "depth_limit"}
    if isinstance(value, Mapping):
        return _jsonable_full_payload_mapping(
            value,
            budget=budget,
            path=path,
            depth=depth,
        )
    if isinstance(value, (list, tuple)):
        return _jsonable_full_payload_sequence(
            value,
            budget=budget,
            path=path,
            depth=depth,
        )
    return _budgeted_full_payload_text(
        str(value),
        budget=budget,
        path=path,
        byte_limit=_field_byte_limit(),
        limit_reason="field_bytes",
    )


def _full_response_body(
    response_body: Any,
    response_content: Optional[bytes],
    *,
    budget: Optional[_FullPayloadBudget] = None,
    path: str = "$.response.body",
) -> Dict[str, Any]:
    body: Dict[str, Any] = {}
    parsed_body = response_body
    if parsed_body is None:
        parse_content = (
            response_content[: _field_byte_limit()]
            if response_content is not None
            else None
        )
        parsed_body = _parse_json_text(_decode_body_bytes(parse_content))
    if parsed_body is not None:
        json_body = _jsonable_full_payload(
            parsed_body,
            budget=budget,
            path=f"{path}.json",
        )
        if json_body is not _FULL_PAYLOAD_OMITTED:
            body["json"] = json_body
    if response_content is not None:
        content_base64, stored_bytes = _budgeted_full_payload_base64(
            response_content,
            budget=budget,
            path=f"{path}.content_base64",
        )
        stored_content = response_content[:stored_bytes]
        body["content_base64"] = content_base64
        decoded_content = _decode_body_bytes(stored_content)
        if decoded_content is not None:
            body["content_text"] = _budgeted_full_payload_text(
                decoded_content,
                budget=budget,
                path=f"{path}.content_text",
                byte_limit=_field_byte_limit(),
                limit_reason="field_bytes",
            )
        body["truncated"] = stored_bytes < len(response_content)
        body["original_bytes"] = len(response_content)
        body["stored_bytes"] = stored_bytes
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
    budget: Optional[_FullPayloadBudget] = None,
) -> Dict[str, Any]:
    if upstream_request is None:
        body = _jsonable_full_payload(
            fallback_request_body,
            budget=budget,
            path="$.request.body",
        )
        return {} if body is _FULL_PAYLOAD_OMITTED else {"body": body}

    request_content = _httpx_request_content(upstream_request)
    payload: Dict[str, Any] = {
        "method": _budgeted_full_payload_text(
            upstream_request.method,
            budget=budget,
            path="$.request.method",
            byte_limit=_field_byte_limit(),
            limit_reason="field_bytes",
        ),
        "url": _budgeted_full_payload_text(
            str(upstream_request.url),
            budget=budget,
            path="$.request.url",
            byte_limit=_field_byte_limit(),
            limit_reason="field_bytes",
        ),
        "headers": _full_payload_headers(
            upstream_request.headers,
            budget=budget,
            path="$.request.headers",
        ),
        "header_items": _full_payload_header_items(
            upstream_request.headers,
            budget=budget,
            path="$.request.header_items",
        ),
    }
    if request_content is not None:
        payload["body"] = _full_response_body(
            None,
            request_content,
            budget=budget,
            path="$.request.body",
        )
    else:
        body = _jsonable_full_payload(
            fallback_request_body, budget=budget, path="$.request.body"
        )
        if body is not _FULL_PAYLOAD_OMITTED:
            payload["body"] = body
            payload["body_source"] = "fallback_request_body"
    if fallback_url_route and str(upstream_request.url) != fallback_url_route:
        payload["logging_url"] = _budgeted_full_payload_text(
            fallback_url_route,
            budget=budget,
            path="$.request.logging_url",
            byte_limit=_field_byte_limit(),
            limit_reason="field_bytes",
        )
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
    budget: Optional[_FullPayloadBudget] = None,
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
            budget=budget,
        ),
        "response": {
            "status_code": response.status_code if response is not None else None,
            "headers": _full_payload_headers(
                _response_headers(response),
                budget=budget,
                path="$.response.headers",
            ),
            "header_items": _full_payload_header_items(
                _response_headers(response),
                budget=budget,
                path="$.response.header_items",
            ),
        },
    }
    if extra_metadata:
        metadata = _jsonable_full_payload(
            extra_metadata,
            budget=budget,
            path="$.metadata",
        )
        if metadata is not _FULL_PAYLOAD_OMITTED:
            artifact["metadata"] = metadata
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


def _ensure_private_dir(path: Path) -> None:
    """Create path (and parents) with mode 0700; tighten existing dirs."""
    parts_to_create: List[Path] = []
    current = path
    while not current.exists():
        parts_to_create.append(current)
        if current.parent == current:
            break
        current = current.parent
    for directory in reversed(parts_to_create):
        try:
            directory.mkdir(mode=0o700, exist_ok=True)
        except FileExistsError:
            pass
        try:
            os.chmod(directory, 0o700)
        except OSError:
            pass
    if path.exists():
        try:
            os.chmod(path, 0o700)
        except OSError:
            pass


def _atomic_write_private_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Atomically write JSON with mode 0600 at creation (no write-then-chmod)."""
    serialized = _serialize_full_payload_json(payload)
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.{_next_counter()}.tmp")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    fd = os.open(str(tmp_path), flags, 0o600)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.chmod(tmp_path, 0o600)
        except OSError:
            pass
        os.replace(str(tmp_path), str(path))
    except Exception:
        try:
            tmp_path.unlink()
        except OSError:
            pass
        raise
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass


def _serialize_full_payload_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(
        payload,
        indent=2,
        sort_keys=True,
        default=str,
        ensure_ascii=False,
        allow_nan=False,
    )


def _full_payload_serialized_bytes(payload: Mapping[str, Any]) -> int:
    return len(_serialize_full_payload_json(payload).encode("utf-8"))


def _full_payload_fits(
    payload: Mapping[str, Any],
    aggregate_limit: int,
) -> bool:
    try:
        return _full_payload_serialized_bytes(payload) <= aggregate_limit
    except (TypeError, ValueError, OverflowError):
        return False


_FULL_PAYLOAD_TRUNCATION_NUMERIC_KEYS = (
    "original_bytes",
    "stored_bytes",
    "dropped_count",
)


def _sanitize_full_payload_truncation_record(
    record: Mapping[str, Any],
) -> Dict[str, Any]:
    sanitized: Dict[str, Any] = {
        "path": _truncate_to_byte_limit(str(record.get("path") or "$"), 256),
        "reason": _truncate_to_byte_limit(
            str(record.get("reason") or "aggregate_limit"),
            64,
        ),
    }
    for key in _FULL_PAYLOAD_TRUNCATION_NUMERIC_KEYS:
        value = record.get(key)
        if isinstance(value, bool):
            continue
        if isinstance(value, int):
            sanitized[key] = max(0, value)
        elif isinstance(value, float) and math.isfinite(value):
            sanitized[key] = max(0, int(value))
    return sanitized


_FULL_PAYLOAD_AGGREGATE_CORE_KEYS = (
    "captured_at",
    "capture_kind",
    "capture_scope",
    "mode",
    "provider",
    "endpoint_type",
    "litellm_call_id",
    "url",
)


def _enforce_full_payload_aggregate_limit(
    artifact: Dict[str, Any],
) -> Dict[str, Any]:
    """Enforce the exact aggregate artifact ceiling before serialization."""
    aggregate_limit = _full_payload_aggregate_max_bytes()
    normalized = dict(artifact)
    truncations: List[Dict[str, Any]] = [
        _sanitize_full_payload_truncation_record(record)
        for record in (normalized.get("truncations") or [])
        if isinstance(record, Mapping)
    ]
    if truncations:
        normalized["truncations"] = truncations
    else:
        normalized.pop("truncations", None)
    normalized["aggregate_limit_bytes"] = aggregate_limit
    if any(
        record.get("reason") == "aggregate_limit" for record in truncations
    ):
        normalized["aggregate_truncated"] = True
    if _full_payload_fits(normalized, aggregate_limit):
        return normalized

    for key in normalized:
        if key in _FULL_PAYLOAD_AGGREGATE_CORE_KEYS or key == "truncations":
            continue
        if key in {"aggregate_limit_bytes", "aggregate_truncated"}:
            continue
        truncations.append(
            {
                "path": f"$.{key}",
                "reason": "aggregate_limit",
            }
        )
    reduced: Dict[str, Any] = {
        key: value
        for key, value in normalized.items()
        if key in _FULL_PAYLOAD_AGGREGATE_CORE_KEYS
    }
    reduced["aggregate_limit_bytes"] = aggregate_limit
    reduced["aggregate_truncated"] = True
    reduced["truncations"] = truncations
    for key in (
        "url",
        "endpoint_type",
        "capture_scope",
        "captured_at",
        "provider",
        "mode",
        "litellm_call_id",
        "capture_kind",
    ):
        if _full_payload_fits(reduced, aggregate_limit):
            return reduced
        reduced.pop(key, None)
    if _full_payload_fits(reduced, aggregate_limit):
        return reduced

    fallback = {
        "aggregate_limit_bytes": aggregate_limit,
        "aggregate_truncated": True,
        "truncations": [
            {
                "path": "$",
                "reason": "aggregate_limit",
                "dropped_count": max(1, len(truncations)),
            }
        ],
    }
    if not _full_payload_fits(fallback, aggregate_limit):
        raise ValueError(
            "full-payload aggregate minimum is too small for truncation metadata"
        )
    return fallback


def _write_full_payload_artifact(artifact: Dict[str, Any]) -> Optional[str]:
    if not passthrough_full_payload_capture_enabled():
        return None
    try:
        capture_dir = _full_payload_capture_dir()
        _ensure_private_dir(capture_dir)
        ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        counter = _next_counter()
        provider = _sanitize_filename_part(artifact.get("provider"))
        mode = _sanitize_filename_part(artifact.get("mode"))
        call_id = _sanitize_filename_part(artifact.get("litellm_call_id"))[:18]
        path = capture_dir / f"{ts}_{counter:04d}_{provider}_{mode}_{call_id}.json"
        limited_artifact = _enforce_full_payload_aggregate_limit(artifact)
        _atomic_write_private_json(path, limited_artifact)
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
        budget = _new_full_payload_budget()
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
            budget=budget,
        )
        full_payload_artifact["response"]["body"] = _full_response_body(
            response_body,
            response_content,
            budget=budget,
            path="$.response.body",
        )
        full_payload_artifact["truncations"] = budget.truncations
        full_payload_artifact["aggregate_limit_bytes"] = budget.limit
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


def _json_shape_payload(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_shape_payload(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_shape_payload(item) for item in value]
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        try:
            dumped = model_dump(exclude_none=True)
            if isinstance(dumped, (Mapping, list, str, int, float, bool)) or dumped is None:
                return _json_shape_payload(dumped)
        except Exception:
            pass
    dict_method = getattr(value, "dict", None)
    if callable(dict_method):
        try:
            dumped = dict_method(exclude_none=True)
            if isinstance(dumped, (Mapping, list, str, int, float, bool)) or dumped is None:
                return _json_shape_payload(dumped)
        except Exception:
            pass
    return {"type": type(value).__name__}


def capture_rerank_shape(
    *,
    url_route: Optional[str] = None,
    request_body: Any = None,
    response_body: Any = None,
    litellm_call_id: Optional[str] = None,
    extra_metadata: Optional[Mapping[str, Any]] = None,
) -> Optional[str]:
    if not diagnostic_payload_capture_enabled():
        return None
    metadata: Dict[str, Any] = dict(extra_metadata or {})
    metadata.setdefault("route_family", "rerank")
    metadata.setdefault("endpoint_template", "/rerank")
    response_payload = _json_shape_payload(response_body)
    return _write_diagnostic_payload_artifact(
        _diagnostic_payload_artifact(
            mode="nonstream",
            provider=str(metadata.get("custom_llm_provider") or "rerank"),
            endpoint_type="rerank",
            url_route=url_route,
            request_body=request_body,
            response=None,
            upstream_request=None,
            response_body=response_payload,
            response_content=None,
            all_chunks=None,
            raw_bytes=None,
            litellm_call_id=litellm_call_id,
            extra_metadata=metadata,
        )
    )


def _full_payload_stream_payload(
    all_chunks: Sequence[str],
    raw_bytes: Optional[Sequence[bytes]],
    *,
    budget: _FullPayloadBudget,
) -> Dict[str, Any]:
    total_lines = len(all_chunks)
    retained_line_limit = min(total_lines, _FULL_PAYLOAD_MAX_LIST_ITEMS)
    lines: List[str] = []
    aggregate_line_dropped_count = 0
    for index in range(retained_line_limit):
        if budget.remaining <= 0:
            aggregate_line_dropped_count += retained_line_limit - index
            break
        lines.append(
            _budgeted_full_payload_text(
                str(all_chunks[index]),
                budget=budget,
                path=f"$.response.stream.lines[{index}]",
                byte_limit=_field_byte_limit(),
                limit_reason="field_bytes",
            )
        )
    stream_payload: Dict[str, Any] = {
        "line_count": total_lines,
        "lines": lines,
    }
    if total_lines > _FULL_PAYLOAD_MAX_LIST_ITEMS:
        _record_truncation(
            budget,
            path="$.response.stream.lines",
            reason="list_item_limit",
            dropped_count=total_lines - _FULL_PAYLOAD_MAX_LIST_ITEMS,
        )
    if aggregate_line_dropped_count:
        _record_truncation(
            budget,
            path="$.response.stream.lines",
            reason="aggregate_limit",
            dropped_count=aggregate_line_dropped_count,
        )

    if raw_bytes is None:
        return stream_payload

    total_raw_chunks = len(raw_bytes)
    retained_raw_limit = min(total_raw_chunks, _FULL_PAYLOAD_MAX_LIST_ITEMS)
    raw_chunks_base64: List[str] = []
    raw_chunks_truncated: List[bool] = []
    aggregate_raw_dropped_count = 0
    raw_total_bytes = 0
    for index in range(total_raw_chunks):
        raw_total_bytes += len(raw_bytes[index])
    for index in range(retained_raw_limit):
        if budget.remaining <= 0:
            aggregate_raw_dropped_count += retained_raw_limit - index
            break
        chunk = raw_bytes[index]
        encoded, stored_bytes = _budgeted_full_payload_base64(
            chunk,
            budget=budget,
            path=f"$.response.stream.raw_chunks[{index}]",
        )
        raw_chunks_base64.append(encoded)
        raw_chunks_truncated.append(stored_bytes < len(chunk))
    stream_payload["raw_chunk_count"] = total_raw_chunks
    stream_payload["raw_total_bytes"] = raw_total_bytes
    stream_payload["raw_chunks_base64"] = raw_chunks_base64
    stream_payload["raw_chunks_truncated"] = raw_chunks_truncated
    if total_raw_chunks > _FULL_PAYLOAD_MAX_LIST_ITEMS:
        _record_truncation(
            budget,
            path="$.response.stream.raw_chunks",
            reason="list_item_limit",
            dropped_count=total_raw_chunks - _FULL_PAYLOAD_MAX_LIST_ITEMS,
        )
    if aggregate_raw_dropped_count:
        _record_truncation(
            budget,
            path="$.response.stream.raw_chunks",
            reason="aggregate_limit",
            dropped_count=aggregate_raw_dropped_count,
        )
    return stream_payload


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
        budget = _new_full_payload_budget()
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
            budget=budget,
        )
        full_payload_artifact["response"]["stream"] = _full_payload_stream_payload(
            all_chunks,
            raw_bytes,
            budget=budget,
        )
        full_payload_artifact["truncations"] = budget.truncations
        full_payload_artifact["aggregate_limit_bytes"] = budget.limit
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
