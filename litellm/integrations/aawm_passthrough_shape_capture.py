"""Sanitized pass-through response shape capture for AAWM investigations.

This is intentionally separate from ``aawm_payload_capture``. It captures the
shape of upstream provider responses as LiteLLM receives them, while avoiding
full prompt/body/content persistence.

Enable with ``AAWM_CAPTURE_PASSTHROUGH_SHAPES=1``. Artifacts are written under
``/tmp/captures/pass_through_shapes`` by default, which maps to ``./captures``
in the local dev compose stack.
"""

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
_DEFAULT_CAPTURE_DIR = Path("/tmp/captures/pass_through_shapes")
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


def _capture_dir() -> Path:
    configured = os.environ.get(_DIR_ENV)
    if configured:
        return Path(configured)
    return _DEFAULT_CAPTURE_DIR


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


def _sanitize_filename_part(value: Any) -> str:
    text = str(value or "unknown").lower()
    text = re.sub(r"[^a-z0-9_.-]+", "_", text)
    return text[:60] or "unknown"


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
        "scheme": parsed.scheme or None,
        "host": parsed.hostname or None,
        "path": parsed.path or None,
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


def _response_headers(response: Optional[httpx.Response]) -> Optional[Mapping[str, Any]]:
    if response is None:
        return None
    return response.headers


def _decode_body_bytes(content: Optional[bytes]) -> Optional[str]:
    if content is None:
        return None
    try:
        return content.decode("utf-8", errors="replace")
    except Exception:
        return None


def _parse_json_text(text: Optional[str]) -> Any:
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        return None


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
        return str(path)
    except Exception as exc:
        verbose_proxy_logger.warning(
            "AawmPassthroughShapeCapture: capture failed: %s", exc
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
    response_body: Any = None,
    response_content: Optional[bytes] = None,
    litellm_call_id: Optional[str] = None,
    extra_metadata: Optional[Mapping[str, Any]] = None,
) -> Optional[str]:
    if not passthrough_shape_capture_enabled():
        return None
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
    return _write_artifact(artifact)


def capture_passthrough_stream_shape(
    *,
    provider: Optional[str],
    endpoint_type: Any = None,
    url_route: Optional[str] = None,
    request_body: Any = None,
    response: Optional[httpx.Response] = None,
    all_chunks: Sequence[str],
    litellm_call_id: Optional[str] = None,
    extra_metadata: Optional[Mapping[str, Any]] = None,
) -> Optional[str]:
    if not passthrough_shape_capture_enabled():
        return None
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
    return _write_artifact(artifact)
