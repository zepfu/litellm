"""Google Code Assist shaping implementation slice.

This module owns provider algorithms. Route-layer dependencies are rebound by
the compatibility facade before entry so existing monkeypatch contracts remain
live without suppressing undefined-name checks.
"""

from __future__ import annotations

from dataclasses import dataclass

import copy
from collections.abc import Mapping
from typing import Any, Optional

import httpx

from litellm.proxy.pass_through_endpoints.pass_through_endpoints import (
    HttpPassThroughEndpointHelpers,
)


@dataclass(frozen=True)
class Runtime:
    _GOOGLE_ADAPTER_TRANSIENT_UPSTREAM_STATUS_CODES: Any
    _GOOGLE_CODE_ASSIST_SCHEMA_SANITIZE_MAX_DEPTH: Any
    _classify_passthrough_hidden_retry_failure: Any
    _extract_embedded_json_payload_candidates: Any
    _google_adapter_hidden_retry_metadata: Any
    _parse_json_payloads_from_text_candidates: Any
    _sanitize_openai_object_schema_properties: Any
    _simplify_google_code_assist_union_schema: Any


_RUNTIME: Optional[Runtime] = None


def bind_runtime(namespace: Mapping[str, object]) -> None:
    global _RUNTIME
    _RUNTIME = Runtime(
        _GOOGLE_ADAPTER_TRANSIENT_UPSTREAM_STATUS_CODES=namespace["_GOOGLE_ADAPTER_TRANSIENT_UPSTREAM_STATUS_CODES"],
        _GOOGLE_CODE_ASSIST_SCHEMA_SANITIZE_MAX_DEPTH=namespace["_GOOGLE_CODE_ASSIST_SCHEMA_SANITIZE_MAX_DEPTH"],
        _classify_passthrough_hidden_retry_failure=namespace["_classify_passthrough_hidden_retry_failure"],
        _extract_embedded_json_payload_candidates=namespace["_extract_embedded_json_payload_candidates"],
        _google_adapter_hidden_retry_metadata=namespace["_google_adapter_hidden_retry_metadata"],
        _parse_json_payloads_from_text_candidates=namespace["_parse_json_payloads_from_text_candidates"],
        _sanitize_openai_object_schema_properties=namespace["_sanitize_openai_object_schema_properties"],
        _simplify_google_code_assist_union_schema=namespace["_simplify_google_code_assist_union_schema"],
    )


def _runtime() -> Runtime:
    if _RUNTIME is None:
        raise RuntimeError("Google shaping runtime is not bound")
    return _RUNTIME


def _extract_google_adapter_exception_status_code(exc: Any) -> Optional[int]:
    for attr_name in ("status_code", "code"):
        raw_status = getattr(exc, attr_name, None)
        if isinstance(raw_status, int):
            return raw_status
        if isinstance(raw_status, str):
            try:
                return int(raw_status)
            except Exception:
                pass
    response = getattr(exc, "response", None)
    response_status = getattr(response, "status_code", None)
    if isinstance(response_status, int):
        return response_status
    return None


def _extract_google_adapter_exception_detail(exc: Any) -> Any:
    for attr_name in ("detail", "message"):
        detail = getattr(exc, attr_name, None)
        if detail is not None:
            return detail
    response = getattr(exc, "response", None)
    if response is not None:
        response_content = getattr(response, "content", None)
        if response_content:
            return response_content
        response_text = getattr(response, "text", None)
        if response_text:
            return response_text
    return str(exc)


def _extract_google_adapter_error_payloads(exc: Any) -> list[Any]:
    detail = _extract_google_adapter_exception_detail(exc)
    return _runtime()._parse_json_payloads_from_text_candidates(
        _runtime()._extract_embedded_json_payload_candidates(detail)
    )


def _extract_google_adapter_error_reason(exc: Any) -> Optional[str]:
    for parsed in _extract_google_adapter_error_payloads(exc):
        error_blocks: list[dict[str, Any]] = []
        if isinstance(parsed, dict):
            error_block = parsed.get("error")
            if isinstance(error_block, dict):
                error_blocks.append(error_block)
        elif isinstance(parsed, list):
            for item in parsed:
                if not isinstance(item, dict):
                    continue
                error_block = item.get("error")
                if isinstance(error_block, dict):
                    error_blocks.append(error_block)
        for error_block in error_blocks:
            details = error_block.get("details")
            if not isinstance(details, list):
                continue
            for item in details:
                if not isinstance(item, dict):
                    continue
                reason = item.get("reason")
                if isinstance(reason, str) and reason:
                    return reason
    return None


def _extract_google_adapter_error_payload_for_logging(exc: Any) -> dict[str, Any]:
    for parsed in _extract_google_adapter_error_payloads(exc):
        if isinstance(parsed, dict) and isinstance(parsed.get("error"), dict):
            return dict(parsed)
        if isinstance(parsed, list):
            for item in parsed:
                if isinstance(item, dict) and isinstance(item.get("error"), dict):
                    return dict(item)
            return {"payload": parsed}
    return {}


def _is_google_adapter_transient_retryable_failure(
    exc: Any,
    *,
    status_code: Optional[int],
    error_reason: Optional[str],
) -> bool:
    if status_code == 429 or error_reason in {
        "MODEL_CAPACITY_EXHAUSTED",
        "RATE_LIMIT_EXCEEDED",
    }:
        return False
    if status_code in _runtime()._GOOGLE_ADAPTER_TRANSIENT_UPSTREAM_STATUS_CODES:
        return True
    if isinstance(exc, (httpx.TimeoutException, httpx.ConnectError)):
        return True
    (
        _status_code,
        failure_class,
        _failure_classification,
    ) = _runtime()._classify_passthrough_hidden_retry_failure(exc)
    return failure_class in {
        "upstream_connectivity_failure",
        "transport_dns_failure",
    }


def _build_google_adapter_terminal_error_log_context(
    passthrough_kwargs: dict[str, Any],
    *,
    status_code: Optional[int],
    failure_classification: Optional[str],
) -> dict[str, Any]:
    metadata = _runtime()._google_adapter_hidden_retry_metadata(passthrough_kwargs)
    request = passthrough_kwargs.get("request")
    endpoint = None
    if request is not None:
        try:
            endpoint = HttpPassThroughEndpointHelpers._get_passthrough_request_url_path(request)
        except Exception:
            endpoint = None
    custom_body = passthrough_kwargs.get("custom_body")
    model = None
    if isinstance(custom_body, dict):
        model = custom_body.get("model")
    return {
        "source": "google_code_assist_adapter",
        "endpoint": endpoint,
        "upstream_url": passthrough_kwargs.get("target"),
        "provider": passthrough_kwargs.get("custom_llm_provider") or metadata.get("custom_llm_provider"),
        "model": model or metadata.get("model_group"),
        "model_alias": metadata.get("requested_model_alias"),
        "route_family": metadata.get("passthrough_route_family"),
        "status_code": status_code,
        "failure_kind": (
            "transient_provider_connectivity"
            if failure_classification in {"transport_dns_failure", "upstream_connectivity_failure"}
            else "expected_upstream_capacity_or_internal"
        ),
        "hidden_retry_final_outcome": metadata.get("aawm_passthrough_hidden_retry_final_outcome"),
        "hidden_retry_failure_classification": failure_classification,
        "hidden_retry_count": metadata.get("aawm_passthrough_hidden_retry_count"),
        "trace_id": metadata.get("trace_id"),
    }


def _normalize_google_completion_adapter_model_name(model: str) -> str:
    normalized_model = model.strip()
    if normalized_model.startswith(("gemini/", "google/")):
        normalized_model = normalized_model.split("/", 1)[1]

    # Claude-facing agent configs use stable shorthand names or newer naming
    # variants; Google Code Assist currently serves the corresponding model ids.
    google_model_aliases = {
        "gemini-3.1": "gemini-3.1-pro-preview",
        "gemini-3.1-flash-lite": "gemini-3.1-flash-lite-preview",
    }
    return google_model_aliases.get(normalized_model, normalized_model)


def _sanitize_google_schema_array_items(
    schema_node: Any,
    *,
    _depth: int = 0,
    _seen: Optional[set[int]] = None,
) -> int:
    # RR-054 #37: depth/cycle bound (same budget as union schema sanitize).
    if _depth > _runtime()._GOOGLE_CODE_ASSIST_SCHEMA_SANITIZE_MAX_DEPTH:
        return 0
    if _seen is None:
        _seen = set()
    if isinstance(schema_node, (dict, list)):
        node_id = id(schema_node)
        if node_id in _seen:
            return 0
        _seen.add(node_id)
    fix_count = 0
    if isinstance(schema_node, dict):
        if schema_node.get("type") == "array":
            items = schema_node.get("items")
            if not isinstance(items, dict) or not items.get("type"):
                schema_node["items"] = {"type": "string"}
                fix_count += 1
        for value in list(schema_node.values()):
            fix_count += _sanitize_google_schema_array_items(value, _depth=_depth + 1, _seen=_seen)
    elif isinstance(schema_node, list):
        for item in schema_node:
            fix_count += _sanitize_google_schema_array_items(item, _depth=_depth + 1, _seen=_seen)
    return fix_count


def _merge_google_code_assist_schema_annotations(
    source: dict[str, Any],
    target: dict[str, Any],
) -> None:
    for key in ("description", "title", "default"):
        if key in source and key not in target:
            target[key] = copy.deepcopy(source[key])


def _sanitize_google_code_assist_union_schemas(
    schema_node: Any,
    *,
    _depth: int = 0,
    _seen: Optional[set[int]] = None,
) -> int:
    # RR-054 #5: bound recursion / cycles so client tool schemas cannot DoS the worker.
    if _depth > _runtime()._GOOGLE_CODE_ASSIST_SCHEMA_SANITIZE_MAX_DEPTH:
        return 0
    if _seen is None:
        _seen = set()
    fix_count = 0
    node_id = id(schema_node)
    if isinstance(schema_node, (dict, list)):
        if node_id in _seen:
            return 0
        _seen.add(node_id)
    try:
        if isinstance(schema_node, dict):
            fix_count += _runtime()._simplify_google_code_assist_union_schema(schema_node)
            for value in list(schema_node.values()):
                fix_count += _sanitize_google_code_assist_union_schemas(value, _depth=_depth + 1, _seen=_seen)
        elif isinstance(schema_node, list):
            for item in schema_node:
                fix_count += _sanitize_google_code_assist_union_schemas(item, _depth=_depth + 1, _seen=_seen)
    except RecursionError:
        return fix_count
    return fix_count


def _sanitize_google_code_assist_tool_schema(schema_node: Any) -> int:
    fix_count = 0
    if not isinstance(schema_node, dict):
        return fix_count

    fix_count += _sanitize_google_code_assist_union_schemas(schema_node)
    if schema_node.get("type") is None:
        schema_node["type"] = "object"
        fix_count += 1
    if schema_node.get("type") == "object" and not isinstance(schema_node.get("properties"), dict):
        schema_node["properties"] = {}
        fix_count += 1

    fix_count += _sanitize_google_schema_array_items(schema_node)
    fix_count += _runtime()._sanitize_openai_object_schema_properties(schema_node)
    return fix_count
