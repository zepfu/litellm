"""Append-only local intake for detected malformed tool-call events."""

from __future__ import annotations

import os
import threading
from datetime import UTC, datetime
from typing import Any, Dict, List, Optional

from litellm._logging import (
    _AAWM_ERROR_LOG_LOCK,
    _get_aawm_error_log_dir,
    _get_aawm_error_log_environment,
    _normalize_aawm_error_log_file_metadata,
    _parse_aawm_error_log_non_negative_int_env,
)
from litellm.integrations.aawm_agent_quality_rules import (
    is_malformed_composer_call_literal_text,
)
from litellm.litellm_core_utils.safe_json_dumps import safe_dumps

MALFORMED_ERROR_JSONL_FILENAME = "malformed-error.jsonl"
MALFORMED_TOOL_CALL_SCHEMA_VERSION = 1
MALFORMED_TOOL_CALL_ERROR_CODE = "aawm_auto_agent_malformed_tool_call_text"
MALFORMED_TOOL_CALL_FAILURE_KIND = "malformed_tool_call"

_DEFAULT_MAX_TEXT_CHARS = 8_192
_DEFAULT_MAX_PAYLOAD_CHARS = 16_384
_MALFORMED_ERROR_LOG_LOCK = threading.Lock()


def _malformed_error_log_enabled() -> bool:
    raw = os.getenv("LITELLM_AAWM_MALFORMED_ERROR_LOG_ENABLED", "").strip()
    if raw:
        return raw.lower() in {"1", "true", "yes", "on"}
    if os.getenv("LITELLM_AAWM_ERROR_LOG_DIR", "").strip():
        return True
    if os.getenv("LITELLM_AAWM_ERROR_LOG_ENABLED", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }:
        return True
    return False


def _get_malformed_error_log_path() -> Optional[str]:
    if not _malformed_error_log_enabled():
        return None
    log_dir = _get_aawm_error_log_dir()
    if not log_dir:
        log_dir = os.path.join(os.getcwd(), ".analysis")
    if not log_dir:
        return None
    return os.path.join(log_dir, MALFORMED_ERROR_JSONL_FILENAME)


def _max_malformed_error_log_file_bytes() -> Optional[int]:
    configured = _parse_aawm_error_log_non_negative_int_env(
        "LITELLM_AAWM_MALFORMED_ERROR_LOG_MAX_BYTES"
    )
    if configured is None or configured <= 0:
        return None
    return configured


def _truncate_text(value: Any, *, max_chars: int) -> Optional[str]:
    if value is None:
        return None
    if not isinstance(value, str):
        value = str(value)
    cleaned = value.strip()
    if not cleaned:
        return None
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[:max_chars] + "...[truncated]"


def _truncate_payload(value: Any, *, max_chars: int) -> Any:
    if value is None:
        return None
    serialized = safe_dumps(value)
    if len(serialized) <= max_chars:
        return value
    return {
        "truncated": True,
        "preview": serialized[:max_chars] + "...[truncated]",
    }


def extract_malformed_tool_call_evidence(
    response_body: dict[str, Any],
    *,
    max_items: Optional[int] = None,
) -> List[Dict[str, Any]]:
    output = response_body.get("output")
    if not isinstance(output, list):
        return []

    evidence: List[Dict[str, Any]] = []
    for item in output:
        if not isinstance(item, dict):
            continue
        if item.get("type") == "message":
            content = item.get("content")
            if not isinstance(content, list):
                continue
            for part in content:
                if not isinstance(part, dict):
                    continue
                if part.get("type") not in {"text", "output_text"}:
                    continue
                text = part.get("text") or ""
                if is_malformed_composer_call_literal_text(text):
                    evidence.append(
                        {
                            "detection_kind": "literal_text",
                            "malformed_tool_call_text": _truncate_text(
                                text,
                                max_chars=_DEFAULT_MAX_TEXT_CHARS,
                            ),
                        }
                    )
            continue

        if item.get("type") in {"function_call", "mcp_call"}:
            name = item.get("name")
            if isinstance(name, str) and name.strip().lower() == "composer_call":
                evidence.append(
                    {
                        "detection_kind": "invalid_tool_name",
                        "malformed_tool_call_text": _truncate_text(
                            name,
                            max_chars=_DEFAULT_MAX_TEXT_CHARS,
                        ),
                        "malformed_tool_call_payload": _truncate_payload(
                            {
                                "type": item.get("type"),
                                "name": item.get("name"),
                                "call_id": item.get("call_id"),
                                "id": item.get("id"),
                                "arguments": item.get("arguments"),
                            },
                            max_chars=_DEFAULT_MAX_PAYLOAD_CHARS,
                        ),
                    }
                )
        if max_items is not None and len(evidence) >= max_items:
            break
    if max_items is not None:
        return evidence[:max_items]
    return evidence


def _coerce_optional_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    if not isinstance(value, str):
        value = str(value)
    cleaned = value.strip()
    return cleaned or None


def _resolve_provider_from_adapter(adapter: str) -> Optional[str]:
    normalized = (adapter or "").strip().lower()
    if not normalized:
        return None
    for token in (
        "openrouter",
        "grok",
        "xai",
        "google",
        "anthropic",
        "openai",
        "opencode",
    ):
        if token in normalized:
            return token
    return None


def build_malformed_tool_call_intake_record(
    *,
    response_body: dict[str, Any],
    adapter_model: str,
    adapter: str,
    adapter_label: str,
    intake_context: Optional[Dict[str, Any]] = None,
    stream_event_summaries: Optional[list[dict[str, Any]]] = None,
    evidence_item: Optional[Dict[str, Any]] = None,
    evidence_index: Optional[int] = None,
    evidence_count: Optional[int] = None,
) -> Dict[str, Any]:
    context = dict(intake_context or {})
    evidence = (
        [evidence_item]
        if isinstance(evidence_item, dict)
        else extract_malformed_tool_call_evidence(response_body)
    )
    primary = evidence[0] if evidence else {}

    response_model = _coerce_optional_str(response_body.get("model"))
    provider = (
        _coerce_optional_str(context.get("provider"))
        or _resolve_provider_from_adapter(adapter)
    )
    route_family = _coerce_optional_str(context.get("route_family")) or adapter

    record: Dict[str, Any] = {
        "schema_version": MALFORMED_TOOL_CALL_SCHEMA_VERSION,
        "observed_at": datetime.now(tz=UTC).isoformat(),
        "environment": _get_aawm_error_log_environment(),
        "failure_kind": MALFORMED_TOOL_CALL_FAILURE_KIND,
        "error_code": MALFORMED_TOOL_CALL_ERROR_CODE,
        "provider": provider,
        "model": response_model or _coerce_optional_str(adapter_model),
        "model_alias": _coerce_optional_str(context.get("model_alias")),
        "route_family": route_family,
        "endpoint": _coerce_optional_str(context.get("endpoint")),
        "upstream_url": _coerce_optional_str(context.get("upstream_url")),
        "repository": _coerce_optional_str(context.get("repository")),
        "agent_name": _coerce_optional_str(context.get("agent_name")),
        "agent_id": _coerce_optional_str(context.get("agent_id")),
        "session_id": _coerce_optional_str(context.get("session_id")),
        "trace_id": _coerce_optional_str(context.get("trace_id")),
        "litellm_call_id": _coerce_optional_str(context.get("litellm_call_id")),
        "request_started_at": _coerce_optional_str(
            context.get("request_started_at")
        ),
        "adapter": adapter,
        "adapter_label": adapter_label,
        "malformed_tool_call_text": primary.get("malformed_tool_call_text"),
        "malformed_tool_call_payload": primary.get("malformed_tool_call_payload"),
        "malformed_tool_call_evidence": evidence or None,
    }
    if evidence_index is not None:
        record["malformed_tool_call_index"] = evidence_index
    if evidence_count is not None:
        record["malformed_tool_call_count"] = evidence_count
    if stream_event_summaries is not None:
        record["stream_event_summaries"] = _truncate_payload(
            stream_event_summaries,
            max_chars=_DEFAULT_MAX_PAYLOAD_CHARS,
        )
    return record


def append_malformed_tool_call_detection(record: Dict[str, Any]) -> bool:
    """Best-effort append of one malformed-tool-call JSON object. Never raises."""
    log_path = _get_malformed_error_log_path()
    if not log_path:
        return False

    try:
        with _MALFORMED_ERROR_LOG_LOCK:
            with _AAWM_ERROR_LOG_LOCK:
                os.makedirs(os.path.dirname(log_path), exist_ok=True)
                max_file_bytes = _max_malformed_error_log_file_bytes()
                if os.path.exists(log_path):
                    if (
                        max_file_bytes is not None
                        and os.path.getsize(log_path) >= max_file_bytes
                    ):
                        return False
                with open(log_path, "a", encoding="utf-8") as intake_file:
                    intake_file.write(safe_dumps(record))
                    intake_file.write("\n")
                _normalize_aawm_error_log_file_metadata(log_path)
        return True
    except Exception:
        return False


def persist_malformed_tool_call_detection(
    *,
    response_body: dict[str, Any],
    adapter_model: str,
    adapter: str,
    adapter_label: str,
    intake_context: Optional[Dict[str, Any]] = None,
    stream_event_summaries: Optional[list[dict[str, Any]]] = None,
) -> bool:
    evidence = extract_malformed_tool_call_evidence(response_body)
    if not evidence:
        record = build_malformed_tool_call_intake_record(
            response_body=response_body,
            adapter_model=adapter_model,
            adapter=adapter,
            adapter_label=adapter_label,
            intake_context=intake_context,
            stream_event_summaries=stream_event_summaries,
        )
        return append_malformed_tool_call_detection(record)

    wrote_any = False
    for index, item in enumerate(evidence):
        record = build_malformed_tool_call_intake_record(
            response_body=response_body,
            adapter_model=adapter_model,
            adapter=adapter,
            adapter_label=adapter_label,
            intake_context=intake_context,
            stream_event_summaries=stream_event_summaries,
            evidence_item=item,
            evidence_index=index,
            evidence_count=len(evidence),
        )
        wrote_any = append_malformed_tool_call_detection(record) or wrote_any
    return wrote_any
