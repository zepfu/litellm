"""Append-only local intake for malformed tools and terminal agent errors."""

from __future__ import annotations

import asyncio
import hashlib
import os
import threading
from datetime import UTC, datetime
from typing import Any, Dict, List, Mapping, Optional

from litellm._logging import (
    _AAWM_ERROR_LOG_LOCK,
    _get_aawm_error_log_dir,
    _get_aawm_error_log_environment,
    _normalize_aawm_error_log_file_metadata,
    _parse_aawm_error_log_non_negative_int_env,
    _redact_string,
)
from litellm.integrations.aawm_agent_quality_rules import (
    is_malformed_composer_call_literal_text,
    is_malformed_grok_literal_tool_label_transcript_text,
)
from litellm.litellm_core_utils.safe_json_dumps import safe_dumps

MALFORMED_ERROR_JSONL_FILENAME = "malformed-error.jsonl"
MALFORMED_TOOL_CALL_SCHEMA_VERSION = 1
MALFORMED_TOOL_CALL_ERROR_CODE = "aawm_auto_agent_malformed_tool_call_text"
MALFORMED_TOOL_CALL_FAILURE_KIND = "malformed_tool_call"
AGENT_TERMINAL_ERROR_SCHEMA_VERSION = 1

_DEFAULT_MAX_TEXT_CHARS = 8_192
_DEFAULT_MAX_PAYLOAD_CHARS = 16_384
# Bound per-response malformed evidence writes so adversarial/malformed
# upstream payloads cannot amplify into unbounded synchronous disk I/O.
_DEFAULT_MAX_MALFORMED_EVIDENCE_ITEMS = 8
# Hard upper clamp for env-configured evidence limits (RR-044).
# Operators may lower the default, but cannot raise past this ceiling.
_HARD_MAX_MALFORMED_EVIDENCE_ITEMS = 64
# Conservative non-None default size cap for opt-in JSONL intake sinks.
_DEFAULT_MAX_ERROR_LOG_FILE_BYTES = 10 * 1024 * 1024
_MALFORMED_ERROR_LOG_LOCK = threading.Lock()
_AGENT_TERMINAL_ERROR_LOG_LOCK = threading.Lock()

_AGENT_TERMINAL_CONTEXT_FIELDS = (
    "source",
    "container",
    "endpoint",
    "incoming_endpoint",
    "upstream_url",
    "outgoing_target",
    "provider",
    "model",
    "model_alias",
    "alias_model",
    "alias_family",
    "route_family",
    "status_code",
    "error_status_code",
    "failure_kind",
    "failure_class",
    "error_code",
    "event_type",
    "candidate_status",
    "failure_phase",
    "attempted_provider_call",
    "repository",
    "tenant_id",
    "agent_name",
    "agent_id",
    "agent_role",
    "agent_profile",
    "thread_source",
    "session_id",
    "thread_id",
    "trace_id",
    "litellm_call_id",
    "dispatch_id",
    "redispatch_ordinal",
    "cooldown_scope",
    "cooldown_state_source",
    "terminal_activity_status",
    "actual_prior_tool_activity_summary",
    "attempt_count",
    "attempts",
    "candidate_count",
    "candidates",
    "hidden_retry_final_outcome",
    "hidden_retry_failure_classification",
    "hidden_retry_count",
    "aawm_passthrough_request_shape_summary",
    "aawm_passthrough_request_shape_fingerprint",
    "aawm_passthrough_request_shape_error_class",
    "aawm_passthrough_request_shape_error_message_class",
    "aawm_passthrough_request_shape_error_body_preview",
    "aawm_passthrough_request_shape_error_fingerprint",
)
_AGENT_TERMINAL_ROUTING_SEQUENCE_FIELDS = (
    "attempt_number",
    "provider",
    "model",
    "route_family",
    "lane_key",
    "reason",
    "selection_reason",
    "status",
    "candidate_status",
    "event_type",
    "failure_class",
    "error_class",
    "error_status_code",
    "error_type",
    "error_code",
    "error_tokens",
    "retry_after_seconds",
    "failure_phase",
    "attempted_provider_call",
    "cooldown_scope",
    "cooldown_seconds",
    "cooldown_state_source",
    "last_resort",
    "in_flight_session",
    "redispatch_required",
    "redispatch_threshold_crossed",
)
_AGENT_TERMINAL_ACTIVITY_SUMMARY_FIELDS = (
    "has_actual_prior_tool_activity",
    "prior_tool_call_count",
    "prior_tool_result_count",
    "prior_tool_names",
    "has_prior_file_edit_activity",
    "prior_file_edit_tool_call_count",
    "prior_file_edit_tool_names",
    "has_previous_response_id",
    "has_continuation_state",
)
_AGENT_TERMINAL_REQUEST_SHAPE_SUMMARY_FIELDS = (
    "body_container_type",
    "body_top_level_keys",
    "input_container_type",
    "input_item_count",
    "input_item_type_counts",
    "input_item_shape_samples",
    "tool_count",
    "tool_type_counts",
)
_AGENT_TERMINAL_REQUEST_SHAPE_SAMPLE_FIELDS = (
    "index",
    "container_type",
    "type",
    "keys",
)
_AGENT_TERMINAL_LIST_FIELDS = {
    "body_top_level_keys",
    "error_tokens",
    "keys",
    "prior_file_edit_tool_names",
    "prior_tool_names",
}
_AGENT_TERMINAL_COUNT_MAPPING_FIELDS = {
    "input_item_type_counts",
    "tool_type_counts",
}


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


def _max_malformed_evidence_items() -> int:
    """Return the per-response evidence item cap with a hard upper clamp.

    ``LITELLM_AAWM_MALFORMED_ERROR_LOG_MAX_ITEMS`` may lower the default, but
    values above ``_HARD_MAX_MALFORMED_EVIDENCE_ITEMS`` are clamped so a
    misconfigured env cannot reintroduce unbounded intake amplification.
    """
    configured = _parse_aawm_error_log_non_negative_int_env(
        "LITELLM_AAWM_MALFORMED_ERROR_LOG_MAX_ITEMS"
    )
    if configured is None or configured <= 0:
        return _DEFAULT_MAX_MALFORMED_EVIDENCE_ITEMS
    return min(configured, _HARD_MAX_MALFORMED_EVIDENCE_ITEMS)


def _max_malformed_error_log_file_bytes() -> int:
    configured = _parse_aawm_error_log_non_negative_int_env(
        "LITELLM_AAWM_MALFORMED_ERROR_LOG_MAX_BYTES"
    )
    if configured is None or configured <= 0:
        return _DEFAULT_MAX_ERROR_LOG_FILE_BYTES
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


def _malformed_text_evidence(text: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(text, str):
        return None
    if is_malformed_composer_call_literal_text(text):
        detection_kind = "literal_text"
    elif is_malformed_grok_literal_tool_label_transcript_text(text):
        detection_kind = "grok_literal_tool_label_transcript"
    else:
        return None
    return {
        "detection_kind": detection_kind,
        "malformed_tool_call_text": _truncate_text(
            text,
            max_chars=_DEFAULT_MAX_TEXT_CHARS,
        ),
    }


def extract_malformed_tool_call_evidence(
    response_body: dict[str, Any],
    *,
    max_items: Optional[int] = None,
) -> List[Dict[str, Any]]:
    output = response_body.get("output")
    if not isinstance(output, list):
        return []

    # Always enforce the hard upper clamp, even for direct callers that pass an
    # explicit max_items (or None). None defaults to the env-resolved cap.
    if max_items is None:
        max_items = _max_malformed_evidence_items()
    else:
        try:
            max_items = int(max_items)
        except (TypeError, ValueError):
            max_items = _max_malformed_evidence_items()
        if max_items < 0:
            max_items = 0
        max_items = min(max_items, _HARD_MAX_MALFORMED_EVIDENCE_ITEMS)

    evidence: List[Dict[str, Any]] = []
    for item in output:
        if not isinstance(item, dict):
            continue
        if item.get("type") == "message":
            content = item.get("content")
            if isinstance(content, str):
                text_evidence = _malformed_text_evidence(content)
                if text_evidence is not None:
                    evidence.append(text_evidence)
                if max_items is not None and len(evidence) >= max_items:
                    break
                continue
            if not isinstance(content, list):
                continue
            for part in content:
                if not isinstance(part, dict):
                    continue
                if part.get("type") not in {"text", "output_text"}:
                    continue
                text_evidence = _malformed_text_evidence(part.get("text"))
                if text_evidence is not None:
                    evidence.append(text_evidence)
                if max_items is not None and len(evidence) >= max_items:
                    break
            if max_items is not None and len(evidence) >= max_items:
                break
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
        [evidence_item] if isinstance(evidence_item, dict) else extract_malformed_tool_call_evidence(response_body)
    )
    primary = evidence[0] if evidence else {}

    response_model = _coerce_optional_str(response_body.get("model"))
    provider = _coerce_optional_str(context.get("provider")) or _resolve_provider_from_adapter(adapter)
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
        "agent_role": _coerce_optional_str(context.get("agent_role")),
        "agent_profile": _coerce_optional_str(context.get("agent_profile")),
        "thread_source": _coerce_optional_str(context.get("thread_source")),
        "session_id": _coerce_optional_str(context.get("session_id")),
        "thread_id": _coerce_optional_str(context.get("thread_id")),
        "trace_id": _coerce_optional_str(context.get("trace_id")),
        "litellm_call_id": _coerce_optional_str(context.get("litellm_call_id")),
        "dispatch_id": _coerce_optional_str(context.get("dispatch_id")),
        "redispatch_ordinal": context.get("redispatch_ordinal"),
        "terminal_outcome": _coerce_optional_str(context.get("terminal_outcome")) or "malformed_tool_call_rejected",
        "fallback_result": _coerce_optional_str(context.get("fallback_result")) or "none",
        "redispatch_required": bool(context.get("redispatch_required", False)),
        "agent_session_killed": bool(context.get("agent_session_killed", True)),
        "request_started_at": _coerce_optional_str(context.get("request_started_at")),
        "adapter": adapter,
        "adapter_label": adapter_label,
        "malformed_tool_call_text": primary.get("malformed_tool_call_text"),
        "malformed_tool_call_payload": primary.get("malformed_tool_call_payload"),
        "malformed_tool_call_evidence": evidence or None,
    }
    fingerprint = _build_agent_terminal_failure_fingerprint(record)
    record["fingerprint"] = fingerprint
    record["failure_fingerprint"] = fingerprint
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
    return append_malformed_tool_call_detections([record])


def _encode_jsonl_record_line(record: Dict[str, Any]) -> str:
    return safe_dumps(record) + "\n"


def _jsonl_record_line_bytes(record: Dict[str, Any]) -> int:
    return len(_encode_jsonl_record_line(record).encode("utf-8"))


def _projected_jsonl_batch_bytes(records: List[Dict[str, Any]]) -> int:
    return sum(_jsonl_record_line_bytes(record) for record in records)


def append_malformed_tool_call_detections(records: List[Dict[str, Any]]) -> bool:
    """Best-effort append of one or more malformed-tool-call JSON objects.

    Writes all records under a single lock acquisition and file open so a
    multi-evidence response cannot amplify into N independent open/stat cycles.
    Enforces a strict max-bytes ceiling against current file size plus the
    projected encoded size of the entire pending batch.
    Never raises.
    """
    if not records:
        return False
    log_path = _get_malformed_error_log_path()
    if not log_path:
        return False

    try:
        encoded_lines = [_encode_jsonl_record_line(record) for record in records]
        pending_bytes = sum(len(line.encode("utf-8")) for line in encoded_lines)
        with _MALFORMED_ERROR_LOG_LOCK:
            with _AAWM_ERROR_LOG_LOCK:
                os.makedirs(os.path.dirname(log_path), exist_ok=True)
                max_file_bytes = _max_malformed_error_log_file_bytes()
                current_size = (
                    os.path.getsize(log_path) if os.path.exists(log_path) else 0
                )
                # Strict ceiling: refuse the whole batch if it would cross max.
                if current_size + pending_bytes > max_file_bytes:
                    return False
                with open(log_path, "a", encoding="utf-8") as intake_file:
                    for line in encoded_lines:
                        intake_file.write(line)
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
    max_items = _max_malformed_evidence_items()
    evidence = extract_malformed_tool_call_evidence(
        response_body,
        max_items=max_items,
    )
    if not evidence:
        record = build_malformed_tool_call_intake_record(
            response_body=response_body,
            adapter_model=adapter_model,
            adapter=adapter,
            adapter_label=adapter_label,
            intake_context=intake_context,
            stream_event_summaries=stream_event_summaries,
        )
        return append_malformed_tool_call_detections([record])

    records: List[Dict[str, Any]] = []
    evidence_count = len(evidence)
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
            evidence_count=evidence_count,
        )
        records.append(record)
    return append_malformed_tool_call_detections(records)


def schedule_persist_malformed_tool_call_detection(
    *,
    response_body: dict[str, Any],
    adapter_model: str,
    adapter: str,
    adapter_label: str,
    intake_context: Optional[Dict[str, Any]] = None,
    stream_event_summaries: Optional[list[dict[str, Any]]] = None,
) -> None:
    """Offload malformed-tool-call intake writes off the async request path.

    When an event loop is running, schedules ``asyncio.to_thread`` so async
    request handlers do not block on synchronous disk I/O. Without a running
    loop (sync callers / unit tests), persists inline so behavior stays
    deterministic. Best-effort: never raises.
    """

    def _run() -> None:
        try:
            persist_malformed_tool_call_detection(
                response_body=response_body,
                adapter_model=adapter_model,
                adapter=adapter,
                adapter_label=adapter_label,
                intake_context=intake_context,
                stream_event_summaries=stream_event_summaries,
            )
        except Exception:
            # Best-effort intake must never surface to request handlers.
            return

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        _run()
        return

    async def _offload() -> None:
        try:
            await asyncio.to_thread(
                persist_malformed_tool_call_detection,
                response_body=response_body,
                adapter_model=adapter_model,
                adapter=adapter,
                adapter_label=adapter_label,
                intake_context=intake_context,
                stream_event_summaries=stream_event_summaries,
            )
        except Exception:
            return

    try:
        loop.create_task(_offload())
    except Exception:
        _run()


def _agent_terminal_error_log_enabled() -> bool:
    raw = os.getenv("LITELLM_AAWM_AGENT_TERMINAL_ERROR_LOG_ENABLED", "").strip()
    if raw:
        return raw.lower() in {"1", "true", "yes", "on"}
    if os.getenv("LITELLM_AAWM_ERROR_LOG_DIR", "").strip():
        return True
    return os.getenv("LITELLM_AAWM_ERROR_LOG_ENABLED", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _build_agent_terminal_failure_fingerprint(
    context: Mapping[str, Any],
) -> str:
    values = (
        context.get("failure_kind"),
        context.get("failure_class"),
        context.get("error_code"),
        context.get("status_code") or context.get("error_status_code"),
        context.get("provider"),
        context.get("model_alias") or context.get("alias_model"),
        context.get("route_family"),
        context.get("endpoint") or context.get("incoming_endpoint"),
        context.get("aawm_passthrough_request_shape_error_fingerprint"),
    )
    normalized = "|".join(str(value or "").strip().lower() for value in values)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _get_agent_terminal_error_log_path() -> Optional[str]:
    if not _agent_terminal_error_log_enabled():
        return None
    log_dir = _get_aawm_error_log_dir()
    if not log_dir:
        log_dir = os.path.join(os.getcwd(), ".analysis")
    if not log_dir:
        return None
    return os.path.join(
        log_dir,
        f"{_get_aawm_error_log_environment()}-error.jsonl",
    )


def _max_agent_terminal_error_log_file_bytes() -> int:
    configured = _parse_aawm_error_log_non_negative_int_env("LITELLM_AAWM_ERROR_LOG_MAX_BYTES")
    if configured is None or configured <= 0:
        return _DEFAULT_MAX_ERROR_LOG_FILE_BYTES
    return configured


def _sanitize_agent_terminal_context_value(
    *,
    field: str,
    value: Any,
) -> Any:
    if field in {"attempts", "candidates"}:
        return _sanitize_agent_terminal_routing_sequence(value)
    if field == "actual_prior_tool_activity_summary":
        return _sanitize_agent_terminal_mapping(
            value,
            allowed_fields=_AGENT_TERMINAL_ACTIVITY_SUMMARY_FIELDS,
        )
    if field == "aawm_passthrough_request_shape_summary":
        return _sanitize_agent_terminal_request_shape_summary(value)
    return _sanitize_agent_terminal_scalar(value)


def _sanitize_agent_terminal_count_mapping(
    value: Any,
) -> Optional[Dict[str, Any]]:
    if not isinstance(value, Mapping):
        return None
    sanitized: Dict[str, Any] = {}
    for raw_key, raw_value in list(value.items())[:50]:
        key = _sanitize_agent_terminal_scalar(raw_key)
        count = _sanitize_agent_terminal_scalar(raw_value)
        if isinstance(key, str) and isinstance(count, (int, float)):
            sanitized[key] = count
    return sanitized or None


def _sanitize_agent_terminal_mapping(
    value: Any,
    *,
    allowed_fields: tuple[str, ...],
) -> Optional[Dict[str, Any]]:
    if not isinstance(value, Mapping):
        return None
    sanitized: Dict[str, Any] = {}
    for field in allowed_fields:
        raw_value = value.get(field)
        if raw_value is None:
            continue
        cleaned: Any
        if field in _AGENT_TERMINAL_LIST_FIELDS:
            cleaned = _sanitize_agent_terminal_scalar_list(raw_value)
        elif field in _AGENT_TERMINAL_COUNT_MAPPING_FIELDS:
            cleaned = _sanitize_agent_terminal_count_mapping(raw_value)
        else:
            cleaned = _sanitize_agent_terminal_scalar(raw_value)
        if cleaned is not None:
            sanitized[field] = cleaned
    return sanitized or None


def _sanitize_agent_terminal_request_shape_summary(
    value: Any,
) -> Optional[Dict[str, Any]]:
    if not isinstance(value, Mapping):
        return None
    sanitized = (
        _sanitize_agent_terminal_mapping(
            value,
            allowed_fields=_AGENT_TERMINAL_REQUEST_SHAPE_SUMMARY_FIELDS,
        )
        or {}
    )
    samples = value.get("input_item_shape_samples")
    if isinstance(samples, list):
        sanitized_samples = [
            cleaned
            for item in samples[:20]
            if (
                cleaned := _sanitize_agent_terminal_mapping(
                    item,
                    allowed_fields=_AGENT_TERMINAL_REQUEST_SHAPE_SAMPLE_FIELDS,
                )
            )
            is not None
        ]
        if sanitized_samples:
            sanitized["input_item_shape_samples"] = sanitized_samples
    return sanitized or None


def _sanitize_agent_terminal_routing_sequence(
    value: Any,
) -> Optional[List[Dict[str, Any]]]:
    if not isinstance(value, list):
        return None
    sanitized = [
        cleaned
        for item in value[:50]
        if (
            cleaned := _sanitize_agent_terminal_mapping(
                item,
                allowed_fields=_AGENT_TERMINAL_ROUTING_SEQUENCE_FIELDS,
            )
        )
        is not None
    ]
    return sanitized or None


def _sanitize_agent_terminal_scalar(value: Any) -> Any:
    if isinstance(value, str):
        return _truncate_text(
            _redact_string(value),
            max_chars=2_048,
        )
    if isinstance(value, (bool, int, float)):
        return value
    return None


def _sanitize_agent_terminal_scalar_list(
    value: Any,
    *,
    max_items: int = 50,
) -> Optional[List[Any]]:
    if not isinstance(value, (list, tuple, set)):
        return None
    sanitized = [
        cleaned for item in list(value)[:max_items] if (cleaned := _sanitize_agent_terminal_scalar(item)) is not None
    ]
    return sanitized or None


def append_agent_terminal_error(record: Dict[str, Any]) -> bool:
    """Best-effort append of one terminal agent error JSON object."""
    log_path = _get_agent_terminal_error_log_path()
    if not log_path:
        return False

    try:
        with _AGENT_TERMINAL_ERROR_LOG_LOCK:
            with _AAWM_ERROR_LOG_LOCK:
                os.makedirs(os.path.dirname(log_path), exist_ok=True)
                max_file_bytes = _max_agent_terminal_error_log_file_bytes()
                encoded_line = _encode_jsonl_record_line(record)
                pending_bytes = len(encoded_line.encode("utf-8"))
                current_size = (
                    os.path.getsize(log_path) if os.path.exists(log_path) else 0
                )
                if current_size + pending_bytes > max_file_bytes:
                    return False
                with open(log_path, "a", encoding="utf-8") as intake_file:
                    intake_file.write(encoded_line)
                _normalize_aawm_error_log_file_metadata(log_path)
        return True
    except Exception:
        return False


def build_agent_terminal_error_record(
    *,
    error_context: Mapping[str, Any],
    terminal_outcome: Optional[str] = None,
    fallback_result: Optional[str] = None,
    redispatch_required: Optional[bool] = None,
    agent_session_killed: bool,
) -> Dict[str, Any]:
    context: Dict[str, Any] = {}
    for field in _AGENT_TERMINAL_CONTEXT_FIELDS:
        value = error_context.get(field)
        if value is None:
            continue
        sanitized = _sanitize_agent_terminal_context_value(
            field=field,
            value=value,
        )
        if sanitized is not None:
            context[field] = sanitized

    context["terminal_outcome"] = _coerce_optional_str(terminal_outcome)
    context["fallback_result"] = _coerce_optional_str(fallback_result)
    context["redispatch_required"] = redispatch_required
    context["agent_session_killed"] = bool(agent_session_killed)
    context = {key: value for key, value in context.items() if value is not None}

    failure_kind = _coerce_optional_str(context.get("failure_kind")) or "agent_terminal_error"
    error_code = (
        _coerce_optional_str(context.get("error_code"))
        or _coerce_optional_str(context.get("failure_class"))
        or failure_kind
    )
    fingerprint = _build_agent_terminal_failure_fingerprint(context)
    return {
        "schema_version": AGENT_TERMINAL_ERROR_SCHEMA_VERSION,
        "observed_at": datetime.now(tz=UTC).isoformat(),
        "environment": _get_aawm_error_log_environment(),
        "logger": "litellm.proxy.agent_terminal",
        "level": "ERROR",
        "message": f"Agent terminal error: {failure_kind}",
        "traceback": None,
        "traceback_text": None,
        "traceback_lines": [],
        "raw_text": None,
        "fingerprint": fingerprint,
        "failure_kind": failure_kind,
        "error_code": error_code,
        "status_code": context.get("status_code") or context.get("error_status_code"),
        "agent_id": context.get("agent_id"),
        "session_id": context.get("session_id"),
        "litellm_call_id": context.get("litellm_call_id"),
        "terminal_outcome": context.get("terminal_outcome"),
        "fallback_result": context.get("fallback_result"),
        "redispatch_required": context.get("redispatch_required"),
        "agent_session_killed": context["agent_session_killed"],
        "context": context,
    }


def persist_agent_terminal_error(
    *,
    error_context: Mapping[str, Any],
    terminal_outcome: Optional[str] = None,
    fallback_result: Optional[str] = None,
    redispatch_required: Optional[bool] = None,
    agent_session_killed: bool,
) -> bool:
    record = build_agent_terminal_error_record(
        error_context=error_context,
        terminal_outcome=terminal_outcome,
        fallback_result=fallback_result,
        redispatch_required=redispatch_required,
        agent_session_killed=agent_session_killed,
    )
    return append_agent_terminal_error(record)
