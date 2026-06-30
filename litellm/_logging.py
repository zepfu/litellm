import ast
from collections import deque
import hashlib
import json
import logging
import os
import re
import sys
import threading
import time
from datetime import UTC, datetime
from logging import Formatter
from typing import Any, Deque, Dict, List, Optional, Set, Tuple
from urllib.parse import unquote

from litellm.litellm_core_utils.safe_json_dumps import safe_dumps
from litellm.litellm_core_utils.safe_json_loads import safe_json_loads

set_verbose = False

if set_verbose is True:
    logging.warning(
        "`litellm.set_verbose` is deprecated. Please set `os.environ['LITELLM_LOG'] = 'DEBUG'` for debug logs."
    )

_ENABLE_SECRET_REDACTION = os.getenv("LITELLM_DISABLE_REDACT_SECRETS", "").lower() != "true"

_REDACTED = "REDACTED"


def _build_secret_patterns() -> re.Pattern:
    patterns: List[str] = [
        # AWS access key IDs
        r"(?:AKIA|ASIA)[0-9A-Z]{16}",
        # AWS secrets / session tokens / access key IDs (key=value)
        r"(?:aws_secret_access_key|aws_session_token|aws_access_key_id)"
        r"\s*[:=]\s*[A-Za-z0-9/+=]{20,}",
        # Bearer tokens (OAuth, JWT, etc.)
        r"Bearer\s+[A-Za-z0-9\-._~+/]{10,}=*",
        # Basic auth headers
        r"Basic\s+[A-Za-z0-9+/]{10,}={0,2}",
        # OpenAI / Anthropic sk- prefixed keys
        r"sk-[A-Za-z0-9\-_]{20,}",
        # Generic api_key / api-key / apikey (handles 'key': 'value' dict repr)
        r"(?:api[_-]?key)['\"]?\s*[:=]\s*['\"]?[^\s,'\"})\]{}>]{8,}",
        # x-api-key / api-key header values (handles 'key': 'value' dict repr)
        r"(?:x-api-key|api-key)['\"]?\s*[:=]\s*['\"]?[^\s,'\"})\]{}>]+",
        # Anthropic internal header keys
        r"x-ak-[A-Za-z0-9\-_]{20,}",
        # Google API keys
        r"AIza[0-9A-Za-z\-_]{35}",
        # Password / secret params (handles key=value and 'key': 'value')
        r"\w*(?:password|passwd|client_secret|secret_key|_secret)"
        r"['\"]?\s*[:=]\s*['\"]?[^\s,'\"})\]{}>]+",
        # Database connection string credentials (scheme://user:pass@host)
        r"(?<=://)[^\s'\"]*:[^\s'\"@]+(?=@)",
        # Databricks personal access tokens
        r"dapi[0-9a-f]{32}",
    ]
    return re.compile("|".join(patterns), re.IGNORECASE)


_SECRET_RE = _build_secret_patterns()


def _redact_string(value: str) -> str:
    return _SECRET_RE.sub(_REDACTED, value)


class SecretRedactionFilter(logging.Filter):
    """Scrubs known secret/credential patterns from log records."""

    _formatter = logging.Formatter()

    def filter(self, record: logging.LogRecord) -> bool:
        if not _ENABLE_SECRET_REDACTION:
            return True

        try:
            record.msg = _redact_string(record.getMessage())
            record.args = None
        except Exception:
            if isinstance(record.msg, str):
                record.msg = _redact_string(record.msg)

        # Redact exception tracebacks
        if record.exc_info and record.exc_info[1] is not None:
            try:
                record.exc_text = _redact_string(
                    self._formatter.formatException(record.exc_info)
                )
            except Exception:
                pass

        # Redact extra fields passed via logger.debug("msg", extra={...})
        for key, value in list(record.__dict__.items()):
            if key not in _STANDARD_RECORD_ATTRS and isinstance(value, str):
                setattr(record, key, _redact_string(value))

        return True


_secret_filter = SecretRedactionFilter()

_AAWM_ERROR_LOG_HANDLER_NAME = "aawm_error_log_file_handler"
_AAWM_ERROR_LOG_LOCK = threading.Lock()


_LANGFUSE_SUPPORT_STRING_COALESCE_DEFAULT_TTL_SECONDS = 300
_LANGFUSE_SUPPORT_STRING_COALESCE_LOCK = threading.Lock()
_langfuse_support_string_coalesce_state: Dict[str, float] = {}


def _get_langfuse_support_string_coalesce_ttl_seconds() -> int:
    configured = _parse_aawm_error_log_non_negative_int_env(
        "LITELLM_AAWM_ERROR_LOG_LANGFUSE_SUPPORT_STRING_COALESCE_TTL_SECONDS"
    )
    if configured is None:
        return _LANGFUSE_SUPPORT_STRING_COALESCE_DEFAULT_TTL_SECONDS
    return configured


def clear_langfuse_support_string_coalesce_state() -> None:
    with _LANGFUSE_SUPPORT_STRING_COALESCE_LOCK:
        _langfuse_support_string_coalesce_state.clear()


def _prune_langfuse_support_string_coalesce_state(*, now: float, ttl_seconds: int) -> None:
    expired_keys = [
        key
        for key, first_seen_at in _langfuse_support_string_coalesce_state.items()
        if (now - first_seen_at) >= ttl_seconds
    ]
    for key in expired_keys:
        _langfuse_support_string_coalesce_state.pop(key, None)


def _build_langfuse_support_string_coalesce_key(payload: Dict[str, Any]) -> Optional[str]:
    context = payload.get("context")
    if not isinstance(context, dict) or not context.get("langfuse_support_string"):
        return None

    key_source = {
        "fingerprint": payload.get("fingerprint"),
        "trace_id": context.get("trace_id"),
        "langfuse_generation_id": context.get("langfuse_generation_id"),
        "langfuse_generation_name": context.get("langfuse_generation_name"),
        "langfuse_call_type": context.get("langfuse_call_type"),
        "langfuse_payload_size_state": context.get("langfuse_payload_size_state"),
        "langfuse_failure_class": context.get("langfuse_failure_class"),
        "langfuse_event_fit_failed": context.get("langfuse_event_fit_failed"),
    }
    return hashlib.sha256(
        json.dumps(
            key_source,
            default=str,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


def _should_suppress_langfuse_support_string_coalesce(payload: Dict[str, Any]) -> bool:
    coalesce_key = _build_langfuse_support_string_coalesce_key(payload)
    if coalesce_key is None:
        return False

    ttl_seconds = _get_langfuse_support_string_coalesce_ttl_seconds()
    now = time.time()
    with _LANGFUSE_SUPPORT_STRING_COALESCE_LOCK:
        _prune_langfuse_support_string_coalesce_state(
            now=now,
            ttl_seconds=ttl_seconds,
        )
        first_seen_at = _langfuse_support_string_coalesce_state.get(coalesce_key)
        if first_seen_at is not None and (now - first_seen_at) < ttl_seconds:
            return True
        _langfuse_support_string_coalesce_state[coalesce_key] = now
        return False


def _env_truthy(value: Optional[str]) -> bool:
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _sanitize_aawm_error_log_env_name(value: Optional[str]) -> str:
    cleaned = (value or "").strip().lower()
    sanitized = re.sub(r"[^a-z0-9_.-]+", "-", cleaned).strip(".-")
    return sanitized[:64] or "unknown"


def _get_aawm_error_log_environment() -> str:
    return _sanitize_aawm_error_log_env_name(
        os.getenv("LITELLM_AAWM_ERROR_LOG_ENV")
        or os.getenv("LITELLM_LANGFUSE_TRACE_ENVIRONMENT")
        or os.getenv("LITELLM_ENV")
        or os.getenv("ENVIRONMENT")
    )


def _get_aawm_error_log_dir() -> Optional[str]:
    configured_dir = os.getenv("LITELLM_AAWM_ERROR_LOG_DIR", "").strip()
    if configured_dir:
        return configured_dir

    if not _env_truthy(os.getenv("LITELLM_AAWM_ERROR_LOG_ENABLED")):
        return None

    return os.path.join(os.getcwd(), ".analysis")


def _get_aawm_error_log_path() -> Optional[str]:
    log_dir = _get_aawm_error_log_dir()
    if not log_dir:
        return None
    return os.path.join(log_dir, f"{_get_aawm_error_log_environment()}-error.jsonl")


def _get_aawm_legacy_error_log_path() -> Optional[str]:
    """Legacy text sink path retained for migration and discovery."""
    log_dir = _get_aawm_error_log_dir()
    if not log_dir:
        return None
    return os.path.join(log_dir, f"{_get_aawm_error_log_environment()}-error.log")


def _parse_aawm_error_log_non_negative_int_env(name: str) -> Optional[int]:
    raw = os.getenv(name, "").strip()
    if not raw:
        return None
    try:
        value = int(raw, 10)
    except ValueError:
        return None
    if value < 0:
        return None
    return value


def _parse_aawm_error_log_file_mode_env() -> Optional[int]:
    raw = os.getenv("LITELLM_AAWM_ERROR_LOG_FILE_MODE", "").strip()
    if not raw:
        return None
    try:
        mode = int(raw, 8)
    except ValueError:
        return None
    if mode < 0 or mode > 0o777:
        return None
    return mode


def _normalize_aawm_error_log_file_metadata(log_path: str) -> None:
    """Best-effort host bind-mount ownership repair for runtime JSONL intake."""
    try:
        parent_dir = os.path.dirname(os.path.abspath(log_path)) or "."
        parent_stat = os.stat(parent_dir)
        target_uid = (
            _parse_aawm_error_log_non_negative_int_env(
                "LITELLM_AAWM_ERROR_LOG_FILE_UID"
            )
            if os.getenv("LITELLM_AAWM_ERROR_LOG_FILE_UID", "").strip()
            else parent_stat.st_uid
        )
        target_gid = (
            _parse_aawm_error_log_non_negative_int_env(
                "LITELLM_AAWM_ERROR_LOG_FILE_GID"
            )
            if os.getenv("LITELLM_AAWM_ERROR_LOG_FILE_GID", "").strip()
            else parent_stat.st_gid
        )
        current_stat = os.stat(log_path)
        if (
            target_uid is not None
            and target_gid is not None
            and (current_stat.st_uid, current_stat.st_gid)
            != (target_uid, target_gid)
            and hasattr(os, "chown")
        ):
            try:
                os.chown(log_path, target_uid, target_gid)
            except OSError:
                pass

        target_mode = _parse_aawm_error_log_file_mode_env()
        if target_mode is not None and (current_stat.st_mode & 0o777) != target_mode:
            try:
                os.chmod(log_path, target_mode)
            except OSError:
                pass
    except Exception:
        # Error intake must never fail the application logging path.
        return


_AAWM_ERROR_LOG_CONTEXT_FIELDS = (
    "source",
    "container",
    "endpoint",
    "upstream_url",
    "provider",
    "model",
    "model_alias",
    "route_family",
    "status_code",
    "auth_header_names",
    "auth_header_sources",
    "auth_credential_source",
    "failure_kind",
    "hidden_retry_final_outcome",
    "hidden_retry_failure_classification",
    "hidden_retry_count",
    "stream_failure_stage",
    "stream_chunks_seen",
    "stream_bytes_seen",
    "stream_hidden_retry_safe",
    "trace_id",
    "litellm_call_id",
    "callback_name",
    "callback_phase",
    "handler_branch",
    "langfuse_failure_class",
    "event_type",
    "worker_timeout_seconds",
    "queue_depth",
    "queue_maxsize",
    "coroutine_name",
    "worker_delivery_state",
    "aawm_passthrough_inbound_content_type",
    "aawm_passthrough_json_egress_content_type_removed",
    "aawm_passthrough_json_egress_content_type_removed_value",
    "aawm_passthrough_body_container_type",
    "aawm_passthrough_body_top_level_keys",
    "aawm_passthrough_input_container_type",
    "aawm_passthrough_input_item_count",
    "aawm_passthrough_input_item_type_counts",
    "aawm_passthrough_tool_count",
    "aawm_passthrough_tool_type_counts",
    "aawm_passthrough_request_shape_error_class",
    "aawm_passthrough_request_shape_error_message_class",
    "aawm_passthrough_request_shape_error_body_preview",
    "aawm_passthrough_request_shape_summary",
    "aawm_passthrough_request_shape_fingerprint",
    "aawm_passthrough_request_shape_error_fingerprint",
    "grok_side_channel",
    "grok_side_channel_endpoint_type",
    "grok_side_channel_endpoint_path_template",
    "grok_side_channel_request_content_type",
    "grok_side_channel_request_body_byte_length",
    "grok_side_channel_request_body_digest_source",
    "grok_side_channel_request_json_container_type",
    "grok_side_channel_request_array_length",
    "recommended_operator_action",
    "langfuse_support_string",
    "langfuse_sdk_background_ingestion_failure",
    "langfuse_payload_size_state",
    "langfuse_total_size_bytes",
    "langfuse_max_event_size_bytes",
    "langfuse_warning_threshold_bytes",
    "langfuse_generation_id",
    "langfuse_generation_name",
    "langfuse_call_type",
    "langfuse_event_fit_failed",
    "langfuse_enqueue_observed_at",
    "aiohttp_owner_kind",
    "aiohttp_creation_site",
    "aiohttp_litellm_owns_session",
    "aiohttp_session_id",
    "aiohttp_connector_id",
    "aiohttp_event_loop_id",
    "aiohttp_pid",
    "aiohttp_container_hostname",
    "aiohttp_context_keys",
    "aiohttp_context_resource",
)


def _build_aawm_error_log_context(record: logging.LogRecord) -> Dict[str, Any]:
    context: Dict[str, Any] = {field: None for field in _AAWM_ERROR_LOG_CONTEXT_FIELDS}
    for field in _AAWM_ERROR_LOG_CONTEXT_FIELDS:
        value = getattr(record, field, None)
        if value is not None:
            context[field] = value
    return context


def _build_aawm_error_log_fingerprint(
    *,
    logger_name: str,
    level: str,
    message: str,
    traceback_text: Optional[str],
    context: Dict[str, Any],
) -> str:
    fingerprint_source = {
        "context": context,
        "logger": logger_name,
        "level": level,
        "message": message,
        "traceback": traceback_text,
        "traceback_text": traceback_text,
    }
    return hashlib.sha256(
        json.dumps(
            fingerprint_source,
            default=str,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


def _build_aawm_error_log_record(
    record: logging.LogRecord,
    *,
    formatter: logging.Formatter,
) -> Dict[str, Any]:
    context = _build_aawm_error_log_context(record)
    message = record.getMessage()
    traceback_text: Optional[str] = None
    traceback_lines: List[str] = []
    if record.exc_info and record.exc_info[1] is not None:
        traceback_text = record.exc_text or formatter.formatException(record.exc_info)
        traceback_lines = traceback_text.splitlines()

    return {
        "schema_version": 1,
        "observed_at": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
        "environment": _get_aawm_error_log_environment(),
        "logger": record.name,
        "level": record.levelname,
        "message": message,
        "traceback": traceback_text,
        "traceback_text": traceback_text,
        "traceback_lines": traceback_lines,
        "raw_text": formatter.format(record),
        "fingerprint": _build_aawm_error_log_fingerprint(
            logger_name=record.name,
            level=record.levelname,
            message=message,
            traceback_text=traceback_text,
            context=context,
        ),
        "context": context,
    }


def _extract_aiohttp_attribution_from_asyncio_context(
    context: Dict[str, Any],
) -> Dict[str, Any]:
    if not isinstance(context, dict):
        return {}

    resource_name = "client_session"
    resource = context.get(resource_name)
    if resource is None:
        resource_name = "connector"
        resource = context.get(resource_name)
    if resource is None:
        return {}

    try:
        from litellm.llms.custom_httpx.aiohttp_transport import (
            get_litellm_aiohttp_session_attribution,
        )
    except Exception:
        return {}

    attribution = get_litellm_aiohttp_session_attribution(resource)
    if not isinstance(attribution, dict):
        return {}

    fields = {
        "aiohttp_owner_kind": attribution.get("owner_kind"),
        "aiohttp_creation_site": attribution.get("creation_site"),
        "aiohttp_litellm_owns_session": attribution.get("litellm_owns_session"),
        "aiohttp_session_id": attribution.get("session_id"),
        "aiohttp_connector_id": attribution.get("connector_id"),
        "aiohttp_event_loop_id": attribution.get("event_loop_id"),
        "aiohttp_pid": attribution.get("pid"),
        "aiohttp_container_hostname": attribution.get("container_hostname"),
        "aiohttp_context_keys": sorted({str(key) for key in context.keys()}),
        "aiohttp_context_resource": resource_name,
    }
    return {key: value for key, value in fields.items() if value is not None}


class AawmErrorLogFileHandler(logging.Handler):
    """Append sanitized LiteLLM ERROR records to the local .analysis intake log."""

    _formatter = logging.Formatter(
        "%(asctime)s - %(name)s:%(levelname)s: %(filename)s:%(lineno)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    _emit_state = threading.local()

    def __init__(self) -> None:
        super().__init__(level=logging.WARNING)
        self.name = _AAWM_ERROR_LOG_HANDLER_NAME
        self.addFilter(_secret_filter)
        self.setFormatter(self._formatter)

    def emit(self, record: logging.LogRecord) -> None:
        if getattr(self._emit_state, "active", False):
            return
        if record.levelno < logging.ERROR and record.exc_info is None:
            return
        if (
            record.exc_info is None
            and record.levelno >= logging.CRITICAL
            and record.getMessage() == "LITELLM_MASTER_KEY is not set"
        ):
            return

        log_path = _get_aawm_error_log_path()
        if not log_path:
            return

        self._emit_state.active = True
        try:
            _apply_langfuse_support_string_diagnostics(record)
            payload = _build_aawm_error_log_record(record, formatter=self._formatter)
            if _should_suppress_langfuse_support_string_coalesce(payload):
                return
            with _AAWM_ERROR_LOG_LOCK:
                os.makedirs(os.path.dirname(log_path), exist_ok=True)
                with open(log_path, "a", encoding="utf-8") as error_log:
                    error_log.write(safe_dumps(payload))
                    error_log.write("\n")
                _normalize_aawm_error_log_file_metadata(log_path)
        except Exception:
            # Never let local error-intake logging break application logging.
            return
        finally:
            self._emit_state.active = False

_EGRESS_GUARD_ALERT_LOCK = threading.Lock()
_EGRESS_GUARD_ALERT_STATE: Dict[str, Any] = {
    "trigger_count": 0,
    "last_triggered_at": None,
    "last_reason": None,
    "last_target": None,
    "last_credential_family": None,
    "last_target_family": None,
}


def _get_egress_guard_alert_suffix() -> str:
    message = "EGRESS GUARD TRIGGERED - INVESTIGATE IMMEDIATELY"
    if json_logs:
        return message
    return f"\033[91m{message}\033[0m"


def get_egress_guard_alert_state() -> Dict[str, Any]:
    with _EGRESS_GUARD_ALERT_LOCK:
        return dict(_EGRESS_GUARD_ALERT_STATE)


def reset_egress_guard_alert_state() -> None:
    with _EGRESS_GUARD_ALERT_LOCK:
        _EGRESS_GUARD_ALERT_STATE.update(
            {
                "trigger_count": 0,
                "last_triggered_at": None,
                "last_reason": None,
                "last_target": None,
                "last_credential_family": None,
                "last_target_family": None,
            }
        )


def trigger_egress_guard_alert(
    *,
    reason: str,
    target: Optional[str] = None,
    credential_family: Optional[str] = None,
    target_family: Optional[str] = None,
) -> Dict[str, Any]:
    with _EGRESS_GUARD_ALERT_LOCK:
        _EGRESS_GUARD_ALERT_STATE["trigger_count"] += 1
        _EGRESS_GUARD_ALERT_STATE["last_triggered_at"] = datetime.utcnow().isoformat()
        _EGRESS_GUARD_ALERT_STATE["last_reason"] = reason
        _EGRESS_GUARD_ALERT_STATE["last_target"] = target
        _EGRESS_GUARD_ALERT_STATE["last_credential_family"] = credential_family
        _EGRESS_GUARD_ALERT_STATE["last_target_family"] = target_family
        return dict(_EGRESS_GUARD_ALERT_STATE)


class EgressGuardAlertFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        state = get_egress_guard_alert_state()
        if state.get("trigger_count", 0) <= 0:
            return True
        if getattr(record, "_egress_guard_alert_suffix_applied", False):
            return True

        suffix = _get_egress_guard_alert_suffix()
        try:
            if isinstance(record.msg, str):
                record.msg = f"{record.msg} [{suffix}]"
            else:
                record.msg = f"{record.getMessage()} [{suffix}]"
                record.args = None
        except Exception:
            if isinstance(record.msg, str):
                record.msg = f"{record.msg} [{suffix}]"
        record._egress_guard_alert_suffix_applied = True
        return True


_egress_guard_alert_filter = EgressGuardAlertFilter()

_AawmRouteAccessLogReplacementKey = Tuple[str, str, str, str]
_AAWM_ROUTE_ACCESS_LOG_REPLACEMENT_LIMIT = 1024
_aawm_route_access_log_replacement_lock = threading.Lock()
_aawm_route_access_log_replacements: Set[_AawmRouteAccessLogReplacementKey] = set()
_aawm_route_access_log_replacement_order: Deque[
    _AawmRouteAccessLogReplacementKey
] = deque()


def _normalize_aawm_route_access_log_replacement_path(full_path: object) -> str:
    return unquote(str(full_path))


def clear_aawm_route_access_log_replacements() -> None:
    with _aawm_route_access_log_replacement_lock:
        _aawm_route_access_log_replacements.clear()
        _aawm_route_access_log_replacement_order.clear()


def register_aawm_route_access_log_replacement(
    *,
    client_addr: Optional[str],
    method: Optional[str],
    full_path: Optional[str],
    http_version: Optional[str],
) -> None:
    if not client_addr or not method or not full_path or not http_version:
        return

    key = (
        str(client_addr),
        str(method),
        _normalize_aawm_route_access_log_replacement_path(full_path),
        str(http_version),
    )
    with _aawm_route_access_log_replacement_lock:
        if key not in _aawm_route_access_log_replacements:
            _aawm_route_access_log_replacements.add(key)
            _aawm_route_access_log_replacement_order.append(key)

        while (
            len(_aawm_route_access_log_replacement_order)
            > _AAWM_ROUTE_ACCESS_LOG_REPLACEMENT_LIMIT
        ):
            stale_key = _aawm_route_access_log_replacement_order.popleft()
            _aawm_route_access_log_replacements.discard(stale_key)


def _aawm_route_access_log_key_from_record(
    record: logging.LogRecord,
) -> Optional[_AawmRouteAccessLogReplacementKey]:
    args = record.args
    if not isinstance(args, (list, tuple)) or len(args) < 4:
        return None

    client_addr, method, full_path, http_version = args[:4]
    if (
        client_addr is None
        or method is None
        or full_path is None
        or http_version is None
    ):
        return None
    return (
        str(client_addr),
        str(method),
        _normalize_aawm_route_access_log_replacement_path(full_path),
        str(http_version),
    )


def _consume_aawm_route_access_log_replacement(
    record: logging.LogRecord,
) -> bool:
    key = _aawm_route_access_log_key_from_record(record)
    if key is None:
        return False

    with _aawm_route_access_log_replacement_lock:
        if key not in _aawm_route_access_log_replacements:
            return False
        _aawm_route_access_log_replacements.remove(key)
        try:
            _aawm_route_access_log_replacement_order.remove(key)
        except ValueError:
            pass
        return True


class AawmRouteAccessLogReplacementFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        if record.name not in {"uvicorn.access", "gunicorn.access"}:
            return True
        return not _consume_aawm_route_access_log_replacement(record)


_aawm_route_access_log_replacement_filter = AawmRouteAccessLogReplacementFilter()


_AAWM_HEALTH_ACCESS_LOG_PATHS = frozenset(
    {
        "/health",
        "/health/",
        "/health/liveliness",
        "/health/readiness",
        "/health/services",
    }
)
_AAWM_HEALTH_ACCESS_LOG_STATUSES = frozenset({200, 204})


class AawmHealthAccessLogFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        if record.name not in {"uvicorn.access", "gunicorn.access"}:
            return True

        args = record.args
        if not isinstance(args, (list, tuple)) or len(args) < 5:
            return True

        _client_addr, method, full_path, _http_version, status_code = args[:5]
        if str(method).upper() != "GET":
            return True
        path = _normalize_aawm_route_access_log_replacement_path(full_path).split(
            "?",
            1,
        )[0]
        if path not in _AAWM_HEALTH_ACCESS_LOG_PATHS:
            return True
        try:
            normalized_status_code = int(status_code)
        except Exception:
            return True
        return normalized_status_code not in _AAWM_HEALTH_ACCESS_LOG_STATUSES


_aawm_health_access_log_filter = AawmHealthAccessLogFilter()


json_logs = bool(os.getenv("JSON_LOGS", False))
# Create a handler for the logger (you may need to adapt this based on your needs)
log_level = os.getenv("LITELLM_LOG", "DEBUG")
numeric_level: str = getattr(logging, log_level.upper())
handler = logging.StreamHandler()
handler.setLevel(numeric_level)
handler.addFilter(_secret_filter)
handler.addFilter(_egress_guard_alert_filter)


def _try_parse_json_message(message: str) -> Optional[Dict[str, Any]]:
    """
    Try to parse a log message as JSON. Returns parsed dict if valid, else None.
    Handles messages that are entirely valid JSON (e.g. json.dumps output).
    Uses shared safe_json_loads for consistent error handling.
    """
    if not message or not isinstance(message, str):
        return None
    msg_stripped = message.strip()
    if not (msg_stripped.startswith("{") or msg_stripped.startswith("[")):
        return None
    parsed = safe_json_loads(message, default=None)
    if parsed is None or not isinstance(parsed, dict):
        return None
    return parsed


def _try_parse_embedded_python_dict(message: str) -> Optional[Dict[str, Any]]:
    """
    Try to find and parse a Python dict repr (e.g. str(d) or repr(d)) embedded in
    the message. Handles patterns like:
    "get_available_deployment for model: X, Selected deployment: {'model_name': '...', ...} for model: X"
    Uses ast.literal_eval for safe parsing. Returns the parsed dict or None.
    """
    if not message or not isinstance(message, str) or "{" not in message:
        return None
    i = 0
    while i < len(message):
        start = message.find("{", i)
        if start == -1:
            break
        depth = 0
        for j in range(start, len(message)):
            c = message[j]
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    substr = message[start : j + 1]
                    try:
                        result = ast.literal_eval(substr)
                        if isinstance(result, dict) and len(result) > 0:
                            return result
                    except (ValueError, SyntaxError, TypeError):
                        pass
                    break
        i = start + 1
    return None


# Standard LogRecord attribute names - used to identify 'extra' fields.
# Derived at runtime so we automatically include version-specific attrs (e.g. taskName).
def _get_standard_record_attrs() -> frozenset:
    """Standard LogRecord attribute names - excludes extra keys from logger.debug(..., extra={...})."""
    return frozenset(logging.LogRecord("", 0, "", 0, "", (), None).__dict__.keys())


_STANDARD_RECORD_ATTRS = _get_standard_record_attrs()

_LANGFUSE_RECENT_ENQUEUE_AUDIT_LOCK = threading.Lock()
_LANGFUSE_RECENT_ENQUEUE_AUDIT_LIMIT = 64
_LANGFUSE_RECENT_ENQUEUE_AUDIT_TTL_SECONDS = 600
_langfuse_recent_enqueue_audits: Deque[Dict[str, Any]] = deque()


def _compact_langfuse_enqueue_audit_summary(size_summary: Dict[str, Any]) -> Dict[str, Any]:
    compact: Dict[str, Any] = {
        "observed_at": datetime.now(tz=UTC).isoformat(),
        "trace_id": size_summary.get("trace_id"),
        "generation_id": size_summary.get("generation_id"),
        "generation_name": size_summary.get("generation_name"),
        "model": size_summary.get("model"),
        "call_type": size_summary.get("call_type"),
        "total_size_bytes": size_summary.get("total_size_bytes"),
        "max_event_size_bytes": size_summary.get("max_event_size_bytes"),
        "warning_threshold_bytes": size_summary.get("warning_threshold_bytes"),
        "input_size_bytes": size_summary.get("input_size_bytes"),
        "output_size_bytes": size_summary.get("output_size_bytes"),
        "metadata_size_bytes": size_summary.get("metadata_size_bytes"),
        "event_fit_failed": bool(size_summary.get("event_fit_failed")),
    }
    compaction = size_summary.get("compaction_savings_audit")
    if isinstance(compaction, dict):
        compact["compaction_event_fit_failed"] = bool(compaction.get("event_fit_failed"))
        compact["compaction_final_total_size_bytes"] = compaction.get(
            "final_total_size_bytes"
        )
    return compact


def record_langfuse_enqueue_size_audit(size_summary: Dict[str, Any]) -> None:
    """Remember compact Langfuse payload-size context for later SDK failure correlation."""
    if not isinstance(size_summary, dict):
        return
    compact = _compact_langfuse_enqueue_audit_summary(size_summary)
    with _LANGFUSE_RECENT_ENQUEUE_AUDIT_LOCK:
        _langfuse_recent_enqueue_audits.append(compact)
        while len(_langfuse_recent_enqueue_audits) > _LANGFUSE_RECENT_ENQUEUE_AUDIT_LIMIT:
            _langfuse_recent_enqueue_audits.popleft()


def clear_langfuse_enqueue_size_audits() -> None:
    with _LANGFUSE_RECENT_ENQUEUE_AUDIT_LOCK:
        _langfuse_recent_enqueue_audits.clear()


def _langfuse_payload_size_state_from_audit(audit: Dict[str, Any]) -> str:
    if bool(audit.get("event_fit_failed")) or bool(audit.get("compaction_event_fit_failed")):
        return "fit_failed_before_enqueue"
    total = audit.get("total_size_bytes")
    max_bytes = audit.get("max_event_size_bytes")
    threshold = audit.get("warning_threshold_bytes")
    if isinstance(total, int) and isinstance(max_bytes, int) and total > max_bytes:
        return "over_limit_before_enqueue"
    if isinstance(total, int) and isinstance(threshold, int) and total >= threshold:
        return "near_limit_before_enqueue"
    return "below_limit_before_enqueue"


def _get_recent_langfuse_enqueue_audit() -> Optional[Dict[str, Any]]:
    now = datetime.now(tz=UTC)
    with _LANGFUSE_RECENT_ENQUEUE_AUDIT_LOCK:
        audits = list(_langfuse_recent_enqueue_audits)
    for audit in reversed(audits):
        observed_at = audit.get("observed_at")
        if not isinstance(observed_at, str):
            continue
        try:
            observed_dt = datetime.fromisoformat(observed_at)
        except ValueError:
            continue
        if observed_dt.tzinfo is None:
            observed_dt = observed_dt.replace(tzinfo=UTC)
        age_seconds = (now - observed_dt).total_seconds()
        if age_seconds <= _LANGFUSE_RECENT_ENQUEUE_AUDIT_TTL_SECONDS:
            return audit
    return None


def _recommended_operator_action_for_langfuse_support_string(
    *, payload_size_state: str
) -> str:
    if payload_size_state == "over_limit_before_enqueue":
        return (
            "Telemetry-only Langfuse SDK background upload failed after enqueueing an "
            "oversized event. Reduce Langfuse payload size (input/output/metadata compaction) "
            "and inspect Langfuse worker/blob storage health near the same timestamp."
        )
    if payload_size_state == "near_limit_before_enqueue":
        return (
            "Telemetry-only Langfuse SDK background upload failed after enqueueing a "
            "near-limit event. Review Langfuse payload shaping/compaction and Langfuse "
            "ingestion/storage health near the same timestamp."
        )
    if payload_size_state == "fit_failed_before_enqueue":
        return (
            "Telemetry-only Langfuse SDK background upload failed after enqueueing an event "
            "that could not be compacted below the SDK size limit. Tighten Langfuse payload "
            "shaping and inspect Langfuse ingestion health near the same timestamp."
        )
    if payload_size_state == "below_limit_before_enqueue":
        return (
            "Telemetry-only Langfuse SDK background upload failed for a below-limit event. "
            "Inspect Langfuse web/worker/blob-storage and network health near the same "
            "timestamp before assuming LiteLLM callback code failed."
        )
    return (
        "Telemetry-only Langfuse SDK background upload failed. Inspect Langfuse "
        "web/worker/blob-storage health near the same timestamp and review recent Langfuse "
        "payload-size warnings for oversized or near-limit events."
    )


_LANGFUSE_SUPPORT_STRING = (
    "Unexpected error occurred. Please check your request and contact support: "
    "https://langfuse.com/support."
)


def _is_langfuse_support_string_message(message: str) -> bool:
    return message == _LANGFUSE_SUPPORT_STRING


def _apply_langfuse_support_string_diagnostics(record: logging.LogRecord) -> None:
    if record.name != "langfuse":
        return
    if getattr(record, "langfuse_support_string", False):
        return

    message = record.getMessage()
    if not _is_langfuse_support_string_message(message):
        return

    recent_audit = _get_recent_langfuse_enqueue_audit()
    payload_size_state = (
        _langfuse_payload_size_state_from_audit(recent_audit)
        if recent_audit is not None
        else "enqueued_but_sdk_failed_no_recent_audit"
    )

    setattr(record, "source", "langfuse_sdk")
    setattr(record, "callback_name", "langfuse")
    setattr(record, "callback_phase", "sdk_background_ingestion_upload")
    setattr(record, "failure_kind", "degraded_langfuse_sdk_background_ingestion")
    setattr(record, "langfuse_support_string", True)
    setattr(record, "langfuse_sdk_background_ingestion_failure", True)
    setattr(
        record,
        "langfuse_failure_class",
        "langfuse_sdk_background_ingestion_upload_failure",
    )
    setattr(record, "langfuse_payload_size_state", payload_size_state)
    setattr(
        record,
        "recommended_operator_action",
        _recommended_operator_action_for_langfuse_support_string(
            payload_size_state=payload_size_state
        ),
    )

    if recent_audit is not None:
        for field, attr in (
            ("trace_id", "trace_id"),
            ("generation_id", "langfuse_generation_id"),
            ("generation_name", "langfuse_generation_name"),
            ("model", "model"),
            ("call_type", "langfuse_call_type"),
            ("total_size_bytes", "langfuse_total_size_bytes"),
            ("max_event_size_bytes", "langfuse_max_event_size_bytes"),
            ("warning_threshold_bytes", "langfuse_warning_threshold_bytes"),
            ("observed_at", "langfuse_enqueue_observed_at"),
        ):
            value = recent_audit.get(field)
            if value is not None and getattr(record, attr, None) is None:
                setattr(record, attr, value)
        setattr(
            record,
            "langfuse_event_fit_failed",
            bool(recent_audit.get("event_fit_failed"))
            or bool(recent_audit.get("compaction_event_fit_failed")),
        )


class LangfuseSupportStringDiagnosticFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        _apply_langfuse_support_string_diagnostics(record)
        return True


_langfuse_support_string_diagnostic_filter = LangfuseSupportStringDiagnosticFilter()


class JsonFormatter(Formatter):
    def __init__(self):
        super(JsonFormatter, self).__init__()

    def formatTime(self, record, datefmt=None):
        # Use datetime to format the timestamp in ISO 8601 format
        dt = datetime.fromtimestamp(record.created)
        return dt.isoformat()

    def format(self, record):
        _apply_langfuse_support_string_diagnostics(record)
        message_str = record.getMessage()
        json_record: Dict[str, Any] = {
            "message": message_str,
            "level": record.levelname,
            "timestamp": self.formatTime(record),
            "logger": record.name,
        }

        # Parse embedded JSON or Python dict repr in message so sub-fields become first-class properties
        parsed = _try_parse_json_message(message_str)
        if parsed is None:
            parsed = _try_parse_embedded_python_dict(message_str)
        if parsed is not None:
            for key, value in parsed.items():
                if key not in json_record:
                    json_record[key] = value

        # Include extra attributes passed via logger.debug("msg", extra={...})
        for key, value in record.__dict__.items():
            if key not in _STANDARD_RECORD_ATTRS and key not in json_record:
                json_record[key] = value

        if record.exc_info:
            json_record["stacktrace"] = record.exc_text or self.formatException(record.exc_info)

        return safe_dumps(json_record)


# Function to set up exception handlers for JSON logging
def _setup_json_exception_handlers(formatter):
    # Create a handler with JSON formatting for exceptions
    error_handler = logging.StreamHandler()
    error_handler.setFormatter(formatter)
    error_handler.addFilter(_secret_filter)
    error_handler.addFilter(_egress_guard_alert_filter)
    aawm_error_handler = AawmErrorLogFileHandler()

    # Setup excepthook for uncaught exceptions
    def json_excepthook(exc_type, exc_value, exc_traceback):
        record = logging.LogRecord(
            name="LiteLLM",
            level=logging.ERROR,
            pathname="",
            lineno=0,
            msg=str(exc_value),
            args=(),
            exc_info=(exc_type, exc_value, exc_traceback),
        )
        error_handler.handle(record)
        aawm_error_handler.handle(record)

    sys.excepthook = json_excepthook

    # Configure asyncio exception handler if possible
    try:
        import asyncio

        def async_json_exception_handler(loop, context):
            exception = context.get("exception")
            if exception:
                exc_type = type(exception)
                record = logging.LogRecord(
                    name="LiteLLM",
                    level=logging.ERROR,
                    pathname="",
                    lineno=0,
                    msg=str(exception),
                    args=(),
                    exc_info=(exc_type, exception, exception.__traceback__),
                )
                error_handler.handle(record)
                aawm_error_handler.handle(record)
                return

            aiohttp_context = _extract_aiohttp_attribution_from_asyncio_context(context)
            if not aiohttp_context:
                loop.default_exception_handler(context)
                return

            record = logging.LogRecord(
                name="asyncio",
                level=logging.ERROR,
                pathname="",
                lineno=0,
                msg=str(context.get("message", "Asyncio error")),
                args=(),
                exc_info=None,
            )
            setattr(record, "source", "asyncio_exception_handler")
            setattr(record, "failure_kind", "aiohttp_lifecycle_warning")
            for key, value in aiohttp_context.items():
                setattr(record, key, value)

            error_handler.handle(record)
            aawm_error_handler.handle(record)

        asyncio.get_event_loop().set_exception_handler(async_json_exception_handler)
    except Exception:
        pass


# Create a formatter and set it for the handler
if json_logs:
    handler.setFormatter(JsonFormatter())
    _setup_json_exception_handlers(JsonFormatter())
else:
    formatter = logging.Formatter(
        "\033[92m%(asctime)s - %(name)s:%(levelname)s\033[0m: %(filename)s:%(lineno)s - %(message)s",
        datefmt="%H:%M:%S",
    )

    handler.setFormatter(formatter)

verbose_proxy_logger = logging.getLogger("LiteLLM Proxy")
verbose_router_logger = logging.getLogger("LiteLLM Router")
verbose_logger = logging.getLogger("LiteLLM")
verbose_aawm_route_logger = logging.getLogger("LiteLLM AAWM Route")
verbose_aawm_route_logger.setLevel(logging.INFO)

_aawm_route_handler = logging.StreamHandler()
_aawm_route_handler.setLevel(logging.INFO)
_aawm_route_handler.addFilter(_secret_filter)
_aawm_route_handler.addFilter(_egress_guard_alert_filter)
if json_logs:
    _aawm_route_handler.setFormatter(JsonFormatter())
else:
    _aawm_route_handler.setFormatter(logging.Formatter("%(message)s"))

# Add the handler to the logger
verbose_router_logger.addHandler(handler)
verbose_proxy_logger.addHandler(handler)
verbose_logger.addHandler(handler)
verbose_aawm_route_logger.addHandler(_aawm_route_handler)
verbose_aawm_route_logger.propagate = False


def _suppress_loggers():
    """Suppress noisy loggers at INFO level"""
    # Suppress httpx request logging at INFO level
    httpx_logger = logging.getLogger("httpx")
    httpx_logger.setLevel(logging.WARNING)

    # Suppress APScheduler logging at INFO level
    apscheduler_executors_logger = logging.getLogger("apscheduler.executors.default")
    apscheduler_executors_logger.setLevel(logging.WARNING)
    apscheduler_scheduler_logger = logging.getLogger("apscheduler.scheduler")
    apscheduler_scheduler_logger.setLevel(logging.WARNING)


# Call the suppression function
_suppress_loggers()

ALL_LOGGERS = [
    logging.getLogger(),
    verbose_logger,
    verbose_router_logger,
    verbose_proxy_logger,
    verbose_aawm_route_logger,
    logging.getLogger("uvicorn"),
    logging.getLogger("uvicorn.error"),
    logging.getLogger("uvicorn.access"),
    logging.getLogger("gunicorn.access"),
]


def _ensure_filter_on_logger(logger: logging.Logger, log_filter: logging.Filter) -> None:
    if any(existing_filter is log_filter for existing_filter in logger.filters):
        return
    logger.addFilter(log_filter)


for _logger in ALL_LOGGERS:
    _ensure_filter_on_logger(_logger, _egress_guard_alert_filter)
    _ensure_filter_on_logger(_logger, _langfuse_support_string_diagnostic_filter)

_ensure_filter_on_logger(
    logging.getLogger("uvicorn.access"),
    _aawm_route_access_log_replacement_filter,
)
_ensure_filter_on_logger(
    logging.getLogger("uvicorn.access"),
    _aawm_health_access_log_filter,
)
_ensure_filter_on_logger(
    logging.getLogger("gunicorn.access"),
    _aawm_route_access_log_replacement_filter,
)
_ensure_filter_on_logger(
    logging.getLogger("gunicorn.access"),
    _aawm_health_access_log_filter,
)


def _get_loggers_to_initialize():
    """
    Get all loggers that should be initialized with the JSON handler.

    Includes third-party integration loggers (like langfuse) if they are
    configured as callbacks.
    """
    loggers = list(ALL_LOGGERS)

    # Add langfuse logger if langfuse is being used as a callback
    langfuse_callbacks = {"langfuse", "langfuse_otel"}
    litellm_module = sys.modules.get("litellm")
    success_callbacks = getattr(litellm_module, "success_callback", []) or []
    failure_callbacks = getattr(litellm_module, "failure_callback", []) or []
    all_callbacks = set(success_callbacks + failure_callbacks)
    if langfuse_callbacks & all_callbacks:
        loggers.append(logging.getLogger("langfuse"))

    return loggers



_aawm_error_log_handler: Optional[logging.Handler] = None


def _get_aawm_error_log_handler() -> Optional[logging.Handler]:
    global _aawm_error_log_handler
    if _aawm_error_log_handler is not None:
        return _aawm_error_log_handler

    log_path = _get_aawm_error_log_path()
    if log_path is not None:
        _aawm_error_log_handler = AawmErrorLogFileHandler()
        _aawm_error_log_handler.name = _AAWM_ERROR_LOG_HANDLER_NAME
        return _aawm_error_log_handler
    return None


def _ensure_aawm_error_log_handler_on_logger(logger: logging.Logger) -> None:
    # Check if a handler with this name is already present in this logger or its parents (if propagating)
    curr = logger
    while curr:
        for h in curr.handlers:
            if getattr(h, "name", None) == _AAWM_ERROR_LOG_HANDLER_NAME:
                return
        if not curr.propagate or curr.parent is None:
            break
        curr = curr.parent

    handler = _get_aawm_error_log_handler()
    if handler:
        logger.addHandler(handler)


def _configure_aawm_error_log_handlers() -> None:
    if _get_aawm_error_log_path() is None:
        return

    for lg in _get_loggers_to_initialize():
        _ensure_aawm_error_log_handler_on_logger(lg)


_configure_aawm_error_log_handlers()
def _initialize_loggers_with_handler(handler: logging.Handler):
    """
    Initialize all loggers with a handler

    - Adds a handler to each logger
    - Prevents bubbling to parent/root (critical to prevent duplicate JSON logs)
    """
    handler.addFilter(_secret_filter)
    handler.addFilter(_egress_guard_alert_filter)
    handler.addFilter(_langfuse_support_string_diagnostic_filter)
    for lg in _get_loggers_to_initialize():
        lg.handlers.clear()  # remove any existing handlers
        _ensure_filter_on_logger(lg, _egress_guard_alert_filter)
        _ensure_filter_on_logger(lg, _langfuse_support_string_diagnostic_filter)
        lg.addHandler(handler)  # add JSON formatter handler
        lg.propagate = False  # prevent bubbling to parent/root
    _configure_aawm_error_log_handlers()


def _get_uvicorn_json_log_config():
    """
    Generate a uvicorn log_config dictionary that applies JSON formatting to all loggers.

    This ensures that uvicorn's access logs, error logs, and all application logs
    are formatted as JSON when json_logs is enabled.
    """
    json_formatter_class = "litellm._logging.JsonFormatter"

    # Use the module-level log_level variable for consistency
    uvicorn_log_level = log_level.upper()

    log_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "json": {
                "()": json_formatter_class,
            },
            "default": {
                "()": json_formatter_class,
            },
            "access": {
                "()": json_formatter_class,
            },
        },
        "filters": {
            "aawm_route_access_replacement": {
                "()": "litellm._logging.AawmRouteAccessLogReplacementFilter",
            },
            "aawm_health_access": {
                "()": "litellm._logging.AawmHealthAccessLogFilter",
            },
        },
        "handlers": {
            "default": {
                "formatter": "json",
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stdout",
            },
            "access": {
                "formatter": "access",
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stdout",
            },
        },
        "loggers": {
            "uvicorn": {
                "handlers": ["default"],
                "level": uvicorn_log_level,
                "propagate": False,
            },
            "uvicorn.error": {
                "handlers": ["default"],
                "level": uvicorn_log_level,
                "propagate": False,
            },
            "uvicorn.access": {
                "handlers": ["access"],
                "filters": [
                    "aawm_route_access_replacement",
                    "aawm_health_access",
                ],
                "level": uvicorn_log_level,
                "propagate": False,
            },
        },
    }

    return log_config


def _turn_on_json():
    """
    Turn on JSON logging

    - Adds a JSON formatter to all loggers
    """
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    _initialize_loggers_with_handler(handler)
    # Set up exception handlers
    _setup_json_exception_handlers(JsonFormatter())


def _turn_on_debug():
    verbose_logger.setLevel(level=logging.DEBUG)  # set package log to debug
    verbose_router_logger.setLevel(level=logging.DEBUG)  # set router logs to debug
    verbose_proxy_logger.setLevel(level=logging.DEBUG)  # set proxy logs to debug


def _disable_debugging():
    verbose_logger.disabled = True
    verbose_router_logger.disabled = True
    verbose_proxy_logger.disabled = True


def _enable_debugging():
    verbose_logger.disabled = False
    verbose_router_logger.disabled = False
    verbose_proxy_logger.disabled = False


def print_verbose(print_statement):
    try:
        if set_verbose:
            print(print_statement)  # noqa
    except Exception:
        pass


def _is_debugging_on() -> bool:
    """
    Returns True if debugging is on
    """
    return verbose_logger.isEnabledFor(logging.DEBUG) or set_verbose is True
