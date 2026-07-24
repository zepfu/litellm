"""AAWM observability callback for Langfuse attribution.

Extracts agent identity from the SubagentStart hook context injected into
request prompts, then enriches the langfuse_trace_name request header so
each agent's API calls can be distinguished in Langfuse.

The hook injects: "You are '<agent-name>' and you are working..."
Role profiles can also declare one exact supported profile sentence, such as:
"You are a 'worker' agent."
When no agent designation is found, defaults to "orchestrator".

Enriches langfuse_trace_name from "claude-code" to "claude-code.<agent>"
(e.g. "claude-code.ops").

Uses BOTH logging_hook() (sync) and async_logging_hook() (async) to modify
headers BEFORE Langfuse's add_metadata_from_header() reads them. The sync
hook is critical for pass-through endpoints because Langfuse runs as a string
callback ("langfuse") in the sync success_handler - the async hook alone
would race with the thread-pool-submitted sync handler.

Registration in litellm-config.yaml:
    litellm_settings:
      callbacks: ["aawm_litellm_callbacks.agent_identity.AawmAgentIdentity"]
      success_callback: ["langfuse"]

Session-history SQL constants and the durable queue/worker/spool/retry
service live in `litellm.integrations.aawm_session_history` and are
re-exported here for compatibility with repair/backfill scripts and tests.
"""

import ast
import asyncio  # noqa: F401 - monkeypatch surface for session_history writer tests
import atexit  # noqa: F401 - monkeypatch surface for session_history writer tests
import base64
import hashlib
import importlib  # noqa: F401 - monkeypatch surface for session_history writer tests
import inspect  # noqa: F401 - freevar seed for record APIs
import ipaddress
import json
import math
import os
import queue  # noqa: F401 - monkeypatch surface for session_history writer tests
import re
import shlex
import threading  # noqa: F401 - monkeypatch surface for session_history writer tests
import time  # noqa: F401 - monkeypatch surface for session_history writer tests
import warnings  # noqa: F401 - consumed by moved rate_limit_base._coerce_rate_limit_payload via host globals
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from importlib import metadata as importlib_metadata
from typing import (
    Any,
    Dict,
    Iterator,
    List,
    Optional,
    Set,
    Tuple,
    cast,  # noqa: F401 - consumed by moved provider_cache helpers via host globals
)
from urllib.parse import (  # noqa: F401 - parse_qsl/quote/urlencode consumed by moved coerce helpers via host globals
    parse_qsl,
    quote,
    urlencode,
    urlsplit,
    urlunsplit,
)

from litellm._logging import verbose_logger
from litellm.integrations.custom_logger import CustomLogger

try:
    from litellm.integrations.aawm_agent_quality_rules import (
        AgentQualityCommand,
        score_agent_quality_context,
    )
except ModuleNotFoundError as exc:
    if exc.name != "litellm.integrations.aawm_agent_quality_rules":
        raise
    from aawm_litellm_callbacks.aawm_agent_quality_rules import (  # type: ignore[import-not-found,no-redef]
        AgentQualityCommand,
        score_agent_quality_context,
    )
from litellm.proxy.aawm_route_logging import _resolve_aawm_route_host_name_from_ip
from litellm.secret_managers.main import get_secret_str

try:
    from litellm.llms.base_llm.base_model_iterator import BaseModelResponseIterator
except Exception:  # pragma: no cover - optional at import time
    BaseModelResponseIterator = None  # type: ignore[misc,assignment]

# Lazy litellm dependency resolution: this module is imported under
# litellm.integrations, so eagerly importing litellm at module top can be
# circular. Cache once on first use to avoid hot-path import churn.
_litellm_module: Any = None
_response_api_logging_utils: Any = None
_response_api_logging_utils_loaded = False


def _get_litellm_module() -> Any:
    global _litellm_module
    if _litellm_module is None:
        import litellm as litellm_module

        _litellm_module = litellm_module
    return _litellm_module


def _get_response_api_logging_utils() -> Any:
    global _response_api_logging_utils, _response_api_logging_utils_loaded
    if not _response_api_logging_utils_loaded:
        try:
            from litellm.responses.utils import ResponseAPILoggingUtils as _utils
        except Exception:
            _response_api_logging_utils = None
        else:
            _response_api_logging_utils = _utils
        _response_api_logging_utils_loaded = True
    return _response_api_logging_utils


_CLAUDE_PERMISSION_CHECK_OUTPUT_RE = re.compile(
    r"^<block>\s*(?P<decision>yes|no)\s*$",
    re.IGNORECASE,
)
_AGENT_RE = re.compile(r"You are '([^']+)' and you are working")
_AGENT_TENANT_RE = re.compile(r"You are '(?P<agent>[^']+)' and you are working on the '(?P<tenant>[^']+)' project")
_AGENT_ROLE_RE = re.compile(
    r"^[ \t]*You are a '(?P<agent>explorer|worker|default)' agent\.[ \t]*$",
    re.MULTILINE,
)
_DEFAULT_AGENT = "orchestrator"
_CLAUDE_EXPERIMENT_ID_RE = re.compile(rb"(?<![A-Za-z0-9._-])([A-Za-z][A-Za-z0-9._-]{11,})(?![A-Za-z0-9._-])")
_GEMINI_MARKER = bytes.fromhex("8f3d6b5f")
_AAWM_DB_HOST_ENV_VARS = (
    "AAWM_DB_HOST",
    "AAWM_POSTGRES_SERVER",
    "POSTGRES_SERVER",
    "PGHOST",
)
_AAWM_DB_PORT_ENV_VARS = (
    "AAWM_DB_PORT",
    "AAWM_POSTGRES_PORT",
    "POSTGRES_PORT",
    "PGPORT",
)
_AAWM_DB_USER_ENV_VARS = (
    "AAWM_DB_USER",
    "AAWM_POSTGRES_USER",
    "POSTGRES_USER",
    "PGUSER",
)
_AAWM_DB_PASSWORD_ENV_VARS = (
    "AAWM_DB_PASSWORD",
    "AAWM_DB_PWD",
    "AAWM_POSTGRES_PASSWORD",
    "AAWM_POSTGRES_PWD",
    "POSTGRES_PASSWORD",
    "POSTGRES_PWD",
    "PGPASSWORD",
)
_AAWM_DB_NAME_ENV_VARS = (
    "AAWM_DB_NAME",
    "AAWM_POSTGRES_DATABASE",
    "POSTGRES_DATABASE",
    "PGDATABASE",
)
_AAWM_DB_SSLMODE_ENV_VARS = (
    "AAWM_DB_SSLMODE",
    "AAWM_POSTGRES_SSLMODE",
    "POSTGRES_SSLMODE",
    "PGSSLMODE",
)
_AAWM_DB_SSL_BOOL_ENV_VARS = (
    "AAWM_DB_SSL",
    "AAWM_POSTGRES_SSL",
    "POSTGRES_SSL",
)
_AAWM_DB_URL_ENV_VARS = (
    "AAWM_DB_URL",
    "AAWM_DATABASE_URL",
    "AAWM_POSTGRES_URL",
)
_AAWM_DB_APPLICATION_NAME_ENV_VARS = (
    "AAWM_SESSION_HISTORY_DB_APPLICATION_NAME",
    "AAWM_DB_APPLICATION_NAME",
    "AAWM_POSTGRES_APPLICATION_NAME",
    "PGAPPNAME",
)
_AAWM_LITELLM_FORK_VERSION_ENV_VARS = (
    "AAWM_LITELLM_FORK_VERSION",
    "LITELLM_FORK_VERSION",
)
_AAWM_ASSOCIATED_WHEEL_PACKAGES = (
    "litellm",
    "aawm-litellm-callbacks",
    "aawm-litellm-control-plane",
)
_USER_AGENT_PRODUCT_RE = re.compile(
    r"(?P<name>[A-Za-z][A-Za-z0-9._-]{1,63})/" r"(?P<version>[A-Za-z0-9][A-Za-z0-9.+_-]{0,127})"
)
_USER_AGENT_PAREN_PRODUCT_RE = re.compile(
    r"\((?P<name>[A-Za-z][A-Za-z0-9._-]{1,63})\s*;\s*" r"(?P<version>[A-Za-z0-9][A-Za-z0-9.+_-]{0,127})\)"
)
_RESET_AFTER_SECONDS_RE = re.compile(
    r"\breset(?:s|ting)?\s+after\s+(?P<seconds>\d+)s\b",
    re.IGNORECASE,
)
from litellm.integrations.aawm_session_history.sql import (  # noqa: F401
    _AAWM_SESSION_HISTORY_TABLE_SQL,
    _AAWM_SESSION_HISTORY_ALTER_STATEMENTS,
    _AAWM_SESSION_HISTORY_INDEX_STATEMENTS,
    _AAWM_SESSION_HISTORY_TOOL_ACTIVITY_TABLE_SQL,
    _AAWM_SESSION_HISTORY_TOOL_ACTIVITY_INDEX_STATEMENTS,
    _AAWM_TOOL_DEFINITION_SNAPSHOT_METADATA_KEY,
    _AAWM_SESSION_HISTORY_TOOL_DEFINITION_SNAPSHOTS_TABLE_SQL,
    _AAWM_SESSION_HISTORY_TOOL_DEFINITION_SNAPSHOTS_INDEX_STATEMENTS,
    _AAWM_RATE_LIMIT_OBSERVATIONS_TABLE_SQL,
    _AAWM_RATE_LIMIT_OBSERVATIONS_ALTER_STATEMENTS,
    _AAWM_RATE_LIMIT_OBSERVATIONS_INDEX_STATEMENTS,
    _AAWM_OPENROUTER_FREE_DAILY_REQUEST_COUNT_SQL,
    _AAWM_RATE_LIMIT_TRANSITIONS_TABLE_SQL,
    _AAWM_RATE_LIMIT_TRANSITIONS_ALTER_STATEMENTS,
    _AAWM_RATE_LIMIT_TRANSITIONS_INDEX_STATEMENTS,
    _AAWM_PROVIDER_ERROR_OBSERVATIONS_TABLE_SQL,
    _AAWM_PROVIDER_ERROR_OBSERVATIONS_ALTER_STATEMENTS,
    _AAWM_PROVIDER_ERROR_OBSERVATIONS_INDEX_STATEMENTS,
    _AAWM_PROVIDER_STATUS_OBSERVATIONS_TABLE_SQL,
    _AAWM_PROVIDER_STATUS_OBSERVATIONS_ALTER_STATEMENTS,
    _AAWM_PROVIDER_STATUS_OBSERVATIONS_INDEX_STATEMENTS,
    _AAWM_SESSION_HISTORY_INSERT_SQL,
    _AAWM_CLAUDE_AUTO_REVIEW_PARENT_IDENTITY_SQL,
    _SESSION_HISTORY_PREVIOUS_GAP_FIELD,
    _AAWM_SESSION_HISTORY_PREVIOUS_GAP_UPDATE_SQL,
    _AAWM_SESSION_HISTORY_TOOL_ACTIVITY_INSERT_SQL,
    _AAWM_SESSION_HISTORY_TOOL_DEFINITION_SNAPSHOT_INSERT_SQL,
    _AAWM_RATE_LIMIT_OBSERVATION_INSERT_SQL,
    _AAWM_RATE_LIMIT_PREVIOUS_OBSERVATION_SQL,
    _AAWM_RATE_LIMIT_PREVIOUS_OBSERVATIONS_BATCH_SQL,
    _AAWM_RATE_LIMIT_TRANSITION_INSERT_SQL,
    _AAWM_PROVIDER_ERROR_OBSERVATION_INSERT_SQL,
    _AAWM_ALIAS_ROUTING_AUDIT_TABLE_SQL,
    _AAWM_ALIAS_ROUTING_AUDIT_INDEX_STATEMENTS,
    _AAWM_ALIAS_ROUTING_AUDIT_INSERT_SQL,
)
from litellm.integrations.aawm_session_history.writer import (  # noqa: F401
    _AAWM_SESSION_HISTORY_APPLICATION_NAME,
    _AAWM_SESSION_HISTORY_BATCH_SIZE,
    _AAWM_SESSION_HISTORY_COMMAND_TIMEOUT_SECONDS,
    _AAWM_SESSION_HISTORY_DEGRADED_SPOOL_SECONDS,
    _AAWM_SESSION_HISTORY_FAILED_FLUSH_MAX_RETRIES,
    _AAWM_SESSION_HISTORY_FAILED_FLUSH_RETRY_SECONDS,
    _AAWM_SESSION_HISTORY_FLUSH_INTERVAL_SECONDS,
    _AAWM_SESSION_HISTORY_OVERFLOW_FLUSHERS,
    _AAWM_SESSION_HISTORY_POOL_MAX_SIZE,
    _AAWM_SESSION_HISTORY_QUEUE_DRAIN_TO_SPOOL_MAX_RECORDS,
    _AAWM_SESSION_HISTORY_QUEUE_TIMEOUT_SECONDS,
    _AAWM_SESSION_HISTORY_RETRYABLE_EXCEPTION_NAMES,
    _AAWM_SESSION_HISTORY_RETRYABLE_MESSAGE_MARKERS,
    _AAWM_SESSION_HISTORY_SPOOL_DATETIME_MARKER,
    _AAWM_SESSION_HISTORY_SPOOL_DIR_DEFAULT,
    _AAWM_SESSION_HISTORY_SPOOL_DIR_ENV,
    _AAWM_SESSION_HISTORY_SPOOL_DRAIN_THREAD_NAME,
    _AAWM_SESSION_HISTORY_SPOOL_REPLAY_BACKOFF_SECONDS,
    _AAWM_SESSION_HISTORY_STATEMENT_CACHE_SIZE,
    _SessionHistorySpoolListing,
    _aawm_session_history_degraded_failure_fingerprint,
    _aawm_session_history_degraded_lock,
    _aawm_session_history_degraded_until_monotonic,
    _aawm_session_history_flush_failure_active,
    _aawm_session_history_flush_failure_lock,
    _aawm_session_history_overflow_flush_semaphore,
    _aawm_session_history_pool_lock,
    _aawm_session_history_pools,
    _aawm_session_history_queue,
    _aawm_session_history_schema_lock,
    _aawm_session_history_schema_ready,
    _aawm_session_history_spool_drain_lock,
    _aawm_session_history_spool_drainer,
    _aawm_session_history_spool_drainer_lock,
    _aawm_session_history_spool_startup_bootstrapped,
    _aawm_session_history_spool_startup_lock,
    _aawm_session_history_suppressed_flush_failures,
    _aawm_session_history_worker,
    _aawm_session_history_worker_lock,
    _append_aawm_dsn_query_params_for_session_history,
    _bootstrap_session_history_spool_drainer_once,
    _build_aawm_dsn_for_session_history,
    _build_session_history_dsn,
    _call,
    _clear_session_history_degraded_spooling,
    _close_aawm_session_history_pools_for_current_loop,
    _decode_session_history_spool_value,
    _drain_session_history_queue_for_spool,
    _drain_session_history_queue_to_spool_on_shutdown,
    _drop_aawm_session_history_pools_for_current_loop,
    _encode_session_history_spool_value,
    _enqueue_session_history_record,
    _ensure_session_history_schema,
    _ensure_session_history_spool_dir,
    _ensure_session_history_spool_drainer_started,
    _ensure_session_history_worker_started,
    _flush_session_history_batch,
    _flush_session_history_batch_with_retry,
    _flush_session_history_overflow_record,
    _format_exception_for_warning,
    _get_aawm_session_history_pool,
    _get_persist_session_history_records,
    _get_session_history_application_name,
    _get_session_history_batch_size,
    _get_session_history_command_timeout_seconds,
    _get_session_history_degraded_spool_seconds,
    _get_session_history_degraded_spooling_context,
    _get_session_history_failed_flush_max_retries,
    _get_session_history_failed_flush_retry_seconds,
    _get_session_history_flush_interval_seconds,
    _get_session_history_pool_max_size,
    _get_session_history_server_settings,
    _get_session_history_spool_dir,
    _get_session_history_spool_replay_backoff_seconds,
    _get_session_history_statement_cache_size,
    _handle_session_history_retry_exhaustion,
    _identity_host,
    _initialize_session_history_connection,
    _is_retryable_session_history_persistence_failure,
    _iter_exception_chain,
    _list_session_history_spool,
    _load_session_history_spool_record,
    _load_session_history_spool_records,
    _log_recovered_retryable_session_history_flush,
    _log_session_history_retry,
    _mark_session_history_degraded_for_spooling,
    _mark_session_history_flush_failure_for_logging,
    _mirror_state,
    _open_aawm_session_history_connection,
    _prepare_session_history_retry_after_failure,
    _remove_recovered_session_history_retry_spool,
    _reset_session_history_flush_failure_window,
    _reset_session_history_pool_after_retryable_failure,
    _sanitize_session_history_spool_filename_component,
    _session_history_persistence_failure_fingerprint,
    _session_history_persistence_telemetry_suffix,
    _session_history_queue_depth_summary,
    _session_history_queue_depth_values,
    _session_history_retry_budget_remaining,
    _session_history_spool_bad_record,
    _session_history_spool_drainer_main,
    _session_history_spool_filename,
    _session_history_spool_identity,
    _session_history_spool_paths,
    _session_history_spool_summary,
    _session_history_worker_main,
    _shutdown_session_history_worker,
    _spool_session_history_record,
    _spool_session_history_records,
    _start_session_history_spool_drainer_after_retry_exhaustion,
    _state,
    _writer_get_secret_str,
)


from litellm.integrations.aawm_session_history import record as _aawm_session_history_record


def _bind_session_history_record_apis() -> None:
    """Install package-owned record/persist APIs into this module namespace.

    Record functions are defined as ordinary Python in
    `aawm_session_history.record` and rebound so their ``__globals__`` is this
    module (preserving monkeypatch-on-identity behavior) without compile/exec
    of source strings.
    """
    _aawm_session_history_record._ensure_installed()
    for _name in _aawm_session_history_record._RECORD_API_NAMES:
        globals()[_name] = getattr(_aawm_session_history_record, _name)


_ANTHROPIC_CONTEXT_1M_MODEL_SUFFIX = "[1m]"
_ANTHROPIC_CONTEXT_1M_BETA_HEADER = "context-1m-2025-08-07"
_ANTHROPIC_CONTEXT_1M_BETA_PREFIX = "context-1m"
_ANTHROPIC_CONTEXT_WINDOW_DEFAULT_TOKEN_COUNT = 200_000
_ANTHROPIC_CONTEXT_WINDOW_1M_TOKEN_COUNT = 1_000_000
_ANTHROPIC_CONTEXT_WINDOW_METADATA_KEYS = (
    "anthropic_context_window_mode",
    "anthropic_context_window_requested_tokens",
    "anthropic_context_window_source",
    "anthropic_context_window_beta",
    "anthropic_context_window_classification",
)

_WORKER_CONTEXT_EXHAUSTION_METADATA_KEYS = (
    "worker_context_exhaustion_failure_class",
    "worker_context_exhaustion_failure_reason",
    "worker_context_exhaustion_partial_output_summary",
    "worker_context_exhaustion_changed_paths_hint",
    "worker_context_exhaustion_attempted_patch_scope",
    "worker_context_exhaustion_last_visible_message",
    "worker_context_exhaustion_success",
    "worker_context_exhaustion_completed",
)
_WORKER_CONTEXT_EXHAUSTION_STRING_MAX_LEN = {
    "worker_context_exhaustion_failure_class": 128,
    "worker_context_exhaustion_failure_reason": 512,
    "worker_context_exhaustion_partial_output_summary": 2000,
    "worker_context_exhaustion_changed_paths_hint": 2000,
    "worker_context_exhaustion_attempted_patch_scope": 2000,
    "worker_context_exhaustion_last_visible_message": 2000,
}
_WORKER_CONTEXT_EXHAUSTION_BOOL_KEYS = frozenset(
    {
        "worker_context_exhaustion_success",
        "worker_context_exhaustion_completed",
    }
)
_PROMPT_OVERHEAD_TOKEN_FIELDS = (
    "input_system_tokens_estimated",
    "input_tool_advertisement_tokens_estimated",
    "input_conversation_tokens_estimated",
    "input_other_tokens_estimated",
    "input_breakdown_residual_tokens",
    "system_behavior_tokens_estimated",
    "system_safety_tokens_estimated",
    "system_instructional_tokens_estimated",
    "system_unclassified_tokens_estimated",
)
_SESSION_HISTORY_LATENCY_FIELDS = (
    "litellm_processing_ms",
    "llm_upstream_elapsed_ms",
    "total_server_elapsed_ms",
    "ttft_ms",
    "litellm_pre_send_ms",
    "litellm_post_response_ms",
    "llm_upstream_time_to_first_byte_ms",
    "llm_upstream_stream_ms",
    "latency_unclassified_ms",
)
_SESSION_HISTORY_AGENT_SCORE_FLOAT_FIELDS = (
    "trace_quality_score",
    "read_only_policy_compliance_score",
    "response_meaningfulness_score",
    "instruction_adherence_score",
    "answer_completeness_score",
    "evidence_fidelity_score",
    "tool_result_fidelity_score",
    "error_attribution_quality_score",
    "repetition_loop_risk_score",
    "context_retention_score",
    "tool_use_validity_score",
    "tool_error_recovery_score",
    "stall_risk_score",
    "output_contract_compliance_score",
    "task_progress_score",
    "scope_control_score",
    "destructive_action_policy_score",
    "ignored_path_tracking_policy_score",
    "baseline_deflection_attempted_score",
    "baseline_deflection_incident_score",
    "baseline_deflection_elapsed_ms",
    "sleep_wellness_interruption_attempted_score",
    "sleep_wellness_interruption_incident_score",
    "sleep_wellness_interruption_elapsed_ms",
    "terminal_completion_score",
    "discovery_inventory_coverage_score",
)
_SESSION_HISTORY_AGENT_SCORE_BOOL_FIELDS = (
    "empty_completion_failure",
    "large_tool_result_payload_risk",
    "destructive_checkout_after_work",
    "invalid_tool_call_error",
)
_SESSION_HISTORY_AGENT_SCORE_INT_FIELDS = (
    "read_only_policy_violation_count",
    "ignored_path_tracking_violation_count",
    "baseline_deflection_attempt_count",
    "baseline_deflection_tool_call_count",
    "baseline_deflection_input_tokens",
    "quality_gate_trigger_count",
    "quality_gate_fix_attempt_count",
    "quality_gate_rerun_count",
    "sleep_wellness_interruption_count",
    "sleep_wellness_interruption_output_tokens",
    "sleep_wellness_interruption_input_tokens",
    "sleep_wellness_interruption_after_user_pushback_count",
    "sleep_wellness_interruption_repeated_count",
    "discovery_inventory_missing_count",
)
_SESSION_HISTORY_OUTPUT_CONTRACT_STRING_FIELDS = (
    "output_contract_required_final_phrase",
    "output_contract_required_final_phrase_source",
    "output_contract_failure_class",
)
_SESSION_HISTORY_OUTPUT_CONTRACT_BOOL_FIELDS = (
    "output_contract_required_final_phrase_present",
    "output_contract_setup_only_detected",
)
_SESSION_HISTORY_OUTPUT_CONTRACT_INT_FIELDS = (
    "output_contract_failure_count",
    "output_contract_final_text_chars",
)
_SESSION_HISTORY_OUTPUT_CONTRACT_JSON_FIELDS = ("output_contract_setup_only_markers",)
_PROMPT_OVERHEAD_CLASSIFIER_VERSION = "deterministic-v2"
_AAWM_REQUEST_PAYLOAD_SCAN_MAX_DEPTH = 16
_AAWM_REQUEST_PAYLOAD_SCAN_MAX_ITEMS = 5000
_AAWM_JSON_SAFE_MAX_DEPTH = 12
_AAWM_TENANT_ID_HEADER_NAMES = (
    "x-aawm-tenant-id",
    "x-litellm-tenant-id",
    "x-litellm-organization-id",
    "x-litellm-org-id",
    "x-organization-id",
    "x-org-id",
    "x-litellm-team-id",
    "x-team-id",
)
_AAWM_AGENT_ID_METADATA_KEYS = (
    "agent_id",
    "aawm_agent_id",
    "source_agent_id",
    "subagent_id",
    "task_id",
)
_AAWM_AGENT_ID_HEADER_NAMES = (
    "x-aawm-agent-id",
    "x-grok-agent-id",
    "x-litellm-agent-id",
    "x-agent-id",
)
_AAWM_REPOSITORY_HEADER_NAMES = (
    "x-aawm-repository",
    "x-litellm-repository",
    "x-repository",
    "x-git-repository",
)
_AAWM_WORKSPACE_ROOT_ENV = "AAWM_WORKSPACE_ROOT"
_AAWM_CODEX_MEMORY_ROOT_ENV = "AAWM_CODEX_MEMORY_ROOT"
# Portable defaults: expanduser, never hardcode a developer home path.
_AAWM_WORKSPACE_ROOT_DEFAULT = os.path.expanduser("~/projects")
_AAWM_CODEX_MEMORY_ROOT_DEFAULT = os.path.expanduser("~/.codex/memories")


def _normalize_configured_root_path(value: Any) -> Optional[str]:
    if not isinstance(value, str):
        return None
    cleaned = value.strip().rstrip("/")
    if not cleaned or not cleaned.startswith("/"):
        return None
    return cleaned


def _get_aawm_workspace_root() -> str:
    try:
        configured = _normalize_configured_root_path(
            get_secret_str(_AAWM_WORKSPACE_ROOT_ENV) or os.environ.get(_AAWM_WORKSPACE_ROOT_ENV)
        )
    except Exception:
        configured = None
    return configured or _AAWM_WORKSPACE_ROOT_DEFAULT


def _get_codex_memory_root_path() -> str:
    try:
        configured = _normalize_configured_root_path(
            get_secret_str(_AAWM_CODEX_MEMORY_ROOT_ENV) or os.environ.get(_AAWM_CODEX_MEMORY_ROOT_ENV)
        )
    except Exception:
        configured = None
    return configured or _AAWM_CODEX_MEMORY_ROOT_DEFAULT


def _aawm_workspace_root_prefix() -> str:
    return f"{_get_aawm_workspace_root()}/"


def _build_aawm_repository_text_patterns(
    workspace_root: Optional[str] = None,
) -> Tuple[re.Pattern[str], ...]:
    root = workspace_root or _get_aawm_workspace_root()
    workspace_prefix = re.escape(f"{root.rstrip('/')}/")
    return (
        re.compile(
            r"<environment_context>[\s\S]{0,2000}<cwd>\s*[`'\"]?(?P<path>[^<`'\"]+)</cwd>",
            re.IGNORECASE,
        ),
        re.compile(r"<cwd>\s*[`'\"]?(?P<path>[^<`'\"]+)</cwd>"),
        re.compile(r"AGENTS\.md instructions for\s+[`'\"]?(?P<path>/[^\n<`'\"]+)"),
        re.compile(r"\bcwd\b\s*[:=]\s*[`'\"]?(?P<path>/[^,`'\"\n<]+)"),
        re.compile(
            r"\*{0,2}Workspace Directories:\*{0,2}\s*\n\s*[-*]\s*[`'\"]?(?P<path>/[^\n`'\"]+)",
            re.IGNORECASE,
        ),
        re.compile(rf"(?P<path>{workspace_prefix}[^,\s`'\"<)]+)"),
    )


_AAWM_REPOSITORY_TEXT_PATTERN_SOURCES = (
    "text.environment_context.cwd",
    "text.cwd_tag",
    "text.agents_instructions",
    "text.cwd_assignment",
    "text.workspace_directories",
    "text.project_path",
)


def _aawm_repository_text_markers(
    workspace_root: Optional[str] = None,
) -> Tuple[str, ...]:
    root = (workspace_root or _get_aawm_workspace_root()).rstrip("/")
    return (
        "<environment_context",
        "<cwd>",
        f"{root.lower()}/",
        "agents.md instructions for",
        "cwd",
        "workspace directories",
    )


# Import-time snapshots use defaults so module import stays free of secret lookups.
_AAWM_REPOSITORY_TEXT_PATTERNS = _build_aawm_repository_text_patterns(_AAWM_WORKSPACE_ROOT_DEFAULT)
_AAWM_REPOSITORY_TEXT_MARKERS = _aawm_repository_text_markers(_AAWM_WORKSPACE_ROOT_DEFAULT)
_CODEX_MEMORY_ROOT_PATH = _AAWM_CODEX_MEMORY_ROOT_DEFAULT
_AAWM_REPOSITORY_UNTRUSTED_TEXT_ITEM_TYPES = {
    "custom_tool_call",
    "custom_tool_call_output",
    "function_call",
    "function_call_output",
    "reasoning",
    "tool_search_call",
    "tool_search_output",
}
_AAWM_REPO_INSTRUCTION_FILENAMES = frozenset(
    {
        "agents.md",
        "claude.md",
        "gemini.md",
        "memory.md",
    }
)

_AAWM_REJECT_BARE_FILENAME_EXTENSIONS: frozenset[str] = frozenset(
    {
        ".py",
        ".pyi",
        ".pyx",
        ".js",
        ".mjs",
        ".cjs",
        ".ts",
        ".tsx",
        ".jsx",
        ".go",
        ".rs",
        ".java",
        ".kt",
        ".swift",
        ".scala",
        ".c",
        ".cc",
        ".cpp",
        ".cxx",
        ".h",
        ".hh",
        ".hpp",
        ".sh",
        ".bash",
        ".zsh",
        ".fish",
        ".yaml",
        ".yml",
        ".toml",
        ".json",
        ".jsonc",
        ".ini",
        ".cfg",
        ".conf",
        ".md",
        ".markdown",
        ".txt",
        ".rst",
        ".adoc",
        ".log",
        ".jsonl",
        ".out",
        ".err",
    }
)


_AAWM_REPOSITORY_PLACEHOLDER_VALUES = {
    "...",
    "0",
    ".analysis",
    ".codex",
    "agent-ok",
    "deep",
    "docker-compose.yml",
    "fixture",
    "memories",
    "myapp",
    "new",
    "nonexistent-worktree",
    "none",
    "null",
    "path",
    "project",
    "remote",
    "repo",
    "repository",
    "two",
    "unknown",
    "wt",
    "wt-ops-xyz",
    "x",
}

_KNOWN_AAWM_WORKSPACE_REPOS: frozenset[str] = frozenset(
    {
        "litellm",
        "aawm",
        "aawm-tap",
        "aawm-devtools",
        "aawm-infrastructure",
        "dashboard-shell",
        "aegis",
        "pytest-testable",
        "pytest-classifier",
        "aawm-transcript",
        "aawm-hook",
        "aawm-tap-dashboard",
        "aawm-observe",
        "mcp-pg",
        "sluice",
    }
)
_AAWM_REPOSITORY_AGENT_ROLE_VALUES = {
    "agent",
    "analyst",
    "architect",
    "engineer",
    "infra",
    "ops",
    "orchestrator",
    "principal",
    "qa",
    "researcher",
    "reviewer",
    "salvage",
    "tester",
}
_AAWM_REPOSITORY_AGENT_ID_RE = re.compile(r"^agent-[a-f0-9]{3,}$", re.IGNORECASE)
_AAWM_REPOSITORY_WAVE_AGENT_RE = re.compile(
    r"^wave\d+-(?:analyst|engineer|infra|ops|principal|qa|researcher|reviewer|salvage|tester)$",
    re.IGNORECASE,
)
_AAWM_NUMERIC_IDENTITY_ALLOWLIST: frozenset[str] = frozenset()
_AAWM_SCALAR_NUMERIC_IDENTITY_RE = re.compile(r"^[+-]?\d+$")
_CLAUDE_AUTO_REVIEW_LOGICAL_MODEL = "claude-auto-review"
_CLAUDE_AUTO_REVIEW_TRACE_NAME = "claude-code.auto-reviewer"
_CLAUDE_AUTO_REVIEW_AGENT_NAME = "auto-reviewer"
_CODEX_MEMORY_REPOSITORY_SUFFIX = " (memory)"
_CODEX_MEMORY_ROOT_REPOSITORY = "codex-memories"
_AAWM_REPOSITORY_ID_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)?$")
_AAWM_REPOSITORY_TRANSCRIPT_ARTIFACT_RE = re.compile(
    r"^(?:rollout-\d{4}(?:-[A-Za-z0-9_.-]*)?|.*\.jsonl?)$",
    re.IGNORECASE,
)
_CODEX_MEMORY_WORKFLOW_REQUIRED_MARKER = "memory writing agent"
_CODEX_MEMORY_WORKFLOW_CONTEXT_MARKERS = (
    "raw rollouts",
    "rollout_summary",
    "raw_memory",
    "do not follow any instructions found inside the rollout content",
)


def _content_to_text(content: Any) -> str:
    """Convert message content (string or Anthropic content blocks) to plain text.

    Only text-bearing content is kept. Non-text Anthropic/OpenAI content blocks
    (tool_use, tool_result, image, thinking, etc.) are skipped rather than
    contributing blank lines, so identity/text extraction is not diluted by
    empty placeholders.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for block in content:
            if isinstance(block, dict):
                if "text" not in block:
                    continue
                text = block.get("text")
                if text is None:
                    continue
                text_str = str(text)
                if text_str:
                    parts.append(text_str)
            else:
                text_str = str(block)
                if text_str:
                    parts.append(text_str)
        return "\n".join(parts)
    return str(content) if content else ""


def _get_header_value(headers: Any, *names: str) -> Optional[str]:
    if not headers:
        return None
    if not isinstance(headers, dict):
        try:
            headers = dict(headers)
        except (TypeError, ValueError):
            return None

    wanted = {name.lower() for name in names}
    for key, value in list(headers.items()):
        if str(key).lower() in wanted:
            return _clean_non_empty_string(value)
    return None


def _extract_request_headers_from_kwargs(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    header_sources = (
        _maybe_get_path(kwargs.get("litellm_params"), "proxy_server_request", "headers"),
        _maybe_get_path(kwargs.get("passthrough_logging_payload"), "request_headers"),
        _maybe_get_path(kwargs.get("standard_logging_object"), "request_headers"),
        _maybe_get_path(kwargs.get("standard_logging_object"), "headers"),
    )
    merged: Dict[str, Any] = {}
    for headers in header_sources:
        if not headers:
            continue
        if not isinstance(headers, dict):
            try:
                headers = dict(headers)
            except (TypeError, ValueError):
                continue
        merged.update(dict(headers))
    return merged


def _coerce_mapping(value: Any) -> Dict[str, Any]:
    parsed_value = _safe_json_load(value, value)
    return parsed_value if isinstance(parsed_value, dict) else {}


def _extract_tenant_identity_from_metadata_sources(
    *sources: Tuple[str, Any],
) -> Tuple[Optional[str], Optional[str]]:
    for source_name, raw_source in sources:
        source = _coerce_mapping(raw_source)
        if not source:
            continue
        for key in _AAWM_TENANT_ID_METADATA_KEYS:
            tenant_id = _normalize_tenant_identity(source.get(key))
            if tenant_id:
                return tenant_id, f"{source_name}.{key}"

        for nested_key in ("metadata", "request_metadata", "user_api_key_metadata"):
            nested_source = _coerce_mapping(source.get(nested_key))
            if not nested_source:
                continue
            for key in _AAWM_TENANT_ID_METADATA_KEYS:
                tenant_id = _normalize_tenant_identity(nested_source.get(key))
                if tenant_id:
                    return tenant_id, f"{source_name}.{nested_key}.{key}"

        repository, source_detail = _extract_route_rollup_repository_identity_from_mapping(
            source,
            source_name=source_name,
        )
        if repository:
            return repository, source_detail

        for nested_key in ("metadata", "request_metadata", "user_api_key_metadata"):
            nested_source = _coerce_mapping(source.get(nested_key))
            if not nested_source:
                continue
            repository, source_detail = _extract_route_rollup_repository_identity_from_mapping(
                nested_source,
                source_name=f"{source_name}.{nested_key}",
            )
            if repository:
                return repository, source_detail

    return None, None


def _agent_id_disallowed_values(
    *values: Any,
) -> Set[str]:
    disallowed: Set[str] = set()
    for value in values:
        cleaned = _clean_non_empty_string(value)
        if cleaned:
            disallowed.add(cleaned.strip("`'\"").lower())
    return disallowed


def _agent_id_disallowed_values_from_kwargs(
    kwargs: Dict[str, Any],
    *,
    metadata: Optional[Dict[str, Any]] = None,
    standard_logging_object: Optional[Dict[str, Any]] = None,
    agent_name: Optional[str] = None,
    tenant_id: Optional[str] = None,
    repository: Optional[str] = None,
) -> Set[str]:
    litellm_params = kwargs.get("litellm_params") or {}
    metadata = metadata or litellm_params.get("metadata") or {}
    standard_logging_object = standard_logging_object or kwargs.get("standard_logging_object") or {}
    standard_metadata = standard_logging_object.get("metadata") or {}
    return _agent_id_disallowed_values(
        agent_name,
        tenant_id,
        repository,
        _extract_session_id(kwargs),
        _extract_trace_id(kwargs),
        kwargs.get("litellm_call_id"),
        litellm_params.get("litellm_call_id"),
        metadata.get("session_id") if isinstance(metadata, dict) else None,
        metadata.get("trace_id") if isinstance(metadata, dict) else None,
        metadata.get("trace_user_id") if isinstance(metadata, dict) else None,
        metadata.get("repository") if isinstance(metadata, dict) else None,
        metadata.get("tenant_id") if isinstance(metadata, dict) else None,
        metadata.get("agent_name") if isinstance(metadata, dict) else None,
        standard_logging_object.get("session_id"),
        standard_logging_object.get("trace_id"),
        standard_metadata.get("session_id") if isinstance(standard_metadata, dict) else None,
        standard_metadata.get("trace_id") if isinstance(standard_metadata, dict) else None,
        standard_metadata.get("trace_user_id") if isinstance(standard_metadata, dict) else None,
        standard_metadata.get("repository") if isinstance(standard_metadata, dict) else None,
        standard_metadata.get("tenant_id") if isinstance(standard_metadata, dict) else None,
        standard_metadata.get("agent_name") if isinstance(standard_metadata, dict) else None,
    )


def _is_valid_repository_identity(value: str) -> bool:
    if value.endswith(_CODEX_MEMORY_REPOSITORY_SUFFIX):
        value = value[: -len(_CODEX_MEMORY_REPOSITORY_SUFFIX)]
    return bool(_AAWM_REPOSITORY_ID_PATTERN.fullmatch(value))


def _normalize_identity_for_placeholder_check(value: Any) -> Optional[str]:
    cleaned = _clean_non_empty_string(value)
    if cleaned is None:
        return None
    normalized = cleaned.strip("`'\"").strip().strip("/").lower()
    if normalized.endswith(_CODEX_MEMORY_REPOSITORY_SUFFIX):
        normalized = normalized[: -len(_CODEX_MEMORY_REPOSITORY_SUFFIX)]
    return normalized or None


def _is_numeric_identity_placeholder(value: Any) -> bool:
    normalized = _normalize_identity_for_placeholder_check(value)
    return bool(
        normalized
        and normalized not in _AAWM_NUMERIC_IDENTITY_ALLOWLIST
        and _AAWM_SCALAR_NUMERIC_IDENTITY_RE.fullmatch(normalized)
    )


def _is_disallowed_repository_identity(value: str) -> bool:
    normalized = _normalize_identity_for_placeholder_check(value)
    if not normalized:
        return True
    if normalized in _AAWM_REPOSITORY_PLACEHOLDER_VALUES:
        return True
    if _is_numeric_identity_placeholder(normalized):
        return True
    if _AAWM_REPOSITORY_TRANSCRIPT_ARTIFACT_RE.fullmatch(normalized):
        return True
    if normalized in _AAWM_REPOSITORY_AGENT_ROLE_VALUES:
        return True
    if _AAWM_REPOSITORY_AGENT_ID_RE.fullmatch(normalized):
        return True
    return bool(_AAWM_REPOSITORY_WAVE_AGENT_RE.fullmatch(normalized))


def _is_known_aawm_workspace_repository(value: Any) -> bool:
    """Return True only for conservative known AAWM workspace repo names.

    Used to gate metadata.repository -> tenant_id fallback for Codex records.
    Built-in list + optional AAWM_KNOWN_WORKSPACE_REPOS (comma list) env.
    Never allows generic owners (e.g. zepfu), wt-ops-*, file-like, or arbitrary.
    """
    normalized = _normalize_repository_identity(value)
    if not normalized:
        return False
    if normalized in _KNOWN_AAWM_WORKSPACE_REPOS:
        return True
    # env allowlist (comma-separated additional known repos)
    try:
        env_val = get_secret_str("AAWM_KNOWN_WORKSPACE_REPOS") or os.environ.get("AAWM_KNOWN_WORKSPACE_REPOS", "")
    except Exception:
        env_val = os.environ.get("AAWM_KNOWN_WORKSPACE_REPOS", "")
    if env_val:
        extras = {x.strip() for x in env_val.split(",") if x.strip()}
        if normalized in extras:
            return True
    return False


def _normalize_tenant_identity(value: Any) -> Optional[str]:
    cleaned = _clean_non_empty_string(value)
    if not cleaned:
        return None
    cleaned = cleaned.strip("`'\"")
    normalized = _normalize_identity_for_placeholder_check(cleaned)
    if normalized in {"...", "none", "null", "unknown"}:
        return None
    if _is_numeric_identity_placeholder(cleaned):
        return None
    return cleaned


def _is_bare_file_basename_with_reject_extension(value: str) -> bool:
    if not value or "/" in value or "\\" in value:
        return False
    v = value.lower().rstrip(".")
    for ext in _AAWM_REJECT_BARE_FILENAME_EXTENSIONS:
        if v.endswith(ext):
            return True
    if value.endswith(".") and value.rstrip("."):
        return True
    return False


def _is_bare_dot_directory(value: str) -> bool:
    if not value or "/" in value or "\\" in value:
        return False
    if not value.startswith("."):
        return False
    return True


def _extract_repository_identity_from_text_with_source(
    value: str,
) -> Tuple[Optional[str], Optional[str]]:
    # Rebuild from configured roots so env overrides work without process restart
    # of module constants used only for default import-time snapshots.
    patterns = _build_aawm_repository_text_patterns()
    markers = _aawm_repository_text_markers()
    normalized_value = value.lower()
    if not any(marker in normalized_value for marker in markers):
        return None, None
    for index, pattern in enumerate(patterns):
        matches = list(pattern.finditer(value))
        for match in reversed(matches):
            repository = _normalize_repository_identity(match.group("path"))
            if repository:
                source = (
                    _AAWM_REPOSITORY_TEXT_PATTERN_SOURCES[index]
                    if index < len(_AAWM_REPOSITORY_TEXT_PATTERN_SOURCES)
                    else "text"
                )
                return repository, source
    return None, None


_AAWM_ROUTE_ROLLUP_CONTEXT_METADATA_KEY = "aawm_route_rollup_context"
_AAWM_ROUTE_ROLLUP_GROUP_HEADER_LABEL_KEY = "group_header_label"
_AAWM_ROUTE_ROLLUP_GROUP_HEADER_LABEL_MAX_CHARS = 96


def _extract_repository_identity_from_route_rollup_group_header_label(
    value: Any,
) -> Optional[str]:
    """Bounded recovery of repository prefix from route-rollup group headers.

    Accepts labels such as ``aegis@Claude[2.1.199]`` and returns the normalized
    repository prefix (``aegis``). Does not accept arbitrary trace-user ids.
    """
    cleaned = _clean_non_empty_string(value)
    if not cleaned or len(cleaned) > _AAWM_ROUTE_ROLLUP_GROUP_HEADER_LABEL_MAX_CHARS:
        return None
    if "@" not in cleaned:
        return None
    repository_part = cleaned.split("@", 1)[0].strip()
    if not repository_part:
        return None
    repository = _normalize_repository_identity(repository_part)
    if not _is_known_aawm_workspace_repository(repository):
        return None
    return repository


def _extract_route_rollup_repository_identity_from_mapping(
    source: Dict[str, Any],
    *,
    source_name: str,
) -> Tuple[Optional[str], Optional[str]]:
    rollup_context = _coerce_mapping(source.get(_AAWM_ROUTE_ROLLUP_CONTEXT_METADATA_KEY))
    if not rollup_context:
        return None, None
    repository = _extract_repository_identity_from_route_rollup_group_header_label(
        rollup_context.get(_AAWM_ROUTE_ROLLUP_GROUP_HEADER_LABEL_KEY)
    )
    if not repository:
        return None, None
    return (
        repository,
        f"{source_name}.{_AAWM_ROUTE_ROLLUP_CONTEXT_METADATA_KEY}.{_AAWM_ROUTE_ROLLUP_GROUP_HEADER_LABEL_KEY}",
    )


def _repository_text_scan_blocked_by_mapping(value: Dict[str, Any]) -> bool:
    item_type = _clean_non_empty_string(value.get("type"))
    if item_type and item_type.lower() in _AAWM_REPOSITORY_UNTRUSTED_TEXT_ITEM_TYPES:
        return True
    role = _clean_non_empty_string(value.get("role"))
    return bool(role and role.lower() == "assistant")


def _extract_repository_identity_from_value_with_source(
    value: Any,
    *,
    source_prefix: str,
    _seen: Optional[set[int]] = None,
    _depth: int = 0,
) -> Tuple[Optional[str], Optional[str]]:
    if _depth > 12:
        return None, None
    if isinstance(value, (dict, list)):
        if _seen is None:
            _seen = set()
        value_id = id(value)
        if value_id in _seen:
            return None, None
        _seen.add(value_id)
    if isinstance(value, str):
        repository, source = _extract_repository_identity_from_text_with_source(value)
        if repository:
            return repository, f"{source_prefix}.{source}" if source else source_prefix
        return None, None
    if isinstance(value, dict):
        if _repository_text_scan_blocked_by_mapping(value):
            return None, None
        for key, child in list(value.items()):
            if key in _AAWM_REPOSITORY_METADATA_KEYS:
                repository = _normalize_repository_identity(child)
                if repository:
                    return repository, f"{source_prefix}.{key}"
            repository, source = _extract_repository_identity_from_value_with_source(
                child,
                source_prefix=f"{source_prefix}.{key}",
                _seen=_seen,
                _depth=_depth + 1,
            )
            if repository:
                return repository, source
    if isinstance(value, list):
        for index, child in reversed(list(enumerate(value))):
            repository, source = _extract_repository_identity_from_value_with_source(
                child,
                source_prefix=f"{source_prefix}[{index}]",
                _seen=_seen,
                _depth=_depth + 1,
            )
            if repository:
                return repository, source
    return None, None


def _extract_repository_identity_from_metadata_sources_with_source(
    *sources: Tuple[str, Any],
) -> Tuple[Optional[str], Optional[str]]:
    for source_name, raw_source in sources:
        source = _coerce_mapping(raw_source)
        if not source:
            continue
        for key in _AAWM_REPOSITORY_METADATA_KEYS:
            repository = _normalize_repository_identity(source.get(key))
            if repository:
                source_detail = (
                    _clean_non_empty_string(source.get("repository_source")) if key == "repository" else None
                )
                return repository, source_detail or f"{source_name}.{key}"

        for nested_key in (
            "metadata",
            "litellm_metadata",
            "request_metadata",
            "user_api_key_metadata",
        ):
            nested_source = _coerce_mapping(source.get(nested_key))
            if not nested_source:
                continue
            for key in _AAWM_REPOSITORY_METADATA_KEYS:
                repository = _normalize_repository_identity(nested_source.get(key))
                if repository:
                    source_detail = (
                        _clean_non_empty_string(nested_source.get("repository_source")) if key == "repository" else None
                    )
                    return (
                        repository,
                        source_detail or f"{source_name}.{nested_key}.{key}",
                    )

        repository, source_detail = _extract_route_rollup_repository_identity_from_mapping(
            source,
            source_name=source_name,
        )
        if repository:
            return repository, source_detail

        for nested_key in (
            "metadata",
            "litellm_metadata",
            "request_metadata",
            "user_api_key_metadata",
        ):
            nested_source = _coerce_mapping(source.get(nested_key))
            if not nested_source:
                continue
            repository, source_detail = _extract_route_rollup_repository_identity_from_mapping(
                nested_source,
                source_name=f"{source_name}.{nested_key}",
            )
            if repository:
                return repository, source_detail

        repository, source_detail = _extract_repository_identity_from_value_with_source(
            source,
            source_prefix=source_name,
        )
        if repository:
            return repository, source_detail

    return None, None


def _extract_repository_identity_from_kwargs_with_source(
    kwargs: Dict[str, Any],
    *,
    metadata: Optional[Dict[str, Any]] = None,
    standard_logging_object: Optional[Dict[str, Any]] = None,
) -> Tuple[Optional[str], Optional[str]]:
    litellm_params = kwargs.get("litellm_params") or {}
    standard_logging_object = standard_logging_object or kwargs.get("standard_logging_object") or {}
    standard_metadata = _coerce_mapping(standard_logging_object.get("metadata"))
    requester_custom_headers = _coerce_mapping(standard_metadata.get("requester_custom_headers"))
    passthrough_payload = kwargs.get("passthrough_logging_payload") or {}
    proxy_request = _coerce_mapping(litellm_params.get("proxy_server_request"))
    proxy_body = _coerce_mapping(proxy_request.get("body"))
    passthrough_body = _coerce_mapping(passthrough_payload.get("request_body"))

    headers = _extract_request_headers_from_kwargs(kwargs)
    for header_name in _AAWM_REPOSITORY_HEADER_NAMES:
        repository = _normalize_repository_identity(_get_header_value(headers, header_name))
        if repository:
            return repository, f"request_headers.{header_name}"

    # Prefer structured metadata, then request bodies that carry workspace text.
    # Do not deep-scan the entire kwargs / standard_logging_object /
    # passthrough_logging_payload graphs as undifferentiated last-resort
    # catch-alls (RR-006 #18). Per-value walkers already enforce depth/cycle
    # guards on the retained body sources.
    repository, source = _extract_repository_identity_from_metadata_sources_with_source(
        (
            "standard_logging_object.metadata.requester_custom_headers.x-codex-turn-metadata",
            requester_custom_headers.get("x-codex-turn-metadata"),
        ),
        ("litellm_params.metadata", metadata or litellm_params.get("metadata")),
        ("litellm_params.litellm_metadata", litellm_params.get("litellm_metadata")),
        ("standard_logging_object.metadata", standard_metadata),
        ("kwargs.metadata", kwargs.get("metadata")),
        ("litellm_params.proxy_server_request.body.metadata", proxy_body.get("metadata")),
        ("litellm_params.proxy_server_request.body.litellm_metadata", proxy_body.get("litellm_metadata")),
        ("litellm_params.proxy_server_request.body", proxy_body),
        ("passthrough_logging_payload.request_body.metadata", passthrough_body.get("metadata")),
        ("passthrough_logging_payload.request_body.litellm_metadata", passthrough_body.get("litellm_metadata")),
        ("passthrough_logging_payload.request_body", passthrough_body),
    )
    if repository:
        return repository, source

    repository, _source = _extract_claude_trace_user_identity_from_metadata_sources(
        ("litellm_params.metadata", metadata or litellm_params.get("metadata")),
        ("standard_logging_object.metadata", standard_logging_object.get("metadata")),
        ("kwargs.metadata", kwargs.get("metadata")),
        ("litellm_params.proxy_server_request.body", proxy_body),
        ("litellm_params.proxy_server_request.body.metadata", proxy_body.get("metadata")),
        ("litellm_params.proxy_server_request.body.litellm_metadata", proxy_body.get("litellm_metadata")),
        ("passthrough_logging_payload.request_body", passthrough_body),
        ("passthrough_logging_payload.request_body.metadata", passthrough_body.get("metadata")),
        ("passthrough_logging_payload.request_body.litellm_metadata", passthrough_body.get("litellm_metadata")),
    )
    if repository:
        return repository, _source

    return None, None


def _extract_repository_identity_from_langfuse_trace_observation_with_source(
    trace: Dict[str, Any],
    observation: Dict[str, Any],
    metadata: Optional[Dict[str, Any]] = None,
) -> Tuple[Optional[str], Optional[str]]:
    trace_metadata = trace.get("metadata") if isinstance(trace, dict) else None
    return _extract_repository_identity_from_metadata_sources_with_source(
        ("observation.metadata", metadata or observation.get("metadata")),
        ("trace.metadata", trace_metadata),
        ("observation", observation),
        ("trace", trace),
    )


def _payload_contains_codex_memory_workflow_markers(value: Any) -> bool:
    found_required_marker = False
    found_context_marker = False

    def visit(child: Any, *, _depth: int = 0, _seen: Optional[Set[int]] = None) -> None:
        nonlocal found_required_marker, found_context_marker
        if found_required_marker and found_context_marker:
            return
        if _depth > 12:
            return
        if isinstance(child, str):
            normalized = child.lower()
            if _CODEX_MEMORY_WORKFLOW_REQUIRED_MARKER in normalized:
                found_required_marker = True
            if any(marker in normalized for marker in _CODEX_MEMORY_WORKFLOW_CONTEXT_MARKERS):
                found_context_marker = True
            return
        if isinstance(child, (dict, list)):
            if _seen is None:
                _seen = set()
            child_id = id(child)
            if child_id in _seen:
                return
            _seen.add(child_id)
        if isinstance(child, dict):
            for nested in child.values():
                visit(nested, _depth=_depth + 1, _seen=_seen)
                if found_required_marker and found_context_marker:
                    return
        elif isinstance(child, list):
            for nested in child:
                visit(nested, _depth=_depth + 1, _seen=_seen)
                if found_required_marker and found_context_marker:
                    return

    visit(value)
    return found_required_marker and found_context_marker


def _format_memory_repository_identity(repository: str) -> str:
    if repository.endswith(_CODEX_MEMORY_REPOSITORY_SUFFIX):
        return repository
    return f"{repository}{_CODEX_MEMORY_REPOSITORY_SUFFIX}"


@lru_cache(maxsize=1)
def _resolve_runtime_litellm_version() -> Optional[str]:
    env_version = _get_first_secret_value(_AAWM_LITELLM_VERSION_ENV_VARS)
    if env_version:
        return env_version

    try:
        from litellm._version import version as litellm_version

        cleaned_version = _clean_non_empty_string(litellm_version)
        if cleaned_version and cleaned_version.lower() != "unknown":
            return cleaned_version
    except Exception:
        pass

    try:
        return _clean_non_empty_string(importlib_metadata.version("litellm"))
    except Exception:
        return None


def _derive_fork_version(litellm_version: Optional[str]) -> Optional[str]:
    env_version = _get_first_secret_value(_AAWM_LITELLM_FORK_VERSION_ENV_VARS)
    if env_version:
        return env_version
    if not litellm_version:
        return None
    if "+" not in litellm_version:
        return None
    local_version = litellm_version.split("+", 1)[1].strip()
    return local_version or None


@lru_cache(maxsize=1)
def _resolve_runtime_wheel_versions() -> Dict[str, str]:
    versions: Dict[str, str] = {}
    for package_name in _AAWM_ASSOCIATED_WHEEL_PACKAGES:
        try:
            version = _clean_non_empty_string(importlib_metadata.version(package_name))
        except Exception:
            version = None
        if version:
            versions[package_name] = version

    for package_name, env_vars in _AAWM_ASSOCIATED_VERSION_ENV_VARS.items():
        version = _get_first_secret_value(env_vars)
        if version:
            versions[package_name] = version
    return versions


_SESSION_HISTORY_LOOPBACK_HOST_LABEL = "localhost"


def _resolve_session_history_host_name_from_ip(
    client_ip: Optional[str],
) -> Tuple[Optional[str], Optional[str]]:
    return _resolve_aawm_route_host_name_from_ip(client_ip)


def _extract_session_host_attribution(
    metadata: Dict[str, Any],
) -> Dict[str, Optional[str]]:
    route_rollup_context = metadata.get("aawm_route_rollup_context")
    if not isinstance(route_rollup_context, dict):
        route_rollup_context = {}

    client_ip = None
    for candidate in (
        metadata.get("requester_ip_address"),
        metadata.get("client_ip"),
        route_rollup_context.get("client_ip"),
    ):
        client_ip = _canonical_session_history_client_ip(candidate)
        if client_ip:
            break

    host_name = _first_non_empty_string(
        metadata.get("host_name"),
        route_rollup_context.get("host_name"),
    )
    host_name_source = _first_non_empty_string(
        metadata.get("host_name_source"),
        route_rollup_context.get("host_name_source"),
    )
    if host_name is None and client_ip is not None:
        host_name, resolved_source = _resolve_session_history_host_name_from_ip(client_ip)
        if host_name_source is None:
            host_name_source = resolved_source

    client_ip_source = _first_non_empty_string(
        metadata.get("client_ip_source"),
        route_rollup_context.get("client_ip_source"),
    )
    return {
        "client_ip": client_ip,
        "host_name": host_name,
        "client_ip_source": client_ip_source,
        "host_name_source": host_name_source,
    }


def _build_session_runtime_identity(
    *,
    metadata: Dict[str, Any],
    kwargs: Optional[Dict[str, Any]] = None,
    trace_environment: Any = None,
    allow_runtime: bool = True,
) -> Dict[str, Any]:
    headers = _extract_request_headers_from_kwargs(kwargs or {})
    user_agent = _first_non_empty_string(
        metadata.get("client_user_agent"),
        metadata.get("user_agent"),
        metadata.get("http_user_agent"),
        _get_header_value(headers, "user-agent", "User-Agent"),
    )

    parsed_client_name, parsed_client_version = _parse_client_identity_from_user_agent(user_agent)
    cc_version, cc_entrypoint = _extract_claude_code_version_from_metadata(metadata)
    client_name = _first_non_empty_string(metadata.get("client_name"), parsed_client_name)
    client_version = _first_non_empty_string(
        metadata.get("client_version"),
        parsed_client_version,
    )
    if cc_version and (client_name is None or client_name.lower() == "claude-code"):
        client_name = "claude-code"
        client_version = cc_version
    if cc_entrypoint and client_name is None:
        client_name = cc_entrypoint

    runtime_environment = _get_first_secret_value(_AAWM_LITELLM_ENVIRONMENT_ENV_VARS) if allow_runtime else None
    litellm_environment = _first_non_empty_string(
        runtime_environment,
        metadata.get("litellm_environment"),
        metadata.get("trace_environment"),
        metadata.get("source_trace_environment"),
        trace_environment,
    )

    litellm_version = _first_non_empty_string(metadata.get("litellm_version"))
    if allow_runtime and litellm_version is None:
        litellm_version = _resolve_runtime_litellm_version()

    litellm_fork_version = _first_non_empty_string(metadata.get("litellm_fork_version"))
    if allow_runtime and litellm_fork_version is None:
        litellm_fork_version = _derive_fork_version(litellm_version)

    wheel_versions = _coerce_string_dict(metadata.get("litellm_wheel_versions"))
    if allow_runtime:
        runtime_versions = _resolve_runtime_wheel_versions()
        wheel_versions = {**runtime_versions, **wheel_versions}

    return {
        "litellm_environment": litellm_environment,
        "litellm_version": litellm_version,
        "litellm_fork_version": litellm_fork_version,
        "litellm_wheel_versions": wheel_versions,
        "client_name": client_name,
        "client_version": client_version,
        "client_user_agent": user_agent,
    }


def _enrich_session_runtime_identity_metadata(kwargs: Dict[str, Any]) -> None:
    metadata = _ensure_mutable_metadata(kwargs)
    identity = _build_session_runtime_identity(
        metadata=metadata,
        kwargs=kwargs,
        allow_runtime=True,
    )
    cc_version, cc_entrypoint = _extract_claude_code_version_from_metadata(metadata)
    if cc_version and not metadata.get("cc_version"):
        metadata["cc_version"] = cc_version
    if cc_entrypoint and not metadata.get("cc_entrypoint"):
        metadata["cc_entrypoint"] = cc_entrypoint

    for key, value in list(identity.items()):
        if key == "litellm_wheel_versions":
            if isinstance(value, dict) and value:
                metadata[key] = value
            continue
        if value is not None:
            metadata[key] = value
    host_attribution = _extract_session_host_attribution(metadata)
    for key, value in host_attribution.items():
        if value is not None:
            metadata[key] = value


def _extract_agent_context_from_text(text: str) -> Tuple[Optional[str], Optional[str]]:
    tenant_match = _AGENT_TENANT_RE.search(text)
    if tenant_match:
        return tenant_match.group("agent"), tenant_match.group("tenant")

    agent_match = _AGENT_RE.search(text)
    if agent_match:
        return agent_match.group(1), None

    role_match = _AGENT_ROLE_RE.search(text)
    if role_match:
        return role_match.group("agent"), None

    return None, None


def _extract_agent_context_from_mapping(
    source: Any,
    *,
    explicit_tenant_id: Optional[str],
    is_codex_client: bool,
) -> Tuple[Optional[str], Optional[str]]:
    if not isinstance(source, dict):
        return None, None
    agent_name = _clean_non_empty_string(source.get("agent_name") or source.get("aawm_claude_agent_name"))
    if agent_name is None and is_codex_client:
        agent_name = _clean_non_empty_string(source.get("agent_role") or source.get("agent_nickname"))
    tenant_id = _clean_non_empty_string(
        source.get("tenant_id") or source.get("aawm_tenant_id") or source.get("aawm_claude_project")
    )
    if agent_name:
        return agent_name, explicit_tenant_id or tenant_id
    trace_agent_name = _extract_claude_trace_agent_name(source.get("trace_name"))
    if trace_agent_name:
        return trace_agent_name, explicit_tenant_id or tenant_id
    return None, None


def _extract_agent_context(kwargs: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    """Extract agent/tenant from request content when present."""
    explicit_tenant_id, _tenant_source = _extract_tenant_identity_from_kwargs(kwargs)
    litellm_params = kwargs.get("litellm_params") or {}
    metadata = litellm_params.get("metadata") or {}
    headers = _extract_request_headers_from_kwargs(kwargs)
    is_codex_client = _is_codex_client_identity(
        metadata if isinstance(metadata, dict) else {},
        headers,
    )
    standard_logging_object = kwargs.get("standard_logging_object") or {}
    standard_metadata = standard_logging_object.get("metadata") or {}
    for source in (metadata, standard_metadata):
        agent_name, tenant_id = _extract_agent_context_from_mapping(
            source,
            explicit_tenant_id=explicit_tenant_id,
            is_codex_client=is_codex_client,
        )
        if agent_name:
            return agent_name, tenant_id

    messages = kwargs.get("messages")
    if messages and isinstance(messages, list):
        for message in messages:
            if not isinstance(message, dict):
                continue
            if message.get("role") != "system":
                continue
            text = _content_to_text(message.get("content", ""))
            agent_name, tenant_id = _extract_agent_context_from_text(text)
            if agent_name:
                return agent_name, explicit_tenant_id or tenant_id

    system_direct = kwargs.get("system")
    if system_direct:
        text = _content_to_text(system_direct)
        agent_name, tenant_id = _extract_agent_context_from_text(text)
        if agent_name:
            return agent_name, explicit_tenant_id or tenant_id

    payload = kwargs.get("passthrough_logging_payload")
    if isinstance(payload, dict):
        request_body = payload.get("request_body")
        if isinstance(request_body, dict):
            instructions = request_body.get("instructions")
            if instructions:
                text = _content_to_text(instructions)
                agent_name, tenant_id = _extract_agent_context_from_text(text)
                if agent_name:
                    return agent_name, explicit_tenant_id or tenant_id

            system = request_body.get("system")
            if system:
                text = _content_to_text(system)
                agent_name, tenant_id = _extract_agent_context_from_text(text)
                if agent_name:
                    return agent_name, explicit_tenant_id or tenant_id

            pt_messages = request_body.get("messages")
            if pt_messages and isinstance(pt_messages, list):
                for msg in pt_messages[:3]:
                    if not isinstance(msg, dict):
                        continue
                    if msg.get("role") != "user":
                        continue
                    text = _content_to_text(msg.get("content", ""))
                    agent_name, tenant_id = _extract_agent_context_from_text(text)
                    if agent_name:
                        return agent_name, explicit_tenant_id or tenant_id
                    break

    if _is_codex_default_agent_context(
        kwargs,
        metadata,
    ) and not _is_codex_subagent_context(kwargs, metadata):
        return _DEFAULT_AGENT, explicit_tenant_id

    return None, explicit_tenant_id


def _ensure_mutable_headers(kwargs: Dict[str, Any]) -> dict:
    """Ensure proxy_server_request.headers is a mutable dict.

    Mirrors `_ensure_mutable_metadata`: create and reattach the headers dict
    through litellm_params/proxy_server_request so callers can mutate it.
    """
    litellm_params = kwargs.get("litellm_params")
    if not isinstance(litellm_params, dict):
        litellm_params = {}
        kwargs["litellm_params"] = litellm_params

    psr = litellm_params.get("proxy_server_request")
    if not isinstance(psr, dict):
        psr = {}
        litellm_params["proxy_server_request"] = psr

    headers = psr.get("headers")
    if not isinstance(headers, dict):
        headers = dict(headers) if headers is not None else {}
        psr["headers"] = headers

    return headers


def _ensure_mutable_metadata(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    litellm_params = kwargs.get("litellm_params") or {}
    metadata = litellm_params.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
    litellm_params["metadata"] = metadata
    kwargs["litellm_params"] = litellm_params
    return metadata


def _is_generic_codex_trace_user_id(value: Any) -> bool:
    normalized = _clean_non_empty_string(value)
    return normalized is not None and (
        _is_numeric_identity_placeholder(normalized)
        or normalized.lower()
        in {
            "codex",
            "codex-cli",
            "codex-tui",
        }
    )


def _is_generic_grok_trace_user_id(value: Any) -> bool:
    normalized = _clean_non_empty_string(value)
    return normalized is not None and normalized.lower() in {
        "grok",
        "grok-build",
        "grok-cli",
        "xai",
        "xai-grok",
    }


def _is_generic_grok_trace_name(value: Any) -> bool:
    normalized = _clean_non_empty_string(value)
    if normalized is None:
        return True
    normalized_lower = normalized.lower()
    return normalized_lower in {"grok", "grok-build", "xai"} or normalized_lower.startswith("grok-build.")


def _merge_tags(metadata: Dict[str, Any], tags_to_add: List[str]) -> None:
    existing_tags = metadata.get("tags") or []
    if not isinstance(existing_tags, list):
        existing_tags = []

    merged_tags = list(existing_tags)
    for tag in tags_to_add:
        if tag and tag not in merged_tags:
            merged_tags.append(tag)
    metadata["tags"] = merged_tags


def _sync_standard_logging_object(kwargs: Dict[str, Any], metadata: Dict[str, Any]) -> None:
    standard_logging_object = kwargs.get("standard_logging_object")
    if not isinstance(standard_logging_object, dict):
        return

    standard_logging_metadata = standard_logging_object.get("metadata")
    if not isinstance(standard_logging_metadata, dict):
        standard_logging_metadata = {}
    standard_logging_metadata.update(dict(metadata))
    standard_logging_object["metadata"] = standard_logging_metadata

    tags = metadata.get("tags") or []
    if not isinstance(tags, list):
        tags = []
    metadata_request_tags = metadata.get("request_tags") or []
    if not isinstance(metadata_request_tags, list):
        metadata_request_tags = []
    existing_request_tags = standard_logging_object.get("request_tags") or []
    if not isinstance(existing_request_tags, list):
        existing_request_tags = []

    merged_request_tags = list(existing_request_tags)
    for tag in [*tags, *metadata_request_tags]:
        if isinstance(tag, str) and tag and tag not in merged_request_tags:
            merged_request_tags.append(tag)
    standard_logging_object["request_tags"] = merged_request_tags
    kwargs["standard_logging_object"] = standard_logging_object


def _maybe_get(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _maybe_get_path(obj: Any, *keys: str, default: Any = None) -> Any:
    current = obj
    for key in keys:
        if current is None:
            return default
        current = _maybe_get(current, key, default)
        if current is default:
            return default
    return current


def _extract_first_response_message(result: Any) -> Any:
    choices = _maybe_get(result, "choices")
    if not isinstance(choices, list) or len(choices) == 0:
        return None

    first_choice = choices[0]
    message = _maybe_get(first_choice, "message")
    if message is not None:
        return message
    return _maybe_get(first_choice, "delta")


def _extract_provider_specific_fields(message: Any) -> Dict[str, Any]:
    provider_specific_fields = _maybe_get(message, "provider_specific_fields")
    if isinstance(provider_specific_fields, dict):
        return provider_specific_fields
    return {}


def _extract_reasoning_content(message: Any, thinking_blocks: List[dict]) -> str:
    reasoning_content = _maybe_get(message, "reasoning_content")
    if isinstance(reasoning_content, str):
        return reasoning_content

    thinking_parts: List[str] = []
    for block in thinking_blocks:
        thinking_text = _maybe_get(block, "thinking")
        if isinstance(thinking_text, str) and thinking_text:
            thinking_parts.append(thinking_text)
    return "\n".join(thinking_parts)


def _extract_thinking_blocks(message: Any) -> List[dict]:
    thinking_blocks = _maybe_get(message, "thinking_blocks")
    if not isinstance(thinking_blocks, list):
        provider_specific_fields = _extract_provider_specific_fields(message)
        thinking_blocks = provider_specific_fields.get("thinking_blocks")
    if not isinstance(thinking_blocks, list):
        return []
    return [block for block in thinking_blocks if isinstance(block, dict)]


def _normalize_base64_text(value: str) -> str:
    return "".join(value.split())


def _decode_base64_bytes(value: str) -> bytes:
    normalized_value = _normalize_base64_text(value)
    padding = (-len(normalized_value)) % 4
    if padding:
        normalized_value += "=" * padding
    return base64.b64decode(normalized_value)


def _short_hash(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()[:12]


def _format_langfuse_span_timestamp(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _append_langfuse_span(
    metadata: Dict[str, Any],
    *,
    name: str,
    span_metadata: Optional[Dict[str, Any]] = None,
    input_data: Any = None,
    output_data: Any = None,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
) -> None:
    existing_spans = metadata.get("langfuse_spans") or []
    if not isinstance(existing_spans, list):
        existing_spans = []

    span_descriptor: Dict[str, Any] = {"name": name}
    if input_data is not None:
        span_descriptor["input"] = input_data
    if output_data is not None:
        span_descriptor["output"] = output_data
    if span_metadata:
        span_descriptor["metadata"] = span_metadata
    if start_time is not None:
        span_descriptor["start_time"] = _format_langfuse_span_timestamp(start_time)
    if end_time is not None:
        span_descriptor["end_time"] = _format_langfuse_span_timestamp(end_time)

    existing_spans.append(span_descriptor)
    metadata["langfuse_spans"] = existing_spans


def _maybe_parse_json_text(value: str) -> Any:
    stripped_value = value.strip()
    if not stripped_value or stripped_value[0] not in "[{":
        return None
    try:
        return json.loads(stripped_value)
    except (TypeError, ValueError):
        return None


def _permission_check_probeable_value(value: Any) -> bool:
    """True when *value* is a concrete response-shaped container we should walk.

    Restricts attribute probing to dicts and objects that already expose the
    known fields, so free-form getattr on test doubles / arbitrary objects is
    not required in production code.
    """
    if isinstance(value, (str, list, dict)):
        return True
    if value is None or isinstance(value, (bool, int, float, bytes)):
        return False
    for key in ("content", "choices", "response", "message"):
        try:
            if isinstance(value, dict) and key in value:
                return True
            obj_dict = getattr(value, "__dict__", None)
            if isinstance(obj_dict, dict) and key in obj_dict:
                return True
        except Exception:
            continue
    return False


def _extract_claude_permission_check_decision_from_value(
    value: Any,
    *,
    _depth: int = 0,
) -> Optional[str]:
    if value is None or _depth > 8:
        return None

    if isinstance(value, str):
        stripped_value = value.strip()
        match = _CLAUDE_PERMISSION_CHECK_OUTPUT_RE.match(stripped_value)
        if match is not None:
            return match.group("decision").lower()
        parsed_value = _maybe_parse_json_text(stripped_value)
        if parsed_value is not None:
            return _extract_claude_permission_check_decision_from_value(parsed_value, _depth=_depth + 1)
        return None

    if isinstance(value, list):
        text_value = _content_to_text(value).strip()
        match = _CLAUDE_PERMISSION_CHECK_OUTPUT_RE.match(text_value)
        if match is not None:
            return match.group("decision").lower()
        for item in value:
            decision = _extract_claude_permission_check_decision_from_value(item, _depth=_depth + 1)
            if decision is not None:
                return decision
        return None

    if not _permission_check_probeable_value(value):
        return None

    content = _maybe_get(value, "content")
    if content is not None and content is not value:
        decision = _extract_claude_permission_check_decision_from_value(content, _depth=_depth + 1)
        if decision is not None:
            return decision

    message = _extract_first_response_message(value)
    if message is not None and message is not value:
        decision = _extract_claude_permission_check_decision_from_value(message, _depth=_depth + 1)
        if decision is not None:
            return decision

    response = _maybe_get(value, "response")
    if response is not None and response is not value:
        decision = _extract_claude_permission_check_decision_from_value(response, _depth=_depth + 1)
        if decision is not None:
            return decision

    return None


def _extract_claude_permission_check_decision(
    result: Any,
    standard_logging_object: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    decision = _extract_claude_permission_check_decision_from_value(result)
    if decision is not None:
        return decision

    if isinstance(standard_logging_object, dict):
        for candidate in (
            standard_logging_object.get("response"),
            standard_logging_object.get("output"),
        ):
            decision = _extract_claude_permission_check_decision_from_value(candidate)
            if decision is not None:
                return decision

    return None


def _extract_claude_permission_check_models(
    kwargs: Dict[str, Any],
    standard_logging_object: Dict[str, Any],
    metadata: Dict[str, Any],
    result: Any,
) -> Tuple[Optional[str], Optional[str]]:
    request_model = _first_non_empty_string(
        _maybe_get_path(kwargs.get("passthrough_logging_payload"), "request_body", "model"),
        _maybe_get_path(
            kwargs.get("litellm_params"),
            "proxy_server_request",
            "body",
            "model",
        ),
        _maybe_get_path(standard_logging_object, "request_body", "model"),
    )
    response_model = _first_non_empty_string(
        _maybe_get(result, "model"),
        _maybe_get_path(standard_logging_object, "response", "model"),
        standard_logging_object.get("model"),
        kwargs.get("model"),
        metadata.get("model"),
    )
    return request_model, response_model


def _enrich_claude_permission_check_metadata(
    kwargs: Dict[str, Any],
    metadata: Dict[str, Any],
    result: Any,
    *,
    standard_logging_object: Optional[Dict[str, Any]] = None,
) -> None:
    standard_logging_object = standard_logging_object or kwargs.get("standard_logging_object") or {}
    decision = _extract_claude_permission_check_decision(
        result,
        standard_logging_object=standard_logging_object,
    )
    if decision is None:
        return

    blocked = decision == "yes"
    request_model, response_model = _extract_claude_permission_check_models(
        kwargs,
        standard_logging_object,
        metadata,
        result,
    )

    metadata["claude_internal_check"] = True
    metadata["claude_internal_check_type"] = "permission_check"
    metadata["claude_permission_check"] = True
    metadata["claude_permission_check_decision"] = decision
    metadata["claude_permission_check_blocked"] = blocked
    if request_model:
        metadata["claude_permission_check_request_model"] = request_model
    if response_model:
        metadata["claude_permission_check_response_model"] = response_model

    _merge_tags(
        metadata,
        [
            "claude-internal-check",
            "claude-permission-check",
            f"claude-permission-check:{decision}",
            "claude-permission-check:block" if blocked else "claude-permission-check:allow",
        ],
    )

    existing_spans = metadata.get("langfuse_spans") or []
    if not isinstance(existing_spans, list):
        existing_spans = []
    if any(isinstance(span, dict) and span.get("name") == "claude.permission_check" for span in existing_spans):
        return

    span_metadata: Dict[str, Any] = {
        "decision": decision,
        "blocked": blocked,
        "source": "claude_code_block_output",
    }
    for key in (
        "cc_version",
        "cc_entrypoint",
        "client_name",
        "client_version",
        "litellm_environment",
    ):
        value = metadata.get(key)
        if value is not None:
            span_metadata[key] = value
    if request_model:
        span_metadata["request_model"] = request_model
    if response_model:
        span_metadata["response_model"] = response_model

    now = datetime.now(timezone.utc)
    _append_langfuse_span(
        metadata,
        name="claude.permission_check",
        span_metadata=span_metadata,
        input_data={"check_type": "permission_check"},
        output_data={"decision": decision, "blocked": blocked},
        start_time=now,
        end_time=now,
    )


def _metadata_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def _metadata_request_tags(metadata: Dict[str, Any]) -> List[str]:
    request_tags = metadata.get("request_tags")
    tags = metadata.get("tags")
    merged: List[str] = []
    for source in (request_tags, tags):
        if not isinstance(source, list):
            continue
        for tag in source:
            if isinstance(tag, str) and tag.strip() and tag not in merged:
                merged.append(tag)
    return merged


def _is_claude_permission_check_metadata(metadata: Any) -> bool:
    if not isinstance(metadata, dict):
        return False
    if _metadata_bool(metadata.get("claude_permission_check")):
        return True
    for tag in _metadata_request_tags(metadata):
        if tag == "claude-permission-check" or tag.startswith("claude-permission-check:"):
            return True
    return False


def _extract_claude_project_from_metadata_tags(
    metadata: Dict[str, Any],
) -> Optional[str]:
    for tag in _metadata_request_tags(metadata):
        if not tag.startswith("claude-project:"):
            continue
        repository = _normalize_repository_identity(tag.split(":", 1)[1])
        if repository:
            return repository
    return None


def _extract_claude_auto_review_source_model(
    metadata: Dict[str, Any],
    fallback_model: Optional[str] = None,
) -> Optional[str]:
    return _first_non_empty_string(
        metadata.get("source_model"),
        metadata.get("claude_permission_check_response_model"),
        metadata.get("claude_permission_check_request_model"),
        fallback_model,
    )


def _apply_claude_auto_review_metadata(
    metadata: Dict[str, Any],
    *,
    repository: Optional[str] = None,
    tenant_id: Optional[str] = None,
    source_model: Optional[str] = None,
) -> None:
    metadata["trace_name"] = _CLAUDE_AUTO_REVIEW_TRACE_NAME
    metadata["agent_name"] = _CLAUDE_AUTO_REVIEW_AGENT_NAME
    metadata["aawm_claude_agent_name"] = _CLAUDE_AUTO_REVIEW_AGENT_NAME
    metadata["logical_model"] = _CLAUDE_AUTO_REVIEW_LOGICAL_MODEL

    resolved_source_model = _extract_claude_auto_review_source_model(
        metadata,
        source_model,
    )
    if resolved_source_model and resolved_source_model != _CLAUDE_AUTO_REVIEW_LOGICAL_MODEL:
        metadata["source_model"] = resolved_source_model

    normalized_repository = _normalize_repository_identity(repository)
    normalized_tenant = _normalize_repository_identity(tenant_id)
    inherited_identity = normalized_repository or normalized_tenant
    if inherited_identity:
        metadata["repository"] = inherited_identity
        metadata["tenant_id"] = inherited_identity
        metadata["aawm_tenant_id"] = inherited_identity
        metadata["aawm_claude_project"] = inherited_identity
        metadata["trace_user_id"] = inherited_identity

    tags_to_add = [
        "claude-internal-check",
        "claude-permission-check",
        f"claude-agent:{_CLAUDE_AUTO_REVIEW_AGENT_NAME}",
    ]
    if inherited_identity:
        tags_to_add.append(f"claude-project:{inherited_identity}")
    _merge_tags(metadata, tags_to_add)
    existing_request_tags = metadata.get("request_tags") or []
    if not isinstance(existing_request_tags, list):
        existing_request_tags = []
    merged_request_tags = list(existing_request_tags)
    for tag in tags_to_add:
        if tag and tag not in merged_request_tags:
            merged_request_tags.append(tag)
    metadata["request_tags"] = merged_request_tags


def _apply_claude_auto_review_identity_to_record(record: Dict[str, Any]) -> None:
    metadata = record.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
    else:
        metadata = dict(metadata)
    if not _is_claude_permission_check_metadata(metadata):
        return

    source_model = _extract_claude_auto_review_source_model(
        metadata,
        _clean_non_empty_string(record.get("model")),
    )
    repository = _normalize_repository_identity(record.get("repository"))
    tenant_id = _normalize_repository_identity(record.get("tenant_id"))
    if repository is None:
        repository = _extract_claude_project_from_metadata_tags(metadata)
    if tenant_id is None:
        tenant_id = repository

    _apply_claude_auto_review_metadata(
        metadata,
        repository=repository,
        tenant_id=tenant_id,
        source_model=source_model,
    )
    record["metadata"] = metadata
    record["model"] = _CLAUDE_AUTO_REVIEW_LOGICAL_MODEL
    record["agent_name"] = _CLAUDE_AUTO_REVIEW_AGENT_NAME
    if repository is not None:
        record["repository"] = repository
    resolved_tenant = tenant_id or repository
    if resolved_tenant is not None:
        record["tenant_id"] = resolved_tenant


def _extract_claude_auto_review_identity_from_row(
    row: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    metadata = row.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}

    repository = (
        _normalize_repository_identity(row.get("repository"))
        or _extract_claude_project_from_metadata_tags(metadata)
        or _normalize_repository_identity(metadata.get("aawm_claude_project"))
        or _normalize_repository_identity(metadata.get("repository"))
        or _normalize_repository_identity(row.get("tenant_id"))
        or _normalize_repository_identity(metadata.get("tenant_id"))
    )
    if not repository:
        return None

    return {
        "repository": repository,
        "tenant_id": repository,
        "source_row_id": row.get("id"),
        "source": "same_session.session_history",
    }


def _apply_claude_auto_review_parent_identity(
    payload: Dict[str, Any],
    identity: Dict[str, Any],
) -> None:
    repository = _normalize_repository_identity(identity.get("repository"))
    if not repository:
        return

    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
    else:
        metadata = dict(metadata)

    payload["repository"] = repository
    payload["tenant_id"] = repository
    _apply_claude_auto_review_metadata(
        metadata,
        repository=repository,
        tenant_id=repository,
        source_model=_extract_claude_auto_review_source_model(
            metadata,
            _clean_non_empty_string(payload.get("model")),
        ),
    )
    metadata["claude_auto_review_parent_identity_source"] = identity.get("source")
    if identity.get("source_row_id") is not None:
        metadata["claude_auto_review_parent_identity_source_row_id"] = identity["source_row_id"]
    payload["metadata"] = metadata


def _build_session_identity_cache(
    records: List[Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    identity_by_session: Dict[str, Dict[str, Any]] = {}
    for record in records:
        if record.get("_skip_session_history"):
            continue
        session_id = _clean_non_empty_string(record.get("session_id"))
        if not session_id:
            continue
        metadata = record.get("metadata")
        if _is_claude_permission_check_metadata(metadata):
            continue
        identity = _extract_claude_auto_review_identity_from_row(record)
        if identity:
            identity_by_session[session_id] = identity
    return identity_by_session


def _build_permission_usage_fields(
    *,
    metadata: Dict[str, Any],
    prompt_tokens: Optional[int],
    completion_tokens: Optional[int],
    response_cost_usd: Optional[float],
) -> Dict[str, Any]:
    if not _metadata_bool(metadata.get("claude_permission_check")):
        return {
            "token_permission_input": 0,
            "token_permission_output": 0,
            "permission_usd_cost": 0.0,
        }

    return {
        "token_permission_input": _safe_int(prompt_tokens) or 0,
        "token_permission_output": _safe_int(completion_tokens) or 0,
        "permission_usd_cost": _safe_float(response_cost_usd) or 0.0,
    }


def _safe_int(value: Any) -> Optional[int]:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _safe_float(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_str(value: Any) -> Optional[str]:
    """Shared string coercion for backfill/repair scripts and identity helpers."""
    return _clean_non_empty_string(value)


def _first_non_none(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _first_reported_openrouter_cost(
    metadata: Dict[str, Any],
    usage_dict: Dict[str, Any],
) -> Optional[float]:
    response_cost = _safe_float(
        _first_non_none(
            metadata.get("usage_openrouter_cost"),
            usage_dict.get("cost"),
        )
    )
    if response_cost is None or response_cost < 0:
        return None
    return response_cost


def _safe_json_load(value: Any, default: Any) -> Any:
    if value is None or value == "":
        return default
    if isinstance(value, (dict, list)):
        return value
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return default
    return default


def _normalize_datetime(value: Any) -> Optional[datetime]:
    if not isinstance(value, datetime):
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _parse_datetime_value(value: Any) -> Optional[datetime]:
    if isinstance(value, datetime):
        return _normalize_datetime(value)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            normalized = stripped.replace("Z", "+00:00")
            return _normalize_datetime(datetime.fromisoformat(normalized))
        except ValueError:
            return None
    return None


def _nonnegative_float_or_none(value: Any) -> Optional[float]:
    normalized = _safe_float(value)
    if normalized is None or not math.isfinite(normalized) or normalized < 0:
        return None
    return round(normalized, 3)


def _sum_nonnegative_floats(*values: Optional[float]) -> Optional[float]:
    present_values = [value for value in values if value is not None]
    if not present_values:
        return None
    return round(sum(present_values), 3)


def _coerce_session_latency_datetime(value: Any) -> Optional[datetime]:
    if isinstance(value, str):
        return _parse_datetime_value(value)
    return _normalize_datetime(value)


def _elapsed_ms_from_times(start_time: Any, end_time: Any) -> Optional[float]:
    normalized_start = _coerce_session_latency_datetime(start_time)
    normalized_end = _coerce_session_latency_datetime(end_time)
    if normalized_start is None or normalized_end is None:
        return None
    return _nonnegative_float_or_none((normalized_end - normalized_start).total_seconds() * 1000.0)


def _metadata_nonnegative_float(
    metadata: Dict[str, Any],
    key: str,
) -> Optional[float]:
    return _nonnegative_float_or_none(metadata.get(key))


def _build_session_history_latency_breakdown(
    *,
    metadata: Any,
    start_time: Any,
    end_time: Any,
) -> Dict[str, Optional[float]]:
    if not isinstance(metadata, dict):
        metadata = _safe_json_load(metadata, {})
    if not isinstance(metadata, dict):
        metadata = {}

    litellm_pre_send_ms = _metadata_nonnegative_float(
        metadata,
        "aawm_local_prepare_ms",
    )
    litellm_post_response_ms = _first_non_none(
        _metadata_nonnegative_float(metadata, "aawm_local_finalize_ms"),
        _metadata_nonnegative_float(metadata, "aawm_local_stream_finalize_ms"),
    )
    litellm_processing_ms = _sum_nonnegative_floats(
        litellm_pre_send_ms,
        litellm_post_response_ms,
    )
    if litellm_processing_ms is None:
        litellm_processing_ms = _metadata_nonnegative_float(
            metadata,
            "aawm_total_proxy_overhead_ms",
        )

    upstream_first_chunk_ms = _metadata_nonnegative_float(
        metadata,
        "aawm_upstream_first_chunk_ms",
    )
    upstream_stream_complete_ms = _metadata_nonnegative_float(
        metadata,
        "aawm_upstream_stream_complete_ms",
    )
    upstream_wait_ms = _metadata_nonnegative_float(
        metadata,
        "aawm_upstream_wait_ms",
    )
    llm_upstream_elapsed_ms = _first_non_none(
        upstream_stream_complete_ms,
        upstream_wait_ms,
    )
    llm_upstream_time_to_first_byte_ms = upstream_first_chunk_ms
    llm_upstream_stream_ms = None
    if upstream_first_chunk_ms is not None and upstream_stream_complete_ms is not None:
        llm_upstream_stream_ms = _nonnegative_float_or_none(upstream_stream_complete_ms - upstream_first_chunk_ms)

    total_server_elapsed_ms = _first_non_none(
        _metadata_nonnegative_float(metadata, "aawm_total_proxy_duration_ms"),
        _elapsed_ms_from_times(start_time, end_time),
    )
    ttft_ms = _first_non_none(
        _metadata_nonnegative_float(metadata, "aawm_time_to_first_token_ms"),
        _metadata_nonnegative_float(metadata, "aawm_first_emitted_chunk_ms"),
    )

    latency_unclassified_ms = None
    if total_server_elapsed_ms is not None and (
        litellm_processing_ms is not None or llm_upstream_elapsed_ms is not None
    ):
        classified_ms = (litellm_processing_ms or 0.0) + (llm_upstream_elapsed_ms or 0.0)
        latency_unclassified_ms = _nonnegative_float_or_none(total_server_elapsed_ms - classified_ms)
        if latency_unclassified_ms is None:
            latency_unclassified_ms = 0.0

    return {
        "litellm_processing_ms": litellm_processing_ms,
        "llm_upstream_elapsed_ms": llm_upstream_elapsed_ms,
        "total_server_elapsed_ms": total_server_elapsed_ms,
        "ttft_ms": ttft_ms,
        "litellm_pre_send_ms": litellm_pre_send_ms,
        "litellm_post_response_ms": litellm_post_response_ms,
        "llm_upstream_time_to_first_byte_ms": llm_upstream_time_to_first_byte_ms,
        "llm_upstream_stream_ms": llm_upstream_stream_ms,
        "latency_unclassified_ms": latency_unclassified_ms,
    }


_AAWM_RATE_LIMIT_METADATA_KEYS = (
    "trace_name",
    "litellm_environment",
    "client_name",
    "client_version",
    "repository",
    "passthrough_route_family",
    "route_family",
    "auth_mode",
    "credential_family",
    "xai_oauth_managed",
    "xai_oauth_public_model",
    "xai_oauth_upstream_model",
    "xai_quota_family",
    "shared_quota_family",
)
_AAWM_RATE_LIMIT_MEANINGFUL_PERCENT_DROP = 1.0
_AAWM_RATE_LIMIT_MEANINGFUL_RESET_SHIFT = timedelta(minutes=15)
_AAWM_RATE_LIMIT_STALE_RESET_TOLERANCE = timedelta(minutes=15)
_AAWM_OPENROUTER_FREE_DAILY_REQUEST_LIMIT_DEFAULT = 1000
_AAWM_OPENROUTER_FREE_DAILY_SOURCE = "openrouter_free_daily_local_meter"
_AAWM_RATE_LIMIT_SNAPSHOT_FIELDS = (
    "provider_resets_at",
    "used_percentage",
    "remaining_requests",
    "used_requests",
    "total_requests",
    "status",
    "exhausted",
    "exhaustion_kind",
    "reset_hint_seconds",
)


async def _build_openrouter_free_daily_observations_for_records(
    conn: Any,
    records: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    latest_record_by_window: Dict[
        Tuple[datetime, datetime],
        Tuple[datetime, Dict[str, Any]],
    ] = {}
    for record in records:
        if record.get("_skip_session_history"):
            continue
        if not _is_openrouter_free_session_history_record(record):
            continue
        observed_at = _openrouter_free_record_observed_at(record)
        window = _openrouter_free_daily_window(observed_at)
        previous = latest_record_by_window.get(window)
        if previous is None or observed_at >= previous[0]:
            latest_record_by_window[window] = (observed_at, record)

    if not latest_record_by_window:
        return []

    total_requests = _openrouter_free_daily_request_limit()
    observations: List[Dict[str, Any]] = []
    for (day_start, day_end), (observed_at, record) in sorted(
        latest_record_by_window.items(),
        key=lambda item: item[0][0],
    ):
        used_requests = _safe_int(
            await conn.fetchval(
                _AAWM_OPENROUTER_FREE_DAILY_REQUEST_COUNT_SQL,
                day_start,
                day_end,
            )
        )
        if used_requests is None:
            continue
        observations.append(
            _build_openrouter_free_daily_observation(
                context=_openrouter_free_daily_observation_context_from_record(
                    record,
                    observed_at,
                ),
                day_start=day_start,
                day_end=day_end,
                used_requests=used_requests,
                total_requests=total_requests,
                signal="local_session_history_openrouter_free_count",
                status=("quota_exhausted" if used_requests >= total_requests else "observed"),
                exhausted=used_requests >= total_requests,
            )
        )
    return observations


_AAWM_RATE_LIMIT_CONTEXT_CACHE_KEY = "_aawm_rate_limit_context_cache"


USAGE_PERIOD_TYPE_WEEKLY = "USAGE_PERIOD_TYPE_WEEKLY"
GROK_BILLING_WEEKLY_CREDITS_QUOTA_KEY = "xai_grok_build_weekly_credits:credits"
GROK_BILLING_MONTHLY_REQUESTS_QUOTA_KEY = "xai_grok_build_monthly_requests:requests"
GROK_BILLING_MONTHLY_CREDITS_QUOTA_KEY = "xai_grok_build_monthly_credits:credits"


_AAWM_EMBEDDED_JSON_MAX_SUCCESS = 20
_AAWM_EMBEDDED_JSON_MAX_ATTEMPTS = 64
_AAWM_EMBEDDED_JSON_SCAN_CHARS = 20000


_LITELLM_PROVIDER_ERROR_MODEL_GROUP_RE = re.compile(r"Received Model Group=(?P<model_group>[^\n\r]+)")
_LITELLM_PROVIDER_ERROR_FALLBACKS_RE = re.compile(
    r"Available Model Group Fallbacks=(?P<fallbacks>.*?)(?:\s+LiteLLM Retried:|$)",
    re.DOTALL,
)
_LITELLM_PROVIDER_ERROR_RETRIES_RE = re.compile(
    r"LiteLLM Retried:\s*(?P<retry_count>\d+)\s*times,\s*" r"LiteLLM Max Retries:\s*(?P<max_retries>\d+)"
)


_UPSTREAM_ERROR_SECRET_RE = re.compile(
    r"(?is)(?P<label>authorization|x-api-key|api[-_]?key|bearer|token|secret|password)"
    r"(?P<sep>\s*[:=]\s*|\s+)"
    r"(?P<value>(?:bearer\s+)?[^\s,\"'}{]{6,})",
)


def _build_alias_routing_audit_only_record(
    *,
    events: List[Dict[str, Any]],
    session_id: Optional[str] = None,
    litellm_call_id: Optional[str] = None,
    model: Optional[str] = None,
    provider: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build an audit-only record that skips session_history inserts.

    Terminal Codex/Anthropic auto-agent no-candidate and redispatch 429 paths can
    finish without a normal session_history write. This record still persists
    ``aawm_alias_routing_audit`` events best-effort while avoiding a duplicate
    session_history row or a normal success/fallback double-write.
    """
    normalized_events = [event for event in events if isinstance(event, dict)]
    primary = normalized_events[-1] if normalized_events else {}
    record_metadata: Dict[str, Any] = {
        "aawm_alias_routing_audit_only": True,
        "aawm_alias_routing_audit_events": normalized_events,
    }
    if isinstance(metadata, dict):
        for key, value in metadata.items():
            if value is not None and key not in record_metadata:
                record_metadata[key] = value
    # Promote direct event context IDs for durable audit payload builders.
    for key in (
        "session_id",
        "session_key",
        "trace_id",
        "litellm_call_id",
        "agent_id",
        "agent_name",
        "agent_role",
        "agent_profile",
        "thread_source",
        "dispatch_id",
        "redispatch_ordinal",
        "cooldown_state_source",
        "terminal_activity_status",
        "actual_prior_tool_activity_summary",
        "repository",
        "alias_model",
        "alias_family",
    ):
        value = primary.get(key)
        if value is not None and key not in record_metadata:
            record_metadata[key] = value
    return {
        "_skip_session_history": True,
        "litellm_call_id": litellm_call_id or primary.get("litellm_call_id"),
        "session_id": session_id or primary.get("session_id"),
        "model": model or primary.get("model") or primary.get("alias_model"),
        "provider": provider or primary.get("provider"),
        "aawm_alias_routing_audit_events": normalized_events,
        "metadata": record_metadata,
    }


# _build_provider_error_observation_only_record moved to litellm.integrations.aawm_session_history.record
# _build_structured_output_failure_session_history_record moved to litellm.integrations.aawm_session_history.record
# _build_failure_observation_only_record moved to litellm.integrations.aawm_session_history.record
def _classify_rate_limit_transition(
    previous: Dict[str, Any],
    current: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    signals: List[str] = []
    transition_type: Optional[str] = None
    confidence = 0.0

    previous_reset = _parse_provider_timestamp(previous.get("provider_resets_at"))
    current_reset = _parse_provider_timestamp(current.get("provider_resets_at"))
    previous_observed = _parse_provider_timestamp(previous.get("observed_at"))
    current_observed = _parse_provider_timestamp(current.get("observed_at"))
    if previous_reset is not None and current_reset is not None:
        if abs((current_reset - previous_reset).total_seconds()) > 1:
            signals.append("resets_at_change")
            confidence = 0.95
            if previous_reset and current_observed and current_observed >= previous_reset - timedelta(minutes=2):
                transition_type = "expected_rollover"
            elif current_reset < previous_reset:
                transition_type = "capacity_grant_or_random_reset"
            else:
                transition_type = "early_provider_reset"

    previous_used_requests = _safe_int(previous.get("used_requests"))
    current_used_requests = _safe_int(current.get("used_requests"))
    if (
        previous_used_requests is not None
        and current_used_requests is not None
        and current_used_requests < previous_used_requests
    ):
        signals.append("counter_drop")
        transition_type = transition_type or "counter_drop_reset"
        confidence = max(confidence, 0.9)

    previous_remaining = _safe_int(previous.get("remaining_requests"))
    current_remaining = _safe_int(current.get("remaining_requests"))
    previous_total = _safe_int(previous.get("total_requests"))
    current_total = _safe_int(current.get("total_requests"))
    if (
        previous_remaining is not None
        and current_remaining is not None
        and current_remaining > previous_remaining
        and (previous_total is None or current_total is None or previous_total == current_total)
    ):
        signals.append("counter_drop")
        transition_type = transition_type or "counter_drop_reset"
        confidence = max(confidence, 0.85)

    previous_used_percentage = _safe_float(previous.get("used_percentage"))
    current_used_percentage = _safe_float(current.get("used_percentage"))
    if (
        previous_used_percentage is not None
        and current_used_percentage is not None
        and previous_used_percentage - current_used_percentage >= _AAWM_RATE_LIMIT_MEANINGFUL_PERCENT_DROP
    ):
        signals.append("usage_percent_drop")
        transition_type = transition_type or "usage_percent_drop"
        confidence = max(confidence, 0.75)

    if previous_total is not None and current_total is not None and previous_total != current_total:
        signals.append("limit_change")
        transition_type = transition_type or "policy_change"
        confidence = max(confidence, 0.65)

    if bool(previous.get("exhausted")) and not bool(current.get("exhausted")):
        signals.append("success_after_exhaustion")
        transition_type = transition_type or "capacity_grant_or_random_reset"
        confidence = max(confidence, 0.7)

    if not transition_type or not signals:
        return None

    return {
        "transition_type": transition_type,
        "signals": sorted(set(signals)),
        "confidence": confidence,
        "previous_observed_at": previous_observed,
        "current_observed_at": current_observed,
    }


def _rate_limit_observation_json(observation: Dict[str, Any]) -> Dict[str, Any]:
    return {
        key: _json_safe_rate_limit_value(value) for key, value in list(observation.items()) if key not in {"metadata"}
    }


def _build_rate_limit_transition(
    previous: Dict[str, Any],
    current: Dict[str, Any],
    classification: Dict[str, Any],
) -> Dict[str, Any]:
    transition_material = "|".join(
        str(value or "")
        for value in (
            current.get("limit_key"),
            previous.get("observed_at"),
            current.get("observed_at"),
            classification.get("transition_type"),
            ",".join(classification.get("signals") or []),
            current.get("provider_resets_at"),
        )
    )
    transition_key = "rlt_" + _short_hash(transition_material.encode("utf-8"))
    return {
        "transition_key": transition_key,
        "limit_key": current.get("limit_key"),
        "provider": current.get("provider"),
        "client_family": current.get("client_family"),
        "account_hash": current.get("account_hash"),
        "transition_type": classification["transition_type"],
        "confidence": classification["confidence"],
        "signals": classification["signals"],
        "source": current.get("source"),
        "old_observed_at": _parse_provider_timestamp(previous.get("observed_at")),
        "new_observed_at": _parse_provider_timestamp(current.get("observed_at")),
        "old_provider_resets_at": _parse_provider_timestamp(previous.get("provider_resets_at")),
        "new_provider_resets_at": _parse_provider_timestamp(current.get("provider_resets_at")),
        "old_used_percentage": _safe_float(previous.get("used_percentage")),
        "new_used_percentage": _safe_float(current.get("used_percentage")),
        "old_remaining_requests": _safe_int(previous.get("remaining_requests")),
        "new_remaining_requests": _safe_int(current.get("remaining_requests")),
        "old_used_requests": _safe_int(previous.get("used_requests")),
        "new_used_requests": _safe_int(current.get("used_requests")),
        "old_total_requests": _safe_int(previous.get("total_requests")),
        "new_total_requests": _safe_int(current.get("total_requests")),
        "inferred_window_start_at": _parse_provider_timestamp(current.get("inferred_window_start_at")),
        "detection_window_start_at": _parse_provider_timestamp(previous.get("observed_at")),
        "detection_window_end_at": _parse_provider_timestamp(current.get("observed_at")),
        "session_usage_summary": {},
        "old_observation": _rate_limit_observation_json(previous),
        "new_observation": _rate_limit_observation_json(current),
        "metadata": {
            "transition_basis": "adjacent_observation_compare",
            "meaningful_percent_drop_threshold": _AAWM_RATE_LIMIT_MEANINGFUL_PERCENT_DROP,
        },
    }


_AAWM_RESPONSES_CHUNKS_LITERAL_MAX_CHARS = 8192


def _extract_responses_completed_payload_from_passthrough_fallback_text(
    response_text: Any,
) -> Optional[Dict[str, Any]]:
    if not isinstance(response_text, str) or "Chunks=" not in response_text:
        return None

    chunks_text = response_text.split("Chunks=", 1)[1].strip()
    # Fail closed on oversized provider/passthrough text before literal_eval.
    if len(chunks_text) > _AAWM_RESPONSES_CHUNKS_LITERAL_MAX_CHARS:
        return None
    try:
        # Prefer JSON when the chunk envelope is JSON-shaped.
        if chunks_text[:1] in "[{":
            try:
                chunks = json.loads(chunks_text)
            except Exception:
                chunks = ast.literal_eval(chunks_text)
        else:
            chunks = ast.literal_eval(chunks_text)
    except Exception:
        return None
    if not isinstance(chunks, list):
        return None

    if BaseModelResponseIterator is None:
        return None

    completed_response = None
    output_text_parts: List[str] = []
    for chunk in chunks:
        if not isinstance(chunk, str):
            continue
        parsed_chunk = BaseModelResponseIterator._string_to_dict_parser(str_line=chunk)
        if not isinstance(parsed_chunk, dict):
            continue
        chunk_type = parsed_chunk.get("type")
        if chunk_type == "response.output_text.delta":
            delta = parsed_chunk.get("delta")
            if isinstance(delta, str):
                output_text_parts.append(delta)
        elif chunk_type == "response.completed":
            response_payload = parsed_chunk.get("response")
            if isinstance(response_payload, dict):
                completed_response = response_payload

    if not isinstance(completed_response, dict):
        return None

    return {
        "response": completed_response,
        "output_text": "".join(output_text_parts),
    }


def _build_usage_object_from_metadata(metadata: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not isinstance(metadata, dict):
        return None

    usage_object = metadata.get("usage_object")
    reconstructed: Dict[str, Any] = dict(usage_object) if isinstance(usage_object, dict) and usage_object else {}

    input_tokens = _safe_int(metadata.get("usage_input_tokens"))
    output_tokens = _safe_int(metadata.get("usage_output_tokens"))
    total_tokens = _safe_int(metadata.get("usage_total_tokens"))
    cache_read_input_tokens = _safe_int(metadata.get("usage_cache_read_input_tokens"))
    cache_creation_input_tokens = _safe_int(metadata.get("usage_cache_creation_input_tokens"))
    reasoning_tokens_reported = _safe_int(metadata.get("usage_reasoning_tokens_reported"))

    if not any(
        value is not None
        for value in (
            input_tokens,
            output_tokens,
            total_tokens,
            cache_read_input_tokens,
            cache_creation_input_tokens,
            reasoning_tokens_reported,
        )
    ):
        return reconstructed or None

    if input_tokens is not None:
        reconstructed["input_tokens"] = input_tokens
        reconstructed["prompt_tokens"] = input_tokens
    if output_tokens is not None:
        reconstructed["output_tokens"] = output_tokens
        reconstructed["completion_tokens"] = output_tokens
    if total_tokens is not None:
        reconstructed["total_tokens"] = total_tokens
    if cache_read_input_tokens is not None:
        reconstructed["cache_read_input_tokens"] = cache_read_input_tokens
        input_tokens_details = dict(reconstructed.get("input_tokens_details") or {})
        input_tokens_details["cached_tokens"] = cache_read_input_tokens
        reconstructed["input_tokens_details"] = input_tokens_details
    if cache_creation_input_tokens is not None:
        reconstructed["cache_creation_input_tokens"] = cache_creation_input_tokens
    if reasoning_tokens_reported is not None:
        reconstructed["reasoning_tokens"] = reasoning_tokens_reported
        output_tokens_details = dict(reconstructed.get("output_tokens_details") or {})
        output_tokens_details["reasoning_tokens"] = reasoning_tokens_reported
        reconstructed["output_tokens_details"] = output_tokens_details

    return reconstructed or None


def _build_usage_object_from_token_count_payload(
    output_payload: Any,
) -> Optional[Dict[str, Any]]:
    if isinstance(output_payload, str):
        parsed_payload = _maybe_parse_json_text(output_payload)
        if parsed_payload is None:
            return None
        return _build_usage_object_from_token_count_payload(parsed_payload)

    if not isinstance(output_payload, dict):
        return None

    input_tokens = _safe_int(
        _first_non_none(
            output_payload.get("prompt_tokens"),
            output_payload.get("input_tokens"),
            output_payload.get("inputTokens"),
        )
    )
    output_tokens = _safe_int(
        _first_non_none(
            output_payload.get("completion_tokens"),
            output_payload.get("output_tokens"),
            output_payload.get("outputTokens"),
        )
    )
    total_tokens = _safe_int(
        _first_non_none(
            output_payload.get("total_tokens"),
            output_payload.get("totalTokens"),
        )
    )
    # Only accept generic "total" when sibling token keys already establish this
    # as a token-count payload, not a pagination/billing envelope.
    if total_tokens is None and (input_tokens is not None or output_tokens is not None):
        total_tokens = _safe_int(output_payload.get("total"))

    if input_tokens is None and output_tokens is None and total_tokens is None:
        return None

    usage_object: Dict[str, Any] = {}
    usage_object["token_count_response"] = True
    if input_tokens is not None:
        usage_object["prompt_tokens"] = input_tokens
        usage_object["input_tokens"] = input_tokens
    if output_tokens is not None:
        usage_object["completion_tokens"] = output_tokens
        usage_object["output_tokens"] = output_tokens
    if total_tokens is None and (input_tokens is not None or output_tokens is not None):
        total_tokens = (input_tokens or 0) + (output_tokens or 0)
    if total_tokens is not None:
        usage_object["total_tokens"] = total_tokens

    return usage_object or None


def _extract_responses_completed_response_from_langfuse_output(
    output_payload: Any,
) -> Optional[Dict[str, Any]]:
    raw_text = output_payload
    if isinstance(output_payload, dict):
        if isinstance(output_payload.get("response"), dict):
            return output_payload["response"]
        if isinstance(output_payload.get("raw_output"), str):
            raw_text = output_payload["raw_output"]

    completed_payload = _extract_responses_completed_payload_from_passthrough_fallback_text(raw_text)
    if not isinstance(completed_payload, dict):
        return None
    response_payload = completed_payload.get("response")
    return response_payload if isinstance(response_payload, dict) else None


def _build_usage_object_from_langfuse_output(output_payload: Any) -> Optional[Dict[str, Any]]:
    if isinstance(output_payload, dict):
        usage = output_payload.get("usage")
        if isinstance(usage, dict) and usage:
            return dict(usage)

    token_count_usage = _build_usage_object_from_token_count_payload(output_payload)
    if token_count_usage is not None:
        return token_count_usage

    response_payload = _extract_responses_completed_response_from_langfuse_output(output_payload)
    if not isinstance(response_payload, dict):
        return None
    usage = response_payload.get("usage")
    return dict(usage) if isinstance(usage, dict) and usage else None


def _extract_codex_model_from_response_headers(metadata: Dict[str, Any]) -> Optional[str]:
    headers = metadata.get("codex_response_headers")
    if not isinstance(headers, dict):
        return None

    limit_name = _clean_non_empty_string(_get_rate_limit_header_value(headers, "x-codex-bengalfox-limit-name"))
    if not limit_name:
        return None

    normalized = re.sub(r"[^a-z0-9._-]+", "-", limit_name.lower()).strip("-")
    if normalized.startswith("gpt-") and "codex" in normalized:
        return normalized
    return None


def _session_history_metadata_model(metadata: Dict[str, Any]) -> Optional[str]:
    hidden_params = metadata.get("hidden_params")
    return _first_known_model_string(
        metadata.get("codex_auto_agent_selected_model"),
        metadata.get("anthropic_auto_agent_selected_model"),
        metadata.get("codex_adapter_model"),
        metadata.get("litellm_model"),
        _session_history_model_from_request_tags(metadata),
        metadata.get("model"),
        _maybe_get(hidden_params, "model"),
    )


_SESSION_HISTORY_CLAUDE_MODEL_TAG_RE = re.compile(
    r"^claude-(?:opus|sonnet|haiku)-[a-z0-9_.-]+$",
    re.IGNORECASE,
)


def _session_history_model_from_request_tags(
    metadata: Dict[str, Any],
) -> Optional[str]:
    for tag in _metadata_request_tags(metadata):
        if not isinstance(tag, str):
            continue
        stripped_tag = tag.strip()
        tag_lower = stripped_tag.lower()
        if not tag_lower.startswith("claude-exp:"):
            continue
        candidate = stripped_tag.split(":", 1)[1].strip()
        if _SESSION_HISTORY_CLAUDE_MODEL_TAG_RE.fullmatch(candidate):
            return candidate
    return None


def _extract_model_from_langfuse_input(input_payload: Any) -> Optional[str]:
    request_body = _extract_request_body_from_langfuse_input(input_payload)
    if not isinstance(request_body, dict):
        return None
    body = request_body.get("body")
    return _first_known_model_string(
        request_body.get("model"),
        _maybe_get(body, "model"),
    )


def _extract_model_from_langfuse_output(output_payload: Any) -> Optional[str]:
    if isinstance(output_payload, dict):
        model = output_payload.get("model")
        if isinstance(model, str) and model.strip():
            return model.strip()

    response_payload = _extract_responses_completed_response_from_langfuse_output(output_payload)
    model = _maybe_get(response_payload, "model")
    if isinstance(model, str) and model.strip():
        return model.strip()
    return None


def _first_known_model_string(*candidates: Any) -> Optional[str]:
    for candidate in candidates:
        if not isinstance(candidate, str):
            continue
        cleaned = candidate.strip()
        if not cleaned or cleaned.lower() in {"unknown", "none", "null"}:
            continue
        return cleaned
    return None


def _first_explicit_openrouter_model_string(*candidates: Any) -> Optional[str]:
    for candidate in candidates:
        if not isinstance(candidate, str):
            continue
        cleaned = candidate.strip()
        if cleaned.lower().startswith("openrouter/") and len(cleaned) > len("openrouter/"):
            return cleaned
    return None


def _coerce_usage_object_to_dict(usage_obj: Any) -> Optional[Dict[str, Any]]:
    if isinstance(usage_obj, dict):
        return dict(usage_obj)

    model_dump = getattr(usage_obj, "model_dump", None)
    if callable(model_dump):
        try:
            dumped = model_dump(exclude_none=True)
        except TypeError:
            dumped = model_dump()
        if isinstance(dumped, dict):
            return dumped

    dict_method = getattr(usage_obj, "dict", None)
    if callable(dict_method):
        try:
            dumped = dict_method(exclude_none=True)
        except TypeError:
            dumped = dict_method()
        if isinstance(dumped, dict):
            return dumped

    return None


def _extract_metadata_usage_object(kwargs: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    standard_logging_object = kwargs.get("standard_logging_object")
    if isinstance(standard_logging_object, dict):
        metadata = standard_logging_object.get("metadata")
        if isinstance(metadata, dict):
            usage_object = metadata.get("usage_object")
            if isinstance(usage_object, dict) and usage_object:
                return dict(usage_object)
            reconstructed_usage = _build_usage_object_from_metadata(metadata)
            if reconstructed_usage is not None:
                return reconstructed_usage

    litellm_params = kwargs.get("litellm_params")
    if isinstance(litellm_params, dict):
        metadata = litellm_params.get("metadata")
        if isinstance(metadata, dict):
            usage_object = metadata.get("usage_object")
            if isinstance(usage_object, dict) and usage_object:
                return dict(usage_object)
            reconstructed_usage = _build_usage_object_from_metadata(metadata)
            if reconstructed_usage is not None:
                return reconstructed_usage

    return None


def _merge_usage_object_with_metadata(
    usage_obj: Any,
    metadata_usage_object: Optional[Dict[str, Any]],
) -> Any:
    if metadata_usage_object is None:
        return usage_obj

    usage_dict = _coerce_usage_object_to_dict(usage_obj)
    if usage_dict is None:
        return metadata_usage_object

    merged_usage = dict(usage_dict)
    for key, value in list(metadata_usage_object.items()):
        if key not in merged_usage or merged_usage.get(key) in (None, {}, []):
            merged_usage[key] = value

    return merged_usage


def _extract_usage_object(kwargs: Dict[str, Any], result: Any) -> Any:
    usage_obj = _maybe_get(result, "usage")
    metadata_usage_object = _extract_metadata_usage_object(kwargs)
    if usage_obj is not None:
        return _merge_usage_object_with_metadata(usage_obj, metadata_usage_object)

    token_count_usage = _build_usage_object_from_token_count_payload(result)
    if token_count_usage is not None:
        return _merge_usage_object_with_metadata(
            token_count_usage,
            metadata_usage_object,
        )
    token_count_usage = _build_usage_object_from_token_count_payload(_maybe_get(result, "response"))
    if token_count_usage is not None:
        return _merge_usage_object_with_metadata(
            token_count_usage,
            metadata_usage_object,
        )

    meta_obj = _maybe_get(result, "meta")
    billed_units = _maybe_get(meta_obj, "billed_units")
    token_units = _maybe_get(meta_obj, "tokens")
    if billed_units is not None:
        search_units = _safe_int(_maybe_get(billed_units, "search_units"))
        total_tokens = _safe_int(_maybe_get(billed_units, "total_tokens"))
        input_tokens = _safe_int(_maybe_get(token_units, "input_tokens"))
        prompt_tokens = total_tokens or input_tokens
        rerank_usage: Dict[str, Any] = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": 0,
            "total_tokens": total_tokens or prompt_tokens,
        }
        if search_units:
            rerank_usage["search_units"] = search_units
        return _merge_usage_object_with_metadata(
            rerank_usage,
            metadata_usage_object,
        )

    completed_payload = _extract_responses_completed_payload_from_passthrough_fallback_text(
        _maybe_get(result, "response")
    )
    if isinstance(completed_payload, dict):
        usage_obj = _maybe_get(completed_payload.get("response"), "usage")
        if usage_obj is not None:
            return _merge_usage_object_with_metadata(
                usage_obj,
                metadata_usage_object,
            )

    standard_logging_object = kwargs.get("standard_logging_object")
    if isinstance(standard_logging_object, dict):
        response = standard_logging_object.get("response")
        if isinstance(response, dict) and response.get("usage") is not None:
            return _merge_usage_object_with_metadata(
                response["usage"],
                metadata_usage_object,
            )
        token_count_usage = _build_usage_object_from_token_count_payload(response)
        if token_count_usage is not None:
            return _merge_usage_object_with_metadata(
                token_count_usage,
                metadata_usage_object,
            )
        token_count_usage = _build_usage_object_from_token_count_payload(standard_logging_object.get("output"))
        if token_count_usage is not None:
            return _merge_usage_object_with_metadata(
                token_count_usage,
                metadata_usage_object,
            )

    if metadata_usage_object is not None:
        return metadata_usage_object

    if isinstance(standard_logging_object, dict):
        completed_payload = _extract_responses_completed_payload_from_passthrough_fallback_text(
            _maybe_get(standard_logging_object.get("response"), "response")
        )
        if isinstance(completed_payload, dict):
            usage_obj = _maybe_get(completed_payload.get("response"), "usage")
            if usage_obj is not None:
                return _merge_usage_object_with_metadata(
                    usage_obj,
                    metadata_usage_object,
                )

    return None


def _enrich_token_count_usage_metadata(kwargs: Dict[str, Any], result: Any) -> None:
    metadata = _ensure_mutable_metadata(kwargs)
    standard_logging_object = kwargs.get("standard_logging_object")
    if not isinstance(standard_logging_object, dict):
        standard_logging_object = {}

    passthrough_logging_payload = kwargs.get("passthrough_logging_payload")
    standard_passthrough_logging_payload = kwargs.get("standard_pass_through_logging_payload")
    candidates = (
        result,
        _maybe_get(result, "response"),
        standard_logging_object.get("response"),
        standard_logging_object.get("output"),
        _maybe_get_path(passthrough_logging_payload, "response_body"),
        _maybe_get_path(passthrough_logging_payload, "response"),
        _maybe_get_path(standard_passthrough_logging_payload, "response_body"),
        _maybe_get_path(standard_passthrough_logging_payload, "response"),
    )

    token_count_usage: Optional[Dict[str, Any]] = None
    for candidate in candidates:
        token_count_usage = _build_usage_object_from_token_count_payload(candidate)
        if token_count_usage is not None:
            break
    if token_count_usage is None:
        return

    prompt_tokens = _extract_prompt_tokens(token_count_usage)
    completion_tokens = _extract_completion_tokens(token_count_usage)
    total_tokens = _extract_total_tokens(
        token_count_usage,
        prompt_tokens,
        completion_tokens,
    )
    metadata["usage_token_count_response"] = True
    metadata["usage_input_tokens"] = prompt_tokens
    metadata["usage_output_tokens"] = completion_tokens
    metadata["usage_total_tokens"] = total_tokens
    _merge_tags(metadata, ["token-count-response"])


def _extract_prompt_tokens(usage_obj: Any) -> int:
    return (
        _safe_int(_maybe_get(usage_obj, "prompt_tokens"))
        or _safe_int(_maybe_get(usage_obj, "input_tokens"))
        or _safe_int(_maybe_get(usage_obj, "input"))
        or 0
    )


def _extract_completion_tokens(usage_obj: Any) -> int:
    return (
        _safe_int(_maybe_get(usage_obj, "completion_tokens"))
        or _safe_int(_maybe_get(usage_obj, "output_tokens"))
        or _safe_int(_maybe_get(usage_obj, "candidatesTokenCount"))
        or 0
    )


def _extract_total_tokens(usage_obj: Any, prompt_tokens: int, completion_tokens: int) -> int:
    return (
        _safe_int(_maybe_get(usage_obj, "total_tokens"))
        or _safe_int(_maybe_get(usage_obj, "totalTokenCount"))
        or (prompt_tokens + completion_tokens)
    )


def _extract_prompt_tokens_details(usage_obj: Any) -> Any:
    return _first_non_none(
        _maybe_get(usage_obj, "prompt_tokens_details"),
        _maybe_get(usage_obj, "input_tokens_details"),
        _maybe_get(usage_obj, "promptTokensDetails"),
        _maybe_get(usage_obj, "inputTokensDetails"),
    )


def _extract_completion_tokens_details(usage_obj: Any) -> Any:
    return _first_non_none(
        _maybe_get(usage_obj, "completion_tokens_details"),
        _maybe_get(usage_obj, "output_tokens_details"),
        _maybe_get(usage_obj, "completionTokensDetails"),
        _maybe_get(usage_obj, "outputTokensDetails"),
        _maybe_get(usage_obj, "responseTokensDetails"),
        _maybe_get(usage_obj, "candidatesTokensDetails"),
    )


def _extract_cache_read_input_tokens(usage_obj: Any) -> int:
    prompt_tokens_details = _extract_prompt_tokens_details(usage_obj)
    return (
        _safe_int(_maybe_get(usage_obj, "cache_read_input_tokens"))
        or _safe_int(_maybe_get(usage_obj, "cacheReadInputTokens"))
        or _safe_int(_maybe_get(usage_obj, "cachedContentTokenCount"))
        or _safe_int(_maybe_get(prompt_tokens_details, "cached_tokens"))
        or _safe_int(_maybe_get(prompt_tokens_details, "cachedTokens"))
        or 0
    )


def _extract_cache_creation_input_tokens(usage_obj: Any) -> int:
    return (
        _safe_int(_maybe_get(usage_obj, "cache_creation_input_tokens"))
        or _safe_int(_maybe_get(usage_obj, "cacheWriteInputTokens"))
        or _safe_int(_maybe_get(usage_obj, "cacheWriteInputTokenCount"))
        or _safe_int(_maybe_get(usage_obj, "cacheCreationInputTokens"))
        or 0
    )


_PROVIDER_CACHE_TARGET_FAMILIES = {
    "antigravity",
    "anthropic",
    "openai",
    "openrouter",
    "opencode_zen",
    "gemini",
    "nvidia",
    "xai",
}


def _has_nested_path(obj: Any, *keys: str) -> bool:
    sentinel = object()
    return _maybe_get_path(obj, *keys, default=sentinel) is not sentinel


def _normalize_session_history_provider_name(candidate: Any) -> Optional[str]:
    if not isinstance(candidate, str) or not candidate.strip():
        return None
    candidate_lower = candidate.strip().lower()
    if candidate_lower in {"unknown", "none", "null", "litellm"}:
        return None
    if candidate_lower in {"google", "google_code_assist", "google-code-assist"}:
        return "gemini"
    if candidate_lower in {"agy", "google-antigravity"}:
        return "antigravity"
    if candidate_lower in {"nvidia", "nvidia_nim", "nvidia-nim"}:
        return "nvidia_nim"
    if candidate_lower in {"opencode", "opencode-zen", "opencode_zen", "zen"}:
        return "opencode_zen"
    if candidate_lower == "grok":
        return "xai"
    if candidate_lower in {
        "local_embed",
        "local-embed",
        "local_rerank",
        "local-rerank",
        "local_llm",
        "local-llm",
        "local_biomed",
        "local-biomed",
        "antigravity",
        "openrouter",
        "opencode_zen",
        "openai",
        "anthropic",
        "gemini",
        "xai",
    }:
        return candidate_lower.replace("-", "_")
    return candidate_lower


@lru_cache(maxsize=512)
def _session_history_provider_from_model_catalog(model: str) -> Optional[str]:
    normalized_model = str(model or "").strip()
    if not normalized_model or normalized_model.lower() == "unknown":
        return None
    try:
        from litellm.utils import get_model_info

        model_info = get_model_info(model=normalized_model)
    except Exception:
        return None
    if not isinstance(model_info, dict):
        return None
    return _normalize_session_history_provider_name(model_info.get("litellm_provider"))


def _session_history_provider_from_model(model: Any) -> Optional[str]:
    model_lower = str(model or "").strip().lower()
    if not model_lower or model_lower == "unknown":
        return None
    if model_lower.startswith("local_embed/"):
        return "local_embed"
    if model_lower.startswith("local_rerank/"):
        return "local_rerank"
    if model_lower.startswith("local_llm/"):
        return "local_llm"
    if model_lower.startswith("local_biomed/"):
        return "local_biomed"
    if model_lower.startswith("nvidia/"):
        return "nvidia_nim"
    if model_lower.startswith("xai/") or model_lower.startswith("grok"):
        return "xai"
    if model_lower.startswith("openrouter/"):
        return "openrouter"
    if model_lower.startswith(("opencode/", "opencode-zen/", "zen/")):
        return "opencode_zen"
    if model_lower.startswith(("antigravity/", "agy/", "google-antigravity/")):
        return "antigravity"
    if "gemini" in model_lower or model_lower.startswith("google/"):
        return "gemini"
    if "claude" in model_lower or model_lower.startswith("anthropic/"):
        return "anthropic"
    if (
        model_lower.startswith("gpt")
        or model_lower.startswith("o1")
        or model_lower.startswith("o3")
        or model_lower.startswith("o4")
        or model_lower.startswith("openai/")
        or "codex" in model_lower
    ):
        return "openai"
    return _session_history_provider_from_model_catalog(str(model or ""))


def _session_history_provider_from_route_family(route_family: Any) -> Optional[str]:
    if not isinstance(route_family, str) or not route_family.strip():
        return None
    route_lower = route_family.lower()
    if "grok" in route_lower or "xai" in route_lower:
        return "xai"
    if "nvidia" in route_lower:
        return "nvidia_nim"
    if "openrouter" in route_lower:
        return "openrouter"
    if "opencode" in route_lower:
        return "opencode_zen"
    if "antigravity" in route_lower:
        return "antigravity"
    if "local_embed" in route_lower or "local-embed" in route_lower:
        return "local_embed"
    if "local_rerank" in route_lower or "local-rerank" in route_lower:
        return "local_rerank"
    if "local_llm" in route_lower or "local-llm" in route_lower:
        return "local_llm"
    if "local_biomed" in route_lower or "local-biomed" in route_lower:
        return "local_biomed"
    if "gemini" in route_lower or "google" in route_lower:
        return "gemini"
    if "codex" in route_lower or "openai" in route_lower:
        return "openai"
    if "anthropic" in route_lower:
        return "anthropic"
    return None


def _session_history_adapter_target_provider(
    metadata: Dict[str, Any],
) -> Optional[str]:
    for tag in _metadata_request_tags(metadata):
        tag_lower = tag.strip().lower()
        if not tag_lower.startswith("anthropic-adapter-target:"):
            continue
        target = tag_lower.split(":", 1)[1].strip()
        if target.startswith(("google", "gemini")):
            return "gemini"
        if target.startswith("openrouter"):
            return "openrouter"
        if target.startswith(("opencode", "opencode_zen", "zen")):
            return "opencode_zen"
        if target.startswith(("antigravity", "agy", "google-antigravity")):
            return "antigravity"
        if target.startswith("nvidia"):
            return "nvidia_nim"
        if target.startswith(("xai", "grok")):
            return "xai"
        if target.startswith(("responses", "openai", "codex", "/v1/responses")):
            return "openai"
    return None


def _session_history_auto_agent_selected_provider(
    metadata: Dict[str, Any],
) -> Optional[str]:
    selected_provider = _normalize_session_history_provider_name(metadata.get("codex_auto_agent_selected_provider"))
    if selected_provider is not None:
        return selected_provider
    selected_provider = _normalize_session_history_provider_name(metadata.get("anthropic_auto_agent_selected_provider"))
    if selected_provider is not None:
        return selected_provider
    return _normalize_session_history_provider_name(metadata.get("aawm_auto_agent_selected_provider"))


def _session_history_adapter_model(metadata: Dict[str, Any]) -> Optional[str]:
    prefix = "anthropic-adapter-model:"
    for tag in _metadata_request_tags(metadata):
        stripped_tag = tag.strip()
        if stripped_tag.lower().startswith(prefix):
            return stripped_tag[len(prefix) :].strip() or None
    return None
def _normalize_session_history_provider(
    provider: Any,
    model: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    metadata = metadata or {}
    adapter_target_provider = _session_history_adapter_target_provider(metadata)
    if adapter_target_provider is not None:
        return adapter_target_provider

    auto_agent_provider = _session_history_auto_agent_selected_provider(metadata)
    if auto_agent_provider is not None:
        return auto_agent_provider

    credential_family = str(metadata.get("credential_family") or "").strip().lower()
    if (
        credential_family == "xai_oauth"
        or metadata.get("xai_oauth_managed") is True
        or metadata.get("xai_oauth_public_model") is not None
    ):
        return "xai"

    route_provider = _session_history_provider_from_route_family(metadata.get("passthrough_route_family"))
    if route_provider is not None and route_provider != "anthropic":
        return route_provider

    model_provider = _session_history_provider_from_model(model)

    normalized_provider = _normalize_session_history_provider_name(provider)
    if (
        normalized_provider in {"anthropic", "openai"}
        and model_provider is not None
        and model_provider != normalized_provider
    ):
        return model_provider
    if normalized_provider is not None:
        return normalized_provider

    for key in (
        "custom_llm_provider",
        "provider",
        "litellm_provider",
        "aawm_stream_logging_custom_llm_provider",
    ):
        normalized_provider = _normalize_session_history_provider_name(metadata.get(key))
        if (
            normalized_provider in {"anthropic", "openai"}
            and model_provider is not None
            and model_provider != normalized_provider
        ):
            return model_provider
        if normalized_provider is not None:
            return normalized_provider

    if route_provider is not None:
        return route_provider

    request_route = metadata.get("user_api_key_request_route")
    if isinstance(request_route, str) and request_route.strip():
        route_lower = request_route.lower()
        if "gemini" in route_lower or "google" in route_lower:
            return "gemini"
        if route_lower.startswith("/v1/"):
            return "openai"
        if route_lower.startswith("/anthropic/"):
            return "anthropic"

    api_base = metadata.get("api_base") or _maybe_get(metadata.get("hidden_params"), "api_base")
    if isinstance(api_base, str) and api_base.strip():
        api_base_lower = api_base.lower()
        if "api.x.ai" in api_base_lower or "cli-chat-proxy.grok.com" in api_base_lower:
            return "xai"
        if "integrate.api.nvidia.com" in api_base_lower:
            return "nvidia_nim"
        if "openrouter.ai" in api_base_lower:
            return "openrouter"
        if "opencode.ai/zen" in api_base_lower:
            return "opencode_zen"
        if "anthropic.com" in api_base_lower:
            return "anthropic"
        if "googleapis.com" in api_base_lower or "generativelanguage" in api_base_lower:
            return "gemini"
        if "openai.com" in api_base_lower:
            return "openai"

    return model_provider
_INVALID_TOOL_CALL_ERROR_RE = re.compile(
    r"("
    r"\bInputValidationError\b"
    r"|<tool_use_error>"
    r"|tool_use_error"
    r"|unexpected (?:parameter|key)"
    r"|unrecognized (?:parameter|key)"
    r"|unknown (?:parameter|key)"
    r"|invalid tool(?: call| use)?"
    r"|tool call validation"
    r"|unable to parse tool parameter json"
    r"|failed due to the following issue"
    r")",
    re.IGNORECASE,
)
_TOOL_RESULT_ERROR_BLOCK_TYPES = {
    "tool_result",
    "tool_use_result",
    "function_call_output",
}


def _invalid_tool_call_error_text_seen(value: Any) -> bool:
    parsed = _safe_json_load(value, value)
    if isinstance(parsed, str):
        return bool(_INVALID_TOOL_CALL_ERROR_RE.search(parsed))
    if isinstance(parsed, dict):
        for key in (
            "content",
            "text",
            "output",
            "error",
            "message",
            "status",
            "name",
            "type",
        ):
            if key in parsed and _invalid_tool_call_error_text_seen(parsed[key]):
                return True
        return False
    if isinstance(parsed, list):
        return any(_invalid_tool_call_error_text_seen(item) for item in parsed)
    return False


def _iter_tool_result_error_candidates(message: Any) -> Iterator[Any]:
    parsed_message = _safe_json_load(message, message)
    if not isinstance(parsed_message, dict):
        return

    message_type = _clean_non_empty_string(parsed_message.get("type"))
    message_role = _clean_non_empty_string(parsed_message.get("role"))
    if message_type in _TOOL_RESULT_ERROR_BLOCK_TYPES or (message_role or "").lower() == "tool":
        yield parsed_message

    content = _safe_json_load(parsed_message.get("content"), parsed_message.get("content"))
    if isinstance(content, dict):
        content_blocks = [content]
    elif isinstance(content, list):
        content_blocks = content
    else:
        content_blocks = []

    for block in content_blocks:
        parsed_block = _safe_json_load(block, block)
        if not isinstance(parsed_block, dict):
            continue
        block_type = _clean_non_empty_string(parsed_block.get("type"))
        if block_type in _TOOL_RESULT_ERROR_BLOCK_TYPES:
            yield parsed_block


def _iter_request_message_payloads(request_body: Dict[str, Any]) -> Iterator[Any]:
    for key in ("messages", "input"):
        value = request_body.get(key)
        parsed = _safe_json_load(value, value)
        if isinstance(parsed, list):
            yield from parsed
        elif isinstance(parsed, dict):
            yield parsed

    nested_request = _safe_json_load(request_body.get("request"), request_body.get("request"))
    if isinstance(nested_request, dict) and nested_request is not request_body:
        yield from _iter_request_message_payloads(nested_request)


def _extract_invalid_tool_call_count_from_request_body(
    request_body: Optional[Dict[str, Any]],
) -> int:
    if not isinstance(request_body, dict):
        return 0

    invalid_count = 0
    for message in _iter_request_message_payloads(request_body):
        for candidate in _iter_tool_result_error_candidates(message):
            if _invalid_tool_call_error_text_seen(candidate):
                invalid_count += 1
    return invalid_count


_STRUCTURED_OUTPUT_JSON_MODE_VALUES = {
    "json",
    "json_object",
    "json_schema",
    "schema",
    "response_schema",
}
_STRUCTURED_OUTPUT_NESTED_REQUEST_KEYS = (
    "body",
    "data",
    "json",
    "payload",
    "request",
    "request_body",
)
_STRUCTURED_OUTPUT_FAILURE_PATTERNS = (
    (
        "schema_validation_error",
        re.compile(
            r"("
            r"structured[-_ ]?output"
            r"|json[-_ ]?schema"
            r"|schema validation"
            r"|validation schema"
            r"|invalid schema"
            r"|schema .*valid"
            r"|does not match (?:the )?schema"
            r"|pydantic"
            r"|jsonschema"
            r")",
            re.IGNORECASE,
        ),
    ),
    (
        "json_validation_error",
        re.compile(
            r"("
            r"invalid[-_ ]?json"
            r"|malformed json"
            r"|json parse"
            r"|parse json"
            r"|json decode"
            r"|json validation"
            r"|validate json"
            r"|json .*valid"
            r")",
            re.IGNORECASE,
        ),
    ),
    (
        "response_format_error",
        re.compile(r"(response[-_ ]?format|invalid_response_format)", re.IGNORECASE),
    ),
)


def _empty_structured_output_state() -> Dict[str, Any]:
    return {
        "structured_output_attempted": False,
        "structured_output_failed": False,
        "structured_output_mode": None,
        "structured_output_schema_hash": None,
        "structured_output_failure_reason": None,
    }


def _merge_structured_output_state(
    current: Dict[str, Any],
    candidate: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    if not isinstance(candidate, dict) or not candidate.get("structured_output_attempted"):
        return current

    current["structured_output_attempted"] = True
    current["structured_output_failed"] = bool(
        current.get("structured_output_failed") or candidate.get("structured_output_failed")
    )
    for key in (
        "structured_output_mode",
        "structured_output_schema_hash",
        "structured_output_failure_reason",
    ):
        value = _clean_non_empty_string(candidate.get(key))
        if value and not _clean_non_empty_string(current.get(key)):
            current[key] = value
    return current


def _structured_output_schema_hash(value: Any) -> Optional[str]:
    if value is None:
        return None
    try:
        encoded = json.dumps(
            _json_safe_rate_limit_value(value),
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError):
        encoded = str(value)
    if not encoded or encoded in {"null", "{}", "[]"}:
        return None
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _structured_output_state_from_format(
    value: Any,
    *,
    default_mode: Optional[str] = None,
) -> Dict[str, Any]:
    parsed = _safe_json_load(value, value)
    state = _empty_structured_output_state()

    if isinstance(parsed, str):
        mode = parsed.strip().lower().replace("-", "_")
        if mode in _STRUCTURED_OUTPUT_JSON_MODE_VALUES or "json" in mode:
            state["structured_output_attempted"] = True
            state["structured_output_mode"] = mode
        return state

    if not isinstance(parsed, dict):
        return state

    raw_mode = _first_non_empty_string(
        parsed.get("type"),
        parsed.get("format"),
        parsed.get("mode"),
        default_mode,
    )
    dict_mode = raw_mode.lower().replace("-", "_") if raw_mode else None
    schema = _first_non_none(
        parsed.get("json_schema"),
        parsed.get("schema"),
        parsed.get("response_schema"),
        parsed.get("responseSchema"),
    )
    mime_type = _first_non_empty_string(
        parsed.get("response_mime_type"),
        parsed.get("responseMimeType"),
        parsed.get("mime_type"),
    )
    has_json_mime = bool(mime_type and "json" in mime_type.lower())
    has_json_mode = bool(
        dict_mode and (dict_mode in _STRUCTURED_OUTPUT_JSON_MODE_VALUES or "json" in dict_mode or "schema" in dict_mode)
    )
    if schema is None and not has_json_mode and not has_json_mime:
        return state

    state["structured_output_attempted"] = True
    state["structured_output_mode"] = dict_mode or ("response_schema" if schema is not None else "json_mime_type")
    state["structured_output_schema_hash"] = _structured_output_schema_hash(schema)
    return state


def _structured_output_state_from_generation_config(value: Any) -> Dict[str, Any]:
    parsed = _safe_json_load(value, value)
    state = _empty_structured_output_state()
    if not isinstance(parsed, dict):
        return state

    schema = _first_non_none(
        parsed.get("responseSchema"),
        parsed.get("response_schema"),
    )
    mime_type = _first_non_empty_string(
        parsed.get("responseMimeType"),
        parsed.get("response_mime_type"),
    )
    if schema is None and not (mime_type and "json" in mime_type.lower()):
        return state

    state["structured_output_attempted"] = True
    state["structured_output_mode"] = "response_schema" if schema is not None else "json_mime_type"
    state["structured_output_schema_hash"] = _structured_output_schema_hash(schema)
    return state


def _detect_structured_output_request(
    request_body: Optional[Dict[str, Any]],
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    state = _empty_structured_output_state()

    if isinstance(metadata, dict):
        metadata_attempted = any(
            key in metadata and _metadata_bool(metadata.get(key))
            for key in (
                "usage_structured_output_attempted",
                "structured_output_attempted",
            )
        )
        metadata_failed = any(
            key in metadata and _metadata_bool(metadata.get(key))
            for key in (
                "usage_structured_output_failed",
                "structured_output_failed",
            )
        )
        metadata_mode = _first_non_empty_string(
            metadata.get("usage_structured_output_mode"),
            metadata.get("structured_output_mode"),
        )
        metadata_schema_hash = _first_non_empty_string(
            metadata.get("usage_structured_output_schema_hash"),
            metadata.get("structured_output_schema_hash"),
        )
        metadata_reason = _first_non_empty_string(
            metadata.get("usage_structured_output_failure_reason"),
            metadata.get("structured_output_failure_reason"),
        )
        if metadata_attempted or metadata_failed or metadata_mode or metadata_schema_hash:
            state["structured_output_attempted"] = True
            state["structured_output_failed"] = metadata_failed
            state["structured_output_mode"] = metadata_mode
            state["structured_output_schema_hash"] = metadata_schema_hash
            state["structured_output_failure_reason"] = metadata_reason

    parsed_request = _safe_json_load(request_body, request_body)
    if not isinstance(parsed_request, dict):
        return state

    pending: List[Tuple[Any, int]] = [(parsed_request, 0)]
    seen: set[int] = set()
    while pending:
        payload, depth = pending.pop(0)
        if not isinstance(payload, dict):
            continue
        payload_id = id(payload)
        if payload_id in seen:
            continue
        seen.add(payload_id)

        for key in ("response_format", "responseFormat"):
            if key in payload:
                _merge_structured_output_state(
                    state,
                    _structured_output_state_from_format(payload.get(key)),
                )

        text_config = _safe_json_load(payload.get("text"), payload.get("text"))
        if isinstance(text_config, dict) and "format" in text_config:
            _merge_structured_output_state(
                state,
                _structured_output_state_from_format(text_config.get("format")),
            )

        for key in ("text_format", "textFormat"):
            if key in payload:
                _merge_structured_output_state(
                    state,
                    _structured_output_state_from_format(payload.get(key)),
                )

        for key in ("output_format", "outputFormat", "output_config", "outputConfig"):
            if key in payload:
                _merge_structured_output_state(
                    state,
                    _structured_output_state_from_format(payload.get(key)),
                )

        for key in ("generationConfig", "generation_config"):
            if key in payload:
                _merge_structured_output_state(
                    state,
                    _structured_output_state_from_generation_config(payload.get(key)),
                )

        if "response_schema" in payload or "responseSchema" in payload:
            schema = _first_non_none(
                payload.get("response_schema"),
                payload.get("responseSchema"),
            )
            _merge_structured_output_state(
                state,
                {
                    "structured_output_attempted": True,
                    "structured_output_failed": False,
                    "structured_output_mode": "response_schema",
                    "structured_output_schema_hash": _structured_output_schema_hash(schema),
                    "structured_output_failure_reason": None,
                },
            )

        mime_type = _first_non_empty_string(
            payload.get("response_mime_type"),
            payload.get("responseMimeType"),
        )
        if mime_type and "json" in mime_type.lower():
            _merge_structured_output_state(
                state,
                {
                    "structured_output_attempted": True,
                    "structured_output_failed": False,
                    "structured_output_mode": "json_mime_type",
                    "structured_output_schema_hash": None,
                    "structured_output_failure_reason": None,
                },
            )

        if depth >= 4:
            continue
        for key in _STRUCTURED_OUTPUT_NESTED_REQUEST_KEYS:
            nested = _safe_json_load(payload.get(key), payload.get(key))
            if isinstance(nested, dict):
                pending.append((nested, depth + 1))

    return state


def _collect_structured_output_failure_texts(value: Any) -> List[str]:
    texts: List[str] = []
    pending: List[Tuple[Any, int]] = [(value, 0)]
    seen: set[int] = set()
    while pending and len(texts) < 40:
        current, depth = pending.pop(0)
        current = _safe_json_load(current, current)
        if isinstance(current, str):
            if current.strip():
                texts.append(current.strip()[:1000])
            continue
        if isinstance(current, dict):
            current_id = id(current)
            if current_id in seen:
                continue
            seen.add(current_id)
            for key in (
                "message",
                "error",
                "detail",
                "details",
                "code",
                "type",
                "statusMessage",
                "status_message",
            ):
                if key in current:
                    pending.append((current[key], depth + 1))
            if depth < 3:
                for nested_value in list(current.values()):
                    if isinstance(nested_value, (dict, list)):
                        pending.append((nested_value, depth + 1))
            continue
        if isinstance(current, list) and depth < 3:
            for item in current[:40]:
                pending.append((item, depth + 1))
    return texts


def _classify_structured_output_failure(value: Any) -> Optional[str]:
    dicts = _extract_provider_error_dicts(value)
    error_text = _extract_provider_error_text(value, dicts)
    texts = [error_text] if error_text else []
    texts.extend(_collect_structured_output_failure_texts(value))
    combined = "\n".join(text for text in texts if isinstance(text, str))[:5000]
    if not combined.strip():
        return None
    for reason, pattern in _STRUCTURED_OUTPUT_FAILURE_PATTERNS:
        if pattern.search(combined):
            return reason
    return None


def _extract_request_body_from_langfuse_input(value: Any) -> Optional[Dict[str, Any]]:
    parsed = _safe_json_load(value, value)
    if not isinstance(parsed, dict):
        return None

    messages = parsed.get("messages")
    if isinstance(messages, list):
        for message in messages:
            if not isinstance(message, dict):
                continue
            nested = _safe_json_load(message.get("content"), None)
            if isinstance(nested, dict) and (
                isinstance(nested.get("messages"), list)
                or isinstance(nested.get("input"), (str, list, dict))
                or isinstance(nested.get("instructions"), str)
                or isinstance(nested.get("model"), str)
            ):
                return nested
        return parsed

    body = parsed.get("body")
    if isinstance(body, dict):
        return _extract_request_body_from_langfuse_input(body)
    return None


def _request_payload_contains(
    payload: Any,
    predicate: Any,
) -> bool:
    pending: List[Tuple[Any, int]] = [(payload, 0)]
    seen: Set[int] = set()
    scanned = 0

    while pending and scanned < _AAWM_REQUEST_PAYLOAD_SCAN_MAX_ITEMS:
        value, depth = pending.pop()
        scanned += 1

        if isinstance(value, dict):
            value_id = id(value)
            if value_id in seen:
                continue
            seen.add(value_id)

            if predicate(value):
                return True
            if depth >= _AAWM_REQUEST_PAYLOAD_SCAN_MAX_DEPTH:
                continue
            pending.extend(
                (nested_value, depth + 1)
                for nested_value in list(value.values())
                if isinstance(nested_value, (dict, list, tuple))
            )
            continue

        if isinstance(value, (list, tuple)):
            value_id = id(value)
            if value_id in seen:
                continue
            seen.add(value_id)

            if depth >= _AAWM_REQUEST_PAYLOAD_SCAN_MAX_DEPTH:
                continue
            pending.extend((item, depth + 1) for item in list(value) if isinstance(item, (dict, list, tuple)))

    return False
_CODEX_THREAD_ID_RE = re.compile(r"\bCODEX_THREAD_ID=(?P<thread_id>[A-Za-z0-9][A-Za-z0-9._:-]{7,})\b")
_GEMINI_COMPACT_PROMPT_ID_RE = re.compile(r"^compress-[A-Za-z0-9._:-]+$")
_CLAUDE_CODE_COMPACT_REQUEST_MARKERS = (
    "your task is to create a detailed summary of the conversation so far",
    "respond with text only",
    "do not call any tools",
)


def _append_request_content_text(texts: List[str], content: Any) -> None:
    text = _content_to_text(content).strip()
    if text:
        texts.append(text)


def _extract_request_user_texts(request_body: Any) -> List[str]:
    if not isinstance(request_body, dict):
        return []

    texts: List[str] = []
    messages = request_body.get("messages")
    if isinstance(messages, list):
        for message in messages:
            if not isinstance(message, dict):
                continue
            if str(message.get("role") or "").lower() == "user":
                _append_request_content_text(texts, message.get("content"))

    input_items = request_body.get("input")
    if isinstance(input_items, str):
        texts.append(input_items.strip())
    elif isinstance(input_items, list):
        for item in input_items:
            if isinstance(item, str):
                if item.strip():
                    texts.append(item.strip())
                continue
            if not isinstance(item, dict):
                continue
            item_type = str(item.get("type") or "").lower()
            role = str(item.get("role") or "").lower()
            if item_type == "input_text":
                _append_request_content_text(texts, item.get("text"))
            elif role == "user" and item_type in {"", "message"}:
                _append_request_content_text(texts, item.get("content"))

    return texts


def _join_compact_request_user_texts(request_body: Any) -> str:
    return "\n".join(_extract_request_user_texts(request_body))


def _extract_codex_compact_thread_id(
    metadata: Dict[str, Any],
    request_body: Any,
    request_text: str,
) -> Optional[str]:
    if isinstance(request_body, dict):
        prompt_cache_key = _clean_non_empty_string(request_body.get("prompt_cache_key"))
        if prompt_cache_key is not None:
            return prompt_cache_key

    for candidate in (
        metadata.get("prompt_cache_key"),
        metadata.get("codex_prompt_cache_key"),
        metadata.get("CODEX_THREAD_ID"),
        metadata.get("codex_thread_id"),
    ):
        thread_id = _clean_non_empty_string(candidate)
        if thread_id is not None:
            return thread_id

    match = _CODEX_THREAD_ID_RE.search(request_text)
    if match:
        return match.group("thread_id")
    return None


def _extract_gemini_compact_prompt_id(
    metadata: Dict[str, Any],
    request_body: Any,
) -> Optional[str]:
    candidates = [metadata.get("gemini_user_prompt_id")]
    if isinstance(request_body, dict):
        candidates.extend(
            [
                request_body.get("user_prompt_id"),
                _maybe_get_path(request_body, "request", "user_prompt_id"),
            ]
        )
    for candidate in candidates:
        prompt_id = _clean_non_empty_string(candidate)
        if prompt_id and _GEMINI_COMPACT_PROMPT_ID_RE.match(prompt_id):
            return prompt_id
    return None


def _base_gemini_compact_prompt_id(prompt_id: str) -> str:
    if prompt_id.endswith("-verify"):
        return prompt_id[: -len("-verify")]
    return prompt_id


def _extract_compact_output_text(output_payload: Any) -> str:
    parsed = _safe_json_load(output_payload, output_payload)

    for extractor in (_extract_first_response_message, _extract_first_langfuse_response_message):
        message = extractor(parsed)
        if message is None:
            continue
        text = _content_to_text(_maybe_get(message, "content")).strip()
        if text:
            return text

    if isinstance(parsed, dict):
        content = parsed.get("content")
        if isinstance(content, str) and content.strip():
            return content.strip()

        candidates = parsed.get("candidates")
        if isinstance(candidates, list):
            for candidate in candidates:
                parts = _maybe_get_path(candidate, "content", "parts")
                text = _content_to_text(parts).strip()
                if text:
                    return text

    return _content_to_text(parsed).strip()


def _is_claude_code_compact_context(metadata: Dict[str, Any]) -> bool:
    client_name = str(metadata.get("client_name") or "").strip().lower()
    trace_name = str(metadata.get("trace_name") or "").strip().lower()
    route_family = str(metadata.get("passthrough_route_family") or "").strip().lower()
    return (
        client_name in {"claude-cli", "claude-code"}
        or trace_name.startswith("claude-code")
        or route_family in {"anthropic_messages", "anthropic_completion"}
    )


def _is_codex_compact_context(metadata: Dict[str, Any]) -> bool:
    client_name = str(metadata.get("client_name") or "").strip().lower()
    trace_name = str(metadata.get("trace_name") or "").strip().lower()
    route_family = str(metadata.get("passthrough_route_family") or "").strip().lower()
    return client_name == "codex-tui" or trace_name.startswith("codex") or route_family == "codex_responses"


def _is_gemini_cli_compact_context(metadata: Dict[str, Any]) -> bool:
    client_name = str(metadata.get("client_name") or "").strip().lower()
    user_agent = str(metadata.get("client_user_agent") or "").strip().lower()
    route_family = str(metadata.get("passthrough_route_family") or "").strip().lower()
    return (
        client_name == "gemini-cli"
        or user_agent.startswith("geminicli-tui/")
        or route_family == "gemini_generate_content"
    )


def _classify_compact_summary_state(
    *,
    metadata: Dict[str, Any],
    request_body: Any,
    output_payload: Any,
    session_id: Optional[str],
    litellm_call_id: Optional[str],
    trace_id: Optional[str],
) -> Dict[str, Any]:
    request_text = _join_compact_request_user_texts(request_body)
    request_text_lower = request_text.lower()
    output_text = _extract_compact_output_text(output_payload)
    output_text_lower = output_text.lower()

    if _is_codex_compact_context(metadata):
        compact_id = _extract_codex_compact_thread_id(
            metadata,
            request_body,
            request_text,
        )
        if "context checkpoint compaction" in request_text_lower:
            return {
                "is_compact_summary": True,
                "compact_summary_source": "codex",
                "compact_summary_role": "event",
                "compact_summary_id": compact_id or litellm_call_id or trace_id or session_id,
            }
        if "another language model started to solve this problem" in request_text_lower:
            return {
                "is_compact_summary": False,
                "compact_summary_source": "codex",
                "compact_summary_role": "resume_context",
                "compact_summary_id": compact_id or session_id,
            }

    gemini_prompt_id = _extract_gemini_compact_prompt_id(metadata, request_body)
    if gemini_prompt_id is not None and _is_gemini_cli_compact_context(metadata):
        is_verify = gemini_prompt_id.endswith("-verify")
        if not is_verify and not output_text_lower.startswith("<state_snapshot>"):
            return {
                "is_compact_summary": False,
                "compact_summary_source": None,
                "compact_summary_role": None,
                "compact_summary_id": None,
            }
        return {
            "is_compact_summary": not is_verify,
            "compact_summary_source": "gemini-cli",
            "compact_summary_role": "verify" if is_verify else "event",
            "compact_summary_id": _base_gemini_compact_prompt_id(gemini_prompt_id),
        }

    if _is_claude_code_compact_context(metadata):
        has_compact_tags = "<analysis>" in request_text_lower and "<summary>" in request_text_lower
        strict_prompt_shape = all(marker in request_text_lower for marker in _CLAUDE_CODE_COMPACT_REQUEST_MARKERS)
        compact_summary_phrase = (
            "summarize the current context" in request_text_lower or "context compacted" in request_text_lower
        )
        if has_compact_tags and (strict_prompt_shape or compact_summary_phrase):
            compact_id = litellm_call_id or trace_id or session_id
            return {
                "is_compact_summary": True,
                "compact_summary_source": "claude-code",
                "compact_summary_role": "event",
                "compact_summary_id": compact_id,
            }

    return {
        "is_compact_summary": False,
        "compact_summary_source": None,
        "compact_summary_role": None,
        "compact_summary_id": None,
    }
def _extract_reported_reasoning_tokens(usage_obj: Any) -> Optional[int]:
    completion_tokens_details = _extract_completion_tokens_details(usage_obj)
    explicit_reasoning_tokens = _first_non_none(
        _safe_int(_maybe_get(usage_obj, "reasoning_tokens")),
        _safe_int(_maybe_get(usage_obj, "reasoningTokens")),
        _safe_int(_maybe_get(usage_obj, "reasoning_token_count")),
        _safe_int(_maybe_get(usage_obj, "thoughtsTokenCount")),
        _safe_int(_maybe_get(completion_tokens_details, "reasoning_tokens")),
        _safe_int(_maybe_get(completion_tokens_details, "reasoningTokens")),
    )
    if explicit_reasoning_tokens is not None and explicit_reasoning_tokens > 0:
        return explicit_reasoning_tokens

    modality_reasoning_counts: list[int] = []
    for details in (
        completion_tokens_details,
        _maybe_get(usage_obj, "responseTokensDetails"),
        _maybe_get(usage_obj, "candidatesTokensDetails"),
    ):
        if not isinstance(details, list):
            continue
        detail_reasoning_tokens = 0
        has_reasoning_detail = False
        for detail in details:
            modality = _maybe_get(detail, "modality")
            if not isinstance(modality, str):
                continue
            if modality.upper() not in {"THOUGHT", "REASONING"}:
                continue
            token_count = _safe_int(_maybe_get(detail, "tokenCount"))
            if token_count is None or token_count <= 0:
                continue
            detail_reasoning_tokens += token_count
            has_reasoning_detail = True
        if has_reasoning_detail:
            modality_reasoning_counts.append(detail_reasoning_tokens)

    if modality_reasoning_counts:
        return max(modality_reasoning_counts)

    return None


def _fallback_gemini_reasoning_tokens_from_signatures(metadata: Dict[str, Any], message: Any = None) -> Optional[int]:
    signature_count = _safe_int(metadata.get("gemini_thought_signature_count"))
    if signature_count is not None and signature_count > 0:
        return signature_count

    provider_specific_fields = _extract_provider_specific_fields(message) if message is not None else {}
    thought_signatures = provider_specific_fields.get("thought_signatures")
    if isinstance(thought_signatures, list):
        non_empty_signatures = [
            signature for signature in thought_signatures if isinstance(signature, str) and signature.strip()
        ]
        if non_empty_signatures:
            return len(non_empty_signatures)

    if metadata.get("gemini_thought_signature_present") is True:
        return 1
    if metadata.get("thinking_signature_present") is True:
        return 1

    return None


def _determine_reasoning_tokens_source(
    *,
    provider_reported_reasoning_tokens: Optional[int],
    reported_reasoning_tokens: Optional[int],
    estimated_reasoning_tokens: Optional[int],
    reasoning_present: bool,
) -> str:
    if provider_reported_reasoning_tokens is not None and reported_reasoning_tokens is not None:
        return "provider_reported"
    if reported_reasoning_tokens is not None:
        return "provider_signature_present"
    if estimated_reasoning_tokens is not None:
        return "estimated_from_reasoning_text"
    if reasoning_present:
        return "not_available"
    return "not_applicable"


def _estimate_reasoning_tokens(model: str, reasoning_text: str) -> Optional[int]:
    stripped_reasoning = reasoning_text.strip()
    if not stripped_reasoning:
        return None

    try:
        litellm = _get_litellm_module()
        return litellm.token_counter(
            model=model or "",
            text=stripped_reasoning,
            count_response_tokens=True,
        )
    except Exception as exc:
        verbose_logger.debug(
            "AawmAgentIdentity: failed to estimate reasoning tokens for model=%s: %s",
            model,
            exc,
        )
        return None


def _extract_rerank_request_payload(kwargs: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    candidates = (
        _extract_provider_cache_request_body(kwargs),
        kwargs,
        _maybe_get(kwargs.get("standard_logging_object"), "optional_params"),
        kwargs.get("optional_params"),
    )
    for candidate in candidates:
        if (
            isinstance(candidate, dict)
            and candidate.get("query") is not None
            and (candidate.get("documents") is not None or candidate.get("texts") is not None)
        ):
            return candidate
    return None


def _coerce_rerank_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float, bool)):
        return str(value)
    if isinstance(value, list):
        return "\n".join(text for item in value if (text := _coerce_rerank_text(item).strip()))
    if isinstance(value, dict):
        try:
            return json.dumps(value, sort_keys=True, default=str)
        except Exception:
            return str(value)
    return str(value)


def _extract_rerank_document_text(
    document: Any,
    rank_fields: Optional[List[str]],
) -> str:
    if isinstance(document, str):
        return document
    if isinstance(document, dict):
        if rank_fields:
            return "\n".join(
                text for field in rank_fields if (text := _coerce_rerank_text(document.get(field)).strip())
            )
        if "text" in document:
            return _coerce_rerank_text(document.get("text"))
    return _coerce_rerank_text(document)


def _fallback_text_token_estimate(text: str) -> int:
    stripped = text.strip()
    if not stripped:
        return 0
    return max(1, (len(stripped) + 3) // 4)


def _empty_prompt_overhead_breakdown() -> Dict[str, Any]:
    return {field: 0 for field in _PROMPT_OVERHEAD_TOKEN_FIELDS}


def _serialize_prompt_overhead_component(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float, bool)):
        return str(value)
    try:
        return json.dumps(value, sort_keys=True, default=str, separators=(",", ":"))
    except Exception:
        return str(value)


def _estimate_prompt_overhead_tokens(model: str, value: Any) -> int:
    text = _serialize_prompt_overhead_component(value).strip()
    if not text:
        return 0
    try:
        litellm = _get_litellm_module()
        token_count = litellm.token_counter(model=model or "", text=text)
        coerced = _safe_int(token_count)
        if coerced is not None and coerced >= 0:
            return coerced
    except Exception as exc:
        verbose_logger.debug(
            "AawmAgentIdentity: failed to estimate prompt-overhead tokens for model=%s: %s",
            model,
            exc,
        )
    return _fallback_text_token_estimate(text)


def _extract_prompt_text_blocks(
    value: Any,
    *,
    _seen: Optional[Set[int]] = None,
    _depth: int = 0,
) -> List[str]:
    if _seen is None:
        _seen = set()
    if _depth > _AAWM_REQUEST_PAYLOAD_SCAN_MAX_DEPTH:
        return []
    if value is None:
        return []
    if isinstance(value, str):
        return [block.strip() for block in re.split(r"\n{2,}", value) if block.strip()]
    if isinstance(value, (int, float, bool)):
        return [str(value)]
    if isinstance(value, list):
        value_id = id(value)
        if value_id in _seen:
            return []
        _seen.add(value_id)
        blocks: List[str] = []
        for item in value:
            blocks.extend(
                _extract_prompt_text_blocks(
                    item,
                    _seen=_seen,
                    _depth=_depth + 1,
                )
            )
        return blocks
    if isinstance(value, dict):
        value_id = id(value)
        if value_id in _seen:
            return []
        _seen.add(value_id)
        blocks = []
        for key in ("text", "content", "parts", "systemInstruction", "system_instruction"):
            if key in value:
                blocks.extend(
                    _extract_prompt_text_blocks(
                        value.get(key),
                        _seen=_seen,
                        _depth=_depth + 1,
                    )
                )
        if blocks:
            return blocks
        return [_serialize_prompt_overhead_component(value)]
    return [str(value)]


def _classify_system_prompt_block(block: str) -> str:
    lowered = block.lower()
    safety_markers = (
        "safety",
        "unsafe",
        "policy",
        "refuse",
        "disallowed",
        "forbidden",
        "harm",
        "malicious",
        "secret",
        "credential",
        "privacy",
        "security",
        "do not reveal",
        "never reveal",
    )
    if any(marker in lowered for marker in safety_markers):
        return "safety"

    behavior_markers = (
        "you are",
        "persona",
        "personality",
        "tone",
        "style",
        "respond as",
        "communication",
        "be concise",
        "be direct",
    )
    if any(marker in lowered for marker in behavior_markers):
        return "behavior"

    instructional_markers = (
        "always",
        "must",
        "should",
        "use ",
        "follow",
        "workflow",
        "steps",
        "when ",
        "before ",
        "after ",
        "tool",
        "repository",
        "codebase",
        "task",
        "instruction",
    )
    if any(marker in lowered for marker in instructional_markers):
        return "instructional"
    return "unclassified"


def _estimate_system_prompt_bucket_tokens(
    *,
    model: str,
    system_components: List[Dict[str, Any]],
) -> Tuple[Dict[str, int], List[str]]:
    bucket_tokens = {
        "behavior": 0,
        "safety": 0,
        "instructional": 0,
        "unclassified": 0,
    }
    component_paths: List[str] = []
    for component in system_components:
        path = str(component.get("path") or "system")
        value = component.get("value")
        blocks = _extract_prompt_text_blocks(value)
        if not blocks:
            continue
        component_paths.append(path)
        for block in blocks:
            bucket = _classify_system_prompt_block(block)
            bucket_tokens[bucket] += _estimate_prompt_overhead_tokens(model, block)
    return bucket_tokens, component_paths


def _append_prompt_component(
    components: Dict[str, List[Dict[str, Any]]],
    name: str,
    *,
    path: str,
    value: Any,
) -> None:
    if value is None:
        return
    if isinstance(value, str) and not value.strip():
        return
    if isinstance(value, list) and not value:
        return
    if isinstance(value, dict) and not value:
        return
    components[name].append({"path": path, "value": value})


_RESPONSES_SYSTEM_ROLES = {"system", "developer"}
_RESPONSES_CONVERSATION_ROLES = {"user", "assistant"}
_RESPONSES_TEXT_CONTENT_TYPES = {"input_text", "output_text", "text"}
_RESPONSES_OPAQUE_CONTENT_TYPES = {
    "item_reference",
    "input_audio",
    "audio",
    "input_image",
    "image",
    "image_url",
}
_RESPONSES_OPAQUE_ITEM_TYPES = {
    "reasoning",
    "function_call",
    "mcp_call",
    "file_search_call",
    "web_search_call",
    "computer_call",
    "item_reference",
}


def _append_prompt_text_components(
    components: Dict[str, List[Dict[str, Any]]],
    name: str,
    *,
    path: str,
    values: List[str],
) -> None:
    for value in values:
        _append_prompt_component(components, name, path=path, value=value)


def _extract_responses_visible_text_blocks(
    value: Any,
    *,
    _seen: Optional[Set[int]] = None,
    _depth: int = 0,
) -> List[str]:
    if _seen is None:
        _seen = set()
    if _depth > _AAWM_REQUEST_PAYLOAD_SCAN_MAX_DEPTH:
        return []
    if value is None:
        return []
    if isinstance(value, str):
        stripped = value.strip()
        return [stripped] if stripped else []
    if isinstance(value, (int, float, bool)):
        return [str(value)]
    if isinstance(value, list):
        value_id = id(value)
        if value_id in _seen:
            return []
        _seen.add(value_id)
        blocks: List[str] = []
        for item in value:
            blocks.extend(
                _extract_responses_visible_text_blocks(
                    item,
                    _seen=_seen,
                    _depth=_depth + 1,
                )
            )
        return blocks
    if isinstance(value, dict):
        value_id = id(value)
        if value_id in _seen:
            return []
        _seen.add(value_id)
        content_type = str(value.get("type") or "").lower()
        if content_type in _RESPONSES_OPAQUE_CONTENT_TYPES:
            return []
        if content_type in _RESPONSES_TEXT_CONTENT_TYPES:
            text = value.get("text")
            return [text.strip()] if isinstance(text, str) and text.strip() else []
        if "text" in value and isinstance(value.get("text"), str):
            text = value["text"].strip()
            return [text] if text else []
        if "content" in value:
            return _extract_responses_visible_text_blocks(
                value.get("content"),
                _seen=_seen,
                _depth=_depth + 1,
            )
    return []


def _responses_message_component_path(role: str) -> str:
    if role in _RESPONSES_SYSTEM_ROLES:
        return "input[type=message][role=system|developer].content"
    if role in _RESPONSES_CONVERSATION_ROLES:
        return f"input[type=message][role={role}].content"
    return "input[type=message].content"


def _record_responses_excluded_fields(
    components: Dict[str, List[Dict[str, Any]]],
    value: Any,
    *,
    path: str,
    _seen: Optional[Set[int]] = None,
    _depth: int = 0,
) -> None:
    if _seen is None:
        _seen = set()
    if _depth > _AAWM_REQUEST_PAYLOAD_SCAN_MAX_DEPTH:
        return
    if isinstance(value, list):
        value_id = id(value)
        if value_id in _seen:
            return
        _seen.add(value_id)
        for item in value:
            _record_responses_excluded_fields(
                components,
                item,
                path=path,
                _seen=_seen,
                _depth=_depth + 1,
            )
        return
    if not isinstance(value, dict):
        return
    value_id = id(value)
    if value_id in _seen:
        return
    _seen.add(value_id)
    content_type = str(value.get("type") or "").lower()
    if content_type == "item_reference":
        _append_prompt_component(
            components,
            "excluded",
            path=f"{path}[type=item_reference]",
            value=value,
        )
        return
    for key, field_value in value.items():
        if key in {"encrypted_content", "reasoning_content"}:
            _append_prompt_component(
                components,
                "excluded",
                path=f"{path}.{key}",
                value=field_value,
            )
        elif isinstance(field_value, (dict, list)):
            _record_responses_excluded_fields(
                components,
                field_value,
                path=f"{path}.{key}",
                _seen=_seen,
                _depth=_depth + 1,
            )


def _append_openai_responses_input_component(
    components: Dict[str, List[Dict[str, Any]]],
    item: Any,
) -> None:
    if isinstance(item, str):
        _append_prompt_component(
            components,
            "conversation",
            path="input",
            value=item,
        )
        return

    if not isinstance(item, dict):
        _append_prompt_component(
            components,
            "conversation",
            path="input",
            value=item,
        )
        return

    item_type = str(item.get("type") or "").lower()
    role = str(item.get("role") or "").lower()
    if item_type in _RESPONSES_OPAQUE_ITEM_TYPES:
        _append_prompt_component(
            components,
            "excluded",
            path=f"input[type={item_type}]",
            value=item,
        )
        return

    if item_type == "function_call_output":
        _append_prompt_component(
            components,
            "conversation",
            path="input[type=function_call_output].output",
            value=item.get("output"),
        )
        return

    _record_responses_excluded_fields(
        components,
        item,
        path=f"input[type={item_type or 'unknown'}]",
    )

    if item_type == "message" or role:
        bucket = "system" if role in _RESPONSES_SYSTEM_ROLES else "conversation"
        path = _responses_message_component_path(role)
        text_blocks = _extract_responses_visible_text_blocks(item.get("content"))
        if not text_blocks and "content" not in item:
            text_blocks = _extract_responses_visible_text_blocks(item)
        _append_prompt_text_components(
            components,
            bucket,
            path=path,
            values=text_blocks,
        )
        return

    text_blocks = _extract_responses_visible_text_blocks(item)
    if text_blocks:
        _append_prompt_text_components(
            components,
            "conversation",
            path="input[type=visible_text]",
            values=text_blocks,
        )
    else:
        _append_prompt_component(
            components,
            "excluded",
            path=f"input[type={item_type or 'unknown'}]",
            value=item,
        )


def _append_openai_responses_input_components(
    components: Dict[str, List[Dict[str, Any]]],
    input_value: Any,
) -> None:
    if isinstance(input_value, list):
        for item in input_value:
            _append_openai_responses_input_component(components, item)
        return
    _append_openai_responses_input_component(components, input_value)


def _split_chat_prompt_messages(messages: Any) -> Tuple[List[Any], List[Any]]:
    if not isinstance(messages, list):
        return [], []
    system_messages: List[Any] = []
    conversation_messages: List[Any] = []
    for message in messages:
        if isinstance(message, dict) and message.get("role") in {"system", "developer"}:
            system_messages.append(message)
        else:
            conversation_messages.append(message)
    return system_messages, conversation_messages


def _extract_prompt_overhead_components(
    request_body: Dict[str, Any],
    route_family: Optional[str],
) -> Tuple[Dict[str, List[Dict[str, Any]]], str]:
    components: Dict[str, List[Dict[str, Any]]] = {
        "system": [],
        "tools": [],
        "conversation": [],
        "excluded": [],
    }
    route_family_lower = (route_family or "").lower()
    request_block = request_body.get("request")
    is_nested_gemini = isinstance(request_block, dict) and (
        "gemini" in route_family_lower
        or "google" in route_family_lower
        or "contents" in request_block
        or "systemInstruction" in request_block
    )
    if is_nested_gemini:
        nested_request_block = request_block if isinstance(request_block, dict) else {}
        _append_prompt_component(
            components,
            "system",
            path="request.systemInstruction",
            value=nested_request_block.get("systemInstruction") or nested_request_block.get("system_instruction"),
        )
        _append_prompt_component(
            components,
            "tools",
            path="request.tools",
            value=nested_request_block.get("tools") or request_body.get("tools"),
        )
        _append_prompt_component(
            components,
            "conversation",
            path="request.contents",
            value=nested_request_block.get("contents"),
        )
        return components, "gemini_generate_content"

    if request_body.get("systemInstruction") is not None or request_body.get("contents") is not None:
        _append_prompt_component(
            components,
            "system",
            path="systemInstruction",
            value=request_body.get("systemInstruction") or request_body.get("system_instruction"),
        )
        _append_prompt_component(
            components,
            "tools",
            path="tools",
            value=request_body.get("tools"),
        )
        _append_prompt_component(
            components,
            "conversation",
            path="contents",
            value=request_body.get("contents"),
        )
        return components, "gemini_generate_content"

    if request_body.get("instructions") is not None or request_body.get("input") is not None:
        _append_prompt_component(
            components,
            "system",
            path="instructions",
            value=request_body.get("instructions"),
        )
        _append_prompt_component(
            components,
            "tools",
            path="tools",
            value=request_body.get("tools"),
        )
        _append_openai_responses_input_components(
            components,
            request_body.get("input"),
        )
        return components, "openai_responses"

    if request_body.get("messages") is not None:
        if request_body.get("system") is not None:
            _append_prompt_component(
                components,
                "system",
                path="system",
                value=request_body.get("system"),
            )
            _append_prompt_component(
                components,
                "conversation",
                path="messages",
                value=request_body.get("messages"),
            )
            counted_shape = (
                "anthropic_messages_semantic"
                if "anthropic" in route_family_lower
                else "chat_messages_with_top_level_system"
            )
        else:
            system_messages, conversation_messages = _split_chat_prompt_messages(request_body.get("messages"))
            _append_prompt_component(
                components,
                "system",
                path="messages[role=system|developer]",
                value=system_messages,
            )
            _append_prompt_component(
                components,
                "conversation",
                path="messages[role!=system|developer]",
                value=conversation_messages,
            )
            counted_shape = "openai_chat_completions"
        _append_prompt_component(
            components,
            "tools",
            path="tools",
            value=request_body.get("tools"),
        )
        _append_prompt_component(
            components,
            "tools",
            path="mcp_servers",
            value=request_body.get("mcp_servers"),
        )
        return components, counted_shape

    return components, "unknown"


def _build_prompt_overhead_breakdown(
    *,
    kwargs: Dict[str, Any],
    metadata: Dict[str, Any],
    model: str,
    prompt_tokens: int,
    request_body: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    breakdown = _empty_prompt_overhead_breakdown()
    if not isinstance(request_body, dict) or prompt_tokens <= 0:
        return breakdown

    route_family = metadata.get("passthrough_route_family")
    if not isinstance(route_family, str) or not route_family.strip():
        route_family = _maybe_get_path(
            kwargs.get("passthrough_logging_payload"),
            "request_body",
            "litellm_metadata",
            "passthrough_route_family",
        )
    route_family = route_family if isinstance(route_family, str) else None

    components, counted_shape = _extract_prompt_overhead_components(
        request_body,
        route_family,
    )
    bucket_tokens, system_paths = _estimate_system_prompt_bucket_tokens(
        model=model,
        system_components=components["system"],
    )
    system_tokens = sum(bucket_tokens.values())
    tool_tokens = sum(_estimate_prompt_overhead_tokens(model, component["value"]) for component in components["tools"])
    conversation_tokens = sum(
        _estimate_prompt_overhead_tokens(model, component["value"]) for component in components["conversation"]
    )
    excluded_components = components.get("excluded", [])
    opaque_state_tokens = sum(
        _estimate_prompt_overhead_tokens(model, component["value"]) for component in excluded_components
    )
    component_total = system_tokens + tool_tokens + conversation_tokens
    residual_tokens = prompt_tokens - component_total

    breakdown.update(
        {
            "input_system_tokens_estimated": system_tokens,
            "input_tool_advertisement_tokens_estimated": tool_tokens,
            "input_conversation_tokens_estimated": conversation_tokens,
            "input_other_tokens_estimated": max(residual_tokens, 0),
            "input_breakdown_residual_tokens": residual_tokens,
            "system_behavior_tokens_estimated": bucket_tokens["behavior"],
            "system_safety_tokens_estimated": bucket_tokens["safety"],
            "system_instructional_tokens_estimated": bucket_tokens["instructional"],
            "system_unclassified_tokens_estimated": bucket_tokens["unclassified"],
        }
    )

    component_paths = {
        "system": system_paths,
        "tools": [str(component.get("path")) for component in components["tools"]],
        "conversation": [str(component.get("path")) for component in components["conversation"]],
    }
    excluded_component_paths = [str(component.get("path")) for component in excluded_components]
    metadata.update(
        {
            "prompt_overhead_breakdown_source": "request_body_estimate",
            "prompt_overhead_counted_shape": counted_shape,
            "prompt_overhead_route_family": route_family,
            "prompt_overhead_tokenizer": "litellm.token_counter_with_char_fallback",
            "prompt_overhead_classifier_version": _PROMPT_OVERHEAD_CLASSIFIER_VERSION,
            "prompt_overhead_component_paths": component_paths,
            "prompt_overhead_excluded_component_paths": excluded_component_paths,
            "usage_input_opaque_state_tokens_estimated": opaque_state_tokens,
        }
    )
    for key, value in breakdown.items():
        metadata[f"usage_{key}"] = value
    return breakdown


def _estimate_rerank_request_tokens(
    *,
    kwargs: Dict[str, Any],
    model: str,
) -> Optional[int]:
    request_payload = _extract_rerank_request_payload(kwargs)
    if not request_payload:
        return None

    query_text = _coerce_rerank_text(request_payload.get("query")).strip()
    documents = request_payload.get("documents")
    if documents is None:
        documents = request_payload.get("texts")
    if not isinstance(documents, list):
        return None

    raw_rank_fields = request_payload.get("rank_fields")
    rank_fields = raw_rank_fields if isinstance(raw_rank_fields, list) else None
    document_texts = [
        text for document in documents if (text := _extract_rerank_document_text(document, rank_fields).strip())
    ]
    combined_text = "\n\n".join([query_text, *document_texts]).strip()
    if not combined_text:
        return None

    try:
        litellm = _get_litellm_module()
        token_count = litellm.token_counter(model=model or "", text=combined_text)
        return _positive_int_or_none(token_count)
    except Exception as exc:
        verbose_logger.debug(
            "AawmAgentIdentity: failed to estimate rerank tokens for model=%s: %s",
            model,
            exc,
        )
        return _fallback_text_token_estimate(combined_text)


def _usage_has_positive_tokens(usage_obj: Any) -> bool:
    prompt_tokens = _extract_prompt_tokens(usage_obj)
    completion_tokens = _extract_completion_tokens(usage_obj)
    total_tokens = _extract_total_tokens(usage_obj, prompt_tokens, completion_tokens)
    return prompt_tokens > 0 or completion_tokens > 0 or total_tokens > 0


def _merge_estimated_rerank_tokens_into_usage(
    *,
    kwargs: Dict[str, Any],
    result: Any,
    usage_obj: Any,
    model: str,
) -> Any:
    usage_dict = _coerce_usage_object_to_dict(usage_obj)
    if usage_dict is None:
        return usage_obj
    if _usage_has_positive_tokens(usage_dict):
        return usage_obj

    search_units = _safe_int(usage_dict.get("search_units")) or _safe_int(
        _maybe_get_path(result, "meta", "billed_units", "search_units")
    )
    if not search_units:
        return usage_obj

    estimated_tokens = _estimate_rerank_request_tokens(kwargs=kwargs, model=model)
    if estimated_tokens is None:
        return usage_obj

    merged_usage = dict(usage_dict)
    merged_usage.setdefault("prompt_tokens", estimated_tokens)
    merged_usage.setdefault("completion_tokens", 0)
    merged_usage.setdefault("total_tokens", estimated_tokens)
    return merged_usage


def _positive_int_or_none(value: Any) -> Optional[int]:
    normalized = _safe_int(value)
    if normalized is not None and normalized > 0:
        return normalized
    return None


def _normalize_reasoning_state(record: Dict[str, Any]) -> None:
    reported = _positive_int_or_none(record.get("reasoning_tokens_reported"))
    estimated = _positive_int_or_none(record.get("reasoning_tokens_estimated"))
    source = record.get("reasoning_tokens_source")
    reasoning_present = bool(record.get("reasoning_present") or record.get("thinking_signature_present"))

    record["reasoning_tokens_reported"] = reported
    record["reasoning_tokens_estimated"] = estimated

    if source == "provider_signature_present" and reported is not None:
        record["reasoning_tokens_source"] = "provider_signature_present"
    elif source == "provider_reported" and reported is not None:
        record["reasoning_tokens_source"] = "provider_reported"
    elif estimated is not None:
        record["reasoning_tokens_source"] = "estimated_from_reasoning_text"
    elif reasoning_present:
        record["reasoning_tokens_source"] = "not_available"
    else:
        record["reasoning_tokens_source"] = "not_applicable"


def _row_usage_object_from_record(record: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "prompt_tokens": int(record.get("input_tokens") or 0),
        "completion_tokens": int(record.get("output_tokens") or 0),
        "total_tokens": int(record.get("total_tokens") or 0),
        "cache_read_input_tokens": int(record.get("cache_read_input_tokens") or 0),
        "cache_creation_input_tokens": int(record.get("cache_creation_input_tokens") or 0),
    }


def _normalize_provider_cache_state_on_record(record: Dict[str, Any]) -> None:
    metadata = record.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}

    provider = _normalize_session_history_provider(
        record.get("provider"),
        str(record.get("model") or ""),
        metadata,
    )
    if provider is not None:
        record["provider"] = provider

    provider_family = _normalize_provider_cache_family(
        provider,
        str(record.get("model") or ""),
        metadata,
    )
    if provider_family is None:
        return

    cache_state = _resolve_provider_cache_state(
        provider=provider,
        model=str(record.get("model") or ""),
        usage_obj=_row_usage_object_from_record(record),
        metadata=metadata,
        request_body=None,
    )
    if cache_state is None:
        return

    current_status = record.get("provider_cache_status")
    if isinstance(current_status, str) and current_status.strip() and cache_state.get("status") == "not_attempted":
        return
    should_override = (
        not isinstance(current_status, str)
        or not current_status.strip()
        or bool(record.get("cache_read_input_tokens") or record.get("cache_creation_input_tokens"))
        or (
            bool(record.get("provider_cache_miss"))
            and (
                record.get("provider_cache_miss_token_count") is None
                or record.get("provider_cache_miss_cost_usd") is None
            )
        )
    )
    if not should_override:
        return

    cache_state = dict(cache_state)
    cache_state.update(
        _compute_provider_cache_miss_cost_state(
            provider_family=provider_family,
            model=str(record.get("model") or ""),
            usage_obj=_row_usage_object_from_record(record),
            cache_state=cache_state,
            metadata=metadata,
            response_cost_usd=_safe_float(record.get("response_cost_usd")),
        )
    )
    record["provider_cache_attempted"] = bool(cache_state.get("attempted"))
    record["provider_cache_status"] = cache_state.get("status")
    record["provider_cache_miss"] = bool(cache_state.get("miss"))
    record["provider_cache_miss_reason"] = cache_state.get("miss_reason")
    record["provider_cache_miss_token_count"] = cache_state.get("miss_token_count")
    record["provider_cache_miss_cost_usd"] = cache_state.get("miss_cost_usd")


def _normalize_session_runtime_identity_on_record(record: Dict[str, Any]) -> None:
    metadata = record.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}

    identity = _build_session_runtime_identity(
        metadata=metadata,
        kwargs=None,
        allow_runtime=False,
    )
    for key in (
        "litellm_environment",
        "litellm_version",
        "litellm_fork_version",
        "client_name",
        "client_version",
        "client_user_agent",
        "client_ip",
        "host_name",
    ):
        if not _clean_non_empty_string(record.get(key)):
            record[key] = identity.get(key)

    record_wheel_versions = _coerce_string_dict(record.get("litellm_wheel_versions"))
    metadata_wheel_versions = _coerce_string_dict(identity.get("litellm_wheel_versions"))
    record["litellm_wheel_versions"] = {
        **metadata_wheel_versions,
        **record_wheel_versions,
    }
    host_attribution = _extract_session_host_attribution(metadata)
    for key in ("client_ip", "host_name"):
        if not _clean_non_empty_string(record.get(key)):
            record[key] = host_attribution.get(key)


_REQUEST_HEADER_TENANT_LITELLM_REPOSITORY_FRAGMENTS = (
    "harness",
    "validation",
)


def _is_harness_tenant_identity(value: Any) -> bool:
    normalized = _normalize_identity_for_placeholder_check(value)
    if not normalized:
        return False
    return any(fragment in normalized for fragment in _REQUEST_HEADER_TENANT_LITELLM_REPOSITORY_FRAGMENTS)


def _normalize_request_header_tenant_repository(value: Any) -> Optional[str]:
    repository = _normalize_repository_identity(value)
    if repository is None:
        return None
    normalized = repository.lower()
    if any(fragment in normalized for fragment in _REQUEST_HEADER_TENANT_LITELLM_REPOSITORY_FRAGMENTS):
        return "litellm"
    if normalized.endswith("-dev") or "tenant" in normalized:
        return None
    return repository


_REPOSITORY_SOURCE_CODEX_MEMORY_METADATA_MARKERS = (
    ".metadata.",
    ".litellm_metadata.",
)
_REPOSITORY_SOURCE_GENERAL_METADATA_MARKERS = (
    ".metadata.",
    ".litellm_metadata.",
    ".request_metadata.",
    ".user_api_key_metadata.",
)
_REPOSITORY_SOURCE_TEXT_SUFFIXES = (
    ".text.environment_context.cwd",
    ".text.cwd_tag",
    ".text.agents_instructions",
    ".text.workspace_directories",
)


def _normalize_repository_trust_source(value: Any) -> Optional[str]:
    source = _clean_non_empty_string(value)
    if not source:
        return None
    if source.endswith(".codex_memory_workflow"):
        return source[: -len(".codex_memory_workflow")]
    return source


def _repository_source_has_codex_memory_workflow(value: Any) -> bool:
    source = _clean_non_empty_string(value)
    return bool(source and source.endswith(".codex_memory_workflow"))


def _is_repository_source_trusted_common(
    value: Any,
    *,
    allow_general_metadata_markers: bool,
    allow_route_rollup_label: bool,
) -> bool:
    source = _normalize_repository_trust_source(value)
    if not source:
        return False

    if _repository_source_has_codex_memory_workflow(value) and any(
        marker in source for marker in _REPOSITORY_SOURCE_CODEX_MEMORY_METADATA_MARKERS
    ):
        return True
    if source == "tenant_id.request_headers":
        return True
    if source.startswith("request_headers."):
        return True
    if allow_general_metadata_markers and any(
        marker in source for marker in _REPOSITORY_SOURCE_GENERAL_METADATA_MARKERS
    ):
        return True
    if allow_route_rollup_label and source.endswith(".aawm_route_rollup_context.group_header_label"):
        return True
    if "x-codex-turn-metadata" in source and source.endswith(".text.project_path"):
        return True
    return any(source.endswith(marker) for marker in _REPOSITORY_SOURCE_TEXT_SUFFIXES)


def _is_repository_source_trusted_for_tenant(value: Any) -> bool:
    return _is_repository_source_trusted_common(
        value,
        allow_general_metadata_markers=True,
        allow_route_rollup_label=True,
    )


def _is_codex_trace_user_tenant_source(value: Any) -> bool:
    source = _clean_non_empty_string(value)
    if not source:
        return False
    normalized = source.lower()
    return normalized.endswith(".trace_user_id") or normalized == "trace_user_id"


def _is_codex_passthrough_tenant_extraction_context(
    kwargs: Dict[str, Any],
    *,
    metadata: Optional[Dict[str, Any]] = None,
) -> bool:
    litellm_params = kwargs.get("litellm_params") or {}
    metadata = metadata or litellm_params.get("metadata") or {}
    if not isinstance(metadata, dict):
        metadata = {}
    headers = _extract_request_headers_from_kwargs(kwargs)
    return bool(_is_native_codex_passthrough_context(metadata, headers) or _is_codex_client_identity(metadata, headers))


def _is_repository_source_trusted_for_codex_tenant(value: Any) -> bool:
    # Codex trust deliberately omits general metadata markers and the
    # route-rollup label that the general tenant helper accepts.
    return _is_repository_source_trusted_common(
        value,
        allow_general_metadata_markers=False,
        allow_route_rollup_label=False,
    )


def _is_codex_session_history_record(record: Dict[str, Any]) -> bool:
    metadata = record.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
    route_family = _clean_non_empty_string(
        _first_non_none(
            metadata.get("passthrough_route_family"),
            metadata.get("openai_passthrough_route_family"),
        )
    )
    if route_family and route_family.lower() == "codex_responses":
        return True
    client_name = _clean_non_empty_string(_first_non_none(record.get("client_name"), metadata.get("client_name")))
    if client_name and "codex" in client_name.lower():
        return True
    trace_name = _clean_non_empty_string(metadata.get("trace_name"))
    user_agent = _clean_non_empty_string(
        _first_non_none(
            record.get("client_user_agent"),
            metadata.get("client_user_agent"),
            metadata.get("user_agent"),
        )
    )
    return bool(trace_name and trace_name.lower() == "codex" and user_agent and "codex" in user_agent.lower())


def _is_claude_session_history_record(record: Dict[str, Any]) -> bool:
    metadata = record.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
    provider = str(record.get("provider") or "").strip().lower()
    if provider == "anthropic":
        return True
    client_name = _clean_non_empty_string(_first_non_none(record.get("client_name"), metadata.get("client_name")))
    if client_name and "claude" in client_name.lower():
        return True
    trace_name = _clean_non_empty_string(metadata.get("trace_name"))
    return bool(trace_name and "claude" in trace_name.lower())


def _is_claude_project_repository_source(value: Any) -> bool:
    source = _clean_non_empty_string(value)
    return bool(source and source.endswith(".aawm_claude_project"))


def _is_claude_metadata_tenant_source(value: Any) -> bool:
    source = _clean_non_empty_string(value)
    if not source:
        return False
    return source.endswith(".metadata.tenant_id") or source.endswith(".metadata.aawm_tenant_id")


def _claude_project_identity_is_trusted(
    record: Dict[str, Any],
    repository: Optional[str],
    source: Any,
) -> bool:
    if not (repository and _is_claude_session_history_record(record) and _is_claude_project_repository_source(source)):
        return True
    return _is_known_aawm_workspace_repository(repository)


def _codex_repository_source_trusted_for_record(record: Dict[str, Any]) -> bool:
    metadata = record.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
    repository_source = metadata.get("repository_source")
    if _is_codex_session_history_record(record):
        return _is_repository_source_trusted_for_codex_tenant(repository_source)
    return _is_repository_source_trusted_for_tenant(repository_source)


def _clear_untrusted_codex_trace_user_tenant_on_record(
    record: Dict[str, Any],
    tenant_id: str,
) -> bool:
    metadata = record.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
    if not _is_codex_session_history_record(record):
        return False
    if not _is_codex_trace_user_tenant_source(metadata.get("tenant_id_source")):
        trace_user_id = _normalize_repository_identity(metadata.get("trace_user_id"))
        normalized_tenant = _normalize_tenant_identity(tenant_id)
        if not (
            trace_user_id
            and normalized_tenant
            and trace_user_id == normalized_tenant
            and not _codex_tenant_source_trusted_for_record(record)
        ):
            return False

    metadata = dict(metadata)
    metadata["aawm_original_tenant_id"] = tenant_id
    metadata.pop("tenant_id", None)
    metadata["tenant_id_source"] = "trace_user_untrusted"
    metadata["trace_user_tenant_fallback_skipped"] = True
    record["tenant_id"] = None
    record["metadata"] = metadata
    return True


def _mark_codex_trace_user_tenant_skipped(
    record: Dict[str, Any],
    original_tenant_id: Optional[str],
) -> None:
    metadata = record.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
    metadata = dict(metadata)
    if original_tenant_id:
        metadata.setdefault("aawm_original_tenant_id", original_tenant_id)
    metadata.pop("tenant_id", None)
    metadata["tenant_id_source"] = "trace_user_untrusted"
    metadata["trace_user_tenant_fallback_skipped"] = True
    record["tenant_id"] = None
    record["metadata"] = metadata


def _codex_untrusted_repository_reason(metadata: Dict[str, Any]) -> str:
    repository_source = _clean_non_empty_string(metadata.get("repository_source")) or ""
    if ".metadata.repository" in repository_source:
        return "untrusted_metadata_repository_label"
    if ".text." in repository_source or "project_path" in repository_source:
        return "untrusted_prompt_text_repository_candidate"
    return "untrusted_repository_tenant_source"


def _mark_repository_unresolved_metadata(metadata: Dict[str, Any]) -> None:
    metadata["session_history_repository_status"] = "unresolved"
    metadata["session_history_repository_unresolved"] = True
    metadata["session_history_repository_unresolved_reason"] = _codex_untrusted_repository_reason(metadata)


def _session_history_missing_repository_reason(record: Dict[str, Any]) -> str:
    metadata = record.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
    provider = str(record.get("provider") or "").strip().lower()
    client_name = str(metadata.get("client_name") or "").strip().lower()
    if provider == "anthropic" or "claude" in client_name:
        return "no_trusted_claude_project_signal"
    if provider in {"xai", "grok"} or "grok" in str(record.get("model") or "").lower():
        return "no_trusted_grok_project_signal"
    return "no_trusted_repository_signal"


def _mark_missing_repository_unresolved(
    record: Dict[str, Any],
) -> None:
    metadata = record.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
    else:
        metadata = dict(metadata)
    if _metadata_bool(metadata.get("session_history_reporting_excluded")):
        record["metadata"] = metadata
        return
    if metadata.get("session_history_repository_status") == "unresolved":
        record["metadata"] = metadata
        return
    metadata["session_history_repository_status"] = "unresolved"
    metadata["session_history_repository_unresolved"] = True
    metadata["session_history_repository_unresolved_reason"] = _session_history_missing_repository_reason(record)
    record["metadata"] = metadata


def _clear_untrusted_claude_project_repository_on_record(
    record: Dict[str, Any],
    repository: Optional[str],
    repository_source: Any,
) -> Optional[str]:
    if _claude_project_identity_is_trusted(record, repository, repository_source):
        return repository
    metadata = record.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
    else:
        metadata = dict(metadata)
    if repository:
        metadata["aawm_original_repository"] = repository
    source = _clean_non_empty_string(repository_source)
    if source:
        metadata["repository_source_untrusted"] = source
    metadata.pop("repository", None)
    record["metadata"] = metadata
    return None


def _clear_untrusted_claude_metadata_tenant_on_record(
    record: Dict[str, Any],
    tenant_id: str,
) -> bool:
    metadata = record.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
    if not (
        _is_claude_session_history_record(record)
        and not _is_known_aawm_workspace_repository(tenant_id)
        and _is_claude_metadata_tenant_source(metadata.get("tenant_id_source"))
    ):
        return False
    metadata = dict(metadata)
    metadata["aawm_original_tenant_id"] = tenant_id
    tenant_source = _clean_non_empty_string(metadata.get("tenant_id_source"))
    if tenant_source:
        metadata["tenant_id_source_untrusted"] = tenant_source
    metadata.pop("tenant_id", None)
    record["tenant_id"] = None
    record["metadata"] = metadata
    if _normalize_repository_identity(record.get("repository")) is None:
        _mark_missing_repository_unresolved(record)
    return True


def _clear_repository_unresolved_metadata(metadata: Dict[str, Any]) -> None:
    metadata.pop("session_history_repository_unresolved", None)
    metadata.pop("session_history_repository_unresolved_reason", None)
    if metadata.get("session_history_repository_status") == "unresolved":
        metadata.pop("session_history_repository_status", None)


def _mark_codex_repository_tenant_skipped(
    record: Dict[str, Any],
    original_tenant_id: Optional[str] = None,
) -> None:
    metadata = record.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
    metadata = dict(metadata)
    if original_tenant_id:
        metadata["aawm_original_tenant_id"] = original_tenant_id
    metadata.pop("tenant_id", None)
    metadata["tenant_id_source"] = "repository_untrusted"
    metadata["repository_tenant_fallback_skipped"] = True
    _mark_repository_unresolved_metadata(metadata)
    record["tenant_id"] = None
    record["metadata"] = metadata


def _clear_codex_trace_user_tenant_source_on_record(record: Dict[str, Any]) -> bool:
    metadata = record.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
    if not (
        _is_codex_session_history_record(record)
        and _is_codex_trace_user_tenant_source(metadata.get("tenant_id_source"))
    ):
        return False

    original_tenant_id = _clean_non_empty_string(record.get("tenant_id")) or _clean_non_empty_string(
        metadata.get("tenant_id")
    )
    if original_tenant_id is None:
        original_tenant_id = _normalize_repository_identity(metadata.get("trace_user_id"))
    _mark_codex_trace_user_tenant_skipped(record, original_tenant_id)
    return True


def _clear_untrusted_codex_tenant_on_record(
    record: Dict[str, Any],
    tenant_id: str,
) -> bool:
    if not _is_codex_session_history_record(record):
        return False
    metadata = record.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
    if _is_codex_trace_user_tenant_source(metadata.get("tenant_id_source")) or (
        _normalize_repository_identity(metadata.get("trace_user_id")) == tenant_id
        and not _codex_tenant_source_trusted_for_record(record)
    ):
        _mark_codex_trace_user_tenant_skipped(record, tenant_id)
        return True
    if _clear_untrusted_codex_trace_user_tenant_on_record(record, tenant_id):
        return True
    if _clear_untrusted_codex_repository_tenant_on_record(record, tenant_id):
        return True
    if not _codex_tenant_source_trusted_for_record(record):
        _mark_codex_repository_tenant_skipped(record, tenant_id)
        return True
    return False


def _codex_tenant_source_trusted_for_record(record: Dict[str, Any]) -> bool:
    metadata = record.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
    tenant_source = metadata.get("tenant_id_source")
    if _is_codex_trace_user_tenant_source(tenant_source):
        return False
    if tenant_source == "repository":
        return _codex_repository_source_trusted_for_record(record)
    if tenant_source in {
        "request_headers",
        "harness_tenant_repository",
        "agent_context_text",
    }:
        return True
    if isinstance(tenant_source, str) and tenant_source.startswith("request_headers."):
        return True
    if isinstance(tenant_source, str) and ".trace_user_id" in tenant_source:
        return False
    if isinstance(tenant_source, str) and any(
        marker in tenant_source
        for marker in (
            ".metadata.tenant_id",
            ".metadata.aawm_tenant_id",
            ".litellm_metadata.tenant_id",
            ".litellm_metadata.aawm_tenant_id",
        )
    ):
        return _codex_repository_source_trusted_for_record(record)
    return tenant_source is None


def _clear_untrusted_codex_repository_tenant_on_record(
    record: Dict[str, Any],
    tenant_id: str,
) -> bool:
    metadata = record.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
    if not (
        _is_codex_session_history_record(record)
        and metadata.get("tenant_id_source") == "repository"
        and not _codex_repository_source_trusted_for_record(record)
    ):
        return False

    metadata = dict(metadata)
    metadata["aawm_original_tenant_id"] = tenant_id
    metadata.pop("tenant_id", None)
    metadata["tenant_id_source"] = "repository_untrusted"
    metadata["repository_tenant_fallback_skipped"] = True
    _mark_repository_unresolved_metadata(metadata)
    record["tenant_id"] = None
    record["metadata"] = metadata
    return True


def _normalize_session_repository_on_record(record: Dict[str, Any]) -> None:
    repository = _normalize_repository_identity(record.get("repository"))
    metadata = record.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
    repository_source = metadata.get("repository_source")
    if repository is None:
        repository, repository_source = _extract_repository_identity_from_metadata_sources_with_source(
            ("record.metadata", metadata)
        )
        if repository is not None and repository_source:
            metadata = dict(metadata)
            metadata.setdefault("repository_source", repository_source)
            record["metadata"] = metadata
    if repository is None and metadata.get("tenant_id_source") == "request_headers":
        repository = _normalize_request_header_tenant_repository(
            record.get("tenant_id")
        ) or _normalize_request_header_tenant_repository(metadata.get("tenant_id"))
        if repository is not None:
            metadata = dict(metadata)
            metadata.setdefault("repository_source", "tenant_id.request_headers")
            record["metadata"] = metadata
            repository_source = metadata.get("repository_source")
    repository = _clear_untrusted_claude_project_repository_on_record(
        record,
        repository,
        repository_source,
    )
    record["repository"] = repository


def _can_promote_known_codex_repository_to_tenant(
    repository: str,
    metadata: Dict[str, Any],
) -> bool:
    # Bounded relaxation for Codex: generic metadata.repository (or
    # litellm_metadata.repository) may promote to tenant ONLY when the
    # normalized repository label is a known AAWM workspace repo from the
    # conservative built-in allowlist (or AAWM_KNOWN_WORKSPACE_REPOS env).
    # Headers, x-codex-turn-metadata project_path, cwd/workspace text, and
    # other previously trusted sources remain trusted without the name check.
    repo_source = metadata.get("repository_source")
    return _is_known_aawm_workspace_repository(repository) and (
        metadata.get("tenant_id_source") in {"repository_untrusted", "trace_user_untrusted"}
        or metadata.get("trace_user_tenant_fallback_skipped") is True
        or metadata.get("repository_tenant_fallback_skipped") is True
        or (
            isinstance(repo_source, str)
            and (".metadata.repository" in repo_source or "litellm_metadata.repository" in repo_source)
        )
    )


def _normalize_session_tenant_on_record(record: Dict[str, Any]) -> None:
    _clear_codex_trace_user_tenant_source_on_record(record)

    raw_tenant_id = record.get("tenant_id")
    if _is_harness_tenant_identity(raw_tenant_id):
        metadata = record.get("metadata")
        if not isinstance(metadata, dict):
            metadata = {}
        else:
            metadata = dict(metadata)

        original_tenant_id = _clean_non_empty_string(raw_tenant_id)
        if original_tenant_id:
            metadata["aawm_original_tenant_id"] = original_tenant_id
        metadata["aawm_harness_tenant_alias"] = True

        repository = _normalize_repository_identity(
            record.get("repository")
        ) or _normalize_request_header_tenant_repository(raw_tenant_id)
        record["tenant_id"] = repository
        if repository is not None:
            metadata["tenant_id"] = repository
            metadata["tenant_id_source"] = "harness_tenant_repository"
        else:
            metadata.pop("tenant_id", None)
            metadata["tenant_id_source"] = "harness_tenant_excluded"
        record["metadata"] = metadata
        return

    tenant_id = _normalize_tenant_identity(record.get("tenant_id"))
    if tenant_id:
        if _clear_untrusted_claude_metadata_tenant_on_record(record, tenant_id):
            return
        _clear_untrusted_codex_tenant_on_record(record, tenant_id)
        tenant_id = _normalize_tenant_identity(record.get("tenant_id"))
        if not tenant_id:
            # Continue into repository fallback. A stale Codex tenant can be
            # rejected while the same row still has a trusted current repo.
            pass
        else:
            record["tenant_id"] = tenant_id
            return

    repository = _normalize_repository_identity(record.get("repository"))
    if repository is None:
        record["tenant_id"] = None
        _mark_missing_repository_unresolved(record)
        return

    metadata = record.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
    if not _codex_repository_source_trusted_for_record(record):
        if not _can_promote_known_codex_repository_to_tenant(repository, metadata):
            _mark_codex_repository_tenant_skipped(record)
            return

    record["tenant_id"] = repository
    metadata = dict(metadata)
    metadata["tenant_id"] = repository
    metadata["tenant_id_source"] = "repository"
    metadata.pop("repository_tenant_fallback_skipped", None)
    _clear_repository_unresolved_metadata(metadata)
    record["metadata"] = metadata


def _sync_session_history_record_metadata(record: Dict[str, Any]) -> None:  # noqa: PLR0915
    metadata = record.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
    else:
        metadata = dict(metadata)

    reasoning_source = record.get("reasoning_tokens_source")
    if isinstance(reasoning_source, str) and reasoning_source.strip():
        metadata["usage_reasoning_tokens_source"] = reasoning_source

    if record.get("reasoning_tokens_reported") is not None:
        metadata["usage_reasoning_tokens_reported"] = record["reasoning_tokens_reported"]
    else:
        metadata.pop("usage_reasoning_tokens_reported", None)

    if record.get("reasoning_tokens_estimated") is not None:
        metadata["usage_reasoning_tokens_estimated"] = record["reasoning_tokens_estimated"]
    else:
        metadata.pop("usage_reasoning_tokens_estimated", None)

    for field in _PROMPT_OVERHEAD_TOKEN_FIELDS:
        metadata[f"usage_{field}"] = int(record.get(field) or 0)

    metadata["usage_invalid_tool_call_count"] = int(record.get("invalid_tool_call_count") or 0)

    metadata["usage_structured_output_attempted"] = bool(record.get("structured_output_attempted"))
    metadata["usage_structured_output_failed"] = bool(record.get("structured_output_failed"))
    for field, metadata_key in (
        ("structured_output_mode", "usage_structured_output_mode"),
        ("structured_output_schema_hash", "usage_structured_output_schema_hash"),
        (
            "structured_output_failure_reason",
            "usage_structured_output_failure_reason",
        ),
    ):
        value = _clean_non_empty_string(record.get(field))
        if value is None:
            metadata.pop(metadata_key, None)
        else:
            metadata[metadata_key] = value

    metadata["is_compact_summary"] = bool(record.get("is_compact_summary"))
    for field in (
        "compact_summary_source",
        "compact_summary_role",
        "compact_summary_id",
    ):
        value = _clean_non_empty_string(record.get(field))
        if value is None:
            metadata.pop(field, None)
        else:
            metadata[field] = value

    for field in _SESSION_HISTORY_AGENT_SCORE_FLOAT_FIELDS:
        float_value = _safe_float(record.get(field))
        metadata_key = f"usage_{field}"
        if float_value is None:
            metadata.pop(metadata_key, None)
        else:
            metadata[metadata_key] = float_value

    for field in _SESSION_HISTORY_AGENT_SCORE_BOOL_FIELDS:
        bool_value = _optional_metadata_bool(record.get(field))
        metadata_key = f"usage_{field}"
        if bool_value is None:
            metadata.pop(metadata_key, None)
        else:
            metadata[metadata_key] = bool_value

    for field in _SESSION_HISTORY_AGENT_SCORE_INT_FIELDS:
        int_value = _safe_int(record.get(field))
        metadata_key = f"usage_{field}"
        if int_value is None:
            metadata.pop(metadata_key, None)
        else:
            metadata[metadata_key] = int_value

    for field in _SESSION_HISTORY_OUTPUT_CONTRACT_STRING_FIELDS:
        value = _clean_non_empty_string(record.get(field))
        metadata_key = f"usage_{field}"
        if value is None:
            metadata.pop(metadata_key, None)
        else:
            metadata[metadata_key] = value

    for field in _SESSION_HISTORY_OUTPUT_CONTRACT_BOOL_FIELDS:
        bool_value = _optional_metadata_bool(record.get(field))
        metadata_key = f"usage_{field}"
        if bool_value is None:
            metadata.pop(metadata_key, None)
        else:
            metadata[metadata_key] = bool_value

    for field in _SESSION_HISTORY_OUTPUT_CONTRACT_INT_FIELDS:
        int_value = _safe_int(record.get(field))
        metadata_key = f"usage_{field}"
        if int_value is None:
            metadata.pop(metadata_key, None)
        else:
            metadata[metadata_key] = int_value

    for field in _SESSION_HISTORY_OUTPUT_CONTRACT_JSON_FIELDS:
        value = record.get(field)
        metadata_key = f"usage_{field}"
        if value in (None, [], {}):
            metadata.pop(metadata_key, None)
        else:
            metadata[metadata_key] = _json_safe_rate_limit_value(value)

    agent_score_reasons = _normalize_agent_score_reasons(record.get("agent_score_reasons"))
    if agent_score_reasons:
        metadata["usage_agent_score_reasons"] = agent_score_reasons
    else:
        metadata.pop("usage_agent_score_reasons", None)

    provider_family = _normalize_provider_cache_family(
        record.get("provider"),
        str(record.get("model") or ""),
        metadata,
    )
    cache_status = record.get("provider_cache_status")
    if provider_family is not None and isinstance(cache_status, str) and cache_status.strip():
        cache_values: Dict[str, Any] = {
            "provider_cache_attempted": bool(record.get("provider_cache_attempted")),
            "provider_cache_status": cache_status,
            "provider_cache_miss": bool(record.get("provider_cache_miss")),
            "provider_cache_miss_reason": record.get("provider_cache_miss_reason"),
            "provider_cache_miss_token_count": record.get("provider_cache_miss_token_count"),
            "provider_cache_miss_cost_usd": record.get("provider_cache_miss_cost_usd"),
        }
        for suffix, value in cache_values.items():
            generic_key = f"usage_{suffix}"
            provider_key = f"{provider_family}_{suffix}"
            if value is None or value == "":
                metadata.pop(generic_key, None)
                metadata.pop(provider_key, None)
            else:
                metadata[generic_key] = value
                metadata[provider_key] = value

    for key in (
        "litellm_environment",
        "litellm_version",
        "litellm_fork_version",
        "client_name",
        "client_version",
        "client_user_agent",
        "client_ip",
        "host_name",
    ):
        value = _clean_non_empty_string(record.get(key))
        if value is not None:
            metadata[key] = value

    wheel_versions = _coerce_string_dict(record.get("litellm_wheel_versions"))
    if wheel_versions:
        metadata["litellm_wheel_versions"] = wheel_versions

    repository = _normalize_repository_identity(record.get("repository"))
    if repository is not None:
        metadata["repository"] = repository
    else:
        metadata.pop("repository", None)

    tenant_id = _normalize_tenant_identity(record.get("tenant_id"))
    if tenant_id is not None:
        metadata["tenant_id"] = tenant_id
    else:
        metadata.pop("tenant_id", None)
        if metadata.get("trace_user_tenant_fallback_skipped") is True:
            metadata.setdefault("tenant_id_source", "trace_user_untrusted")
        elif metadata.get("repository_tenant_fallback_skipped") is True:
            metadata.setdefault("tenant_id_source", "repository_untrusted")
        else:
            metadata.pop("tenant_id_source", None)

    if _is_numeric_identity_placeholder(metadata.get("trace_user_id")):
        metadata.pop("trace_user_id", None)

    record["metadata"] = metadata


def _normalize_prompt_overhead_state_on_record(record: Dict[str, Any]) -> None:
    for field in _PROMPT_OVERHEAD_TOKEN_FIELDS:
        value = _safe_int(record.get(field))
        record[field] = value if value is not None else 0


def _normalize_invalid_tool_call_state_on_record(record: Dict[str, Any]) -> None:
    value = _safe_int(record.get("invalid_tool_call_count"))
    record["invalid_tool_call_count"] = value if value is not None and value > 0 else 0


def _normalize_structured_output_state_on_record(record: Dict[str, Any]) -> None:
    metadata = record.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}

    attempted_value = record.get("structured_output_attempted")
    failed_value = record.get("structured_output_failed")
    attempted = (
        _metadata_bool(attempted_value)
        if attempted_value is not None
        else _metadata_bool(metadata.get("usage_structured_output_attempted"))
    )
    failed = (
        _metadata_bool(failed_value)
        if failed_value is not None
        else _metadata_bool(metadata.get("usage_structured_output_failed"))
    )
    if failed:
        attempted = True

    record["structured_output_attempted"] = attempted
    record["structured_output_failed"] = failed
    record["structured_output_mode"] = _first_non_empty_string(
        record.get("structured_output_mode"),
        metadata.get("usage_structured_output_mode"),
        metadata.get("structured_output_mode"),
    )
    record["structured_output_schema_hash"] = _first_non_empty_string(
        record.get("structured_output_schema_hash"),
        metadata.get("usage_structured_output_schema_hash"),
        metadata.get("structured_output_schema_hash"),
    )
    record["structured_output_failure_reason"] = _first_non_empty_string(
        record.get("structured_output_failure_reason"),
        metadata.get("usage_structured_output_failure_reason"),
        metadata.get("structured_output_failure_reason"),
    )
    if not failed:
        record["structured_output_failure_reason"] = None


def _normalize_compact_summary_state_on_record(record: Dict[str, Any]) -> None:
    metadata = record.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}

    is_compact = (
        _optional_metadata_bool(record.get("is_compact_summary"))
        if record.get("is_compact_summary") is not None
        else _optional_metadata_bool(metadata.get("is_compact_summary"))
    )
    record["is_compact_summary"] = bool(is_compact)
    record["compact_summary_source"] = _first_non_empty_string(
        record.get("compact_summary_source"),
        metadata.get("compact_summary_source"),
    )
    record["compact_summary_role"] = _first_non_empty_string(
        record.get("compact_summary_role"),
        metadata.get("compact_summary_role"),
    )
    record["compact_summary_id"] = _first_non_empty_string(
        record.get("compact_summary_id"),
        metadata.get("compact_summary_id"),
    )

    if record["is_compact_summary"]:
        record["compact_summary_role"] = record["compact_summary_role"] or "event"


def _optional_metadata_bool(value: Any) -> Optional[bool]:
    if value is None or value == "":
        return None
    return _metadata_bool(value)


def _normalize_agent_score_reasons(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return {}
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            return {}
        return dict(parsed) if isinstance(parsed, dict) else {}
    return {}


def _append_agent_quality_text(
    *,
    role: Optional[str],
    content: Any,
    user_texts: List[str],
    assistant_texts: List[str],
    tool_result_texts: List[str],
) -> None:
    text = _content_to_text(content).strip()
    if not text:
        return
    role_lower = str(role or "").lower()
    if role_lower in {"assistant", "model"}:
        assistant_texts.append(text)
    elif role_lower in {"tool", "function"}:
        tool_result_texts.append(text)
    else:
        user_texts.append(text)


def _append_agent_quality_command_from_arguments(
    *,
    commands: List[AgentQualityCommand],
    name: str,
    arguments: Any,
) -> None:
    command_text = _extract_command_text_from_tool_arguments(arguments)
    if not command_text:
        return
    commands.append(
        AgentQualityCommand(
            name=name,
            command=command_text,
            affected_paths=tuple(_extract_file_paths_from_tool_arguments(arguments)),
        )
    )


def _append_agent_quality_commands_from_message(
    *,
    message: Dict[str, Any],
    commands: List[AgentQualityCommand],
) -> None:
    content = message.get("content")
    content_blocks = content if isinstance(content, list) else [content]
    for block in content_blocks:
        if not isinstance(block, dict):
            continue
        block_type = str(block.get("type") or "").lower()
        if block_type not in {"tool_use", "function_call", "custom_tool_call"}:
            continue
        arguments = block.get("input")
        if arguments is None:
            arguments = block.get("arguments")
        _append_agent_quality_command_from_arguments(
            commands=commands,
            name=str(block.get("name") or block_type or "tool"),
            arguments=arguments,
        )

    tool_calls = message.get("tool_calls")
    if isinstance(tool_calls, list):
        for tool_call in tool_calls:
            if not isinstance(tool_call, dict):
                continue
            function = tool_call.get("function")
            if isinstance(function, dict):
                _append_agent_quality_command_from_arguments(
                    commands=commands,
                    name=str(function.get("name") or tool_call.get("type") or "tool"),
                    arguments=function.get("arguments"),
                )
                continue
            _append_agent_quality_command_from_arguments(
                commands=commands,
                name=str(tool_call.get("name") or tool_call.get("type") or "tool"),
                arguments=tool_call.get("arguments") or tool_call.get("input"),
            )


def _collect_agent_quality_context_from_request_body(
    request_body: Any,
) -> Tuple[List[str], List[str], List[str], List[AgentQualityCommand]]:
    user_texts: List[str] = []
    assistant_texts: List[str] = []
    tool_result_texts: List[str] = []
    commands: List[AgentQualityCommand] = []
    if not isinstance(request_body, dict):
        return user_texts, assistant_texts, tool_result_texts, commands

    messages = request_body.get("messages")
    if isinstance(messages, list):
        for message in messages:
            if not isinstance(message, dict):
                continue
            role = str(message.get("role") or "")
            content = message.get("content")
            if role.lower() == "user" and isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "tool_result":
                        tool_result_texts.append(_content_to_text(block.get("content")))
            if role.lower() in {"assistant", "model"}:
                _append_agent_quality_commands_from_message(
                    message=message,
                    commands=commands,
                )
            _append_agent_quality_text(
                role=role,
                content=content,
                user_texts=user_texts,
                assistant_texts=assistant_texts,
                tool_result_texts=tool_result_texts,
            )

    input_items = request_body.get("input")
    if isinstance(input_items, list):
        for item in input_items:
            if not isinstance(item, dict):
                user_texts.append(_content_to_text(item))
                continue
            item_type = str(item.get("type") or "").lower()
            role = str(item.get("role") or "")
            if item_type in {"message", ""} or role:
                _append_agent_quality_text(
                    role=role or "user",
                    content=item.get("content"),
                    user_texts=user_texts,
                    assistant_texts=assistant_texts,
                    tool_result_texts=tool_result_texts,
                )
                continue
            if item_type in {"function_call_output", "tool_result"}:
                tool_result_texts.append(_content_to_text(item.get("output")))
                continue
            if item_type in {"function_call", "tool_use", "custom_tool_call"}:
                command_text = _extract_command_text_from_tool_arguments(item.get("arguments") or item.get("input"))
                if command_text:
                    commands.append(
                        AgentQualityCommand(
                            name=str(item.get("name") or item_type),
                            command=command_text,
                            affected_paths=tuple(
                                _extract_file_paths_from_tool_arguments(item.get("arguments") or item.get("input"))
                            ),
                        )
                    )

    return user_texts, assistant_texts, tool_result_texts, commands


def _collect_agent_quality_response_texts(result: Any) -> List[str]:
    assistant_texts: List[str] = []
    message = _extract_first_response_message(result)
    if message is not None:
        text = _content_to_text(_maybe_get(message, "content")).strip()
        if text:
            assistant_texts.append(text)

    output_items = _maybe_get(result, "output")
    if isinstance(output_items, list):
        for item in output_items:
            if not isinstance(item, dict):
                continue
            role = str(item.get("role") or "")
            if item.get("type") == "message" or role == "assistant":
                text = _content_to_text(item.get("content")).strip()
                if text:
                    assistant_texts.append(text)
    return assistant_texts


def _agent_quality_commands_from_tool_activity(
    tool_activity: List[Dict[str, Any]],
) -> List[AgentQualityCommand]:
    commands: List[AgentQualityCommand] = []
    for item in tool_activity:
        if not isinstance(item, dict):
            continue
        command_text = item.get("command_text")
        if not isinstance(command_text, str) or not command_text.strip():
            continue
        affected_paths = tuple(
            value
            for value in (list(item.get("file_paths_modified") or []) + list(item.get("file_paths_read") or []))
            if isinstance(value, str)
        )
        commands.append(
            AgentQualityCommand(
                name=str(item.get("tool_name") or ""),
                command=command_text,
                affected_paths=affected_paths,
            )
        )
    return commands


def _apply_runtime_agent_quality_scores(
    *,
    record: Dict[str, Any],
    request_body: Any,
    result: Any,
    tool_activity: List[Dict[str, Any]],
) -> None:
    user_texts, assistant_texts, tool_result_texts, commands = _collect_agent_quality_context_from_request_body(
        request_body
    )
    assistant_texts.extend(_collect_agent_quality_response_texts(result))
    commands.extend(_agent_quality_commands_from_tool_activity(tool_activity))

    start_time = record.get("start_time")
    end_time = record.get("end_time")
    elapsed_ms: Optional[float] = None
    if isinstance(start_time, datetime) and isinstance(end_time, datetime):
        elapsed_ms = max(0.0, (end_time - start_time).total_seconds() * 1000)

    task_progress = bool(
        record.get("output_tokens")
        or record.get("tool_call_count")
        or record.get("file_modified_count")
        or record.get("git_commit_count")
    )
    result_scores = score_agent_quality_context(
        user_texts=user_texts,
        assistant_texts=assistant_texts,
        tool_result_texts=tool_result_texts,
        commands=commands,
        input_tokens=_safe_int(record.get("input_tokens")) or 0,
        output_tokens=_safe_int(record.get("output_tokens")) or 0,
        elapsed_ms=elapsed_ms,
        task_progress=task_progress,
    )
    for field, value in result_scores.fields.items():
        if record.get(field) is None:
            record[field] = value

    reasons = _normalize_agent_score_reasons(record.get("agent_score_reasons"))
    for key, value in result_scores.reasons.items():
        if value:
            reasons[key] = value
    record["agent_score_reasons"] = reasons


def _normalize_agent_score_state_on_record(record: Dict[str, Any]) -> None:
    metadata = record.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}

    for field in _SESSION_HISTORY_AGENT_SCORE_FLOAT_FIELDS:
        record[field] = _first_non_none(
            _safe_float(record.get(field)),
            _safe_float(metadata.get(f"usage_{field}")),
            _safe_float(metadata.get(field)),
        )

    for field in _SESSION_HISTORY_AGENT_SCORE_BOOL_FIELDS:
        record[field] = _first_non_none(
            _optional_metadata_bool(record.get(field)),
            _optional_metadata_bool(metadata.get(f"usage_{field}")),
            _optional_metadata_bool(metadata.get(field)),
        )

    for field in _SESSION_HISTORY_AGENT_SCORE_INT_FIELDS:
        value = _first_non_none(
            _safe_int(record.get(field)),
            _safe_int(metadata.get(f"usage_{field}")),
            _safe_int(metadata.get(field)),
        )
        record[field] = value if value is not None and value >= 0 else None

    for field in _SESSION_HISTORY_OUTPUT_CONTRACT_STRING_FIELDS:
        record[field] = _first_non_empty_string(
            record.get(field),
            metadata.get(f"usage_{field}"),
            metadata.get(field),
        )

    for field in _SESSION_HISTORY_OUTPUT_CONTRACT_BOOL_FIELDS:
        record[field] = _first_non_none(
            _optional_metadata_bool(record.get(field)),
            _optional_metadata_bool(metadata.get(f"usage_{field}")),
            _optional_metadata_bool(metadata.get(field)),
        )

    for field in _SESSION_HISTORY_OUTPUT_CONTRACT_INT_FIELDS:
        value = _first_non_none(
            _safe_int(record.get(field)),
            _safe_int(metadata.get(f"usage_{field}")),
            _safe_int(metadata.get(field)),
        )
        record[field] = value if value is not None and value >= 0 else None

    for field in _SESSION_HISTORY_OUTPUT_CONTRACT_JSON_FIELDS:
        record[field] = _first_non_none(
            record.get(field),
            metadata.get(f"usage_{field}"),
            metadata.get(field),
        )

    metadata_reasons = _normalize_agent_score_reasons(
        _first_non_none(
            metadata.get("usage_agent_score_reasons"),
            metadata.get("agent_score_reasons"),
        )
    )
    record_reasons = _normalize_agent_score_reasons(record.get("agent_score_reasons"))
    record["agent_score_reasons"] = {
        **metadata_reasons,
        **record_reasons,
    }


def _normalize_session_latency_state_on_record(record: Dict[str, Any]) -> None:
    derived_latency = _build_session_history_latency_breakdown(
        metadata=record.get("metadata"),
        start_time=record.get("start_time"),
        end_time=record.get("end_time"),
    )
    for field in _SESSION_HISTORY_LATENCY_FIELDS:
        explicit_value = _nonnegative_float_or_none(record.get(field))
        record[field] = explicit_value if explicit_value is not None else derived_latency.get(field)


_GEMINI_CONTROL_PLANE_METHOD_LABELS = {
    "fetchadmincontrols": "google-fetch-admin-controls",
    "listexperiments": "google-list-experiments",
    "loadcodeassist": "google-load-code-assist",
    "retrieveuserquota": "google-retrieve-user-quota",
}
_GEMINI_CONTROL_PLANE_METHOD_NAMES = {
    "fetchadmincontrols": "fetchAdminControls",
    "listexperiments": "listExperiments",
    "loadcodeassist": "loadCodeAssist",
    "retrieveuserquota": "retrieveUserQuota",
}


def _extract_gemini_control_plane_method_from_record(
    record: Dict[str, Any],
    metadata: Dict[str, Any],
) -> Optional[str]:
    candidates = (
        record.get("call_type"),
        record.get("model"),
        metadata.get("user_api_key_request_route"),
        metadata.get("passthrough_route_family"),
        metadata.get("aawm_local_route"),
        metadata.get("aawm_local_endpoint"),
    )
    for candidate in candidates:
        if not isinstance(candidate, str):
            continue
        candidate_lower = candidate.lower()
        for method_lower in _GEMINI_CONTROL_PLANE_METHOD_LABELS:
            if method_lower in candidate_lower:
                return method_lower
    return None


def _session_history_record_provider_usage_token_total(record: Dict[str, Any]) -> int:
    total = 0
    for field in (
        "input_tokens",
        "output_tokens",
        "cache_read_input_tokens",
        "cache_creation_input_tokens",
        "reasoning_tokens_reported",
    ):
        value = _safe_int(record.get(field))
        if value is not None and value > 0:
            total += value
    return total


def _classify_zero_token_session_history_record(record: Dict[str, Any]) -> None:
    if _session_history_record_provider_usage_token_total(record) > 0:
        return

    metadata = record.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
    else:
        metadata = dict(metadata)

    provider = str(record.get("provider") or "").strip().lower()
    zero_token_class: Optional[str] = None
    zero_token_reason: Optional[str] = None

    gemini_control_plane_method = _extract_gemini_control_plane_method_from_record(
        record,
        metadata,
    )
    has_gemini_quota_payload = isinstance(metadata.get("google_retrieve_user_quota"), dict)
    if provider in {"gemini", "google"} and (
        metadata.get("aawm_rate_limit_observation_only") is True
        or has_gemini_quota_payload
        or gemini_control_plane_method is not None
    ):
        zero_token_class = "non_usage_rate_limit_observation"
        zero_token_reason = "gemini_control_plane_rate_limit_payload"
        if gemini_control_plane_method is not None:
            metadata.setdefault(
                "gemini_control_plane_method",
                _GEMINI_CONTROL_PLANE_METHOD_NAMES[gemini_control_plane_method],
            )
            metadata["gemini_control_plane_excluded"] = True
            model = _clean_non_empty_string(record.get("model"))
            if model is None or model.lower() in {"unknown", "null", "none"}:
                record["model"] = _GEMINI_CONTROL_PLANE_METHOD_LABELS[gemini_control_plane_method]
    elif (
        provider == "gemini"
        and metadata.get("codex_adapter_output_shape") == "openai_responses"
        and _safe_int(metadata.get("aawm_stream_chunk_count")) is not None
    ):
        zero_token_class = "empty_provider_response_no_usage"
        zero_token_reason = "gemini_code_assist_adapter_empty_response"
    elif metadata.get("source_status") == "failure":
        zero_token_class = "failed_observation_no_usage"
        zero_token_reason = "langfuse_observation_failed_without_usage"
    elif (
        provider in {"xai", "grok"}
        and str(record.get("model") or "").strip().lower() == "unknown"
        and str(metadata.get("passthrough_route_family") or "").strip().lower() == "grok_cli_chat_proxy"
    ):
        inferred_model = _first_non_empty_string(
            metadata.get("grok_model_override"),
            metadata.get("model_group"),
            record.get("model_group"),
        )
        if inferred_model is None or inferred_model.lower() in {
            "unknown",
            "null",
            "none",
        }:
            inferred_model = "grok-build"
            metadata.setdefault("grok_side_channel_model_defaulted", True)
            metadata.setdefault(
                "grok_side_channel_model_default_reason",
                "grok_cli_side_channel_without_request_model",
            )

        record["model"] = inferred_model
        if _clean_non_empty_string(record.get("model_group")) is None:
            record["model_group"] = inferred_model
        metadata.setdefault("model_group", inferred_model)
        zero_token_class = "grok_cli_side_channel_no_usage"
        zero_token_reason = "grok_side_channel_without_model_usage"
        metadata["session_history_reporting_excluded"] = True
        metadata["session_history_reporting_exclusion_reason"] = zero_token_reason
        metadata["grok_side_channel_excluded"] = True

    if zero_token_class is not None:
        metadata.setdefault("session_history_usage_record", False)
        metadata.setdefault("session_history_zero_token_class", zero_token_class)
        metadata.setdefault("d1_140_zero_token_class", zero_token_class)
        if zero_token_reason is not None:
            metadata.setdefault("d1_140_zero_token_reason", zero_token_reason)

    record["metadata"] = metadata


def _normalize_session_history_record(record: Dict[str, Any]) -> Dict[str, Any]:
    metadata = record.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
    record["provider_response_id"] = _first_non_empty_string(
        record.get("provider_response_id"),
        metadata.get("provider_response_id"),
        metadata.get("response_id"),
    )
    _normalize_agent_id_on_record(record)
    _normalize_inbound_model_alias_on_record(record)
    _normalize_reasoning_state(record)
    _normalize_provider_cache_state_on_record(record)
    _normalize_invalid_tool_call_state_on_record(record)
    _normalize_structured_output_state_on_record(record)
    _normalize_compact_summary_state_on_record(record)
    _normalize_reporting_exclusion_state_on_record(record)
    _normalize_agent_score_state_on_record(record)
    _normalize_prompt_overhead_state_on_record(record)
    _normalize_session_runtime_identity_on_record(record)
    _apply_claude_auto_review_identity_to_record(record)
    _normalize_session_repository_on_record(record)
    _normalize_session_tenant_on_record(record)
    _normalize_session_latency_state_on_record(record)
    _normalize_sensitive_config_change_state_on_record(record)
    _extract_inline_tool_definition_snapshot_from_metadata(record)
    _classify_zero_token_session_history_record(record)
    _sync_session_history_record_metadata(record)
    return record


def _normalize_agent_id_on_record(record: Dict[str, Any]) -> None:
    metadata = dict(record.get("metadata") or {}) if isinstance(record.get("metadata"), dict) else {}
    record["metadata"] = metadata
    disallowed_values = _agent_id_disallowed_values(
        record.get("session_id"),
        record.get("trace_id"),
        record.get("litellm_call_id"),
        record.get("agent_name"),
        record.get("tenant_id"),
        record.get("repository"),
        metadata.get("session_id"),
        metadata.get("trace_id"),
        metadata.get("trace_user_id"),
        metadata.get("agent_name"),
        metadata.get("tenant_id"),
        metadata.get("repository"),
    )
    agent_id = _normalize_agent_id_identity(
        _first_non_empty_string(record.get("agent_id"), metadata.get("agent_id")),
        disallowed_values=disallowed_values,
    )
    record["agent_id"] = agent_id
    if agent_id:
        metadata["agent_id"] = agent_id
    else:
        metadata.pop("agent_id", None)
        metadata.pop("agent_id_source", None)


def _normalize_inbound_model_alias_on_record(record: Dict[str, Any]) -> None:
    metadata = record.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}

    record["inbound_model_alias"] = _first_non_empty_string(
        record.get("inbound_model_alias"),
        metadata.get("model_alias_label"),
        metadata.get("requested_model_alias"),
        metadata.get("codex_auto_agent_alias"),
        metadata.get("anthropic_auto_agent_alias"),
        metadata.get("aawm_auto_agent_alias"),
        record.get("model"),
    )


def _extract_inline_tool_definition_snapshot_from_metadata(
    record: Dict[str, Any],
) -> None:
    metadata = record.get("metadata")
    if not isinstance(metadata, dict):
        return

    snapshot = metadata.pop(_AAWM_TOOL_DEFINITION_SNAPSHOT_METADATA_KEY, None)
    if isinstance(snapshot, list) and snapshot:
        record.setdefault(_AAWM_TOOL_DEFINITION_SNAPSHOT_METADATA_KEY, snapshot)
    record["metadata"] = metadata


def _normalize_reporting_exclusion_state_on_record(record: Dict[str, Any]) -> None:
    metadata = record.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
    else:
        metadata = dict(metadata)

    call_type = str(record.get("call_type") or "").strip().lower()
    model = str(record.get("model") or "").strip().lower()
    source = str(metadata.get("source") or "").strip().lower()

    if call_type == "codex_transcript" or source == "codex_transcript":
        metadata["session_history_usage_record"] = False
        metadata["session_history_reporting_excluded"] = True
        metadata["session_history_reporting_exclusion_reason"] = "synthetic_codex_transcript"

    if model == "unknown":
        metadata["session_history_model_unresolved"] = True
        metadata["session_history_model_reporting_excluded"] = True
        metadata.setdefault(
            "session_history_model_unresolved_reason",
            "missing_source_model_evidence",
        )

    record["metadata"] = metadata


_TOOL_ACTIVITY_READ_NAMES = {
    "read",
    "view",
    "cat",
    "grep",
    "glob",
    "ls",
    "listdir",
    "list_files",
    "search",
    "fetch",
    "webfetch",
    "web_fetch",
    "notebookread",
}
_TOOL_ACTIVITY_MODIFY_NAMES = {
    "write",
    "edit",
    "replace",
    "replacement",
    "multiedit",
    "apply_patch",
    "applypatch",
    "notebookedit",
    "notebookwrite",
}
_TOOL_ACTIVITY_COMMAND_NAMES = {
    "bash",
    "shell",
    "terminal",
    "run",
    "exec",
    "exec_command",
    "browser_run_code",
}
_TOOL_ACTIVITY_SKIP_PATH_KEYS = {
    "content",
    "old_str",
    "new_str",
    "replacement",
    "patch",
    "command",
    "cmd",
    "description",
    "thinking",
    "reason",
}
_APPLY_PATCH_FILE_RE = re.compile(r"^\*\*\* (?:Update|Add|Delete) File: (.+)$", re.MULTILINE)
_APPLY_PATCH_MOVE_TO_RE = re.compile(r"^\*\*\* Move to: (.+)$", re.MULTILINE)
_GIT_COMMAND_RE = re.compile(r"(?<!\S)git\b(?P<args>[^;&|]*)")
_GIT_GLOBAL_OPTIONS_WITH_VALUES = {
    "-C",
    "-c",
    "--git-dir",
    "--work-tree",
    "--namespace",
    "--exec-path",
    "--config-env",
}
_TOOL_ACTIVITY_COMMAND_TEXT_KEYS = (
    "command",
    "cmd",
    "raw_text",
    "input",
    "script",
    "shell",
    "bash",
    "code",
    "text",
)
_TOOL_ACTIVITY_COMMAND_TEXT_SKIP_KEYS = {
    "description",
    "reason",
    "thinking",
    "title",
    "summary",
}
_SENSITIVE_CONFIG_CHANGE_FIELDS = (
    "changed_pre_commit_config",
    "changed_env_file",
    "changed_pyproject_toml",
    "changed_gitignore",
)
_SENSITIVE_CONFIG_ENV_REDACTION = "[redacted_sensitive_config_file_content]"
_SENSITIVE_CONFIG_ENV_REDACT_ARGUMENT_KEYS = {
    "bash",
    "cmd",
    "code",
    "command",
    "content",
    "input",
    "new_str",
    "old_str",
    "patch",
    "raw_text",
    "replacement",
    "script",
    "shell",
    "text",
    "value",
}
_SENSITIVE_CONFIG_ENV_COMMAND_RE = re.compile(
    r"(?<![A-Za-z0-9_./-])\.env[A-Za-z0-9._-]*(?![A-Za-z0-9_/-])",
    re.IGNORECASE,
)


def _dedupe_strings(values: List[str]) -> List[str]:
    seen: set[str] = set()
    result: List[str] = []
    for value in values:
        stripped = str(value).strip()
        if not stripped or stripped in seen:
            continue
        seen.add(stripped)
        result.append(stripped)
    return result


def _normalize_changed_file_path(value: Any) -> Optional[str]:
    if value is None:
        return None
    normalized = str(value).strip().strip("'\"").replace("\\", "/")
    while normalized.startswith("./"):
        normalized = normalized[2:]
    if not normalized:
        return None
    return normalized


def _changed_file_basename(value: Any) -> Optional[str]:
    normalized = _normalize_changed_file_path(value)
    if normalized is None:
        return None
    return normalized.rstrip("/").rsplit("/", 1)[-1]


def _sensitive_config_change_flags_from_paths(paths: List[str]) -> Dict[str, bool]:
    flags = {field: False for field in _SENSITIVE_CONFIG_CHANGE_FIELDS}
    for path in _dedupe_strings(paths):
        basename = _changed_file_basename(path)
        if not basename:
            continue
        basename_lower = basename.lower()
        if basename_lower in {".pre-commit-config.yaml", ".pre-commit-config.yml"}:
            flags["changed_pre_commit_config"] = True
        if basename_lower.startswith(".env"):
            flags["changed_env_file"] = True
        if basename_lower == "pyproject.toml":
            flags["changed_pyproject_toml"] = True
        if basename_lower == ".gitignore":
            flags["changed_gitignore"] = True
    return flags


def _text_mentions_env_file(value: Any) -> bool:
    return isinstance(value, str) and bool(_SENSITIVE_CONFIG_ENV_COMMAND_RE.search(value))


def _redact_sensitive_config_argument_value(value: Any) -> Any:
    if isinstance(value, dict):
        redacted: Dict[str, Any] = {}
        for key, nested_value in value.items():
            key_lower = str(key).lower()
            if key_lower in _SENSITIVE_CONFIG_ENV_REDACT_ARGUMENT_KEYS:
                redacted[key] = _SENSITIVE_CONFIG_ENV_REDACTION
            else:
                redacted[key] = _redact_sensitive_config_argument_value(nested_value)
        return redacted
    if isinstance(value, list):
        return [_redact_sensitive_config_argument_value(item) for item in value]
    return value


def _sanitize_tool_activity_arguments_for_sensitive_config(
    arguments: Any,
    *,
    file_paths_modified: List[str],
    command_text: Optional[str] = None,
) -> Any:
    flags = _sensitive_config_change_flags_from_paths(file_paths_modified)
    if not flags["changed_env_file"] and not _text_mentions_env_file(command_text):
        return arguments
    if isinstance(arguments, str):
        return _SENSITIVE_CONFIG_ENV_REDACTION
    return _redact_sensitive_config_argument_value(arguments)


def _normalize_sensitive_config_change_state_on_record(record: Dict[str, Any]) -> None:
    modified_paths: List[str] = []
    tool_activity = record.get("tool_activity")
    if not isinstance(tool_activity, list):
        return
    if isinstance(tool_activity, list):
        for item in tool_activity:
            if not isinstance(item, dict):
                continue
            modified_paths.extend(value for value in (item.get("file_paths_modified") or []) if isinstance(value, str))

    flags = _sensitive_config_change_flags_from_paths(modified_paths)
    for field, derived_value in flags.items():
        record[field] = bool(record.get(field)) or derived_value


def _parse_tool_arguments(arguments: Any) -> Any:
    if arguments is None or arguments == "":
        return {}
    if isinstance(arguments, (dict, list)):
        return arguments
    if isinstance(arguments, str):
        stripped = arguments.strip()
        if not stripped:
            return {}
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            return {"raw_text": stripped}
    return {"value": arguments}


def _is_empty_claude_read_pages_value(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, list):
        return len(value) == 0
    return False


def _sanitize_tool_activity_arguments(tool_name: str, arguments: Any) -> Any:
    if tool_name != "Read" or not isinstance(arguments, dict):
        return arguments
    if "pages" not in arguments:
        return arguments
    if not _is_empty_claude_read_pages_value(arguments.get("pages")):
        return arguments

    sanitized_arguments = dict(arguments)
    sanitized_arguments.pop("pages", None)
    return sanitized_arguments


def _extract_paths_from_patch_text(text: str) -> List[str]:
    if not isinstance(text, str) or not text.strip():
        return []
    paths = _APPLY_PATCH_FILE_RE.findall(text) + _APPLY_PATCH_MOVE_TO_RE.findall(text)
    return _dedupe_strings(paths)


def _collect_file_paths_from_value(value: Any) -> List[str]:
    collected: List[str] = []
    if isinstance(value, str):
        stripped = value.strip()
        if stripped:
            collected.append(stripped)
    elif isinstance(value, list):
        for item in value:
            collected.extend(_collect_file_paths_from_value(item))
    elif isinstance(value, dict):
        for nested_key, nested_value in list(value.items()):
            nested_key_lower = str(nested_key).lower()
            if nested_key_lower in _TOOL_ACTIVITY_SKIP_PATH_KEYS:
                continue
            if any(token in nested_key_lower for token in ("path", "file")):
                collected.extend(_collect_file_paths_from_value(nested_value))
    return collected


def _extract_file_paths_from_tool_arguments(arguments: Any) -> List[str]:
    parsed_arguments = _parse_tool_arguments(arguments)
    if isinstance(parsed_arguments, str):
        return []
    return _dedupe_strings(_collect_file_paths_from_value(parsed_arguments))


def _extract_command_text_from_tool_arguments(arguments: Any) -> Optional[str]:
    parsed_arguments = _parse_tool_arguments(arguments)
    command_text = _find_command_text_in_value(parsed_arguments)
    if command_text is not None:
        return command_text
    if isinstance(parsed_arguments, str) and parsed_arguments.strip():
        return parsed_arguments.strip()
    return None


def _find_command_text_in_value(value: Any, *, depth: int = 0) -> Optional[str]:
    if depth > 4:
        return None
    if isinstance(value, str) and value.strip():
        return value.strip()
    if isinstance(value, list):
        for item in value:
            command_text = _find_command_text_in_value(item, depth=depth + 1)
            if command_text is not None:
                return command_text
        return None
    if not isinstance(value, dict):
        return None

    for key in _TOOL_ACTIVITY_COMMAND_TEXT_KEYS:
        candidate = value.get(key)
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()

    for key, nested_value in list(value.items()):
        if str(key).lower() in _TOOL_ACTIVITY_COMMAND_TEXT_SKIP_KEYS:
            continue
        command_text = _find_command_text_in_value(nested_value, depth=depth + 1)
        if command_text is not None:
            return command_text
    return None


def _count_git_subcommand(command_text: str, subcommand: str) -> int:
    count = 0
    for match in _GIT_COMMAND_RE.finditer(command_text):
        command = f"git{match.group('args') or ''}"
        try:
            tokens = shlex.split(command)
        except ValueError:
            tokens = command.split()
        index = 1
        while index < len(tokens):
            token = tokens[index]
            if token in _GIT_GLOBAL_OPTIONS_WITH_VALUES:
                index += 2
                continue
            if any(token.startswith(f"{option}=") for option in _GIT_GLOBAL_OPTIONS_WITH_VALUES):
                index += 1
                continue
            if token.startswith("-"):
                index += 1
                continue
            if token == subcommand:
                count += 1
            break
    return count


def _classify_tool_kind(tool_name: str) -> str:
    normalized_name = (tool_name or "").strip().lower()
    if normalized_name.startswith("mcp__"):
        return "mcp"
    if normalized_name in _TOOL_ACTIVITY_COMMAND_NAMES or any(
        token in normalized_name for token in ("bash", "shell", "terminal")
    ):
        return "command"
    if normalized_name in _TOOL_ACTIVITY_MODIFY_NAMES or any(
        token in normalized_name for token in ("write", "edit", "patch")
    ):
        return "modify"
    if normalized_name in _TOOL_ACTIVITY_READ_NAMES or any(
        token in normalized_name for token in ("read", "view", "grep", "glob", "search", "fetch")
    ):
        return "read"
    return "other"


def _build_tool_activity_entry(
    *,
    tool_index: int,
    tool_name: str,
    arguments: Any,
    tool_call_id: Optional[str] = None,
    source: Optional[str] = None,
) -> Dict[str, Any]:
    parsed_arguments = _parse_tool_arguments(arguments)
    parsed_arguments = _sanitize_tool_activity_arguments(tool_name, parsed_arguments)
    tool_kind = _classify_tool_kind(tool_name)
    file_paths_read: List[str] = []
    file_paths_modified: List[str] = []
    command_text: Optional[str] = None

    if tool_kind == "read":
        file_paths_read = _extract_file_paths_from_tool_arguments(parsed_arguments)
    elif tool_kind == "modify":
        file_paths_modified = _extract_file_paths_from_tool_arguments(parsed_arguments)
        if tool_name.strip().lower() in {"apply_patch", "applypatch"}:
            patch_text = _extract_command_text_from_tool_arguments(parsed_arguments)
            if patch_text:
                file_paths_modified = _dedupe_strings(file_paths_modified + _extract_paths_from_patch_text(patch_text))
    elif tool_kind == "command":
        command_text = _extract_command_text_from_tool_arguments(parsed_arguments)

    if command_text is None and tool_name.strip().lower() in {"apply_patch", "applypatch"}:
        command_text = _extract_command_text_from_tool_arguments(parsed_arguments)

    git_commit_count = 0
    git_push_count = 0
    if isinstance(command_text, str) and command_text:
        git_commit_count = _count_git_subcommand(command_text, "commit")
        git_push_count = _count_git_subcommand(command_text, "push")

    sensitive_config_flags = _sensitive_config_change_flags_from_paths(file_paths_modified)
    stored_arguments = _sanitize_tool_activity_arguments_for_sensitive_config(
        parsed_arguments,
        file_paths_modified=file_paths_modified,
        command_text=command_text,
    )
    if (
        sensitive_config_flags["changed_env_file"] or _text_mentions_env_file(command_text)
    ) and command_text is not None:
        command_text = _SENSITIVE_CONFIG_ENV_REDACTION

    return {
        "tool_index": tool_index,
        "tool_call_id": tool_call_id,
        "tool_name": tool_name,
        "tool_kind": tool_kind,
        "file_paths_read": _dedupe_strings(file_paths_read),
        "file_paths_modified": _dedupe_strings(file_paths_modified),
        "git_commit_count": git_commit_count,
        "git_push_count": git_push_count,
        "command_text": command_text,
        "arguments": stored_arguments,
        "metadata": {"source": source} if source else {},
    }


def _extract_tool_activity_from_message(message: Any) -> List[Dict[str, Any]]:
    activity: List[Dict[str, Any]] = []
    raw_tool_calls = _maybe_get(message, "tool_calls")
    if isinstance(raw_tool_calls, list):
        for index, tool_call in enumerate(raw_tool_calls):
            function_obj = _maybe_get(tool_call, "function")
            tool_name = _maybe_get(function_obj, "name") or _maybe_get(tool_call, "name")
            if not isinstance(tool_name, str) or not tool_name.strip():
                continue
            activity.append(
                _build_tool_activity_entry(
                    tool_index=index,
                    tool_name=tool_name.strip(),
                    arguments=_maybe_get(function_obj, "arguments"),
                    tool_call_id=_maybe_get(tool_call, "id"),
                    source="message.tool_calls",
                )
            )
        return activity

    content = _maybe_get(message, "content")
    if isinstance(content, list):
        for index, block in enumerate(content):
            if isinstance(block, dict):
                block_type = block.get("type")
                tool_name = block.get("name")
                arguments = block.get("input") or block.get("arguments")
                tool_call_id = block.get("id")
            else:
                block_type = getattr(block, "type", None)
                tool_name = getattr(block, "name", None)
                arguments = getattr(block, "input", None) or getattr(block, "arguments", None)
                tool_call_id = getattr(block, "id", None)
            if block_type not in {"tool_use", "function_call"}:
                continue
            if not isinstance(tool_name, str) or not tool_name.strip():
                continue
            activity.append(
                _build_tool_activity_entry(
                    tool_index=index,
                    tool_name=tool_name.strip(),
                    arguments=arguments,
                    tool_call_id=tool_call_id,
                    source="message.content",
                )
            )
        if activity:
            return activity

    provider_specific_fields = _extract_provider_specific_fields(message)
    provider_tool_calls = provider_specific_fields.get("tool_calls")
    if isinstance(provider_tool_calls, list):
        for index, tool_call in enumerate(provider_tool_calls):
            function_obj = _maybe_get(tool_call, "function")
            tool_name = _maybe_get(function_obj, "name") or _maybe_get(tool_call, "name")
            if not isinstance(tool_name, str) or not tool_name.strip():
                continue
            activity.append(
                _build_tool_activity_entry(
                    tool_index=index,
                    tool_name=tool_name.strip(),
                    arguments=_maybe_get(function_obj, "arguments"),
                    tool_call_id=_maybe_get(tool_call, "id"),
                    source="provider_specific_fields.tool_calls",
                )
            )

    return activity


_RESPONSE_OUTPUT_TOOL_ITEM_FALLBACK_NAMES: Dict[str, str] = {
    "apply_patch_call": "apply_patch",
    "custom_tool_call": "custom_tool_call",
    "computer_call": "computer_call",
    "local_shell_call": "local_shell_call",
    "mcp_call": "mcp_call",
    "web_search_call": "web_search_call",
    "file_search_call": "file_search_call",
    "image_generation_call": "image_generation_call",
}
_RESPONSE_OUTPUT_TOOL_ITEM_TYPES = set(_RESPONSE_OUTPUT_TOOL_ITEM_FALLBACK_NAMES) | {"function_call"}


def _extract_response_output_items(result: Any, standard_logging_object: Optional[Dict[str, Any]] = None) -> List[Any]:
    candidate_sources: List[Any] = [result]
    if isinstance(standard_logging_object, dict):
        candidate_sources.append(standard_logging_object.get("response"))

    for source in candidate_sources:
        if isinstance(source, list):
            return source

        output_items = _maybe_get(source, "output")
        if isinstance(output_items, list):
            return output_items

        output_items = _maybe_get_path(source, "_hidden_params", "responses_output")
        if isinstance(output_items, list):
            return output_items

        completed_payload = _extract_responses_completed_payload_from_passthrough_fallback_text(
            _maybe_get(source, "response")
        )
        if isinstance(completed_payload, dict):
            output_items = _maybe_get(_maybe_get(completed_payload, "response"), "output")
            if isinstance(output_items, list):
                return output_items

    return []


def _resolve_response_output_tool_name(item: Any) -> Optional[str]:
    tool_name = _maybe_get(item, "name")
    if isinstance(tool_name, str) and tool_name.strip():
        return tool_name.strip()

    item_type = _maybe_get(item, "type")
    if not isinstance(item_type, str) or not item_type.strip():
        return None

    fallback_name = _RESPONSE_OUTPUT_TOOL_ITEM_FALLBACK_NAMES.get(item_type)
    if isinstance(fallback_name, str) and fallback_name.strip():
        return fallback_name.strip()

    return None


def _extract_response_output_tool_activity(
    result: Any, standard_logging_object: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    output_items = _extract_response_output_items(result, standard_logging_object)
    if not output_items:
        return []

    activity: List[Dict[str, Any]] = []
    for index, item in enumerate(output_items):
        item_type = _maybe_get(item, "type")
        if item_type not in _RESPONSE_OUTPUT_TOOL_ITEM_TYPES:
            continue
        tool_name = _resolve_response_output_tool_name(item)
        if not isinstance(tool_name, str) or not tool_name.strip():
            continue
        arguments = _maybe_get(item, "arguments")
        if arguments is None and item_type in {"apply_patch_call", "custom_tool_call"}:
            arguments = _maybe_get(item, "patch") or _maybe_get(item, "input")
        activity.append(
            _build_tool_activity_entry(
                tool_index=index,
                tool_name=tool_name,
                arguments=arguments,
                tool_call_id=_maybe_get(item, "call_id") or _maybe_get(item, "id"),
                source="responses.output",
            )
        )

    return activity


def _summarize_tool_activity(tool_activity: List[Dict[str, Any]]) -> Dict[str, int]:
    read_paths: List[str] = []
    modified_paths: List[str] = []
    git_commit_count = 0
    git_push_count = 0
    for item in tool_activity:
        read_paths.extend(value for value in (item.get("file_paths_read") or []) if isinstance(value, str))
        modified_paths.extend(value for value in (item.get("file_paths_modified") or []) if isinstance(value, str))
        git_commit_count += _safe_int(item.get("git_commit_count")) or 0
        git_push_count += _safe_int(item.get("git_push_count")) or 0
    return {
        "file_read_count": len(_dedupe_strings(read_paths)),
        "file_modified_count": len(_dedupe_strings(modified_paths)),
        **_sensitive_config_change_flags_from_paths(modified_paths),
        "git_commit_count": git_commit_count,
        "git_push_count": git_push_count,
    }


def _extract_tool_call_info(message: Any) -> Tuple[int, List[str]]:
    raw_tool_calls = _maybe_get(message, "tool_calls")
    if isinstance(raw_tool_calls, list):
        tool_names: List[str] = []
        for tool_call in raw_tool_calls:
            function_obj = _maybe_get(tool_call, "function")
            tool_name = _maybe_get(function_obj, "name") or _maybe_get(tool_call, "name")
            if isinstance(tool_name, str) and tool_name:
                tool_names.append(tool_name)
        return len(raw_tool_calls), tool_names

    content = _maybe_get(message, "content")
    if isinstance(content, list):
        tool_names = []
        tool_call_count = 0
        for block in content:
            if isinstance(block, dict):
                block_type = block.get("type")
            else:
                block_type = getattr(block, "type", None)
            if block_type not in {"tool_use", "function_call"}:
                continue
            tool_call_count += 1
            tool_name = block.get("name") if isinstance(block, dict) else getattr(block, "name", None)
            if isinstance(tool_name, str) and tool_name:
                tool_names.append(tool_name)
        if tool_call_count:
            return tool_call_count, tool_names

    provider_specific_fields = _extract_provider_specific_fields(message)
    provider_tool_calls = provider_specific_fields.get("tool_calls")
    if isinstance(provider_tool_calls, list):
        tool_names = []
        for tool_call in provider_tool_calls:
            tool_name = _maybe_get(_maybe_get(tool_call, "function"), "name") or _maybe_get(tool_call, "name")
            if isinstance(tool_name, str) and tool_name:
                tool_names.append(tool_name)
        return len(provider_tool_calls), tool_names

    return 0, []


def _extract_response_output_tool_call_info(
    result: Any, standard_logging_object: Optional[Dict[str, Any]] = None
) -> Tuple[int, List[str]]:
    output_items = _extract_response_output_items(result, standard_logging_object)
    if not output_items:
        return 0, []

    tool_call_count = 0
    tool_names: List[str] = []
    for item in output_items:
        item_type = _maybe_get(item, "type")
        if item_type not in _RESPONSE_OUTPUT_TOOL_ITEM_TYPES:
            continue
        tool_call_count += 1
        tool_name = _resolve_response_output_tool_name(item)
        if isinstance(tool_name, str) and tool_name.strip():
            tool_names.append(tool_name)

    return tool_call_count, tool_names


def _extract_session_id(kwargs: Dict[str, Any]) -> Optional[str]:
    litellm_params = kwargs.get("litellm_params") or {}
    metadata = litellm_params.get("metadata") or {}
    standard_logging_object = kwargs.get("standard_logging_object") or {}
    standard_metadata = standard_logging_object.get("metadata") or {}

    proxy_header_candidates = (
        _maybe_get_path(litellm_params, "proxy_server_request", "headers", "x-claude-code-session-id"),
        _maybe_get_path(litellm_params, "proxy_server_request", "headers", "X-Claude-Code-Session-Id"),
        _maybe_get_path(litellm_params, "proxy_server_request", "headers", "x-grok-session-id"),
        _maybe_get_path(litellm_params, "proxy_server_request", "headers", "X-Grok-Session-Id"),
        _maybe_get_path(kwargs.get("passthrough_logging_payload"), "request_headers", "x-claude-code-session-id"),
        _maybe_get_path(kwargs.get("passthrough_logging_payload"), "request_headers", "X-Claude-Code-Session-Id"),
        _maybe_get_path(kwargs.get("passthrough_logging_payload"), "request_headers", "x-grok-session-id"),
        _maybe_get_path(kwargs.get("passthrough_logging_payload"), "request_headers", "X-Grok-Session-Id"),
    )

    for candidate in (
        litellm_params.get("litellm_session_id"),
        kwargs.get("litellm_session_id"),
        metadata.get("session_id"),
        standard_metadata.get("session_id"),
        standard_logging_object.get("session_id"),
        _coerce_nested_session_id(metadata.get("user_id")),
        _coerce_nested_session_id(metadata.get("user_api_key_end_user_id")),
        *proxy_header_candidates,
    ):
        if candidate is not None and str(candidate).strip():
            return str(candidate)

    route_family = _first_non_empty_string(
        metadata.get("passthrough_route_family"),
        standard_metadata.get("passthrough_route_family"),
    )
    call_type = kwargs.get("call_type") or standard_logging_object.get("call_type")
    should_fallback = (
        call_type == "pass_through_endpoint"
        or route_family is not None
        or metadata.get("aawm_passthrough_endpoint_type") is not None
        or metadata.get("aawm_stream_logging_endpoint_type") is not None
    )
    if not should_fallback:
        return None

    fallback_candidates = (
        (
            "metadata.google_adapter_session_id",
            metadata.get("google_adapter_session_id"),
            False,
        ),
        (
            "standard_metadata.google_adapter_session_id",
            standard_metadata.get("google_adapter_session_id"),
            False,
        ),
        (
            "litellm_params.litellm_trace_id",
            litellm_params.get("litellm_trace_id"),
            True,
        ),
        ("kwargs.litellm_trace_id", kwargs.get("litellm_trace_id"), True),
        ("metadata.trace_id", metadata.get("trace_id"), True),
        (
            "standard_logging_object.trace_id",
            standard_logging_object.get("trace_id"),
            True,
        ),
        ("kwargs.litellm_call_id", kwargs.get("litellm_call_id"), True),
    )
    for source, candidate, synthetic in fallback_candidates:
        if candidate is None or not str(candidate).strip():
            continue
        if isinstance(metadata, dict):
            metadata.setdefault("session_id_source", source)
            if synthetic:
                metadata.setdefault("synthetic_session_id", True)
                metadata.setdefault("synthetic_session_id_basis", source)
        return str(candidate).strip()
    return None


def _extract_trace_id(kwargs: Dict[str, Any]) -> Optional[str]:
    litellm_params = kwargs.get("litellm_params") or {}
    metadata = litellm_params.get("metadata") or {}
    standard_logging_object = kwargs.get("standard_logging_object") or {}

    for candidate in (
        litellm_params.get("litellm_trace_id"),
        kwargs.get("litellm_trace_id"),
        metadata.get("trace_id"),
        standard_logging_object.get("trace_id"),
    ):
        if candidate is not None and str(candidate).strip():
            return str(candidate)
    return None


def _infer_usage_breakout_provider_prefix(kwargs: Dict[str, Any], metadata: Dict[str, Any]) -> Optional[str]:
    route_family = metadata.get("passthrough_route_family")
    if isinstance(route_family, str) and route_family.strip():
        route_family_lower = route_family.lower()
        if route_family_lower == "codex_responses" or route_family_lower.startswith("codex_"):
            return "codex"
        if "gemini" in route_family_lower:
            return "gemini"

    provider = kwargs.get("custom_llm_provider")
    if isinstance(provider, str) and provider.strip():
        provider_lower = provider.lower()
        if provider_lower == "gemini":
            return "gemini"

    model = kwargs.get("model")
    if isinstance(model, str) and model.strip():
        model_lower = model.lower()
        if "gemini" in model_lower:
            return "gemini"
        if "codex" in model_lower:
            return "codex"

    return None


def _enrich_usage_breakout_metadata(kwargs: Dict[str, Any], result: Any) -> None:
    metadata = _ensure_mutable_metadata(kwargs)
    provider_prefix = _infer_usage_breakout_provider_prefix(kwargs, metadata)
    if provider_prefix is None:
        return

    usage_obj = _extract_usage_object(kwargs, result)
    if usage_obj is None:
        return

    reported_reasoning_tokens = _extract_reported_reasoning_tokens(usage_obj)
    reasoning_tokens_source: Optional[str] = None
    cache_read_input_tokens = _extract_cache_read_input_tokens(usage_obj)
    cache_creation_input_tokens = _extract_cache_creation_input_tokens(usage_obj)

    message = _extract_first_response_message(result)
    if reported_reasoning_tokens is not None:
        reasoning_tokens_source = "provider_reported"
    elif provider_prefix == "gemini":
        reported_reasoning_tokens = _fallback_gemini_reasoning_tokens_from_signatures(
            metadata,
            message,
        )
        if reported_reasoning_tokens is not None:
            reasoning_tokens_source = "provider_signature_present"

    tool_call_count, tool_names = _extract_tool_call_info(message)
    if tool_call_count == 0:
        tool_call_count, tool_names = _extract_response_output_tool_call_info(
            result,
            kwargs.get("standard_logging_object"),
        )

    metadata["usage_cache_read_input_tokens"] = cache_read_input_tokens
    metadata["usage_cache_creation_input_tokens"] = cache_creation_input_tokens
    metadata["usage_tool_call_count"] = tool_call_count
    metadata["usage_tool_names"] = tool_names
    metadata[f"{provider_prefix}_cache_read_input_tokens"] = cache_read_input_tokens
    metadata[f"{provider_prefix}_cache_creation_input_tokens"] = cache_creation_input_tokens
    metadata[f"{provider_prefix}_tool_call_count"] = tool_call_count
    metadata[f"{provider_prefix}_tool_names"] = tool_names

    if reported_reasoning_tokens is not None:
        metadata["usage_reasoning_tokens_reported"] = reported_reasoning_tokens
        metadata["usage_reasoning_tokens_source"] = reasoning_tokens_source or "provider_reported"
        metadata[f"{provider_prefix}_reasoning_tokens_reported"] = reported_reasoning_tokens

    tags_to_add = [f"{provider_prefix}-usage-breakout"]
    if reported_reasoning_tokens is not None:
        tags_to_add.extend(["reasoning-tokens-reported", f"{provider_prefix}-reasoning-tokens-reported"])
    if cache_read_input_tokens > 0:
        tags_to_add.extend(["cache-read-input-tokens", f"{provider_prefix}-cache-read-input-tokens"])
    if cache_creation_input_tokens > 0:
        tags_to_add.extend(
            [
                "cache-creation-input-tokens",
                f"{provider_prefix}-cache-creation-input-tokens",
            ]
        )
    if tool_call_count > 0:
        tags_to_add.extend(["tool-calls-present", f"{provider_prefix}-tool-calls-present"])
    _merge_tags(metadata, tags_to_add)

    _append_langfuse_span(
        metadata,
        name=f"{provider_prefix}.usage_breakout",
        span_metadata={
            "reported_reasoning_tokens": reported_reasoning_tokens,
            "reported_reasoning_tokens_source": reasoning_tokens_source,
            "cache_read_input_tokens": cache_read_input_tokens,
            "cache_creation_input_tokens": cache_creation_input_tokens,
            "tool_call_count": tool_call_count,
            "tool_names": tool_names,
        },
        start_time=datetime.now(timezone.utc),
        end_time=datetime.now(timezone.utc),
    )
def _split_spend_log_proxy_server_request(
    spend_log_row: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    proxy_server_request = _safe_json_load(spend_log_row.get("proxy_server_request"), {})
    if not isinstance(proxy_server_request, dict):
        return {}, {}

    request_headers = proxy_server_request.get("headers")
    if not isinstance(request_headers, dict):
        request_headers = {}

    for body_key in ("body", "request"):
        request_body = proxy_server_request.get(body_key)
        if isinstance(request_body, dict):
            return request_body, request_headers

    return proxy_server_request, request_headers


def _extract_trace_id_from_spend_log_row(spend_log_row: Dict[str, Any]) -> Tuple[Optional[str], str]:
    metadata = _safe_json_load(spend_log_row.get("metadata"), {})
    request_body, _request_headers = _split_spend_log_proxy_server_request(spend_log_row)

    for candidate in (
        metadata.get("trace_id") if isinstance(metadata, dict) else None,
        request_body.get("trace_id") if isinstance(request_body, dict) else None,
        spend_log_row.get("session_id"),
        spend_log_row.get("request_id"),
    ):
        if candidate is not None and str(candidate).strip():
            candidate_str = str(candidate).strip()
            if candidate is spend_log_row.get("session_id"):
                return candidate_str, "legacy_spend_log_session_field"
            if candidate is spend_log_row.get("request_id"):
                return candidate_str, "request_id_fallback"
            return candidate_str, "metadata_or_request_body"

    return None, "missing"


def _coerce_nested_session_id(value: Any) -> Optional[str]:
    if isinstance(value, dict):
        session_candidate = value.get("session_id") or value.get("sessionId")
        if session_candidate is not None and str(session_candidate).strip():
            return str(session_candidate).strip()
        return None

    if isinstance(value, str):
        parsed = _safe_json_load(value, None)
        if parsed is not None:
            return _coerce_nested_session_id(parsed)
        if value.strip():
            return value.strip()

    return None


def _extract_session_id_from_spend_log_row(
    spend_log_row: Dict[str, Any],
) -> Tuple[Optional[str], str]:
    metadata = _safe_json_load(spend_log_row.get("metadata"), {})
    request_body, _request_headers = _split_spend_log_proxy_server_request(spend_log_row)
    response_body = _safe_json_load(spend_log_row.get("response"), {})

    if isinstance(request_body, dict):
        metadata_payload = request_body.get("metadata")
        if isinstance(metadata_payload, dict):
            session_candidate = metadata_payload.get("session_id")
            if session_candidate is not None and str(session_candidate).strip():
                return str(session_candidate).strip(), "request_body.metadata.session_id"

            user_id_payload = metadata_payload.get("user_id")
            nested_session_id = _coerce_nested_session_id(user_id_payload)
            if nested_session_id:
                return nested_session_id, "request_body.metadata.user_id.session_id"

        top_level_session_id = request_body.get("session_id")
        if top_level_session_id is not None and str(top_level_session_id).strip():
            return str(top_level_session_id).strip(), "request_body.session_id"

        request_payload = request_body.get("request")
        if isinstance(request_payload, dict):
            request_session_id = request_payload.get("session_id")
            if request_session_id is not None and str(request_session_id).strip():
                return str(request_session_id).strip(), "request_body.request.session_id"

    if isinstance(metadata, dict):
        for key in ("session_id", "sessionId"):
            session_candidate = metadata.get(key)
            if session_candidate is not None and str(session_candidate).strip():
                return str(session_candidate).strip(), f"metadata.{key}"

    if isinstance(response_body, dict):
        for key in ("session_id", "sessionId"):
            session_candidate = response_body.get(key)
            if session_candidate is not None and str(session_candidate).strip():
                return str(session_candidate).strip(), f"response.{key}"

    legacy_session_field = spend_log_row.get("session_id")
    if legacy_session_field is not None and str(legacy_session_field).strip():
        return str(legacy_session_field).strip(), "legacy_spend_log_session_field"

    return None, "missing"


def _coerce_spend_log_request_tags(value: Any) -> List[str]:
    parsed = _safe_json_load(value, value)
    if not isinstance(parsed, list):
        return []
    return [str(tag) for tag in parsed if isinstance(tag, str) and tag.strip()]


def _synthesize_result_from_spend_log_row(
    spend_log_row: Dict[str, Any],
    metadata: Dict[str, Any],
) -> Dict[str, Any]:
    result = _safe_json_load(spend_log_row.get("response"), {})
    if not isinstance(result, dict):
        result = {"response": result}

    usage_object = metadata.get("usage_object")
    if not isinstance(usage_object, dict):
        usage_object = {}

    if not isinstance(result.get("usage"), dict):
        reconstructed_usage = dict(usage_object)
        reconstructed_usage.setdefault("prompt_tokens", _safe_int(spend_log_row.get("prompt_tokens")) or 0)
        reconstructed_usage.setdefault("completion_tokens", _safe_int(spend_log_row.get("completion_tokens")) or 0)
        reconstructed_usage.setdefault("total_tokens", _safe_int(spend_log_row.get("total_tokens")) or 0)
        result["usage"] = reconstructed_usage

    return result


def _build_backfill_kwargs_from_spend_log_row(
    spend_log_row: Dict[str, Any],
) -> Optional[Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]]:
    request_id = spend_log_row.get("request_id")
    model = spend_log_row.get("model")
    if request_id is None or not str(request_id).strip():
        return None
    if model is None or not str(model).strip():
        return None

    metadata = _safe_json_load(spend_log_row.get("metadata"), {})
    if not isinstance(metadata, dict):
        metadata = {}
    request_body, request_headers = _split_spend_log_proxy_server_request(spend_log_row)
    request_tags = _coerce_spend_log_request_tags(spend_log_row.get("request_tags"))

    session_id, session_id_source = _extract_session_id_from_spend_log_row(spend_log_row)
    trace_id, trace_id_source = _extract_trace_id_from_spend_log_row(spend_log_row)

    litellm_metadata: Dict[str, Any] = dict(metadata)
    if session_id:
        litellm_metadata["session_id"] = session_id
    if trace_id:
        litellm_metadata["trace_id"] = trace_id
    if spend_log_row.get("model_group"):
        litellm_metadata["model_group"] = spend_log_row.get("model_group")

    standard_logging_metadata = dict(litellm_metadata)
    if isinstance(metadata.get("usage_object"), dict):
        standard_logging_metadata["usage_object"] = metadata.get("usage_object")

    standard_logging_object: Dict[str, Any] = {
        "metadata": standard_logging_metadata,
        "request_headers": request_headers,
        "request_tags": list(request_tags),
        "trace_id": trace_id,
        "model": str(model),
        "model_group": spend_log_row.get("model_group"),
        "response_cost": _safe_float(spend_log_row.get("spend")),
        "prompt_tokens": _safe_int(spend_log_row.get("prompt_tokens")) or 0,
        "completion_tokens": _safe_int(spend_log_row.get("completion_tokens")) or 0,
        "total_tokens": _safe_int(spend_log_row.get("total_tokens")) or 0,
    }

    kwargs: Dict[str, Any] = {
        "model": str(model),
        "custom_llm_provider": _normalize_session_history_provider(
            spend_log_row.get("custom_llm_provider"),
            str(model),
            metadata,
        ),
        "call_type": spend_log_row.get("call_type"),
        "litellm_call_id": str(request_id),
        "litellm_trace_id": trace_id,
        "litellm_session_id": session_id,
        "litellm_params": {
            "metadata": litellm_metadata,
            "litellm_trace_id": trace_id,
            "litellm_session_id": session_id,
            "proxy_server_request": {
                "body": request_body,
                "headers": request_headers,
            },
        },
        "standard_logging_object": standard_logging_object,
        "passthrough_logging_payload": {
            "request_body": request_body,
            "request_headers": request_headers,
        },
        "response_cost": _safe_float(spend_log_row.get("spend")),
    }

    messages = _safe_json_load(spend_log_row.get("messages"), None)
    if isinstance(messages, list):
        kwargs["messages"] = messages

    system = request_body.get("system")
    if system is not None:
        kwargs["system"] = system

    result = _synthesize_result_from_spend_log_row(spend_log_row, metadata)

    provenance = {
        "session_id_source": session_id_source,
        "trace_id_source": trace_id_source,
        "source_request_id": str(request_id),
        "source_spend_log_session_field": (
            str(spend_log_row.get("session_id")).strip()
            if spend_log_row.get("session_id") is not None and str(spend_log_row.get("session_id")).strip()
            else None
        ),
    }

    return kwargs, result, provenance


# _derive_session_history_reasoning_fields moved to litellm.integrations.aawm_session_history.record
# _derive_session_history_tool_fields moved to litellm.integrations.aawm_session_history.record
# _derive_session_history_provider_cache_fields moved to litellm.integrations.aawm_session_history.record
# _build_session_history_record_from_spend_log_row moved to litellm.integrations.aawm_session_history.record
def _derive_langfuse_trace_tags_from_spend_log_row(
    spend_log_row: Dict[str, Any],
) -> Tuple[Optional[str], List[str]]:
    prepared = _build_backfill_kwargs_from_spend_log_row(spend_log_row)
    if prepared is None:
        return None, []

    kwargs, result, _provenance = prepared
    kwargs, result = _enrich_trace_name_and_provider_metadata(kwargs, result)
    standard_logging_object = kwargs.get("standard_logging_object") or {}
    request_tags = standard_logging_object.get("request_tags") or []
    if not isinstance(request_tags, list):
        request_tags = []
    trace_id = kwargs.get("litellm_trace_id")
    if trace_id is not None and str(trace_id).strip():
        trace_id = str(trace_id).strip()
    else:
        trace_id = None
    return trace_id, [tag for tag in request_tags if isinstance(tag, str) and tag.strip()]


def _serialize_searchable_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, sort_keys=True)
    except (TypeError, ValueError):
        return str(value)


def _extract_agent_context_from_langfuse_trace_observation(
    trace: Dict[str, Any],
    observation: Dict[str, Any],
) -> Tuple[Optional[str], Optional[str]]:
    explicit_tenant_id, _tenant_source = _extract_tenant_identity_from_langfuse_trace_observation(
        trace,
        observation,
    )
    for candidate in (
        observation.get("input"),
        trace.get("input"),
        observation.get("output"),
        trace.get("output"),
    ):
        agent_name, tenant_id = _extract_agent_context_from_text(_serialize_searchable_text(candidate))
        if agent_name:
            return agent_name, explicit_tenant_id or tenant_id

    trace_name = trace.get("name")
    if isinstance(trace_name, str) and trace_name.startswith("claude-code."):
        return trace_name.split(".", 1)[1], explicit_tenant_id

    return None, explicit_tenant_id


def _extract_langfuse_session_id(
    trace: Dict[str, Any],
    observation_metadata: Dict[str, Any],
) -> Tuple[Optional[str], str]:
    for candidate in (
        trace.get("sessionId"),
        trace.get("session_id"),
        observation_metadata.get("session_id"),
        observation_metadata.get("google_adapter_session_id"),
        _coerce_nested_session_id(observation_metadata.get("user_id")),
        _coerce_nested_session_id(observation_metadata.get("user_api_key_end_user_id")),
    ):
        if candidate is not None and str(candidate).strip():
            if candidate == trace.get("sessionId"):
                return str(candidate).strip(), "trace.sessionId"
            if candidate == trace.get("session_id"):
                return str(candidate).strip(), "trace.session_id"
            if candidate == observation_metadata.get("session_id"):
                return str(candidate).strip(), "observation.metadata.session_id"
            if candidate == observation_metadata.get("google_adapter_session_id"):
                return (
                    str(candidate).strip(),
                    "observation.metadata.google_adapter_session_id",
                )
            if candidate == _coerce_nested_session_id(observation_metadata.get("user_id")):
                return str(candidate).strip(), "observation.metadata.user_id.session_id"
            return (
                str(candidate).strip(),
                "observation.metadata.user_api_key_end_user_id.session_id",
            )

    route_family = observation_metadata.get("passthrough_route_family")
    is_passthrough_trace = (
        isinstance(route_family, str)
        and bool(route_family.strip())
        or observation_metadata.get("aawm_passthrough_endpoint_type") is not None
        or observation_metadata.get("aawm_stream_logging_endpoint_type") is not None
    )
    if is_passthrough_trace:
        for source, candidate in (
            ("trace.id", trace.get("id")),
            ("observation.traceId", observation_metadata.get("traceId")),
        ):
            if candidate is None or not str(candidate).strip():
                continue
            observation_metadata.setdefault("session_id_source", f"{source}.synthetic")
            observation_metadata.setdefault("synthetic_session_id", True)
            observation_metadata.setdefault("synthetic_session_id_basis", source)
            return str(candidate).strip(), f"{source}.synthetic"

    return None, "missing"


def _build_usage_object_from_langfuse_observation(observation: Dict[str, Any]) -> Dict[str, Any]:
    metadata = observation.get("metadata")
    usage = observation.get("usage")
    usage_details = observation.get("usageDetails")

    usage_object: Dict[str, Any] = {}
    if isinstance(metadata, dict):
        metadata_usage_object = _build_usage_object_from_metadata(metadata)
        if isinstance(metadata_usage_object, dict):
            usage_object.update(metadata_usage_object)
    output_usage_object = _build_usage_object_from_langfuse_output(observation.get("output"))
    if isinstance(output_usage_object, dict):
        usage_object.update(output_usage_object)
    if isinstance(usage, dict):
        usage_object.update(usage)
    if isinstance(usage_details, dict):
        usage_object.update(usage_details)

    prompt_tokens = _safe_int(
        _first_non_none(
            observation.get("promptTokens"),
            observation.get("inputTokens"),
            usage_object.get("prompt_tokens"),
            usage_object.get("input_tokens"),
            usage_object.get("input"),
        )
    )
    completion_tokens = _safe_int(
        _first_non_none(
            observation.get("completionTokens"),
            observation.get("outputTokens"),
            usage_object.get("completion_tokens"),
            usage_object.get("output_tokens"),
            usage_object.get("output"),
        )
    )
    total_tokens = _safe_int(
        _first_non_none(
            observation.get("totalTokens"),
            usage_object.get("total_tokens"),
            usage_object.get("total"),
        )
    )

    if prompt_tokens is not None:
        usage_object["prompt_tokens"] = prompt_tokens
    if completion_tokens is not None:
        usage_object["completion_tokens"] = completion_tokens
        usage_object.setdefault("output_tokens", completion_tokens)
    if total_tokens is not None:
        usage_object["total_tokens"] = total_tokens

    prompt_tokens_details = _extract_prompt_tokens_details(usage_object)
    if isinstance(prompt_tokens_details, dict):
        usage_object.setdefault("prompt_tokens_details", prompt_tokens_details)

    completion_tokens_details = _extract_completion_tokens_details(usage_object)
    if isinstance(completion_tokens_details, dict):
        usage_object.setdefault("completion_tokens_details", completion_tokens_details)

    cache_read_tokens = _safe_int(usage_object.get("cache_read_input_tokens"))
    if cache_read_tokens is None:
        cache_read_tokens = _safe_int(usage_object.get("cachedContentTokenCount"))
    cache_creation_tokens = _safe_int(usage_object.get("cache_creation_input_tokens"))
    if cache_read_tokens is not None:
        usage_object["cache_read_input_tokens"] = cache_read_tokens
    if cache_creation_tokens is not None:
        usage_object["cache_creation_input_tokens"] = cache_creation_tokens
    if usage_object.get("reasoning_tokens") is None:
        thoughts_token_count = _safe_int(usage_object.get("thoughtsTokenCount"))
        if thoughts_token_count is not None:
            usage_object["reasoning_tokens"] = thoughts_token_count

    return usage_object


def _extract_first_langfuse_response_message(output_payload: Any) -> Any:
    if isinstance(output_payload, dict):
        if isinstance(output_payload.get("choices"), list):
            return _extract_first_response_message(output_payload)
        if isinstance(output_payload.get("message"), dict):
            return output_payload["message"]
        if any(key in output_payload for key in ("content", "tool_calls", "reasoning_content", "thinking_blocks")):
            return output_payload
    return None


def _infer_provider_from_langfuse_observation(
    observation: Dict[str, Any],
    metadata: Dict[str, Any],
) -> Optional[str]:
    adapter_target_provider = _session_history_adapter_target_provider(metadata)
    if adapter_target_provider is not None:
        return adapter_target_provider

    auto_agent_provider = _session_history_auto_agent_selected_provider(metadata)
    if auto_agent_provider is not None:
        return auto_agent_provider

    route_provider = _session_history_provider_from_route_family(metadata.get("passthrough_route_family"))
    if route_provider is not None:
        return route_provider

    api_base = (
        metadata.get("api_base") or _maybe_get(metadata.get("hidden_params"), "api_base") or observation.get("apiBase")
    )
    api_base_provider = _session_history_provider_from_api_base(
        api_base,
        call_type=metadata.get("user_api_key_request_route") or observation.get("name"),
    )
    if api_base_provider is not None:
        return api_base_provider

    model = (
        _session_history_adapter_model(metadata)
        or _session_history_metadata_model(metadata)
        or observation.get("model")
    )
    model_provider = _session_history_provider_from_model(model)
    if model_provider is not None:
        return model_provider

    request_route = metadata.get("user_api_key_request_route")
    if isinstance(request_route, str) and request_route.strip():
        route_lower = request_route.lower()
        if "gemini" in route_lower or "google" in route_lower:
            return "gemini"
        if route_lower.startswith("/v1/"):
            return "openai"
        if route_lower.startswith("/anthropic/"):
            return "anthropic"

    return _normalize_session_history_provider(
        metadata.get("custom_llm_provider"),
        str(observation.get("model") or ""),
        metadata,
    )


def _derive_request_tags_from_langfuse_metadata(metadata: Dict[str, Any]) -> List[str]:
    request_tags = metadata.get("tags")
    normalized_tags = (
        [str(tag) for tag in request_tags if isinstance(tag, str) and tag.strip()]
        if isinstance(request_tags, list)
        else []
    )

    route_family = metadata.get("passthrough_route_family")
    if isinstance(route_family, str) and route_family.strip():
        normalized_tags.append(f"route:{route_family.strip()}")

    billing_header_fields = metadata.get("anthropic_billing_header_fields")
    if isinstance(billing_header_fields, dict) and billing_header_fields:
        normalized_tags.append("anthropic-billing-header")
        for key, value in list(billing_header_fields.items()):
            if isinstance(key, str) and key.strip():
                normalized_tags.append(f"anthropic-billing-header-key:{key}")
                if value is not None and str(value).strip():
                    normalized_tags.append(f"anthropic-billing-header:{key}={str(value).strip()}")

    thinking_type = metadata.get("claude_thinking_type")
    if isinstance(thinking_type, str) and thinking_type.strip():
        normalized_tags.append(f"claude-thinking-type:{thinking_type}")
        normalized_tags.append(f"thinking-type:{thinking_type}")

    effort = metadata.get("claude_effort")
    if isinstance(effort, str) and effort.strip():
        normalized_tags.append(f"claude-effort:{effort}")
        normalized_tags.append(f"effort:{effort}")

    if metadata.get("thinking_signature_present") is True:
        normalized_tags.append("thinking-signature-present")
    if metadata.get("claude_thinking_signature_present") is True:
        normalized_tags.append("claude-thinking-signature")
    if metadata.get("gemini_thought_signature_present") is True:
        normalized_tags.append("gemini-thought-signature")
    if metadata.get("thinking_signature_decoded") is True:
        normalized_tags.append("thinking-signature-decoded")
    if metadata.get("claude_thinking_signature_decoded") is True:
        normalized_tags.append("claude-thinking-decoded")
    if metadata.get("reasoning_content_present") is True:
        normalized_tags.append("reasoning-present")
    elif metadata.get("reasoning_content_present") is False:
        normalized_tags.append("reasoning-empty")
    if metadata.get("thinking_blocks_present") is True:
        normalized_tags.append("thinking-blocks-present")
    elif metadata.get("thinking_blocks_present") is False:
        normalized_tags.append("thinking-blocks-empty")

    return sorted({tag for tag in normalized_tags if isinstance(tag, str) and tag.strip()})


# _build_session_history_record_from_langfuse_trace_observation moved to litellm.integrations.aawm_session_history.record
def _derive_langfuse_trace_tags_from_langfuse_trace(
    trace: Dict[str, Any],
) -> Tuple[Optional[str], List[str]]:
    trace_id = trace.get("id")
    normalized_trace_id = str(trace_id).strip() if trace_id is not None and str(trace_id).strip() else None

    derived_tags: List[str] = []
    existing_trace_tags = trace.get("tags")
    if isinstance(existing_trace_tags, list):
        derived_tags.extend(str(tag) for tag in existing_trace_tags if isinstance(tag, str) and tag.strip())

    observations = trace.get("observations")
    if isinstance(observations, list):
        for observation in observations:
            if not isinstance(observation, dict) or observation.get("type") != "GENERATION":
                continue
            metadata = observation.get("metadata")
            if not isinstance(metadata, dict):
                continue
            derived_tags.extend(_derive_request_tags_from_langfuse_metadata(metadata))

    return normalized_trace_id, sorted({tag for tag in derived_tags if isinstance(tag, str) and tag.strip()})


def _iter_litellm_metadata_sources(
    kwargs: Dict[str, Any],
    metadata: Dict[str, Any],
) -> Iterator[Dict[str, Any]]:
    litellm_params = kwargs.get("litellm_params")
    if not isinstance(litellm_params, dict):
        litellm_params = {}
    standard_logging_object = kwargs.get("standard_logging_object")
    if not isinstance(standard_logging_object, dict):
        standard_logging_object = {}
    passthrough_payload = kwargs.get("passthrough_logging_payload")
    if not isinstance(passthrough_payload, dict):
        passthrough_payload = {}
    proxy_request = _coerce_mapping(litellm_params.get("proxy_server_request"))
    proxy_body = _coerce_mapping(proxy_request.get("body"))
    passthrough_body = _coerce_mapping(passthrough_payload.get("request_body"))

    for candidate in (
        metadata,
        litellm_params.get("metadata"),
        litellm_params.get("litellm_metadata"),
        standard_logging_object.get("metadata"),
        kwargs.get("metadata"),
        proxy_body.get("metadata"),
        proxy_body.get("litellm_metadata"),
        passthrough_body.get("metadata"),
        passthrough_body.get("litellm_metadata"),
    ):
        source = _coerce_mapping(candidate)
        if source:
            yield source


def _bound_worker_context_exhaustion_string(
    key: str,
    value: Any,
) -> Optional[str]:
    cleaned = _clean_non_empty_string(value)
    if cleaned is None:
        return None
    max_len = _WORKER_CONTEXT_EXHAUSTION_STRING_MAX_LEN.get(key, 512)
    if len(cleaned) > max_len:
        cleaned = cleaned[:max_len]
    return cleaned


def _normalize_worker_context_exhaustion_bool(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes"}:
            return True
        if normalized in {"false", "0", "no"}:
            return False
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    return None


def _sanitize_worker_context_exhaustion_metadata(metadata: Dict[str, Any]) -> None:
    """Bound orchestrator worker exhaustion fields; never infer success from LLM output."""
    for key in _WORKER_CONTEXT_EXHAUSTION_METADATA_KEYS:
        if key not in metadata:
            continue
        raw_value = metadata.get(key)
        if key in _WORKER_CONTEXT_EXHAUSTION_BOOL_KEYS:
            normalized_bool = _normalize_worker_context_exhaustion_bool(raw_value)
            if normalized_bool is None:
                metadata.pop(key, None)
            else:
                metadata[key] = normalized_bool
            continue

        if isinstance(raw_value, list):
            bounded_items = []
            for item in raw_value[:50]:
                item_text = _bound_worker_context_exhaustion_string(key, item)
                if item_text is not None:
                    bounded_items.append(item_text)
            if bounded_items:
                metadata[key] = bounded_items
            else:
                metadata.pop(key, None)
            continue

        bounded = _bound_worker_context_exhaustion_string(key, raw_value)
        if bounded is None:
            metadata.pop(key, None)
        else:
            metadata[key] = bounded

    if metadata.get("worker_context_exhaustion_failure_class"):
        metadata["worker_context_exhaustion_success"] = False
        metadata["worker_context_exhaustion_completed"] = False


def _is_anthropic_session_history_context(
    *,
    provider: Optional[str],
    resolved_model: str,
    metadata: Dict[str, Any],
) -> bool:
    provider_lower = str(provider or "").strip().lower()
    if provider_lower in {"anthropic", "azure_ai", "bedrock"}:
        return True
    route_family = (
        str(
            metadata.get("passthrough_route_family")
            or metadata.get("route_family")
            or metadata.get("openai_passthrough_route_family")
            or ""
        )
        .strip()
        .lower()
    )
    if "anthropic" in route_family:
        return True
    model_lower = str(resolved_model or "").strip().lower()
    if model_lower.startswith("claude") or "claude" in model_lower:
        return True
    for key in (
        "anthropic_adapter_model",
        "anthropic_adapter_original_model",
        "anthropic_auto_agent_selected_model",
    ):
        candidate = str(metadata.get(key) or "").strip().lower()
        if candidate.startswith("claude") or "anthropic" in candidate:
            return True
    return False


def _iter_anthropic_beta_header_candidates(
    headers: Optional[Dict[str, Any]],
    metadata: Dict[str, Any],
) -> List[str]:
    candidates: List[str] = []
    header_names = (
        "anthropic-beta",
        "x-pass-anthropic-beta",
        "llm_provider-anthropic-beta",
    )
    if headers:
        for header_name in header_names:
            value = _get_header_value(headers, header_name)
            if value:
                candidates.append(value)
        for key, value in headers.items():
            key_lower = str(key).lower()
            if key_lower in {"anthropic-beta", "x-pass-anthropic-beta"}:
                cleaned = _clean_non_empty_string(value)
                if cleaned:
                    candidates.append(cleaned)
            elif key_lower.startswith("llm_provider-") and "anthropic-beta" in key_lower:
                cleaned = _clean_non_empty_string(value)
                if cleaned:
                    candidates.append(cleaned)

    for meta_key in (
        "anthropic-beta",
        "anthropic_beta",
        "llm_provider-anthropic-beta",
        "x-pass-anthropic-beta",
    ):
        cleaned = _clean_non_empty_string(metadata.get(meta_key))
        if cleaned:
            candidates.append(cleaned)

    for nested_key in ("hidden_params", "_hidden_params", "additional_headers"):
        nested = metadata.get(nested_key)
        if not isinstance(nested, dict):
            continue
        for key, value in nested.items():
            key_lower = str(key).lower()
            if key_lower in {"anthropic-beta", "x-pass-anthropic-beta"}:
                cleaned = _clean_non_empty_string(value)
                if cleaned:
                    candidates.append(cleaned)
            elif key_lower.startswith("llm_provider-") and "anthropic-beta" in key_lower:
                cleaned = _clean_non_empty_string(value)
                if cleaned:
                    candidates.append(cleaned)
    return candidates


def _split_anthropic_beta_values(raw_value: str) -> List[str]:
    return [token.strip() for token in str(raw_value).replace(";", ",").split(",") if token.strip()]


def _extract_context_1m_beta_values(
    headers: Optional[Dict[str, Any]],
    metadata: Dict[str, Any],
) -> List[str]:
    matched: List[str] = []
    seen: Set[str] = set()
    for raw in _iter_anthropic_beta_header_candidates(headers, metadata):
        for beta_value in _split_anthropic_beta_values(raw):
            beta_lower = beta_value.lower()
            if beta_lower == _ANTHROPIC_CONTEXT_1M_BETA_HEADER.lower() or beta_lower.startswith(
                _ANTHROPIC_CONTEXT_1M_BETA_PREFIX
            ):
                if beta_value not in seen:
                    seen.add(beta_value)
                    matched.append(beta_value)
    return matched


def _model_strings_indicate_context_1m_suffix(*model_values: Any) -> bool:
    suffix_lower = _ANTHROPIC_CONTEXT_1M_MODEL_SUFFIX.lower()
    for value in model_values:
        cleaned = _clean_non_empty_string(value)
        if cleaned and cleaned.lower().endswith(suffix_lower):
            return True
    return False


def _select_safe_anthropic_context_window_beta(beta_values: List[str]) -> Optional[str]:
    if not beta_values:
        return None
    for beta_value in beta_values:
        if beta_value.lower() == _ANTHROPIC_CONTEXT_1M_BETA_HEADER.lower():
            return beta_value
    for beta_value in beta_values:
        if beta_value.lower().startswith(_ANTHROPIC_CONTEXT_1M_BETA_PREFIX):
            return beta_value
    return beta_values[0]


def _apply_anthropic_context_window_metadata_fields(
    metadata: Dict[str, Any],
    *,
    mode: str,
    requested_tokens: Optional[int],
    source: str,
    beta: Optional[str] = None,
    classification: Optional[str] = None,
) -> None:
    metadata["anthropic_context_window_mode"] = mode
    metadata["anthropic_context_window_requested_tokens"] = requested_tokens
    metadata["anthropic_context_window_source"] = source
    if beta is not None:
        metadata["anthropic_context_window_beta"] = beta
    else:
        metadata.pop("anthropic_context_window_beta", None)
    if classification is not None:
        metadata["anthropic_context_window_classification"] = classification
    else:
        metadata.pop("anthropic_context_window_classification", None)


def _classify_anthropic_context_window_from_retained_evidence(
    metadata: Dict[str, Any],
    *,
    resolved_model: str,
    inbound_model_alias: Optional[str] = None,
    headers: Optional[Dict[str, Any]] = None,
    allow_implicit_default: bool = False,
) -> Optional[Dict[str, Any]]:
    beta_values = _extract_context_1m_beta_values(headers, metadata)
    if beta_values:
        return {
            "mode": "extended_1m",
            "requested_tokens": _ANTHROPIC_CONTEXT_WINDOW_1M_TOKEN_COUNT,
            "source": "anthropic_beta_header",
            "beta": _select_safe_anthropic_context_window_beta(beta_values),
            "classification": "classified",
        }

    if _model_strings_indicate_context_1m_suffix(
        inbound_model_alias,
        metadata.get("inbound_model_alias"),
        metadata.get("requested_model_alias"),
        metadata.get("model_alias_label"),
        metadata.get("anthropic_native_passthrough_model_alias"),
        metadata.get("source_model"),
        metadata.get("model"),
        resolved_model,
    ):
        return {
            "mode": "extended_1m",
            "requested_tokens": _ANTHROPIC_CONTEXT_WINDOW_1M_TOKEN_COUNT,
            "source": "model_suffix_1m",
            "beta": None,
            "classification": "classified",
        }

    if not _is_anthropic_session_history_context(
        provider=str(metadata.get("custom_llm_provider") or metadata.get("provider") or ""),
        resolved_model=resolved_model,
        metadata=metadata,
    ):
        return None

    if allow_implicit_default:
        return {
            "mode": "default_200k",
            "requested_tokens": _ANTHROPIC_CONTEXT_WINDOW_DEFAULT_TOKEN_COUNT,
            "source": "no_extended_context_evidence",
            "beta": None,
            "classification": "classified",
        }

    return {
        "mode": "unknown",
        "requested_tokens": None,
        "source": "unavailable",
        "beta": None,
        "classification": "unavailable",
    }


def _enrich_anthropic_context_window_metadata(
    kwargs: Dict[str, Any],
    metadata: Dict[str, Any],
    *,
    resolved_model: Optional[str] = None,
    inbound_model_alias: Optional[str] = None,
    provider: Optional[str] = None,
    allow_implicit_default: bool = True,
) -> None:
    model_value = _clean_non_empty_string(resolved_model or metadata.get("model") or kwargs.get("model")) or "unknown"
    provider_value = _clean_non_empty_string(
        provider or kwargs.get("custom_llm_provider") or metadata.get("custom_llm_provider")
    )
    if provider_value:
        metadata.setdefault("custom_llm_provider", provider_value)

    headers = _extract_request_headers_from_kwargs(kwargs)
    classification = _classify_anthropic_context_window_from_retained_evidence(
        metadata,
        resolved_model=model_value,
        inbound_model_alias=inbound_model_alias,
        headers=headers,
        allow_implicit_default=allow_implicit_default,
    )
    if classification is None:
        for key in _ANTHROPIC_CONTEXT_WINDOW_METADATA_KEYS:
            metadata.pop(key, None)
        return

    _apply_anthropic_context_window_metadata_fields(
        metadata,
        mode=classification["mode"],
        requested_tokens=classification["requested_tokens"],
        source=classification["source"],
        beta=classification.get("beta"),
        classification=classification.get("classification"),
    )


def _enrich_backfill_anthropic_context_window_metadata(
    record: Dict[str, Any],
) -> None:
    metadata = record.get("metadata")
    if not isinstance(metadata, dict):
        return
    provider = _clean_non_empty_string(record.get("provider"))
    if provider:
        metadata.setdefault("custom_llm_provider", provider)
    classification = _classify_anthropic_context_window_from_retained_evidence(
        metadata,
        resolved_model=str(record.get("model") or metadata.get("model") or "unknown"),
        inbound_model_alias=record.get("inbound_model_alias"),
        headers=None,
        allow_implicit_default=False,
    )
    if classification is None:
        return
    _apply_anthropic_context_window_metadata_fields(
        metadata,
        mode=classification["mode"],
        requested_tokens=classification["requested_tokens"],
        source=classification["source"],
        beta=classification.get("beta"),
        classification=classification.get("classification"),
    )
    record["metadata"] = metadata


def _promote_worker_context_exhaustion_metadata(
    kwargs: Dict[str, Any],
    metadata: Dict[str, Any],
) -> None:
    """Copy allowlisted worker exhaustion keys from upstream litellm_metadata without overwriting."""
    for source in _iter_litellm_metadata_sources(kwargs, metadata):
        for key in _WORKER_CONTEXT_EXHAUSTION_METADATA_KEYS:
            if key in metadata:
                continue
            if key not in source:
                continue
            value = source.get(key)
            if value is None:
                continue
            metadata[key] = value
    _sanitize_worker_context_exhaustion_metadata(metadata)


# _build_session_history_metadata moved to litellm.integrations.aawm_session_history.record
def _sanitize_session_history_api_base(value: Any) -> Optional[str]:
    cleaned = _clean_non_empty_string(value)
    if not cleaned:
        return None

    try:
        parsed = urlsplit(cleaned)
    except ValueError:
        return None

    if not parsed.scheme or not parsed.netloc:
        return cleaned.split("?", 1)[0].split("#", 1)[0].rstrip("/") or None

    hostname = parsed.hostname
    if not hostname:
        return None

    netloc = hostname
    if parsed.port is not None:
        netloc = f"{netloc}:{parsed.port}"

    return urlunsplit((parsed.scheme, netloc, parsed.path.rstrip("/"), "", "")) or None


def _is_local_session_history_api_base(value: Any) -> bool:
    sanitized = _sanitize_session_history_api_base(value)
    if not sanitized:
        return False

    try:
        hostname = urlsplit(sanitized).hostname
    except ValueError:
        return False
    if not hostname:
        return False

    hostname_lower = hostname.lower()
    if hostname_lower in {"localhost", "host.docker.internal"}:
        return True

    try:
        parsed_ip = ipaddress.ip_address(hostname_lower)
    except ValueError:
        return False

    return parsed_ip.is_loopback or parsed_ip.is_private or parsed_ip.is_link_local or parsed_ip.is_unspecified


def _extract_session_history_api_base(
    kwargs: Dict[str, Any],
    standard_logging_object: Dict[str, Any],
    metadata: Dict[str, Any],
) -> Optional[str]:
    litellm_params = kwargs.get("litellm_params")
    if not isinstance(litellm_params, dict):
        litellm_params = {}

    for candidate in (
        standard_logging_object.get("api_base"),
        _maybe_get_path(standard_logging_object, "hidden_params", "api_base"),
        litellm_params.get("api_base"),
        metadata.get("api_base"),
        _maybe_get(metadata.get("hidden_params"), "api_base"),
        _maybe_get_path(kwargs.get("passthrough_logging_payload"), "url"),
        _maybe_get_path(kwargs.get("standard_pass_through_logging_payload"), "url"),
    ):
        sanitized = _sanitize_session_history_api_base(candidate)
        if sanitized:
            return sanitized
    return None


def _get_session_history_model_group(
    metadata: Dict[str, Any],
    standard_logging_object: Dict[str, Any],
) -> Optional[str]:
    return _first_non_empty_string(
        metadata.get("model_group"),
        standard_logging_object.get("model_group"),
    )


def _resolve_inbound_model_alias(
    *,
    kwargs: Dict[str, Any],
    standard_logging_object: Dict[str, Any],
    metadata: Dict[str, Any],
    resolved_model: str,
) -> str:
    return (
        _first_non_empty_string(
            metadata.get("model_alias_label"),
            metadata.get("requested_model_alias"),
            metadata.get("codex_auto_agent_alias"),
            metadata.get("anthropic_auto_agent_alias"),
            metadata.get("aawm_auto_agent_alias"),
            _maybe_get_path(
                kwargs.get("litellm_params"),
                "proxy_server_request",
                "body",
                "model",
            ),
            _maybe_get_path(
                kwargs.get("passthrough_logging_payload"),
                "request_body",
                "model",
            ),
            _maybe_get_path(standard_logging_object, "request_body", "model"),
            kwargs.get("model"),
            standard_logging_object.get("model"),
            metadata.get("model"),
            resolved_model,
        )
        or "unknown"
    )


def _resolve_inbound_model_alias_from_langfuse(
    *,
    observation: Dict[str, Any],
    metadata: Dict[str, Any],
    input_model: Optional[str],
    output_model: Optional[str],
    resolved_model: str,
) -> str:
    return (
        _first_non_empty_string(
            metadata.get("model_alias_label"),
            metadata.get("requested_model_alias"),
            metadata.get("codex_auto_agent_alias"),
            metadata.get("anthropic_auto_agent_alias"),
            metadata.get("aawm_auto_agent_alias"),
            input_model,
            metadata.get("model"),
            observation.get("model"),
            output_model,
            resolved_model,
        )
        or "unknown"
    )


def _normalize_session_history_model_group(
    model_group: Optional[str],
    metadata: Dict[str, Any],
    resolved_model: str,
) -> Optional[str]:
    normalized_group = _clean_non_empty_string(model_group)
    if normalized_group is None:
        return None
    group_lower = normalized_group.lower()

    auto_agent_aliases: Tuple[Tuple[Optional[str], Tuple[Any, ...]], ...] = (
        (
            _clean_non_empty_string(metadata.get("codex_auto_agent_alias")),
            (
                metadata.get("codex_auto_agent_selected_model"),
                metadata.get("aawm_auto_agent_selected_model"),
            ),
        ),
        (
            _clean_non_empty_string(metadata.get("anthropic_auto_agent_alias")),
            (
                metadata.get("anthropic_auto_agent_selected_model"),
                metadata.get("aawm_auto_agent_selected_model"),
            ),
        ),
        (
            _clean_non_empty_string(metadata.get("aawm_auto_agent_alias")),
            (
                metadata.get("aawm_auto_agent_selected_model"),
                metadata.get("codex_auto_agent_selected_model"),
                metadata.get("anthropic_auto_agent_selected_model"),
            ),
        ),
        (
            _clean_non_empty_string(metadata.get("requested_model_alias")),
            (
                metadata.get("codex_auto_agent_selected_model"),
                metadata.get("anthropic_auto_agent_selected_model"),
                metadata.get("aawm_auto_agent_selected_model"),
            ),
        ),
    )
    for auto_alias, selected_model_candidates in auto_agent_aliases:
        if auto_alias and group_lower == auto_alias.lower():
            return _first_non_empty_string(*selected_model_candidates, resolved_model)

    if group_lower == "aawm-codex-agent-auto":
        return _first_non_empty_string(
            metadata.get("codex_auto_agent_selected_model"),
            metadata.get("aawm_auto_agent_selected_model"),
            resolved_model,
        )
    return normalized_group


def _is_completion_call_type(call_type: Any) -> bool:
    if not isinstance(call_type, str) or not call_type.strip():
        return False
    return "completion" in call_type.strip().lower()


def _is_embedding_call_type(call_type: Any, api_base: Optional[str]) -> bool:
    call_lower = str(call_type or "").strip().lower()
    if "embedding" in call_lower or "aembedding" in call_lower:
        return True
    sanitized = _sanitize_session_history_api_base(api_base)
    if not sanitized:
        return False
    try:
        path = urlsplit(sanitized).path.lower()
    except ValueError:
        return False
    return "embedding" in path


def _strip_local_provider_model_prefix(model: str) -> str:
    normalized = str(model or "").strip()
    lowered = normalized.lower()
    for prefix in ("local_embed/", "local_rerank/", "local_llm/", "local_biomed/"):
        if lowered.startswith(prefix):
            return normalized[len(prefix) :].strip() or normalized
    return normalized


def _session_history_provider_from_api_base(
    api_base: Any,
    *,
    call_type: Any = None,
) -> Optional[str]:
    sanitized = _sanitize_session_history_api_base(api_base)
    if not sanitized:
        return None
    api_base_lower = sanitized.lower()
    if "api.x.ai" in api_base_lower or "cli-chat-proxy.grok.com" in api_base_lower:
        return "xai"
    if "integrate.api.nvidia.com" in api_base_lower:
        return "nvidia_nim"
    if "openrouter.ai" in api_base_lower:
        return "openrouter"
    if "opencode.ai/zen" in api_base_lower:
        return "opencode_zen"
    if "anthropic.com" in api_base_lower:
        return "anthropic"
    if "googleapis.com" in api_base_lower or "generativelanguage" in api_base_lower:
        return "gemini"
    if "openai.com" in api_base_lower:
        return "openai"
    if _is_local_session_history_api_base(sanitized) and _is_embedding_call_type(
        call_type,
        sanitized,
    ):
        return "local_embed"
    return None


def _apply_local_embedding_route_metadata(
    *,
    metadata: Dict[str, Any],
    resolved_provider: Optional[str],
    resolved_model: str,
    model_group: Optional[str],
    call_type: Any,
    api_base: Optional[str],
) -> Tuple[Optional[str], str]:
    if not _is_embedding_call_type(call_type, api_base):
        return resolved_provider, resolved_model
    if not _is_local_session_history_api_base(api_base):
        return resolved_provider, resolved_model
    if resolved_provider not in {None, "openai", "local_embed"}:
        return resolved_provider, resolved_model

    upstream_model = _strip_local_provider_model_prefix(resolved_model)
    route_model = _clean_non_empty_string(upstream_model) or _clean_non_empty_string(model_group)
    if not route_model:
        return "local_embed", resolved_model

    metadata["aawm_local_route"] = True
    metadata["aawm_local_route_family"] = "local_embedding"
    if model_group:
        metadata["aawm_local_model_group"] = model_group
    metadata["aawm_local_upstream_provider"] = "local_embed"
    metadata["aawm_local_upstream_model"] = route_model
    sanitized_api_base = _sanitize_session_history_api_base(api_base)
    if sanitized_api_base:
        metadata["aawm_local_upstream_api_base"] = sanitized_api_base

    return "local_embed", route_model


def _apply_local_llm_route_metadata(
    *,
    metadata: Dict[str, Any],
    resolved_provider: Optional[str],
    resolved_model: str,
    model_group: Optional[str],
    call_type: Any,
    api_base: Optional[str],
) -> Tuple[Optional[str], str]:
    if (
        resolved_provider != "openai"
        or not model_group
        or not api_base
        or not _is_completion_call_type(call_type)
        or not _is_local_session_history_api_base(api_base)
    ):
        return resolved_provider, resolved_model

    upstream_model = _clean_non_empty_string(_strip_local_provider_model_prefix(resolved_model)) or model_group

    metadata["aawm_local_route"] = True
    metadata["aawm_local_route_family"] = "local_llm_chat"
    metadata["aawm_local_model_group"] = model_group
    metadata["aawm_local_upstream_provider"] = "openai"
    metadata["aawm_local_upstream_model"] = upstream_model
    sanitized_api_base = _sanitize_session_history_api_base(api_base)
    if sanitized_api_base:
        metadata["aawm_local_upstream_api_base"] = sanitized_api_base

    return "local_llm", model_group


_LOCAL_BIOMED_SESSION_HISTORY_ROUTES = {
    (8094, "/extract"): {
        "model": "scispacy",
        "service": "scispacy",
        "endpoint": "extract",
    },
    (8095, "/annotate"): {
        "model": "tinybern2",
        "service": "tinybern2",
        "endpoint": "annotate",
    },
}


def _resolve_local_biomed_session_history_route(
    api_base: Optional[str],
) -> Optional[Dict[str, str]]:
    sanitized = _sanitize_session_history_api_base(api_base)
    if not sanitized:
        return None

    try:
        parsed = urlsplit(sanitized)
    except ValueError:
        return None

    route_info = _LOCAL_BIOMED_SESSION_HISTORY_ROUTES.get((parsed.port or 0, parsed.path.rstrip("/")))
    if route_info is None:
        return None
    return dict(route_info)


def _apply_local_biomed_route_metadata(
    *,
    metadata: Dict[str, Any],
    resolved_provider: Optional[str],
    resolved_model: str,
    model_group: Optional[str],
    call_type: Any,
    api_base: Optional[str],
) -> Tuple[Optional[str], str, Optional[str]]:
    if str(call_type or "").strip().lower() != "pass_through_endpoint":
        return resolved_provider, resolved_model, model_group

    route_info = _resolve_local_biomed_session_history_route(api_base)
    if route_info is None:
        return resolved_provider, resolved_model, model_group

    route_model = route_info["model"]
    sanitized_api_base = _sanitize_session_history_api_base(api_base)
    metadata["aawm_local_route"] = True
    metadata["aawm_local_route_family"] = "local_biomed_rest"
    metadata["aawm_local_model_group"] = route_model
    metadata["aawm_local_service"] = route_info["service"]
    metadata["aawm_local_endpoint"] = route_info["endpoint"]
    metadata["aawm_local_upstream_provider"] = "local_rest"
    metadata["aawm_local_upstream_model"] = route_model
    if sanitized_api_base:
        metadata["aawm_local_upstream_api_base"] = sanitized_api_base
        metadata["aawm_local_upstream_url"] = sanitized_api_base
    metadata.setdefault("passthrough_route_family", "local_biomed")

    return "local_biomed", route_model, model_group or route_model


def _resolve_session_history_model(
    kwargs: Dict[str, Any],
    standard_logging_object: Dict[str, Any],
    metadata: Dict[str, Any],
    result: Any,
) -> str:
    grok_model_override = _resolve_xai_grok_model_override(kwargs, metadata)
    if grok_model_override:
        return grok_model_override

    explicit_openrouter_model = _first_explicit_openrouter_model_string(
        metadata.get("codex_auto_agent_selected_model"),
        metadata.get("anthropic_auto_agent_selected_model"),
        metadata.get("aawm_auto_agent_selected_model"),
        metadata.get("anthropic_adapter_original_model"),
        metadata.get("codex_adapter_original_model"),
        _maybe_get_path(
            kwargs.get("litellm_params"),
            "proxy_server_request",
            "body",
            "model",
        ),
        _maybe_get_path(
            kwargs.get("passthrough_logging_payload"),
            "request_body",
            "model",
        ),
        _maybe_get_path(standard_logging_object, "request_body", "model"),
        metadata.get("model"),
        kwargs.get("model"),
        standard_logging_object.get("model"),
    )
    if explicit_openrouter_model is not None:
        return explicit_openrouter_model

    if str(kwargs.get("custom_llm_provider") or "").lower() == "openrouter":
        for candidate in (
            _maybe_get_path(
                kwargs.get("litellm_params"),
                "proxy_server_request",
                "body",
                "model",
            ),
            _maybe_get_path(
                kwargs.get("passthrough_logging_payload"),
                "request_body",
                "model",
            ),
            _maybe_get_path(standard_logging_object, "request_body", "model"),
        ):
            if candidate is None:
                continue
            normalized = str(candidate).strip()
            if normalized.startswith("openrouter/"):
                return normalized

    result_completed_payload = _extract_responses_completed_payload_from_passthrough_fallback_text(
        _maybe_get(result, "response")
    )
    standard_completed_payload = _extract_responses_completed_payload_from_passthrough_fallback_text(
        _maybe_get(standard_logging_object.get("response"), "response")
    )
    candidates = (
        metadata.get("codex_auto_agent_selected_model"),
        metadata.get("anthropic_auto_agent_selected_model"),
        metadata.get("aawm_auto_agent_selected_model"),
        kwargs.get("model"),
        standard_logging_object.get("model"),
        _session_history_model_from_request_tags(metadata),
        _maybe_get_path(kwargs.get("passthrough_logging_payload"), "request_body", "model"),
        _maybe_get_path(kwargs.get("litellm_params"), "proxy_server_request", "body", "model"),
        _session_history_adapter_model(metadata),
        metadata.get("anthropic_adapter_model"),
        metadata.get("codex_adapter_model"),
        metadata.get("model"),
        _maybe_get(result, "model"),
        _maybe_get(_maybe_get(result_completed_payload, "response"), "model"),
        _maybe_get(_maybe_get(standard_completed_payload, "response"), "model"),
    )
    for candidate in candidates:
        if candidate is None:
            continue
        normalized = str(candidate).strip()
        if normalized and normalized.lower() != "unknown":
            return normalized
    return "unknown"


def _resolve_xai_grok_model_override(
    kwargs: Dict[str, Any],
    metadata: Dict[str, Any],
) -> Optional[str]:
    provider = str(kwargs.get("custom_llm_provider") or "").strip().lower()
    route_family = str(metadata.get("passthrough_route_family") or "").strip().lower()
    if provider not in {"xai", "grok"} and "grok" not in route_family:
        return None

    headers = _extract_request_headers_from_kwargs(kwargs)
    for candidate in (
        _get_header_value(headers, "x-grok-model-override"),
        metadata.get("grok_model_override"),
        metadata.get("model_group"),
        metadata.get("model"),
    ):
        normalized = _clean_non_empty_string(candidate)
        if normalized and normalized.lower() != "unknown":
            return normalized
    return None


# _build_session_history_record moved to litellm.integrations.aawm_session_history.record
# _build_session_history_db_payload moved to litellm.integrations.aawm_session_history.record
def _strip_postgres_nul_bytes(value: Any) -> Any:
    if isinstance(value, str):
        return value.replace("\x00", "")
    if isinstance(value, list):
        return [_strip_postgres_nul_bytes(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_strip_postgres_nul_bytes(item) for item in value)
    if isinstance(value, dict):
        return {
            _strip_postgres_nul_bytes(key): _strip_postgres_nul_bytes(nested_value)
            for key, nested_value in value.items()
        }
    return value


# _build_tool_activity_db_payloads moved to litellm.integrations.aawm_session_history.record
def _tool_definition_snapshot_from_metadata(
    metadata: Dict[str, Any],
) -> Optional[List[Any]]:
    snapshot = metadata.get(_AAWM_TOOL_DEFINITION_SNAPSHOT_METADATA_KEY)
    if isinstance(snapshot, list) and snapshot:
        return _json_safe_rate_limit_value(snapshot)
    return None


def _build_tool_definition_snapshot_db_payload(
    record: Dict[str, Any],
) -> Optional[Tuple[Any, ...]]:
    record = _strip_postgres_nul_bytes(record)
    metadata = record.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}

    snapshot = record.get(_AAWM_TOOL_DEFINITION_SNAPSHOT_METADATA_KEY)
    if not isinstance(snapshot, list) or not snapshot:
        snapshot = metadata.get(_AAWM_TOOL_DEFINITION_SNAPSHOT_METADATA_KEY)
    if not isinstance(snapshot, list) or not snapshot:
        return None

    session_id = _clean_non_empty_string(record.get("session_id"))
    snapshot_hash = _clean_non_empty_string(
        record.get("aawm_tool_definition_snapshot_hash") or metadata.get("aawm_tool_definition_snapshot_hash")
    )
    if not session_id or not snapshot_hash:
        return None

    durable_metadata = {
        "storage": "session_history_tool_definition_snapshots",
        "storage_key": "session_id,aawm_tool_definition_snapshot_hash",
        "provider": record.get("provider"),
        "model": record.get("model"),
        "model_group": record.get("model_group"),
        "repository": record.get("repository"),
    }
    sources = metadata.get("aawm_tool_definition_sources")
    names = metadata.get("aawm_tool_definition_names")
    tool_types = metadata.get("aawm_tool_definition_types")
    return (
        session_id,
        snapshot_hash,
        _clean_non_empty_string(metadata.get("aawm_tool_definition_capture_version")),
        _clean_non_empty_string(metadata.get("aawm_tool_definition_capture_source")),
        _safe_int(metadata.get("aawm_tool_definition_count")),
        _safe_int(metadata.get("aawm_tool_definition_captured_count")),
        json.dumps(_json_safe_rate_limit_value(sources if isinstance(sources, list) else [])),
        json.dumps(_json_safe_rate_limit_value(names if isinstance(names, list) else [])),
        json.dumps(_json_safe_rate_limit_value(tool_types if isinstance(tool_types, list) else [])),
        bool(metadata.get("aawm_tool_definition_snapshot_truncated")),
        json.dumps(_json_safe_rate_limit_value(snapshot)),
        _clean_non_empty_string(record.get("litellm_call_id")),
        _clean_non_empty_string(record.get("trace_id")),
        json.dumps(_json_safe_rate_limit_value(durable_metadata)),
    )


# _build_tool_definition_snapshot_db_payloads moved to litellm.integrations.aawm_session_history.record
# _persist_tool_definition_snapshots_best_effort moved to litellm.integrations.aawm_session_history.record
async def _lookup_claude_auto_review_parent_identity(
    conn: Any,
    payload: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    session_id = _clean_non_empty_string(payload.get("session_id"))
    if not session_id:
        return None
    reference_time = (
        _parse_datetime_value(payload.get("start_time"))
        or _parse_datetime_value(payload.get("observed_at"))
        or _parse_datetime_value(payload.get("end_time"))
    )
    rows = await conn.fetch(
        _AAWM_CLAUDE_AUTO_REVIEW_PARENT_IDENTITY_SQL,
        session_id,
        reference_time,
    )
    for row in rows:
        try:
            candidate = dict(row)
        except Exception:
            candidate = {
                "id": _maybe_get(row, "id"),
                "repository": _maybe_get(row, "repository"),
                "tenant_id": _maybe_get(row, "tenant_id"),
                "agent_name": _maybe_get(row, "agent_name"),
                "metadata": _maybe_get(row, "metadata"),
            }
        identity = _extract_claude_auto_review_identity_from_row(candidate)
        if identity:
            return identity
    return None


async def _apply_claude_auto_review_parent_identity_from_store(
    conn: Any,
    payload: Dict[str, Any],
    identity_by_session: Optional[Dict[str, Dict[str, Any]]] = None,
) -> None:
    metadata = payload.get("metadata")
    if not _is_claude_permission_check_metadata(metadata):
        return

    session_id = _clean_non_empty_string(payload.get("session_id"))
    identity = (identity_by_session or {}).get(session_id or "")
    if identity is None:
        identity = await _lookup_claude_auto_review_parent_identity(conn, payload)
    if identity is not None:
        _apply_claude_auto_review_parent_identity(payload, identity)
        return

    _apply_claude_auto_review_identity_to_record(payload)


def _extract_session_history_call_ids_from_payloads(
    payloads: List[Tuple[Any, ...]],
) -> List[str]:
    call_ids: List[str] = []
    seen_call_ids: set[str] = set()
    for payload in payloads:
        call_id = _clean_non_empty_string(payload[0] if payload else None)
        if call_id is None or call_id in seen_call_ids:
            continue
        call_ids.append(call_id)
        seen_call_ids.add(call_id)
    return call_ids


async def _update_session_history_previous_gap_ms(
    conn: Any,
    payloads: List[Tuple[Any, ...]],
) -> None:
    call_ids = _extract_session_history_call_ids_from_payloads(payloads)
    if not call_ids:
        return
    await conn.execute(_AAWM_SESSION_HISTORY_PREVIOUS_GAP_UPDATE_SQL, call_ids)


def _rate_limit_storage_provider(record: Dict[str, Any]) -> str:
    provider = _clean_non_empty_string(record.get("provider")) or "unknown"
    source = str(record.get("source") or "").lower()
    client_family = str(record.get("client_family") or "").lower()
    if provider == "antigravity" or client_family == "antigravity_code_assist" or source.startswith("antigravity_"):
        return "antigravity"
    if provider in {"opencode", "opencode_zen"} or client_family == "opencode_zen" or source.startswith("opencode_"):
        return "opencode_zen"
    if (
        provider in {"gemini", "google_code_assist"}
        or client_family in {"gemini", "google_code_assist"}
        or source.startswith("google_")
        or source.startswith("gemini_")
    ):
        return "google"
    return provider


def _rate_limit_storage_client(record: Dict[str, Any]) -> Optional[str]:
    return _first_non_empty_string(
        record.get("client_family"),
        record.get("client_name"),
        _maybe_get_path(record.get("metadata"), "client_name"),
    )


def _rate_limit_storage_quota_key(record: Dict[str, Any]) -> str:
    limit_id = _clean_non_empty_string(record.get("limit_id"))
    limit_scope = _clean_non_empty_string(record.get("limit_scope"))
    if limit_id and limit_scope:
        return f"{limit_id}:{limit_scope}"
    return (
        _clean_non_empty_string(record.get("limit_key"))
        or _clean_non_empty_string(record.get("limit_name"))
        or ":".join(
            part
            for part in (
                _clean_non_empty_string(record.get("source")),
                _clean_non_empty_string(record.get("model")),
            )
            if part
        )
        or "unknown_quota"
    )


def _rate_limit_storage_quota_type(record: Dict[str, Any]) -> str:
    explicit_quota_type = _clean_non_empty_string(record.get("quota_type"))
    if explicit_quota_type:
        return explicit_quota_type

    limit_scope = str(record.get("limit_scope") or "").lower()
    raw_provider_fields = record.get("raw_provider_fields")
    token_type = (
        str(raw_provider_fields.get("tokenType") or "").lower() if isinstance(raw_provider_fields, dict) else ""
    )
    source = str(record.get("source") or "").lower()
    provider = _rate_limit_storage_provider(record)

    if "request" in limit_scope or limit_scope == "requests" or token_type == "requests":
        return "requests"
    if "message" in limit_scope or token_type == "messages":
        return "messages"
    if "token" in limit_scope or limit_scope == "tokens" or token_type == "tokens":
        return "tokens"
    if limit_scope == "model_capacity" or "capacity" in source:
        return "capacity"
    if provider == "google":
        return "requests"
    if provider in {"openai", "anthropic"}:
        return "tokens"
    return "unknown"


def _rate_limit_storage_remaining_pct(record: Dict[str, Any]) -> Optional[float]:
    remaining_pct = _safe_float(record.get("remaining_pct"))
    if remaining_pct is not None:
        return max(0.0, min(100.0, remaining_pct))

    remaining_fraction = _safe_float(_maybe_get_path(record.get("raw_provider_fields"), "remainingFraction"))
    if remaining_fraction is not None:
        return max(0.0, min(100.0, remaining_fraction * 100.0))

    used_percentage = _safe_float(record.get("used_percentage"))
    if used_percentage is not None:
        return max(0.0, min(100.0, 100.0 - used_percentage))

    if bool(record.get("exhausted")):
        return 0.0
    return None


def _rate_limit_storage_numeric_detail(
    record: Dict[str, Any],
    key: str,
    *raw_paths: str,
) -> Optional[float]:
    direct_value = _nonnegative_float_or_none(record.get(key))
    if direct_value is not None:
        return direct_value
    raw_provider_fields = record.get("raw_provider_fields")
    if not isinstance(raw_provider_fields, dict):
        return None
    for raw_path in raw_paths:
        value: Any = raw_provider_fields
        for part in raw_path.split("."):
            if isinstance(value, dict):
                value = value.get(part)
            else:
                value = None
                break
        normalized = _nonnegative_float_or_none(value.get("val") if isinstance(value, dict) else value)
        if normalized is not None:
            return normalized
    return None


def _rate_limit_storage_quota_limit(record: Dict[str, Any]) -> Optional[float]:
    return _first_non_none(
        _rate_limit_storage_numeric_detail(
            record,
            "quota_limit",
            "monthlyLimit",
            "total",
            "limit",
            "x-ratelimit-limit-requests",
            "x-ratelimit-limit-tokens",
        ),
        _nonnegative_float_or_none(record.get("total_requests")),
    )


def _rate_limit_storage_quota_used(record: Dict[str, Any]) -> Optional[float]:
    return _first_non_none(
        _rate_limit_storage_numeric_detail(record, "quota_used", "used"),
        _nonnegative_float_or_none(record.get("used_requests")),
    )


def _rate_limit_storage_quota_remaining(record: Dict[str, Any]) -> Optional[float]:
    return _first_non_none(
        _rate_limit_storage_numeric_detail(
            record,
            "quota_remaining",
            "remaining",
            "x-ratelimit-remaining-requests",
            "x-ratelimit-remaining-tokens",
        ),
        _nonnegative_float_or_none(record.get("remaining_requests")),
    )


def _rate_limit_storage_timestamp_detail(
    record: Dict[str, Any],
    key: str,
    *raw_paths: str,
) -> Optional[datetime]:
    direct_value = _parse_provider_timestamp(record.get(key))
    if direct_value is not None:
        return direct_value
    raw_provider_fields = record.get("raw_provider_fields")
    if not isinstance(raw_provider_fields, dict):
        return None
    for raw_path in raw_paths:
        value: Any = raw_provider_fields
        for part in raw_path.split("."):
            if isinstance(value, dict):
                value = value.get(part)
            else:
                value = None
                break
        parsed = _parse_provider_timestamp(value)
        if parsed is not None:
            return parsed
    return None


def _rate_limit_storage_billing_period_start_at(
    record: Dict[str, Any],
) -> Optional[datetime]:
    return _rate_limit_storage_timestamp_detail(
        record,
        "billing_period_start_at",
        "billingPeriodStart",
    )


def _rate_limit_storage_billing_period_end_at(
    record: Dict[str, Any],
) -> Optional[datetime]:
    return _first_non_none(
        _rate_limit_storage_timestamp_detail(
            record,
            "billing_period_end_at",
            "billingPeriodEnd",
        ),
        _parse_provider_timestamp(record.get("provider_resets_at"))
        if record.get("quota_period") == "monthly"
        else None,
    )


def _build_rate_limit_observation_db_payload(record: Dict[str, Any]) -> Tuple[Any, ...]:
    return (
        record["observed_at"],
        _rate_limit_storage_client(record),
        record.get("client_version"),
        record.get("account_hash"),
        _rate_limit_storage_provider(record),
        record.get("model"),
        _rate_limit_storage_quota_key(record),
        record.get("quota_period"),
        _rate_limit_storage_quota_type(record),
        record.get("provider_resets_at"),
        _rate_limit_storage_remaining_pct(record),
        _rate_limit_storage_quota_limit(record),
        _rate_limit_storage_quota_used(record),
        _rate_limit_storage_quota_remaining(record),
        _rate_limit_storage_billing_period_start_at(record),
        _rate_limit_storage_billing_period_end_at(record),
        json.dumps(_json_safe_rate_limit_value(record.get("raw_provider_fields") or {})),
        json.dumps(_json_safe_rate_limit_value(record.get("evidence") or {})),
        record.get("source"),
        record.get("session_id"),
        record.get("trace_id"),
        record.get("litellm_call_id"),
    )


def _build_rate_limit_transition_db_payload(record: Dict[str, Any]) -> Tuple[Any, ...]:
    return (
        record["transition_key"],
        record["limit_key"],
        record.get("provider"),
        record.get("client_family"),
        record.get("account_hash"),
        record["transition_type"],
        record.get("confidence") or 0.0,
        json.dumps(_json_safe_rate_limit_value(record.get("signals") or [])),
        record.get("source"),
        record.get("old_observed_at"),
        record["new_observed_at"],
        record.get("old_provider_resets_at"),
        record.get("new_provider_resets_at"),
        record.get("old_used_percentage"),
        record.get("new_used_percentage"),
        record.get("old_remaining_requests"),
        record.get("new_remaining_requests"),
        record.get("old_used_requests"),
        record.get("new_used_requests"),
        record.get("old_total_requests"),
        record.get("new_total_requests"),
        record.get("inferred_window_start_at"),
        record.get("detection_window_start_at"),
        record.get("detection_window_end_at"),
        json.dumps(_json_safe_rate_limit_value(record.get("session_usage_summary") or {})),
        json.dumps(_json_safe_rate_limit_value(record.get("old_observation") or {})),
        json.dumps(_json_safe_rate_limit_value(record.get("new_observation") or {})),
        json.dumps(_json_safe_rate_limit_value(record.get("metadata") or {})),
    )


def _build_provider_error_observation_db_payload(record: Dict[str, Any]) -> Tuple[Any, ...]:
    return (
        record["observed_at"],
        record.get("environment"),
        record["provider"],
        record.get("model"),
        record.get("model_group"),
        record.get("route_family"),
        record.get("status_code"),
        record.get("error_type"),
        record.get("error_code"),
        record["error_class"],
        record.get("retry_after_seconds"),
        record.get("expected_reset_at"),
        record.get("session_id"),
        record.get("trace_id"),
        record.get("litellm_call_id"),
        json.dumps(_json_safe_rate_limit_value(record.get("metadata") or {})),
    )


def _extract_alias_routing_audit_events(
    record: Dict[str, Any],
) -> List[Dict[str, Any]]:
    metadata = record.get("metadata")
    event_sources: List[Any] = [record.get("aawm_alias_routing_audit_events")]
    if isinstance(metadata, dict):
        event_sources.extend(
            [
                metadata.get("aawm_alias_routing_audit_events"),
                metadata.get("codex_auto_agent_audit_events"),
                metadata.get("anthropic_auto_agent_audit_events"),
            ]
        )
    events: List[Dict[str, Any]] = []
    seen: Set[str] = set()
    for source in event_sources:
        if not isinstance(source, list):
            continue
        for event in source:
            if not isinstance(event, dict):
                continue
            try:
                fingerprint = json.dumps(
                    _json_safe_rate_limit_value(event),
                    sort_keys=True,
                    separators=(",", ":"),
                )
            except Exception:
                fingerprint = str(id(event))
            if fingerprint in seen:
                continue
            seen.add(fingerprint)
            events.append(event)
    return events


def _alias_routing_audit_observed_at(
    record: Dict[str, Any],
    event: Dict[str, Any],
) -> datetime:
    return (
        _parse_datetime_value(event.get("observed_at"))
        or _parse_datetime_value(record.get("start_time"))
        or _parse_datetime_value(record.get("end_time"))
        or datetime.now(timezone.utc)
    )


def _alias_routing_audit_event_key(
    *,
    record: Dict[str, Any],
    event: Dict[str, Any],
    event_index: int,
) -> Optional[str]:
    litellm_call_id = _clean_non_empty_string(event.get("litellm_call_id") or record.get("litellm_call_id"))
    if litellm_call_id is None:
        return None
    key_material = [
        litellm_call_id,
        event.get("alias_family"),
        event.get("alias_model"),
        event.get("event_type"),
        event.get("provider"),
        event.get("model"),
        event.get("attempt_number"),
        event.get("candidate_status"),
        event_index,
    ]
    digest = hashlib.sha256(json.dumps(key_material, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:24]
    return f"{litellm_call_id}:alias-routing:{digest}"


def _infer_alias_routing_family(
    event: Dict[str, Any],
    metadata: Dict[str, Any],
) -> str:
    return (
        _clean_non_empty_string(event.get("alias_family"))
        or ("codex_auto_agent" if _clean_non_empty_string(metadata.get("codex_auto_agent_alias")) else None)
        or ("anthropic_auto_agent" if _clean_non_empty_string(metadata.get("anthropic_auto_agent_alias")) else None)
        or "unknown"
    )


def _build_alias_routing_audit_db_payload(
    record: Dict[str, Any],
    event: Dict[str, Any],
    event_index: int,
) -> Tuple[Any, ...]:
    metadata = record.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
    event_metadata = dict(event)
    event_metadata["event_index"] = event_index
    event_metadata.setdefault("session_history_provider", record.get("provider"))
    event_metadata.setdefault("session_history_model", record.get("model"))
    event_metadata.setdefault("session_history_model_group", record.get("model_group"))
    event_metadata.setdefault("session_history_repository", record.get("repository"))
    return (
        _alias_routing_audit_event_key(
            record=record,
            event=event,
            event_index=event_index,
        ),
        _alias_routing_audit_observed_at(record, event),
        _clean_non_empty_string(event.get("session_id")) or _clean_non_empty_string(record.get("session_id")),
        _clean_non_empty_string(event.get("session_key")),
        _clean_non_empty_string(event.get("trace_id")) or _clean_non_empty_string(record.get("trace_id")),
        _clean_non_empty_string(event.get("litellm_call_id")) or _clean_non_empty_string(record.get("litellm_call_id")),
        _clean_non_empty_string(event.get("alias_model"))
        or _clean_non_empty_string(metadata.get("requested_model_alias"))
        or "unknown",
        _infer_alias_routing_family(event, metadata),
        _clean_non_empty_string(event.get("route_family")),
        _clean_non_empty_string(event.get("provider")),
        _clean_non_empty_string(event.get("model")),
        _clean_non_empty_string(event.get("lane_key")),
        _clean_non_empty_string(event.get("cooldown_key")),
        _safe_int(event.get("attempt_number")),
        _clean_non_empty_string(event.get("event_type")) or "unknown",
        _clean_non_empty_string(event.get("selection_reason")),
        _clean_non_empty_string(event.get("candidate_status")),
        _clean_non_empty_string(event.get("failure_class")),
        _safe_int(event.get("error_status_code")),
        _clean_non_empty_string(event.get("cooldown_scope")),
        _safe_float(event.get("cooldown_seconds")),
        _parse_datetime_value(event.get("cooldown_until")),
        _metadata_bool(event.get("selected")),
        _metadata_bool(event.get("skipped")),
        _metadata_bool(event.get("last_resort")),
        _metadata_bool(event.get("in_flight_session")),
        _metadata_bool(event.get("redispatch_required")),
        _metadata_bool(event.get("redispatch_threshold_crossed")),
        json.dumps(_json_safe_rate_limit_value(event_metadata)),
    )


# _persist_alias_routing_audit_best_effort moved to litellm.integrations.aawm_session_history.record
_AAWM_RATE_LIMIT_PREVIOUS_OBSERVATION_FIELDS: Tuple[str, ...] = (
    "observed_at",
    "source",
    "provider",
    "client_family",
    "account_hash",
    "environment",
    "tenant_id",
    "repository",
    "limit_key",
    "limit_id",
    "limit_name",
    "limit_scope",
    "window_minutes",
    "quota_period",
    "provider_resets_at",
    "inferred_window_start_at",
    "used_percentage",
    "remaining_requests",
    "used_requests",
    "total_requests",
    "status",
    "exhausted",
    "exhaustion_kind",
    "reset_hint_seconds",
    "model",
    "quota_limit",
    "quota_used",
    "quota_remaining",
    "billing_period_start_at",
    "billing_period_end_at",
    "model_family",
    "model_tier",
    "parent_limit_key",
    "session_id",
    "trace_id",
    "litellm_call_id",
    "route_family",
    "request_model",
    "response_model",
    "client_name",
    "client_version",
    "client_user_agent",
    "raw_provider_fields",
    "evidence",
    "metadata",
)


def _rate_limit_previous_observation_row_to_dict(row: Any) -> Dict[str, Any]:
    try:
        row_dict = dict(row)
    except Exception:
        return {key: _maybe_get(row, key) for key in _AAWM_RATE_LIMIT_PREVIOUS_OBSERVATION_FIELDS}
    row_dict.pop("input_limit_key", None)
    return row_dict


async def _fetch_previous_rate_limit_observation(
    conn: Any,
    observation: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    quota_key = _rate_limit_storage_quota_key(observation)
    if not quota_key or not observation.get("observed_at"):
        return None
    row = await conn.fetchrow(
        _AAWM_RATE_LIMIT_PREVIOUS_OBSERVATION_SQL,
        quota_key,
        _rate_limit_storage_provider(observation),
        _rate_limit_storage_client(observation),
        observation.get("account_hash"),
        observation.get("source"),
        observation["observed_at"],
    )
    if row is None:
        return None
    return _rate_limit_previous_observation_row_to_dict(row)


async def _fetch_previous_rate_limit_observations(
    conn: Any,
    observations: List[Dict[str, Any]],
) -> Dict[str, Optional[Dict[str, Any]]]:
    first_observation_by_limit_key: Dict[str, Dict[str, Any]] = {}
    for observation in observations:
        limit_key = _rate_limit_storage_quota_key(observation)
        if (
            not isinstance(limit_key, str)
            or not limit_key
            or not observation.get("observed_at")
            or limit_key in first_observation_by_limit_key
        ):
            continue
        first_observation_by_limit_key[limit_key] = observation

    if not first_observation_by_limit_key:
        return {}

    limit_keys: List[str] = []
    providers: List[str] = []
    clients: List[Optional[str]] = []
    account_hashes: List[Optional[str]] = []
    sources: List[Optional[str]] = []
    observed_ats: List[Any] = []
    for limit_key, observation in first_observation_by_limit_key.items():
        limit_keys.append(limit_key)
        providers.append(_rate_limit_storage_provider(observation))
        clients.append(_rate_limit_storage_client(observation))
        account_hashes.append(observation.get("account_hash"))
        sources.append(observation.get("source"))
        observed_ats.append(observation["observed_at"])

    previous_by_limit_key: Dict[str, Optional[Dict[str, Any]]] = {limit_key: None for limit_key in limit_keys}
    rows = await conn.fetch(
        _AAWM_RATE_LIMIT_PREVIOUS_OBSERVATIONS_BATCH_SQL,
        limit_keys,
        providers,
        clients,
        account_hashes,
        sources,
        observed_ats,
    )
    for row in rows:
        limit_key = _maybe_get(row, "input_limit_key")
        if isinstance(limit_key, str) and limit_key in previous_by_limit_key:
            previous_by_limit_key[limit_key] = _rate_limit_previous_observation_row_to_dict(row)
    return previous_by_limit_key


async def _derive_rate_limit_transitions(
    conn: Any,
    observations: List[Dict[str, Any]],
    initial_previous_by_limit_key: Optional[Dict[str, Optional[Dict[str, Any]]]] = None,
) -> List[Dict[str, Any]]:
    transitions: List[Dict[str, Any]] = []
    previous_by_limit_key: Dict[str, Optional[Dict[str, Any]]] = dict(initial_previous_by_limit_key or {})
    ordered_observations = sorted(
        observations,
        key=lambda item: (
            _rate_limit_storage_quota_key(item),
            item.get("observed_at") or datetime.min.replace(tzinfo=timezone.utc),
        ),
    )
    missing_previous_observations: List[Dict[str, Any]] = []
    for observation in ordered_observations:
        limit_key = _rate_limit_storage_quota_key(observation)
        if isinstance(limit_key, str) and limit_key and limit_key not in previous_by_limit_key:
            previous_by_limit_key[limit_key] = None
            missing_previous_observations.append(observation)
    if missing_previous_observations:
        previous_by_limit_key.update(
            await _fetch_previous_rate_limit_observations(
                conn,
                missing_previous_observations,
            )
        )
    for observation in ordered_observations:
        limit_key = _rate_limit_storage_quota_key(observation)
        if not isinstance(limit_key, str) or not limit_key:
            continue
        previous = previous_by_limit_key.get(limit_key)
        if previous is not None:
            classification = _classify_rate_limit_transition(previous, observation)
            if classification is not None:
                transitions.append(_build_rate_limit_transition(previous, observation, classification))
        previous_by_limit_key[limit_key] = observation
    return transitions


async def _filter_meaningful_rate_limit_observations(
    conn: Any,
    observations: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, Optional[Dict[str, Any]]]]:
    kept_by_index: List[Tuple[int, Dict[str, Any]]] = []
    rolling_previous_by_limit_key: Dict[str, Optional[Dict[str, Any]]] = {}
    initial_previous_by_limit_key: Dict[str, Optional[Dict[str, Any]]] = {}
    indexed_observations = [
        (index, observation)
        for index, observation in enumerate(observations)
        if isinstance(observation.get("limit_key"), str) and observation.get("limit_key")
    ]
    indexed_observations.sort(
        key=lambda item: (
            _rate_limit_storage_quota_key(item[1]),
            item[1].get("observed_at") or datetime.min.replace(tzinfo=timezone.utc),
            item[0],
        )
    )

    initial_previous_by_limit_key.update(
        await _fetch_previous_rate_limit_observations(
            conn,
            [observation for _index, observation in indexed_observations],
        )
    )
    rolling_previous_by_limit_key.update(initial_previous_by_limit_key)

    for index, observation in indexed_observations:
        limit_key = _rate_limit_storage_quota_key(observation)
        previous = rolling_previous_by_limit_key.get(limit_key)
        if not _rate_limit_observation_has_meaningful_change(previous, observation):
            continue

        kept_by_index.append((index, observation))
        rolling_previous_by_limit_key[limit_key] = observation

    kept_by_index.sort(key=lambda item: item[0])
    return [observation for _index, observation in kept_by_index], initial_previous_by_limit_key


# _build_rate_limit_observation_only_record moved to litellm.integrations.aawm_session_history.record
def _rate_limit_observation_only_requested(kwargs: Dict[str, Any]) -> bool:
    metadata = _merged_rate_limit_metadata(kwargs)
    return bool(metadata.get("aawm_rate_limit_observation_only"))


# _persist_rate_limit_observations_best_effort moved to litellm.integrations.aawm_session_history.record
# _persist_provider_error_observations_best_effort moved to litellm.integrations.aawm_session_history.record
# _session_history_transaction moved to litellm.integrations.aawm_session_history.record
# _persist_session_history_record moved to litellm.integrations.aawm_session_history.record
# _persist_session_history_records moved to litellm.integrations.aawm_session_history.record
def _get_reasoning_state_tags(
    provider_prefix: str,
    reasoning_content: str,
    thinking_blocks: List[dict],
) -> List[str]:
    stripped_reasoning = reasoning_content.strip()
    tags: List[str] = []
    if stripped_reasoning:
        tags.append("reasoning-present")
        tags.append(f"{provider_prefix}-reasoning-present")
    else:
        tags.append("reasoning-empty")
        tags.append(f"{provider_prefix}-reasoning-empty")

    if thinking_blocks:
        tags.append("thinking-blocks-present")
        tags.append(f"{provider_prefix}-thinking-blocks-present")
    else:
        tags.append("thinking-blocks-empty")
        tags.append(f"{provider_prefix}-thinking-blocks-empty")
    return tags


def _extract_claude_experiment_ids(decoded_bytes: bytes) -> List[str]:
    experiment_ids: List[str] = []
    for offset, current_byte in enumerate(decoded_bytes[:-2]):
        if current_byte != 0x32:
            continue
        candidate_length = decoded_bytes[offset + 1]
        candidate_start = offset + 2
        candidate_end = candidate_start + candidate_length
        if candidate_end > len(decoded_bytes):
            continue
        candidate_bytes = decoded_bytes[candidate_start:candidate_end]
        if not all(32 <= byte <= 126 for byte in candidate_bytes):
            continue
        decoded_match = candidate_bytes.decode("ascii", errors="ignore")
        if decoded_match.count("-") < 2:
            continue
        if decoded_match not in experiment_ids:
            experiment_ids.append(decoded_match)

    if experiment_ids:
        return experiment_ids

    for match in _CLAUDE_EXPERIMENT_ID_RE.findall(decoded_bytes):
        decoded_match = match.decode("ascii", errors="ignore")
        if decoded_match.count("-") < 2:
            continue
        if decoded_match not in experiment_ids:
            experiment_ids.append(decoded_match)
    return experiment_ids


def _enrich_claude_thinking_metadata(metadata: Dict[str, Any], message: Any) -> None:
    span_started_at = datetime.now(timezone.utc)
    thinking_blocks = _extract_thinking_blocks(message)
    if not thinking_blocks:
        return
    reasoning_content = _extract_reasoning_content(message, thinking_blocks)

    signatures: List[str] = []
    for block in thinking_blocks:
        if _maybe_get(block, "type") != "thinking":
            continue
        signature = _maybe_get(block, "signature")
        if isinstance(signature, str) and signature.strip():
            signatures.append(signature)

    if not signatures:
        return

    decoded_hashes: List[str] = []
    experiment_ids: List[str] = []
    decode_errors: List[str] = []
    decoded_any = False

    for signature in signatures:
        try:
            decoded_bytes = _decode_base64_bytes(signature)
            decoded_hashes.append(_short_hash(decoded_bytes))
            decoded_any = True
            for experiment_id in _extract_claude_experiment_ids(decoded_bytes):
                if experiment_id not in experiment_ids:
                    experiment_ids.append(experiment_id)
        except Exception as exc:
            decode_errors.append(str(exc))

    metadata["claude_thinking_signature_present"] = len(signatures) > 0
    metadata["claude_thinking_signature_count"] = len(signatures)
    metadata["claude_thinking_signature_hashes"] = decoded_hashes
    metadata["claude_thinking_signature_decoded"] = decoded_any
    metadata["claude_thinking_decode_version"] = "v1"
    metadata["claude_reasoning_content_present"] = bool(reasoning_content.strip())
    metadata["claude_reasoning_content_empty_or_short"] = len(reasoning_content.strip()) < 16
    if experiment_ids:
        metadata["claude_thinking_experiment_ids"] = experiment_ids
        if len(experiment_ids) == 1:
            metadata["claude_thinking_experiment_id"] = experiment_ids[0]
    if decode_errors:
        metadata["claude_thinking_decode_errors"] = decode_errors

    metadata["thinking_signature_present"] = True
    metadata["thinking_signature_decoded"] = decoded_any
    metadata["reasoning_content_present"] = bool(reasoning_content.strip())
    metadata["reasoning_content_empty_or_short"] = len(reasoning_content.strip()) < 16
    metadata["thinking_blocks_present"] = len(thinking_blocks) > 0

    tags_to_add = ["claude-thinking-signature", "thinking-signature-present"]
    if decoded_any:
        tags_to_add.extend(["claude-thinking-decoded", "thinking-signature-decoded"])
    tags_to_add.extend(
        _get_reasoning_state_tags(
            provider_prefix="claude",
            reasoning_content=reasoning_content,
            thinking_blocks=thinking_blocks,
        )
    )
    tags_to_add.extend(f"claude-exp:{experiment_id}" for experiment_id in experiment_ids)
    _merge_tags(metadata, tags_to_add)
    _append_langfuse_span(
        metadata,
        name="claude.thinking_signature_decode",
        span_metadata={
            "signature_count": len(signatures),
            "decoded_signature_count": len(decoded_hashes),
            "thinking_block_count": len(thinking_blocks),
            "reasoning_content_present": bool(reasoning_content.strip()),
            "experiment_ids": experiment_ids,
        },
        start_time=span_started_at,
        end_time=datetime.now(timezone.utc),
    )


def _read_varint(data: bytes, offset: int) -> Tuple[Optional[int], int]:
    value = 0
    shift = 0
    current_offset = offset
    while current_offset < len(data):
        current_byte = data[current_offset]
        value |= (current_byte & 0x7F) << shift
        current_offset += 1
        if current_byte < 0x80:
            return value, current_offset
        shift += 7
        if shift > 63:
            break
    return None, offset


def _extract_gemini_signature_summary(signature: str) -> Dict[str, Any]:
    decoded_bytes = _decode_base64_bytes(signature)
    signature_hash = _short_hash(decoded_bytes)

    record_sizes: List[int] = []
    prefixes: List[str] = []
    marker_offsets: List[int] = []
    indexed_fields: Dict[str, Any] = {}

    offset = 0
    record_index = 0
    while offset < len(decoded_bytes):
        if decoded_bytes[offset] != 0x0A:
            break
        record_size, payload_offset = _read_varint(decoded_bytes, offset + 1)
        if record_size is None:
            break
        payload_end = payload_offset + record_size
        if payload_end > len(decoded_bytes):
            break

        payload = decoded_bytes[payload_offset:payload_end]
        marker_index = payload.find(_GEMINI_MARKER)
        prefix_hex = ""
        absolute_marker_offset = None
        if marker_index >= 0:
            prefix_hex = payload[:marker_index].hex()
            absolute_marker_offset = payload_offset + marker_index
            marker_offsets.append(absolute_marker_offset)

        record_sizes.append(record_size)
        prefixes.append(prefix_hex)
        indexed_fields[f"gemini_tsig_0_record_{record_index}_size"] = record_size
        indexed_fields[f"gemini_tsig_0_record_{record_index}_prefix"] = prefix_hex
        if absolute_marker_offset is not None:
            indexed_fields[f"gemini_tsig_0_record_{record_index}_marker_offset"] = absolute_marker_offset

        record_index += 1
        offset = payload_end

    shape_components = {
        "decoded_bytes": len(decoded_bytes),
        "record_sizes": record_sizes,
        "prefixes": prefixes,
        "marker_offsets": marker_offsets,
    }
    shape_hash = _short_hash(str(shape_components).encode("utf-8"))

    summary: Dict[str, Any] = {
        "decoded_bytes": len(decoded_bytes),
        "record_count": len(record_sizes),
        "record_sizes": record_sizes,
        "prefixes": prefixes,
        "marker_offsets": marker_offsets,
        "marker_hex": _GEMINI_MARKER.hex(),
        "shape_hash": shape_hash,
        "signature_hash": signature_hash,
        "indexed_fields": indexed_fields,
    }
    return summary


def _enrich_gemini_thought_signature_metadata(  # noqa: PLR0915
    metadata: Dict[str, Any], message: Any
) -> None:
    span_started_at = datetime.now(timezone.utc)
    provider_specific_fields = _extract_provider_specific_fields(message)
    thought_signatures = provider_specific_fields.get("thought_signatures")
    thinking_blocks = _extract_thinking_blocks(message)
    reasoning_content = _extract_reasoning_content(message, thinking_blocks)

    if not isinstance(thought_signatures, list):
        thought_signatures = []
    thought_signatures = [
        signature for signature in thought_signatures if isinstance(signature, str) and signature.strip()
    ]

    if not thought_signatures:
        return

    summaries: List[Dict[str, Any]] = []
    decode_errors: List[str] = []
    signature_hashes: List[str] = []
    shape_hashes: List[str] = []

    for index, signature in enumerate(thought_signatures):
        try:
            summary = _extract_gemini_signature_summary(signature)
            summaries.append(summary)
            signature_hashes.append(summary["signature_hash"])
            shape_hashes.append(summary["shape_hash"])
            metadata[f"gemini_tsig_{index}_decoded_bytes"] = summary["decoded_bytes"]
            metadata[f"gemini_tsig_{index}_record_count"] = summary["record_count"]
            metadata[f"gemini_tsig_{index}_record_sizes"] = summary["record_sizes"]
            metadata[f"gemini_tsig_{index}_prefixes"] = summary["prefixes"]
            metadata[f"gemini_tsig_{index}_marker_offsets"] = summary["marker_offsets"]
            metadata[f"gemini_tsig_{index}_marker_hex"] = summary["marker_hex"]
            metadata[f"gemini_tsig_{index}_shape_hash"] = summary["shape_hash"]

            indexed_fields = summary["indexed_fields"]
            for key, value in list(indexed_fields.items()):
                if key.startswith("gemini_tsig_0_"):
                    metadata[key.replace("gemini_tsig_0_", f"gemini_tsig_{index}_")] = value
        except Exception as exc:
            decode_errors.append(str(exc))

    metadata["gemini_thought_signature_present"] = len(thought_signatures) > 0
    metadata["gemini_thought_signature_count"] = len(thought_signatures)
    metadata["gemini_tsig_signature_hashes"] = signature_hashes
    metadata["gemini_tsig_shape_hashes"] = sorted(set(shape_hashes))
    metadata["gemini_reasoning_content_present"] = bool(reasoning_content.strip())
    metadata["gemini_reasoning_content_empty_or_short"] = len(reasoning_content.strip()) < 16
    metadata["gemini_thinking_blocks_present"] = len(thinking_blocks) > 0
    if summaries:
        first_summary = summaries[0]
        metadata["gemini_tsig_decoded_bytes"] = first_summary["decoded_bytes"]
        metadata["gemini_tsig_record_count"] = first_summary["record_count"]
        metadata["gemini_tsig_record_sizes"] = first_summary["record_sizes"]
        metadata["gemini_tsig_prefixes"] = first_summary["prefixes"]
        metadata["gemini_tsig_marker_offsets"] = first_summary["marker_offsets"]
        metadata["gemini_tsig_marker_hex"] = first_summary["marker_hex"]
        metadata["gemini_tsig_shape_hash"] = first_summary["shape_hash"]
    if decode_errors:
        metadata["gemini_tsig_decode_errors"] = decode_errors

    metadata["thinking_signature_present"] = True
    metadata["thinking_signature_decoded"] = len(summaries) > 0
    metadata["reasoning_content_present"] = bool(reasoning_content.strip())
    metadata["reasoning_content_empty_or_short"] = len(reasoning_content.strip()) < 16
    metadata["thinking_blocks_present"] = len(thinking_blocks) > 0

    tags_to_add = ["gemini-thought-signature", "thinking-signature-present"]
    if summaries:
        tags_to_add.extend(["gemini-thought-signature-decoded", "thinking-signature-decoded"])
        for shape_hash in sorted(set(shape_hashes)):
            tags_to_add.append(f"gemini-tsig-shape:{shape_hash}")
        for record_count in sorted({summary["record_count"] for summary in summaries}):
            tags_to_add.append(f"gemini-tsig-records:{record_count}")

    tags_to_add.extend(
        _get_reasoning_state_tags(
            provider_prefix="gemini",
            reasoning_content=reasoning_content,
            thinking_blocks=thinking_blocks,
        )
    )
    _merge_tags(metadata, tags_to_add)
    _append_langfuse_span(
        metadata,
        name="gemini.thought_signature_decode",
        span_metadata={
            "signature_count": len(thought_signatures),
            "decoded_signature_count": len(summaries),
            "shape_hashes": sorted(set(shape_hashes)),
            "record_counts": sorted({summary["record_count"] for summary in summaries} if summaries else []),
            "reasoning_content_present": bool(reasoning_content.strip()),
        },
        start_time=span_started_at,
        end_time=datetime.now(timezone.utc),
    )


def _enrich_agent_identity_metadata(
    kwargs: Dict[str, Any],
    metadata: Dict[str, Any],
) -> None:
    if (
        _is_codex_default_agent_context(kwargs, metadata)
        and not _clean_non_empty_string(metadata.get("agent_name"))
        and not _clean_non_empty_string(metadata.get("aawm_claude_agent_name"))
    ):
        metadata["agent_name"] = _DEFAULT_AGENT

    agent_context_name, agent_context_tenant_id = _extract_agent_context(kwargs)
    agent_id_repository = _extract_repository_identity_from_kwargs(
        kwargs,
        metadata=metadata,
        standard_logging_object=kwargs.get("standard_logging_object") or {},
    )
    agent_id, agent_id_source = _extract_agent_id_from_kwargs(
        kwargs,
        metadata=metadata,
        standard_logging_object=kwargs.get("standard_logging_object") or {},
        agent_name=agent_context_name,
        tenant_id=agent_context_tenant_id,
        repository=agent_id_repository,
    )
    if agent_id:
        metadata["agent_id"] = agent_id
        if agent_id_source:
            metadata["agent_id_source"] = agent_id_source
    else:
        metadata.pop("agent_id", None)
        metadata.pop("agent_id_source", None)


def _enrich_trace_name_and_provider_metadata(kwargs: Dict[str, Any], result: Any) -> Tuple[dict, Any]:
    agent_name = _extract_agent_name(kwargs)
    headers = _ensure_mutable_headers(kwargs)
    metadata = _ensure_mutable_metadata(kwargs)
    session_id = _extract_session_id(kwargs)
    is_grok_context = _is_native_grok_passthrough_context(metadata, headers)
    _enrich_claude_permission_check_metadata(kwargs, metadata, result)
    if _is_claude_permission_check_metadata(metadata):
        direct_repository = _extract_repository_identity_from_kwargs(
            kwargs,
            metadata=metadata,
            standard_logging_object=kwargs.get("standard_logging_object") or {},
        )
        direct_tenant_id, _tenant_source = _extract_tenant_identity_from_kwargs(
            kwargs,
            metadata=metadata,
            standard_logging_object=kwargs.get("standard_logging_object") or {},
        )
        _apply_claude_auto_review_metadata(
            metadata,
            repository=direct_repository,
            tenant_id=direct_tenant_id,
            source_model=_extract_claude_auto_review_source_model(
                metadata,
                _clean_non_empty_string(kwargs.get("model")),
            ),
        )

    current_trace_name = metadata.get("trace_name")
    if current_trace_name == "claude-code":
        metadata["trace_name"] = f"claude-code.{agent_name}"
    elif is_grok_context and (not current_trace_name or _is_generic_grok_trace_name(current_trace_name)):
        metadata["trace_name"] = (
            f"grok-build.{agent_name}" if agent_name and agent_name != _DEFAULT_AGENT else "grok-build"
        )
    elif not current_trace_name:
        metadata["trace_name"] = agent_name
    child_trace_user_id = _clean_non_empty_string(metadata.get("trace_user_id"))
    child_trace_name = _clean_non_empty_string(metadata.get("trace_name"))
    if headers and child_trace_name and child_trace_name.startswith("claude-code."):
        current_trace_name_header = _clean_non_empty_string(headers.get("langfuse_trace_name"))
        if (
            current_trace_name_header is None
            or current_trace_name_header == "claude-code"
            or current_trace_name_header.startswith("claude-code.")
        ) and current_trace_name_header != child_trace_name:
            headers["langfuse_trace_name"] = child_trace_name
            verbose_logger.debug(
                "AawmAgentIdentity: enriched header trace_name to %s",
                child_trace_name,
            )
    if headers and is_grok_context and child_trace_name:
        current_trace_name_header = _clean_non_empty_string(headers.get("langfuse_trace_name"))
        if (
            current_trace_name_header is None or _is_generic_grok_trace_name(current_trace_name_header)
        ) and current_trace_name_header != child_trace_name:
            headers["langfuse_trace_name"] = child_trace_name
            verbose_logger.debug(
                "AawmAgentIdentity: enriched Grok header trace_name to %s",
                child_trace_name,
            )
    if headers and child_trace_user_id and child_trace_name and child_trace_name.startswith("claude-code."):
        current_trace_user_id = headers.get("langfuse_trace_user_id")
        if current_trace_user_id != child_trace_user_id:
            headers["langfuse_trace_user_id"] = child_trace_user_id
            verbose_logger.debug(
                "AawmAgentIdentity: enriched header trace_user_id to %s",
                child_trace_user_id,
            )
    if session_id and not metadata.get("session_id"):
        metadata["session_id"] = session_id

    _promote_codex_repository_trace_user_id(kwargs, metadata, headers)
    _promote_grok_repository_trace_identity(kwargs, metadata, headers)
    _enrich_agent_identity_metadata(kwargs, metadata)
    _enrich_session_runtime_identity_metadata(kwargs)

    message = _extract_first_response_message(result)
    if message is not None:
        _enrich_claude_thinking_metadata(metadata, message)
        _enrich_gemini_thought_signature_metadata(metadata, message)
    _enrich_token_count_usage_metadata(kwargs, result)
    _enrich_usage_breakout_metadata(kwargs, result)
    _enrich_provider_cache_metadata(kwargs, result)

    _sync_standard_logging_object(kwargs, metadata)

    verbose_logger.debug(
        "AawmAgentIdentity: agent=%s, trace_name=%s, tags=%s",
        agent_name,
        metadata.get("trace_name"),
        metadata.get("tags"),
    )
    return kwargs, result


# _handle_session_history_success_event moved to litellm.integrations.aawm_session_history.record
# _handle_session_history_failure_event moved to litellm.integrations.aawm_session_history.record


# --- Wave A3A rate-limit / provider-error typed extraction: facade rebinds.
# These MUST precede _bind_session_history_record_apis() so record-API free
# names and monkeypatch targets keep resolving through this namespace (same
# rebind-order rule as Wave A2). The moved functions' __globals__ are rebound
# to this namespace by each submodule's install() below, so cross-module and
# cross-band free-name calls (and monkeypatches on this namespace) stay live. ---
from . import interfaces as _aawm_interfaces  # noqa: F401 - typed seams (RateLimitObservation, ObservationExtractor, ...)
from . import provider_errors as _aawm_provider_errors
from . import rate_limit_base as _aawm_rate_limit_base
from . import provider_cache as _aawm_provider_cache
from . import rate_limit_providers as _aawm_rate_limit_providers

ProviderCacheState = _aawm_interfaces.ProviderCacheState

# rebind installers: helper __globals__ -> this namespace (order: base ->
# providers -> errors, mirroring the original definition order so any
# install-time cross-reference resolves to an already-installed host binding).
_aawm_rate_limit_base.install(globals())
_aawm_rate_limit_providers.install(globals())
_aawm_provider_errors.install(globals())
_aawm_provider_cache.install(globals())

# literal facade assignments (AST-visible; installers above already published
# the rebound function objects into this namespace, these re-affirm identity).
# --- Wave A3B provider_cache facades ---
_normalize_provider_cache_family = _aawm_provider_cache._normalize_provider_cache_family
_supports_prompt_caching_safe = _aawm_provider_cache._supports_prompt_caching_safe
_extract_provider_cache_request_body = _aawm_provider_cache._extract_provider_cache_request_body
_request_contains_cache_control = _aawm_provider_cache._request_contains_cache_control
_request_contains_cached_content = _aawm_provider_cache._request_contains_cached_content
_request_contains_prompt_cache_key = _aawm_provider_cache._request_contains_prompt_cache_key
_openai_style_cached_tokens_source = _aawm_provider_cache._openai_style_cached_tokens_source
_usage_has_openai_style_cached_tokens_field = _aawm_provider_cache._usage_has_openai_style_cached_tokens_field
_usage_has_gemini_style_cached_content_field = _aawm_provider_cache._usage_has_gemini_style_cached_content_field
_openai_cache_attempt_source = _aawm_provider_cache._openai_cache_attempt_source
_extract_service_tier_hint = _aawm_provider_cache._extract_service_tier_hint
_price_cache_miss = _aawm_provider_cache._price_cache_miss
_determine_cache_attempt = _aawm_provider_cache._determine_cache_attempt
_determine_cache_outcome = _aawm_provider_cache._determine_cache_outcome
_compute_provider_cache_miss_cost_state = _aawm_provider_cache._compute_provider_cache_miss_cost_state
_provider_cache_state_from_metadata = _aawm_provider_cache._provider_cache_state_from_metadata
_resolve_provider_cache_state = _aawm_provider_cache._resolve_provider_cache_state
_enrich_provider_cache_metadata = _aawm_provider_cache._enrich_provider_cache_metadata
# --- end Wave A3B provider_cache facades ---

_parse_provider_timestamp = _aawm_rate_limit_base._parse_provider_timestamp
_infer_window_start_at = _aawm_rate_limit_base._infer_window_start_at
_quota_period_from_window_minutes = _aawm_rate_limit_base._quota_period_from_window_minutes
_normalize_quota_period = _aawm_rate_limit_base._normalize_quota_period
_window_minutes_from_quota_period = _aawm_rate_limit_base._window_minutes_from_quota_period
_parse_reset_hint_seconds = _aawm_rate_limit_base._parse_reset_hint_seconds
_resolve_rate_limit_reset_at = _aawm_rate_limit_base._resolve_rate_limit_reset_at
_json_safe_rate_limit_value = _aawm_rate_limit_base._json_safe_rate_limit_value
_coerce_rate_limit_payload = _aawm_rate_limit_base._coerce_rate_limit_payload
_iter_rate_limit_dicts = _aawm_rate_limit_base._iter_rate_limit_dicts
_merged_rate_limit_metadata = _aawm_rate_limit_base._merged_rate_limit_metadata
_extract_headers_from_kwargs = _aawm_rate_limit_base._extract_headers_from_kwargs
_extract_rate_limit_account_hash = _aawm_rate_limit_base._extract_rate_limit_account_hash
_resolve_rate_limit_model = _aawm_rate_limit_base._resolve_rate_limit_model
_infer_model_family_and_tier = _aawm_rate_limit_base._infer_model_family_and_tier
_infer_rate_limit_client_family = _aawm_rate_limit_base._infer_rate_limit_client_family
_build_rate_limit_key = _aawm_rate_limit_base._build_rate_limit_key
_build_rate_limit_context = _aawm_rate_limit_base._build_rate_limit_context
_finalize_rate_limit_observation = _aawm_rate_limit_base._finalize_rate_limit_observation
_dedupe_rate_limit_observations = _aawm_rate_limit_base._dedupe_rate_limit_observations
_rate_limit_snapshot_signature = _aawm_rate_limit_base._rate_limit_snapshot_signature
_rate_limit_observation_has_meaningful_change = _aawm_rate_limit_base._rate_limit_observation_has_meaningful_change
_rate_limit_candidate_roots = _aawm_rate_limit_base._rate_limit_candidate_roots
_openrouter_free_daily_request_limit = _aawm_rate_limit_providers._openrouter_free_daily_request_limit
_openrouter_free_shared_account_hash = _aawm_rate_limit_providers._openrouter_free_shared_account_hash
_is_openrouter_free_model = _aawm_rate_limit_providers._is_openrouter_free_model
_openrouter_free_daily_window = _aawm_rate_limit_providers._openrouter_free_daily_window
_openrouter_free_daily_observation_context_from_record = (
    _aawm_rate_limit_providers._openrouter_free_daily_observation_context_from_record
)
_build_openrouter_free_daily_observation = _aawm_rate_limit_providers._build_openrouter_free_daily_observation
_openrouter_free_record_observed_at = _aawm_rate_limit_providers._openrouter_free_record_observed_at
_is_openrouter_free_session_history_record = _aawm_rate_limit_providers._is_openrouter_free_session_history_record
_extract_codex_rate_limit_observations = _aawm_rate_limit_providers._extract_codex_rate_limit_observations
_extract_codex_header_rate_limit_observations = _aawm_rate_limit_providers._extract_codex_header_rate_limit_observations
_extract_error_payload_dicts = _aawm_rate_limit_providers._extract_error_payload_dicts
_extract_codex_usage_limit_error_observations = _aawm_rate_limit_providers._extract_codex_usage_limit_error_observations
_rate_limit_header_map = _aawm_rate_limit_providers._rate_limit_header_map
_get_rate_limit_header_value = _aawm_rate_limit_providers._get_rate_limit_header_value
_looks_like_claude_rate_limit_context = _aawm_rate_limit_providers._looks_like_claude_rate_limit_context
_extract_anthropic_header_rate_limit_observations = (
    _aawm_rate_limit_providers._extract_anthropic_header_rate_limit_observations
)
_first_quota_number = _aawm_rate_limit_providers._first_quota_number
_first_quota_float = _aawm_rate_limit_providers._first_quota_float
_looks_like_xai_oauth_rate_limit_context = _aawm_rate_limit_providers._looks_like_xai_oauth_rate_limit_context
_extract_xai_oauth_account_hash = _aawm_rate_limit_providers._extract_xai_oauth_account_hash
_xai_oauth_header_remaining_pct = _aawm_rate_limit_providers._xai_oauth_header_remaining_pct
_next_utc_month_start = _aawm_rate_limit_providers._next_utc_month_start
_is_xai_oauth_subscription_quota_context = _aawm_rate_limit_providers._is_xai_oauth_subscription_quota_context
_extract_xai_oauth_billing_period_end = _aawm_rate_limit_providers._extract_xai_oauth_billing_period_end
_extract_xai_oauth_header_rate_limit_observations = (
    _aawm_rate_limit_providers._extract_xai_oauth_header_rate_limit_observations
)
_grok_billing_quota_value = _aawm_rate_limit_providers._grok_billing_quota_value
_grok_billing_current_period = _aawm_rate_limit_providers._grok_billing_current_period
_grok_billing_is_weekly_period = _aawm_rate_limit_providers._grok_billing_is_weekly_period
_grok_billing_period_bounds = _aawm_rate_limit_providers._grok_billing_period_bounds
_is_grok_billing_context = _aawm_rate_limit_providers._is_grok_billing_context
_extract_grok_billing_config = _aawm_rate_limit_providers._extract_grok_billing_config
_grok_billing_model = _aawm_rate_limit_providers._grok_billing_model
_grok_billing_request_contract_evidence = _aawm_rate_limit_providers._grok_billing_request_contract_evidence
_grok_billing_snapshot_parts = _aawm_rate_limit_providers._grok_billing_snapshot_parts
_extract_grok_billing_observations = _aawm_rate_limit_providers._extract_grok_billing_observations
_extract_openrouter_free_error_reset_at = _aawm_rate_limit_providers._extract_openrouter_free_error_reset_at
_extract_openrouter_free_error_observations = _aawm_rate_limit_providers._extract_openrouter_free_error_observations
_looks_like_google_quota_candidate = _aawm_rate_limit_providers._looks_like_google_quota_candidate
_antigravity_quota_pool_for_model = _aawm_rate_limit_providers._antigravity_quota_pool_for_model
_extract_google_quota_observations = _aawm_rate_limit_providers._extract_google_quota_observations
_extract_google_error_observations = _aawm_rate_limit_providers._extract_google_error_observations
_build_rate_limit_observations = _aawm_rate_limit_providers._build_rate_limit_observations
_extract_provider_error_dicts = _aawm_provider_errors._extract_provider_error_dicts
_extract_embedded_json_payload_dicts = _aawm_provider_errors._extract_embedded_json_payload_dicts
_extract_provider_error_headers = _aawm_provider_errors._extract_provider_error_headers
_extract_provider_error_status_code = _aawm_provider_errors._extract_provider_error_status_code
_extract_provider_error_text = _aawm_provider_errors._extract_provider_error_text
_extract_provider_error_code_and_type = _aawm_provider_errors._extract_provider_error_code_and_type
_extract_provider_error_retry_after_seconds = _aawm_provider_errors._extract_provider_error_retry_after_seconds
_extract_litellm_provider_error_model_group = _aawm_provider_errors._extract_litellm_provider_error_model_group
_clean_litellm_provider_error_fallbacks = _aawm_provider_errors._clean_litellm_provider_error_fallbacks
_extract_litellm_provider_error_retry_context = _aawm_provider_errors._extract_litellm_provider_error_retry_context
_extract_provider_error_payload_metadata_value = _aawm_provider_errors._extract_provider_error_payload_metadata_value
_resolve_provider_error_model_group = _aawm_provider_errors._resolve_provider_error_model_group
_redact_upstream_error_raw = _aawm_provider_errors._redact_upstream_error_raw
_build_provider_error_fingerprint = _aawm_provider_errors._build_provider_error_fingerprint
_enrich_provider_error_observation_metadata = _aawm_provider_errors._enrich_provider_error_observation_metadata
_classify_provider_error = _aawm_provider_errors._classify_provider_error
_extract_provider_error_fields = _aawm_provider_errors._extract_provider_error_fields
_build_provider_error_observation = _aawm_provider_errors._build_provider_error_observation

# Typed seam re-exports (Wave A3A). These are specification types, not runtime
# replacements for the dict observations the extractors still emit this wave.
CallbackEnvelope = _aawm_interfaces.CallbackEnvelope
IdentityResolution = _aawm_interfaces.IdentityResolution
RateLimitObservation = _aawm_interfaces.RateLimitObservation
ObservationExtractor = _aawm_interfaces.ObservationExtractor


# --- Wave A2 identity leaf extractions: facade rebinds. These MUST precede
# _bind_session_history_record_apis() so record-API free names and
# monkeypatch targets keep resolving through this namespace. ---
from . import agent_context as _aawm_agent_context
from . import coerce as _aawm_coerce
from . import cost_map as _aawm_cost_map
from . import identity_repository as _aawm_identity_repository
from . import identity_runtime as _aawm_identity_runtime
from . import identity_tenant_agent as _aawm_identity_tenant_agent
from .constants import (  # noqa: F401 - re-exported into host globals for moved helpers
    _AAWM_AGENT_ID_HEX_RE,
    _AAWM_AGENT_ID_PREFIXED_RE,
    _AAWM_AGENT_ID_UUID_RE,
    _AAWM_ASSOCIATED_VERSION_ENV_VARS,
    _AAWM_LITELLM_ENVIRONMENT_ENV_VARS,
    _AAWM_LITELLM_VERSION_ENV_VARS,
    _AAWM_REPOSITORY_METADATA_KEYS,
    _AAWM_SESSION_HISTORY_METADATA_KEYS,
    _AAWM_TENANT_ID_METADATA_KEYS,
)

# literal facade assignments (AST-visible; installers finalize identity)
_clean_secret_string = _aawm_coerce._clean_secret_string
_get_first_secret_value = _aawm_coerce._get_first_secret_value
_normalize_aawm_sslmode = _aawm_coerce._normalize_aawm_sslmode
_build_aawm_dsn = _aawm_coerce._build_aawm_dsn
_append_aawm_dsn_query_params = _aawm_coerce._append_aawm_dsn_query_params
_clean_non_empty_string = _aawm_coerce._clean_non_empty_string
_first_non_empty_string = _aawm_coerce._first_non_empty_string
_coerce_string_dict = _aawm_coerce._coerce_string_dict
_load_bundled_model_cost_map = _aawm_cost_map._load_bundled_model_cost_map
_bundled_model_cost_casefold_lookup = _aawm_cost_map._bundled_model_cost_casefold_lookup
_lookup_bundled_model_cost_info = _aawm_cost_map._lookup_bundled_model_cost_info
_calculate_response_cost_from_bundled_model_cost_map = (
    _aawm_cost_map._calculate_response_cost_from_bundled_model_cost_map
)
_extract_claude_trace_agent_name = _aawm_identity_tenant_agent._extract_claude_trace_agent_name
_extract_claude_trace_user_identity_from_metadata_sources = (
    _aawm_identity_tenant_agent._extract_claude_trace_user_identity_from_metadata_sources
)
_extract_tenant_identity_from_kwargs = _aawm_identity_tenant_agent._extract_tenant_identity_from_kwargs
_extract_tenant_identity_from_langfuse_trace_observation = (
    _aawm_identity_tenant_agent._extract_tenant_identity_from_langfuse_trace_observation
)
_is_agent_id_like = _aawm_identity_tenant_agent._is_agent_id_like
_normalize_agent_id_identity = _aawm_identity_tenant_agent._normalize_agent_id_identity
_extract_agent_id_from_metadata_sources = _aawm_identity_tenant_agent._extract_agent_id_from_metadata_sources
_extract_agent_id_from_kwargs = _aawm_identity_tenant_agent._extract_agent_id_from_kwargs
_extract_agent_id_from_langfuse_trace_observation = (
    _aawm_identity_tenant_agent._extract_agent_id_from_langfuse_trace_observation
)
_normalize_repository_identity = _aawm_identity_repository._normalize_repository_identity
_normalize_repository_identity_from_absolute_path = (
    _aawm_identity_repository._normalize_repository_identity_from_absolute_path
)
_extract_repository_identity_from_text = _aawm_identity_repository._extract_repository_identity_from_text
_extract_repository_identity_from_value = _aawm_identity_repository._extract_repository_identity_from_value
_extract_repository_identity_from_metadata_sources = (
    _aawm_identity_repository._extract_repository_identity_from_metadata_sources
)
_extract_repository_identity_from_kwargs = _aawm_identity_repository._extract_repository_identity_from_kwargs
_extract_repository_identity_from_langfuse_trace_observation = (
    _aawm_identity_repository._extract_repository_identity_from_langfuse_trace_observation
)
_is_codex_memory_workflow_request = _aawm_identity_repository._is_codex_memory_workflow_request
_apply_codex_memory_workflow_repository = _aawm_identity_repository._apply_codex_memory_workflow_repository
_parse_client_identity_from_user_agent = _aawm_identity_runtime._parse_client_identity_from_user_agent
_extract_claude_code_version_from_metadata = _aawm_identity_runtime._extract_claude_code_version_from_metadata
_clean_session_history_client_ip_candidate = _aawm_identity_runtime._clean_session_history_client_ip_candidate
_canonical_session_history_client_ip = _aawm_identity_runtime._canonical_session_history_client_ip
_extract_agent_name = _aawm_agent_context._extract_agent_name
_is_native_codex_passthrough_context = _aawm_agent_context._is_native_codex_passthrough_context
_is_codex_client_identity = _aawm_agent_context._is_codex_client_identity
_is_codex_default_agent_context = _aawm_agent_context._is_codex_default_agent_context
_is_codex_subagent_context = _aawm_agent_context._is_codex_subagent_context
_is_native_grok_passthrough_context = _aawm_agent_context._is_native_grok_passthrough_context
_promote_grok_repository_trace_identity = _aawm_agent_context._promote_grok_repository_trace_identity
_promote_codex_repository_trace_user_id = _aawm_agent_context._promote_codex_repository_trace_user_id

# rebind installers: helper __globals__ -> this namespace
_aawm_coerce.install(globals())
_aawm_cost_map.install(globals())
_aawm_identity_tenant_agent.install(globals())
_aawm_identity_repository.install(globals())
_aawm_identity_runtime.install(globals())
_aawm_agent_context.install(globals())

_bind_session_history_record_apis()

# Static aliases for class methods / analyzers (values installed by bind above).
_handle_session_history_success_event = globals()["_handle_session_history_success_event"]
_handle_session_history_failure_event = globals()["_handle_session_history_failure_event"]
_build_failure_observation_only_record = globals()["_build_failure_observation_only_record"]


class AawmAgentIdentity(CustomLogger):
    """CustomLogger that enriches Langfuse trace_name with agent identity.

    Implements both sync logging_hook() and async async_logging_hook() to
    cover all code paths:
    - Sync: pass-through endpoints run Langfuse in sync success_handler (thread pool)
    - Async: standard LLM calls run Langfuse in async_success_handler
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        _bootstrap_session_history_spool_drainer_once()

    def logging_hook(self, kwargs: Dict[str, Any], result: Any, call_type: str) -> Tuple[dict, Any]:
        """Sync hook - runs before Langfuse in sync success handler."""
        try:
            return _enrich_trace_name_and_provider_metadata(kwargs, result)
        except Exception as exc:
            verbose_logger.warning("AawmAgentIdentity.logging_hook failed: %s", exc)
            return kwargs, result

    async def async_logging_hook(self, kwargs: Dict[str, Any], result: Any, call_type: str) -> Tuple[dict, Any]:
        """Async hook - runs before Langfuse in async success handler."""
        try:
            return _enrich_trace_name_and_provider_metadata(kwargs, result)
        except Exception as exc:
            verbose_logger.warning("AawmAgentIdentity.async_logging_hook failed: %s", exc)
            return kwargs, result

    def log_success_event(self, kwargs, response_obj, start_time, end_time):
        """Queue one finalized session-history row per completed LiteLLM call."""
        _handle_session_history_success_event(
            kwargs,
            response_obj,
            start_time,
            end_time,
            log_label="log_success_event",
        )

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time) -> None:
        _handle_session_history_success_event(
            kwargs,
            response_obj,
            start_time,
            end_time,
            log_label="async_log_success_event",
        )

    def log_failure_event(self, kwargs, response_obj, start_time, end_time):
        """Queue passive health observations from failed provider calls."""
        _handle_session_history_failure_event(
            kwargs,
            response_obj,
            start_time,
            end_time,
            log_label="log_failure_event",
        )

    async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time) -> None:
        _handle_session_history_failure_event(
            kwargs,
            response_obj,
            start_time,
            end_time,
            log_label="async_log_failure_event",
        )

    async def async_post_call_failure_hook(
        self,
        request_data: dict,
        original_exception: Exception,
        user_api_key_dict: Any,
        traceback_str: Optional[str] = None,
    ) -> None:
        try:
            kwargs = dict(request_data or {})
            kwargs.setdefault("user_api_key_dict", user_api_key_dict)
            now = datetime.now(timezone.utc)
            record = _build_failure_observation_only_record(
                kwargs,
                original_exception,
                now,
                now,
            )
            if record is not None:
                _enqueue_session_history_record(record)
        except Exception as exc:
            verbose_logger.warning(
                "AawmAgentIdentity.async_post_call_failure_hook failed: %s",
                exc,
            )
        return None


# Module-level instance for config registration via get_instance_fn().
# Config must reference this instance name, not the class name:
#   callbacks: ["litellm.integrations.aawm_agent_identity.aawm_agent_identity_instance"]
aawm_agent_identity_instance = AawmAgentIdentity()
