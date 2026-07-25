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
import base64  # noqa: F401 - consumed by moved enrich helpers via host globals
import asyncio  # noqa: F401 - monkeypatch surface for session_history writer tests
import atexit  # noqa: F401 - monkeypatch surface for session_history writer tests
import hashlib
import importlib  # noqa: F401 - monkeypatch surface for session_history writer tests
import inspect  # noqa: F401 - freevar seed for record APIs
import ipaddress  # noqa: F401 - consumed by moved provider_normalize helpers via host globals
import json
import math
import os
import queue  # noqa: F401 - monkeypatch surface for session_history writer tests
import re
import shlex  # noqa: F401 - consumed by moved tool_activity helpers via host globals
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


_AGENT_RE = re.compile(r"You are '([^']+)' and you are working")
_AGENT_TENANT_RE = re.compile(r"You are '(?P<agent>[^']+)' and you are working on the '(?P<tenant>[^']+)' project")
_AGENT_ROLE_RE = re.compile(
    r"^[ \t]*You are a '(?P<agent>explorer|worker|default)' agent\.[ \t]*$",
    re.MULTILINE,
)
_DEFAULT_AGENT = "orchestrator"
_CLAUDE_EXPERIMENT_ID_RE = re.compile(rb"(?<![A-Za-z0-9._-])([A-Za-z][A-Za-z0-9._-]{11,})(?![A-Za-z0-9._-])")
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




# _build_session_history_metadata moved to litellm.integrations.aawm_session_history.record
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
















# _handle_session_history_success_event moved to litellm.integrations.aawm_session_history.record
# _handle_session_history_failure_event moved to litellm.integrations.aawm_session_history.record


# --- Wave A4B tool-activity/claude-review/enrichment extraction.
# These MUST precede _bind_session_history_record_apis() so record-API free
# names and monkeypatch targets keep resolving through this namespace. ---
from . import tool_activity as _aawm_tool_activity
from . import claude_review as _aawm_claude_review
from . import enrich as _aawm_enrich

# rebind installers: helper __globals__ -> this namespace
_aawm_tool_activity.install(globals())
_aawm_claude_review.install(globals())
_aawm_enrich.install(globals())

# literal facade assignments (AST-visible; installers above already published
# the rebound function objects into this namespace, these re-affirm identity).
# --- Wave A4B tool_activity facades ---
_dedupe_strings = _aawm_tool_activity._dedupe_strings
_normalize_changed_file_path = _aawm_tool_activity._normalize_changed_file_path
_changed_file_basename = _aawm_tool_activity._changed_file_basename
_sensitive_config_change_flags_from_paths = _aawm_tool_activity._sensitive_config_change_flags_from_paths
_text_mentions_env_file = _aawm_tool_activity._text_mentions_env_file
_redact_sensitive_config_argument_value = _aawm_tool_activity._redact_sensitive_config_argument_value
_sanitize_tool_activity_arguments_for_sensitive_config = (
    _aawm_tool_activity._sanitize_tool_activity_arguments_for_sensitive_config
)
_normalize_sensitive_config_change_state_on_record = (
    _aawm_tool_activity._normalize_sensitive_config_change_state_on_record
)
_parse_tool_arguments = _aawm_tool_activity._parse_tool_arguments
_is_empty_claude_read_pages_value = _aawm_tool_activity._is_empty_claude_read_pages_value
_sanitize_tool_activity_arguments = _aawm_tool_activity._sanitize_tool_activity_arguments
_extract_paths_from_patch_text = _aawm_tool_activity._extract_paths_from_patch_text
_extract_file_paths_from_tool_arguments = _aawm_tool_activity._extract_file_paths_from_tool_arguments
_extract_command_text_from_tool_arguments = _aawm_tool_activity._extract_command_text_from_tool_arguments
_count_git_subcommand = _aawm_tool_activity._count_git_subcommand
_collect_file_paths_from_value = _aawm_tool_activity._collect_file_paths_from_value
_find_command_text_in_value = _aawm_tool_activity._find_command_text_in_value
_classify_tool_kind = _aawm_tool_activity._classify_tool_kind
_build_tool_activity_entry = _aawm_tool_activity._build_tool_activity_entry
_extract_tool_activity_from_message = _aawm_tool_activity._extract_tool_activity_from_message
_extract_response_output_items = _aawm_tool_activity._extract_response_output_items
_resolve_response_output_tool_name = _aawm_tool_activity._resolve_response_output_tool_name
_extract_response_output_tool_activity = _aawm_tool_activity._extract_response_output_tool_activity
_summarize_tool_activity = _aawm_tool_activity._summarize_tool_activity
_extract_tool_call_info = _aawm_tool_activity._extract_tool_call_info
_extract_response_output_tool_call_info = _aawm_tool_activity._extract_response_output_tool_call_info
_TOOL_ACTIVITY_READ_NAMES = _aawm_tool_activity._TOOL_ACTIVITY_READ_NAMES
_TOOL_ACTIVITY_MODIFY_NAMES = _aawm_tool_activity._TOOL_ACTIVITY_MODIFY_NAMES
_TOOL_ACTIVITY_COMMAND_NAMES = _aawm_tool_activity._TOOL_ACTIVITY_COMMAND_NAMES
_TOOL_ACTIVITY_SKIP_PATH_KEYS = _aawm_tool_activity._TOOL_ACTIVITY_SKIP_PATH_KEYS
_APPLY_PATCH_FILE_RE = _aawm_tool_activity._APPLY_PATCH_FILE_RE
_APPLY_PATCH_MOVE_TO_RE = _aawm_tool_activity._APPLY_PATCH_MOVE_TO_RE
_GIT_COMMAND_RE = _aawm_tool_activity._GIT_COMMAND_RE
_GIT_GLOBAL_OPTIONS_WITH_VALUES = _aawm_tool_activity._GIT_GLOBAL_OPTIONS_WITH_VALUES
_TOOL_ACTIVITY_COMMAND_TEXT_KEYS = _aawm_tool_activity._TOOL_ACTIVITY_COMMAND_TEXT_KEYS
_TOOL_ACTIVITY_COMMAND_TEXT_SKIP_KEYS = _aawm_tool_activity._TOOL_ACTIVITY_COMMAND_TEXT_SKIP_KEYS
_SENSITIVE_CONFIG_CHANGE_FIELDS = _aawm_tool_activity._SENSITIVE_CONFIG_CHANGE_FIELDS
_SENSITIVE_CONFIG_ENV_REDACTION = _aawm_tool_activity._SENSITIVE_CONFIG_ENV_REDACTION
_SENSITIVE_CONFIG_ENV_REDACT_ARGUMENT_KEYS = _aawm_tool_activity._SENSITIVE_CONFIG_ENV_REDACT_ARGUMENT_KEYS
_SENSITIVE_CONFIG_ENV_COMMAND_RE = _aawm_tool_activity._SENSITIVE_CONFIG_ENV_COMMAND_RE
_RESPONSE_OUTPUT_TOOL_ITEM_FALLBACK_NAMES = _aawm_tool_activity._RESPONSE_OUTPUT_TOOL_ITEM_FALLBACK_NAMES
_RESPONSE_OUTPUT_TOOL_ITEM_TYPES = _aawm_tool_activity._RESPONSE_OUTPUT_TOOL_ITEM_TYPES
# --- Wave A4B claude_review facades ---
_permission_check_probeable_value = _aawm_claude_review._permission_check_probeable_value
_extract_claude_permission_check_decision_from_value = (
    _aawm_claude_review._extract_claude_permission_check_decision_from_value
)
_extract_claude_permission_check_decision = _aawm_claude_review._extract_claude_permission_check_decision
_extract_claude_permission_check_models = _aawm_claude_review._extract_claude_permission_check_models
_enrich_claude_permission_check_metadata = _aawm_claude_review._enrich_claude_permission_check_metadata
_metadata_bool = _aawm_claude_review._metadata_bool
_metadata_request_tags = _aawm_claude_review._metadata_request_tags
_is_claude_permission_check_metadata = _aawm_claude_review._is_claude_permission_check_metadata
_extract_claude_project_from_metadata_tags = _aawm_claude_review._extract_claude_project_from_metadata_tags
_extract_claude_auto_review_source_model = _aawm_claude_review._extract_claude_auto_review_source_model
_apply_claude_auto_review_metadata = _aawm_claude_review._apply_claude_auto_review_metadata
_apply_claude_auto_review_identity_to_record = _aawm_claude_review._apply_claude_auto_review_identity_to_record
_extract_claude_auto_review_identity_from_row = _aawm_claude_review._extract_claude_auto_review_identity_from_row
_apply_claude_auto_review_parent_identity = _aawm_claude_review._apply_claude_auto_review_parent_identity
_build_session_identity_cache = _aawm_claude_review._build_session_identity_cache
_build_permission_usage_fields = _aawm_claude_review._build_permission_usage_fields
_lookup_claude_auto_review_parent_identity = _aawm_claude_review._lookup_claude_auto_review_parent_identity
_apply_claude_auto_review_parent_identity_from_store = (
    _aawm_claude_review._apply_claude_auto_review_parent_identity_from_store
)
_CLAUDE_PERMISSION_CHECK_OUTPUT_RE = _aawm_claude_review._CLAUDE_PERMISSION_CHECK_OUTPUT_RE
_CLAUDE_AUTO_REVIEW_LOGICAL_MODEL = _aawm_claude_review._CLAUDE_AUTO_REVIEW_LOGICAL_MODEL
_CLAUDE_AUTO_REVIEW_TRACE_NAME = _aawm_claude_review._CLAUDE_AUTO_REVIEW_TRACE_NAME
_CLAUDE_AUTO_REVIEW_AGENT_NAME = _aawm_claude_review._CLAUDE_AUTO_REVIEW_AGENT_NAME
# --- Wave A4B enrich facades ---
_bound_worker_context_exhaustion_string = _aawm_enrich._bound_worker_context_exhaustion_string
_normalize_worker_context_exhaustion_bool = _aawm_enrich._normalize_worker_context_exhaustion_bool
_sanitize_worker_context_exhaustion_metadata = _aawm_enrich._sanitize_worker_context_exhaustion_metadata
_promote_worker_context_exhaustion_metadata = _aawm_enrich._promote_worker_context_exhaustion_metadata
_infer_usage_breakout_provider_prefix = _aawm_enrich._infer_usage_breakout_provider_prefix
_enrich_usage_breakout_metadata = _aawm_enrich._enrich_usage_breakout_metadata
_enrich_claude_thinking_metadata = _aawm_enrich._enrich_claude_thinking_metadata
_read_varint = _aawm_enrich._read_varint
_extract_gemini_signature_summary = _aawm_enrich._extract_gemini_signature_summary
_enrich_gemini_thought_signature_metadata = _aawm_enrich._enrich_gemini_thought_signature_metadata
_enrich_agent_identity_metadata = _aawm_enrich._enrich_agent_identity_metadata
_enrich_trace_name_and_provider_metadata = _aawm_enrich._enrich_trace_name_and_provider_metadata
_get_reasoning_state_tags = _aawm_enrich._get_reasoning_state_tags
_extract_claude_experiment_ids = _aawm_enrich._extract_claude_experiment_ids
_extract_reasoning_content = _aawm_enrich._extract_reasoning_content
_extract_thinking_blocks = _aawm_enrich._extract_thinking_blocks
_normalize_base64_text = _aawm_enrich._normalize_base64_text
_decode_base64_bytes = _aawm_enrich._decode_base64_bytes
_short_hash = _aawm_enrich._short_hash
_WORKER_CONTEXT_EXHAUSTION_METADATA_KEYS = _aawm_enrich._WORKER_CONTEXT_EXHAUSTION_METADATA_KEYS
_WORKER_CONTEXT_EXHAUSTION_STRING_MAX_LEN = _aawm_enrich._WORKER_CONTEXT_EXHAUSTION_STRING_MAX_LEN
_WORKER_CONTEXT_EXHAUSTION_BOOL_KEYS = _aawm_enrich._WORKER_CONTEXT_EXHAUSTION_BOOL_KEYS
_GEMINI_MARKER = _aawm_enrich._GEMINI_MARKER
# --- end Wave A4B facades ---

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


# --- Wave A4A usage/provider-normalize/request-signals/prompt-overhead extraction.
# These MUST precede _bind_session_history_record_apis() so record-API free
# names and monkeypatch targets keep resolving through this namespace. ---
from . import usage_extract as _aawm_usage_extract
from . import provider_normalize as _aawm_provider_normalize
from . import request_signals as _aawm_request_signals
from . import prompt_overhead as _aawm_prompt_overhead

# rebind installers: helper __globals__ -> this namespace
_aawm_usage_extract.install(globals())
_aawm_provider_normalize.install(globals())
_aawm_request_signals.install(globals())
_aawm_prompt_overhead.install(globals())

# literal facade assignments (AST-visible; installers above already published
# the rebound function objects into this namespace, these re-affirm identity).
# --- Wave A4A usage_extract facades ---
_build_usage_object_from_metadata = _aawm_usage_extract._build_usage_object_from_metadata
_build_usage_object_from_token_count_payload = _aawm_usage_extract._build_usage_object_from_token_count_payload
_extract_responses_completed_response_from_langfuse_output = (
    _aawm_usage_extract._extract_responses_completed_response_from_langfuse_output
)
_build_usage_object_from_langfuse_output = _aawm_usage_extract._build_usage_object_from_langfuse_output
_extract_codex_model_from_response_headers = _aawm_usage_extract._extract_codex_model_from_response_headers
_session_history_metadata_model = _aawm_usage_extract._session_history_metadata_model
_SESSION_HISTORY_CLAUDE_MODEL_TAG_RE = _aawm_usage_extract._SESSION_HISTORY_CLAUDE_MODEL_TAG_RE
_session_history_model_from_request_tags = _aawm_usage_extract._session_history_model_from_request_tags
_extract_model_from_langfuse_input = _aawm_usage_extract._extract_model_from_langfuse_input
_extract_model_from_langfuse_output = _aawm_usage_extract._extract_model_from_langfuse_output
_first_known_model_string = _aawm_usage_extract._first_known_model_string
_first_explicit_openrouter_model_string = _aawm_usage_extract._first_explicit_openrouter_model_string
_coerce_usage_object_to_dict = _aawm_usage_extract._coerce_usage_object_to_dict
_extract_metadata_usage_object = _aawm_usage_extract._extract_metadata_usage_object
_merge_usage_object_with_metadata = _aawm_usage_extract._merge_usage_object_with_metadata
_extract_usage_object = _aawm_usage_extract._extract_usage_object
_enrich_token_count_usage_metadata = _aawm_usage_extract._enrich_token_count_usage_metadata
_extract_prompt_tokens = _aawm_usage_extract._extract_prompt_tokens
_extract_completion_tokens = _aawm_usage_extract._extract_completion_tokens
_extract_total_tokens = _aawm_usage_extract._extract_total_tokens
_extract_prompt_tokens_details = _aawm_usage_extract._extract_prompt_tokens_details
_extract_completion_tokens_details = _aawm_usage_extract._extract_completion_tokens_details
_extract_cache_read_input_tokens = _aawm_usage_extract._extract_cache_read_input_tokens
_extract_cache_creation_input_tokens = _aawm_usage_extract._extract_cache_creation_input_tokens
_has_nested_path = _aawm_usage_extract._has_nested_path
_extract_reported_reasoning_tokens = _aawm_usage_extract._extract_reported_reasoning_tokens
_fallback_gemini_reasoning_tokens_from_signatures = (
    _aawm_usage_extract._fallback_gemini_reasoning_tokens_from_signatures
)
_determine_reasoning_tokens_source = _aawm_usage_extract._determine_reasoning_tokens_source
_estimate_reasoning_tokens = _aawm_usage_extract._estimate_reasoning_tokens
_extract_rerank_request_payload = _aawm_usage_extract._extract_rerank_request_payload
_coerce_rerank_text = _aawm_usage_extract._coerce_rerank_text
_extract_rerank_document_text = _aawm_usage_extract._extract_rerank_document_text
# --- Wave A4A provider_normalize facades ---
_normalize_session_history_provider_name = _aawm_provider_normalize._normalize_session_history_provider_name
_session_history_provider_from_model_catalog = (
    _aawm_provider_normalize._session_history_provider_from_model_catalog
)
_session_history_provider_from_model = _aawm_provider_normalize._session_history_provider_from_model
_session_history_provider_from_route_family = (
    _aawm_provider_normalize._session_history_provider_from_route_family
)
_session_history_adapter_target_provider = _aawm_provider_normalize._session_history_adapter_target_provider
_session_history_auto_agent_selected_provider = (
    _aawm_provider_normalize._session_history_auto_agent_selected_provider
)
_session_history_adapter_model = _aawm_provider_normalize._session_history_adapter_model
_normalize_session_history_provider = _aawm_provider_normalize._normalize_session_history_provider
_sanitize_session_history_api_base = _aawm_provider_normalize._sanitize_session_history_api_base
_is_local_session_history_api_base = _aawm_provider_normalize._is_local_session_history_api_base
_extract_session_history_api_base = _aawm_provider_normalize._extract_session_history_api_base
_get_session_history_model_group = _aawm_provider_normalize._get_session_history_model_group
_resolve_inbound_model_alias = _aawm_provider_normalize._resolve_inbound_model_alias
_resolve_inbound_model_alias_from_langfuse = (
    _aawm_provider_normalize._resolve_inbound_model_alias_from_langfuse
)
_normalize_session_history_model_group = _aawm_provider_normalize._normalize_session_history_model_group
_is_completion_call_type = _aawm_provider_normalize._is_completion_call_type
_is_embedding_call_type = _aawm_provider_normalize._is_embedding_call_type
_strip_local_provider_model_prefix = _aawm_provider_normalize._strip_local_provider_model_prefix
_session_history_provider_from_api_base = _aawm_provider_normalize._session_history_provider_from_api_base
_apply_local_embedding_route_metadata = _aawm_provider_normalize._apply_local_embedding_route_metadata
_apply_local_llm_route_metadata = _aawm_provider_normalize._apply_local_llm_route_metadata
_LOCAL_BIOMED_SESSION_HISTORY_ROUTES = _aawm_provider_normalize._LOCAL_BIOMED_SESSION_HISTORY_ROUTES
_resolve_local_biomed_session_history_route = (
    _aawm_provider_normalize._resolve_local_biomed_session_history_route
)
_apply_local_biomed_route_metadata = _aawm_provider_normalize._apply_local_biomed_route_metadata
_resolve_session_history_model = _aawm_provider_normalize._resolve_session_history_model
_resolve_xai_grok_model_override = _aawm_provider_normalize._resolve_xai_grok_model_override
# --- Wave A4A request_signals facades ---
_INVALID_TOOL_CALL_ERROR_RE = _aawm_request_signals._INVALID_TOOL_CALL_ERROR_RE
_TOOL_RESULT_ERROR_BLOCK_TYPES = _aawm_request_signals._TOOL_RESULT_ERROR_BLOCK_TYPES
_invalid_tool_call_error_text_seen = _aawm_request_signals._invalid_tool_call_error_text_seen
_iter_tool_result_error_candidates = _aawm_request_signals._iter_tool_result_error_candidates
_iter_request_message_payloads = _aawm_request_signals._iter_request_message_payloads
_extract_invalid_tool_call_count_from_request_body = (
    _aawm_request_signals._extract_invalid_tool_call_count_from_request_body
)
_STRUCTURED_OUTPUT_JSON_MODE_VALUES = _aawm_request_signals._STRUCTURED_OUTPUT_JSON_MODE_VALUES
_STRUCTURED_OUTPUT_NESTED_REQUEST_KEYS = _aawm_request_signals._STRUCTURED_OUTPUT_NESTED_REQUEST_KEYS
_STRUCTURED_OUTPUT_FAILURE_PATTERNS = _aawm_request_signals._STRUCTURED_OUTPUT_FAILURE_PATTERNS
_empty_structured_output_state = _aawm_request_signals._empty_structured_output_state
_merge_structured_output_state = _aawm_request_signals._merge_structured_output_state
_structured_output_schema_hash = _aawm_request_signals._structured_output_schema_hash
_structured_output_state_from_format = _aawm_request_signals._structured_output_state_from_format
_structured_output_state_from_generation_config = (
    _aawm_request_signals._structured_output_state_from_generation_config
)
_detect_structured_output_request = _aawm_request_signals._detect_structured_output_request
_collect_structured_output_failure_texts = _aawm_request_signals._collect_structured_output_failure_texts
_classify_structured_output_failure = _aawm_request_signals._classify_structured_output_failure
_extract_request_body_from_langfuse_input = _aawm_request_signals._extract_request_body_from_langfuse_input
_request_payload_contains = _aawm_request_signals._request_payload_contains
_CODEX_THREAD_ID_RE = _aawm_request_signals._CODEX_THREAD_ID_RE
_GEMINI_COMPACT_PROMPT_ID_RE = _aawm_request_signals._GEMINI_COMPACT_PROMPT_ID_RE
_CLAUDE_CODE_COMPACT_REQUEST_MARKERS = _aawm_request_signals._CLAUDE_CODE_COMPACT_REQUEST_MARKERS
_append_request_content_text = _aawm_request_signals._append_request_content_text
_extract_request_user_texts = _aawm_request_signals._extract_request_user_texts
_join_compact_request_user_texts = _aawm_request_signals._join_compact_request_user_texts
_extract_codex_compact_thread_id = _aawm_request_signals._extract_codex_compact_thread_id
_extract_gemini_compact_prompt_id = _aawm_request_signals._extract_gemini_compact_prompt_id
_base_gemini_compact_prompt_id = _aawm_request_signals._base_gemini_compact_prompt_id
_extract_compact_output_text = _aawm_request_signals._extract_compact_output_text
_is_claude_code_compact_context = _aawm_request_signals._is_claude_code_compact_context
_is_codex_compact_context = _aawm_request_signals._is_codex_compact_context
_is_gemini_cli_compact_context = _aawm_request_signals._is_gemini_cli_compact_context
_classify_compact_summary_state = _aawm_request_signals._classify_compact_summary_state
# --- Wave A4A prompt_overhead facades ---
_fallback_text_token_estimate = _aawm_prompt_overhead._fallback_text_token_estimate
_empty_prompt_overhead_breakdown = _aawm_prompt_overhead._empty_prompt_overhead_breakdown
_serialize_prompt_overhead_component = _aawm_prompt_overhead._serialize_prompt_overhead_component
_estimate_prompt_overhead_tokens = _aawm_prompt_overhead._estimate_prompt_overhead_tokens
_extract_prompt_text_blocks = _aawm_prompt_overhead._extract_prompt_text_blocks
_classify_system_prompt_block = _aawm_prompt_overhead._classify_system_prompt_block
_estimate_system_prompt_bucket_tokens = _aawm_prompt_overhead._estimate_system_prompt_bucket_tokens
_append_prompt_component = _aawm_prompt_overhead._append_prompt_component
_RESPONSES_SYSTEM_ROLES = _aawm_prompt_overhead._RESPONSES_SYSTEM_ROLES
_RESPONSES_CONVERSATION_ROLES = _aawm_prompt_overhead._RESPONSES_CONVERSATION_ROLES
_RESPONSES_TEXT_CONTENT_TYPES = _aawm_prompt_overhead._RESPONSES_TEXT_CONTENT_TYPES
_RESPONSES_OPAQUE_CONTENT_TYPES = _aawm_prompt_overhead._RESPONSES_OPAQUE_CONTENT_TYPES
_RESPONSES_OPAQUE_ITEM_TYPES = _aawm_prompt_overhead._RESPONSES_OPAQUE_ITEM_TYPES
_append_prompt_text_components = _aawm_prompt_overhead._append_prompt_text_components
_extract_responses_visible_text_blocks = _aawm_prompt_overhead._extract_responses_visible_text_blocks
_responses_message_component_path = _aawm_prompt_overhead._responses_message_component_path
_record_responses_excluded_fields = _aawm_prompt_overhead._record_responses_excluded_fields
_append_openai_responses_input_component = _aawm_prompt_overhead._append_openai_responses_input_component
_append_openai_responses_input_components = (
    _aawm_prompt_overhead._append_openai_responses_input_components
)
_split_chat_prompt_messages = _aawm_prompt_overhead._split_chat_prompt_messages
_extract_prompt_overhead_components = _aawm_prompt_overhead._extract_prompt_overhead_components
_build_prompt_overhead_breakdown = _aawm_prompt_overhead._build_prompt_overhead_breakdown
_estimate_rerank_request_tokens = _aawm_prompt_overhead._estimate_rerank_request_tokens
_usage_has_positive_tokens = _aawm_prompt_overhead._usage_has_positive_tokens
_merge_estimated_rerank_tokens_into_usage = (
    _aawm_prompt_overhead._merge_estimated_rerank_tokens_into_usage
)
_positive_int_or_none = _aawm_prompt_overhead._positive_int_or_none
# --- end Wave A4A facades ---


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
