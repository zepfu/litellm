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
import warnings
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
    cast,
)
from urllib.parse import parse_qsl, quote, urlencode, urlsplit, urlunsplit

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
_AGENT_TENANT_RE = re.compile(
    r"You are '(?P<agent>[^']+)' and you are working on the '(?P<tenant>[^']+)' project"
)
_AGENT_ROLE_RE = re.compile(
    r"^[ \t]*You are a '(?P<agent>explorer|worker|default)' agent\.[ \t]*$",
    re.MULTILINE,
)
_DEFAULT_AGENT = "orchestrator"
_CLAUDE_EXPERIMENT_ID_RE = re.compile(
    rb"(?<![A-Za-z0-9._-])([A-Za-z][A-Za-z0-9._-]{11,})(?![A-Za-z0-9._-])"
)
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
_AAWM_LITELLM_ENVIRONMENT_ENV_VARS = (
    "AAWM_LITELLM_ENVIRONMENT",
    "LITELLM_INSTANCE_ENVIRONMENT",
    "LITELLM_ENVIRONMENT",
    "LITELLM_ENV",
    "LITELLM_LANGFUSE_TRACE_ENVIRONMENT",
    "LANGFUSE_TRACING_ENVIRONMENT",
    "AAWM_ENVIRONMENT",
)
_AAWM_LITELLM_VERSION_ENV_VARS = (
    "AAWM_LITELLM_VERSION",
    "LITELLM_VERSION",
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
_AAWM_ASSOCIATED_VERSION_ENV_VARS = {
    "aawm-litellm-callbacks": (
        "AAWM_CALLBACK_WHEEL_VERSION",
        "AAWM_LITELLM_CALLBACKS_VERSION",
    ),
    "aawm-litellm-control-plane": (
        "AAWM_CONTROL_PLANE_WHEEL_VERSION",
        "AAWM_LITELLM_CONTROL_PLANE_VERSION",
    ),
    "litellm-model-config": (
        "AAWM_MODEL_CONFIG_VERSION",
        "LITELLM_MODEL_CONFIG_VERSION",
    ),
    "litellm-local-ci-harness": (
        "AAWM_HARNESS_VERSION",
        "LITELLM_LOCAL_CI_HARNESS_VERSION",
    ),
}
_USER_AGENT_PRODUCT_RE = re.compile(
    r"(?P<name>[A-Za-z][A-Za-z0-9._-]{1,63})/"
    r"(?P<version>[A-Za-z0-9][A-Za-z0-9.+_-]{0,127})"
)
_USER_AGENT_PAREN_PRODUCT_RE = re.compile(
    r"\((?P<name>[A-Za-z][A-Za-z0-9._-]{1,63})\s*;\s*"
    r"(?P<version>[A-Za-z0-9][A-Za-z0-9.+_-]{0,127})\)"
)
_RESET_AFTER_SECONDS_RE = re.compile(
    r"\breset(?:s|ting)?\s+after\s+(?P<seconds>\d+)s\b",
    re.IGNORECASE,
)
_AAWM_AGENT_ID_UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    re.IGNORECASE,
)
_AAWM_AGENT_ID_HEX_RE = re.compile(r"^(?=.*[a-f])[a-f0-9]{12,64}$", re.IGNORECASE)
_AAWM_AGENT_ID_PREFIXED_RE = re.compile(
    r"^(?:agent|subagent|task)[-_][A-Za-z0-9][A-Za-z0-9._:-]{5,127}$",
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

_AAWM_SESSION_HISTORY_METADATA_KEYS = (
    "agent_id",
    "agent_id_source",
    "tenant_id_source",
    "repository_source",
    "workload_type",
    "workload_subtype",
    "source_repository",
    "memory_workload_label",
    "trace_name",
    "trace_user_id",
    "trace_environment",
    "session_id_source",
    "synthetic_session_id",
    "synthetic_session_id_basis",
    "source_status",
    "source_model",
    "logical_model",
    "aawm_claude_agent_name",
    "aawm_claude_project",
    "aawm_tenant_id",
    "cc_version",
    "cc_entrypoint",
    "litellm_environment",
    "litellm_version",
    "litellm_fork_version",
    "litellm_wheel_versions",
    "client_name",
    "client_version",
    "client_user_agent",
    "client_ip",
    "client_ip_source",
    "host_name",
    "host_name_source",
    "repository",
    "route_tag",
    "openai_passthrough_route_family",
    "passthrough_route_family",
    "route_family",
    "auth_mode",
    "credential_family",
    "xai_oauth_managed",
    "xai_oauth_public_model",
    "xai_oauth_upstream_model",
    "xai_quota_family",
    "shared_quota_family",
    "grok_subscription_quota_shared",
    "xai_responses_request_sanitized",
    "xai_responses_sanitized_removed_params",
    "xai_responses_sanitized_tool_count",
    "xai_responses_sanitized_tool_types",
    "xai_tool_choice_without_tools_removed",
    "xai_tool_choice_without_tools_removed_reason",
    "claude_internal_check",
    "claude_internal_check_type",
    "claude_permission_check",
    "claude_permission_check_decision",
    "claude_permission_check_blocked",
    "claude_permission_check_request_model",
    "claude_permission_check_response_model",
    "reasoning_content_present",
    "thinking_signature_present",
    "usage_reasoning_tokens_reported",
    "usage_reasoning_tokens_source",
    "usage_token_count_response",
    "aawm_rate_limit_observation_only",
    "session_history_usage_record",
    "session_history_zero_token_class",
    "d1_140_zero_token_class",
    "d1_140_zero_token_reason",
    "gemini_control_plane_excluded",
    "gemini_control_plane_method",
    "reasoning_effort_requested",
    "reasoning_effort_source",
    "reasoning_effort_native_provider",
    "reasoning_effort_native_value",
    "reasoning_effort_native_field",
    "reasoning_effort_supported_ceiling",
    "reasoning_effort_resolved_model",
    "reasoning_effort_resolved_provider",
    "reasoning_effort_candidate_attempt",
    "reasoning_effort_mapping_reason",
    "reasoning_effort_clamped_from",
    "reasoning_effort_clamp_reason",
    "codex_reasoning_effort",
    "openai_reasoning_effort",
    "gemini_reasoning_effort",
    "openrouter_reasoning_effort",
    "nvidia_reasoning_effort",
    "usage_cache_read_input_tokens",
    "usage_cache_creation_input_tokens",
    "usage_provider_cache_attempted",
    "usage_provider_cache_status",
    "usage_provider_cache_miss",
    "usage_provider_cache_miss_reason",
    "usage_provider_cache_miss_token_count",
    "usage_provider_cache_miss_cost_usd",
    "usage_provider_cache_miss_cost_basis",
    "usage_provider_cache_source",
    "openai_prompt_cache_key_present",
    "anthropic_adapter_cache_control_present",
    "anthropic_adapter_unsupported_hosted_tools",
    "anthropic_adapter_unsupported_hosted_tool_choice",
    "anthropic_adapter_model",
    "anthropic_adapter_original_model",
    "anthropic_adapter_target_endpoint",
    "codex_unsupported_hosted_tool_removed_count",
    "codex_unsupported_hosted_tool_types_removed",
    "codex_unsupported_hosted_tools_removed",
    "codex_unsupported_hosted_tool_choice_removed",
    "aawm_tool_definition_capture_version",
    "aawm_tool_definition_capture_source",
    "aawm_tool_definition_count",
    "aawm_tool_definition_captured_count",
    "aawm_tool_definition_sources",
    "aawm_tool_definition_names",
    "aawm_tool_definition_types",
    "aawm_tool_definition_snapshot_hash",
    "aawm_tool_definition_snapshot_truncated",
    "aawm_tool_definition_snapshot_storage",
    "aawm_tool_definition_snapshot_storage_key",
    "opencode_zen_removed_unsupported_tool_count",
    "opencode_zen_removed_unsupported_tool_types",
    "opencode_zen_removed_unsupported_tool_names",
    "codex_adapter_model",
    "codex_adapter_original_model",
    "codex_adapter_target_endpoint",
    "codex_adapter_input_shape",
    "codex_adapter_output_shape",
    "model_alias_label",
    "requested_model_alias",
    "codex_auto_agent_alias",
    "codex_auto_agent_selected_provider",
    "codex_auto_agent_selected_model",
    "codex_auto_agent_selected_route_family",
    "codex_auto_agent_selected_last_resort",
    "codex_auto_agent_selection_reason",
    "codex_auto_agent_lane_key",
    "codex_auto_agent_attempts",
    "codex_auto_agent_skipped_candidates",
    "codex_auto_agent_audit_events",
    "anthropic_auto_agent_alias",
    "anthropic_auto_agent_selected_provider",
    "anthropic_auto_agent_selected_model",
    "anthropic_auto_agent_selected_route_family",
    "anthropic_auto_agent_selected_last_resort",
    "anthropic_auto_agent_selection_reason",
    "anthropic_auto_agent_lane_key",
    "anthropic_auto_agent_attempts",
    "anthropic_auto_agent_skipped_candidates",
    "anthropic_auto_agent_audit_events",
    "aawm_alias_routing_audit_events",
    "codex_google_code_assist_dropped_response_tool_types",
    "google_retrieve_user_quota",
    "usage_tool_call_count",
    "usage_invalid_tool_call_count",
    "usage_structured_output_attempted",
    "usage_structured_output_failed",
    "usage_structured_output_mode",
    "usage_structured_output_schema_hash",
    "usage_structured_output_failure_reason",
    "usage_trace_quality_score",
    "usage_empty_completion_failure",
    "usage_large_tool_result_payload_risk",
    "usage_destructive_checkout_after_work",
    "usage_invalid_tool_call_error",
    "usage_read_only_policy_compliance_score",
    "usage_read_only_policy_violation_count",
    "usage_response_meaningfulness_score",
    "usage_instruction_adherence_score",
    "usage_answer_completeness_score",
    "usage_evidence_fidelity_score",
    "usage_tool_result_fidelity_score",
    "usage_error_attribution_quality_score",
    "usage_repetition_loop_risk_score",
    "usage_context_retention_score",
    "usage_tool_use_validity_score",
    "usage_tool_error_recovery_score",
    "usage_stall_risk_score",
    "usage_output_contract_compliance_score",
    "usage_task_progress_score",
    "usage_scope_control_score",
    "usage_destructive_action_policy_score",
    "usage_ignored_path_tracking_policy_score",
    "usage_ignored_path_tracking_violation_count",
    "usage_baseline_deflection_attempted_score",
    "usage_baseline_deflection_incident_score",
    "usage_baseline_deflection_attempt_count",
    "usage_baseline_deflection_tool_call_count",
    "usage_baseline_deflection_input_tokens",
    "usage_baseline_deflection_elapsed_ms",
    "usage_quality_gate_trigger_count",
    "usage_quality_gate_fix_attempt_count",
    "usage_quality_gate_rerun_count",
    "usage_sleep_wellness_interruption_attempted_score",
    "usage_sleep_wellness_interruption_incident_score",
    "usage_sleep_wellness_interruption_count",
    "usage_sleep_wellness_interruption_output_tokens",
    "usage_sleep_wellness_interruption_input_tokens",
    "usage_sleep_wellness_interruption_elapsed_ms",
    "usage_sleep_wellness_interruption_after_user_pushback_count",
    "usage_sleep_wellness_interruption_repeated_count",
    "usage_terminal_completion_score",
    "usage_discovery_inventory_coverage_score",
    "usage_discovery_inventory_missing_count",
    "usage_output_contract_required_final_phrase",
    "usage_output_contract_required_final_phrase_present",
    "usage_output_contract_required_final_phrase_source",
    "usage_output_contract_failure_class",
    "usage_output_contract_failure_count",
    "usage_output_contract_setup_only_detected",
    "usage_output_contract_setup_only_markers",
    "usage_output_contract_final_text_chars",
    "usage_agent_score_reasons",
    "usage_agent_score_source",
    "gemini_user_prompt_id",
    "is_compact_summary",
    "compact_summary_source",
    "compact_summary_role",
    "compact_summary_id",
    "usage_tool_names",
    "google_adapter_system_prompt_policy_name",
    "google_adapter_system_prompt_policy",
    "google_adapter_system_prompt_policy_version",
    "google_adapter_system_prompt_original_chars",
    "google_adapter_system_prompt_rewritten_chars",
    "google_adapter_system_prompt_removed_claude_overhead_chars",
    "google_adapter_system_prompt_preserved_instruction_chars",
    "google_adapter_system_prompt_policy_applied",
    "codex_google_code_assist_tool_contract_policy_name",
    "codex_google_code_assist_tool_contract_policy",
    "codex_google_code_assist_tool_contract_policy_version",
    "codex_google_code_assist_tool_contract_policy_applied",
    "anthropic_context_window_mode",
    "anthropic_context_window_requested_tokens",
    "anthropic_context_window_source",
    "anthropic_context_window_beta",
    "anthropic_context_window_classification",
    "codex_google_code_assist_tool_contract_prompt_chars",
    "usage_search_units",
    "usage_openrouter_cost",
    "openrouter_provider",
    "openrouter_response_model",
    "aawm_local_prepare_ms",
    "aawm_upstream_wait_ms",
    "aawm_time_to_first_token_ms",
    "aawm_upstream_first_chunk_ms",
    "aawm_first_emitted_chunk_ms",
    "aawm_stream_emit_gap_ms",
    "aawm_upstream_stream_complete_ms",
    "aawm_local_stream_finalize_ms",
    "aawm_local_finalize_ms",
    "aawm_total_proxy_overhead_ms",
    "aawm_total_proxy_duration_ms",
    "aawm_stream_chunk_count",
    "aawm_stream_total_bytes",
    "aawm_passthrough_endpoint_type",
    "responses_stream_event_types",
    "responses_stream_event_counts",
    "responses_stream_tool_call_count",
    "responses_stream_tool_names",
    "aawm_stream_logging_endpoint_type",
    "aawm_stream_logging_custom_llm_provider",
    "aawm_stream_logging_is_openai_responses",
    "aawm_local_route",
    "aawm_local_route_family",
    "aawm_local_model_group",
    "aawm_local_service",
    "aawm_local_endpoint",
    "aawm_local_upstream_provider",
    "aawm_local_upstream_model",
    "aawm_local_upstream_api_base",
    "aawm_local_upstream_url",
    "usage_input_system_tokens_estimated",
    "usage_input_tool_advertisement_tokens_estimated",
    "usage_input_conversation_tokens_estimated",
    "usage_input_other_tokens_estimated",
    "usage_input_breakdown_residual_tokens",
    "usage_input_opaque_state_tokens_estimated",
    "usage_system_behavior_tokens_estimated",
    "usage_system_safety_tokens_estimated",
    "usage_system_instructional_tokens_estimated",
    "usage_system_unclassified_tokens_estimated",
    "prompt_overhead_breakdown_source",
    "prompt_overhead_counted_shape",
    "prompt_overhead_route_family",
    "prompt_overhead_tokenizer",
    "prompt_overhead_classifier_version",
    "prompt_overhead_component_paths",
    "prompt_overhead_excluded_component_paths",
    "worker_context_exhaustion_failure_class",
    "worker_context_exhaustion_failure_reason",
    "worker_context_exhaustion_partial_output_summary",
    "worker_context_exhaustion_changed_paths_hint",
    "worker_context_exhaustion_attempted_patch_scope",
    "worker_context_exhaustion_last_visible_message",
    "worker_context_exhaustion_success",
    "worker_context_exhaustion_completed",
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
_SESSION_HISTORY_OUTPUT_CONTRACT_JSON_FIELDS = (
    "output_contract_setup_only_markers",
)
_PROMPT_OVERHEAD_CLASSIFIER_VERSION = "deterministic-v2"
_AAWM_REQUEST_PAYLOAD_SCAN_MAX_DEPTH = 16
_AAWM_REQUEST_PAYLOAD_SCAN_MAX_ITEMS = 5000
_AAWM_JSON_SAFE_MAX_DEPTH = 12
_AAWM_TENANT_ID_METADATA_KEYS = (
    "tenant_id",
    "aawm_tenant_id",
    "user_api_key_org_id",
    "organization_id",
    "org_id",
    "litellm_organization_id",
    "litellm_org_id",
    "user_api_key_team_id",
    "team_id",
    "litellm_team_id",
)
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
_AAWM_REPOSITORY_METADATA_KEYS = (
    "repository",
    "aawm_repository",
    "repo",
    "repo_name",
    "repository_name",
    "git_repository",
    "vcs_repository",
    "workspace_root",
    "workspaceRoot",
    "project_root",
    "projectRoot",
    "root_path",
    "rootPath",
    "working_directory",
    "workingDirectory",
    "cwd_path",
    "cwdPath",
    "cwd_uri",
    "cwdUri",
    "aawm_claude_project",
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
            get_secret_str(_AAWM_WORKSPACE_ROOT_ENV)
            or os.environ.get(_AAWM_WORKSPACE_ROOT_ENV)
        )
    except Exception:
        configured = None
    return configured or _AAWM_WORKSPACE_ROOT_DEFAULT


def _get_codex_memory_root_path() -> str:
    try:
        configured = _normalize_configured_root_path(
            get_secret_str(_AAWM_CODEX_MEMORY_ROOT_ENV)
            or os.environ.get(_AAWM_CODEX_MEMORY_ROOT_ENV)
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
_AAWM_REPOSITORY_TEXT_PATTERNS = _build_aawm_repository_text_patterns(
    _AAWM_WORKSPACE_ROOT_DEFAULT
)
_AAWM_REPOSITORY_TEXT_MARKERS = _aawm_repository_text_markers(
    _AAWM_WORKSPACE_ROOT_DEFAULT
)
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
_AAWM_REPOSITORY_ID_PATTERN = re.compile(
    r"^[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)?$"
)
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


def _clean_secret_string(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None

    cleaned = value.strip()
    if len(cleaned) >= 2 and cleaned[0] == cleaned[-1] and cleaned[0] in {'"', "'"}:
        cleaned = cleaned[1:-1].strip()
    return cleaned or None


def _get_first_secret_value(secret_names: tuple[str, ...]) -> Optional[str]:
    for secret_name in secret_names:
        value = _clean_secret_string(get_secret_str(secret_name))
        if value:
            return value
    return None


def _normalize_aawm_sslmode(value: Optional[str]) -> Optional[str]:
    cleaned = _clean_secret_string(value)
    if not cleaned:
        return None

    lowered = cleaned.lower()
    if lowered in {"1", "true", "yes", "on"}:
        return "require"
    if lowered in {"0", "false", "no", "off"}:
        return "disable"
    return cleaned


def _build_aawm_dsn() -> Optional[str]:
    host = _get_first_secret_value(_AAWM_DB_HOST_ENV_VARS)
    port = _get_first_secret_value(_AAWM_DB_PORT_ENV_VARS)
    user = _get_first_secret_value(_AAWM_DB_USER_ENV_VARS)
    password = _get_first_secret_value(_AAWM_DB_PASSWORD_ENV_VARS)
    database = _get_first_secret_value(_AAWM_DB_NAME_ENV_VARS)
    sslmode = _normalize_aawm_sslmode(
        _get_first_secret_value(_AAWM_DB_SSLMODE_ENV_VARS)
        or _get_first_secret_value(_AAWM_DB_SSL_BOOL_ENV_VARS)
    )

    has_component_config = any((host, port, user, password, database, sslmode))
    if has_component_config:
        if not host or not user or not database:
            return None

        credentials = quote(user, safe="")
        if password:
            credentials += f":{quote(password, safe='')}"
        dsn = (
            f"postgresql://{credentials}@{host}:{port or '5432'}/"
            f"{quote(database, safe='')}"
        )
        if sslmode:
            dsn += f"?{urlencode({'sslmode': sslmode})}"
        return dsn

    return _get_first_secret_value(_AAWM_DB_URL_ENV_VARS)


def _append_aawm_dsn_query_params(
    dsn: str,
    params: Dict[str, Optional[str]],
) -> str:
    parsed = urlsplit(dsn)
    if not parsed.scheme:
        return dsn

    query_items = parse_qsl(parsed.query, keep_blank_values=True)
    existing_keys = {key for key, _value in query_items}
    for key, value in params.items():
        cleaned_value = _clean_secret_string(value)
        if cleaned_value and key not in existing_keys:
            query_items.append((key, cleaned_value))
            existing_keys.add(key)
    return urlunsplit(
        (
            parsed.scheme,
            parsed.netloc,
            parsed.path,
            urlencode(query_items),
            parsed.fragment,
        )
    )










def _clean_non_empty_string(value: Any) -> Optional[str]:
    if value is None:
        return None
    cleaned = str(value).strip()
    return cleaned or None


def _first_non_empty_string(*values: Any) -> Optional[str]:
    for value in values:
        cleaned = _clean_non_empty_string(value)
        if cleaned:
            return cleaned
    return None


def _coerce_string_dict(value: Any) -> Dict[str, str]:
    parsed_value = _safe_json_load(value, value)
    if not isinstance(parsed_value, dict):
        return {}

    result: Dict[str, str] = {}
    for key, nested_value in list(parsed_value.items()):
        key_text = _clean_non_empty_string(key)
        value_text = _clean_non_empty_string(nested_value)
        if key_text and value_text:
            result[key_text] = value_text
    return result


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


def _extract_claude_trace_agent_name(value: Any) -> Optional[str]:
    trace_name = _clean_non_empty_string(value)
    if not trace_name or not trace_name.startswith("claude-code."):
        return None
    agent_name = _clean_non_empty_string(trace_name.split(".", 1)[1])
    return agent_name


def _extract_claude_trace_user_identity_from_metadata_sources(
    *sources: Tuple[str, Any],
) -> Tuple[Optional[str], Optional[str]]:
    for source_name, raw_source in sources:
        source = _coerce_mapping(raw_source)
        if not source:
            continue

        trace_user_id = _normalize_repository_identity(source.get("trace_user_id"))
        if trace_user_id and _clean_non_empty_string(
            source.get("trace_name")
        ) and str(source.get("trace_name")).startswith("claude-code"):
            return trace_user_id, f"{source_name}.trace_user_id"

        nested_source = _coerce_mapping(source.get("metadata"))
        if not nested_source:
            continue
        trace_user_id = _normalize_repository_identity(
            nested_source.get("trace_user_id")
        )
        if trace_user_id and _clean_non_empty_string(
            nested_source.get("trace_name")
        ) and str(nested_source.get("trace_name")).startswith("claude-code"):
            return trace_user_id, f"{source_name}.metadata.trace_user_id"

    return None, None


def _extract_tenant_identity_from_kwargs(
    kwargs: Dict[str, Any],
    *,
    metadata: Optional[Dict[str, Any]] = None,
    standard_logging_object: Optional[Dict[str, Any]] = None,
) -> Tuple[Optional[str], Optional[str]]:
    litellm_params = kwargs.get("litellm_params") or {}
    standard_logging_object = standard_logging_object or kwargs.get("standard_logging_object") or {}
    passthrough_payload = kwargs.get("passthrough_logging_payload") or {}
    proxy_request = _coerce_mapping(litellm_params.get("proxy_server_request"))
    proxy_body = _coerce_mapping(proxy_request.get("body"))
    passthrough_body = _coerce_mapping(passthrough_payload.get("request_body"))

    tenant_id, source = _extract_tenant_identity_from_metadata_sources(
        ("litellm_params.metadata", metadata or litellm_params.get("metadata")),
        ("standard_logging_object.metadata", standard_logging_object.get("metadata")),
        ("kwargs.metadata", kwargs.get("metadata")),
        ("litellm_params.proxy_server_request.body", proxy_body),
        ("litellm_params.proxy_server_request.body.metadata", proxy_body.get("metadata")),
        ("passthrough_logging_payload", passthrough_payload),
        ("passthrough_logging_payload.request_body", passthrough_body),
        ("passthrough_logging_payload.request_body.metadata", passthrough_body.get("metadata")),
        ("standard_logging_object", standard_logging_object),
        ("kwargs", kwargs),
    )
    metadata_mapping = _coerce_mapping(metadata or litellm_params.get("metadata"))
    if tenant_id and _is_codex_passthrough_tenant_extraction_context(
        kwargs,
        metadata=metadata_mapping,
    ):
        trace_user_id = _normalize_repository_identity(
            metadata_mapping.get("trace_user_id")
        )
        tenant_source = _clean_non_empty_string(metadata_mapping.get("tenant_id_source"))
        if _is_codex_trace_user_tenant_source(source) or _is_codex_trace_user_tenant_source(
            tenant_source
        ):
            tenant_id, source = None, None
        elif isinstance(source, str) and source.endswith(".trace_user_id"):
            tenant_id, source = None, None
        elif trace_user_id and tenant_id == trace_user_id and not _is_repository_source_trusted_for_codex_tenant(
            metadata_mapping.get("repository_source")
        ):
            tenant_id, source = None, None
        elif isinstance(source, str) and any(
            source.endswith(marker)
            for marker in (".tenant_id", ".aawm_tenant_id")
        ) and trace_user_id and tenant_id == trace_user_id:
            tenant_id, source = None, None
    if tenant_id:
        return tenant_id, source

    headers = _extract_request_headers_from_kwargs(kwargs)
    tenant_id = _normalize_tenant_identity(
        _get_header_value(headers, *_AAWM_TENANT_ID_HEADER_NAMES)
    )
    if tenant_id:
        return tenant_id, "request_headers"

    tenant_id, source = _extract_claude_trace_user_identity_from_metadata_sources(
        ("litellm_params.metadata", metadata or litellm_params.get("metadata")),
        ("standard_logging_object.metadata", standard_logging_object.get("metadata")),
        ("kwargs.metadata", kwargs.get("metadata")),
        ("litellm_params.proxy_server_request.body", proxy_body),
        ("litellm_params.proxy_server_request.body.metadata", proxy_body.get("metadata")),
        ("passthrough_logging_payload", passthrough_payload),
        ("passthrough_logging_payload.request_body", passthrough_body),
        ("passthrough_logging_payload.request_body.metadata", passthrough_body.get("metadata")),
        ("standard_logging_object", standard_logging_object),
        ("kwargs", kwargs),
    )
    if tenant_id:
        return tenant_id, source

    return None, None


def _extract_tenant_identity_from_langfuse_trace_observation(
    trace: Dict[str, Any],
    observation: Dict[str, Any],
    metadata: Optional[Dict[str, Any]] = None,
) -> Tuple[Optional[str], Optional[str]]:
    trace_metadata = trace.get("metadata") if isinstance(trace, dict) else None
    tenant_id, source = _extract_tenant_identity_from_metadata_sources(
        ("observation.metadata", metadata or observation.get("metadata")),
        ("trace.metadata", trace_metadata),
        ("observation", observation),
        ("trace", trace),
    )
    if tenant_id:
        return tenant_id, source
    trace_user_id = _normalize_tenant_identity(
        trace.get("userId") if isinstance(trace, dict) else None
    )
    if trace_user_id:
        return trace_user_id, "trace.userId"
    return None, None


def _is_agent_id_like(value: str) -> bool:
    normalized = value.strip()
    if not normalized:
        return False
    return bool(
        _AAWM_AGENT_ID_UUID_RE.fullmatch(normalized)
        or _AAWM_AGENT_ID_HEX_RE.fullmatch(normalized)
        or _AAWM_AGENT_ID_PREFIXED_RE.fullmatch(normalized)
    )


def _normalize_agent_id_identity(
    value: Any,
    *,
    disallowed_values: Optional[Set[str]] = None,
) -> Optional[str]:
    cleaned = _clean_non_empty_string(value)
    if not cleaned:
        return None
    cleaned = cleaned.strip("`'\"")
    normalized = cleaned.lower()
    if normalized in {"none", "null", "unknown", "orchestrator"}:
        return None
    if disallowed_values and normalized in disallowed_values:
        return None
    if not _is_agent_id_like(cleaned):
        return None
    return cleaned


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


def _extract_agent_id_from_metadata_sources(
    *sources: Tuple[str, Any],
    disallowed_values: Optional[Set[str]] = None,
) -> Tuple[Optional[str], Optional[str]]:
    for source_name, raw_source in sources:
        source = _coerce_mapping(raw_source)
        if not source:
            continue
        for key in _AAWM_AGENT_ID_METADATA_KEYS:
            agent_id = _normalize_agent_id_identity(
                source.get(key),
                disallowed_values=disallowed_values,
            )
            if agent_id:
                return agent_id, f"{source_name}.{key}"

        for nested_key in (
            "metadata",
            "litellm_metadata",
            "request_metadata",
            "user_api_key_metadata",
        ):
            nested_source = _coerce_mapping(source.get(nested_key))
            if not nested_source:
                continue
            for key in _AAWM_AGENT_ID_METADATA_KEYS:
                agent_id = _normalize_agent_id_identity(
                    nested_source.get(key),
                    disallowed_values=disallowed_values,
                )
                if agent_id:
                    return agent_id, f"{source_name}.{nested_key}.{key}"

    return None, None


def _extract_agent_id_from_kwargs(
    kwargs: Dict[str, Any],
    *,
    metadata: Optional[Dict[str, Any]] = None,
    standard_logging_object: Optional[Dict[str, Any]] = None,
    agent_name: Optional[str] = None,
    tenant_id: Optional[str] = None,
    repository: Optional[str] = None,
) -> Tuple[Optional[str], Optional[str]]:
    litellm_params = kwargs.get("litellm_params") or {}
    standard_logging_object = standard_logging_object or kwargs.get("standard_logging_object") or {}
    passthrough_payload = kwargs.get("passthrough_logging_payload") or {}
    proxy_request = _coerce_mapping(litellm_params.get("proxy_server_request"))
    proxy_body = _coerce_mapping(proxy_request.get("body"))
    passthrough_body = _coerce_mapping(passthrough_payload.get("request_body"))
    disallowed_values = _agent_id_disallowed_values_from_kwargs(
        kwargs,
        metadata=metadata,
        standard_logging_object=standard_logging_object,
        agent_name=agent_name,
        tenant_id=tenant_id,
        repository=repository,
    )

    agent_id, source = _extract_agent_id_from_metadata_sources(
        ("litellm_params.metadata", metadata or litellm_params.get("metadata")),
        ("standard_logging_object.metadata", standard_logging_object.get("metadata")),
        ("kwargs.metadata", kwargs.get("metadata")),
        ("litellm_params.proxy_server_request.body", proxy_body),
        ("litellm_params.proxy_server_request.body.metadata", proxy_body.get("metadata")),
        ("litellm_params.proxy_server_request.body.litellm_metadata", proxy_body.get("litellm_metadata")),
        ("passthrough_logging_payload", passthrough_payload),
        ("passthrough_logging_payload.request_body", passthrough_body),
        ("passthrough_logging_payload.request_body.metadata", passthrough_body.get("metadata")),
        ("passthrough_logging_payload.request_body.litellm_metadata", passthrough_body.get("litellm_metadata")),
        disallowed_values=disallowed_values,
    )
    if agent_id:
        return agent_id, source

    headers = _extract_request_headers_from_kwargs(kwargs)
    for header_name in _AAWM_AGENT_ID_HEADER_NAMES:
        agent_id = _normalize_agent_id_identity(
            _get_header_value(headers, header_name),
            disallowed_values=disallowed_values,
        )
        if agent_id:
            return agent_id, f"request_headers.{header_name}"

    return None, None


def _extract_agent_id_from_langfuse_trace_observation(
    trace: Dict[str, Any],
    observation: Dict[str, Any],
    metadata: Optional[Dict[str, Any]] = None,
    *,
    agent_name: Optional[str] = None,
    tenant_id: Optional[str] = None,
    repository: Optional[str] = None,
) -> Tuple[Optional[str], Optional[str]]:
    trace_metadata = trace.get("metadata") if isinstance(trace, dict) else None
    disallowed_values = _agent_id_disallowed_values(
        agent_name,
        tenant_id,
        repository,
        trace.get("sessionId") if isinstance(trace, dict) else None,
        trace.get("session_id") if isinstance(trace, dict) else None,
        trace.get("id") if isinstance(trace, dict) else None,
        observation.get("traceId") if isinstance(observation, dict) else None,
        observation.get("id") if isinstance(observation, dict) else None,
        metadata.get("session_id") if isinstance(metadata, dict) else None,
        metadata.get("trace_id") if isinstance(metadata, dict) else None,
        metadata.get("trace_user_id") if isinstance(metadata, dict) else None,
        metadata.get("repository") if isinstance(metadata, dict) else None,
        metadata.get("tenant_id") if isinstance(metadata, dict) else None,
        metadata.get("agent_name") if isinstance(metadata, dict) else None,
    )
    return _extract_agent_id_from_metadata_sources(
        ("observation.metadata", metadata or observation.get("metadata")),
        ("trace.metadata", trace_metadata),
        disallowed_values=disallowed_values,
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
        env_val = get_secret_str("AAWM_KNOWN_WORKSPACE_REPOS") or os.environ.get(
            "AAWM_KNOWN_WORKSPACE_REPOS", ""
        )
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


def _normalize_repository_identity_from_absolute_path(
    normalized_path: str,
) -> Optional[str]:
    codex_memory_root = _get_codex_memory_root_path()
    workspace_prefix = _aawm_workspace_root_prefix()
    if normalized_path == codex_memory_root:
        return _CODEX_MEMORY_ROOT_REPOSITORY

    path_parts = normalized_path.rsplit("/", 1)
    basename = path_parts[-1]
    if basename.lower() in _AAWM_REPO_INSTRUCTION_FILENAMES and len(path_parts) > 1:
        parent_path = path_parts[0].rstrip("/")
        if parent_path == codex_memory_root:
            return _CODEX_MEMORY_ROOT_REPOSITORY
        if parent_path.startswith(workspace_prefix):
            return parent_path.rsplit("/", 1)[-1]
        return None

    # Trusted workspace roots map to repos; nested prompt-text file paths under
    # a project are references, not session ownership.
    if normalized_path.startswith(workspace_prefix):
        sub = normalized_path[len(workspace_prefix) :].strip("/")
        if not sub:
            return None
        if "/" not in sub:
            return sub
        first, sep, rest = sub.partition("/")
        if sep and rest and rest.lower() in _AAWM_REPO_INSTRUCTION_FILENAMES:
            return first
        return None

    return basename


def _normalize_repository_identity(value: Any) -> Optional[str]:
    if not isinstance(value, str):
        return None

    cleaned = _clean_non_empty_string(value)
    if not cleaned:
        return None
    cleaned = cleaned.strip("`'\"")

    if "..." in cleaned:
        return None

    if cleaned.startswith("git@") and ":" in cleaned:
        cleaned = cleaned.split(":", 1)[1]
    elif "://" in cleaned:
        try:
            parsed = urlsplit(cleaned)
            netloc = parsed.netloc.split("@", 1)[-1]
            path = parsed.path.strip("/")
            if parsed.scheme == "file" and path:
                cleaned = path.rstrip("/").rsplit("/", 1)[-1]
            elif netloc.lower().endswith("github.com") and path:
                cleaned = path
            else:
                cleaned = urlunsplit(("", netloc, path, "", "")).strip("/")
        except Exception:
            pass
    elif cleaned.startswith("/"):
        cleaned = _normalize_repository_identity_from_absolute_path(
            cleaned.rstrip("/")
        )
        if cleaned is None:
            return None

    if cleaned.lower() in _AAWM_REPO_INSTRUCTION_FILENAMES:
        return None

    if cleaned.endswith(".git"):
        cleaned = cleaned[:-4]
    cleaned = cleaned.strip().strip("/")
    if _is_bare_file_basename_with_reject_extension(cleaned):
        return None
    if _is_bare_dot_directory(cleaned):
        return None
    if (
        not cleaned
        or not _is_valid_repository_identity(cleaned)
        or _is_disallowed_repository_identity(cleaned)
    ):
        return None
    return cleaned


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


def _extract_repository_identity_from_text(value: str) -> Optional[str]:
    repository, _source = _extract_repository_identity_from_text_with_source(value)
    return repository


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


def _extract_repository_identity_from_value(
    value: Any,
    *,
    _seen: Optional[set[int]] = None,
    _depth: int = 0,
) -> Optional[str]:
    repository, _source = _extract_repository_identity_from_value_with_source(
        value,
        source_prefix="value",
        _seen=_seen,
        _depth=_depth,
    )
    return repository


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
                    _clean_non_empty_string(source.get("repository_source"))
                    if key == "repository"
                    else None
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
                        _clean_non_empty_string(nested_source.get("repository_source"))
                        if key == "repository"
                        else None
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


def _extract_repository_identity_from_metadata_sources(
    *sources: Tuple[str, Any],
) -> Optional[str]:
    repository, _source = _extract_repository_identity_from_metadata_sources_with_source(
        *sources
    )
    return repository


def _extract_repository_identity_from_kwargs_with_source(
    kwargs: Dict[str, Any],
    *,
    metadata: Optional[Dict[str, Any]] = None,
    standard_logging_object: Optional[Dict[str, Any]] = None,
) -> Tuple[Optional[str], Optional[str]]:
    litellm_params = kwargs.get("litellm_params") or {}
    standard_logging_object = standard_logging_object or kwargs.get("standard_logging_object") or {}
    standard_metadata = _coerce_mapping(standard_logging_object.get("metadata"))
    requester_custom_headers = _coerce_mapping(
        standard_metadata.get("requester_custom_headers")
    )
    passthrough_payload = kwargs.get("passthrough_logging_payload") or {}
    proxy_request = _coerce_mapping(litellm_params.get("proxy_server_request"))
    proxy_body = _coerce_mapping(proxy_request.get("body"))
    passthrough_body = _coerce_mapping(passthrough_payload.get("request_body"))

    headers = _extract_request_headers_from_kwargs(kwargs)
    for header_name in _AAWM_REPOSITORY_HEADER_NAMES:
        repository = _normalize_repository_identity(
            _get_header_value(headers, header_name)
        )
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


def _extract_repository_identity_from_kwargs(
    kwargs: Dict[str, Any],
    *,
    metadata: Optional[Dict[str, Any]] = None,
    standard_logging_object: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    repository, _source = _extract_repository_identity_from_kwargs_with_source(
        kwargs,
        metadata=metadata,
        standard_logging_object=standard_logging_object,
    )
    return repository


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


def _extract_repository_identity_from_langfuse_trace_observation(
    trace: Dict[str, Any],
    observation: Dict[str, Any],
    metadata: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    repository, _source = _extract_repository_identity_from_langfuse_trace_observation_with_source(
        trace,
        observation,
        metadata,
    )
    return repository


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
            if any(
                marker in normalized
                for marker in _CODEX_MEMORY_WORKFLOW_CONTEXT_MARKERS
            ):
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


def _is_codex_memory_workflow_request(
    kwargs: Dict[str, Any],
    metadata: Dict[str, Any],
    *,
    request_body: Optional[Dict[str, Any]] = None,
) -> bool:
    headers = _extract_request_headers_from_kwargs(kwargs)
    if not _is_native_codex_passthrough_context(metadata, headers):
        return False

    payload = request_body
    if payload is None:
        payload = _extract_provider_cache_request_body(kwargs)
    return _payload_contains_codex_memory_workflow_markers(payload)


def _format_memory_repository_identity(repository: str) -> str:
    if repository.endswith(_CODEX_MEMORY_REPOSITORY_SUFFIX):
        return repository
    return f"{repository}{_CODEX_MEMORY_REPOSITORY_SUFFIX}"


def _apply_codex_memory_workflow_repository(
    kwargs: Dict[str, Any],
    metadata: Dict[str, Any],
    repository: Optional[str],
    *,
    request_body: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    if not repository:
        return repository
    if not _is_codex_memory_workflow_request(
        kwargs,
        metadata,
        request_body=request_body,
    ):
        return repository

    source_repository = repository
    if source_repository.endswith(_CODEX_MEMORY_REPOSITORY_SUFFIX):
        source_repository = source_repository[: -len(_CODEX_MEMORY_REPOSITORY_SUFFIX)]

    metadata["workload_type"] = "agent_memory"
    metadata["workload_subtype"] = "codex_memory_writer"
    metadata["source_repository"] = source_repository
    metadata["repository"] = source_repository
    metadata["memory_workload_label"] = _format_memory_repository_identity(
        source_repository
    )
    _merge_tags(metadata, ["codex-memory-workflow", "agent-memory-workload"])
    return source_repository


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


def _parse_client_identity_from_user_agent(
    user_agent: Optional[str],
) -> Tuple[Optional[str], Optional[str]]:
    if not user_agent:
        return None, None

    known_patterns = (
        (re.compile(r"\bclaude-code/(?P<version>[A-Za-z0-9.+_-]+)"), "claude-code"),
        (re.compile(r"\bcodex-tui/(?P<version>[A-Za-z0-9.+_-]+)"), "codex-tui"),
        (
            re.compile(r"\bGeminiCLI(?:-tui)?/(?P<version>[A-Za-z0-9.+_-]+)"),
            "gemini-cli",
        ),
        (re.compile(r"\bOpenAI/Python\s+(?P<version>[A-Za-z0-9.+_-]+)"), "openai-python"),
        (re.compile(r"\bAnthropic/Python\s+(?P<version>[A-Za-z0-9.+_-]+)"), "anthropic-python"),
    )
    for pattern, client_name in known_patterns:
        match = pattern.search(user_agent)
        if match:
            return client_name, match.group("version")

    for pattern in (_USER_AGENT_PRODUCT_RE, _USER_AGENT_PAREN_PRODUCT_RE):
        match = pattern.search(user_agent)
        if match:
            return match.group("name"), match.group("version")

    return None, None


def _extract_claude_code_version_from_metadata(
    metadata: Dict[str, Any],
) -> Tuple[Optional[str], Optional[str]]:
    billing_header_fields = metadata.get("anthropic_billing_header_fields")
    if not isinstance(billing_header_fields, dict):
        billing_header_fields = {}
    return (
        _first_non_empty_string(metadata.get("cc_version"), billing_header_fields.get("cc_version")),
        _first_non_empty_string(metadata.get("cc_entrypoint"), billing_header_fields.get("cc_entrypoint")),
    )

_SESSION_HISTORY_LOOPBACK_HOST_LABEL = "localhost"


def _clean_session_history_client_ip_candidate(value: Any) -> Optional[str]:
    cleaned = _clean_non_empty_string(value)
    if not cleaned:
        return None
    if "," in cleaned:
        cleaned = cleaned.split(",", 1)[0].strip()
    return cleaned or None


def _canonical_session_history_client_ip(value: Any) -> Optional[str]:
    cleaned = _clean_session_history_client_ip_candidate(value)
    if not cleaned:
        return None
    try:
        return str(ipaddress.ip_address(cleaned))
    except ValueError:
        if cleaned.lower() == _SESSION_HISTORY_LOOPBACK_HOST_LABEL:
            return _SESSION_HISTORY_LOOPBACK_HOST_LABEL
        return None


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
        host_name, resolved_source = _resolve_session_history_host_name_from_ip(
            client_ip
        )
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

    parsed_client_name, parsed_client_version = _parse_client_identity_from_user_agent(
        user_agent
    )
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

    runtime_environment = (
        _get_first_secret_value(_AAWM_LITELLM_ENVIRONMENT_ENV_VARS)
        if allow_runtime
        else None
    )
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
    agent_name = _clean_non_empty_string(
        source.get("agent_name") or source.get("aawm_claude_agent_name")
    )
    if agent_name is None and is_codex_client:
        agent_name = _clean_non_empty_string(
            source.get("agent_role") or source.get("agent_nickname")
        )
    tenant_id = _clean_non_empty_string(
        source.get("tenant_id")
        or source.get("aawm_tenant_id")
        or source.get("aawm_claude_project")
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


def _extract_agent_name(kwargs: Dict[str, Any]) -> str:
    agent_name, _tenant_id = _extract_agent_context(kwargs)
    return agent_name or _DEFAULT_AGENT


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
        or normalized.lower() in {
            "codex",
            "codex-cli",
            "codex-tui",
        }
    )


def _is_native_codex_passthrough_context(
    metadata: Dict[str, Any], headers: Dict[str, Any]
) -> bool:
    route_family = _clean_non_empty_string(metadata.get("passthrough_route_family"))
    if route_family and route_family.lower() == "codex_responses":
        return True

    trace_name = _first_non_empty_string(
        metadata.get("trace_name"),
        _get_header_value(headers, "langfuse_trace_name"),
    )
    user_agent = _get_header_value(headers, "user-agent")
    return bool(
        trace_name
        and trace_name.lower() == "codex"
        and user_agent
        and "codex" in user_agent.lower()
    )


def _is_codex_client_identity(metadata: Dict[str, Any], headers: Dict[str, Any]) -> bool:
    user_agent = _first_non_empty_string(
        metadata.get("client_user_agent"),
        metadata.get("user_agent"),
        metadata.get("http_user_agent"),
        _get_header_value(headers, "user-agent", "User-Agent"),
    )
    parsed_client_name, _parsed_client_version = _parse_client_identity_from_user_agent(
        user_agent
    )
    client_name = _first_non_empty_string(metadata.get("client_name"), parsed_client_name)
    return bool(
        (client_name and "codex" in client_name.lower())
        or (user_agent and "codex" in user_agent.lower())
    )


def _is_codex_default_agent_context(
    kwargs: Dict[str, Any],
    metadata: Optional[Dict[str, Any]] = None,
) -> bool:
    metadata = metadata or _ensure_mutable_metadata(kwargs)
    headers = _extract_request_headers_from_kwargs(kwargs)
    return bool(
        _is_native_codex_passthrough_context(metadata, headers)
        and _is_codex_client_identity(metadata, headers)
        and not _is_codex_subagent_context(kwargs, metadata)
    )


def _is_codex_subagent_context(
    kwargs: Dict[str, Any],
    metadata: Optional[Dict[str, Any]] = None,
) -> bool:
    litellm_params = kwargs.get("litellm_params") or {}
    metadata = metadata or litellm_params.get("metadata") or {}
    standard_logging_object = kwargs.get("standard_logging_object") or {}
    passthrough_payload = kwargs.get("passthrough_logging_payload") or {}
    proxy_request = _coerce_mapping(litellm_params.get("proxy_server_request"))
    proxy_body = _coerce_mapping(proxy_request.get("body"))
    passthrough_body = _coerce_mapping(passthrough_payload.get("request_body"))
    sources = (
        metadata,
        standard_logging_object.get("metadata"),
        proxy_body,
        proxy_body.get("metadata"),
        proxy_body.get("litellm_metadata"),
        passthrough_payload,
        passthrough_body,
        passthrough_body.get("metadata"),
        passthrough_body.get("litellm_metadata"),
    )
    for raw_source in sources:
        source = _coerce_mapping(raw_source)
        if not source:
            continue
        thread_source = _clean_non_empty_string(source.get("thread_source"))
        if thread_source and thread_source.lower() == "subagent":
            return True
        nested_source = _coerce_mapping(source.get("source"))
        nested_thread_source = _clean_non_empty_string(
            nested_source.get("thread_source")
        )
        if nested_thread_source and nested_thread_source.lower() == "subagent":
            return True
        if nested_source.get("subagent"):
            return True
    return False


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
    return normalized_lower in {"grok", "grok-build", "xai"} or normalized_lower.startswith(
        "grok-build."
    )


def _is_native_grok_passthrough_context(
    metadata: Dict[str, Any], headers: Dict[str, Any]
) -> bool:
    route_family = str(metadata.get("passthrough_route_family") or "").lower()
    if "grok" in route_family or "xai" in route_family:
        return True

    client_name = str(metadata.get("client_name") or "").lower()
    if client_name == "grok-build":
        return True

    trace_name = _first_non_empty_string(
        metadata.get("trace_name"),
        _get_header_value(headers, "langfuse_trace_name"),
    )
    if trace_name and str(trace_name).lower().startswith("grok-build"):
        return True

    return any(
        str(header_name).lower().startswith("x-grok-")
        or str(header_name).lower() == "x-xai-token-auth"
        for header_name in headers
    )


def _promote_grok_repository_trace_identity(
    kwargs: Dict[str, Any],
    metadata: Dict[str, Any],
    headers: Dict[str, Any],
) -> None:
    if not _is_native_grok_passthrough_context(metadata, headers):
        return

    repository = _extract_repository_identity_from_kwargs(
        kwargs,
        metadata=metadata,
    )
    if repository:
        metadata["repository"] = repository

    tenant_id, tenant_source = _extract_tenant_identity_from_kwargs(
        kwargs,
        metadata=metadata,
    )
    if not tenant_id:
        _agent_name, agent_context_tenant_id = _extract_agent_context(kwargs)
        if agent_context_tenant_id:
            tenant_id = agent_context_tenant_id
            tenant_source = "agent_context_text"
    if tenant_id and not metadata.get("tenant_id"):
        metadata["tenant_id"] = tenant_id
    if tenant_id and tenant_source and not metadata.get("tenant_id_source"):
        metadata["tenant_id_source"] = tenant_source

    metadata_trace_user_id = _clean_non_empty_string(metadata.get("trace_user_id"))
    header_trace_user_id = _get_header_value(headers, "langfuse_trace_user_id")
    desired_trace_user_id = repository or tenant_id
    if not desired_trace_user_id:
        return

    if metadata_trace_user_id is None or _is_generic_grok_trace_user_id(
        metadata_trace_user_id
    ):
        metadata["trace_user_id"] = desired_trace_user_id
    if header_trace_user_id is None or _is_generic_grok_trace_user_id(
        header_trace_user_id
    ):
        headers["langfuse_trace_user_id"] = desired_trace_user_id


def _promote_codex_repository_trace_user_id(
    kwargs: Dict[str, Any],
    metadata: Dict[str, Any],
    headers: Dict[str, Any],
) -> None:
    if not _is_native_codex_passthrough_context(metadata, headers):
        return

    if _is_numeric_identity_placeholder(metadata.get("repository")):
        metadata.pop("repository", None)
    if _is_numeric_identity_placeholder(metadata.get("tenant_id")):
        metadata.pop("tenant_id", None)
        metadata.pop("tenant_id_source", None)
    if _is_numeric_identity_placeholder(metadata.get("trace_user_id")):
        metadata.pop("trace_user_id", None)
    if _is_numeric_identity_placeholder(
        _get_header_value(headers, "langfuse_trace_user_id")
    ):
        headers.pop("langfuse_trace_user_id", None)

    metadata_trace_user_id = _normalize_repository_identity(
        metadata.get("trace_user_id")
    )
    header_trace_user_id = _get_header_value(headers, "langfuse_trace_user_id")
    repository, repository_source = _extract_repository_identity_from_kwargs_with_source(
        kwargs,
        metadata=metadata,
    )
    repository_before_memory_workflow = repository
    repository = _apply_codex_memory_workflow_repository(
        kwargs,
        metadata,
        repository,
    )
    if repository and repository_source:
        if repository != repository_before_memory_workflow:
            repository_source = f"{repository_source}.codex_memory_workflow"
        metadata["repository_source"] = repository_source

    desired_trace_user_id: Optional[str] = None
    if metadata_trace_user_id and not _is_generic_codex_trace_user_id(
        metadata_trace_user_id
    ):
        desired_trace_user_id = metadata_trace_user_id
    elif (
        repository
        and _is_repository_source_trusted_for_tenant(repository_source)
        and (
            metadata_trace_user_id is None
            or _is_generic_codex_trace_user_id(metadata_trace_user_id)
        )
        and (
            header_trace_user_id is None
            or _is_generic_codex_trace_user_id(header_trace_user_id)
        )
    ):
        desired_trace_user_id = repository

    if not desired_trace_user_id:
        return

    metadata["trace_user_id"] = desired_trace_user_id
    if header_trace_user_id is None or _is_generic_codex_trace_user_id(
        header_trace_user_id
    ):
        headers["langfuse_trace_user_id"] = desired_trace_user_id


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
            return _extract_claude_permission_check_decision_from_value(
                parsed_value, _depth=_depth + 1
            )
        return None

    if isinstance(value, list):
        text_value = _content_to_text(value).strip()
        match = _CLAUDE_PERMISSION_CHECK_OUTPUT_RE.match(text_value)
        if match is not None:
            return match.group("decision").lower()
        for item in value:
            decision = _extract_claude_permission_check_decision_from_value(
                item, _depth=_depth + 1
            )
            if decision is not None:
                return decision
        return None

    if not _permission_check_probeable_value(value):
        return None

    content = _maybe_get(value, "content")
    if content is not None and content is not value:
        decision = _extract_claude_permission_check_decision_from_value(
            content, _depth=_depth + 1
        )
        if decision is not None:
            return decision

    message = _extract_first_response_message(value)
    if message is not None and message is not value:
        decision = _extract_claude_permission_check_decision_from_value(
            message, _depth=_depth + 1
        )
        if decision is not None:
            return decision

    response = _maybe_get(value, "response")
    if response is not None and response is not value:
        decision = _extract_claude_permission_check_decision_from_value(
            response, _depth=_depth + 1
        )
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
        _maybe_get_path(
            kwargs.get("passthrough_logging_payload"), "request_body", "model"
        ),
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
    standard_logging_object = standard_logging_object or kwargs.get(
        "standard_logging_object"
    ) or {}
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
            "claude-permission-check:block"
            if blocked
            else "claude-permission-check:allow",
        ],
    )

    existing_spans = metadata.get("langfuse_spans") or []
    if not isinstance(existing_spans, list):
        existing_spans = []
    if any(
        isinstance(span, dict) and span.get("name") == "claude.permission_check"
        for span in existing_spans
    ):
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
        if tag == "claude-permission-check" or tag.startswith(
            "claude-permission-check:"
        ):
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
    if (
        resolved_source_model
        and resolved_source_model != _CLAUDE_AUTO_REVIEW_LOGICAL_MODEL
    ):
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
        metadata["claude_auto_review_parent_identity_source_row_id"] = identity[
            "source_row_id"
        ]
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
    return _nonnegative_float_or_none(
        (normalized_end - normalized_start).total_seconds() * 1000.0
    )


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
        llm_upstream_stream_ms = _nonnegative_float_or_none(
            upstream_stream_complete_ms - upstream_first_chunk_ms
        )

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
        classified_ms = (litellm_processing_ms or 0.0) + (
            llm_upstream_elapsed_ms or 0.0
        )
        latency_unclassified_ms = _nonnegative_float_or_none(
            total_server_elapsed_ms - classified_ms
        )
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


def _parse_provider_timestamp(value: Any) -> Optional[datetime]:
    if isinstance(value, (int, float)):
        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            return None
        if numeric_value <= 0:
            return None
        if numeric_value > 1_000_000_000_000:
            numeric_value = numeric_value / 1000.0
        try:
            return datetime.fromtimestamp(numeric_value, tz=timezone.utc)
        except (OverflowError, OSError, ValueError):
            return None
    parsed = _parse_datetime_value(value)
    if parsed is not None:
        return parsed
    if isinstance(value, str):
        numeric_string_value = _safe_float(value.strip())
        if numeric_string_value is not None:
            return _parse_provider_timestamp(numeric_string_value)
    return None


def _infer_window_start_at(
    provider_resets_at: Optional[datetime],
    window_minutes: Optional[int],
) -> Optional[datetime]:
    if provider_resets_at is None or window_minutes is None or window_minutes <= 0:
        return None
    return provider_resets_at - timedelta(minutes=window_minutes)


def _quota_period_from_window_minutes(window_minutes: Optional[int]) -> Optional[str]:
    if window_minutes is None:
        return None
    if window_minutes == 60:
        return "hourly"
    if window_minutes == 300:
        return "five_hour"
    if window_minutes == 10080:
        return "seven_day"
    if window_minutes == 1440:
        return "daily"
    return f"{window_minutes}_minutes"


def _normalize_quota_period(value: Any) -> Optional[str]:
    normalized = _clean_non_empty_string(value)
    if normalized is None:
        return None
    normalized = normalized.strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "1h": "hourly",
        "hour": "hourly",
        "hourly": "hourly",
        "5h": "five_hour",
        "five_h": "five_hour",
        "five_hour": "five_hour",
        "five_hours": "five_hour",
        "daily": "daily",
        "day": "daily",
        "1d": "daily",
        "7d": "seven_day",
        "seven_day": "seven_day",
        "seven_days": "seven_day",
        "weekly": "weekly",
        "week": "weekly",
        "monthly": "monthly",
        "month": "monthly",
    }
    return aliases.get(normalized, normalized)


def _window_minutes_from_quota_period(quota_period: Optional[str]) -> Optional[int]:
    if quota_period == "hourly":
        return 60
    if quota_period == "five_hour":
        return 300
    if quota_period == "daily":
        return 1440
    if quota_period in {"seven_day", "weekly"}:
        return 10080
    return None


def _parse_reset_hint_seconds(*values: Any) -> Optional[int]:
    for value in values:
        parsed = _safe_int(value)
        if parsed is not None and parsed >= 0:
            return parsed
        if not isinstance(value, str):
            continue
        match = _RESET_AFTER_SECONDS_RE.search(value)
        if match is None:
            continue
        parsed = _safe_int(match.group("seconds"))
        if parsed is not None and parsed >= 0:
            return parsed
    return None


def _resolve_rate_limit_reset_at(
    reset_value: Any,
    observed_at: Any,
    reset_hint_seconds: Optional[int] = None,
) -> Tuple[Optional[datetime], bool]:
    provider_resets_at = _parse_provider_timestamp(reset_value)
    observed_dt = _normalize_datetime(observed_at)
    if (
        provider_resets_at is not None
        and observed_dt is not None
        and provider_resets_at
        < observed_dt - _AAWM_RATE_LIMIT_STALE_RESET_TOLERANCE
    ):
        if reset_hint_seconds is not None:
            return observed_dt + timedelta(seconds=reset_hint_seconds), False
        return None, True
    if (
        provider_resets_at is None
        and reset_hint_seconds is not None
        and observed_dt is not None
    ):
        return observed_dt + timedelta(seconds=reset_hint_seconds), False
    return provider_resets_at, False


def _json_safe_rate_limit_value(
    value: Any,
    *,
    _seen: Optional[Set[int]] = None,
    _depth: int = 0,
) -> Any:
    if _seen is None:
        _seen = set()
    if _depth > _AAWM_JSON_SAFE_MAX_DEPTH:
        return "<max_depth>"
    if isinstance(value, datetime):
        return _format_langfuse_span_timestamp(value)
    if isinstance(value, dict):
        value_id = id(value)
        if value_id in _seen:
            return "<recursive>"
        _seen.add(value_id)
        try:
            return {
                str(key): _json_safe_rate_limit_value(
                    nested_value,
                    _seen=_seen,
                    _depth=_depth + 1,
                )
                for key, nested_value in list(value.items())
                if isinstance(key, (str, int, float, bool))
            }
        finally:
            _seen.discard(value_id)
    if isinstance(value, list):
        value_id = id(value)
        if value_id in _seen:
            return ["<recursive>"]
        _seen.add(value_id)
        try:
            return [
                _json_safe_rate_limit_value(
                    item,
                    _seen=_seen,
                    _depth=_depth + 1,
                )
                for item in value[:100]
            ]
        finally:
            _seen.discard(value_id)
    if isinstance(value, tuple):
        value_id = id(value)
        if value_id in _seen:
            return ["<recursive>"]
        _seen.add(value_id)
        try:
            return [
                _json_safe_rate_limit_value(
                    item,
                    _seen=_seen,
                    _depth=_depth + 1,
                )
                for item in value[:100]
            ]
        finally:
            _seen.discard(value_id)
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8", errors="replace")[:500]
        except Exception:
            return "<bytes>"
    if isinstance(value, (str, int, float, bool)) or value is None:
        if isinstance(value, str):
            return value[:1000]
        return value
    return str(value)[:500]


def _coerce_rate_limit_payload(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return value
    if hasattr(value, "items"):
        try:
            return {
                str(key): nested_value
                for key, nested_value in list(value.items())
            }
        except Exception:
            return None
    if isinstance(value, bytes):
        try:
            return _coerce_rate_limit_payload(value.decode("utf-8", errors="replace"))
        except Exception:
            return None
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    if not stripped:
        return None
    # Fail closed on unbounded attacker/provider-influenced text before
    # JSON/literal evaluation (ast.literal_eval is not DoS-safe on deep nests).
    if len(stripped) > 8192:
        return None
    parsed = _safe_json_load(stripped, None)
    if parsed is not None:
        return parsed
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            literal_value = ast.literal_eval(stripped)
    except Exception:
        return None
    if isinstance(literal_value, bytes):
        return _coerce_rate_limit_payload(literal_value)
    if isinstance(literal_value, (dict, list)):
        return literal_value
    return None


def _iter_rate_limit_dicts(*roots: Any) -> List[Dict[str, Any]]:
    pending: List[Tuple[Any, int]] = [(root, 0) for root in roots if root is not None]
    seen: set = set()
    dicts: List[Dict[str, Any]] = []
    while pending and len(seen) < 512:
        value, depth = pending.pop(0)
        coerced = _coerce_rate_limit_payload(value)
        if coerced is not None:
            value = coerced
        value_id = id(value)
        if value_id in seen:
            continue
        seen.add(value_id)
        if isinstance(value, dict):
            dicts.append(value)
            if depth >= 6:
                continue
            for nested_value in list(value.values()):
                if isinstance(nested_value, (dict, list, str, bytes)):
                    pending.append((nested_value, depth + 1))
        elif isinstance(value, list):
            if depth >= 6:
                continue
            for item in value[:200]:
                if isinstance(item, (dict, list, str, bytes)):
                    pending.append((item, depth + 1))
    return dicts


def _merged_rate_limit_metadata(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    metadata: Dict[str, Any] = {}
    standard_logging_object = kwargs.get("standard_logging_object")
    if isinstance(standard_logging_object, dict):
        standard_metadata = standard_logging_object.get("metadata")
        if isinstance(standard_metadata, dict):
            metadata.update(dict(standard_metadata))
    litellm_params = kwargs.get("litellm_params")
    if isinstance(litellm_params, dict):
        litellm_metadata = litellm_params.get("metadata")
        if isinstance(litellm_metadata, dict):
            metadata.update(dict(litellm_metadata))
    return metadata


def _extract_headers_from_kwargs(kwargs: Dict[str, Any]) -> Dict[str, str]:
    headers: Dict[str, str] = {}
    for candidate in (
        _maybe_get_path(kwargs.get("litellm_params"), "proxy_server_request", "headers"),
        _maybe_get_path(kwargs.get("passthrough_logging_payload"), "request_headers"),
        _maybe_get_path(kwargs.get("standard_pass_through_logging_payload"), "request_headers"),
        _maybe_get_path(kwargs.get("standard_logging_object"), "request_headers"),
        kwargs.get("headers"),
    ):
        if not isinstance(candidate, dict):
            continue
        for key, value in list(candidate.items()):
            if isinstance(key, str) and value is not None:
                headers[key.lower()] = str(value)
    return headers


def _extract_rate_limit_account_hash(
    kwargs: Dict[str, Any],
    metadata: Dict[str, Any],
) -> Optional[str]:
    headers = _extract_headers_from_kwargs(kwargs)
    user_api_key_dict = kwargs.get("user_api_key_dict") or kwargs.get("user_api_key")
    candidates = [
        metadata.get("user_api_key_hash"),
        metadata.get("api_key_hash"),
        metadata.get("provider_account_hash"),
        metadata.get("provider_account_id"),
        metadata.get("organization_id"),
        metadata.get("org_id"),
        kwargs.get("user_api_key_hash"),
        _maybe_get(user_api_key_dict, "api_key_hash"),
        _maybe_get(user_api_key_dict, "token"),
        _maybe_get(user_api_key_dict, "api_key"),
        headers.get("x-litellm-user-api-key-hash"),
        headers.get("x-api-key-hash"),
        headers.get("x-goog-user-project"),
        headers.get("anthropic-organization-id"),
        headers.get("openai-organization"),
        headers.get("x-grok-user-id"),
        headers.get("x-userid"),
        headers.get("x-teamid"),
        headers.get("x-email"),
        headers.get("x-xai-token-auth"),
        headers.get("authorization"),
    ]
    for candidate in candidates:
        if candidate is None:
            continue
        candidate_text = str(candidate).strip()
        if not candidate_text:
            continue
        return _short_hash(candidate_text.encode("utf-8"))
    return None


def _resolve_rate_limit_model(
    kwargs: Dict[str, Any],
    result: Any,
    metadata: Dict[str, Any],
) -> str:
    standard_logging_object = kwargs.get("standard_logging_object") or {}
    try:
        model = _resolve_session_history_model(
            kwargs=kwargs,
            standard_logging_object=standard_logging_object,
            metadata=metadata,
            result=result,
        )
        if model and model != "unknown":
            return model
    except Exception:
        pass
    for candidate in (
        kwargs.get("model"),
        metadata.get("model"),
        _maybe_get_path(kwargs.get("passthrough_logging_payload"), "request_body", "model"),
        _maybe_get_path(kwargs.get("litellm_params"), "proxy_server_request", "body", "model"),
        _maybe_get(result, "model"),
    ):
        if candidate is not None and str(candidate).strip():
            return str(candidate).strip()
    return "unknown"


def _infer_model_family_and_tier(*values: Any) -> Tuple[Optional[str], Optional[str]]:
    text = " ".join(str(value) for value in values if value is not None).lower()

    def _has_token(token: str) -> bool:
        return re.search(rf"(?<![a-z0-9_]){re.escape(token)}(?![a-z0-9_])", text) is not None

    model_tier = None
    if _has_token("sonnet"):
        model_tier = "sonnet"
    elif _has_token("opus"):
        model_tier = "opus"
    elif _has_token("haiku"):
        model_tier = "haiku"
    elif "flash-lite" in text or "flash_lite" in text or _has_token("flash-lite"):
        model_tier = "flash_lite"
    elif _has_token("flash"):
        model_tier = "flash"
    elif _has_token("pro"):
        model_tier = "pro"

    if "claude" in text or model_tier in {"sonnet", "opus", "haiku"}:
        return "claude", model_tier
    # Prefer explicit OpenAI/Codex markers over tier-based gemini inference so
    # names like gpt-5-pro / o1-pro and metadata containing "project"/"prod"
    # do not get misclassified as gemini solely because of a "pro" token.
    if (
        "gpt" in text
        or "openai" in text
        or _has_token("o1")
        or _has_token("o3")
        or _has_token("o4")
    ):
        return "openai", model_tier
    if "codex" in text:
        return "codex", model_tier
    if "gemini" in text or "gemma" in text or model_tier in {"pro", "flash", "flash_lite"}:
        return "gemini", model_tier
    return None, model_tier


def _infer_rate_limit_client_family(
    provider: Optional[str],
    model: str,
    metadata: Dict[str, Any],
    source: Optional[str],
) -> Optional[str]:
    source_lower = str(source or "").lower()
    route_family = str(metadata.get("passthrough_route_family") or "").lower()
    model_lower = str(model or "").lower()
    client_text = " ".join(
        str(value)
        for value in (
            metadata.get("client_name"),
            metadata.get("client_version"),
            metadata.get("trace_name"),
            metadata.get("cc_version"),
            metadata.get("cc_entrypoint"),
        )
        if value is not None
    ).lower()
    credential_family = str(metadata.get("credential_family") or "").lower()
    if (
        provider == "antigravity"
        or source_lower.startswith("antigravity_")
        or "antigravity" in route_family
        or metadata.get("aawm_stream_logging_custom_llm_provider") == "antigravity"
        or str(metadata.get("custom_llm_provider") or "").lower() == "antigravity"
        or model_lower.startswith(("antigravity/", "agy/", "google-antigravity/"))
    ):
        return "antigravity_code_assist"
    if (
        "opencode" in source_lower
        or credential_family == "opencode"
        or metadata.get("opencode_zen") is True
        or "opencode" in route_family
        or model_lower.startswith(("opencode/", "opencode-zen/", "zen/"))
    ):
        return "opencode_zen"
    if (
        "xai_oauth" in source_lower
        or credential_family == "xai_oauth"
        or metadata.get("xai_oauth_managed") is True
        or metadata.get("xai_oauth_public_model") is not None
        or "xai_oauth" in route_family
    ):
        return "xai_oauth"
    if (
        "google_code_assist" in source_lower
        or "google_retrieve_user_quota" in source_lower
        or "code_assist" in route_family
    ):
        return "google_code_assist"
    if "codex" in source_lower or "codex" in route_family or "codex" in model_lower:
        return "codex"
    if "gemini" in source_lower or "gemini" in route_family or "gemini" in model_lower:
        return "gemini"
    if (
        "grok" in source_lower
        or "grok" in route_family
        or "xai" in route_family
        or "grok" in model_lower
        or "grok-build" in client_text
    ):
        return "grok-build"
    if (
        "claude" in source_lower
        or "claude" in route_family
        or "claude" in client_text
        or "cc_version" in metadata
    ):
        return "claude"
    return provider


def _openrouter_free_daily_request_limit() -> int:
    configured_limit = _safe_int(
        get_secret_str("AAWM_OPENROUTER_FREE_DAILY_REQUEST_LIMIT")
    )
    if configured_limit is not None and configured_limit > 0:
        return configured_limit
    return _AAWM_OPENROUTER_FREE_DAILY_REQUEST_LIMIT_DEFAULT


def _openrouter_free_shared_account_hash() -> str:
    return _short_hash(b"openrouter_free_daily_shared_pool")


def _is_openrouter_free_model(model: Any) -> bool:
    return str(model or "").strip().lower().endswith(":free")


def _openrouter_free_daily_window(observed_at: Any) -> Tuple[datetime, datetime]:
    observed_dt = _normalize_datetime(observed_at) or datetime.now(timezone.utc)
    day_start = observed_dt.astimezone(timezone.utc).replace(
        hour=0,
        minute=0,
        second=0,
        microsecond=0,
    )
    return day_start, day_start + timedelta(days=1)


def _openrouter_free_daily_observation_context_from_record(
    record: Dict[str, Any],
    observed_at: datetime,
) -> Dict[str, Any]:
    metadata = record.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
    return {
        "observed_at": observed_at,
        "provider": "openrouter",
        "client_family": "openrouter",
        "account_hash": _openrouter_free_shared_account_hash(),
        "environment": record.get("litellm_environment"),
        "tenant_id": record.get("tenant_id"),
        "repository": record.get("repository"),
        "session_id": record.get("session_id"),
        "trace_id": record.get("trace_id"),
        "litellm_call_id": record.get("litellm_call_id"),
        "route_family": metadata.get("passthrough_route_family"),
        "request_model": record.get("model"),
        "response_model": None,
        "model": None,
        "model_family": "openrouter",
        "model_tier": "free",
        "client_name": record.get("client_name"),
        "client_version": record.get("client_version"),
        "client_user_agent": record.get("client_user_agent"),
        "metadata": metadata,
    }


def _build_openrouter_free_daily_observation(
    *,
    context: Dict[str, Any],
    day_start: datetime,
    day_end: datetime,
    used_requests: int,
    total_requests: int,
    signal: str,
    status: str = "observed",
    exhausted: bool = False,
    reset_hint_seconds: Optional[int] = None,
    provider_resets_at: Optional[datetime] = None,
) -> Dict[str, Any]:
    bounded_total = max(total_requests, 1)
    bounded_used = max(0, used_requests)
    remaining_requests = max(0, bounded_total - bounded_used)
    used_percentage = round(
        min(100.0, (bounded_used / bounded_total) * 100.0),
        3,
    )
    remaining_pct = round(max(0.0, 100.0 - used_percentage), 3)
    return _finalize_rate_limit_observation(
        {
            "observed_at": context["observed_at"],
            "source": _AAWM_OPENROUTER_FREE_DAILY_SOURCE,
            "provider": "openrouter",
            "client_family": "openrouter",
            "account_hash": _openrouter_free_shared_account_hash(),
            "limit_id": "openrouter_free_daily_requests",
            "limit_name": "OpenRouter free daily requests",
            "limit_scope": "requests",
            "window_minutes": 1440,
            "quota_period": "daily",
            "quota_type": "requests",
            "provider_resets_at": provider_resets_at or day_end,
            "remaining_pct": remaining_pct,
            "used_percentage": used_percentage,
            "remaining_requests": remaining_requests,
            "used_requests": bounded_used,
            "total_requests": bounded_total,
            "status": status,
            "exhausted": exhausted or remaining_requests <= 0,
            "exhaustion_kind": "request_quota" if exhausted else None,
            "reset_hint_seconds": reset_hint_seconds,
            "model": None,
            "model_family": "openrouter",
            "model_tier": "free",
            "raw_provider_fields": {
                "dailyLimit": bounded_total,
                "usedRequests": bounded_used,
                "remainingRequests": remaining_requests,
                "windowStart": _json_safe_rate_limit_value(day_start),
                "windowEnd": _json_safe_rate_limit_value(day_end),
                "reset_anchor": "utc_midnight",
                "model_scope": "openrouter_:free_shared_pool",
                "meter_source": "local_session_history",
            },
            "evidence": {
                "signals": [signal],
                "provider_fields": [],
                "scope_note": (
                    "OpenRouter documents free-model quota as account-level; "
                    "provider does not expose current free request usage."
                ),
            },
        },
        context,
    )


def _openrouter_free_record_observed_at(record: Dict[str, Any]) -> datetime:
    return (
        _normalize_datetime(record.get("end_time"))
        or _normalize_datetime(record.get("start_time"))
        or datetime.now(timezone.utc)
    )


def _is_openrouter_free_session_history_record(record: Dict[str, Any]) -> bool:
    model = record.get("model")
    if not _is_openrouter_free_model(model):
        return False
    metadata = record.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
    provider = _normalize_session_history_provider(
        record.get("provider"),
        str(model or ""),
        metadata,
    )
    return provider == "openrouter"


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
                status=(
                    "quota_exhausted"
                    if used_requests >= total_requests
                    else "observed"
                ),
                exhausted=used_requests >= total_requests,
            )
        )
    return observations


def _build_rate_limit_key(
    *,
    provider: Optional[str],
    client_family: Optional[str],
    account_hash: Optional[str],
    limit_id: Optional[str],
    limit_name: Optional[str],
    limit_scope: Optional[str],
    quota_period: Optional[str],
    window_minutes: Optional[int],
    model: Optional[str],
    model_family: Optional[str],
) -> str:
    identity = (
        _clean_non_empty_string(limit_id)
        or _clean_non_empty_string(limit_name)
        or (
            _clean_non_empty_string(model)
            if str(limit_scope or "").startswith("model")
            else None
        )
        or _clean_non_empty_string(model_family)
        or "default"
    )
    parts = (
        provider or "unknown_provider",
        client_family or "unknown_client",
        account_hash or "unknown_account",
        identity,
        limit_scope or quota_period or "unknown_scope",
        str(window_minutes or "unknown_window"),
    )
    normalized_parts = [
        re.sub(r"[^a-z0-9_.-]+", "_", str(part).strip().lower()).strip("_") or "unknown"
        for part in parts
    ]
    return ":".join(normalized_parts)


_AAWM_RATE_LIMIT_CONTEXT_CACHE_KEY = "_aawm_rate_limit_context_cache"


def _build_rate_limit_context(
    kwargs: Dict[str, Any],
    result: Any,
    end_time: Any,
    source: Optional[str],
) -> Dict[str, Any]:
    """Build (and request-cache) shared rate-limit observation context.

    Repository/tenant extraction can deep-walk large request payloads. Cache the
    expensive identity fields once per kwargs object so the nine extractors that
    call this helper do not re-scan the miss path.
    """
    cache: Optional[Dict[str, Any]] = None
    if isinstance(kwargs, dict):
        raw_cache = kwargs.get(_AAWM_RATE_LIMIT_CONTEXT_CACHE_KEY)
        if isinstance(raw_cache, dict):
            cache = raw_cache
        else:
            cache = {}
            kwargs[_AAWM_RATE_LIMIT_CONTEXT_CACHE_KEY] = cache

    cache_key = (
        id(result),
        source,
        id(end_time) if not isinstance(end_time, (str, int, float)) else end_time,
    )
    if cache is not None and cache_key in cache:
        cached = cache[cache_key]
        if isinstance(cached, dict):
            # Return a shallow copy so extractors can mutate client_family safely.
            return dict(cached)

    metadata = _merged_rate_limit_metadata(kwargs)
    model = _resolve_rate_limit_model(kwargs, result, metadata)
    provider = _normalize_session_history_provider(
        kwargs.get("custom_llm_provider"),
        model,
        metadata,
    )
    client_family = _infer_rate_limit_client_family(provider, model, metadata, source)
    model_family, model_tier = _infer_model_family_and_tier(
        model,
        metadata.get("model"),
        metadata.get("anthropic_adapter_model"),
        metadata.get("codex_adapter_model"),
    )
    runtime_identity = _build_session_runtime_identity(
        metadata=metadata,
        kwargs=kwargs,
        allow_runtime=True,
    )

    identity_cache_key = "_identity"
    identity: Optional[Dict[str, Any]] = None
    if cache is not None and isinstance(cache.get(identity_cache_key), dict):
        identity = cache[identity_cache_key]
    if identity is None:
        tenant_id, _tenant_source = _extract_tenant_identity_from_kwargs(
            kwargs,
            metadata=metadata,
            standard_logging_object=kwargs.get("standard_logging_object") or {},
        )
        repository = _extract_repository_identity_from_kwargs(
            kwargs,
            metadata=metadata,
            standard_logging_object=kwargs.get("standard_logging_object") or {},
        )
        identity = {
            "tenant_id": tenant_id,
            "repository": repository,
            "session_id": _extract_session_id(kwargs),
            "trace_id": _extract_trace_id(kwargs),
            "account_hash": _extract_rate_limit_account_hash(kwargs, metadata),
            "environment": runtime_identity.get("litellm_environment"),
            "client_name": runtime_identity.get("client_name"),
            "client_version": runtime_identity.get("client_version"),
            "client_user_agent": runtime_identity.get("client_user_agent"),
        }
        if cache is not None:
            cache[identity_cache_key] = identity

    context = {
        "observed_at": _normalize_datetime(end_time) or datetime.now(timezone.utc),
        "provider": provider,
        "client_family": client_family,
        "account_hash": identity["account_hash"],
        "environment": identity["environment"],
        "tenant_id": identity["tenant_id"],
        "repository": identity["repository"],
        "session_id": identity["session_id"],
        "trace_id": identity["trace_id"],
        "litellm_call_id": kwargs.get("litellm_call_id"),
        "route_family": metadata.get("passthrough_route_family"),
        "request_model": _first_non_empty_string(
            _maybe_get_path(kwargs.get("passthrough_logging_payload"), "request_body", "model"),
            _maybe_get_path(kwargs.get("litellm_params"), "proxy_server_request", "body", "model"),
        ),
        "response_model": _first_non_empty_string(
            _maybe_get(result, "model"),
            _maybe_get_path(kwargs.get("standard_logging_object"), "response", "model"),
        ),
        "model": model,
        "model_family": model_family,
        "model_tier": model_tier,
        "client_name": identity["client_name"],
        "client_version": identity["client_version"],
        "client_user_agent": identity["client_user_agent"],
        "metadata": metadata,
    }
    if cache is not None:
        cache[cache_key] = context
    return dict(context)


def _finalize_rate_limit_observation(
    observation: Dict[str, Any],
    context: Dict[str, Any],
) -> Dict[str, Any]:
    finalized = dict(context)
    finalized.update(observation)
    finalized["observed_at"] = (
        _normalize_datetime(finalized.get("observed_at"))
        or context.get("observed_at")
        or datetime.now(timezone.utc)
    )
    finalized["provider_resets_at"] = _parse_provider_timestamp(
        finalized.get("provider_resets_at")
    )
    window_minutes = _safe_int(finalized.get("window_minutes"))
    finalized["window_minutes"] = window_minutes
    if finalized.get("quota_period") is None:
        finalized["quota_period"] = _quota_period_from_window_minutes(window_minutes)
    finalized["inferred_window_start_at"] = _infer_window_start_at(
        finalized.get("provider_resets_at"),
        window_minutes,
    )
    finalized["used_percentage"] = _safe_float(finalized.get("used_percentage"))
    finalized["remaining_requests"] = _safe_int(finalized.get("remaining_requests"))
    finalized["used_requests"] = _safe_int(finalized.get("used_requests"))
    finalized["total_requests"] = _safe_int(finalized.get("total_requests"))
    finalized["reset_hint_seconds"] = _safe_int(finalized.get("reset_hint_seconds"))
    model_family, model_tier = _infer_model_family_and_tier(
        finalized.get("model"),
        finalized.get("limit_name"),
        finalized.get("raw_provider_fields"),
    )
    finalized["model_family"] = finalized.get("model_family") or model_family
    finalized["model_tier"] = finalized.get("model_tier") or model_tier
    finalized_metadata = finalized.get("metadata")
    if not isinstance(finalized_metadata, dict):
        finalized_metadata = {}
    finalized["client_family"] = finalized.get(
        "client_family"
    ) or _infer_rate_limit_client_family(
        finalized.get("provider"),
        str(finalized.get("model") or ""),
        finalized_metadata,
        finalized.get("source"),
    )
    finalized["limit_key"] = _build_rate_limit_key(
        provider=finalized.get("provider"),
        client_family=finalized.get("client_family"),
        account_hash=finalized.get("account_hash"),
        limit_id=_clean_non_empty_string(finalized.get("limit_id")),
        limit_name=_clean_non_empty_string(finalized.get("limit_name")),
        limit_scope=_clean_non_empty_string(finalized.get("limit_scope")),
        quota_period=_clean_non_empty_string(finalized.get("quota_period")),
        window_minutes=window_minutes,
        model=_clean_non_empty_string(finalized.get("model")),
        model_family=_clean_non_empty_string(finalized.get("model_family")),
    )
    raw_provider_fields = finalized.get("raw_provider_fields")
    if not isinstance(raw_provider_fields, dict):
        raw_provider_fields = {}
    finalized["raw_provider_fields"] = raw_provider_fields
    evidence = finalized.get("evidence")
    if not isinstance(evidence, dict):
        evidence = {}
    finalized["evidence"] = evidence
    metadata = finalized.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
    finalized["metadata"] = {
        key: _json_safe_rate_limit_value(metadata.get(key))
        for key in _AAWM_RATE_LIMIT_METADATA_KEYS
        if metadata.get(key) is not None
    }
    finalized["exhausted"] = bool(finalized.get("exhausted"))
    if finalized.get("status") is None:
        finalized["status"] = "exhausted" if finalized["exhausted"] else "observed"
    return finalized


def _dedupe_rate_limit_observations(
    observations: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    deduped: List[Dict[str, Any]] = []
    seen = set()
    for observation in observations:
        key = (
            observation.get("source"),
            observation.get("limit_key"),
            observation.get("provider_resets_at"),
            observation.get("used_percentage"),
            observation.get("remaining_requests"),
            observation.get("used_requests"),
            observation.get("total_requests"),
            observation.get("status"),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(observation)
    return deduped


def _rate_limit_snapshot_signature(
    observation: Dict[str, Any],
    *,
    include_reset: bool = True,
) -> Tuple[Any, ...]:
    provider_resets_at = _parse_provider_timestamp(observation.get("provider_resets_at"))
    body = (
        _safe_float(observation.get("used_percentage")),
        _safe_int(observation.get("remaining_requests")),
        _safe_int(observation.get("used_requests")),
        _safe_int(observation.get("total_requests")),
        _rate_limit_storage_quota_limit(observation),
        _rate_limit_storage_quota_used(observation),
        _rate_limit_storage_quota_remaining(observation),
        _rate_limit_storage_billing_period_start_at(observation),
        _rate_limit_storage_billing_period_end_at(observation),
        _clean_non_empty_string(observation.get("status")),
        bool(observation.get("exhausted")),
        _clean_non_empty_string(observation.get("exhaustion_kind")),
        None
        if provider_resets_at is not None
        else _safe_int(observation.get("reset_hint_seconds")),
    )
    if include_reset:
        return (provider_resets_at, *body)
    return body


def _rate_limit_observation_has_meaningful_change(
    previous: Optional[Dict[str, Any]],
    current: Dict[str, Any],
) -> bool:
    if previous is None:
        return True

    previous_reset = _parse_provider_timestamp(previous.get("provider_resets_at"))
    current_reset = _parse_provider_timestamp(current.get("provider_resets_at"))
    previous_without_reset = _rate_limit_snapshot_signature(
        previous, include_reset=False
    )
    current_without_reset = _rate_limit_snapshot_signature(
        current, include_reset=False
    )
    if previous_without_reset != current_without_reset:
        return True
    if previous_reset is None or current_reset is None:
        return previous_reset != current_reset
    return (
        abs((current_reset - previous_reset).total_seconds())
        >= _AAWM_RATE_LIMIT_MEANINGFUL_RESET_SHIFT.total_seconds()
    )


def _rate_limit_candidate_roots(kwargs: Dict[str, Any], result: Any) -> List[Any]:
    metadata = _merged_rate_limit_metadata(kwargs)
    standard_logging_object = kwargs.get("standard_logging_object") or {}
    litellm_params = kwargs.get("litellm_params") or {}
    roots: List[Any] = [
        result,
        metadata,
        standard_logging_object.get("metadata") if isinstance(standard_logging_object, dict) else None,
        standard_logging_object.get("response") if isinstance(standard_logging_object, dict) else None,
        standard_logging_object.get("output") if isinstance(standard_logging_object, dict) else None,
        kwargs.get("passthrough_logging_payload"),
        kwargs.get("standard_pass_through_logging_payload"),
        litellm_params.get("metadata") if isinstance(litellm_params, dict) else None,
    ]
    for candidate in (
        result,
        standard_logging_object.get("response")
        if isinstance(standard_logging_object, dict)
        else None,
        standard_logging_object.get("output")
        if isinstance(standard_logging_object, dict)
        else None,
    ):
        for attr_name in (
            "_hidden_params",
            "hidden_params",
            "additional_headers",
            "_response_headers",
            "response_headers",
            "headers",
            "upstream_headers",
        ):
            attr_value = _maybe_get(candidate, attr_name)
            if attr_value is not None:
                roots.append(attr_value)
                additional_headers = _maybe_get(attr_value, "additional_headers")
                if additional_headers is not None:
                    roots.append(additional_headers)
    for key in (
        "rate_limits",
        "codex_rate_limits",
        "codex_token_count",
        "codex_response_headers",
        "anthropic_response_headers",
        "anthropic_rate_limit_headers",
        "xai_oauth_response_headers",
        "google_retrieve_user_quota",
        "google_generate_content_error",
        "google_user_quota",
        "gemini_model_status",
        "google_model_status",
    ):
        if key in metadata:
            roots.append(metadata.get(key))
    return [root for root in roots if root is not None]


def _extract_codex_rate_limit_observations(
    kwargs: Dict[str, Any],
    result: Any,
    observed_at: Any,
) -> List[Dict[str, Any]]:
    context = _build_rate_limit_context(kwargs, result, observed_at, "codex_token_count")
    observations: List[Dict[str, Any]] = []
    for candidate in _iter_rate_limit_dicts(*_rate_limit_candidate_roots(kwargs, result)):
        rate_limits = candidate.get("rate_limits")
        if not isinstance(rate_limits, dict):
            continue
        if not (
            isinstance(rate_limits.get("primary"), dict)
            or isinstance(rate_limits.get("secondary"), dict)
        ):
            continue
        limit_id = _clean_non_empty_string(rate_limits.get("limit_id"))
        limit_name = _clean_non_empty_string(rate_limits.get("limit_name"))
        for limit_scope in ("primary", "secondary"):
            window = rate_limits.get(limit_scope)
            if not isinstance(window, dict):
                continue
            window_minutes = _safe_int(window.get("window_minutes"))
            used_percentage = _safe_float(window.get("used_percent"))
            provider_resets_at = _parse_provider_timestamp(window.get("resets_at"))
            observations.append(
                _finalize_rate_limit_observation(
                    {
                        "observed_at": context["observed_at"],
                        "source": "codex_token_count",
                        "provider": "openai",
                        "client_family": "codex",
                        "limit_id": limit_id,
                        "limit_name": limit_name,
                        "limit_scope": limit_scope,
                        "window_minutes": window_minutes,
                        "provider_resets_at": provider_resets_at,
                        "used_percentage": used_percentage,
                        "exhausted": bool(
                            used_percentage is not None and used_percentage >= 100
                        ),
                        "exhaustion_kind": (
                            rate_limits.get("rate_limit_reached_type")
                            if rate_limits.get("rate_limit_reached_type")
                            else None
                        ),
                        "raw_provider_fields": {
                            "limit_id": limit_id,
                            "limit_name": limit_name,
                            "limit_scope": limit_scope,
                            "window_minutes": window.get("window_minutes"),
                            "used_percent": window.get("used_percent"),
                            "resets_at": window.get("resets_at"),
                            "plan_type": rate_limits.get("plan_type"),
                            "rate_limit_reached_type": rate_limits.get("rate_limit_reached_type"),
                        },
                        "evidence": {
                            "signals": ["provider_rate_limits"],
                            "provider_fields": [
                                f"rate_limits.{limit_scope}.used_percent",
                                f"rate_limits.{limit_scope}.window_minutes",
                                f"rate_limits.{limit_scope}.resets_at",
                            ],
                        },
                    },
                    context,
                )
            )
    return _dedupe_rate_limit_observations(observations)


def _extract_codex_header_rate_limit_observations(
    kwargs: Dict[str, Any],
    result: Any,
    observed_at: Any,
) -> List[Dict[str, Any]]:
    context = _build_rate_limit_context(
        kwargs,
        result,
        observed_at,
        "codex_response_headers",
    )
    observations: List[Dict[str, Any]] = []
    for candidate in _iter_rate_limit_dicts(*_rate_limit_candidate_roots(kwargs, result)):
        lower_headers = _rate_limit_header_map(candidate)
        source = str(candidate.get("source") or "").lower()
        has_codex_header = any(
            isinstance(key, str) and key.lower().startswith("x-codex-")
            for key in list(candidate.keys())
        )
        if not has_codex_header and source != "codex_response_headers":
            continue

        active_limit = _get_rate_limit_header_value(
            candidate,
            "x-codex-active-limit",
            lower_headers=lower_headers,
        )
        header_groups = [
            {
                "header_prefix": "x-codex",
                "limit_id": "codex",
                "limit_name": (
                    f"Codex {active_limit}" if active_limit else "Codex"
                ),
            }
        ]
        bengalfox_limit_name = _clean_non_empty_string(
            _get_rate_limit_header_value(
                candidate,
                "x-codex-bengalfox-limit-name",
                lower_headers=lower_headers,
            )
        )
        if bengalfox_limit_name:
            header_groups.append(
                {
                    "header_prefix": "x-codex-bengalfox",
                    "limit_id": "codex_bengalfox",
                    "limit_name": bengalfox_limit_name,
                }
            )

        for header_group in header_groups:
            header_prefix = header_group["header_prefix"]
            for limit_scope, window_minutes in (
                ("primary", 300),
                ("secondary", 10080),
            ):
                reset_key = f"{header_prefix}-{limit_scope}-reset-at"
                reset_after_key = f"{header_prefix}-{limit_scope}-reset-after-seconds"
                used_percent_key = f"{header_prefix}-{limit_scope}-used-percent"
                window_minutes_key = f"{header_prefix}-{limit_scope}-window-minutes"
                over_limit_key = (
                    f"{header_prefix}-{limit_scope}-over-secondary-limit-percent"
                )
                reset_value = _get_rate_limit_header_value(
            candidate, reset_key,
            lower_headers=lower_headers
        )
                reset_hint_seconds = _safe_int(
                    _get_rate_limit_header_value(
            candidate, reset_after_key,
            lower_headers=lower_headers
        )
                )
                used_percentage = _safe_float(
                    _get_rate_limit_header_value(
            candidate, used_percent_key,
            lower_headers=lower_headers
        )
                )
                raw_window_minutes = _get_rate_limit_header_value(
            candidate,
                    window_minutes_key,
            lower_headers=lower_headers
        )
                parsed_window_minutes = _safe_int(raw_window_minutes)
                if raw_window_minutes is not None and (
                    parsed_window_minutes is None or parsed_window_minutes <= 0
                ):
                    continue
                observed_window_minutes = (
                    parsed_window_minutes
                    or window_minutes
                )
                over_limit_percent = _safe_float(
                    _get_rate_limit_header_value(
            candidate, over_limit_key,
            lower_headers=lower_headers
        )
                )
                if (
                    reset_value is None
                    and reset_hint_seconds is None
                    and used_percentage is None
                    and over_limit_percent is None
                ):
                    continue
                provider_resets_at, stale_reset = _resolve_rate_limit_reset_at(
                    reset_value,
                    context["observed_at"],
                    reset_hint_seconds,
                )
                if stale_reset:
                    continue
                observations.append(
                    _finalize_rate_limit_observation(
                        {
                            "observed_at": context["observed_at"],
                            "source": "codex_response_headers",
                            "provider": "openai",
                            "client_family": "codex",
                            "limit_id": header_group["limit_id"],
                            "limit_name": header_group["limit_name"],
                            "limit_scope": limit_scope,
                            "window_minutes": observed_window_minutes,
                            "provider_resets_at": provider_resets_at,
                            "used_percentage": used_percentage,
                            "reset_hint_seconds": reset_hint_seconds,
                            "exhausted": (
                                (used_percentage is not None and used_percentage >= 100)
                                or (
                                    over_limit_percent is not None
                                    and over_limit_percent > 0
                                )
                            ),
                            "raw_provider_fields": {
                                reset_key: reset_value,
                                reset_after_key: _get_rate_limit_header_value(
            candidate,
                                    reset_after_key,
            lower_headers=lower_headers
        ),
                                over_limit_key: _get_rate_limit_header_value(
            candidate,
                                    over_limit_key,
            lower_headers=lower_headers
        ),
                                used_percent_key: _get_rate_limit_header_value(
            candidate,
                                    used_percent_key,
            lower_headers=lower_headers
        ),
                                window_minutes_key: _get_rate_limit_header_value(
            candidate,
                                    window_minutes_key,
            lower_headers=lower_headers
        ),
                                "x-codex-active-limit": _get_rate_limit_header_value(
            candidate,
                                    "x-codex-active-limit",
            lower_headers=lower_headers
        ),
                                "x-codex-credits-unlimited": _get_rate_limit_header_value(
            candidate,
                                    "x-codex-credits-unlimited",
            lower_headers=lower_headers
        ),
                            },
                            "evidence": {
                                "signals": ["codex_response_rate_limit_headers"],
                                "provider_fields": [
                                    reset_key,
                                    reset_after_key,
                                    used_percent_key,
                                    window_minutes_key,
                                    over_limit_key,
                                ],
                            },
                        },
                        context,
                    )
                )
    return _dedupe_rate_limit_observations(observations)


def _extract_error_payload_dicts(value: Any) -> List[Dict[str, Any]]:
    roots: List[Any] = [value, str(value)]
    for attr in (
        "detail",
        "body",
        "response",
        "message",
        "_hidden_params",
        "hidden_params",
        "additional_headers",
        "headers",
        "response_headers",
        "upstream_headers",
    ):
        try:
            attr_value = getattr(value, attr)
        except Exception:
            attr_value = None
        if attr_value is not None:
            roots.append(attr_value)
            if attr == "response":
                roots.extend(
                    [
                        getattr(attr_value, "text", None),
                        getattr(attr_value, "content", None),
                        getattr(attr_value, "headers", None),
                    ]
                )
    return _iter_rate_limit_dicts(*roots)


def _extract_codex_usage_limit_error_observations(
    kwargs: Dict[str, Any],
    result: Any,
    observed_at: Any,
) -> List[Dict[str, Any]]:
    context = _build_rate_limit_context(
        kwargs,
        result,
        observed_at,
        "codex_usage_limit_error",
    )
    observations: List[Dict[str, Any]] = []
    for candidate in _extract_error_payload_dicts(result) + _iter_rate_limit_dicts(
        *_rate_limit_candidate_roots(kwargs, result)
    ):
        error = candidate.get("error") if isinstance(candidate.get("error"), dict) else candidate
        if not isinstance(error, dict):
            continue
        error_type = _clean_non_empty_string(error.get("type")) or _clean_non_empty_string(
            error.get("code")
        )
        message = _clean_non_empty_string(error.get("message"))
        if error_type != "usage_limit_reached" and not (
            isinstance(message, str) and "usage limit" in message.lower()
        ):
            continue
        reset_hint_seconds = _parse_reset_hint_seconds(
            error.get("resets_in_seconds"),
            message,
        )
        provider_resets_at = _parse_provider_timestamp(error.get("resets_at"))
        if provider_resets_at is None and reset_hint_seconds is not None:
            provider_resets_at = context["observed_at"] + timedelta(
                seconds=reset_hint_seconds
            )
        limit_name = (
            _clean_non_empty_string(error.get("limit_name"))
            or _clean_non_empty_string(context.get("model"))
            or "codex"
        )
        observations.append(
            _finalize_rate_limit_observation(
                {
                    "observed_at": context["observed_at"],
                    "source": "codex_usage_limit_error",
                    "provider": "openai",
                    "client_family": "codex",
                    "limit_id": _clean_non_empty_string(error.get("limit_id")),
                    "limit_name": limit_name,
                    "limit_scope": _clean_non_empty_string(
                        error.get("rate_limit_reached_type")
                    )
                    or "usage_limit",
                    "provider_resets_at": provider_resets_at,
                    "used_percentage": 100.0,
                    "status": "exhausted",
                    "exhausted": True,
                    "exhaustion_kind": "usage_limit_reached",
                    "reset_hint_seconds": reset_hint_seconds,
                    "raw_provider_fields": {
                        "type": error_type,
                        "message": message,
                        "plan_type": error.get("plan_type"),
                        "resets_at": error.get("resets_at"),
                        "resets_in_seconds": error.get("resets_in_seconds"),
                        "rate_limit_reached_type": error.get("rate_limit_reached_type"),
                    },
                    "evidence": {
                        "signals": ["usage_limit_error"],
                        "provider_fields": [
                            "error.type",
                            "error.resets_at",
                            "error.resets_in_seconds",
                        ],
                    },
                },
                context,
            )
        )
    return _dedupe_rate_limit_observations(observations)


def _rate_limit_header_map(candidate: Dict[str, Any]) -> Dict[str, Any]:
    """Lowercase header keys once per candidate for repeated lookups."""
    return {
        str(key).lower(): value
        for key, value in list(candidate.items())
        if isinstance(key, str)
    }


def _get_rate_limit_header_value(
    candidate: Dict[str, Any],
    *header_names: str,
    lower_headers: Optional[Dict[str, Any]] = None,
) -> Any:
    if lower_headers is None:
        lower_headers = _rate_limit_header_map(candidate)
    for header_name in header_names:
        normalized_header_name = header_name.lower()
        for candidate_name in (
            normalized_header_name,
            f"llm_provider-{normalized_header_name}",
        ):
            value = lower_headers.get(candidate_name)
            if value is not None:
                return value
    return None


def _looks_like_claude_rate_limit_context(context: Dict[str, Any]) -> bool:
    metadata = context.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
    context_text = " ".join(
        str(value)
        for value in (
            context.get("client_name"),
            context.get("client_user_agent"),
            context.get("route_family"),
            metadata.get("trace_name"),
            metadata.get("client_name"),
            metadata.get("cc_version"),
            metadata.get("cc_entrypoint"),
        )
        if value is not None
    ).lower()
    return "claude" in context_text or "cc_version" in metadata


def _extract_anthropic_header_rate_limit_observations(
    kwargs: Dict[str, Any],
    result: Any,
    observed_at: Any,
) -> List[Dict[str, Any]]:
    context = _build_rate_limit_context(
        kwargs,
        result,
        observed_at,
        "anthropic_response_headers",
    )
    observations: List[Dict[str, Any]] = []
    client_family = (
        "claude" if _looks_like_claude_rate_limit_context(context) else "anthropic"
    )
    for candidate in _iter_rate_limit_dicts(*_rate_limit_candidate_roots(kwargs, result)):
        lower_headers = _rate_limit_header_map(candidate)
        source = str(candidate.get("source") or "").lower()
        has_anthropic_header = any(
            isinstance(key, str)
            and (
                key.lower().startswith("anthropic-ratelimit-")
                or key.lower().startswith("llm_provider-anthropic-ratelimit-")
            )
            for key in list(candidate.keys())
        )
        if not has_anthropic_header and source != "anthropic_response_headers":
            continue
        for limit_scope, display_name, window_minutes in (
            ("5h", "Anthropic unified 5h", 300),
            ("7d", "Anthropic unified 7d", 10080),
            ("7d_oi", "Anthropic unified 7d overage included", 10080),
            ("7d_sonnet", "Anthropic unified 7d Sonnet", 10080),
        ):
            reset_key = f"anthropic-ratelimit-unified-{limit_scope}-reset"
            status_key = f"anthropic-ratelimit-unified-{limit_scope}-status"
            utilization_key = (
                f"anthropic-ratelimit-unified-{limit_scope}-utilization"
            )
            threshold_key = (
                f"anthropic-ratelimit-unified-{limit_scope}-surpassed-threshold"
            )
            reset_value = _get_rate_limit_header_value(
            candidate, reset_key,
            lower_headers=lower_headers
        )
            status_value = _clean_non_empty_string(
                _get_rate_limit_header_value(
            candidate, status_key,
            lower_headers=lower_headers
        )
            )
            utilization = _safe_float(
                _get_rate_limit_header_value(
            candidate, utilization_key,
            lower_headers=lower_headers
        )
            )
            threshold = _safe_float(
                _get_rate_limit_header_value(
            candidate, threshold_key,
            lower_headers=lower_headers
        )
            )
            if reset_value is None and status_value is None and utilization is None:
                continue
            provider_resets_at, stale_reset = _resolve_rate_limit_reset_at(
                reset_value,
                context["observed_at"],
            )
            if stale_reset:
                continue
            used_percentage = (
                utilization * 100
                if utilization is not None and utilization <= 1
                else utilization
            )
            observations.append(
                _finalize_rate_limit_observation(
                    {
                        "observed_at": context["observed_at"],
                        "source": "anthropic_response_headers",
                        "provider": "anthropic",
                        "client_family": client_family,
                        "limit_id": f"anthropic_unified_{limit_scope}",
                        "limit_name": display_name,
                        "limit_scope": limit_scope,
                        "window_minutes": window_minutes,
                        "provider_resets_at": provider_resets_at,
                        "used_percentage": used_percentage,
                        "status": status_value,
                        "exhausted": status_value in {"rejected", "exhausted"},
                        "raw_provider_fields": {
                            reset_key: reset_value,
                            status_key: status_value,
                            utilization_key: _get_rate_limit_header_value(
            candidate,
                                utilization_key,
            lower_headers=lower_headers
        ),
                            threshold_key: _get_rate_limit_header_value(
            candidate,
                                threshold_key,
            lower_headers=lower_headers
        ),
                            "surpassed_threshold": threshold,
                            "anthropic-ratelimit-unified-representative-claim": _get_rate_limit_header_value(
            candidate,
                                "anthropic-ratelimit-unified-representative-claim",
            lower_headers=lower_headers
        ),
                            "anthropic-ratelimit-unified-overage-status": _get_rate_limit_header_value(
            candidate,
                                "anthropic-ratelimit-unified-overage-status",
            lower_headers=lower_headers
        ),
                        },
                        "evidence": {
                            "signals": ["anthropic_unified_rate_limit_headers"],
                            "provider_fields": [
                                reset_key,
                                status_key,
                                utilization_key,
                                threshold_key,
                            ],
                        },
                    },
                    context,
                )
            )
        for limit_scope, total_key, remaining_key, reset_key in (
            (
                "requests",
                "anthropic-ratelimit-requests-limit",
                "anthropic-ratelimit-requests-remaining",
                "anthropic-ratelimit-requests-reset",
            ),
            (
                "tokens",
                "anthropic-ratelimit-tokens-limit",
                "anthropic-ratelimit-tokens-remaining",
                "anthropic-ratelimit-tokens-reset",
            ),
        ):
            total = _safe_int(_get_rate_limit_header_value(
            candidate, total_key,
            lower_headers=lower_headers
        ))
            remaining = _safe_int(
                _get_rate_limit_header_value(
            candidate, remaining_key,
            lower_headers=lower_headers
        )
            )
            reset_value = _get_rate_limit_header_value(
            candidate, reset_key,
            lower_headers=lower_headers
        )
            if total is None and remaining is None and reset_value is None:
                continue
            provider_resets_at, stale_reset = _resolve_rate_limit_reset_at(
                reset_value,
                context["observed_at"],
            )
            if stale_reset:
                continue
            used = (
                max(0, total - remaining)
                if total is not None and remaining is not None
                else None
            )
            used_percentage = (
                (used / total) * 100
                if used is not None and total is not None and total > 0
                else None
            )
            observations.append(
                _finalize_rate_limit_observation(
                    {
                        "observed_at": context["observed_at"],
                        "source": "anthropic_response_headers",
                        "provider": "anthropic",
                        "client_family": client_family,
                        "limit_id": f"anthropic_{limit_scope}",
                        "limit_name": f"Anthropic {limit_scope} rate limit",
                        "limit_scope": limit_scope,
                        "provider_resets_at": provider_resets_at,
                        "used_percentage": used_percentage,
                        "remaining_requests": remaining,
                        "used_requests": used,
                        "total_requests": total,
                        "raw_provider_fields": {
                            total_key: _get_rate_limit_header_value(
            candidate, total_key,
            lower_headers=lower_headers
        ),
                            remaining_key: _get_rate_limit_header_value(
            candidate,
                                remaining_key,
            lower_headers=lower_headers
        ),
                            reset_key: reset_value,
                        },
                        "evidence": {
                            "signals": ["anthropic_response_rate_limit_headers"],
                            "provider_fields": [
                                total_key,
                                remaining_key,
                                reset_key,
                            ],
                        },
                    },
                    context,
                )
            )
    return _dedupe_rate_limit_observations(observations)


def _first_quota_number(candidate: Dict[str, Any], *keys: str) -> Optional[int]:
    for key in keys:
        value = _safe_int(candidate.get(key))
        if value is not None:
            return value
    return None


def _first_quota_float(candidate: Dict[str, Any], *keys: str) -> Optional[float]:
    for key in keys:
        value = _safe_float(candidate.get(key))
        if value is not None:
            return value
    return None


def _looks_like_xai_oauth_rate_limit_context(context: Dict[str, Any]) -> bool:
    metadata = context.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
    credential_family = str(metadata.get("credential_family") or "").lower()
    route_family = str(
        metadata.get("passthrough_route_family")
        or metadata.get("route_family")
        or context.get("route_family")
        or ""
    ).lower()
    model = str(context.get("model") or "").lower()
    request_model = str(context.get("request_model") or "").lower()
    return (
        credential_family == "xai_oauth"
        or metadata.get("xai_oauth_managed") is True
        or metadata.get("xai_oauth_public_model") is not None
        or "xai_oauth" in route_family
        or model.startswith("oa_xai/")
        or request_model.startswith("oa_xai/")
    )


def _extract_xai_oauth_account_hash(metadata: Dict[str, Any]) -> Optional[str]:
    for key in ("xai_oauth_account_hash", "provider_account_hash"):
        value = _clean_non_empty_string(metadata.get(key))
        if value:
            return value
    for key in (
        "xai_oauth_account_id",
        "provider_account_id",
        "organization_id",
        "org_id",
    ):
        value = _clean_non_empty_string(metadata.get(key))
        if value:
            return _short_hash(value.encode("utf-8"))
    return None


def _xai_oauth_header_remaining_pct(
    total: Optional[int],
    remaining: Optional[int],
) -> Optional[float]:
    if total is None or remaining is None or total <= 0:
        return None
    return round(max(0.0, min(100.0, (remaining / total) * 100.0)), 3)


def _next_utc_month_start(value: Any) -> Optional[datetime]:
    observed_dt = _normalize_datetime(value)
    if observed_dt is None:
        return None
    observed_dt = observed_dt.astimezone(timezone.utc)
    if observed_dt.month == 12:
        return datetime(observed_dt.year + 1, 1, 1, tzinfo=timezone.utc)
    return datetime(observed_dt.year, observed_dt.month + 1, 1, tzinfo=timezone.utc)


def _is_xai_oauth_subscription_quota_context(metadata: Dict[str, Any]) -> bool:
    quota_family = str(
        metadata.get("xai_quota_family") or metadata.get("shared_quota_family") or ""
    ).strip().lower()
    return (
        quota_family == "xai_grok_subscription"
        or metadata.get("grok_subscription_quota_shared") is True
    )


def _extract_xai_oauth_billing_period_end(
    *,
    candidate: Dict[str, Any],
    metadata: Dict[str, Any],
    observed_at: Any,
) -> Tuple[Optional[datetime], Optional[str]]:
    for source, value in (
        ("payload_billing_period_end", candidate.get("billingPeriodEnd")),
        (
            "payload_config_billing_period_end",
            _maybe_get_path(candidate, "config", "billingPeriodEnd"),
        ),
        ("metadata_billing_period_end", metadata.get("billingPeriodEnd")),
        (
            "metadata_xai_oauth_billing_period_end",
            metadata.get("xai_oauth_billing_period_end"),
        ),
    ):
        parsed = _parse_provider_timestamp(value)
        if parsed is not None:
            return parsed, source

    if _is_xai_oauth_subscription_quota_context(metadata):
        fallback = _next_utc_month_start(observed_at)
        if fallback is not None:
            return fallback, "xai_grok_subscription_month_boundary"

    return None, None


def _extract_xai_oauth_header_rate_limit_observations(
    kwargs: Dict[str, Any],
    result: Any,
    observed_at: Any,
) -> List[Dict[str, Any]]:
    context = _build_rate_limit_context(
        kwargs,
        result,
        observed_at,
        "xai_oauth_response_headers",
    )
    if context.get("provider") != "xai" or not _looks_like_xai_oauth_rate_limit_context(
        context
    ):
        return []
    raw_metadata = context.get("metadata")
    metadata: Dict[str, Any] = raw_metadata if isinstance(raw_metadata, dict) else {}
    account_hash = _extract_xai_oauth_account_hash(metadata)
    model = _clean_non_empty_string(metadata.get("xai_oauth_public_model")) or (
        _clean_non_empty_string(context.get("model"))
        if context.get("model") != "unknown"
        else None
    )
    observations: List[Dict[str, Any]] = []
    for candidate in _iter_rate_limit_dicts(*_rate_limit_candidate_roots(kwargs, result)):
        lower_headers = _rate_limit_header_map(candidate)
        source = str(candidate.get("source") or "").lower()
        has_xai_header = any(
            isinstance(key, str) and key.lower().startswith("x-ratelimit-")
            for key in list(candidate.keys())
        )
        if not has_xai_header and source != "xai_oauth_response_headers":
            continue

        for limit_scope, total_key, remaining_key, reset_keys in (
            (
                "requests",
                "x-ratelimit-limit-requests",
                "x-ratelimit-remaining-requests",
                (
                    "x-ratelimit-reset-requests",
                    "x-ratelimit-reset-request",
                    "x-ratelimit-reset",
                ),
            ),
            (
                "tokens",
                "x-ratelimit-limit-tokens",
                "x-ratelimit-remaining-tokens",
                (
                    "x-ratelimit-reset-tokens",
                    "x-ratelimit-reset-token",
                    "x-ratelimit-reset",
                ),
            ),
        ):
            total = _safe_int(_get_rate_limit_header_value(
            candidate, total_key,
            lower_headers=lower_headers
        ))
            remaining = _safe_int(_get_rate_limit_header_value(
            candidate, remaining_key,
            lower_headers=lower_headers
        ))
            reset_value = _get_rate_limit_header_value(
            candidate, *reset_keys,
            lower_headers=lower_headers
        )
            reset_hint_seconds = _parse_reset_hint_seconds(
                _get_rate_limit_header_value(
            candidate, "retry-after",
            lower_headers=lower_headers
        )
            )
            if (
                total is None
                and remaining is None
                and reset_value is None
                and reset_hint_seconds is None
            ):
                continue
            if total is not None and total <= 0:
                continue
            provider_resets_at, stale_reset = _resolve_rate_limit_reset_at(
                reset_value,
                context["observed_at"],
                reset_hint_seconds,
            )
            if stale_reset:
                continue
            reset_source = "response_header" if provider_resets_at is not None else None
            if provider_resets_at is None and reset_hint_seconds is None:
                provider_resets_at, reset_source = _extract_xai_oauth_billing_period_end(
                    candidate=candidate,
                    metadata=metadata,
                    observed_at=context["observed_at"],
                )
            elif reset_hint_seconds is not None and provider_resets_at is not None:
                reset_source = "retry_after"
            used = (
                max(0, total - remaining)
                if total is not None and remaining is not None
                else None
            )
            remaining_pct = _xai_oauth_header_remaining_pct(total, remaining)
            used_percentage = (
                round(max(0.0, min(100.0, 100.0 - remaining_pct)), 3)
                if remaining_pct is not None
                else None
            )
            exhausted = remaining is not None and remaining <= 0
            observations.append(
                _finalize_rate_limit_observation(
                    {
                        "observed_at": context["observed_at"],
                        "source": "xai_oauth_response_headers",
                        "provider": "xai",
                        "client_family": "xai_oauth",
                        "account_hash": account_hash,
                        "limit_id": f"xai_oauth_{limit_scope}",
                        "limit_name": f"xAI OAuth {limit_scope} rate limit",
                        "limit_scope": limit_scope,
                        "quota_period": (
                            "monthly"
                            if reset_source
                            in {
                                "payload_billing_period_end",
                                "payload_config_billing_period_end",
                                "metadata_billing_period_end",
                                "metadata_xai_oauth_billing_period_end",
                                "xai_grok_subscription_month_boundary",
                            }
                            else None
                        ),
                        "quota_type": limit_scope,
                        "provider_resets_at": provider_resets_at,
                        "remaining_pct": remaining_pct,
                        "quota_limit": float(total) if total is not None else None,
                        "quota_used": float(used) if used is not None else None,
                        "quota_remaining": (
                            float(remaining) if remaining is not None else None
                        ),
                        "billing_period_end_at": provider_resets_at
                        if reset_source
                        in {
                            "payload_billing_period_end",
                            "payload_config_billing_period_end",
                            "metadata_billing_period_end",
                            "metadata_xai_oauth_billing_period_end",
                            "xai_grok_subscription_month_boundary",
                        }
                        else None,
                        "used_percentage": used_percentage,
                        "remaining_requests": remaining,
                        "used_requests": used,
                        "total_requests": total,
                        "status": "quota_exhausted" if exhausted else "observed",
                        "exhausted": exhausted,
                        "exhaustion_kind": "rate_limit" if exhausted else None,
                        "reset_hint_seconds": reset_hint_seconds,
                        "model": model,
                        "model_family": "grok",
                        "raw_provider_fields": {
                            total_key: _get_rate_limit_header_value(
            candidate, total_key,
            lower_headers=lower_headers
        ),
                            remaining_key: _get_rate_limit_header_value(
            candidate,
                                remaining_key,
            lower_headers=lower_headers
        ),
                            "reset": reset_value,
                            "retry-after": _get_rate_limit_header_value(
            candidate,
                                "retry-after",
            lower_headers=lower_headers
        ),
                            "billingPeriodEnd": _json_safe_rate_limit_value(
                                _maybe_get_path(candidate, "config", "billingPeriodEnd")
                                or candidate.get("billingPeriodEnd")
                                or metadata.get("xai_oauth_billing_period_end")
                                or metadata.get("billingPeriodEnd")
                            ),
                            "quota_unit": f"xai_oauth_{limit_scope}",
                            "quota_unit_interpretation": limit_scope,
                        },
                        "evidence": {
                            "signals": ["xai_oauth_response_rate_limit_headers"],
                            "provider_fields": [
                                total_key,
                                remaining_key,
                                *reset_keys,
                                "retry-after",
                            ],
                            "reset_absent": provider_resets_at is None,
                            "reset_header_absent": (
                                reset_value is None and reset_hint_seconds is None
                            ),
                            "reset_source": reset_source,
                        },
                    },
                    context,
                )
            )
    return _dedupe_rate_limit_observations(observations)


USAGE_PERIOD_TYPE_WEEKLY = "USAGE_PERIOD_TYPE_WEEKLY"
GROK_BILLING_WEEKLY_CREDITS_QUOTA_KEY = "xai_grok_build_weekly_credits:credits"
GROK_BILLING_MONTHLY_REQUESTS_QUOTA_KEY = "xai_grok_build_monthly_requests:requests"
GROK_BILLING_MONTHLY_CREDITS_QUOTA_KEY = "xai_grok_build_monthly_credits:credits"


def _grok_billing_quota_value(value: Any) -> Optional[float]:
    if isinstance(value, dict):
        value = value.get("val")
    return _safe_float(value)


def _grok_billing_current_period(config: Dict[str, Any]) -> Dict[str, Any]:
    current_period = config.get("currentPeriod")
    return current_period if isinstance(current_period, dict) else {}


def _grok_billing_is_weekly_period(config: Dict[str, Any]) -> bool:
    current_period = _grok_billing_current_period(config)
    period_type = str(current_period.get("type") or "").strip()
    return period_type == USAGE_PERIOD_TYPE_WEEKLY


def _grok_billing_period_bounds(
    config: Dict[str, Any],
) -> Tuple[Optional[datetime], Optional[datetime]]:
    current_period = _grok_billing_current_period(config)
    billing_period_start_at = _parse_provider_timestamp(
        config.get("billingPeriodStart") or current_period.get("start")
    )
    billing_period_end_at = _parse_provider_timestamp(
        config.get("billingPeriodEnd") or current_period.get("end")
    )
    return billing_period_start_at, billing_period_end_at


def _is_grok_billing_context(
    kwargs: Dict[str, Any],
    metadata: Dict[str, Any],
) -> bool:
    route_text = " ".join(
        str(value)
        for value in (
            metadata.get("passthrough_route_family"),
            metadata.get("user_api_key_request_route"),
            metadata.get("api_base"),
            _maybe_get_path(kwargs.get("standard_pass_through_logging_payload"), "url"),
            _maybe_get_path(kwargs.get("passthrough_logging_payload"), "url"),
        )
        if value is not None
    ).lower()
    if "/billing" in route_text and (
        "grok" in route_text or "xai" in route_text or "x.ai" in route_text
    ):
        return True
    if (
        metadata.get("grok_cli_chat_proxy") is True
        or metadata.get("xai_cli_chat_proxy") is True
    ):
        return True
    headers = _extract_headers_from_kwargs(kwargs)
    return any(
        header_name.startswith("x-grok-") or header_name == "x-xai-token-auth"
        for header_name in headers
    )


def _extract_grok_billing_config(candidate: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    config = (
        candidate.get("config") if isinstance(candidate.get("config"), dict) else candidate
    )
    if not isinstance(config, dict):
        return None
    has_absolute_quota = isinstance(config.get("monthlyLimit"), dict) and isinstance(
        config.get("used"),
        dict,
    )
    has_percentage_quota = _safe_float(config.get("creditUsagePercent")) is not None
    has_weekly_period = _grok_billing_is_weekly_period(config)
    billing_period_start_at, billing_period_end_at = _grok_billing_period_bounds(config)
    has_period_bounds = (
        billing_period_start_at is not None or billing_period_end_at is not None
    )
    if (
        not has_absolute_quota
        and not has_percentage_quota
        and not (has_weekly_period and has_period_bounds)
    ):
        return None
    return config


def _grok_billing_model(
    context: Dict[str, Any],
    metadata: Dict[str, Any],
) -> str:
    return (
        _clean_non_empty_string(context.get("model"))
        if context.get("model") != "unknown"
        else None
    ) or _clean_non_empty_string(
        metadata.get("grok_model_override")
    ) or "grok-build"


def _grok_billing_request_contract_evidence(
    metadata: Dict[str, Any],
) -> Dict[str, Any]:
    fingerprint = _clean_non_empty_string(
        metadata.get("grok_billing_passthrough_request_contract_fingerprint")
    )
    if not fingerprint:
        return {}

    evidence: Dict[str, Any] = {
        "request_contract_fingerprint": fingerprint,
    }
    for metadata_key, evidence_key in (
        ("grok_billing_passthrough_http_client", "request_contract_http_client"),
        ("grok_billing_passthrough_request_method", "request_contract_method"),
        ("grok_billing_passthrough_target_host", "request_contract_target_host"),
        ("grok_billing_passthrough_target_path", "request_contract_target_path"),
        ("grok_billing_passthrough_user_agent", "request_contract_user_agent"),
    ):
        value = _clean_non_empty_string(metadata.get(metadata_key))
        if value:
            evidence[evidence_key] = value

    for metadata_key, evidence_key in (
        ("grok_billing_passthrough_query_keys", "request_contract_query_keys"),
        ("grok_billing_passthrough_header_names", "request_contract_header_names"),
    ):
        value = metadata.get(metadata_key)
        if isinstance(value, list):
            evidence[evidence_key] = [
                str(item)
                for item in value
                if isinstance(item, (str, int, float)) and str(item)
            ]

    configured = metadata.get("grok_billing_passthrough_x_xai_token_auth_configured")
    if configured is not None:
        evidence["request_contract_x_xai_token_auth_configured"] = bool(configured)

    return evidence


def _grok_billing_snapshot_parts(
    config: Dict[str, Any],
    *,
    base_evidence: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    if not isinstance(config, dict):
        return None

    billing_period_start_at, billing_period_end_at = _grok_billing_period_bounds(config)
    is_weekly = _grok_billing_is_weekly_period(config)
    monthly_limit = _grok_billing_quota_value(config.get("monthlyLimit"))
    used = _grok_billing_quota_value(config.get("used"))
    credit_usage_percent = _safe_float(config.get("creditUsagePercent"))

    evidence: Dict[str, Any] = dict(base_evidence or {})
    signals = list(evidence.get("signals") or [])
    if "grok_billing_payload" not in signals:
        signals.append("grok_billing_payload")
    evidence["signals"] = signals

    if monthly_limit is not None and monthly_limit > 0 and used is not None and used >= 0:
        used_percentage = max(0.0, min(100.0, (used / monthly_limit) * 100.0))
        remaining_pct = float(
            int(math.floor(max(0.0, min(100.0, 100.0 - used_percentage)) + 0.5))
        )
        quota_remaining = max(0.0, monthly_limit - used)
        if "grok_billing_monthly_counter" not in signals:
            signals.append("grok_billing_monthly_counter")
        return {
            "quota_key": GROK_BILLING_MONTHLY_REQUESTS_QUOTA_KEY,
            "quota_period": "monthly",
            "quota_type": "requests",
            "limit_id": "xai_grok_build_monthly_requests",
            "limit_name": "Grok Build monthly requests",
            "limit_scope": "requests",
            "remaining_pct": remaining_pct,
            "used_percentage": float(100.0 - remaining_pct),
            "quota_limit": monthly_limit,
            "quota_used": used,
            "quota_remaining": quota_remaining,
            "billing_period_start_at": billing_period_start_at,
            "billing_period_end_at": billing_period_end_at,
            "raw_provider_fields": {
                "monthlyLimit": _json_safe_rate_limit_value(config.get("monthlyLimit")),
                "used": _json_safe_rate_limit_value(config.get("used")),
                "onDemandCap": _json_safe_rate_limit_value(config.get("onDemandCap")),
                "billingPeriodStart": config.get("billingPeriodStart"),
                "billingPeriodEnd": config.get("billingPeriodEnd"),
                "quota_unit": "grok_billing_used",
                "quota_unit_interpretation": "requests",
            },
            "evidence": {
                **evidence,
                "provider_fields": [
                    "config.monthlyLimit.val",
                    "config.used.val",
                    "config.billingPeriodEnd",
                ],
                "rounding": "whole_remaining_percentage",
                "unit_note": (
                    "Grok billing does not label used.val; observed tool "
                    "traffic behaves request-like."
                ),
            },
        }

    if is_weekly and credit_usage_percent is not None:
        used_percentage = max(0.0, min(100.0, credit_usage_percent))
        remaining_pct = max(0.0, min(100.0, 100.0 - used_percentage))
        for signal in (
            "grok_billing_weekly_credit",
            "grok_billing_percentage_only",
        ):
            if signal not in signals:
                signals.append(signal)
        return {
            "quota_key": GROK_BILLING_WEEKLY_CREDITS_QUOTA_KEY,
            "quota_period": "weekly",
            "quota_type": "credits",
            "limit_id": "xai_grok_build_weekly_credits",
            "limit_name": "Grok Build weekly credits",
            "limit_scope": "credits",
            "remaining_pct": remaining_pct,
            "used_percentage": used_percentage,
            "quota_limit": None,
            "quota_used": None,
            "quota_remaining": None,
            "billing_period_start_at": billing_period_start_at,
            "billing_period_end_at": billing_period_end_at,
            "raw_provider_fields": {
                "creditUsagePercent": _json_safe_rate_limit_value(
                    config.get("creditUsagePercent")
                ),
                "productUsage": _json_safe_rate_limit_value(config.get("productUsage")),
                "currentPeriod": _json_safe_rate_limit_value(config.get("currentPeriod")),
                "billingPeriodStart": config.get("billingPeriodStart"),
                "billingPeriodEnd": config.get("billingPeriodEnd"),
                "quota_unit": "grok_billing_credit_usage_percent",
                "quota_unit_interpretation": "percent_of_credit_quota",
            },
            "evidence": {
                **evidence,
                "provider_fields": [
                    "config.creditUsagePercent",
                    "config.productUsage",
                    "config.currentPeriod.type",
                    "config.billingPeriodEnd",
                ],
                "rounding": "none",
                "unit_note": (
                    "Grok billing provided percentage-only weekly credit usage; "
                    "absolute quota counts are intentionally left null."
                ),
            },
        }

    if is_weekly and (
        billing_period_start_at is not None or billing_period_end_at is not None
    ):
        if "grok_billing_weekly_fresh_period" not in signals:
            signals.append("grok_billing_weekly_fresh_period")
        return {
            "quota_key": GROK_BILLING_WEEKLY_CREDITS_QUOTA_KEY,
            "quota_period": "weekly",
            "quota_type": "credits",
            "limit_id": "xai_grok_build_weekly_credits",
            "limit_name": "Grok Build weekly credits",
            "limit_scope": "credits",
            "remaining_pct": 100.0,
            "used_percentage": 0.0,
            "quota_limit": None,
            "quota_used": None,
            "quota_remaining": None,
            "billing_period_start_at": billing_period_start_at,
            "billing_period_end_at": billing_period_end_at,
            "raw_provider_fields": {
                "currentPeriod": _json_safe_rate_limit_value(config.get("currentPeriod")),
                "billingPeriodStart": config.get("billingPeriodStart"),
                "billingPeriodEnd": config.get("billingPeriodEnd"),
                "quota_unit": "grok_billing_weekly_credit_fresh_period",
                "quota_unit_interpretation": "percent_of_credit_quota",
            },
            "evidence": {
                **evidence,
                "provider_fields": [
                    "config.currentPeriod.type",
                    "config.currentPeriod.start",
                    "config.currentPeriod.end",
                    "config.billingPeriodEnd",
                ],
                "rounding": "none",
                "unit_note": (
                    "Fresh weekly Grok Build credit periods omit creditUsagePercent; "
                    "remaining percent is inferred as 100% used / 0% consumed."
                ),
            },
        }

    if credit_usage_percent is not None:
        used_percentage = max(0.0, min(100.0, credit_usage_percent))
        remaining_pct = max(0.0, min(100.0, 100.0 - used_percentage))
        if "grok_billing_percentage_only" not in signals:
            signals.append("grok_billing_percentage_only")
        if "grok_billing_legacy_monthly_credit" not in signals:
            signals.append("grok_billing_legacy_monthly_credit")
        return {
            "quota_key": GROK_BILLING_MONTHLY_CREDITS_QUOTA_KEY,
            "quota_period": "monthly",
            "quota_type": "credits",
            "limit_id": "xai_grok_build_monthly_credits",
            "limit_name": "Grok Build monthly credits",
            "limit_scope": "credits",
            "remaining_pct": remaining_pct,
            "used_percentage": used_percentage,
            "quota_limit": None,
            "quota_used": None,
            "quota_remaining": None,
            "billing_period_start_at": billing_period_start_at,
            "billing_period_end_at": billing_period_end_at,
            "raw_provider_fields": {
                "creditUsagePercent": _json_safe_rate_limit_value(
                    config.get("creditUsagePercent")
                ),
                "productUsage": _json_safe_rate_limit_value(config.get("productUsage")),
                "billingPeriodStart": config.get("billingPeriodStart"),
                "billingPeriodEnd": config.get("billingPeriodEnd"),
                "quota_unit": "grok_billing_credit_usage_percent",
                "quota_unit_interpretation": "percent_of_credit_quota",
            },
            "evidence": {
                **evidence,
                "provider_fields": [
                    "config.creditUsagePercent",
                    "config.productUsage",
                    "config.billingPeriodEnd",
                ],
                "rounding": "none",
                "unit_note": (
                    "Grok billing provided percentage-only credit usage; "
                    "absolute quota counts are intentionally left null."
                ),
            },
        }

    return None


def _extract_grok_billing_observations(
    kwargs: Dict[str, Any],
    result: Any,
    observed_at: Any,
) -> List[Dict[str, Any]]:
    metadata = _merged_rate_limit_metadata(kwargs)
    if not _is_grok_billing_context(kwargs, metadata):
        return []

    context = _build_rate_limit_context(
        kwargs,
        result,
        observed_at,
        "grok_billing",
    )
    request_contract_evidence = _grok_billing_request_contract_evidence(metadata)
    observations: List[Dict[str, Any]] = []
    for candidate in _iter_rate_limit_dicts(*_rate_limit_candidate_roots(kwargs, result)):
        config = _extract_grok_billing_config(candidate)
        if config is None:
            continue

        snapshot = _grok_billing_snapshot_parts(
            config,
            base_evidence=request_contract_evidence,
        )
        if snapshot is None:
            continue

        model = _grok_billing_model(context, metadata)
        provider_resets_at = snapshot.get("billing_period_end_at")
        observations.append(
            _finalize_rate_limit_observation(
                {
                    "observed_at": context["observed_at"],
                    "source": "grok_billing",
                    "provider": "xai",
                    "client_family": "grok-build",
                    "limit_id": snapshot["limit_id"],
                    "limit_name": snapshot["limit_name"],
                    "limit_scope": snapshot["limit_scope"],
                    "quota_period": snapshot["quota_period"],
                    "quota_type": snapshot["quota_type"],
                    "provider_resets_at": provider_resets_at,
                    "remaining_pct": snapshot["remaining_pct"],
                    "quota_limit": snapshot["quota_limit"],
                    "quota_used": snapshot["quota_used"],
                    "quota_remaining": snapshot["quota_remaining"],
                    "billing_period_start_at": snapshot["billing_period_start_at"],
                    "billing_period_end_at": snapshot["billing_period_end_at"],
                    "used_percentage": snapshot["used_percentage"],
                    "model": model,
                    "model_family": "grok",
                    "raw_provider_fields": snapshot["raw_provider_fields"],
                    "evidence": snapshot["evidence"],
                },
                context,
            )
        )
    return _dedupe_rate_limit_observations(observations)


def _extract_openrouter_free_error_reset_at(
    kwargs: Dict[str, Any],
    result: Any,
    dicts: List[Dict[str, Any]],
    error_text: str,
    observed_at: datetime,
) -> Tuple[datetime, Optional[int]]:
    headers = _extract_headers_from_kwargs(kwargs)
    headers.update(_extract_provider_error_headers(result))
    retry_after_seconds = _extract_provider_error_retry_after_seconds(
        kwargs=kwargs,
        result=result,
        dicts=dicts,
        error_text=error_text,
    )
    reset_hint_seconds = (
        int(retry_after_seconds)
        if retry_after_seconds is not None and retry_after_seconds >= 0
        else None
    )
    reset_value = _first_non_empty_string(
        headers.get("x-ratelimit-reset"),
        headers.get("x-rate-limit-reset"),
        headers.get("x-ratelimit-reset-at"),
        headers.get("x-rate-limit-reset-at"),
    )
    provider_resets_at, stale_reset = _resolve_rate_limit_reset_at(
        reset_value,
        observed_at,
        reset_hint_seconds,
    )
    if provider_resets_at is not None and not stale_reset:
        return provider_resets_at, reset_hint_seconds
    _day_start, day_end = _openrouter_free_daily_window(observed_at)
    return day_end, reset_hint_seconds


def _extract_openrouter_free_error_observations(
    kwargs: Dict[str, Any],
    result: Any,
    observed_at: Any,
) -> List[Dict[str, Any]]:
    context = _build_rate_limit_context(
        kwargs,
        result,
        observed_at,
        _AAWM_OPENROUTER_FREE_DAILY_SOURCE,
    )
    if context.get("provider") != "openrouter":
        return []
    model_candidates = (
        context.get("model"),
        context.get("request_model"),
        _maybe_get_path(
            kwargs.get("passthrough_logging_payload"),
            "request_body",
            "model",
        ),
        _maybe_get_path(
            kwargs.get("litellm_params"),
            "proxy_server_request",
            "body",
            "model",
        ),
    )
    if not any(_is_openrouter_free_model(model) for model in model_candidates):
        return []

    dicts = _extract_provider_error_dicts(result)
    status_code = _extract_provider_error_status_code(result, dicts)
    error_text = _extract_provider_error_text(result, dicts)
    error_code, error_type = _extract_provider_error_code_and_type(result, dicts)
    error_class = _classify_provider_error(
        status_code=status_code,
        error_code=error_code,
        error_type=error_type,
        error_text=error_text,
    )
    if error_class not in {"rate_limited", "usage_limit_reached"}:
        return []

    observed_dt = context["observed_at"]
    day_start, day_end = _openrouter_free_daily_window(observed_dt)
    provider_resets_at, reset_hint_seconds = _extract_openrouter_free_error_reset_at(
        kwargs,
        result,
        dicts,
        error_text,
        observed_dt,
    )
    context = dict(context)
    context["account_hash"] = _openrouter_free_shared_account_hash()
    context["client_family"] = "openrouter"
    context["model"] = None
    total_requests = _openrouter_free_daily_request_limit()
    return [
        _build_openrouter_free_daily_observation(
            context=context,
            day_start=day_start,
            day_end=day_end,
            used_requests=total_requests,
            total_requests=total_requests,
            signal="openrouter_free_model_rate_limit_error",
            status="quota_exhausted",
            exhausted=True,
            reset_hint_seconds=reset_hint_seconds,
            provider_resets_at=provider_resets_at,
        )
    ]


def _looks_like_google_quota_candidate(candidate: Dict[str, Any]) -> bool:
    request_quota_keys = {
        "buckets",
        "modelId",
        "tokenType",
        "remainingFraction",
        "remainingRequests",
        "remaining_requests",
        "requestsRemaining",
        "usedRequests",
        "used_requests",
        "requestsUsed",
        "totalRequests",
        "total_requests",
        "requestLimit",
        "dailyLimit",
        "quotaId",
        "quotaName",
    }
    weak_quota_keys = {
        "usagePercentage",
        "usedPercentage",
        "used_percentage",
    }
    candidate_keys = set(candidate.keys())
    if request_quota_keys.intersection(candidate_keys):
        return True
    source = str(candidate.get("source") or "").lower()
    return bool(weak_quota_keys.intersection(candidate_keys)) and (
        "google" in source or "gemini" in source
    )


def _antigravity_quota_pool_for_model(model: Optional[str]) -> Tuple[str, str, str]:
    normalized = str(model or "").strip().lower()
    if normalized.startswith("claude-") or normalized.startswith("gpt-oss"):
        return (
            "vertex_pool",
            "Antigravity Code Assist Vertex pool",
            "vertex",
        )
    return (
        "gemini_pool",
        "Antigravity Code Assist Gemini pool",
        "gemini",
    )


def _extract_google_quota_observations(  # noqa: PLR0915
    kwargs: Dict[str, Any],
    result: Any,
    observed_at: Any,
) -> List[Dict[str, Any]]:
    context = _build_rate_limit_context(
        kwargs,
        result,
        observed_at,
        "google_retrieve_user_quota",
    )
    observations: List[Dict[str, Any]] = []
    raw_metadata = context.get("metadata")
    metadata: Dict[str, Any] = raw_metadata if isinstance(raw_metadata, dict) else {}
    default_quota_source = _clean_non_empty_string(
        _maybe_get(metadata.get("google_retrieve_user_quota"), "source")
    )
    for candidate in _iter_rate_limit_dicts(*_rate_limit_candidate_roots(kwargs, result)):
        if not _looks_like_google_quota_candidate(candidate):
            continue
        quota_source = (
            _clean_non_empty_string(candidate.get("source"))
            or default_quota_source
            or "google_retrieve_user_quota"
        )
        remaining_requests = _first_quota_number(
            candidate,
            "remainingRequests",
            "remaining_requests",
            "requestsRemaining",
            "remaining",
        )
        used_requests = _first_quota_number(
            candidate,
            "usedRequests",
            "used_requests",
            "requestsUsed",
            "currentUsage",
            "used",
        )
        total_requests = _first_quota_number(
            candidate,
            "totalRequests",
            "total_requests",
            "requestLimit",
            "dailyLimit",
            "limit",
            "quota",
        )
        used_percentage = _first_quota_float(
            candidate,
            "usedPercentage",
            "used_percentage",
            "usagePercentage",
        )
        remaining_fraction = _first_quota_float(
            candidate,
            "remainingFraction",
            "remaining_fraction",
        )
        if used_percentage is None and used_requests is not None and total_requests:
            used_percentage = (used_requests / total_requests) * 100
        if used_percentage is None and remaining_fraction is not None:
            used_percentage = max(0.0, min(100.0, (1 - remaining_fraction) * 100))
        reset_source_value = _first_non_none(
            candidate.get("resetsAt"),
            candidate.get("resets_at"),
            candidate.get("resetAt"),
            candidate.get("resetTime"),
        )
        provider_resets_at, stale_reset = _resolve_rate_limit_reset_at(
            reset_source_value,
            context["observed_at"],
        )
        if stale_reset:
            continue
        if (
            remaining_requests is None
            and used_requests is None
            and total_requests is None
            and used_percentage is None
            and reset_source_value is None
            and _clean_non_empty_string(candidate.get("quotaId")) is None
            and _clean_non_empty_string(candidate.get("quotaName")) is None
        ):
            continue
        model = (
            _clean_non_empty_string(candidate.get("modelId"))
            or _clean_non_empty_string(candidate.get("model"))
            or context.get("model")
        )
        model_family, model_tier = _infer_model_family_and_tier(model)
        token_type = _clean_non_empty_string(candidate.get("tokenType"))
        provider = context.get("provider") or "gemini"
        client_family = context.get("client_family") or _infer_rate_limit_client_family(
            provider,
            str(model or ""),
            metadata,
            quota_source,
        )
        is_antigravity_quota = (
            provider == "antigravity"
            or client_family == "antigravity_code_assist"
            or str(quota_source or "").lower().startswith("antigravity_")
        )
        if is_antigravity_quota and provider_resets_at is None:
            continue
        explicit_quota_period = (
            _normalize_quota_period(candidate.get("quotaPeriod"))
            or _normalize_quota_period(candidate.get("period"))
        )
        quota_period = explicit_quota_period or (
            "five_hour" if is_antigravity_quota else "daily"
        )
        window_minutes = None
        for window_candidate in (
            candidate.get("windowMinutes"),
            candidate.get("window_minutes"),
            candidate.get("windowMinutesEstimate"),
        ):
            parsed_window_minutes = _safe_int(window_candidate)
            if parsed_window_minutes is not None and parsed_window_minutes > 0:
                window_minutes = parsed_window_minutes
                break
        window_minutes = window_minutes or _window_minutes_from_quota_period(
            quota_period
        )
        if is_antigravity_quota:
            limit_scope, limit_name, model_family = _antigravity_quota_pool_for_model(
                model
            )
            limit_id = "antigravity_code_assist"
            stored_model = None
            quota_type = "wtus"
            model_tier = None
            provider = "antigravity"
            client_family = "antigravity_code_assist"
        else:
            limit_scope = (
                "model_requests"
                if _clean_non_empty_string(candidate.get("modelId"))
                or _clean_non_empty_string(candidate.get("model"))
                else "daily_request_pool"
            )
            limit_id = (
                f"google_code_assist_requests_{model}"
                if limit_scope == "model_requests" and model
                else "google_code_assist_requests"
            )
            limit_name = (
                f"Google Code Assist {model} requests"
                if model
                else "Google Code Assist requests"
            )
            stored_model = model
            quota_type = None
        observations.append(
            _finalize_rate_limit_observation(
                {
                    "observed_at": context["observed_at"],
                    "source": quota_source,
                    "provider": provider,
                    "client_family": client_family,
                    "limit_id": limit_id
                    if is_antigravity_quota
                    else _clean_non_empty_string(candidate.get("quotaId")) or limit_id,
                    "limit_name": limit_name
                    if is_antigravity_quota
                    else _clean_non_empty_string(candidate.get("quotaName"))
                    or limit_name,
                    "limit_scope": limit_scope,
                    "window_minutes": window_minutes,
                    "quota_period": quota_period,
                    "quota_type": quota_type,
                    "provider_resets_at": provider_resets_at,
                    "used_percentage": used_percentage,
                    "remaining_requests": remaining_requests,
                    "used_requests": used_requests,
                    "total_requests": total_requests,
                    "model": stored_model,
                    "model_family": model_family,
                    "model_tier": model_tier,
                    "raw_provider_fields": {
                        key: candidate.get(key)
                        for key in (
                            "modelId",
                            "tokenType",
                            "remainingFraction",
                            "remainingRequests",
                            "remaining_requests",
                            "usedRequests",
                            "used_requests",
                            "totalRequests",
                            "total_requests",
                            "usagePercentage",
                            "usedPercentage",
                            "quotaPeriod",
                            "period",
                            "model",
                            "quotaId",
                            "resetsAt",
                            "resets_at",
                            "resetAt",
                            "resetTime",
                            "windowMinutes",
                            "window_minutes",
                            "windowMinutesEstimate",
                        )
                        if key in candidate
                    },
                    "evidence": {
                        "signals": ["google_quota_payload"],
                        "provider_fields": sorted(
                            key
                            for key in list(candidate.keys())
                            if "quota" in key.lower()
                            or "request" in key.lower()
                            or "usage" in key.lower()
                            or "fraction" in key.lower()
                            or "reset" in key.lower()
                            or key in {"modelId", "tokenType"}
                        )[:20],
                        "token_type": token_type,
                    },
                },
                context,
            )
        )
    return _dedupe_rate_limit_observations(observations)


def _extract_google_error_observations(
    kwargs: Dict[str, Any],
    result: Any,
    observed_at: Any,
) -> List[Dict[str, Any]]:
    context = _build_rate_limit_context(
        kwargs,
        result,
        observed_at,
        "google_generate_content_error",
    )
    observations: List[Dict[str, Any]] = []
    for candidate in _extract_error_payload_dicts(result) + _iter_rate_limit_dicts(
        *_rate_limit_candidate_roots(kwargs, result)
    ):
        error = candidate.get("error") if isinstance(candidate.get("error"), dict) else candidate
        if not isinstance(error, dict):
            continue
        status_text = _clean_non_empty_string(error.get("status"))
        code = _safe_int(error.get("code"))
        message = _clean_non_empty_string(error.get("message")) or ""
        raw_details = error.get("details")
        details: List[Any] = raw_details if isinstance(raw_details, list) else []
        reasons = [
            _clean_non_empty_string(_maybe_get(detail, "reason"))
            for detail in details
            if isinstance(detail, dict)
        ]
        reasons = [reason for reason in reasons if reason]
        metadata_models = [
            _clean_non_empty_string(_maybe_get_path(detail, "metadata", "model"))
            for detail in details
            if isinstance(detail, dict)
        ]
        metadata_models = [model for model in metadata_models if model]
        is_resource_exhausted = (
            code == 429
            or status_text in {"RESOURCE_EXHAUSTED", "RATE_LIMIT_EXCEEDED"}
            or any(reason in {"MODEL_CAPACITY_EXHAUSTED", "RATE_LIMIT_EXCEEDED"} for reason in reasons)
        )
        if not is_resource_exhausted:
            continue
        is_capacity = any(reason == "MODEL_CAPACITY_EXHAUSTED" for reason in reasons)
        reset_hint_seconds = _parse_reset_hint_seconds(message)
        provider_resets_at = (
            context["observed_at"] + timedelta(seconds=reset_hint_seconds)
            if reset_hint_seconds is not None
            else None
        )
        model = metadata_models[0] if metadata_models else context.get("model")
        model_family, model_tier = _infer_model_family_and_tier(model)
        observations.append(
            _finalize_rate_limit_observation(
                {
                    "observed_at": context["observed_at"],
                    "source": (
                        "google_model_capacity_error"
                        if is_capacity
                        else "google_generate_content_error"
                    ),
                    "provider": "gemini",
                    "client_family": "google_code_assist",
                    "limit_id": (
                        "google_model_capacity"
                        if is_capacity
                        else "google_code_assist_requests"
                    ),
                    "limit_name": (
                        "Google model capacity"
                        if is_capacity
                        else "Google Code Assist requests"
                    ),
                    "limit_scope": "model_capacity" if is_capacity else "daily_request_pool",
                    "quota_period": None if is_capacity else "daily",
                    "window_minutes": None if is_capacity else 1440,
                    "provider_resets_at": provider_resets_at,
                    "status": (
                        "model_capacity_exhausted"
                        if is_capacity
                        else "quota_exhausted"
                    ),
                    "exhausted": not is_capacity,
                    "exhaustion_kind": (
                        "model_capacity"
                        if is_capacity
                        else "request_quota"
                    ),
                    "reset_hint_seconds": reset_hint_seconds,
                    "model": model,
                    "model_family": model_family,
                    "model_tier": model_tier,
                    "raw_provider_fields": {
                        "code": code,
                        "status": status_text,
                        "message": message,
                        "reasons": reasons,
                        "metadata_models": metadata_models,
                    },
                    "evidence": {
                        "signals": [
                            "google_resource_exhausted",
                            "model_capacity" if is_capacity else "quota_exhaustion",
                        ],
                        "corroboration_required": is_capacity,
                    },
                },
                context,
            )
        )
    return _dedupe_rate_limit_observations(observations)


def _build_rate_limit_observations(
    kwargs: Dict[str, Any],
    result: Any,
    start_time: Any,
    end_time: Any,
) -> List[Dict[str, Any]]:
    observed_at = (
        _parse_datetime_value(end_time)
        or _parse_datetime_value(start_time)
        or datetime.now(timezone.utc)
    )
    observations: List[Dict[str, Any]] = []
    observations.extend(_extract_codex_rate_limit_observations(kwargs, result, observed_at))
    observations.extend(_extract_codex_header_rate_limit_observations(kwargs, result, observed_at))
    observations.extend(_extract_codex_usage_limit_error_observations(kwargs, result, observed_at))
    observations.extend(_extract_anthropic_header_rate_limit_observations(kwargs, result, observed_at))
    observations.extend(_extract_xai_oauth_header_rate_limit_observations(kwargs, result, observed_at))
    observations.extend(_extract_grok_billing_observations(kwargs, result, observed_at))
    observations.extend(_extract_openrouter_free_error_observations(kwargs, result, observed_at))
    observations.extend(_extract_google_quota_observations(kwargs, result, observed_at))
    observations.extend(_extract_google_error_observations(kwargs, result, observed_at))
    return _dedupe_rate_limit_observations(observations)


def _extract_provider_error_dicts(value: Any) -> List[Dict[str, Any]]:
    dicts: List[Dict[str, Any]] = []
    if isinstance(value, dict):
        dicts.append(value)
    dicts.extend(_extract_error_payload_dicts(value))
    for source in (
        value,
        str(value) if value is not None else None,
        getattr(value, "detail", None),
        getattr(value, "message", None),
        getattr(value, "body", None),
    ):
        dicts.extend(_extract_embedded_json_payload_dicts(source))

    seen: set[str] = set()
    deduped: List[Dict[str, Any]] = []
    for candidate in dicts:
        try:
            key = json.dumps(
                _json_safe_rate_limit_value(candidate),
                sort_keys=True,
                default=str,
            )
        except Exception:
            key = str(id(candidate))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(candidate)
    return deduped


_AAWM_EMBEDDED_JSON_MAX_SUCCESS = 20
_AAWM_EMBEDDED_JSON_MAX_ATTEMPTS = 64
_AAWM_EMBEDDED_JSON_SCAN_CHARS = 20000


def _extract_embedded_json_payload_dicts(value: Any) -> List[Dict[str, Any]]:
    text = _clean_non_empty_string(value)
    if text is None:
        return []

    decoder = json.JSONDecoder()
    dicts: List[Dict[str, Any]] = []
    attempts = 0
    for match in re.finditer(r"\{", text[:_AAWM_EMBEDDED_JSON_SCAN_CHARS]):
        if len(dicts) >= _AAWM_EMBEDDED_JSON_MAX_SUCCESS:
            break
        if attempts >= _AAWM_EMBEDDED_JSON_MAX_ATTEMPTS:
            break
        attempts += 1
        try:
            parsed, _end = decoder.raw_decode(text[match.start() :])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, (dict, list)):
            dicts.extend(_iter_rate_limit_dicts(parsed))
    return dicts


def _extract_provider_error_headers(value: Any) -> Dict[str, str]:
    headers: Dict[str, str] = {}
    candidates = [
        value,
        getattr(value, "response", None),
        getattr(value, "headers", None),
        getattr(value, "response_headers", None),
        getattr(value, "upstream_headers", None),
    ]
    response = getattr(value, "response", None)
    if response is not None:
        candidates.extend(
            [
                getattr(response, "headers", None),
                getattr(response, "response_headers", None),
            ]
        )

    for candidate in candidates:
        if candidate is None:
            continue
        if not isinstance(candidate, dict) and hasattr(candidate, "items"):
            try:
                candidate = dict(candidate.items())
            except Exception:
                continue
        if not isinstance(candidate, dict):
            continue
        for key, nested_value in list(candidate.items()):
            key_text = _clean_non_empty_string(key)
            value_text = _clean_non_empty_string(nested_value)
            if key_text and value_text:
                headers[key_text.lower()] = value_text
    return headers


def _extract_provider_error_status_code(result: Any, dicts: List[Dict[str, Any]]) -> Optional[int]:
    for candidate in (
        getattr(result, "status_code", None),
        getattr(getattr(result, "response", None), "status_code", None),
        getattr(result, "code", None),
    ):
        status_code = _safe_int(candidate)
        if status_code is not None:
            return status_code

    for candidate in dicts:
        error = candidate.get("error") if isinstance(candidate.get("error"), dict) else candidate
        for key in ("status_code", "statusCode", "http_status", "code"):
            status_code = _safe_int(error.get(key)) if isinstance(error, dict) else None
            if status_code is not None:
                return status_code
    return None


def _extract_provider_error_text(result: Any, dicts: List[Dict[str, Any]]) -> str:
    parts: List[str] = []
    for candidate in (
        getattr(result, "message", None),
        getattr(result, "detail", None),
        str(result) if result is not None else None,
    ):
        cleaned = _clean_non_empty_string(candidate)
        if cleaned:
            parts.append(cleaned)

    for candidate in dicts:
        error = candidate.get("error") if isinstance(candidate.get("error"), dict) else candidate
        if not isinstance(error, dict):
            continue
        for key in ("message", "detail", "status", "type", "code", "reason"):
            cleaned = _clean_non_empty_string(error.get(key))
            if cleaned:
                parts.append(cleaned)
        raw_details = error.get("details")
        details: List[Any] = raw_details if isinstance(raw_details, list) else []
        for detail in details:
            if not isinstance(detail, dict):
                continue
            for key in ("reason", "domain"):
                cleaned = _clean_non_empty_string(detail.get(key))
                if cleaned:
                    parts.append(cleaned)
            metadata = detail.get("metadata")
            if isinstance(metadata, dict):
                for key in ("reason", "model"):
                    cleaned = _clean_non_empty_string(metadata.get(key))
                    if cleaned:
                        parts.append(cleaned)
    return " ".join(parts)


def _extract_provider_error_code_and_type(
    result: Any,
    dicts: List[Dict[str, Any]],
) -> Tuple[Optional[str], Optional[str]]:
    error_code = _first_non_empty_string(
        getattr(result, "code", None),
        getattr(result, "error_code", None),
    )
    error_type = _first_non_empty_string(
        getattr(result, "type", None),
        getattr(result, "error_type", None),
        type(result).__name__ if result is not None else None,
    )
    for candidate in dicts:
        error = candidate.get("error") if isinstance(candidate.get("error"), dict) else candidate
        if not isinstance(error, dict):
            continue
        error_code = error_code or _first_non_empty_string(
            error.get("status"),
            error.get("reason"),
            error.get("code"),
        )
        error_type = error_type or _first_non_empty_string(
            error.get("type"),
            error.get("error_type"),
            error.get("status"),
        )
    return error_code, error_type


def _extract_provider_error_retry_after_seconds(
    *,
    kwargs: Dict[str, Any],
    result: Any,
    dicts: List[Dict[str, Any]],
    error_text: str,
) -> Optional[float]:
    headers = _extract_headers_from_kwargs(kwargs)
    headers.update(_extract_provider_error_headers(result))
    retry_after = _first_non_empty_string(
        headers.get("retry-after"),
        headers.get("x-ratelimit-reset-after"),
        headers.get("x-codex-primary-reset-after-seconds"),
        headers.get("x-codex-secondary-reset-after-seconds"),
    )
    retry_after_seconds = _safe_float(retry_after)
    if retry_after_seconds is not None and retry_after_seconds >= 0:
        return retry_after_seconds

    for candidate in dicts:
        error = candidate.get("error") if isinstance(candidate.get("error"), dict) else candidate
        if not isinstance(error, dict):
            continue
        for key in ("retry_after_seconds", "retryAfterSeconds", "resetAfterSeconds"):
            parsed = _safe_float(error.get(key))
            if parsed is not None and parsed >= 0:
                return parsed
    reset_hint = _parse_reset_hint_seconds(error_text)
    return float(reset_hint) if reset_hint is not None else None


_LITELLM_PROVIDER_ERROR_MODEL_GROUP_RE = re.compile(
    r"Received Model Group=(?P<model_group>[^\n\r]+)"
)
_LITELLM_PROVIDER_ERROR_FALLBACKS_RE = re.compile(
    r"Available Model Group Fallbacks=(?P<fallbacks>.*?)(?:\s+LiteLLM Retried:|$)",
    re.DOTALL,
)
_LITELLM_PROVIDER_ERROR_RETRIES_RE = re.compile(
    r"LiteLLM Retried:\s*(?P<retry_count>\d+)\s*times,\s*"
    r"LiteLLM Max Retries:\s*(?P<max_retries>\d+)"
)


def _extract_litellm_provider_error_model_group(error_text: str) -> Optional[str]:
    match = _LITELLM_PROVIDER_ERROR_MODEL_GROUP_RE.search(error_text)
    if not match:
        return None
    return _clean_non_empty_string(match.group("model_group"))


def _clean_litellm_provider_error_fallbacks(value: Any) -> Optional[str]:
    fallbacks = _clean_non_empty_string(value)
    if fallbacks is None:
        return None
    for marker in (
        " LiteLLM Retried:",
        " litellm.",
        " RateLimitError:",
        " Traceback ",
        " During handling ",
    ):
        marker_index = fallbacks.find(marker)
        if marker_index > 0:
            fallbacks = fallbacks[:marker_index].strip()
    if fallbacks.lower().startswith("none"):
        return "None"
    return fallbacks[:500]


def _extract_litellm_provider_error_retry_context(error_text: str) -> Dict[str, Any]:
    metadata: Dict[str, Any] = {}
    retry_match = _LITELLM_PROVIDER_ERROR_RETRIES_RE.search(error_text)
    if retry_match:
        retry_count = _safe_int(retry_match.group("retry_count"))
        max_retries = _safe_int(retry_match.group("max_retries"))
        if retry_count is not None:
            metadata["litellm_retry_count"] = retry_count
        if max_retries is not None:
            metadata["litellm_max_retries"] = max_retries
        if retry_count is not None and max_retries is not None:
            metadata["litellm_retries_exhausted"] = retry_count >= max_retries

    fallbacks_match = _LITELLM_PROVIDER_ERROR_FALLBACKS_RE.search(error_text)
    if fallbacks_match:
        fallbacks = _clean_litellm_provider_error_fallbacks(
            fallbacks_match.group("fallbacks")
        )
        if fallbacks is not None:
            metadata["available_model_group_fallbacks"] = fallbacks
            metadata["no_model_group_fallbacks"] = fallbacks.lower() in {
                "none",
                "null",
                "[]",
            }
    return metadata


def _extract_provider_error_payload_metadata_value(
    dicts: List[Dict[str, Any]],
    *keys: str,
) -> Any:
    for candidate in dicts:
        pools: List[Dict[str, Any]] = []
        if isinstance(candidate, dict):
            pools.append(candidate)
            error = candidate.get("error")
            if isinstance(error, dict):
                pools.append(error)
                error_metadata = error.get("metadata")
                if isinstance(error_metadata, dict):
                    pools.append(error_metadata)
            metadata = candidate.get("metadata")
            if isinstance(metadata, dict):
                pools.append(metadata)
        for pool in pools:
            for key in keys:
                if key in pool and pool[key] not in (None, ""):
                    return pool[key]
    return None


def _resolve_provider_error_model_group(
    *,
    kwargs: Dict[str, Any],
    metadata: Dict[str, Any],
    standard_logging_object: Dict[str, Any],
    error_text: str,
    model: str,
) -> Optional[str]:
    return _first_non_empty_string(
        _get_session_history_model_group(metadata, standard_logging_object),
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
        _maybe_get_path(
            kwargs.get("standard_pass_through_logging_payload"),
            "request_body",
            "model",
        ),
        _maybe_get_path(standard_logging_object, "request_body", "model"),
        _extract_litellm_provider_error_model_group(error_text),
        kwargs.get("model"),
        metadata.get("model"),
        model if model != "unknown" else None,
    )


_UPSTREAM_ERROR_SECRET_RE = re.compile(
    r"(?is)(?P<label>authorization|x-api-key|api[-_]?key|bearer|token|secret|password)"
    r"(?P<sep>\s*[:=]\s*|\s+)"
    r"(?P<value>(?:bearer\s+)?[^\s,\"'}{]{6,})",
)


def _redact_upstream_error_raw(value: Any) -> Optional[str]:
    """Redact auth-header-shaped substrings from upstream error raw text."""
    text = _clean_non_empty_string(value)
    if text is None:
        return None

    def _replace(match: re.Match[str]) -> str:
        return f"{match.group('label')}{match.group('sep')}[REDACTED]"

    return _UPSTREAM_ERROR_SECRET_RE.sub(_replace, text)


def _build_provider_error_fingerprint(
    *,
    provider: str,
    model: Optional[str],
    model_group: Optional[str],
    status_code: Optional[int],
    error_code: Optional[str],
    error_type: Optional[str],
    error_class: str,
    observation_metadata: Dict[str, Any],
) -> str:
    # Exclude volatile upstream_error_raw so fingerprints can dedupe the same
    # error class across request-specific raw bodies.
    fingerprint_source = {
        "provider": provider,
        "model": model,
        "model_group": model_group,
        "status_code": status_code,
        "error_code": error_code,
        "error_type": error_type,
        "error_class": error_class,
        "upstream_provider_name": observation_metadata.get("upstream_provider_name"),
    }
    return hashlib.sha256(
        json.dumps(
            _json_safe_rate_limit_value(fingerprint_source),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _enrich_provider_error_observation_metadata(
    *,
    observation_metadata: Dict[str, Any],
    dicts: List[Dict[str, Any]],
    error_text: str,
) -> None:
    upstream_provider_name = _clean_non_empty_string(
        _extract_provider_error_payload_metadata_value(dicts, "provider_name")
    )
    if upstream_provider_name is not None:
        observation_metadata["upstream_provider_name"] = upstream_provider_name

    upstream_is_byok = _extract_provider_error_payload_metadata_value(dicts, "is_byok")
    if upstream_is_byok is not None:
        observation_metadata["upstream_is_byok"] = _metadata_bool(upstream_is_byok)

    upstream_error_raw = _redact_upstream_error_raw(
        _extract_provider_error_payload_metadata_value(dicts, "raw")
    )
    if upstream_error_raw is not None:
        observation_metadata["upstream_error_raw"] = upstream_error_raw[:1000]

    observation_metadata.update(
        _extract_litellm_provider_error_retry_context(error_text)
    )


def _classify_provider_error(
    *,
    status_code: Optional[int],
    error_code: Optional[str],
    error_type: Optional[str],
    error_text: str,
) -> str:
    normalized = " ".join(
        part
        for part in (
            str(status_code or ""),
            error_code or "",
            error_type or "",
            error_text,
        )
        if part
    ).lower()
    if "usage_limit_reached" in normalized:
        return "usage_limit_reached"
    if (
        "model_capacity_exhausted" in normalized
        or "capacity_exhausted" in normalized
        or "model is overloaded" in normalized
        or "overloaded" in normalized
    ):
        return "capacity_exhausted"
    if (
        status_code == 429
        or "resource_exhausted" in normalized
        or "rate_limit" in normalized
        or "rate limit" in normalized
        or "too many requests" in normalized
    ):
        return "rate_limited"
    if status_code in {401, 403} or any(
        marker in normalized
        for marker in (
            "x-api-key",
            "api key",
            "authentication",
            "unauthorized",
            "permission denied",
            "forbidden",
        )
    ):
        return "auth_failed"
    if status_code is not None and status_code >= 500:
        return "provider_5xx"
    if any(marker in normalized for marker in ("timeout", "timed out", "deadline")):
        return "provider_timeout"
    if any(
        marker in normalized
        for marker in (
            "connection error",
            "connection refused",
            "connection reset",
            "dns",
            "tls",
            "ssl",
            "network",
        )
    ):
        return "network_error"
    return "adapter_error"


def _build_provider_error_observation(
    kwargs: Dict[str, Any],
    result: Any,
    start_time: Any,
    end_time: Any,
) -> Optional[Dict[str, Any]]:
    observed_at = (
        _parse_datetime_value(end_time)
        or _parse_datetime_value(start_time)
        or datetime.now(timezone.utc)
    )
    metadata = _merged_rate_limit_metadata(kwargs)
    standard_logging_object = kwargs.get("standard_logging_object") or {}
    if not isinstance(standard_logging_object, dict):
        standard_logging_object = {}
    model = _resolve_rate_limit_model(kwargs, result, metadata)
    source_model = model if model != "unknown" else None
    if _is_claude_permission_check_metadata(metadata):
        repository = _extract_repository_identity_from_kwargs(
            kwargs,
            metadata=metadata,
            standard_logging_object=standard_logging_object,
        )
        tenant_id, _tenant_source = _extract_tenant_identity_from_kwargs(
            kwargs,
            metadata=metadata,
            standard_logging_object=standard_logging_object,
        )
        _apply_claude_auto_review_metadata(
            metadata,
            repository=repository,
            tenant_id=tenant_id,
            source_model=source_model,
        )
        model = _CLAUDE_AUTO_REVIEW_LOGICAL_MODEL
    provider = (
        _normalize_session_history_provider(
            kwargs.get("custom_llm_provider"),
            model,
            metadata,
        )
        or "unknown"
    )
    dicts = _extract_provider_error_dicts(result)
    status_code = _extract_provider_error_status_code(result, dicts)
    error_text = _extract_provider_error_text(result, dicts)
    error_code, error_type = _extract_provider_error_code_and_type(result, dicts)
    error_class = _classify_provider_error(
        status_code=status_code,
        error_code=error_code,
        error_type=error_type,
        error_text=error_text,
    )
    retry_after_seconds = _extract_provider_error_retry_after_seconds(
        kwargs=kwargs,
        result=result,
        dicts=dicts,
        error_text=error_text,
    )
    expected_reset_at = (
        observed_at + timedelta(seconds=retry_after_seconds)
        if retry_after_seconds is not None
        else None
    )
    runtime_identity = _build_session_runtime_identity(
        metadata=metadata,
        kwargs=kwargs,
        allow_runtime=True,
    )
    observation_metadata = {
        "client_name": runtime_identity.get("client_name"),
        "client_version": runtime_identity.get("client_version"),
        "client_user_agent": runtime_identity.get("client_user_agent"),
        "normalized_error_text": error_text[:500] if error_text else None,
        "observed_signal": "normal_traffic_failure",
    }
    structured_output_state = _detect_structured_output_request(
        _extract_provider_cache_request_body(kwargs),
        metadata,
    )
    if structured_output_state.get("structured_output_attempted"):
        structured_failure_reason = _first_non_empty_string(
            structured_output_state.get("structured_output_failure_reason"),
            _classify_structured_output_failure(result),
        )
        observation_metadata["structured_output_attempted"] = True
        observation_metadata["structured_output_failed"] = bool(
            structured_output_state.get("structured_output_failed")
            or structured_failure_reason
        )
        for key in (
            "structured_output_mode",
            "structured_output_schema_hash",
        ):
            value = _clean_non_empty_string(structured_output_state.get(key))
            if value is not None:
                observation_metadata[key] = value
        if structured_failure_reason is not None:
            observation_metadata[
                "structured_output_failure_reason"
            ] = structured_failure_reason
    _enrich_provider_error_observation_metadata(
        observation_metadata=observation_metadata,
        dicts=dicts,
        error_text=error_text,
    )
    if _is_claude_permission_check_metadata(metadata):
        for key in (
            "source_model",
            "logical_model",
            "trace_name",
            "trace_user_id",
            "repository",
            "tenant_id",
            "request_tags",
            "tags",
            "claude_permission_check",
            "claude_permission_check_decision",
            "claude_permission_check_blocked",
            "claude_permission_check_request_model",
            "claude_permission_check_response_model",
        ):
            value = metadata.get(key)
            if value is not None:
                observation_metadata[key] = value
    for key in (
        "auth_mode",
        "credential_family",
        "xai_oauth_managed",
        "xai_oauth_public_model",
        "xai_oauth_upstream_model",
        "xai_quota_family",
        "shared_quota_family",
        "grok_subscription_quota_shared",
        "passthrough_route_family",
        "grok_side_channel",
        "grok_side_channel_endpoint_type",
        "grok_side_channel_endpoint_path_template",
        "grok_side_channel_request_content_type",
        "grok_side_channel_request_body_byte_length",
        "grok_side_channel_request_body_digest_source",
        "grok_side_channel_request_json_container_type",
        "grok_side_channel_request_array_length",
    ):
        value = metadata.get(key)
        if value is not None:
            observation_metadata[key] = value
    model_group = _resolve_provider_error_model_group(
        kwargs=kwargs,
        metadata=metadata,
        standard_logging_object=standard_logging_object,
        error_text=error_text,
        model=model,
    )
    observation_metadata["provider_error_fingerprint"] = (
        _build_provider_error_fingerprint(
            provider=provider,
            model=model if model != "unknown" else None,
            model_group=model_group,
            status_code=status_code,
            error_code=error_code,
            error_type=error_type,
            error_class=error_class,
            observation_metadata=observation_metadata,
        )
    )
    return {
        "observed_at": observed_at,
        "environment": runtime_identity.get("litellm_environment"),
        "provider": provider,
        "model": model if model != "unknown" else None,
        "model_group": model_group,
        "route_family": _clean_non_empty_string(
            metadata.get("passthrough_route_family")
            or metadata.get("codex_auto_agent_selected_route_family")
            or metadata.get("aawm_local_route_family")
        ),
        "status_code": status_code,
        "error_type": error_type,
        "error_code": error_code,
        "error_class": error_class,
        "retry_after_seconds": retry_after_seconds,
        "expected_reset_at": expected_reset_at,
        "session_id": _extract_session_id(kwargs),
        "trace_id": _first_non_empty_string(
            metadata.get("trace_id"),
            metadata.get("langfuse_trace_id"),
            standard_logging_object.get("trace_id"),
            kwargs.get("trace_id"),
        ),
        "litellm_call_id": kwargs.get("litellm_call_id"),
        "metadata": observation_metadata,
    }


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
    normalized_events = [
        event for event in events if isinstance(event, dict)
    ]
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
        and previous_used_percentage - current_used_percentage
        >= _AAWM_RATE_LIMIT_MEANINGFUL_PERCENT_DROP
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
        key: _json_safe_rate_limit_value(value)
        for key, value in list(observation.items())
        if key not in {"metadata"}
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
        "inferred_window_start_at": _parse_provider_timestamp(
            current.get("inferred_window_start_at")
        ),
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
    reconstructed: Dict[str, Any] = (
        dict(usage_object) if isinstance(usage_object, dict) and usage_object else {}
    )

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

    completed_payload = _extract_responses_completed_payload_from_passthrough_fallback_text(
        raw_text
    )
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

    response_payload = _extract_responses_completed_response_from_langfuse_output(
        output_payload
    )
    if not isinstance(response_payload, dict):
        return None
    usage = response_payload.get("usage")
    return dict(usage) if isinstance(usage, dict) and usage else None


def _extract_codex_model_from_response_headers(metadata: Dict[str, Any]) -> Optional[str]:
    headers = metadata.get("codex_response_headers")
    if not isinstance(headers, dict):
        return None

    limit_name = _clean_non_empty_string(
        _get_rate_limit_header_value(headers, "x-codex-bengalfox-limit-name")
    )
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

    response_payload = _extract_responses_completed_response_from_langfuse_output(
        output_payload
    )
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
        if cleaned.lower().startswith("openrouter/") and len(cleaned) > len(
            "openrouter/"
        ):
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
    token_count_usage = _build_usage_object_from_token_count_payload(
        _maybe_get(result, "response")
    )
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
        token_count_usage = _build_usage_object_from_token_count_payload(
            standard_logging_object.get("output")
        )
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
    standard_passthrough_logging_payload = kwargs.get(
        "standard_pass_through_logging_payload"
    )
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
    selected_provider = _normalize_session_history_provider_name(
        metadata.get("codex_auto_agent_selected_provider")
    )
    if selected_provider is not None:
        return selected_provider
    selected_provider = _normalize_session_history_provider_name(
        metadata.get("anthropic_auto_agent_selected_provider")
    )
    if selected_provider is not None:
        return selected_provider
    return _normalize_session_history_provider_name(
        metadata.get("aawm_auto_agent_selected_provider")
    )


def _session_history_adapter_model(metadata: Dict[str, Any]) -> Optional[str]:
    prefix = "anthropic-adapter-model:"
    for tag in _metadata_request_tags(metadata):
        stripped_tag = tag.strip()
        if stripped_tag.lower().startswith(prefix):
            return stripped_tag[len(prefix) :].strip() or None
    return None


def _normalize_provider_cache_family(
    provider: Any,
    model: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    route_family = (metadata or {}).get("passthrough_route_family")
    if isinstance(route_family, str) and route_family.strip():
        route_family_lower = route_family.lower()
        if "grok" in route_family_lower or "xai" in route_family_lower:
            return "xai"
        if "nvidia" in route_family_lower:
            return "nvidia"
        if "openrouter" in route_family_lower:
            return "openrouter"
        if "opencode" in route_family_lower:
            return "opencode_zen"
        if "antigravity" in route_family_lower:
            return "antigravity"
        if "gemini" in route_family_lower or "google" in route_family_lower:
            return "gemini"
        if "anthropic" in route_family_lower:
            return "anthropic"
        if "openai" in route_family_lower or "codex" in route_family_lower:
            return "openai"

    if isinstance(provider, str) and provider.strip():
        provider_lower = provider.strip().lower()
        if provider_lower == "google":
            return "gemini"
        if provider_lower in {"antigravity", "agy", "google-antigravity"}:
            return "antigravity"
        if provider_lower in {"nvidia_nim", "nvidia-nim"}:
            return "nvidia"
        if provider_lower in {"opencode", "opencode-zen", "opencode_zen", "zen"}:
            return "opencode_zen"
        if provider_lower in _PROVIDER_CACHE_TARGET_FAMILIES:
            return provider_lower

    model_lower = str(model or "").strip().lower()
    if model_lower.startswith("nvidia_nim/") or model_lower.startswith("nvidia/"):
        return "nvidia"
    if model_lower.startswith("xai/") or model_lower.startswith("grok"):
        return "xai"
    if model_lower.startswith("openrouter/"):
        return "openrouter"
    if model_lower.startswith(("opencode/", "opencode-zen/", "zen/")):
        return "opencode_zen"
    if model_lower.startswith(("antigravity/", "agy/", "google-antigravity/")):
        return "antigravity"
    if (
        "gemini" in model_lower
        or "gemma" in model_lower
        or model_lower.startswith("google/")
    ):
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

    route_provider = _session_history_provider_from_route_family(
        metadata.get("passthrough_route_family")
    )
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


def _supports_prompt_caching_safe(
    *,
    model: str,
    provider: Optional[str],
) -> Optional[bool]:
    normalized_model = str(model or "").strip()
    if not normalized_model or normalized_model.lower() == "unknown":
        return None
    try:
        litellm = _get_litellm_module()
        return bool(
            litellm.supports_prompt_caching(
                model=normalized_model,
                custom_llm_provider=provider,
            )
        )
    except Exception:
        return None


def _extract_provider_cache_request_body(kwargs: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    candidates = (
        _maybe_get(kwargs.get("passthrough_logging_payload"), "request_body"),
        _maybe_get_path(kwargs.get("litellm_params"), "proxy_server_request", "body"),
        kwargs.get("request_body"),
        _maybe_get(kwargs.get("standard_logging_object"), "request_body"),
        _maybe_get_path(kwargs.get("standard_logging_object"), "request", "body"),
    )
    for candidate in candidates:
        if isinstance(candidate, dict):
            return candidate
    return None


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
    if (
        message_type in _TOOL_RESULT_ERROR_BLOCK_TYPES
        or (message_role or "").lower() == "tool"
    ):
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
        current.get("structured_output_failed")
        or candidate.get("structured_output_failed")
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
        dict_mode
        and (
            dict_mode in _STRUCTURED_OUTPUT_JSON_MODE_VALUES
            or "json" in dict_mode
            or "schema" in dict_mode
        )
    )
    if schema is None and not has_json_mode and not has_json_mime:
        return state

    state["structured_output_attempted"] = True
    state["structured_output_mode"] = dict_mode or (
        "response_schema" if schema is not None else "json_mime_type"
    )
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
    state["structured_output_mode"] = (
        "response_schema" if schema is not None else "json_mime_type"
    )
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
            pending.extend(
                (item, depth + 1)
                for item in list(value)
                if isinstance(item, (dict, list, tuple))
            )

    return False


def _request_contains_cache_control(payload: Any) -> bool:
    return _request_payload_contains(
        payload,
        lambda item: item.get("cache_control") is not None
        or item.get("cacheControl") is not None,
    )


def _request_contains_cached_content(payload: Any) -> bool:
    def _has_cached_content(item: Dict[str, Any]) -> bool:
        cached_content = item.get("cachedContent")
        if isinstance(cached_content, str) and cached_content.strip():
            return True
        cached_content_alias = item.get("cached_content")
        return isinstance(cached_content_alias, str) and bool(
            cached_content_alias.strip()
        )

    return _request_payload_contains(payload, _has_cached_content)


def _request_contains_prompt_cache_key(payload: Any) -> bool:
    if not isinstance(payload, dict):
        return False
    prompt_cache_key = payload.get("prompt_cache_key")
    return isinstance(prompt_cache_key, str) and bool(prompt_cache_key.strip())


_CODEX_THREAD_ID_RE = re.compile(
    r"\bCODEX_THREAD_ID=(?P<thread_id>[A-Za-z0-9][A-Za-z0-9._:-]{7,})\b"
)
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
    return (
        client_name == "codex-tui"
        or trace_name.startswith("codex")
        or route_family == "codex_responses"
    )


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
        if (
            "context checkpoint compaction" in request_text_lower
        ):
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
        strict_prompt_shape = all(
            marker in request_text_lower for marker in _CLAUDE_CODE_COMPACT_REQUEST_MARKERS
        )
        compact_summary_phrase = (
            "summarize the current context" in request_text_lower
            or "context compacted" in request_text_lower
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


def _openai_style_cached_tokens_source(usage_obj: Any) -> Optional[str]:
    for path, source in (
        (
            ("prompt_tokens_details", "cached_tokens"),
            "usage.prompt_tokens_details.cached_tokens",
        ),
        (
            ("prompt_tokens_details", "cachedTokens"),
            "usage.prompt_tokens_details.cachedTokens",
        ),
        (
            ("input_tokens_details", "cached_tokens"),
            "usage.input_tokens_details.cached_tokens",
        ),
        (
            ("input_tokens_details", "cachedTokens"),
            "usage.input_tokens_details.cachedTokens",
        ),
        (
            ("promptTokensDetails", "cached_tokens"),
            "usage.promptTokensDetails.cached_tokens",
        ),
        (
            ("promptTokensDetails", "cachedTokens"),
            "usage.promptTokensDetails.cachedTokens",
        ),
        (
            ("inputTokensDetails", "cached_tokens"),
            "usage.inputTokensDetails.cached_tokens",
        ),
        (
            ("inputTokensDetails", "cachedTokens"),
            "usage.inputTokensDetails.cachedTokens",
        ),
    ):
        if _has_nested_path(usage_obj, *path):
            return source
    return None


def _usage_has_openai_style_cached_tokens_field(usage_obj: Any) -> bool:
    return _openai_style_cached_tokens_source(usage_obj) is not None


def _usage_has_gemini_style_cached_content_field(usage_obj: Any) -> bool:
    return _has_nested_path(usage_obj, "cachedContentTokenCount")


def _openai_cache_attempt_source(
    usage_obj: Any, request_body: Optional[Dict[str, Any]]
) -> Optional[Tuple[str, str]]:
    if _request_contains_prompt_cache_key(request_body):
        return "prompt_cache_key_requested_without_hit", "request.prompt_cache_key"
    cached_tokens_source = _openai_style_cached_tokens_source(usage_obj)
    if cached_tokens_source is not None:
        return "cached_tokens_reported_zero", cached_tokens_source
    return None


def _extract_service_tier_hint(
    usage_obj: Any,
    metadata: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    metadata = metadata or {}
    for candidate in (
        _maybe_get(usage_obj, "service_tier"),
        _maybe_get(usage_obj, "serviceTier"),
        metadata.get("service_tier"),
        metadata.get("serviceTier"),
        metadata.get("openai_service_tier"),
    ):
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()
    return None


def _compute_provider_cache_miss_cost_state(  # noqa: PLR0915
    *,
    provider_family: Optional[str],
    model: str,
    usage_obj: Any,
    cache_state: Optional[Dict[str, Any]],
    metadata: Optional[Dict[str, Any]] = None,
    response_cost_usd: Optional[float] = None,
) -> Dict[str, Any]:
    existing_miss_token_count = (
        _safe_int(cache_state.get("miss_token_count"))
        if isinstance(cache_state, dict)
        else None
    )
    existing_miss_cost_usd = (
        _safe_float(cache_state.get("miss_cost_usd"))
        if isinstance(cache_state, dict)
        else None
    )
    existing_miss_cost_basis = (
        str(cache_state.get("miss_cost_basis")).strip()
        if isinstance(cache_state, dict)
        and cache_state.get("miss_cost_basis") is not None
        and str(cache_state.get("miss_cost_basis")).strip()
        else None
    )
    result: Dict[str, Any] = {
        "miss_token_count": existing_miss_token_count,
        "miss_cost_usd": existing_miss_cost_usd,
        "miss_cost_basis": existing_miss_cost_basis,
    }
    if provider_family is None or cache_state is None:
        return result

    cost_provider_family = "nvidia_nim" if provider_family == "nvidia" else provider_family
    cache_status = cache_state.get("status")
    cache_missed = bool(cache_state.get("miss"))
    cache_miss_reason = cache_state.get("miss_reason")
    service_tier = _extract_service_tier_hint(usage_obj, metadata)

    def _fallback_miss_cost(
        miss_token_count: int,
    ) -> Tuple[Optional[float], Optional[str]]:
        model_info = _lookup_bundled_model_cost_info(
            model=model,
            custom_llm_provider=cost_provider_family,
        )
        input_cost_per_token = (
            _safe_float(model_info.get("input_cost_per_token"))
            if isinstance(model_info, dict)
            else None
        )
        if input_cost_per_token is not None:
            return (
                max(float(input_cost_per_token) * float(miss_token_count), 0.0),
                "prompt_input_cost_no_cache_read_pricing",
            )

        response_cost = _safe_float(
            _first_non_none(
                response_cost_usd,
                _maybe_get(usage_obj, "cost"),
                _maybe_get(usage_obj, "response_cost"),
                _maybe_get(usage_obj, "responseCost"),
                (metadata or {}).get("litellm_response_cost"),
                (metadata or {}).get("response_cost"),
                (metadata or {}).get("usage_openrouter_cost"),
            )
        )
        if response_cost is None or response_cost < 0:
            return None, None
        if response_cost == 0:
            return 0.0, "response_cost_zero"

        prompt_tokens = _extract_prompt_tokens(usage_obj)
        completion_tokens = _extract_completion_tokens(usage_obj)
        total_tokens = _extract_total_tokens(
            usage_obj,
            prompt_tokens,
            completion_tokens,
        )
        if total_tokens > 0:
            token_share = min(float(miss_token_count) / float(total_tokens), 1.0)
            return (
                max(float(response_cost) * token_share, 0.0),
                "response_cost_token_share_estimate",
            )
        return float(response_cost), "response_cost_estimate"

    def _populate_prompt_vs_cache_read_delta_cost(miss_token_count: int) -> Dict[str, Any]:
        if result["miss_cost_usd"] is not None:
            return result
        try:
            from litellm.litellm_core_utils.llm_cost_calc.utils import (
                _get_token_base_cost,
            )
            from litellm.types.utils import ModelInfo, Usage
            from litellm.utils import get_model_info

            usage_for_cost = Usage(
                prompt_tokens=miss_token_count,
                completion_tokens=0,
                total_tokens=miss_token_count,
            )
            try:
                model_info: Any = get_model_info(
                    model=model,
                    custom_llm_provider=cost_provider_family,
                )
            except Exception:
                model_info = _lookup_bundled_model_cost_info(
                    model=model,
                    custom_llm_provider=cost_provider_family,
                )
            if not isinstance(model_info, dict):
                fallback_cost, fallback_basis = _fallback_miss_cost(miss_token_count)
                if fallback_cost is not None:
                    result["miss_cost_usd"] = fallback_cost
                    result["miss_cost_basis"] = fallback_basis
                return result
            if "cache_read_input_token_cost" not in model_info:
                fallback_cost, fallback_basis = _fallback_miss_cost(miss_token_count)
                if fallback_cost is not None:
                    result["miss_cost_usd"] = fallback_cost
                    result["miss_cost_basis"] = fallback_basis
                return result
            typed_model_info = cast(ModelInfo, model_info)
            (
                prompt_base_cost,
                _completion_base_cost,
                _cache_creation_cost,
                _cache_creation_cost_above_1hr,
                cache_read_cost,
            ) = _get_token_base_cost(
                model_info=typed_model_info,
                usage=usage_for_cost,
                service_tier=service_tier,
            )

            if prompt_base_cost is None or cache_read_cost is None:
                return result

            miss_cost = max(
                (float(prompt_base_cost) - float(cache_read_cost))
                * float(miss_token_count),
                0.0,
            )
            result["miss_cost_usd"] = miss_cost
            result["miss_cost_basis"] = "prompt_vs_cache_read_delta"
            return result
        except Exception:
            fallback_cost, fallback_basis = _fallback_miss_cost(miss_token_count)
            if fallback_cost is not None:
                result["miss_cost_usd"] = fallback_cost
                result["miss_cost_basis"] = fallback_basis
            return result

    if (
        cache_status == "hit"
        and cache_missed
        and cache_miss_reason == "partial_cache_hit"
    ):
        miss_token_count = (
            existing_miss_token_count
            if existing_miss_token_count is not None and existing_miss_token_count > 0
            else None
        )
        if miss_token_count is None:
            prompt_tokens = _extract_prompt_tokens(usage_obj)
            cache_read_input_tokens = _extract_cache_read_input_tokens(usage_obj)
            if prompt_tokens > cache_read_input_tokens > 0:
                miss_token_count = prompt_tokens - cache_read_input_tokens
        if miss_token_count is None or miss_token_count <= 0:
            return result
        result["miss_token_count"] = miss_token_count
        return _populate_prompt_vs_cache_read_delta_cost(miss_token_count)

    if cache_status == "miss" and cache_missed:
        miss_token_count = _extract_prompt_tokens(usage_obj)
        if miss_token_count <= 0:
            if existing_miss_token_count is not None and existing_miss_token_count > 0:
                result["miss_token_count"] = existing_miss_token_count
                if result["miss_cost_usd"] is not None:
                    return result
                fallback_cost, fallback_basis = _fallback_miss_cost(
                    existing_miss_token_count
                )
                if fallback_cost is not None:
                    result["miss_cost_usd"] = fallback_cost
                    result["miss_cost_basis"] = fallback_basis
                return result
            result["miss_token_count"] = 0
            fallback_cost, fallback_basis = _fallback_miss_cost(0)
            if fallback_cost is not None:
                result["miss_cost_usd"] = fallback_cost
                result["miss_cost_basis"] = fallback_basis
            return result

        result["miss_token_count"] = miss_token_count
        return _populate_prompt_vs_cache_read_delta_cost(miss_token_count)

    if cache_status != "write":
        return result

    cache_creation_input_tokens = _extract_cache_creation_input_tokens(usage_obj)
    if cache_creation_input_tokens <= 0:
        return result

    result["miss_token_count"] = cache_creation_input_tokens
    prompt_tokens = max(_extract_prompt_tokens(usage_obj), cache_creation_input_tokens)

    try:
        from litellm.litellm_core_utils.llm_cost_calc.utils import (
            _get_token_base_cost,
            calculate_cache_writing_cost,
        )
        from litellm.types.utils import CacheCreationTokenDetails, Usage
        from litellm.utils import get_model_info

        prompt_tokens_details = _extract_prompt_tokens_details(usage_obj)
        cache_creation_token_details = None
        if isinstance(prompt_tokens_details, dict):
            detail_5m = _safe_int(
                _maybe_get(prompt_tokens_details, "ephemeral_5m_input_tokens")
            )
            detail_1h = _safe_int(
                _maybe_get(prompt_tokens_details, "ephemeral_1h_input_tokens")
            )
            if detail_5m is not None or detail_1h is not None:
                cache_creation_token_details = CacheCreationTokenDetails(
                    ephemeral_5m_input_tokens=detail_5m,
                    ephemeral_1h_input_tokens=detail_1h,
                )

        usage_for_cost = Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=0,
            total_tokens=prompt_tokens,
        )
        model_info = get_model_info(
            model=model,
            custom_llm_provider=cost_provider_family,
        )
        (
            _prompt_base_cost,
            _completion_base_cost,
            cache_creation_cost,
            cache_creation_cost_above_1hr,
            cache_read_cost,
        ) = _get_token_base_cost(
            model_info=model_info,
            usage=usage_for_cost,
            service_tier=service_tier,
        )

        write_cost = calculate_cache_writing_cost(
            cache_creation_tokens=cache_creation_input_tokens,
            cache_creation_token_details=cache_creation_token_details,
            cache_creation_cost_above_1hr=cache_creation_cost_above_1hr,
            cache_creation_cost=cache_creation_cost,
        )
        read_cost = float(cache_creation_input_tokens) * float(cache_read_cost or 0.0)
        miss_cost = max(float(write_cost) - float(read_cost), 0.0)
        result["miss_cost_usd"] = miss_cost
        result["miss_cost_basis"] = "write_vs_read_delta"
        return result
    except Exception:
        return result


def _provider_cache_state_from_metadata(
    metadata: Dict[str, Any],
    provider_family: Optional[str],
) -> Optional[Dict[str, Any]]:
    status = metadata.get("usage_provider_cache_status")
    if status is None and provider_family:
        status = metadata.get(f"{provider_family}_provider_cache_status")
    attempted = metadata.get("usage_provider_cache_attempted")
    if attempted is None and provider_family:
        attempted = metadata.get(f"{provider_family}_provider_cache_attempted")
    miss = metadata.get("usage_provider_cache_miss")
    if miss is None and provider_family:
        miss = metadata.get(f"{provider_family}_provider_cache_miss")
    miss_reason = metadata.get("usage_provider_cache_miss_reason")
    if miss_reason is None and provider_family:
        miss_reason = metadata.get(f"{provider_family}_provider_cache_miss_reason")
    miss_token_count = metadata.get("usage_provider_cache_miss_token_count")
    if miss_token_count is None and provider_family:
        miss_token_count = metadata.get(f"{provider_family}_provider_cache_miss_token_count")
    miss_cost_usd = metadata.get("usage_provider_cache_miss_cost_usd")
    if miss_cost_usd is None and provider_family:
        miss_cost_usd = metadata.get(f"{provider_family}_provider_cache_miss_cost_usd")
    miss_cost_basis = metadata.get("usage_provider_cache_miss_cost_basis")
    if miss_cost_basis is None and provider_family:
        miss_cost_basis = metadata.get(f"{provider_family}_provider_cache_miss_cost_basis")
    source = metadata.get("usage_provider_cache_source")
    if source is None and provider_family:
        source = metadata.get(f"{provider_family}_provider_cache_source")
    if (
        status is None
        and attempted is None
        and miss is None
        and miss_reason is None
        and miss_token_count is None
        and miss_cost_usd is None
        and miss_cost_basis is None
        and source is None
    ):
        return None
    normalized_status = str(status).strip() if isinstance(status, str) and status.strip() else None
    return {
        "attempted": bool(attempted) if attempted is not None else bool(normalized_status and normalized_status != "not_attempted"),
        "status": normalized_status,
        "miss": bool(miss) if miss is not None else normalized_status in {"miss", "write"},
        "miss_reason": (
            str(miss_reason).strip()
            if isinstance(miss_reason, str) and str(miss_reason).strip()
            else None
        ),
        "miss_token_count": _safe_int(miss_token_count),
        "miss_cost_usd": _safe_float(miss_cost_usd),
        "miss_cost_basis": (
            str(miss_cost_basis).strip()
            if isinstance(miss_cost_basis, str) and str(miss_cost_basis).strip()
            else None
        ),
        "source": str(source).strip() if isinstance(source, str) and str(source).strip() else None,
    }


def _resolve_provider_cache_state(  # noqa: PLR0915
    *,
    provider: Any,
    model: str,
    usage_obj: Any,
    metadata: Optional[Dict[str, Any]] = None,
    request_body: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    metadata = metadata or {}
    provider_family = _normalize_provider_cache_family(provider, model, metadata)
    if provider_family is None:
        return None

    cache_read_input_tokens = _extract_cache_read_input_tokens(usage_obj)
    cache_creation_input_tokens = _extract_cache_creation_input_tokens(usage_obj)
    state_from_metadata = _provider_cache_state_from_metadata(metadata, provider_family)
    prompt_tokens = _extract_prompt_tokens(usage_obj)
    if (
        provider_family == "xai"
        and state_from_metadata is not None
        and state_from_metadata.get("status") == "hit"
        and cache_read_input_tokens > 0
        and prompt_tokens > cache_read_input_tokens
    ):
        return {
            "attempted": True,
            "status": "hit",
            "miss": True,
            "miss_reason": "partial_cache_hit",
            "miss_token_count": prompt_tokens - cache_read_input_tokens,
            "source": state_from_metadata.get("source") or "usage.cache_read_input_tokens",
            "supports_prompt_caching": state_from_metadata.get("supports_prompt_caching"),
        }
    if state_from_metadata is not None and state_from_metadata.get("status") is not None:
        return state_from_metadata

    request_has_cache_control = _request_contains_cache_control(request_body)
    request_has_cached_content = _request_contains_cached_content(request_body)
    openai_cached_tokens_source = _openai_style_cached_tokens_source(usage_obj)
    usage_has_openai_cached_tokens = openai_cached_tokens_source is not None
    usage_has_gemini_cached_content = _usage_has_gemini_style_cached_content_field(usage_obj)
    supports_prompt_caching = _supports_prompt_caching_safe(
        model=model,
        provider=provider_family,
    )

    if cache_read_input_tokens > 0:
        if provider_family == "xai" and prompt_tokens > cache_read_input_tokens:
            return {
                "attempted": True,
                "status": "hit",
                "miss": True,
                "miss_reason": "partial_cache_hit",
                "miss_token_count": prompt_tokens - cache_read_input_tokens,
                "source": "usage.cache_read_input_tokens",
                "supports_prompt_caching": supports_prompt_caching,
            }
        return {
            "attempted": True,
            "status": "hit",
            "miss": False,
            "miss_reason": None,
            "source": "usage.cache_read_input_tokens",
            "supports_prompt_caching": supports_prompt_caching,
        }

    if cache_creation_input_tokens > 0:
        return {
            "attempted": True,
            "status": "write",
            "miss": True,
            "miss_reason": "cache_write_only",
            "source": "usage.cache_creation_input_tokens",
            "supports_prompt_caching": supports_prompt_caching,
        }

    attempted = False
    miss_reason: Optional[str] = None
    source: Optional[str] = None

    if state_from_metadata is not None and state_from_metadata.get("attempted"):
        attempted = True
        miss_reason = state_from_metadata.get("miss_reason") or "cache_attempted_without_hit"
        source = state_from_metadata.get("source") or "metadata.provider_cache_attempted"

    if provider_family == "anthropic":
        attempted = attempted or request_has_cache_control
        if attempted:
            miss_reason = miss_reason or "cache_control_requested_without_hit"
            source = source or "request.cache_control"
    elif provider_family == "openrouter":
        if request_has_cache_control:
            attempted = True
            miss_reason = miss_reason or "cache_control_requested_without_hit"
            source = source or "request.cache_control"
        elif usage_has_openai_cached_tokens:
            attempted = True
            miss_reason = miss_reason or "cached_tokens_reported_zero"
            source = source or openai_cached_tokens_source
    elif provider_family == "gemini":
        if request_has_cached_content:
            attempted = True
            miss_reason = miss_reason or "cached_content_requested_without_hit"
            source = source or "request.cached_content"
        elif usage_has_gemini_cached_content:
            attempted = True
            miss_reason = miss_reason or "cached_tokens_reported_zero"
            source = source or "usage.cached_content_token_count"
    elif provider_family == "openai":
        openai_cache_attempt_source = _openai_cache_attempt_source(
            usage_obj, request_body
        )
        if openai_cache_attempt_source:
            attempted = True
            source_miss_reason, source_name = openai_cache_attempt_source
            miss_reason = miss_reason or source_miss_reason
            source = source or source_name

    if not attempted:
        return {
            "attempted": False,
            "status": "not_attempted",
            "miss": False,
            "miss_reason": None,
            "source": None,
            "supports_prompt_caching": supports_prompt_caching,
        }

    if supports_prompt_caching is False and source and source.startswith("request."):
        return {
            "attempted": True,
            "status": "unsupported",
            "miss": False,
            "miss_reason": None,
            "source": source,
            "supports_prompt_caching": supports_prompt_caching,
        }

    return {
        "attempted": True,
        "status": "miss",
        "miss": True,
        "miss_reason": miss_reason,
        "source": source,
        "supports_prompt_caching": supports_prompt_caching,
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


def _fallback_gemini_reasoning_tokens_from_signatures(
    metadata: Dict[str, Any], message: Any = None
) -> Optional[int]:
    signature_count = _safe_int(metadata.get("gemini_thought_signature_count"))
    if signature_count is not None and signature_count > 0:
        return signature_count

    provider_specific_fields = (
        _extract_provider_specific_fields(message) if message is not None else {}
    )
    thought_signatures = provider_specific_fields.get("thought_signatures")
    if isinstance(thought_signatures, list):
        non_empty_signatures = [
            signature
            for signature in thought_signatures
            if isinstance(signature, str) and signature.strip()
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
            and (
                candidate.get("documents") is not None
                or candidate.get("texts") is not None
            )
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
        return "\n".join(
            text for item in value if (text := _coerce_rerank_text(item).strip())
        )
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
                text
                for field in rank_fields
                if (text := _coerce_rerank_text(document.get(field)).strip())
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
        return [
            block.strip()
            for block in re.split(r"\n{2,}", value)
            if block.strip()
        ]
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
            value=nested_request_block.get("systemInstruction")
            or nested_request_block.get("system_instruction"),
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
            value=request_body.get("systemInstruction")
            or request_body.get("system_instruction"),
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
            system_messages, conversation_messages = _split_chat_prompt_messages(
                request_body.get("messages")
            )
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
    tool_tokens = sum(
        _estimate_prompt_overhead_tokens(model, component["value"])
        for component in components["tools"]
    )
    conversation_tokens = sum(
        _estimate_prompt_overhead_tokens(model, component["value"])
        for component in components["conversation"]
    )
    excluded_components = components.get("excluded", [])
    opaque_state_tokens = sum(
        _estimate_prompt_overhead_tokens(model, component["value"])
        for component in excluded_components
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
        "conversation": [
            str(component.get("path")) for component in components["conversation"]
        ],
    }
    excluded_component_paths = [
        str(component.get("path")) for component in excluded_components
    ]
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
        text
        for document in documents
        if (text := _extract_rerank_document_text(document, rank_fields).strip())
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

    search_units = (
        _safe_int(usage_dict.get("search_units"))
        or _safe_int(_maybe_get_path(result, "meta", "billed_units", "search_units"))
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


@lru_cache(maxsize=1)
def _load_bundled_model_cost_map() -> Dict[str, Any]:
    try:
        from litellm.litellm_core_utils.get_model_cost_map import GetModelCostMap

        return GetModelCostMap.load_local_model_cost_map()
    except Exception as exc:
        verbose_logger.debug(
            "AawmAgentIdentity: failed to load bundled model cost map: %s",
            exc,
        )
        return {}


@lru_cache(maxsize=1)
def _bundled_model_cost_casefold_lookup() -> Dict[str, str]:
    return {
        key.lower(): key
        for key in _load_bundled_model_cost_map()
        if isinstance(key, str)
    }


def _lookup_bundled_model_cost_info(
    *,
    model: str,
    custom_llm_provider: Optional[str],
) -> Optional[Dict[str, Any]]:
    model_cost = _load_bundled_model_cost_map()
    if not model_cost:
        return None

    # Prefer provider-qualified keys when an explicit provider is supplied so
    # ambiguous bare model names cannot win over the intended provider entry.
    candidates: List[str] = []
    if custom_llm_provider:
        provider_prefix = f"{custom_llm_provider}/"
        if model.startswith(provider_prefix):
            candidates.append(model)
            stripped_model = model[len(provider_prefix) :]
            if stripped_model:
                candidates.append(stripped_model)
        else:
            candidates.append(f"{provider_prefix}{model}")
            candidates.append(model)
    else:
        candidates.append(model)

    lookup = _bundled_model_cost_casefold_lookup()
    for candidate in candidates:
        if not isinstance(candidate, str) or not candidate.strip():
            continue
        if candidate in model_cost and isinstance(model_cost[candidate], dict):
            return model_cost[candidate]
        matched_key = lookup.get(candidate.lower())
        if matched_key is not None and isinstance(model_cost.get(matched_key), dict):
            return model_cost[matched_key]

    return None


def _calculate_response_cost_from_bundled_model_cost_map(
    *,
    model: str,
    custom_llm_provider: Optional[str],
    prompt_tokens: int,
    completion_tokens: int,
    usage_obj: Any,
) -> Optional[float]:
    model_info = _lookup_bundled_model_cost_info(
        model=model,
        custom_llm_provider=custom_llm_provider,
    )
    if not model_info:
        return None

    search_units = _safe_int(_maybe_get(usage_obj, "search_units"))
    input_cost_per_query = _safe_float(model_info.get("input_cost_per_query"))
    if search_units and input_cost_per_query is not None and input_cost_per_query > 0:
        return search_units * input_cost_per_query

    has_token_pricing = (
        "input_cost_per_token" in model_info or "output_cost_per_token" in model_info
    )
    if not has_token_pricing:
        return None

    input_cost_per_token = _safe_float(model_info.get("input_cost_per_token")) or 0.0
    output_cost_per_token = _safe_float(model_info.get("output_cost_per_token")) or 0.0
    return (
        (prompt_tokens * input_cost_per_token)
        + (completion_tokens * output_cost_per_token)
    )


def _positive_int_or_none(value: Any) -> Optional[int]:
    normalized = _safe_int(value)
    if normalized is not None and normalized > 0:
        return normalized
    return None


def _normalize_reasoning_state(record: Dict[str, Any]) -> None:
    reported = _positive_int_or_none(record.get("reasoning_tokens_reported"))
    estimated = _positive_int_or_none(record.get("reasoning_tokens_estimated"))
    source = record.get("reasoning_tokens_source")
    reasoning_present = bool(
        record.get("reasoning_present") or record.get("thinking_signature_present")
    )

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
    if (
        isinstance(current_status, str)
        and current_status.strip()
        and cache_state.get("status") == "not_attempted"
    ):
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
    return any(
        fragment in normalized
        for fragment in _REQUEST_HEADER_TENANT_LITELLM_REPOSITORY_FRAGMENTS
    )


def _normalize_request_header_tenant_repository(value: Any) -> Optional[str]:
    repository = _normalize_repository_identity(value)
    if repository is None:
        return None
    normalized = repository.lower()
    if any(
        fragment in normalized
        for fragment in _REQUEST_HEADER_TENANT_LITELLM_REPOSITORY_FRAGMENTS
    ):
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
    if (
        allow_route_rollup_label
        and source.endswith(".aawm_route_rollup_context.group_header_label")
    ):
        return True
    if (
        "x-codex-turn-metadata" in source
        and source.endswith(".text.project_path")
    ):
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
    return bool(
        _is_native_codex_passthrough_context(metadata, headers)
        or _is_codex_client_identity(metadata, headers)
    )


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
    client_name = _clean_non_empty_string(
        _first_non_none(record.get("client_name"), metadata.get("client_name"))
    )
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
    return bool(
        trace_name
        and trace_name.lower() == "codex"
        and user_agent
        and "codex" in user_agent.lower()
    )


def _is_claude_session_history_record(record: Dict[str, Any]) -> bool:
    metadata = record.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
    provider = str(record.get("provider") or "").strip().lower()
    if provider == "anthropic":
        return True
    client_name = _clean_non_empty_string(
        _first_non_none(record.get("client_name"), metadata.get("client_name"))
    )
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
    return source.endswith(".metadata.tenant_id") or source.endswith(
        ".metadata.aawm_tenant_id"
    )


def _claude_project_identity_is_trusted(
    record: Dict[str, Any],
    repository: Optional[str],
    source: Any,
) -> bool:
    if not (
        repository
        and _is_claude_session_history_record(record)
        and _is_claude_project_repository_source(source)
    ):
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
    metadata["session_history_repository_unresolved_reason"] = (
        _codex_untrusted_repository_reason(metadata)
    )


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
    metadata["session_history_repository_unresolved_reason"] = (
        _session_history_missing_repository_reason(record)
    )
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

    original_tenant_id = _clean_non_empty_string(
        record.get("tenant_id")
    ) or _clean_non_empty_string(metadata.get("tenant_id"))
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
    if (
        _is_codex_trace_user_tenant_source(metadata.get("tenant_id_source"))
        or (
            _normalize_repository_identity(metadata.get("trace_user_id")) == tenant_id
            and not _codex_tenant_source_trusted_for_record(record)
        )
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
        repository = (
            _normalize_request_header_tenant_repository(record.get("tenant_id"))
            or _normalize_request_header_tenant_repository(metadata.get("tenant_id"))
        )
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
        metadata.get("tenant_id_source")
        in {"repository_untrusted", "trace_user_untrusted"}
        or metadata.get("trace_user_tenant_fallback_skipped") is True
        or metadata.get("repository_tenant_fallback_skipped") is True
        or (
            isinstance(repo_source, str)
            and (
                ".metadata.repository" in repo_source
                or "litellm_metadata.repository" in repo_source
            )
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

        repository = (
            _normalize_repository_identity(record.get("repository"))
            or _normalize_request_header_tenant_repository(raw_tenant_id)
        )
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

    metadata["usage_invalid_tool_call_count"] = int(
        record.get("invalid_tool_call_count") or 0
    )

    metadata["usage_structured_output_attempted"] = bool(
        record.get("structured_output_attempted")
    )
    metadata["usage_structured_output_failed"] = bool(
        record.get("structured_output_failed")
    )
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

    agent_score_reasons = _normalize_agent_score_reasons(
        record.get("agent_score_reasons")
    )
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
                command_text = _extract_command_text_from_tool_arguments(
                    item.get("arguments") or item.get("input")
                )
                if command_text:
                    commands.append(
                        AgentQualityCommand(
                            name=str(item.get("name") or item_type),
                            command=command_text,
                            affected_paths=tuple(
                                _extract_file_paths_from_tool_arguments(
                                    item.get("arguments") or item.get("input")
                                )
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
            for value in (
                list(item.get("file_paths_modified") or [])
                + list(item.get("file_paths_read") or [])
            )
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
    user_texts, assistant_texts, tool_result_texts, commands = (
        _collect_agent_quality_context_from_request_body(request_body)
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
        record[field] = (
            explicit_value if explicit_value is not None else derived_latency.get(field)
        )


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
    if (
        provider in {"gemini", "google"}
        and (
            metadata.get("aawm_rate_limit_observation_only") is True
            or has_gemini_quota_payload
            or gemini_control_plane_method is not None
        )
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
                record["model"] = _GEMINI_CONTROL_PLANE_METHOD_LABELS[
                    gemini_control_plane_method
                ]
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
        and str(metadata.get("passthrough_route_family") or "").strip().lower()
        == "grok_cli_chat_proxy"
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
        metadata["session_history_reporting_exclusion_reason"] = (
            "synthetic_codex_transcript"
        )

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
            modified_paths.extend(
                value
                for value in (item.get("file_paths_modified") or [])
                if isinstance(value, str)
            )

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
        token in normalized_name
        for token in ("read", "view", "grep", "glob", "search", "fetch")
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
                file_paths_modified = _dedupe_strings(
                    file_paths_modified + _extract_paths_from_patch_text(patch_text)
                )
    elif tool_kind == "command":
        command_text = _extract_command_text_from_tool_arguments(parsed_arguments)

    if command_text is None and tool_name.strip().lower() in {"apply_patch", "applypatch"}:
        command_text = _extract_command_text_from_tool_arguments(parsed_arguments)

    git_commit_count = 0
    git_push_count = 0
    if isinstance(command_text, str) and command_text:
        git_commit_count = _count_git_subcommand(command_text, "commit")
        git_push_count = _count_git_subcommand(command_text, "push")

    sensitive_config_flags = _sensitive_config_change_flags_from_paths(
        file_paths_modified
    )
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
_RESPONSE_OUTPUT_TOOL_ITEM_TYPES = set(_RESPONSE_OUTPUT_TOOL_ITEM_FALLBACK_NAMES) | {
    "function_call"
}


def _extract_response_output_items(
    result: Any, standard_logging_object: Optional[Dict[str, Any]] = None
) -> List[Any]:
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
        read_paths.extend(
            value for value in (item.get("file_paths_read") or []) if isinstance(value, str)
        )
        modified_paths.extend(
            value
            for value in (item.get("file_paths_modified") or [])
            if isinstance(value, str)
        )
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
            tool_name = _maybe_get(function_obj, "name") or _maybe_get(
                tool_call, "name"
            )
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
            tool_name = _maybe_get(_maybe_get(tool_call, "function"), "name") or _maybe_get(
                tool_call, "name"
            )
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


def _infer_usage_breakout_provider_prefix(
    kwargs: Dict[str, Any], metadata: Dict[str, Any]
) -> Optional[str]:
    route_family = metadata.get("passthrough_route_family")
    if isinstance(route_family, str) and route_family.strip():
        route_family_lower = route_family.lower()
        if route_family_lower == "codex_responses":
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
    metadata[f"{provider_prefix}_cache_creation_input_tokens"] = (
        cache_creation_input_tokens
    )
    metadata[f"{provider_prefix}_tool_call_count"] = tool_call_count
    metadata[f"{provider_prefix}_tool_names"] = tool_names

    if reported_reasoning_tokens is not None:
        metadata["usage_reasoning_tokens_reported"] = reported_reasoning_tokens
        metadata["usage_reasoning_tokens_source"] = (
            reasoning_tokens_source or "provider_reported"
        )
        metadata[f"{provider_prefix}_reasoning_tokens_reported"] = (
            reported_reasoning_tokens
        )

    tags_to_add = [f"{provider_prefix}-usage-breakout"]
    if reported_reasoning_tokens is not None:
        tags_to_add.extend(
            ["reasoning-tokens-reported", f"{provider_prefix}-reasoning-tokens-reported"]
        )
    if cache_read_input_tokens > 0:
        tags_to_add.extend(
            ["cache-read-input-tokens", f"{provider_prefix}-cache-read-input-tokens"]
        )
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


def _enrich_provider_cache_metadata(kwargs: Dict[str, Any], result: Any) -> None:  # noqa: PLR0915
    metadata = _ensure_mutable_metadata(kwargs)
    standard_logging_object = kwargs.get("standard_logging_object") or {}
    resolved_model = _resolve_session_history_model(
        kwargs=kwargs,
        standard_logging_object=standard_logging_object,
        metadata=metadata,
        result=result,
    )
    usage_obj = _extract_usage_object(kwargs, result)
    request_body = _extract_provider_cache_request_body(kwargs)
    response_cost_usd = _safe_float(
        _first_non_none(
            kwargs.get("response_cost"),
            (kwargs.get("standard_logging_object") or {}).get("response_cost"),
            metadata.get("litellm_response_cost"),
            metadata.get("response_cost"),
            metadata.get("usage_openrouter_cost"),
            _maybe_get(usage_obj, "cost"),
        )
    )
    provider_family = _normalize_provider_cache_family(
        kwargs.get("custom_llm_provider"),
        resolved_model,
        metadata,
    )
    cache_state = _resolve_provider_cache_state(
        provider=kwargs.get("custom_llm_provider"),
        model=resolved_model,
        usage_obj=usage_obj,
        metadata=metadata,
        request_body=request_body,
    )
    if provider_family is None or cache_state is None:
        return
    cache_miss_cost_state = _compute_provider_cache_miss_cost_state(
        provider_family=provider_family,
        model=resolved_model,
        usage_obj=usage_obj,
        cache_state=cache_state,
        metadata=metadata,
        response_cost_usd=response_cost_usd,
    )
    cache_state.update(cache_miss_cost_state)

    metadata["usage_provider_cache_attempted"] = cache_state["attempted"]
    metadata["usage_provider_cache_status"] = cache_state["status"]
    metadata["usage_provider_cache_miss"] = cache_state["miss"]
    if cache_state.get("miss_reason"):
        metadata["usage_provider_cache_miss_reason"] = cache_state["miss_reason"]
    if cache_state.get("miss_token_count") is not None:
        metadata["usage_provider_cache_miss_token_count"] = cache_state["miss_token_count"]
    if cache_state.get("miss_cost_usd") is not None:
        metadata["usage_provider_cache_miss_cost_usd"] = cache_state["miss_cost_usd"]
    if cache_state.get("miss_cost_basis"):
        metadata["usage_provider_cache_miss_cost_basis"] = cache_state["miss_cost_basis"]
    if cache_state.get("source"):
        metadata["usage_provider_cache_source"] = cache_state["source"]

    metadata[f"{provider_family}_provider_cache_attempted"] = cache_state["attempted"]
    metadata[f"{provider_family}_provider_cache_status"] = cache_state["status"]
    metadata[f"{provider_family}_provider_cache_miss"] = cache_state["miss"]
    if cache_state.get("miss_reason"):
        metadata[f"{provider_family}_provider_cache_miss_reason"] = cache_state[
            "miss_reason"
        ]
    if cache_state.get("miss_token_count") is not None:
        metadata[f"{provider_family}_provider_cache_miss_token_count"] = cache_state[
            "miss_token_count"
        ]
    if cache_state.get("miss_cost_usd") is not None:
        metadata[f"{provider_family}_provider_cache_miss_cost_usd"] = cache_state[
            "miss_cost_usd"
        ]
    if cache_state.get("miss_cost_basis"):
        metadata[f"{provider_family}_provider_cache_miss_cost_basis"] = cache_state[
            "miss_cost_basis"
        ]
    if cache_state.get("source"):
        metadata[f"{provider_family}_provider_cache_source"] = cache_state["source"]

    tags_to_add = []
    status = cache_state.get("status")
    if isinstance(status, str) and status in {"hit", "write", "miss", "unsupported"}:
        tags_to_add.extend(
            [
                f"provider-cache-status:{status}",
                f"{provider_family}-provider-cache-status:{status}",
            ]
        )
    if cache_state.get("miss") is True:
        tags_to_add.extend(
            [
                "provider-cache-miss",
                f"{provider_family}-provider-cache-miss",
            ]
        )
    if status == "hit":
        tags_to_add.extend(["provider-cache-hit", f"{provider_family}-provider-cache-hit"])
    elif status == "write":
        tags_to_add.extend(
            ["provider-cache-write", f"{provider_family}-provider-cache-write"]
        )
    elif status == "unsupported":
        tags_to_add.extend(
            ["provider-cache-unsupported", f"{provider_family}-provider-cache-unsupported"]
        )
    if cache_state.get("miss_reason") == "partial_cache_hit":
        tags_to_add.extend(
            ["provider-cache-partial-hit", f"{provider_family}-provider-cache-partial-hit"]
        )
    if tags_to_add:
        _merge_tags(metadata, tags_to_add)
        _append_langfuse_span(
            metadata,
            name=f"{provider_family}.provider_cache",
            span_metadata={
                "attempted": cache_state["attempted"],
                "status": cache_state["status"],
                "miss": cache_state["miss"],
                "miss_reason": cache_state.get("miss_reason"),
                "miss_token_count": cache_state.get("miss_token_count"),
                "miss_cost_usd": cache_state.get("miss_cost_usd"),
                "miss_cost_basis": cache_state.get("miss_cost_basis"),
                "source": cache_state.get("source"),
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
        reconstructed_usage.setdefault(
            "prompt_tokens", _safe_int(spend_log_row.get("prompt_tokens")) or 0
        )
        reconstructed_usage.setdefault(
            "completion_tokens", _safe_int(spend_log_row.get("completion_tokens")) or 0
        )
        reconstructed_usage.setdefault(
            "total_tokens", _safe_int(spend_log_row.get("total_tokens")) or 0
        )
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
            if spend_log_row.get("session_id") is not None
            and str(spend_log_row.get("session_id")).strip()
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
        agent_name, tenant_id = _extract_agent_context_from_text(
            _serialize_searchable_text(candidate)
        )
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
    output_usage_object = _build_usage_object_from_langfuse_output(
        observation.get("output")
    )
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
        usage_object.setdefault(
            "completion_tokens_details", completion_tokens_details
        )

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
        if any(
            key in output_payload
            for key in ("content", "tool_calls", "reasoning_content", "thinking_blocks")
        ):
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

    route_provider = _session_history_provider_from_route_family(
        metadata.get("passthrough_route_family")
    )
    if route_provider is not None:
        return route_provider

    api_base = (
        metadata.get("api_base")
        or _maybe_get(metadata.get("hidden_params"), "api_base")
        or observation.get("apiBase")
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
    normalized_tags = [
        str(tag) for tag in request_tags if isinstance(tag, str) and tag.strip()
    ] if isinstance(request_tags, list) else []

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
                    normalized_tags.append(
                        f"anthropic-billing-header:{key}={str(value).strip()}"
                    )

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
    normalized_trace_id = (
        str(trace_id).strip() if trace_id is not None and str(trace_id).strip() else None
    )

    derived_tags: List[str] = []
    existing_trace_tags = trace.get("tags")
    if isinstance(existing_trace_tags, list):
        derived_tags.extend(
            str(tag) for tag in existing_trace_tags if isinstance(tag, str) and tag.strip()
        )

    observations = trace.get("observations")
    if isinstance(observations, list):
        for observation in observations:
            if not isinstance(observation, dict) or observation.get("type") != "GENERATION":
                continue
            metadata = observation.get("metadata")
            if not isinstance(metadata, dict):
                continue
            derived_tags.extend(_derive_request_tags_from_langfuse_metadata(metadata))

    return normalized_trace_id, sorted(
        {tag for tag in derived_tags if isinstance(tag, str) and tag.strip()}
    )




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
    route_family = str(
        metadata.get("passthrough_route_family")
        or metadata.get("route_family")
        or metadata.get("openai_passthrough_route_family")
        or ""
    ).strip().lower()
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
    return [
        token.strip()
        for token in str(raw_value).replace(";", ",").split(",")
        if token.strip()
    ]


def _extract_context_1m_beta_values(
    headers: Optional[Dict[str, Any]],
    metadata: Dict[str, Any],
) -> List[str]:
    matched: List[str] = []
    seen: Set[str] = set()
    for raw in _iter_anthropic_beta_header_candidates(headers, metadata):
        for beta_value in _split_anthropic_beta_values(raw):
            beta_lower = beta_value.lower()
            if (
                beta_lower == _ANTHROPIC_CONTEXT_1M_BETA_HEADER.lower()
                or beta_lower.startswith(_ANTHROPIC_CONTEXT_1M_BETA_PREFIX)
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
    model_value = _clean_non_empty_string(
        resolved_model
        or metadata.get("model")
        or kwargs.get("model")
    ) or "unknown"
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

    return (
        urlunsplit((parsed.scheme, netloc, parsed.path.rstrip("/"), "", ""))
        or None
    )


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

    return (
        parsed_ip.is_loopback
        or parsed_ip.is_private
        or parsed_ip.is_link_local
        or parsed_ip.is_unspecified
    )


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
    return _first_non_empty_string(
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
    ) or "unknown"


def _resolve_inbound_model_alias_from_langfuse(
    *,
    observation: Dict[str, Any],
    metadata: Dict[str, Any],
    input_model: Optional[str],
    output_model: Optional[str],
    resolved_model: str,
) -> str:
    return _first_non_empty_string(
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
    ) or "unknown"


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
    route_model = _clean_non_empty_string(upstream_model) or _clean_non_empty_string(
        model_group
    )
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

    upstream_model = (
        _clean_non_empty_string(_strip_local_provider_model_prefix(resolved_model))
        or model_group
    )

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

    route_info = _LOCAL_BIOMED_SESSION_HISTORY_ROUTES.get(
        (parsed.port or 0, parsed.path.rstrip("/"))
    )
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
        record.get("aawm_tool_definition_snapshot_hash")
        or metadata.get("aawm_tool_definition_snapshot_hash")
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
        _clean_non_empty_string(
            metadata.get("aawm_tool_definition_capture_version")
        ),
        _clean_non_empty_string(metadata.get("aawm_tool_definition_capture_source")),
        _safe_int(metadata.get("aawm_tool_definition_count")),
        _safe_int(metadata.get("aawm_tool_definition_captured_count")),
        json.dumps(
            _json_safe_rate_limit_value(sources if isinstance(sources, list) else [])
        ),
        json.dumps(
            _json_safe_rate_limit_value(names if isinstance(names, list) else [])
        ),
        json.dumps(
            _json_safe_rate_limit_value(
                tool_types if isinstance(tool_types, list) else []
            )
        ),
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
    if (
        provider == "antigravity"
        or client_family == "antigravity_code_assist"
        or source.startswith("antigravity_")
    ):
        return "antigravity"
    if (
        provider in {"opencode", "opencode_zen"}
        or client_family == "opencode_zen"
        or source.startswith("opencode_")
    ):
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
        str(raw_provider_fields.get("tokenType") or "").lower()
        if isinstance(raw_provider_fields, dict)
        else ""
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

    remaining_fraction = _safe_float(
        _maybe_get_path(record.get("raw_provider_fields"), "remainingFraction")
    )
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
    litellm_call_id = _clean_non_empty_string(
        event.get("litellm_call_id") or record.get("litellm_call_id")
    )
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
    digest = hashlib.sha256(
        json.dumps(key_material, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()[:24]
    return f"{litellm_call_id}:alias-routing:{digest}"


def _infer_alias_routing_family(
    event: Dict[str, Any],
    metadata: Dict[str, Any],
) -> str:
    return (
        _clean_non_empty_string(event.get("alias_family"))
        or (
            "codex_auto_agent"
            if _clean_non_empty_string(metadata.get("codex_auto_agent_alias"))
            else None
        )
        or (
            "anthropic_auto_agent"
            if _clean_non_empty_string(metadata.get("anthropic_auto_agent_alias"))
            else None
        )
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
        _clean_non_empty_string(event.get("session_id"))
        or _clean_non_empty_string(record.get("session_id")),
        _clean_non_empty_string(event.get("session_key")),
        _clean_non_empty_string(event.get("trace_id"))
        or _clean_non_empty_string(record.get("trace_id")),
        _clean_non_empty_string(event.get("litellm_call_id"))
        or _clean_non_empty_string(record.get("litellm_call_id")),
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
        return {
            key: _maybe_get(row, key)
            for key in _AAWM_RATE_LIMIT_PREVIOUS_OBSERVATION_FIELDS
        }
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

    previous_by_limit_key: Dict[str, Optional[Dict[str, Any]]] = {
        limit_key: None for limit_key in limit_keys
    }
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
            previous_by_limit_key[limit_key] = (
                _rate_limit_previous_observation_row_to_dict(row)
            )
    return previous_by_limit_key


async def _derive_rate_limit_transitions(
    conn: Any,
    observations: List[Dict[str, Any]],
    initial_previous_by_limit_key: Optional[Dict[str, Optional[Dict[str, Any]]]] = None,
) -> List[Dict[str, Any]]:
    transitions: List[Dict[str, Any]] = []
    previous_by_limit_key: Dict[str, Optional[Dict[str, Any]]] = dict(
        initial_previous_by_limit_key or {}
    )
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
        if (
            isinstance(limit_key, str)
            and limit_key
            and limit_key not in previous_by_limit_key
        ):
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
                transitions.append(
                    _build_rate_limit_transition(previous, observation, classification)
                )
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
        if isinstance(observation.get("limit_key"), str)
        and observation.get("limit_key")
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
    metadata["claude_reasoning_content_empty_or_short"] = (
        len(reasoning_content.strip()) < 16
    )
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
            indexed_fields[
                f"gemini_tsig_0_record_{record_index}_marker_offset"
            ] = absolute_marker_offset

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
        signature
        for signature in thought_signatures
        if isinstance(signature, str) and signature.strip()
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
    metadata["gemini_reasoning_content_empty_or_short"] = (
        len(reasoning_content.strip()) < 16
    )
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
        tags_to_add.extend(
            ["gemini-thought-signature-decoded", "thinking-signature-decoded"]
        )
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
            "record_counts": sorted(
                {summary["record_count"] for summary in summaries} if summaries else []
            ),
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


def _enrich_trace_name_and_provider_metadata(
    kwargs: Dict[str, Any], result: Any
) -> Tuple[dict, Any]:
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
    elif is_grok_context and (
        not current_trace_name or _is_generic_grok_trace_name(current_trace_name)
    ):
        metadata["trace_name"] = (
            f"grok-build.{agent_name}"
            if agent_name and agent_name != _DEFAULT_AGENT
            else "grok-build"
        )
    elif not current_trace_name:
        metadata["trace_name"] = agent_name
    child_trace_user_id = _clean_non_empty_string(metadata.get("trace_user_id"))
    child_trace_name = _clean_non_empty_string(metadata.get("trace_name"))
    if headers and child_trace_name and child_trace_name.startswith("claude-code."):
        current_trace_name_header = _clean_non_empty_string(
            headers.get("langfuse_trace_name")
        )
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
        current_trace_name_header = _clean_non_empty_string(
            headers.get("langfuse_trace_name")
        )
        if (
            current_trace_name_header is None
            or _is_generic_grok_trace_name(current_trace_name_header)
        ) and current_trace_name_header != child_trace_name:
            headers["langfuse_trace_name"] = child_trace_name
            verbose_logger.debug(
                "AawmAgentIdentity: enriched Grok header trace_name to %s",
                child_trace_name,
            )
    if (
        headers
        and child_trace_user_id
        and child_trace_name
        and child_trace_name.startswith("claude-code.")
    ):
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

    def logging_hook(
        self, kwargs: Dict[str, Any], result: Any, call_type: str
    ) -> Tuple[dict, Any]:
        """Sync hook - runs before Langfuse in sync success handler."""
        try:
            return _enrich_trace_name_and_provider_metadata(kwargs, result)
        except Exception as exc:
            verbose_logger.warning("AawmAgentIdentity.logging_hook failed: %s", exc)
            return kwargs, result

    async def async_logging_hook(
        self, kwargs: Dict[str, Any], result: Any, call_type: str
    ) -> Tuple[dict, Any]:
        """Async hook - runs before Langfuse in async success handler."""
        try:
            return _enrich_trace_name_and_provider_metadata(kwargs, result)
        except Exception as exc:
            verbose_logger.warning(
                "AawmAgentIdentity.async_logging_hook failed: %s", exc
            )
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

    async def async_log_success_event(
        self, kwargs, response_obj, start_time, end_time
    ) -> None:
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

    async def async_log_failure_event(
        self, kwargs, response_obj, start_time, end_time
    ) -> None:
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
