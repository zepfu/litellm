"""AAWM observability callback for Langfuse attribution.

Extracts agent identity from the SubagentStart hook context injected into
request prompts, then enriches the langfuse_trace_name request header so
each agent's API calls can be distinguished in Langfuse.

The hook injects: "You are '<agent-name>' and you are working..."
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
"""

import ast
import asyncio
import atexit
import base64
import hashlib
import importlib
import json
import queue
import re
import shlex
import threading
import time
from datetime import datetime, timezone
from functools import lru_cache
from importlib import metadata as importlib_metadata
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote, urlencode, urlsplit, urlunsplit

from litellm._logging import verbose_logger
from litellm.integrations.custom_logger import CustomLogger
from litellm.secret_managers.main import get_secret_str

_CLAUDE_PERMISSION_CHECK_OUTPUT_RE = re.compile(
    r"^<block>\s*(?P<decision>yes|no)\s*$",
    re.IGNORECASE,
)
_AGENT_RE = re.compile(r"You are '([^']+)' and you are working")
_AGENT_TENANT_RE = re.compile(
    r"You are '(?P<agent>[^']+)' and you are working on the '(?P<tenant>[^']+)' project"
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
_AAWM_SESSION_HISTORY_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS public.session_history (
    id BIGSERIAL PRIMARY KEY,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    litellm_call_id TEXT UNIQUE,
    session_id TEXT NOT NULL,
    trace_id TEXT,
    provider_response_id TEXT,
    provider TEXT,
    model TEXT NOT NULL,
    model_group TEXT,
    agent_name TEXT,
    tenant_id TEXT,
    call_type TEXT,
    start_time TIMESTAMPTZ,
    end_time TIMESTAMPTZ,
    input_tokens INTEGER NOT NULL DEFAULT 0,
    output_tokens INTEGER NOT NULL DEFAULT 0,
    total_tokens INTEGER NOT NULL DEFAULT 0,
    cache_read_input_tokens INTEGER NOT NULL DEFAULT 0,
    cache_creation_input_tokens INTEGER NOT NULL DEFAULT 0,
    reasoning_tokens_reported INTEGER,
    reasoning_tokens_estimated INTEGER,
    reasoning_tokens_source TEXT,
    reasoning_present BOOLEAN NOT NULL DEFAULT FALSE,
    thinking_signature_present BOOLEAN NOT NULL DEFAULT FALSE,
    provider_cache_attempted BOOLEAN NOT NULL DEFAULT FALSE,
    provider_cache_status TEXT,
    provider_cache_miss BOOLEAN NOT NULL DEFAULT FALSE,
    provider_cache_miss_reason TEXT,
    provider_cache_miss_token_count INTEGER,
    provider_cache_miss_cost_usd DOUBLE PRECISION,
    tool_call_count INTEGER NOT NULL DEFAULT 0,
    tool_names JSONB NOT NULL DEFAULT '[]'::jsonb,
    file_read_count INTEGER NOT NULL DEFAULT 0,
    file_modified_count INTEGER NOT NULL DEFAULT 0,
    git_commit_count INTEGER NOT NULL DEFAULT 0,
    git_push_count INTEGER NOT NULL DEFAULT 0,
    response_cost_usd DOUBLE PRECISION,
    litellm_environment TEXT,
    litellm_version TEXT,
    litellm_fork_version TEXT,
    litellm_wheel_versions JSONB NOT NULL DEFAULT '{}'::jsonb,
    client_name TEXT,
    client_version TEXT,
    client_user_agent TEXT,
    token_permission_input INTEGER NOT NULL DEFAULT 0,
    token_permission_output INTEGER NOT NULL DEFAULT 0,
    permission_usd_cost DOUBLE PRECISION NOT NULL DEFAULT 0,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    repository TEXT
)
"""
_AAWM_SESSION_HISTORY_ALTER_STATEMENTS = (
    "ALTER TABLE public.session_history ADD COLUMN IF NOT EXISTS tenant_id TEXT",
    "ALTER TABLE public.session_history ADD COLUMN IF NOT EXISTS file_read_count INTEGER NOT NULL DEFAULT 0",
    "ALTER TABLE public.session_history ADD COLUMN IF NOT EXISTS file_modified_count INTEGER NOT NULL DEFAULT 0",
    "ALTER TABLE public.session_history ADD COLUMN IF NOT EXISTS git_commit_count INTEGER NOT NULL DEFAULT 0",
    "ALTER TABLE public.session_history ADD COLUMN IF NOT EXISTS git_push_count INTEGER NOT NULL DEFAULT 0",
    "ALTER TABLE public.session_history ADD COLUMN IF NOT EXISTS provider_cache_attempted BOOLEAN NOT NULL DEFAULT FALSE",
    "ALTER TABLE public.session_history ADD COLUMN IF NOT EXISTS provider_cache_status TEXT",
    "ALTER TABLE public.session_history ADD COLUMN IF NOT EXISTS provider_cache_miss BOOLEAN NOT NULL DEFAULT FALSE",
    "ALTER TABLE public.session_history ADD COLUMN IF NOT EXISTS provider_cache_miss_reason TEXT",
    "ALTER TABLE public.session_history ADD COLUMN IF NOT EXISTS provider_cache_miss_token_count INTEGER",
    "ALTER TABLE public.session_history ADD COLUMN IF NOT EXISTS provider_cache_miss_cost_usd DOUBLE PRECISION",
    "ALTER TABLE public.session_history ADD COLUMN IF NOT EXISTS litellm_environment TEXT",
    "ALTER TABLE public.session_history ADD COLUMN IF NOT EXISTS litellm_version TEXT",
    "ALTER TABLE public.session_history ADD COLUMN IF NOT EXISTS litellm_fork_version TEXT",
    "ALTER TABLE public.session_history ADD COLUMN IF NOT EXISTS litellm_wheel_versions JSONB NOT NULL DEFAULT '{}'::jsonb",
    "ALTER TABLE public.session_history ADD COLUMN IF NOT EXISTS client_name TEXT",
    "ALTER TABLE public.session_history ADD COLUMN IF NOT EXISTS client_version TEXT",
    "ALTER TABLE public.session_history ADD COLUMN IF NOT EXISTS client_user_agent TEXT",
    "ALTER TABLE public.session_history ADD COLUMN IF NOT EXISTS token_permission_input INTEGER NOT NULL DEFAULT 0",
    "ALTER TABLE public.session_history ADD COLUMN IF NOT EXISTS token_permission_output INTEGER NOT NULL DEFAULT 0",
    "ALTER TABLE public.session_history ADD COLUMN IF NOT EXISTS permission_usd_cost DOUBLE PRECISION NOT NULL DEFAULT 0",
    "ALTER TABLE public.session_history ADD COLUMN IF NOT EXISTS repository TEXT",
)
_AAWM_SESSION_HISTORY_INDEX_STATEMENTS = (
    "CREATE INDEX IF NOT EXISTS session_history_session_created_idx ON public.session_history (session_id, created_at DESC)",
    "CREATE INDEX IF NOT EXISTS session_history_session_model_created_idx ON public.session_history (session_id, model, created_at DESC)",
    "CREATE INDEX IF NOT EXISTS session_history_litellm_environment_created_idx ON public.session_history (litellm_environment, created_at DESC)",
    "CREATE INDEX IF NOT EXISTS session_history_client_created_idx ON public.session_history (client_name, client_version, created_at DESC)",
    "CREATE INDEX IF NOT EXISTS session_history_repository_created_idx ON public.session_history (repository, created_at DESC)",
)
_AAWM_SESSION_HISTORY_TOOL_ACTIVITY_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS public.session_history_tool_activity (
    id BIGSERIAL PRIMARY KEY,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    litellm_call_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    trace_id TEXT,
    provider TEXT,
    model TEXT NOT NULL,
    agent_name TEXT,
    tool_index INTEGER NOT NULL,
    tool_call_id TEXT,
    tool_name TEXT NOT NULL,
    tool_kind TEXT,
    file_paths_read JSONB NOT NULL DEFAULT '[]'::jsonb,
    file_paths_modified JSONB NOT NULL DEFAULT '[]'::jsonb,
    git_commit_count INTEGER NOT NULL DEFAULT 0,
    git_push_count INTEGER NOT NULL DEFAULT 0,
    command_text TEXT,
    arguments JSONB NOT NULL DEFAULT '{}'::jsonb,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    UNIQUE (litellm_call_id, tool_index)
)
"""
_AAWM_SESSION_HISTORY_TOOL_ACTIVITY_INDEX_STATEMENTS = (
    "CREATE INDEX IF NOT EXISTS session_history_tool_activity_session_created_idx ON public.session_history_tool_activity (session_id, created_at DESC)",
    "CREATE INDEX IF NOT EXISTS session_history_tool_activity_tool_name_idx ON public.session_history_tool_activity (tool_name)",
)
_AAWM_SESSION_HISTORY_INSERT_SQL = """
INSERT INTO public.session_history (
    litellm_call_id,
    session_id,
    trace_id,
    provider_response_id,
    provider,
    model,
    model_group,
    agent_name,
    tenant_id,
    call_type,
    start_time,
    end_time,
    input_tokens,
    output_tokens,
    total_tokens,
    cache_read_input_tokens,
    cache_creation_input_tokens,
    reasoning_tokens_reported,
    reasoning_tokens_estimated,
    reasoning_tokens_source,
    reasoning_present,
    thinking_signature_present,
    provider_cache_attempted,
    provider_cache_status,
    provider_cache_miss,
    provider_cache_miss_reason,
    provider_cache_miss_token_count,
    provider_cache_miss_cost_usd,
    tool_call_count,
    tool_names,
    file_read_count,
    file_modified_count,
    git_commit_count,
    git_push_count,
    response_cost_usd,
    litellm_environment,
    litellm_version,
    litellm_fork_version,
    litellm_wheel_versions,
    client_name,
    client_version,
    client_user_agent,
    token_permission_input,
    token_permission_output,
    permission_usd_cost,
    metadata,
    repository
) VALUES (
    $1, $2, $3, $4, $5, $6, $7, $8, $9, $10,
    $11, $12, $13, $14, $15, $16, $17, $18, $19, $20,
    $21, $22, $23, $24, $25, $26, $27, $28, $29, $30::jsonb,
    $31, $32, $33, $34, $35, $36, $37, $38, $39::jsonb, $40, $41, $42, $43, $44, $45, $46::jsonb, $47
)
ON CONFLICT (litellm_call_id) DO UPDATE SET
    session_id = COALESCE(NULLIF(EXCLUDED.session_id, ''), session_history.session_id),
    trace_id = COALESCE(NULLIF(EXCLUDED.trace_id, ''), session_history.trace_id),
    provider_response_id = COALESCE(
        NULLIF(EXCLUDED.provider_response_id, ''),
        session_history.provider_response_id
    ),
    provider = COALESCE(NULLIF(EXCLUDED.provider, ''), session_history.provider),
    model = COALESCE(NULLIF(EXCLUDED.model, ''), session_history.model),
    model_group = COALESCE(NULLIF(EXCLUDED.model_group, ''), session_history.model_group),
    agent_name = COALESCE(NULLIF(EXCLUDED.agent_name, ''), session_history.agent_name),
    tenant_id = COALESCE(NULLIF(EXCLUDED.tenant_id, ''), session_history.tenant_id),
    call_type = COALESCE(NULLIF(EXCLUDED.call_type, ''), session_history.call_type),
    start_time = COALESCE(session_history.start_time, EXCLUDED.start_time),
    end_time = COALESCE(EXCLUDED.end_time, session_history.end_time),
    input_tokens = GREATEST(session_history.input_tokens, EXCLUDED.input_tokens),
    output_tokens = GREATEST(session_history.output_tokens, EXCLUDED.output_tokens),
    total_tokens = GREATEST(session_history.total_tokens, EXCLUDED.total_tokens),
    cache_read_input_tokens = GREATEST(
        session_history.cache_read_input_tokens,
        EXCLUDED.cache_read_input_tokens
    ),
    cache_creation_input_tokens = GREATEST(
        session_history.cache_creation_input_tokens,
        EXCLUDED.cache_creation_input_tokens
    ),
    reasoning_tokens_reported = COALESCE(
        GREATEST(
            NULLIF(session_history.reasoning_tokens_reported, 0),
            NULLIF(EXCLUDED.reasoning_tokens_reported, 0)
        ),
        NULLIF(session_history.reasoning_tokens_reported, 0),
        NULLIF(EXCLUDED.reasoning_tokens_reported, 0)
    ),
    reasoning_tokens_estimated = COALESCE(
        GREATEST(
            NULLIF(session_history.reasoning_tokens_estimated, 0),
            NULLIF(EXCLUDED.reasoning_tokens_estimated, 0)
        ),
        NULLIF(session_history.reasoning_tokens_estimated, 0),
        NULLIF(EXCLUDED.reasoning_tokens_estimated, 0)
    ),
    reasoning_tokens_source = COALESCE(
        NULLIF(EXCLUDED.reasoning_tokens_source, ''),
        session_history.reasoning_tokens_source
    ),
    reasoning_present = session_history.reasoning_present OR EXCLUDED.reasoning_present,
    thinking_signature_present = session_history.thinking_signature_present OR EXCLUDED.thinking_signature_present,
    provider_cache_attempted = session_history.provider_cache_attempted OR EXCLUDED.provider_cache_attempted,
    provider_cache_status = COALESCE(
        NULLIF(EXCLUDED.provider_cache_status, ''),
        session_history.provider_cache_status
    ),
    provider_cache_miss = session_history.provider_cache_miss OR EXCLUDED.provider_cache_miss,
    provider_cache_miss_reason = COALESCE(
        NULLIF(EXCLUDED.provider_cache_miss_reason, ''),
        session_history.provider_cache_miss_reason
    ),
    provider_cache_miss_token_count = COALESCE(
        GREATEST(session_history.provider_cache_miss_token_count, EXCLUDED.provider_cache_miss_token_count),
        session_history.provider_cache_miss_token_count,
        EXCLUDED.provider_cache_miss_token_count
    ),
    provider_cache_miss_cost_usd = COALESCE(
        GREATEST(session_history.provider_cache_miss_cost_usd, EXCLUDED.provider_cache_miss_cost_usd),
        session_history.provider_cache_miss_cost_usd,
        EXCLUDED.provider_cache_miss_cost_usd
    ),
    tool_call_count = GREATEST(session_history.tool_call_count, EXCLUDED.tool_call_count),
    tool_names = CASE
        WHEN jsonb_array_length(EXCLUDED.tool_names) > jsonb_array_length(session_history.tool_names)
            THEN EXCLUDED.tool_names
        ELSE session_history.tool_names
    END,
    file_read_count = GREATEST(session_history.file_read_count, EXCLUDED.file_read_count),
    file_modified_count = GREATEST(session_history.file_modified_count, EXCLUDED.file_modified_count),
    git_commit_count = GREATEST(session_history.git_commit_count, EXCLUDED.git_commit_count),
    git_push_count = GREATEST(session_history.git_push_count, EXCLUDED.git_push_count),
    response_cost_usd = COALESCE(
        GREATEST(session_history.response_cost_usd, EXCLUDED.response_cost_usd),
        session_history.response_cost_usd,
        EXCLUDED.response_cost_usd
    ),
    token_permission_input = COALESCE(
        GREATEST(
            session_history.token_permission_input,
            EXCLUDED.token_permission_input
        ),
        session_history.token_permission_input,
        EXCLUDED.token_permission_input
    ),
    token_permission_output = COALESCE(
        GREATEST(
            session_history.token_permission_output,
            EXCLUDED.token_permission_output
        ),
        session_history.token_permission_output,
        EXCLUDED.token_permission_output
    ),
    permission_usd_cost = COALESCE(
        GREATEST(
            session_history.permission_usd_cost,
            EXCLUDED.permission_usd_cost
        ),
        session_history.permission_usd_cost,
        EXCLUDED.permission_usd_cost
    ),
    litellm_environment = COALESCE(
        NULLIF(EXCLUDED.litellm_environment, ''),
        session_history.litellm_environment
    ),
    litellm_version = COALESCE(
        NULLIF(EXCLUDED.litellm_version, ''),
        session_history.litellm_version
    ),
    litellm_fork_version = COALESCE(
        NULLIF(EXCLUDED.litellm_fork_version, ''),
        session_history.litellm_fork_version
    ),
    litellm_wheel_versions = COALESCE(session_history.litellm_wheel_versions, '{}'::jsonb) || COALESCE(EXCLUDED.litellm_wheel_versions, '{}'::jsonb),
    client_name = COALESCE(NULLIF(EXCLUDED.client_name, ''), session_history.client_name),
    client_version = COALESCE(
        NULLIF(EXCLUDED.client_version, ''),
        session_history.client_version
    ),
    client_user_agent = COALESCE(
        NULLIF(EXCLUDED.client_user_agent, ''),
        session_history.client_user_agent
    ),
    repository = COALESCE(NULLIF(EXCLUDED.repository, ''), session_history.repository),
    metadata = COALESCE(session_history.metadata, '{}'::jsonb) || COALESCE(EXCLUDED.metadata, '{}'::jsonb)
"""
_AAWM_SESSION_HISTORY_TOOL_ACTIVITY_INSERT_SQL = """
INSERT INTO public.session_history_tool_activity (
    litellm_call_id,
    session_id,
    trace_id,
    provider,
    model,
    agent_name,
    tool_index,
    tool_call_id,
    tool_name,
    tool_kind,
    file_paths_read,
    file_paths_modified,
    git_commit_count,
    git_push_count,
    command_text,
    arguments,
    metadata
) VALUES (
    $1, $2, $3, $4, $5, $6, $7, $8, $9, $10,
    $11::jsonb, $12::jsonb, $13, $14, $15, $16::jsonb, $17::jsonb
)
ON CONFLICT (litellm_call_id, tool_index) DO UPDATE SET
    session_id = COALESCE(NULLIF(EXCLUDED.session_id, ''), session_history_tool_activity.session_id),
    trace_id = COALESCE(NULLIF(EXCLUDED.trace_id, ''), session_history_tool_activity.trace_id),
    provider = COALESCE(NULLIF(EXCLUDED.provider, ''), session_history_tool_activity.provider),
    model = COALESCE(NULLIF(EXCLUDED.model, ''), session_history_tool_activity.model),
    agent_name = COALESCE(NULLIF(EXCLUDED.agent_name, ''), session_history_tool_activity.agent_name),
    tool_call_id = COALESCE(NULLIF(EXCLUDED.tool_call_id, ''), session_history_tool_activity.tool_call_id),
    tool_name = COALESCE(NULLIF(EXCLUDED.tool_name, ''), session_history_tool_activity.tool_name),
    tool_kind = COALESCE(NULLIF(EXCLUDED.tool_kind, ''), session_history_tool_activity.tool_kind),
    file_paths_read = CASE
        WHEN jsonb_array_length(EXCLUDED.file_paths_read) > jsonb_array_length(session_history_tool_activity.file_paths_read)
            THEN EXCLUDED.file_paths_read
        ELSE session_history_tool_activity.file_paths_read
    END,
    file_paths_modified = CASE
        WHEN jsonb_array_length(EXCLUDED.file_paths_modified) > jsonb_array_length(session_history_tool_activity.file_paths_modified)
            THEN EXCLUDED.file_paths_modified
        ELSE session_history_tool_activity.file_paths_modified
    END,
    git_commit_count = GREATEST(session_history_tool_activity.git_commit_count, EXCLUDED.git_commit_count),
    git_push_count = GREATEST(session_history_tool_activity.git_push_count, EXCLUDED.git_push_count),
    command_text = COALESCE(NULLIF(EXCLUDED.command_text, ''), session_history_tool_activity.command_text),
    arguments = COALESCE(session_history_tool_activity.arguments, '{}'::jsonb) || COALESCE(EXCLUDED.arguments, '{}'::jsonb),
    metadata = COALESCE(session_history_tool_activity.metadata, '{}'::jsonb) || COALESCE(EXCLUDED.metadata, '{}'::jsonb)
"""
_AAWM_SESSION_HISTORY_METADATA_KEYS = (
    "tenant_id_source",
    "trace_name",
    "trace_environment",
    "cc_version",
    "cc_entrypoint",
    "litellm_environment",
    "litellm_version",
    "litellm_fork_version",
    "litellm_wheel_versions",
    "client_name",
    "client_version",
    "client_user_agent",
    "repository",
    "route_tag",
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
    "reasoning_effort_requested",
    "reasoning_effort_source",
    "reasoning_effort_native_provider",
    "reasoning_effort_native_value",
    "reasoning_effort_native_field",
    "reasoning_effort_clamped_from",
    "reasoning_effort_clamp_reason",
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
    "usage_tool_call_count",
    "usage_tool_names",
    "google_adapter_system_prompt_policy_name",
    "google_adapter_system_prompt_policy",
    "google_adapter_system_prompt_policy_version",
    "google_adapter_system_prompt_original_chars",
    "google_adapter_system_prompt_rewritten_chars",
    "google_adapter_system_prompt_removed_claude_overhead_chars",
    "google_adapter_system_prompt_preserved_instruction_chars",
    "google_adapter_system_prompt_policy_applied",
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
)
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
)
_AAWM_REPOSITORY_HEADER_NAMES = (
    "x-aawm-repository",
    "x-litellm-repository",
    "x-repository",
    "x-git-repository",
)
_AAWM_REPOSITORY_TEXT_PATTERNS = (
    re.compile(r"AGENTS\.md instructions for\s+[`'\"]?(?P<path>/[^\n<`'\"]+)"),
    re.compile(r"<cwd>\s*[`'\"]?(?P<path>[^<`'\"]+)</cwd>"),
    re.compile(r"\bcwd\b\s*[:=]\s*[`'\"]?(?P<path>/[^`'\"\n<]+)"),
    re.compile(
        r"\*{0,2}Workspace Directories:\*{0,2}\s*\n\s*[-*]\s*[`'\"]?(?P<path>/[^\n`'\"]+)",
        re.IGNORECASE,
    ),
)
_AAWM_SESSION_HISTORY_BATCH_SIZE = 32
_AAWM_SESSION_HISTORY_FLUSH_INTERVAL_SECONDS = 0.25
_AAWM_SESSION_HISTORY_QUEUE_TIMEOUT_SECONDS = 0.1
_AAWM_SESSION_HISTORY_POOL_MAX_SIZE = 2
_AAWM_SESSION_HISTORY_OVERFLOW_FLUSHERS = 1
_aawm_session_history_schema_ready = False
_aawm_session_history_schema_lock = threading.Lock()
_aawm_session_history_queue: "queue.Queue[Optional[Dict[str, Any]]]" = queue.Queue(maxsize=1024)
_aawm_session_history_worker: Optional[threading.Thread] = None
_aawm_session_history_worker_lock = threading.Lock()
_aawm_session_history_pool_lock = threading.Lock()
_aawm_session_history_pools: Dict[Tuple[Any, str], Any] = {}
_aawm_session_history_overflow_flush_semaphore = threading.BoundedSemaphore(
    value=_AAWM_SESSION_HISTORY_OVERFLOW_FLUSHERS
)


def _get_session_history_batch_size() -> int:
    raw_value = get_secret_str("AAWM_SESSION_HISTORY_BATCH_SIZE") or ""
    try:
        parsed_value = int(str(raw_value).strip())
    except (TypeError, ValueError):
        parsed_value = _AAWM_SESSION_HISTORY_BATCH_SIZE
    return max(1, parsed_value)



def _get_session_history_flush_interval_seconds() -> float:
    raw_value = get_secret_str("AAWM_SESSION_HISTORY_FLUSH_INTERVAL_MS") or ""
    try:
        parsed_value = float(str(raw_value).strip()) / 1000.0
    except (TypeError, ValueError):
        parsed_value = _AAWM_SESSION_HISTORY_FLUSH_INTERVAL_SECONDS
    return max(0.01, parsed_value)


def _get_session_history_pool_max_size() -> int:
    raw_value = get_secret_str("AAWM_SESSION_HISTORY_POOL_MAX_SIZE") or ""
    try:
        parsed_value = int(str(raw_value).strip())
    except (TypeError, ValueError):
        parsed_value = _AAWM_SESSION_HISTORY_POOL_MAX_SIZE
    return max(1, parsed_value)


async def _close_aawm_session_history_pools_for_current_loop() -> None:
    loop = asyncio.get_running_loop()
    pools_to_close: List[Any] = []
    with _aawm_session_history_pool_lock:
        for key, pool in list(_aawm_session_history_pools.items()):
            if key[0] is loop:
                pools_to_close.append(pool)
                del _aawm_session_history_pools[key]

    for pool in pools_to_close:
        await pool.close()


def _flush_session_history_batch(
    records: List[Dict[str, Any]],
    loop: Optional[asyncio.AbstractEventLoop] = None,
) -> None:
    if not records:
        return

    started_at = time.perf_counter()
    try:
        if loop is None:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(_persist_session_history_records(records))
            finally:
                loop.run_until_complete(
                    _close_aawm_session_history_pools_for_current_loop()
                )
                loop.close()
                asyncio.set_event_loop(None)
        else:
            loop.run_until_complete(_persist_session_history_records(records))
    except Exception as exc:
        verbose_logger.warning(
            "AawmAgentIdentity: failed to flush %d session_history records: %s",
            len(records),
            exc,
        )
        return

    verbose_logger.debug(
        "AawmAgentIdentity: flushed %d session_history records in %.2fms",
        len(records),
        (time.perf_counter() - started_at) * 1000.0,
    )


def _flush_session_history_overflow_record(record: Dict[str, Any]) -> None:
    try:
        _flush_session_history_batch([record])
    finally:
        _aawm_session_history_overflow_flush_semaphore.release()


def _session_history_worker_main() -> None:
    flush_interval = _get_session_history_flush_interval_seconds()
    batch_size = _get_session_history_batch_size()
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    try:
        while True:
            try:
                first_item = _aawm_session_history_queue.get(timeout=flush_interval)
            except queue.Empty:
                continue

            if first_item is None:
                break

            batch: List[Dict[str, Any]] = [first_item]
            deadline = time.monotonic() + flush_interval
            while len(batch) < batch_size:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    next_item = _aawm_session_history_queue.get(timeout=remaining)
                except queue.Empty:
                    break
                if next_item is None:
                    _flush_session_history_batch(batch, loop=loop)
                    return
                batch.append(next_item)

            _flush_session_history_batch(batch, loop=loop)
    finally:
        loop.run_until_complete(_close_aawm_session_history_pools_for_current_loop())
        loop.close()



def _ensure_session_history_worker_started() -> None:
    global _aawm_session_history_worker

    if _aawm_session_history_worker is not None and _aawm_session_history_worker.is_alive():
        return

    with _aawm_session_history_worker_lock:
        if _aawm_session_history_worker is not None and _aawm_session_history_worker.is_alive():
            return

        _aawm_session_history_worker = threading.Thread(
            target=_session_history_worker_main,
            name="aawm-session-history-writer",
            daemon=True,
        )
        _aawm_session_history_worker.start()



def _shutdown_session_history_worker() -> None:
    worker = _aawm_session_history_worker
    if worker is None:
        return

    try:
        _aawm_session_history_queue.put(
            None,
            timeout=max(
                _AAWM_SESSION_HISTORY_QUEUE_TIMEOUT_SECONDS,
                _get_session_history_flush_interval_seconds(),
            ),
        )
    except queue.Full:
        verbose_logger.warning(
            "AawmAgentIdentity: session_history queue full during shutdown; worker pool cleanup may be delayed"
        )

    worker.join(timeout=1.0)



def _enqueue_session_history_record(record: Dict[str, Any]) -> None:
    _ensure_session_history_worker_started()
    try:
        _aawm_session_history_queue.put(record, timeout=_AAWM_SESSION_HISTORY_QUEUE_TIMEOUT_SECONDS)
    except queue.Full:
        if not _aawm_session_history_overflow_flush_semaphore.acquire(blocking=False):
            try:
                _aawm_session_history_queue.put(
                    record,
                    timeout=max(
                        _AAWM_SESSION_HISTORY_QUEUE_TIMEOUT_SECONDS,
                        _get_session_history_flush_interval_seconds(),
                    ),
                )
                return
            except queue.Full:
                pass

            verbose_logger.warning(
                "AawmAgentIdentity: session_history queue full and overflow flusher busy; dropping overflow record"
            )
            return

        verbose_logger.warning(
            "AawmAgentIdentity: session_history queue full; flushing overflow record in background"
        )
        try:
            threading.Thread(
                target=_flush_session_history_overflow_record,
                args=(record,),
                name="aawm-session-history-overflow",
                daemon=True,
            ).start()
        except Exception as exc:
            verbose_logger.warning(
                "AawmAgentIdentity: failed to start session_history overflow flusher; flushing inline: %s",
                exc,
            )
            try:
                _flush_session_history_batch([record])
            finally:
                _aawm_session_history_overflow_flush_semaphore.release()


atexit.register(_shutdown_session_history_worker)


def _content_to_text(content: Any) -> str:
    """Convert message content (string or Anthropic content blocks) to plain text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for block in content:
            if isinstance(block, dict):
                parts.append(str(block.get("text", "")))
            else:
                parts.append(str(block))
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
    for key, nested_value in parsed_value.items():
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
    for key, value in headers.items():
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
        merged.update(headers)
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
            tenant_id = _clean_non_empty_string(source.get(key))
            if tenant_id:
                return tenant_id, f"{source_name}.{key}"

        for nested_key in ("metadata", "request_metadata", "user_api_key_metadata"):
            nested_source = _coerce_mapping(source.get(nested_key))
            if not nested_source:
                continue
            for key in _AAWM_TENANT_ID_METADATA_KEYS:
                tenant_id = _clean_non_empty_string(nested_source.get(key))
                if tenant_id:
                    return tenant_id, f"{source_name}.{nested_key}.{key}"

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
    if tenant_id:
        return tenant_id, source

    headers = _extract_request_headers_from_kwargs(kwargs)
    tenant_id = _get_header_value(headers, *_AAWM_TENANT_ID_HEADER_NAMES)
    if tenant_id:
        return tenant_id, "request_headers"

    return None, None


def _extract_tenant_identity_from_langfuse_trace_observation(
    trace: Dict[str, Any],
    observation: Dict[str, Any],
    metadata: Optional[Dict[str, Any]] = None,
) -> Tuple[Optional[str], Optional[str]]:
    trace_metadata = trace.get("metadata") if isinstance(trace, dict) else None
    return _extract_tenant_identity_from_metadata_sources(
        ("observation.metadata", metadata or observation.get("metadata")),
        ("trace.metadata", trace_metadata),
        ("observation", observation),
        ("trace", trace),
    )


def _normalize_repository_identity(value: Any) -> Optional[str]:
    cleaned = _clean_non_empty_string(value)
    if not cleaned:
        return None
    cleaned = cleaned.strip("`'\"")

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
        cleaned = cleaned.rstrip("/").rsplit("/", 1)[-1]

    if cleaned.endswith(".git"):
        cleaned = cleaned[:-4]
    return cleaned.strip().strip("/") or None


def _extract_repository_identity_from_text(value: str) -> Optional[str]:
    for pattern in _AAWM_REPOSITORY_TEXT_PATTERNS:
        match = pattern.search(value)
        if not match:
            continue
        repository = _normalize_repository_identity(match.group("path"))
        if repository:
            return repository
    return None


def _extract_repository_identity_from_value(value: Any) -> Optional[str]:
    if isinstance(value, str):
        return _extract_repository_identity_from_text(value)
    if isinstance(value, dict):
        for key, child in value.items():
            if key in _AAWM_REPOSITORY_METADATA_KEYS:
                repository = _normalize_repository_identity(child)
                if repository:
                    return repository
            repository = _extract_repository_identity_from_value(child)
            if repository:
                return repository
    if isinstance(value, list):
        for child in value:
            repository = _extract_repository_identity_from_value(child)
            if repository:
                return repository
    return None


def _extract_repository_identity_from_metadata_sources(
    *sources: Tuple[str, Any],
) -> Optional[str]:
    for _source_name, raw_source in sources:
        source = _coerce_mapping(raw_source)
        if not source:
            continue
        for key in _AAWM_REPOSITORY_METADATA_KEYS:
            repository = _normalize_repository_identity(source.get(key))
            if repository:
                return repository

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
                    return repository

        repository = _extract_repository_identity_from_value(source)
        if repository:
            return repository

    return None


def _extract_repository_identity_from_kwargs(
    kwargs: Dict[str, Any],
    *,
    metadata: Optional[Dict[str, Any]] = None,
    standard_logging_object: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    litellm_params = kwargs.get("litellm_params") or {}
    standard_logging_object = standard_logging_object or kwargs.get("standard_logging_object") or {}
    passthrough_payload = kwargs.get("passthrough_logging_payload") or {}
    proxy_request = _coerce_mapping(litellm_params.get("proxy_server_request"))
    proxy_body = _coerce_mapping(proxy_request.get("body"))
    passthrough_body = _coerce_mapping(passthrough_payload.get("request_body"))

    repository = _extract_repository_identity_from_metadata_sources(
        ("litellm_params.metadata", metadata or litellm_params.get("metadata")),
        ("litellm_params.litellm_metadata", litellm_params.get("litellm_metadata")),
        ("standard_logging_object.metadata", standard_logging_object.get("metadata")),
        ("kwargs.metadata", kwargs.get("metadata")),
        ("litellm_params.proxy_server_request.body", proxy_body),
        ("litellm_params.proxy_server_request.body.metadata", proxy_body.get("metadata")),
        ("litellm_params.proxy_server_request.body.litellm_metadata", proxy_body.get("litellm_metadata")),
        ("passthrough_logging_payload", passthrough_payload),
        ("passthrough_logging_payload.request_body", passthrough_body),
        ("passthrough_logging_payload.request_body.metadata", passthrough_body.get("metadata")),
        ("passthrough_logging_payload.request_body.litellm_metadata", passthrough_body.get("litellm_metadata")),
        ("standard_logging_object", standard_logging_object),
        ("kwargs", kwargs),
    )
    if repository:
        return repository

    headers = _extract_request_headers_from_kwargs(kwargs)
    for header_name in _AAWM_REPOSITORY_HEADER_NAMES:
        repository = _normalize_repository_identity(
            _get_header_value(headers, header_name)
        )
        if repository:
            return repository

    return None


def _extract_repository_identity_from_langfuse_trace_observation(
    trace: Dict[str, Any],
    observation: Dict[str, Any],
    metadata: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    trace_metadata = trace.get("metadata") if isinstance(trace, dict) else None
    return _extract_repository_identity_from_metadata_sources(
        ("observation.metadata", metadata or observation.get("metadata")),
        ("trace.metadata", trace_metadata),
        ("observation", observation),
        ("trace", trace),
    )


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
        (re.compile(r"\bGeminiCLI/(?P<version>[A-Za-z0-9.+_-]+)"), "gemini-cli"),
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
    elif cc_version and client_name is None:
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

    for key, value in identity.items():
        if key == "litellm_wheel_versions":
            if isinstance(value, dict) and value:
                metadata[key] = value
            continue
        if value is not None:
            metadata[key] = value


def _extract_agent_context_from_text(text: str) -> Tuple[Optional[str], Optional[str]]:
    tenant_match = _AGENT_TENANT_RE.search(text)
    if tenant_match:
        return tenant_match.group("agent"), tenant_match.group("tenant")

    agent_match = _AGENT_RE.search(text)
    if agent_match:
        return agent_match.group(1), None

    return None, None


def _extract_agent_context(kwargs: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    """Extract agent/tenant from request content when present."""
    explicit_tenant_id, _tenant_source = _extract_tenant_identity_from_kwargs(kwargs)
    litellm_params = kwargs.get("litellm_params") or {}
    metadata = litellm_params.get("metadata") or {}
    standard_logging_object = kwargs.get("standard_logging_object") or {}
    standard_metadata = standard_logging_object.get("metadata") or {}
    for source in (metadata, standard_metadata):
        if not isinstance(source, dict):
            continue
        agent_name = _clean_non_empty_string(
            source.get("agent_name") or source.get("aawm_claude_agent_name")
        )
        tenant_id = _clean_non_empty_string(
            source.get("tenant_id")
            or source.get("aawm_tenant_id")
            or source.get("aawm_claude_project")
        )
        if agent_name:
            return agent_name, explicit_tenant_id or tenant_id

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

    return None, explicit_tenant_id


def _extract_agent_name(kwargs: Dict[str, Any]) -> str:
    agent_name, _tenant_id = _extract_agent_context(kwargs)
    return agent_name or _DEFAULT_AGENT


def _ensure_mutable_headers(kwargs: Dict[str, Any]) -> dict:
    """Ensure proxy_server_request.headers is a mutable dict."""
    litellm_params = kwargs.get("litellm_params") or {}
    psr = litellm_params.get("proxy_server_request") or {}
    headers = psr.get("headers")

    if headers is None:
        return {}

    if not isinstance(headers, dict):
        headers = dict(headers)
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
    standard_logging_metadata.update(metadata)
    standard_logging_object["metadata"] = standard_logging_metadata

    tags = metadata.get("tags") or []
    if not isinstance(tags, list):
        tags = []
    existing_request_tags = standard_logging_object.get("request_tags") or []
    if not isinstance(existing_request_tags, list):
        existing_request_tags = []

    merged_request_tags = list(existing_request_tags)
    for tag in tags:
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


def _extract_claude_permission_check_decision_from_value(value: Any) -> Optional[str]:
    if value is None:
        return None
    if type(value).__module__.startswith("unittest.mock"):
        return None

    if isinstance(value, str):
        stripped_value = value.strip()
        match = _CLAUDE_PERMISSION_CHECK_OUTPUT_RE.match(stripped_value)
        if match is not None:
            return match.group("decision").lower()
        parsed_value = _maybe_parse_json_text(stripped_value)
        if parsed_value is not None:
            return _extract_claude_permission_check_decision_from_value(parsed_value)
        return None

    if isinstance(value, list):
        text_value = _content_to_text(value).strip()
        match = _CLAUDE_PERMISSION_CHECK_OUTPUT_RE.match(text_value)
        if match is not None:
            return match.group("decision").lower()
        for item in value:
            decision = _extract_claude_permission_check_decision_from_value(item)
            if decision is not None:
                return decision
        return None

    content = _maybe_get(value, "content")
    if content is not None:
        decision = _extract_claude_permission_check_decision_from_value(content)
        if decision is not None:
            return decision

    message = _extract_first_response_message(value)
    if message is not None and message is not value:
        decision = _extract_claude_permission_check_decision_from_value(message)
        if decision is not None:
            return decision

    response = _maybe_get(value, "response")
    if response is not None and response is not value:
        decision = _extract_claude_permission_check_decision_from_value(response)
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


def _first_non_empty_string(*values: Any) -> Optional[str]:
    for value in values:
        if isinstance(value, str) and value.strip():
            return value.strip()
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


def _first_non_none(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


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


def _extract_responses_completed_payload_from_passthrough_fallback_text(
    response_text: Any,
) -> Optional[Dict[str, Any]]:
    if not isinstance(response_text, str) or "Chunks=" not in response_text:
        return None

    try:
        chunks = ast.literal_eval(response_text.split("Chunks=", 1)[1].strip())
    except Exception:
        return None
    if not isinstance(chunks, list):
        return None

    try:
        from litellm.llms.base_llm.base_model_iterator import (
            BaseModelResponseIterator,
        )
    except Exception:
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
    if isinstance(usage_object, dict) and usage_object:
        return dict(usage_object)

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
        return None

    reconstructed: Dict[str, Any] = {}
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
        reconstructed["input_tokens_details"] = {
            "cached_tokens": cache_read_input_tokens
        }
    if cache_creation_input_tokens is not None:
        reconstructed["cache_creation_input_tokens"] = cache_creation_input_tokens
    if reasoning_tokens_reported is not None:
        reconstructed["reasoning_tokens"] = reasoning_tokens_reported
        output_tokens_details = dict(reconstructed.get("output_tokens_details") or {})
        output_tokens_details["reasoning_tokens"] = reasoning_tokens_reported
        reconstructed["output_tokens_details"] = output_tokens_details

    return reconstructed or None


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
    for key, value in metadata_usage_object.items():
        if key not in merged_usage or merged_usage.get(key) in (None, {}, []):
            merged_usage[key] = value

    return merged_usage



def _extract_usage_object(kwargs: Dict[str, Any], result: Any) -> Any:
    usage_obj = _maybe_get(result, "usage")
    metadata_usage_object = _extract_metadata_usage_object(kwargs)
    if usage_obj is not None:
        return _merge_usage_object_with_metadata(usage_obj, metadata_usage_object)

    meta_obj = _maybe_get(result, "meta")
    billed_units = _maybe_get(meta_obj, "billed_units")
    if billed_units is not None:
        search_units = _safe_int(_maybe_get(billed_units, "search_units"))
        total_tokens = _safe_int(_maybe_get(billed_units, "total_tokens"))
        rerank_usage: Dict[str, Any] = {
            "prompt_tokens": total_tokens,
            "completion_tokens": 0,
            "total_tokens": total_tokens,
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
    "anthropic",
    "openai",
    "openrouter",
    "gemini",
    "nvidia",
}


def _has_nested_path(obj: Any, *keys: str) -> bool:
    sentinel = object()
    return _maybe_get_path(obj, *keys, default=sentinel) is not sentinel


def _normalize_provider_cache_family(
    provider: Any,
    model: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    route_family = (metadata or {}).get("passthrough_route_family")
    if isinstance(route_family, str) and route_family.strip():
        route_family_lower = route_family.lower()
        if "nvidia" in route_family_lower:
            return "nvidia"
        if "openrouter" in route_family_lower:
            return "openrouter"
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
        if provider_lower in {"nvidia_nim", "nvidia-nim"}:
            return "nvidia"
        if provider_lower in _PROVIDER_CACHE_TARGET_FAMILIES:
            return provider_lower

    model_lower = str(model or "").strip().lower()
    if model_lower.startswith("nvidia_nim/") or model_lower.startswith("nvidia/"):
        return "nvidia"
    if model_lower.startswith("openrouter/"):
        return "openrouter"
    if "gemini" in model_lower:
        return "gemini"
    if "claude" in model_lower or model_lower.startswith("anthropic/"):
        return "anthropic"
    if model_lower.startswith("gpt") or model_lower.startswith("o1") or model_lower.startswith("o3"):
        return "openai"
    if model_lower.startswith("openai/") or "codex" in model_lower:
        return "openai"
    return None


def _normalize_session_history_provider(
    provider: Any,
    model: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    metadata = metadata or {}

    def _normalize_known_provider(candidate: Any) -> Optional[str]:
        if not isinstance(candidate, str) or not candidate.strip():
            return None
        candidate_lower = candidate.strip().lower()
        if candidate_lower in {"unknown", "none", "null"}:
            return None
        if candidate_lower == "google":
            return "gemini"
        if candidate_lower in {"nvidia", "nvidia_nim", "nvidia-nim"}:
            return "nvidia_nim"
        return candidate_lower

    normalized_provider = _normalize_known_provider(provider)
    if normalized_provider is not None:
        return normalized_provider

    for key in ("custom_llm_provider", "provider", "litellm_provider"):
        normalized_provider = _normalize_known_provider(metadata.get(key))
        if normalized_provider is not None:
            return normalized_provider

    route_family = metadata.get("passthrough_route_family")
    if isinstance(route_family, str) and route_family.strip():
        route_lower = route_family.lower()
        if "nvidia" in route_lower:
            return "nvidia_nim"
        if "openrouter" in route_lower:
            return "openrouter"
        if "gemini" in route_lower or "google" in route_lower:
            return "gemini"
        if "codex" in route_lower or "openai" in route_lower:
            return "openai"
        if "anthropic" in route_lower:
            return "anthropic"

    request_route = metadata.get("user_api_key_request_route")
    if isinstance(request_route, str) and request_route.strip():
        route_lower = request_route.lower()
        if route_lower.startswith("/anthropic/"):
            return "anthropic"
        if "gemini" in route_lower or "google" in route_lower:
            return "gemini"
        if route_lower.startswith("/v1/"):
            return "openai"

    api_base = metadata.get("api_base") or _maybe_get(metadata.get("hidden_params"), "api_base")
    if isinstance(api_base, str) and api_base.strip():
        api_base_lower = api_base.lower()
        if "integrate.api.nvidia.com" in api_base_lower:
            return "nvidia_nim"
        if "openrouter.ai" in api_base_lower:
            return "openrouter"
        if "anthropic.com" in api_base_lower:
            return "anthropic"
        if "googleapis.com" in api_base_lower or "generativelanguage" in api_base_lower:
            return "gemini"
        if "openai.com" in api_base_lower:
            return "openai"

    model_lower = str(model or "").strip().lower()
    if not model_lower or model_lower == "unknown":
        return None
    if model_lower.startswith("nvidia/"):
        return "nvidia_nim"
    if model_lower.startswith("openrouter/"):
        return "openrouter"
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
    return None


def _supports_prompt_caching_safe(
    *,
    model: str,
    provider: Optional[str],
) -> Optional[bool]:
    normalized_model = str(model or "").strip()
    if not normalized_model or normalized_model.lower() == "unknown":
        return None
    try:
        import litellm

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


def _request_contains_cache_control(payload: Any) -> bool:
    if isinstance(payload, dict):
        if payload.get("cache_control") is not None or payload.get("cacheControl") is not None:
            return True
        return any(_request_contains_cache_control(value) for value in payload.values())
    if isinstance(payload, list):
        return any(_request_contains_cache_control(item) for item in payload)
    return False


def _request_contains_cached_content(payload: Any) -> bool:
    if isinstance(payload, dict):
        cached_content = payload.get("cachedContent")
        if isinstance(cached_content, str) and cached_content.strip():
            return True
        cached_content_alias = payload.get("cached_content")
        if isinstance(cached_content_alias, str) and cached_content_alias.strip():
            return True
        return any(_request_contains_cached_content(value) for value in payload.values())
    if isinstance(payload, list):
        return any(_request_contains_cached_content(item) for item in payload)
    return False


def _request_contains_prompt_cache_key(payload: Any) -> bool:
    if not isinstance(payload, dict):
        return False
    prompt_cache_key = payload.get("prompt_cache_key")
    return isinstance(prompt_cache_key, str) and bool(prompt_cache_key.strip())


def _usage_has_openai_style_cached_tokens_field(usage_obj: Any) -> bool:
    return any(
        _has_nested_path(usage_obj, *path)
        for path in (
            ("input_tokens_details", "cached_tokens"),
            ("input_tokens_details", "cachedTokens"),
            ("inputTokensDetails", "cached_tokens"),
            ("inputTokensDetails", "cachedTokens"),
        )
    )


def _usage_has_gemini_style_cached_content_field(usage_obj: Any) -> bool:
    return _has_nested_path(usage_obj, "cachedContentTokenCount")


def _openai_cache_attempt_source(
    usage_obj: Any, request_body: Optional[Dict[str, Any]]
) -> Optional[Tuple[str, str]]:
    if _request_contains_prompt_cache_key(request_body):
        return "prompt_cache_key_requested_without_hit", "request.prompt_cache_key"
    if _usage_has_openai_style_cached_tokens_field(usage_obj):
        return "cached_tokens_reported_zero", "usage.input_tokens_details.cached_tokens"
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


def _compute_provider_cache_miss_cost_state(
    *,
    provider_family: Optional[str],
    model: str,
    usage_obj: Any,
    cache_state: Optional[Dict[str, Any]],
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "miss_token_count": None,
        "miss_cost_usd": None,
        "miss_cost_basis": None,
    }
    if provider_family is None or cache_state is None:
        return result
    if cache_state.get("status") != "write":
        return result

    cache_creation_input_tokens = _extract_cache_creation_input_tokens(usage_obj)
    if cache_creation_input_tokens <= 0:
        return result

    service_tier = _extract_service_tier_hint(usage_obj, metadata)
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
            custom_llm_provider=provider_family,
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
        result["miss_token_count"] = cache_creation_input_tokens
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


def _resolve_provider_cache_state(
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
    if state_from_metadata is not None and state_from_metadata.get("status") is not None:
        return state_from_metadata

    request_has_cache_control = _request_contains_cache_control(request_body)
    request_has_cached_content = _request_contains_cached_content(request_body)
    usage_has_openai_cached_tokens = _usage_has_openai_style_cached_tokens_field(usage_obj)
    usage_has_gemini_cached_content = _usage_has_gemini_style_cached_content_field(usage_obj)
    supports_prompt_caching = _supports_prompt_caching_safe(
        model=model,
        provider=provider_family,
    )

    if cache_read_input_tokens > 0:
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
            source = source or "usage.input_tokens_details.cached_tokens"
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
        import litellm

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
            and candidate.get("documents") is not None
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
        import litellm

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

    candidates = [model]
    if custom_llm_provider:
        provider_prefix = f"{custom_llm_provider}/"
        if model.startswith(provider_prefix):
            stripped_model = model[len(provider_prefix) :]
            candidates.append(stripped_model)
        else:
            candidates.append(f"{provider_prefix}{model}")

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
    if search_units and input_cost_per_query is not None:
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
    should_override = (
        not isinstance(current_status, str)
        or not current_status.strip()
        or bool(record.get("cache_read_input_tokens") or record.get("cache_creation_input_tokens"))
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
    ):
        if not _clean_non_empty_string(record.get(key)):
            record[key] = identity.get(key)

    record_wheel_versions = _coerce_string_dict(record.get("litellm_wheel_versions"))
    metadata_wheel_versions = _coerce_string_dict(identity.get("litellm_wheel_versions"))
    record["litellm_wheel_versions"] = {
        **metadata_wheel_versions,
        **record_wheel_versions,
    }


def _normalize_session_repository_on_record(record: Dict[str, Any]) -> None:
    repository = _normalize_repository_identity(record.get("repository"))
    if repository is None:
        metadata = record.get("metadata")
        if isinstance(metadata, dict):
            repository = _extract_repository_identity_from_metadata_sources(
                ("record.metadata", metadata)
            )
    record["repository"] = repository


def _normalize_session_tenant_on_record(record: Dict[str, Any]) -> None:
    tenant_id = _clean_non_empty_string(record.get("tenant_id"))
    if tenant_id:
        record["tenant_id"] = tenant_id
        return

    repository = _normalize_repository_identity(record.get("repository"))
    if repository is None:
        record["tenant_id"] = None
        return

    record["tenant_id"] = repository
    metadata = record.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
    else:
        metadata = dict(metadata)
    metadata["tenant_id"] = repository
    metadata["tenant_id_source"] = "repository"
    record["metadata"] = metadata


def _sync_session_history_record_metadata(record: Dict[str, Any]) -> None:
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

    provider_family = _normalize_provider_cache_family(
        record.get("provider"),
        str(record.get("model") or ""),
        metadata,
    )
    cache_status = record.get("provider_cache_status")
    if provider_family is not None and isinstance(cache_status, str) and cache_status.strip():
        cache_values = {
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

    tenant_id = _clean_non_empty_string(record.get("tenant_id"))
    if tenant_id is not None:
        metadata["tenant_id"] = tenant_id

    record["metadata"] = metadata


def _normalize_session_history_record(record: Dict[str, Any]) -> Dict[str, Any]:
    _normalize_reasoning_state(record)
    _normalize_provider_cache_state_on_record(record)
    _normalize_session_runtime_identity_on_record(record)
    _normalize_session_repository_on_record(record)
    _normalize_session_tenant_on_record(record)
    _sync_session_history_record_metadata(record)
    return record


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
        for nested_key, nested_value in value.items():
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

    for key, nested_value in value.items():
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
        "arguments": parsed_arguments,
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


def _maybe_get_path(obj: Any, *keys: str, default: Any = None) -> Any:
    """Safely traverse nested dict/object paths."""
    current = obj
    for key in keys:
        if isinstance(current, dict):
            current = current.get(key, default)
        elif hasattr(current, key):
            current = getattr(current, key)
        else:
            return default
        if current is default:
            return default
    return current


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
        _maybe_get_path(kwargs.get("passthrough_logging_payload"), "request_headers", "x-claude-code-session-id"),
        _maybe_get_path(kwargs.get("passthrough_logging_payload"), "request_headers", "X-Claude-Code-Session-Id"),
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


def _maybe_get_path(obj: Any, *keys: str, default: Any = None) -> Any:
    """Safely traverse nested dict/object paths."""
    current = obj
    for key in keys:
        if isinstance(current, dict):
            current = current.get(key, default)
        elif hasattr(current, key):
            current = getattr(current, key)
        else:
            return default
        if current is default:
            return default
    return current


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


def _enrich_provider_cache_metadata(kwargs: Dict[str, Any], result: Any) -> None:
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


def _build_session_history_record_from_spend_log_row(
    spend_log_row: Dict[str, Any],
    *,
    backfill_run_id: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    prepared = _build_backfill_kwargs_from_spend_log_row(spend_log_row)
    if prepared is None:
        return None

    kwargs, result, provenance = prepared
    kwargs, result = _enrich_trace_name_and_provider_metadata(kwargs, result)

    record = _build_session_history_record(
        kwargs=kwargs,
        result=result,
        start_time=_parse_datetime_value(spend_log_row.get("startTime")),
        end_time=_parse_datetime_value(spend_log_row.get("endTime")),
        allow_runtime_identity=False,
    )
    if record is None:
        return None

    metadata = record.get("metadata") or {}
    metadata.update(
        {
            "backfilled": True,
            "backfill_source": "LiteLLM_SpendLogs",
            "backfill_run_id": backfill_run_id,
            "source_request_id": provenance["source_request_id"],
            "source_spend_log_session_field": provenance["source_spend_log_session_field"],
            "session_id_source": provenance["session_id_source"],
            "trace_id_source": provenance["trace_id_source"],
            "source_status": spend_log_row.get("status"),
        }
    )
    if spend_log_row.get("agent_id") is not None:
        metadata["source_agent_id"] = spend_log_row.get("agent_id")
    record["metadata"] = metadata
    record["trace_id"] = kwargs.get("litellm_trace_id") or record.get("trace_id")
    return record


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
            if candidate == _coerce_nested_session_id(observation_metadata.get("user_id")):
                return str(candidate).strip(), "observation.metadata.user_id.session_id"
            return (
                str(candidate).strip(),
                "observation.metadata.user_api_key_end_user_id.session_id",
            )

    return None, "missing"


def _build_usage_object_from_langfuse_observation(observation: Dict[str, Any]) -> Dict[str, Any]:
    metadata = observation.get("metadata")
    usage = observation.get("usage")
    usage_details = observation.get("usageDetails")

    usage_object: Dict[str, Any] = {}
    if isinstance(metadata, dict) and isinstance(metadata.get("usage_object"), dict):
        usage_object.update(metadata["usage_object"])
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
    route_family = metadata.get("passthrough_route_family")
    if isinstance(route_family, str) and route_family.strip():
        route_lower = route_family.lower()
        if "anthropic" in route_lower:
            return "anthropic"
        if "gemini" in route_lower:
            return "gemini"
        if "codex" in route_lower:
            return "openai"
        if "openai" in route_lower:
            return "openai"

    request_route = metadata.get("user_api_key_request_route")
    if isinstance(request_route, str) and request_route.strip():
        route_lower = request_route.lower()
        if route_lower.startswith("/anthropic/"):
            return "anthropic"
        if "gemini" in route_lower:
            return "gemini"
        if route_lower.startswith("/v1/"):
            return "openai"

    api_base = (
        metadata.get("api_base")
        or _maybe_get(metadata.get("hidden_params"), "api_base")
        or observation.get("apiBase")
    )
    if isinstance(api_base, str) and api_base.strip():
        api_base_lower = api_base.lower()
        if "anthropic.com" in api_base_lower:
            return "anthropic"
        if "googleapis.com" in api_base_lower or "generativelanguage" in api_base_lower:
            return "gemini"
        if "openai.com" in api_base_lower:
            return "openai"

    model = observation.get("model")
    if isinstance(model, str) and model.strip():
        model_lower = model.lower()
        if "claude" in model_lower:
            return "anthropic"
        if "gemini" in model_lower:
            return "gemini"
        if (
            model_lower.startswith("gpt")
            or model_lower.startswith("o1")
            or model_lower.startswith("o3")
            or model_lower.startswith("o4")
            or "codex" in model_lower
            or "text-embedding" in model_lower
        ):
            return "openai"

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
        for key, value in billing_header_fields.items():
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


def _build_session_history_record_from_langfuse_trace_observation(
    trace: Dict[str, Any],
    observation: Dict[str, Any],
    *,
    backfill_run_id: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    if observation.get("type") != "GENERATION":
        return None

    metadata = observation.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}

    session_id, session_id_source = _extract_langfuse_session_id(trace, metadata)
    if not session_id:
        return None

    trace_id = trace.get("id") or observation.get("traceId")
    if trace_id is not None and str(trace_id).strip():
        trace_id = str(trace_id).strip()
    else:
        trace_id = None

    usage_object = _build_usage_object_from_langfuse_observation(observation)
    prompt_tokens = _extract_prompt_tokens(usage_object)
    completion_tokens = _extract_completion_tokens(usage_object)
    total_tokens = _extract_total_tokens(usage_object, prompt_tokens, completion_tokens)
    cache_read_input_tokens = _extract_cache_read_input_tokens(usage_object)
    cache_creation_input_tokens = _extract_cache_creation_input_tokens(usage_object)
    reported_reasoning_tokens = _extract_reported_reasoning_tokens(usage_object)
    provider_reported_reasoning_tokens = reported_reasoning_tokens
    provider = _infer_provider_from_langfuse_observation(observation, metadata)

    output_payload = observation.get("output")
    permission_decision = _extract_claude_permission_check_decision_from_value(
        output_payload
    )
    if permission_decision is not None:
        permission_blocked = permission_decision == "yes"
        metadata["claude_internal_check"] = True
        metadata["claude_internal_check_type"] = "permission_check"
        metadata["claude_permission_check"] = True
        metadata["claude_permission_check_decision"] = permission_decision
        metadata["claude_permission_check_blocked"] = permission_blocked
        observation_model = str(observation.get("model") or "").strip()
        if observation_model:
            metadata["claude_permission_check_response_model"] = observation_model
        _merge_tags(
            metadata,
            [
                "claude-internal-check",
                "claude-permission-check",
                f"claude-permission-check:{permission_decision}",
                "claude-permission-check:block"
                if permission_blocked
                else "claude-permission-check:allow",
            ],
        )

    message = _extract_first_langfuse_response_message(output_payload)
    if reported_reasoning_tokens is None and provider == "gemini":
        reported_reasoning_tokens = _fallback_gemini_reasoning_tokens_from_signatures(
            metadata,
            message,
        )
    thinking_blocks = _extract_thinking_blocks(message) if message is not None else []
    reasoning_text = (
        _extract_reasoning_content(message, thinking_blocks)
        if message is not None
        else ""
    )

    reasoning_present = bool(
        (isinstance(reasoning_text, str) and reasoning_text.strip())
        or thinking_blocks
        or metadata.get("reasoning_content_present")
        or (reported_reasoning_tokens and reported_reasoning_tokens > 0)
    )
    estimated_reasoning_tokens = None
    if reported_reasoning_tokens is None and reasoning_present:
        estimated_reasoning_tokens = _estimate_reasoning_tokens(
            model=str(observation.get("model") or ""),
            reasoning_text=reasoning_text,
        )
    reasoning_tokens_source = _determine_reasoning_tokens_source(
        provider_reported_reasoning_tokens=provider_reported_reasoning_tokens,
        reported_reasoning_tokens=reported_reasoning_tokens,
        estimated_reasoning_tokens=estimated_reasoning_tokens,
        reasoning_present=reasoning_present,
    )

    tool_call_count, tool_names = _extract_tool_call_info(message)
    tool_activity = _extract_tool_activity_from_message(message) if message is not None else []
    if tool_call_count == 0:
        output_tool_call_count, output_tool_names = _extract_response_output_tool_call_info(
            output_payload
        )
        if output_tool_call_count > 0:
            tool_call_count, tool_names = output_tool_call_count, output_tool_names
    if not tool_activity:
        tool_activity = _extract_response_output_tool_activity(output_payload)
    if tool_call_count == 0:
        fallback_tool_names = observation.get("toolCallNames") or observation.get(
            "tool_call_names"
        )
        if isinstance(fallback_tool_names, list):
            normalized_tool_names = [
                str(tool_name)
                for tool_name in fallback_tool_names
                if isinstance(tool_name, str) and tool_name.strip()
            ]
            if normalized_tool_names:
                tool_call_count = len(normalized_tool_names)
                tool_names = normalized_tool_names
    if tool_call_count == 0:
        metadata_tool_names = metadata.get("usage_tool_names")
        if isinstance(metadata_tool_names, list):
            normalized_tool_names = [
                str(tool_name)
                for tool_name in metadata_tool_names
                if isinstance(tool_name, str) and tool_name.strip()
            ]
            if normalized_tool_names:
                tool_call_count = len(normalized_tool_names)
                tool_names = normalized_tool_names
    tool_activity_summary = _summarize_tool_activity(tool_activity)
    agent_name, tenant_id = _extract_agent_context_from_langfuse_trace_observation(
        trace,
        observation,
    )
    explicit_tenant_id, tenant_source = _extract_tenant_identity_from_langfuse_trace_observation(
        trace,
        observation,
        metadata,
    )
    if explicit_tenant_id:
        tenant_id = explicit_tenant_id
    if tenant_id and tenant_source:
        metadata["tenant_id_source"] = tenant_source
    elif tenant_id:
        metadata["tenant_id_source"] = "agent_context_text"
    repository = _extract_repository_identity_from_langfuse_trace_observation(
        trace,
        observation,
        metadata,
    )
    if repository:
        metadata["repository"] = repository
    request_tags = _derive_request_tags_from_langfuse_metadata(metadata)
    provider_cache_state = _resolve_provider_cache_state(
        provider=provider,
        model=str(observation.get("model") or ""),
        usage_obj=usage_object,
        metadata=metadata,
        request_body=None,
    )
    provider_cache_state = dict(provider_cache_state or {})
    if provider_cache_state:
        provider_cache_state.update(
            _compute_provider_cache_miss_cost_state(
                provider_family=provider,
                model=str(observation.get("model") or ""),
                usage_obj=usage_object,
                cache_state=provider_cache_state,
                metadata=metadata,
            )
        )

    response_cost_usd = _safe_float(
        _first_non_none(
            _maybe_get(observation.get("costDetails"), "total"),
            observation.get("calculatedTotalCost"),
            metadata.get("litellm_response_cost"),
            trace.get("totalCost"),
        )
    )
    permission_usage_fields = _build_permission_usage_fields(
        metadata=metadata,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        response_cost_usd=response_cost_usd,
    )

    history_metadata = _build_session_history_metadata(
        metadata=metadata,
        request_tags=request_tags,
        tenant_id=tenant_id,
    )
    history_metadata.update(
        {
            "backfilled": True,
            "backfill_source": "LangfuseTraces",
            "backfill_run_id": backfill_run_id,
            "source_trace_id": trace_id,
            "source_observation_id": observation.get("id"),
            "session_id_source": session_id_source,
            "trace_id_source": "trace.id" if trace_id else "missing",
            "source_trace_environment": trace.get("environment"),
            "source_status": "failure"
            if observation.get("statusMessage")
            else "success",
        }
    )
    runtime_identity = _build_session_runtime_identity(
        metadata=history_metadata,
        trace_environment=trace.get("environment"),
        allow_runtime=False,
    )

    return _normalize_session_history_record({
        "litellm_call_id": observation.get("id"),
        "session_id": session_id,
        "trace_id": trace_id,
        "provider_response_id": _maybe_get(output_payload, "id"),
        "provider": provider,
        "model": str(observation.get("model") or ""),
        "model_group": metadata.get("model_group"),
        "agent_name": agent_name,
        "tenant_id": tenant_id,
        "repository": repository,
        "call_type": metadata.get("user_api_key_request_route") or observation.get("name"),
        "start_time": _parse_datetime_value(observation.get("startTime")),
        "end_time": _parse_datetime_value(observation.get("endTime")),
        "input_tokens": prompt_tokens,
        "output_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "cache_read_input_tokens": cache_read_input_tokens,
        "cache_creation_input_tokens": cache_creation_input_tokens,
        "reasoning_tokens_reported": reported_reasoning_tokens,
        "reasoning_tokens_estimated": estimated_reasoning_tokens,
        "reasoning_tokens_source": reasoning_tokens_source,
        "reasoning_present": reasoning_present,
        "thinking_signature_present": bool(metadata.get("thinking_signature_present")),
        "provider_cache_attempted": bool(
            provider_cache_state and provider_cache_state.get("attempted")
        ),
        "provider_cache_status": (
            provider_cache_state.get("status") if provider_cache_state else None
        ),
        "provider_cache_miss": bool(
            provider_cache_state and provider_cache_state.get("miss")
        ),
        "provider_cache_miss_reason": (
            provider_cache_state.get("miss_reason") if provider_cache_state else None
        ),
        "provider_cache_miss_token_count": (
            provider_cache_state.get("miss_token_count") if provider_cache_state else None
        ),
        "provider_cache_miss_cost_usd": (
            provider_cache_state.get("miss_cost_usd") if provider_cache_state else None
        ),
        "tool_call_count": tool_call_count,
        "tool_names": tool_names,
        "file_read_count": tool_activity_summary["file_read_count"],
        "file_modified_count": tool_activity_summary["file_modified_count"],
        "git_commit_count": tool_activity_summary["git_commit_count"],
        "git_push_count": tool_activity_summary["git_push_count"],
        "tool_activity": tool_activity,
        "response_cost_usd": response_cost_usd,
        **permission_usage_fields,
        "litellm_environment": runtime_identity["litellm_environment"],
        "litellm_version": runtime_identity["litellm_version"],
        "litellm_fork_version": runtime_identity["litellm_fork_version"],
        "litellm_wheel_versions": runtime_identity["litellm_wheel_versions"],
        "client_name": runtime_identity["client_name"],
        "client_version": runtime_identity["client_version"],
        "client_user_agent": runtime_identity["client_user_agent"],
        "metadata": history_metadata,
    })


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


def _build_session_history_metadata(
    *,
    metadata: Dict[str, Any],
    request_tags: List[str],
    tenant_id: Optional[str],
) -> Dict[str, Any]:
    history_metadata: Dict[str, Any] = {"request_tags": request_tags}
    if tenant_id:
        history_metadata["tenant_id"] = tenant_id

    for key in _AAWM_SESSION_HISTORY_METADATA_KEYS:
        value = metadata.get(key)
        if value is not None:
            history_metadata[key] = value

    return history_metadata


def _resolve_session_history_model(
    kwargs: Dict[str, Any],
    standard_logging_object: Dict[str, Any],
    metadata: Dict[str, Any],
    result: Any,
) -> str:
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
        kwargs.get("model"),
        standard_logging_object.get("model"),
        _maybe_get_path(kwargs.get("passthrough_logging_payload"), "request_body", "model"),
        _maybe_get_path(kwargs.get("litellm_params"), "proxy_server_request", "body", "model"),
        metadata.get("anthropic_adapter_model"),
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



def _build_session_history_record(
    kwargs: Dict[str, Any],
    result: Any,
    start_time: Any,
    end_time: Any,
    allow_runtime_identity: bool = True,
) -> Optional[Dict[str, Any]]:
    session_id = _extract_session_id(kwargs)
    if not session_id:
        return None

    litellm_params = kwargs.get("litellm_params") or {}
    metadata = litellm_params.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
        litellm_params["metadata"] = metadata
        kwargs["litellm_params"] = litellm_params
    standard_logging_object = kwargs.get("standard_logging_object") or {}
    _enrich_claude_permission_check_metadata(
        kwargs,
        metadata,
        result,
        standard_logging_object=standard_logging_object,
    )
    _sync_standard_logging_object(kwargs, metadata)
    standard_logging_object = kwargs.get("standard_logging_object") or standard_logging_object
    request_tags = (
        standard_logging_object.get("request_tags") or metadata.get("tags") or []
    )
    if not isinstance(request_tags, list):
        request_tags = []

    resolved_model = _resolve_session_history_model(
        kwargs=kwargs,
        standard_logging_object=standard_logging_object,
        metadata=metadata,
        result=result,
    )

    usage_obj = _extract_usage_object(kwargs, result)
    hidden_params = getattr(result, "_hidden_params", {}) or {}
    usage_obj = _merge_estimated_rerank_tokens_into_usage(
        kwargs=kwargs,
        result=result,
        usage_obj=usage_obj,
        model=resolved_model,
    )
    usage_dict = _coerce_usage_object_to_dict(usage_obj) or {}
    search_units = _safe_int(usage_dict.get("search_units"))
    if search_units:
        metadata["usage_search_units"] = search_units
    usage_cost = _safe_float(usage_dict.get("cost"))
    if usage_cost is not None:
        metadata["usage_openrouter_cost"] = usage_cost
    openrouter_usage = hidden_params.get("openrouter_usage")
    if isinstance(openrouter_usage, dict):
        search_units = _safe_int(openrouter_usage.get("search_units"))
        if search_units:
            metadata["usage_search_units"] = search_units
        openrouter_cost = _safe_float(openrouter_usage.get("cost"))
        if openrouter_cost is not None:
            metadata["usage_openrouter_cost"] = openrouter_cost
    if hidden_params.get("openrouter_provider") is not None:
        metadata["openrouter_provider"] = hidden_params.get("openrouter_provider")
    if hidden_params.get("openrouter_response_model") is not None:
        metadata["openrouter_response_model"] = hidden_params.get(
            "openrouter_response_model"
        )

    prompt_tokens = _extract_prompt_tokens(usage_obj)
    completion_tokens = _extract_completion_tokens(usage_obj)
    total_tokens = _extract_total_tokens(usage_obj, prompt_tokens, completion_tokens)
    cache_read_input_tokens = _extract_cache_read_input_tokens(usage_obj)
    cache_creation_input_tokens = _extract_cache_creation_input_tokens(usage_obj)
    reported_reasoning_tokens = _extract_reported_reasoning_tokens(usage_obj)
    provider_reported_reasoning_tokens = reported_reasoning_tokens
    provider_prefix = _infer_usage_breakout_provider_prefix(kwargs, metadata)
    resolved_provider = _normalize_session_history_provider(
        kwargs.get("custom_llm_provider"),
        resolved_model,
        metadata,
    )

    message = _extract_first_response_message(result)
    if reported_reasoning_tokens is None and provider_prefix == "gemini":
        reported_reasoning_tokens = _fallback_gemini_reasoning_tokens_from_signatures(
            metadata,
            message,
        )
    thinking_blocks = _extract_thinking_blocks(message) if message is not None else []
    reasoning_text = (
        _extract_reasoning_content(message, thinking_blocks)
        if message is not None
        else ""
    )

    reasoning_present = bool(
        (isinstance(reasoning_text, str) and reasoning_text.strip())
        or thinking_blocks
        or metadata.get("reasoning_content_present")
        or (reported_reasoning_tokens and reported_reasoning_tokens > 0)
    )
    estimated_reasoning_tokens = None
    if reported_reasoning_tokens is None and reasoning_present:
        estimated_reasoning_tokens = _estimate_reasoning_tokens(
            model=resolved_model,
            reasoning_text=reasoning_text,
        )
    reasoning_tokens_source = _determine_reasoning_tokens_source(
        provider_reported_reasoning_tokens=provider_reported_reasoning_tokens,
        reported_reasoning_tokens=reported_reasoning_tokens,
        estimated_reasoning_tokens=estimated_reasoning_tokens,
        reasoning_present=reasoning_present,
    )

    tool_call_count, tool_names = _extract_tool_call_info(message)
    tool_activity = _extract_tool_activity_from_message(message) if message is not None else []
    if tool_call_count == 0:
        output_tool_call_count, output_tool_names = _extract_response_output_tool_call_info(
            result,
            standard_logging_object,
        )
        if output_tool_call_count > 0:
            tool_call_count, tool_names = output_tool_call_count, output_tool_names
    if not tool_activity:
        tool_activity = _extract_response_output_tool_activity(
            result,
            standard_logging_object,
        )
    tool_activity_summary = _summarize_tool_activity(tool_activity)
    agent_name, tenant_id = _extract_agent_context(kwargs)
    explicit_tenant_id, tenant_source = _extract_tenant_identity_from_kwargs(
        kwargs,
        metadata=metadata,
        standard_logging_object=standard_logging_object,
    )
    if explicit_tenant_id:
        tenant_id = explicit_tenant_id
    if tenant_id and tenant_source:
        metadata["tenant_id_source"] = tenant_source
    elif tenant_id:
        metadata["tenant_id_source"] = "agent_context_text"
    repository = _extract_repository_identity_from_kwargs(
        kwargs,
        metadata=metadata,
        standard_logging_object=standard_logging_object,
    )
    if repository:
        metadata["repository"] = repository

    response_cost_usd = _safe_float(
        _first_non_none(
            kwargs.get("response_cost"),
            standard_logging_object.get("response_cost"),
            hidden_params.get("response_cost"),
            _maybe_get_path(
                hidden_params,
                "additional_headers",
                "llm_provider-x-litellm-response-cost",
            ),
            metadata.get("litellm_response_cost"),
            metadata.get("response_cost"),
            metadata.get("usage_openrouter_cost"),
            usage_dict.get("cost"),
        )
    )
    if (
        (response_cost_usd is None or response_cost_usd == 0)
        and prompt_tokens > 0
        and resolved_model != "unknown"
    ):
        try:
            import litellm
            from litellm.responses.utils import ResponseAPILoggingUtils

            usage_for_cost = None
            if isinstance(usage_obj, dict) and {
                "input_tokens",
                "output_tokens",
            }.issubset(usage_obj.keys()):
                usage_for_cost = (
                    ResponseAPILoggingUtils._transform_response_api_usage_to_chat_usage(
                        dict(usage_obj)
                    )
                )
            prompt_cost, completion_cost = litellm.cost_per_token(
                model=resolved_model,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                custom_llm_provider=kwargs.get("custom_llm_provider"),
                cache_creation_input_tokens=cache_creation_input_tokens,
                cache_read_input_tokens=cache_read_input_tokens,
                usage_object=usage_for_cost,
                call_type="responses",
            )
            response_cost_usd = prompt_cost + completion_cost
        except Exception as exc:
            response_cost_usd = _calculate_response_cost_from_bundled_model_cost_map(
                model=resolved_model,
                custom_llm_provider=kwargs.get("custom_llm_provider"),
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                usage_obj=usage_obj,
            )
            verbose_logger.debug(
                "AawmAgentIdentity: failed to backfill response cost for model=%s: %s",
                resolved_model,
                exc,
            )

        if response_cost_usd == 0:
            bundled_response_cost = _calculate_response_cost_from_bundled_model_cost_map(
                model=resolved_model,
                custom_llm_provider=kwargs.get("custom_llm_provider"),
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                usage_obj=usage_obj,
            )
            if bundled_response_cost is not None:
                response_cost_usd = bundled_response_cost

    permission_usage_fields = _build_permission_usage_fields(
        metadata=metadata,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        response_cost_usd=response_cost_usd,
    )

    provider_cache_state = _resolve_provider_cache_state(
        provider=resolved_provider,
        model=resolved_model,
        usage_obj=usage_obj,
        metadata=metadata,
        request_body=_extract_provider_cache_request_body(kwargs),
    )
    provider_cache_state = dict(provider_cache_state or {})
    if provider_cache_state:
        provider_cache_state.update(
            _compute_provider_cache_miss_cost_state(
                provider_family=_normalize_provider_cache_family(
                    resolved_provider,
                    resolved_model,
                    metadata,
                ),
                model=resolved_model,
                usage_obj=usage_obj,
                cache_state=provider_cache_state,
                metadata=metadata,
            )
        )

    runtime_identity = _build_session_runtime_identity(
        metadata=metadata,
        kwargs=kwargs,
        allow_runtime=allow_runtime_identity,
    )

    return _normalize_session_history_record({
        "litellm_call_id": kwargs.get("litellm_call_id"),
        "session_id": session_id,
        "trace_id": _extract_trace_id(kwargs),
        "provider_response_id": _maybe_get(result, "id"),
        "provider": resolved_provider,
        "model": resolved_model,
        "model_group": metadata.get("model_group") or standard_logging_object.get("model_group"),
        "agent_name": agent_name,
        "tenant_id": tenant_id,
        "repository": repository,
        "call_type": kwargs.get("call_type") or standard_logging_object.get("call_type"),
        "start_time": _normalize_datetime(start_time),
        "end_time": _normalize_datetime(end_time),
        "input_tokens": prompt_tokens,
        "output_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "cache_read_input_tokens": cache_read_input_tokens,
        "cache_creation_input_tokens": cache_creation_input_tokens,
        "reasoning_tokens_reported": reported_reasoning_tokens,
        "reasoning_tokens_estimated": estimated_reasoning_tokens,
        "reasoning_tokens_source": reasoning_tokens_source,
        "reasoning_present": reasoning_present,
        "thinking_signature_present": bool(metadata.get("thinking_signature_present")),
        "provider_cache_attempted": bool(
            provider_cache_state and provider_cache_state.get("attempted")
        ),
        "provider_cache_status": (
            provider_cache_state.get("status") if provider_cache_state else None
        ),
        "provider_cache_miss": bool(
            provider_cache_state and provider_cache_state.get("miss")
        ),
        "provider_cache_miss_reason": (
            provider_cache_state.get("miss_reason") if provider_cache_state else None
        ),
        "provider_cache_miss_token_count": (
            provider_cache_state.get("miss_token_count") if provider_cache_state else None
        ),
        "provider_cache_miss_cost_usd": (
            provider_cache_state.get("miss_cost_usd") if provider_cache_state else None
        ),
        "tool_call_count": tool_call_count,
        "tool_names": tool_names,
        "file_read_count": tool_activity_summary["file_read_count"],
        "file_modified_count": tool_activity_summary["file_modified_count"],
        "git_commit_count": tool_activity_summary["git_commit_count"],
        "git_push_count": tool_activity_summary["git_push_count"],
        "tool_activity": tool_activity,
        "response_cost_usd": response_cost_usd,
        **permission_usage_fields,
        "litellm_environment": runtime_identity["litellm_environment"],
        "litellm_version": runtime_identity["litellm_version"],
        "litellm_fork_version": runtime_identity["litellm_fork_version"],
        "litellm_wheel_versions": runtime_identity["litellm_wheel_versions"],
        "client_name": runtime_identity["client_name"],
        "client_version": runtime_identity["client_version"],
        "client_user_agent": runtime_identity["client_user_agent"],
        "metadata": _build_session_history_metadata(
            metadata=metadata,
            request_tags=[tag for tag in request_tags if isinstance(tag, str)],
            tenant_id=tenant_id,
        ),
    })


async def _open_aawm_session_history_connection() -> Any:
    dsn = _build_aawm_dsn()
    if not dsn:
        raise RuntimeError("AAWM session history database configuration is missing")

    try:
        asyncpg = importlib.import_module("asyncpg")
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "AAWM session history requires asyncpg to be installed"
        ) from exc

    return await asyncpg.connect(dsn=dsn, command_timeout=10)


async def _get_aawm_session_history_pool() -> Any:
    dsn = _build_aawm_dsn()
    if not dsn:
        raise RuntimeError("AAWM session history database configuration is missing")

    loop = asyncio.get_running_loop()
    pool_key = (loop, dsn)
    with _aawm_session_history_pool_lock:
        pool = _aawm_session_history_pools.get(pool_key)
    if pool is not None:
        return pool

    try:
        asyncpg = importlib.import_module("asyncpg")
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "AAWM session history requires asyncpg to be installed"
        ) from exc

    created_pool = await asyncpg.create_pool(
        dsn=dsn,
        min_size=0,
        max_size=_get_session_history_pool_max_size(),
        command_timeout=10,
    )
    with _aawm_session_history_pool_lock:
        existing_pool = _aawm_session_history_pools.get(pool_key)
        if existing_pool is None:
            _aawm_session_history_pools[pool_key] = created_pool
            return created_pool

    await created_pool.close()
    return existing_pool


async def _ensure_session_history_schema(conn: Any) -> None:
    global _aawm_session_history_schema_ready

    if _aawm_session_history_schema_ready:
        return

    with _aawm_session_history_schema_lock:
        if _aawm_session_history_schema_ready:
            return

        await conn.execute(_AAWM_SESSION_HISTORY_TABLE_SQL)
        for statement in _AAWM_SESSION_HISTORY_ALTER_STATEMENTS:
            await conn.execute(statement)
        for statement in _AAWM_SESSION_HISTORY_INDEX_STATEMENTS:
            await conn.execute(statement)
        await conn.execute(_AAWM_SESSION_HISTORY_TOOL_ACTIVITY_TABLE_SQL)
        for statement in _AAWM_SESSION_HISTORY_TOOL_ACTIVITY_INDEX_STATEMENTS:
            await conn.execute(statement)

        _aawm_session_history_schema_ready = True


def _build_session_history_db_payload(record: Dict[str, Any]) -> Tuple[Any, ...]:
    record = _normalize_session_history_record(dict(record))
    return (
        record["litellm_call_id"],
        record["session_id"],
        record["trace_id"],
        record["provider_response_id"],
        record["provider"],
        record["model"],
        record["model_group"],
        record["agent_name"],
        record["tenant_id"],
        record["call_type"],
        record["start_time"],
        record["end_time"],
        record["input_tokens"],
        record["output_tokens"],
        record["total_tokens"],
        record["cache_read_input_tokens"],
        record["cache_creation_input_tokens"],
        record["reasoning_tokens_reported"],
        record["reasoning_tokens_estimated"],
        record["reasoning_tokens_source"],
        record["reasoning_present"],
        record["thinking_signature_present"],
        record.get("provider_cache_attempted", False),
        record.get("provider_cache_status"),
        record.get("provider_cache_miss", False),
        record.get("provider_cache_miss_reason"),
        record.get("provider_cache_miss_token_count"),
        record.get("provider_cache_miss_cost_usd"),
        record["tool_call_count"],
        json.dumps(record["tool_names"]),
        record.get("file_read_count", 0),
        record.get("file_modified_count", 0),
        record.get("git_commit_count", 0),
        record.get("git_push_count", 0),
        record["response_cost_usd"],
        record.get("litellm_environment"),
        record.get("litellm_version"),
        record.get("litellm_fork_version"),
        json.dumps(record.get("litellm_wheel_versions") or {}),
        record.get("client_name"),
        record.get("client_version"),
        record.get("client_user_agent"),
        record.get("token_permission_input", 0),
        record.get("token_permission_output", 0),
        record.get("permission_usd_cost", 0),
        json.dumps(record["metadata"]),
        record.get("repository"),
    )


def _build_tool_activity_db_payloads(record: Dict[str, Any]) -> List[Tuple[Any, ...]]:
    tool_activity = record.get("tool_activity") or []
    if not isinstance(tool_activity, list):
        return []

    payloads: List[Tuple[Any, ...]] = []
    for index, item in enumerate(tool_activity):
        if not isinstance(item, dict):
            continue
        payloads.append(
            (
                record["litellm_call_id"],
                record["session_id"],
                record.get("trace_id"),
                record.get("provider"),
                record["model"],
                record.get("agent_name"),
                _safe_int(item.get("tool_index")) if _safe_int(item.get("tool_index")) is not None else index,
                item.get("tool_call_id"),
                item.get("tool_name"),
                item.get("tool_kind"),
                json.dumps(item.get("file_paths_read") or []),
                json.dumps(item.get("file_paths_modified") or []),
                _safe_int(item.get("git_commit_count")) or 0,
                _safe_int(item.get("git_push_count")) or 0,
                item.get("command_text"),
                json.dumps(item.get("arguments") or {}),
                json.dumps(item.get("metadata") or {}),
            )
        )
    return payloads


async def _persist_session_history_record(record: Dict[str, Any]) -> None:
    pool = await _get_aawm_session_history_pool()
    async with pool.acquire() as conn:
        await _ensure_session_history_schema(conn)

        history_payload = _build_session_history_db_payload(record)
        tool_activity_payloads = _build_tool_activity_db_payloads(record)

        await conn.execute(_AAWM_SESSION_HISTORY_INSERT_SQL, *history_payload)
        if tool_activity_payloads:
            await conn.executemany(
                _AAWM_SESSION_HISTORY_TOOL_ACTIVITY_INSERT_SQL, tool_activity_payloads
            )


async def _persist_session_history_records(records: List[Dict[str, Any]]) -> None:
    if not records:
        return

    pool = await _get_aawm_session_history_pool()
    async with pool.acquire() as conn:
        await _ensure_session_history_schema(conn)

        payloads = [_build_session_history_db_payload(record) for record in records]
        tool_activity_payloads: List[Tuple[Any, ...]] = []
        for record in records:
            tool_activity_payloads.extend(_build_tool_activity_db_payloads(record))

        await conn.executemany(_AAWM_SESSION_HISTORY_INSERT_SQL, payloads)
        if tool_activity_payloads:
            await conn.executemany(
                _AAWM_SESSION_HISTORY_TOOL_ACTIVITY_INSERT_SQL, tool_activity_payloads
            )


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


def _enrich_gemini_thought_signature_metadata(
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
            for key, value in indexed_fields.items():
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


def _enrich_trace_name_and_provider_metadata(
    kwargs: Dict[str, Any], result: Any
) -> Tuple[dict, Any]:
    agent_name = _extract_agent_name(kwargs)
    headers = _ensure_mutable_headers(kwargs)
    metadata = _ensure_mutable_metadata(kwargs)
    session_id = _extract_session_id(kwargs)

    current_trace_name = metadata.get("trace_name")
    if current_trace_name == "claude-code":
        metadata["trace_name"] = f"claude-code.{agent_name}"
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

    _enrich_session_runtime_identity_metadata(kwargs)

    message = _extract_first_response_message(result)
    if message is not None:
        _enrich_claude_thinking_metadata(metadata, message)
        _enrich_gemini_thought_signature_metadata(metadata, message)
    _enrich_claude_permission_check_metadata(kwargs, metadata, result)
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


class AawmAgentIdentity(CustomLogger):
    """CustomLogger that enriches Langfuse trace_name with agent identity.

    Implements both sync logging_hook() and async async_logging_hook() to
    cover all code paths:
    - Sync: pass-through endpoints run Langfuse in sync success_handler (thread pool)
    - Async: standard LLM calls run Langfuse in async_success_handler
    """

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
        try:
            record = _build_session_history_record(
                kwargs=kwargs,
                result=response_obj,
                start_time=start_time,
                end_time=end_time,
            )
            if record is None:
                return

            _enqueue_session_history_record(record)
        except Exception as exc:
            verbose_logger.warning(
                "AawmAgentIdentity.log_success_event failed: %s", exc
            )

    async def async_log_success_event(
        self, kwargs, response_obj, start_time, end_time
    ) -> None:
        try:
            record = _build_session_history_record(
                kwargs=kwargs,
                result=response_obj,
                start_time=start_time,
                end_time=end_time,
            )
            if record is None:
                return

            _enqueue_session_history_record(record)
        except Exception as exc:
            verbose_logger.warning(
                "AawmAgentIdentity.async_log_success_event failed: %s", exc
            )


# Module-level instance for config registration via get_instance_fn().
# Config must reference this instance name, not the class name:
#   callbacks: ["litellm.integrations.aawm_agent_identity.aawm_agent_identity_instance"]
aawm_agent_identity_instance = AawmAgentIdentity()
