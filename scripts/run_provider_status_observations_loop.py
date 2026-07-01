#!/usr/bin/env python3
"""Run provider status observations on a fixed container-friendly cadence."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import re
import signal
import sys
import time
import traceback
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence
from urllib import error as urllib_error
from urllib import request as urllib_request
from urllib.parse import parse_qsl, urlsplit

try:
    from scripts import record_provider_status_observations as probes
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    probes = importlib.import_module("record_provider_status_observations")

try:
    from scripts import grok_oidc_refresh
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    grok_oidc_refresh = importlib.import_module("grok_oidc_refresh")

try:
    from scripts import codex_oauth_refresh
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    codex_oauth_refresh = importlib.import_module("codex_oauth_refresh")

try:
    from scripts import xai_oauth_refresh
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    xai_oauth_refresh = importlib.import_module("xai_oauth_refresh")


TRUE_VALUES = {"1", "true", "yes", "on"}
FALSE_VALUES = {"0", "false", "no", "off"}
PROVIDER_FAILURE_SUMMARY_LIMIT = 8
PROVIDER_FAILURE_FIELD_LIMIT = 160
PROVIDER_FAILURE_MESSAGE_LIMIT = 240
DEFAULT_GROK_OIDC_AUTH_FILE = "/home/zepfu/.grok/auth.json"
DEFAULT_GROK_OIDC_LOCK_FILE = "/home/zepfu/.grok/auth.json.lock"
GROK_SIDECAR_NATIVE_AUTH_FILE_ENV_VARS = (
    "LITELLM_XAI_GROK_AUTH_FILE",
    "LITELLM_XAI_OAUTH_GROK_AUTH_FILE",
    "GROK_AUTH_FILE",
)
DEFAULT_GROK_OIDC_REFRESH_INTERVAL_SECONDS = 3600.0
DEFAULT_GROK_OIDC_HTTP_TIMEOUT_SECONDS = 30.0
DEFAULT_CODEX_AUTH_FILE = codex_oauth_refresh.DEFAULT_CODEX_AUTH_FILE
DEFAULT_CODEX_LOCK_FILE = codex_oauth_refresh.DEFAULT_CODEX_LOCK_FILE
CODEX_SIDECAR_AUTH_FILE_ENV_VARS = (
    "LITELLM_CODEX_AUTH_FILE",
    "CHATGPT_AUTH_FILE",
)
CODEX_SIDECAR_TOKEN_DIR_ENV_VARS = (
    "LITELLM_CODEX_TOKEN_DIR",
    "CHATGPT_TOKEN_DIR",
)
CODEX_SIDECAR_DEFAULT_AUTH_PATHS = (
    "/home/zepfu/.codex/auth.json",
    "~/.codex/auth.json",
    "~/.config/litellm/chatgpt/auth.json",
)
DEFAULT_CODEX_OAUTH_REFRESH_INTERVAL_SECONDS = 3600.0
DEFAULT_CODEX_OAUTH_HTTP_TIMEOUT_SECONDS = 30.0
DEFAULT_XAI_OAUTH_AUTH_FILE = xai_oauth_refresh.DEFAULT_XAI_OAUTH_AUTH_FILE
DEFAULT_XAI_OAUTH_LOCK_FILE = xai_oauth_refresh.DEFAULT_XAI_OAUTH_LOCK_FILE
XAI_OAUTH_SIDECAR_AUTH_FILE_ENV_VARS = (
    "LITELLM_XAI_OAUTH_AUTH_FILE",
    "LITELLM_XAI_OAUTH_MIGRATED_AUTH_FILE",
)
DEFAULT_XAI_OAUTH_REFRESH_INTERVAL_SECONDS = 3600.0
DEFAULT_XAI_OAUTH_HTTP_TIMEOUT_SECONDS = 30.0
DEFAULT_GROK_BILLING_POLL_ENABLED = False
DEFAULT_GROK_BILLING_POLL_INTERVAL_SECONDS = 3600.0
DEFAULT_GROK_BILLING_POLL_HTTP_TIMEOUT_SECONDS = 30.0
DEFAULT_GROK_BILLING_URL = "https://cli-chat-proxy.grok.com/v1/billing?format=credits"
DEFAULT_GROK_BILLING_CLIENT_VERSION = "0.2.55"
DEFAULT_GROK_BILLING_CLIENT_IDENTIFIER = "grok-cli"
DEFAULT_GROK_BILLING_XAI_TOKEN_AUTH = "xai-grok-cli"
DEFAULT_GROK_BILLING_MODEL = "grok-build"
DEFAULT_GROK_BILLING_HTTP_METHOD = "GET"
DEFAULT_GROK_BILLING_POLL_MAX_ATTEMPTS = 3
DEFAULT_GROK_BILLING_POLL_RETRY_BACKOFF_SECONDS = 0.5
DEFAULT_CODEX_RESET_CREDIT_POLL_ENABLED = False
DEFAULT_CODEX_RESET_CREDIT_POLL_INTERVAL_SECONDS = 3600.0
DEFAULT_CODEX_RESET_CREDIT_POLL_HTTP_TIMEOUT_SECONDS = 30.0
DEFAULT_CODEX_USAGE_URL = "https://chatgpt.com/backend-api/wham/rate-limit-reset-credits"
DEFAULT_CODEX_RESET_CREDIT_POLL_MAX_ATTEMPTS = 3
DEFAULT_CODEX_RESET_CREDIT_POLL_RETRY_BACKOFF_SECONDS = 0.5
DEFAULT_CODEX_RESET_CREDIT_CREDIT_FAMILY = "codex_rate_limit_reset"
DEFAULT_CODEX_RESET_CREDIT_CREDIT_TYPE = "reset_credit"
DEFAULT_CODEX_RESET_CREDIT_SOURCE = "codex_reset_credit_poll"
DEFAULT_CODEX_RESET_CREDIT_LATEST_VISIBLE_SOURCE_URL = (
    "https://x.com/thsottiaux/status/2070653282440405046"
)
CODEX_RESET_CREDIT_SANITIZED_RAW_FIELD_KEYS = frozenset(
    {
        "id",
        "credit_id",
        "status",
        "reset_type",
        "resetType",
        "granted_at",
        "grantedAt",
        "expires_at",
        "expiresAt",
        "redeem_started_at",
        "redeemStartedAt",
        "redeemed_at",
        "redeemedAt",
    }
)

DEFAULT_OBSERVABILITY_ANOMALY_SCAN_ENABLED = False
DEFAULT_OBSERVABILITY_ANOMALY_SCAN_INTERVAL_SECONDS = 3600.0
DEFAULT_OBSERVABILITY_ANOMALY_SCAN_LOOKBACK_HOURS = 4.0
DEFAULT_OBSERVABILITY_ANOMALY_SCAN_ERROR_LOG_DIR = "/app/.analysis"
OBSERVABILITY_ANOMALY_SAMPLE_LIMIT = 5
GROK_BILLING_IDENTITY_HEADER_FIELDS = (
    ("x-userid", "user_id"),
    ("x-grok-user-id", "user_id"),
    ("x-teamid", "team_id"),
    ("x-email", "email"),
)
GROK_BILLING_RETRYABLE_HTTP_STATUS_CODES = {408, 425, 500, 502, 503, 504}
GROK_BILLING_NON_RETRYABLE_HTTP_STATUS_CODES = {
    400,
    401,
    403,
    404,
    405,
    409,
    422,
    429,
}
GROK_BILLING_RETRYABLE_ERROR_HINTS = (
    "operation was cancelled",
    "timeout expired",
)
GROK_BILLING_POLL_SLEEP_FN: Callable[[float], None] = time.sleep
CODEX_RESET_CREDIT_POLL_SLEEP_FN: Callable[[float], None] = time.sleep
OBSERVABILITY_ANOMALY_ALIAS_METADATA_KEYS = (
    "requested_model_alias",
    "model_alias_label",
    "codex_auto_agent_alias",
    "anthropic_auto_agent_alias",
    "aawm_auto_agent_alias",
)
PROVIDER_FAILURE_SECRET_RE = re.compile(
    "|".join(
        (
            r"Bearer\s+[A-Za-z0-9\-._~+/]{10,}=*",
            r"Basic\s+[A-Za-z0-9+/]{10,}={0,2}",
            r"sk-[A-Za-z0-9\-_]{20,}",
            r"(?:api[_-]?key|x-api-key|api-key|token|password|passwd|secret|"
            r"x[_-]?xai[_-]?token[_-]?auth|x[_-]?userid|x[_-]?grok[_-]?user[_-]?id|"
            r"x[_-]?teamid|x[_-]?email)"
            r"['\"]?\s*[:=]\s*['\"]?[^\s,'\"})\]{}>]+",
            r"(?<=://)[^\s'\"]*:[^\s'\"@]+(?=@)",
            r"\b(?:\d{1,3}\.){3}\d{1,3}\b",
            r"\b(?:[0-9A-Fa-f]{1,4}:){2,}[0-9A-Fa-f:.]{1,}\b",
        )
    ),
    re.IGNORECASE,
)


class GrokBillingPollError(ValueError):
    """Sanitized billing poll failure with retry metadata for sidecar events."""

    def __init__(
        self,
        message: str,
        *,
        status_code: Optional[int],
        attempt_count: int,
        retry_count: int,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.attempt_count = max(1, attempt_count)
        self.retry_count = max(0, retry_count)


class CodexResetCreditPollError(ValueError):
    """Sanitized Codex reset-credit poll failure with retry metadata."""

    def __init__(
        self,
        message: str,
        *,
        status_code: Optional[int],
        attempt_count: int,
        retry_count: int,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.attempt_count = max(1, attempt_count)
        self.retry_count = max(0, retry_count)

GROK_BILLING_RATE_LIMIT_INSERT_SQL = """
WITH candidate AS (
    SELECT
        %s::timestamptz AS observed_at,
        %s::text AS client,
        %s::text AS client_version,
        %s::text AS account_hash,
        %s::text AS provider,
        %s::text AS model,
        %s::text AS quota_key,
        %s::text AS quota_period,
        %s::text AS quota_type,
        %s::timestamptz AS expected_reset_at,
        %s::double precision AS remaining_pct,
        %s::double precision AS quota_limit,
        %s::double precision AS quota_used,
        %s::double precision AS quota_remaining,
        %s::timestamptz AS billing_period_start_at,
        %s::timestamptz AS billing_period_end_at,
        %s::jsonb AS raw_provider_fields,
        %s::jsonb AS evidence,
        %s::text AS source,
        %s::text AS session_id,
        %s::text AS trace_id,
        %s::text AS litellm_call_id
),
locked AS (
    SELECT pg_advisory_xact_lock(
        hashtext(
            CONCAT_WS(
                '|',
                candidate.provider,
                COALESCE(candidate.client, '<null>'),
                COALESCE(candidate.account_hash, '<null>'),
                candidate.quota_key,
                COALESCE(candidate.source, '<null>')
            )
        )::bigint
    ) AS lock_acquired
    FROM candidate
)
INSERT INTO public.rate_limit_observations (
    observed_at,
    client,
    client_version,
    account_hash,
    provider,
    model,
    quota_key,
    quota_period,
    quota_type,
    expected_reset_at,
    remaining_pct,
    quota_limit,
    quota_used,
    quota_remaining,
    billing_period_start_at,
    billing_period_end_at,
    raw_provider_fields,
    evidence,
    source,
    session_id,
    trace_id,
    litellm_call_id
)
SELECT
    candidate.observed_at,
    candidate.client,
    candidate.client_version,
    candidate.account_hash,
    candidate.provider,
    candidate.model,
    candidate.quota_key,
    candidate.quota_period,
    candidate.quota_type,
    candidate.expected_reset_at,
    candidate.remaining_pct,
    candidate.quota_limit,
    candidate.quota_used,
    candidate.quota_remaining,
    candidate.billing_period_start_at,
    candidate.billing_period_end_at,
    COALESCE(candidate.raw_provider_fields, '{}'::jsonb),
    COALESCE(candidate.evidence, '{}'::jsonb),
    candidate.source,
    candidate.session_id,
    candidate.trace_id,
    candidate.litellm_call_id
FROM candidate
CROSS JOIN locked
WHERE NOT EXISTS (
    SELECT 1
    FROM (
        SELECT
            latest.model,
            latest.quota_period,
            latest.quota_type,
            latest.expected_reset_at,
            latest.remaining_pct,
            latest.quota_limit,
            latest.quota_used,
            latest.quota_remaining,
            latest.billing_period_start_at,
            latest.billing_period_end_at,
            latest.raw_provider_fields,
            latest.evidence
        FROM public.rate_limit_observations AS latest
        WHERE latest.provider = candidate.provider
          AND latest.quota_key = candidate.quota_key
          AND latest.client IS NOT DISTINCT FROM candidate.client
          AND latest.account_hash IS NOT DISTINCT FROM candidate.account_hash
          AND latest.source IS NOT DISTINCT FROM candidate.source
        ORDER BY latest.observed_at DESC, latest.id DESC
        LIMIT 1
    ) AS latest
    WHERE latest.model IS NOT DISTINCT FROM candidate.model
      AND latest.quota_period IS NOT DISTINCT FROM candidate.quota_period
      AND latest.quota_type IS NOT DISTINCT FROM candidate.quota_type
      AND (
          latest.expected_reset_at IS NOT DISTINCT FROM candidate.expected_reset_at
          OR (
              latest.expected_reset_at IS NOT NULL
              AND candidate.expected_reset_at IS NOT NULL
              AND ABS(EXTRACT(EPOCH FROM (candidate.expected_reset_at - latest.expected_reset_at))) < 900
          )
      )
      AND latest.remaining_pct IS NOT DISTINCT FROM candidate.remaining_pct
      AND latest.quota_limit IS NOT DISTINCT FROM candidate.quota_limit
      AND latest.quota_used IS NOT DISTINCT FROM candidate.quota_used
      AND latest.quota_remaining IS NOT DISTINCT FROM candidate.quota_remaining
      AND latest.billing_period_start_at IS NOT DISTINCT FROM candidate.billing_period_start_at
      AND latest.billing_period_end_at IS NOT DISTINCT FROM candidate.billing_period_end_at
      AND latest.raw_provider_fields IS NOT DISTINCT FROM COALESCE(candidate.raw_provider_fields, '{}'::jsonb)
      AND latest.evidence IS NOT DISTINCT FROM COALESCE(candidate.evidence, '{}'::jsonb)
)
"""

OBSERVABILITY_SESSION_HISTORY_ANOMALY_SQL = """
WITH recent_session_history AS (
    SELECT
        id,
        created_at,
        COALESCE(end_time, start_time, created_at) AS observed_at,
        litellm_call_id,
        session_id,
        trace_id,
        provider,
        model,
        inbound_model_alias,
        model_group,
        agent_name,
        agent_id,
        repository,
        client_name,
        client_version,
        call_type,
        input_tokens,
        output_tokens,
        total_tokens,
        cache_read_input_tokens,
        cache_creation_input_tokens,
        response_cost_usd,
        tool_call_count,
        git_commit_count,
        git_push_count,
        metadata
    FROM public.session_history
    WHERE COALESCE(end_time, start_time, created_at) >= (
        NOW() - (%s::double precision * INTERVAL '1 hour')
    )
),
tool_activity AS (
    SELECT
        litellm_call_id,
        COUNT(*)::int AS tool_activity_count,
        COALESCE(SUM(git_commit_count), 0)::int AS activity_git_commit_count,
        COALESCE(SUM(git_push_count), 0)::int AS activity_git_push_count,
        BOOL_OR(
            COALESCE(command_text, '') ~* '(^|[[:space:];|&])git[[:space:]]+commit([[:space:]]|$)'
        ) AS activity_git_commit_command,
        BOOL_OR(
            COALESCE(command_text, '') ~* '(^|[[:space:];|&])git[[:space:]]+push([[:space:]]|$)'
        ) AS activity_git_push_command
    FROM public.session_history_tool_activity
    WHERE created_at >= (NOW() - (%s::double precision * INTERVAL '1 hour'))
      AND litellm_call_id IS NOT NULL
    GROUP BY litellm_call_id
),
base AS (
    SELECT
        sh.*,
        COALESCE(ta.tool_activity_count, 0) AS tool_activity_count,
        COALESCE(ta.activity_git_commit_count, 0) AS activity_git_commit_count,
        COALESCE(ta.activity_git_push_count, 0) AS activity_git_push_count,
        COALESCE(ta.activity_git_commit_command, FALSE) AS activity_git_commit_command,
        COALESCE(ta.activity_git_push_command, FALSE) AS activity_git_push_command,
        jsonb_strip_nulls(
            jsonb_build_object(
                'row_id', sh.id,
                'created_at', sh.created_at,
                'observed_at', sh.observed_at,
                'litellm_call_id', sh.litellm_call_id,
                'session_id', sh.session_id,
                'trace_id', sh.trace_id,
                'provider', sh.provider,
                'model', sh.model,
                'inbound_model_alias', sh.inbound_model_alias,
                'model_group', sh.model_group,
                'agent_name', sh.agent_name,
                'agent_id', sh.agent_id,
                'repository', sh.repository,
                'client_name', sh.client_name,
                'client_version', sh.client_version,
                'call_type', sh.call_type
            )
        ) AS sample_base,
        jsonb_strip_nulls(
            jsonb_build_object(
                'requested_model_alias', sh.metadata->>'requested_model_alias',
                'model_alias_label', sh.metadata->>'model_alias_label',
                'codex_auto_agent_alias', sh.metadata->>'codex_auto_agent_alias',
                'anthropic_auto_agent_alias', sh.metadata->>'anthropic_auto_agent_alias',
                'aawm_auto_agent_alias', sh.metadata->>'aawm_auto_agent_alias'
            )
        ) AS metadata_aliases
    FROM recent_session_history AS sh
    LEFT JOIN tool_activity AS ta
      ON ta.litellm_call_id = sh.litellm_call_id
),
anomalies AS (
    SELECT
        id,
        observed_at,
        'missing_provider' AS anomaly_class,
        'session_history.provider should be populated for non-excluded usage rows' AS expected,
        sample_base || jsonb_build_object(
            'observed',
            jsonb_build_object('provider', provider)
        ) AS sample
    FROM base
    WHERE NULLIF(BTRIM(COALESCE(provider, '')), '') IS NULL
      AND COALESCE(metadata->>'session_history_reporting_excluded', 'false') <> 'true'

    UNION ALL

    SELECT
        id,
        observed_at,
        'missing_model' AS anomaly_class,
        'session_history.model should be populated for persisted rows' AS expected,
        sample_base || jsonb_build_object(
            'observed',
            jsonb_build_object('model', model)
        ) AS sample
    FROM base
    WHERE NULLIF(BTRIM(COALESCE(model, '')), '') IS NULL

    UNION ALL

    SELECT
        id,
        observed_at,
        'missing_repository_for_agent_context' AS anomaly_class,
        'TUI agent traffic with agent or AAWM alias context should include repository when derivable' AS expected,
        sample_base || jsonb_build_object(
            'observed',
            jsonb_build_object(
                'repository', repository,
                'client_name', client_name,
                'agent_name', agent_name,
                'agent_id', agent_id,
                'inbound_model_alias', inbound_model_alias
            )
        ) AS sample
    FROM base
    WHERE NULLIF(BTRIM(COALESCE(repository, '')), '') IS NULL
      AND LOWER(COALESCE(client_name, '')) ~ '^(claude|codex|grok)'
      AND (
          NULLIF(BTRIM(COALESCE(agent_name, '')), '') IS NOT NULL
          OR NULLIF(BTRIM(COALESCE(agent_id, '')), '') IS NOT NULL
          OR LEFT(COALESCE(inbound_model_alias, ''), 5) = 'aawm-'
      )
      AND COALESCE(metadata->>'session_history_reporting_excluded', 'false') <> 'true'
      AND NOT (
          LOWER(COALESCE(provider, '')) = 'xai'
          AND LOWER(COALESCE(client_name, '')) = 'grok-build'
          AND LOWER(COALESCE(metadata->>'passthrough_route_family', '')) = 'grok_cli_chat_proxy'
          AND LEFT(COALESCE(inbound_model_alias, ''), 5) <> 'aawm-'
          AND NULLIF(BTRIM(COALESCE(agent_name, '')), '') IS NULL
          AND (
              LOWER(COALESCE(metadata->>'client_user_agent', '')) LIKE 'grok-pager/%%'
              OR LOWER(COALESCE(metadata->>'client_user_agent', '')) LIKE 'grok-shell/%%'
              OR LOWER(COALESCE(metadata->>'client_user_agent', '')) LIKE '%% grok-shell/%%'
          )
      )

    UNION ALL

    SELECT
        id,
        observed_at,
        'alias_metadata_not_promoted' AS anomaly_class,
        'inbound_model_alias should be populated when alias metadata is present' AS expected,
        sample_base || jsonb_build_object(
            'observed',
            jsonb_build_object(
                'inbound_model_alias', inbound_model_alias,
                'metadata_aliases', metadata_aliases
            )
        ) AS sample
    FROM base
    WHERE NULLIF(BTRIM(COALESCE(inbound_model_alias, '')), '') IS NULL
      AND metadata ?| ARRAY[
          'requested_model_alias',
          'model_alias_label',
          'codex_auto_agent_alias',
          'anthropic_auto_agent_alias',
          'aawm_auto_agent_alias'
      ]

    UNION ALL

    SELECT
        id,
        observed_at,
        'negative_token_or_cost_value' AS anomaly_class,
        'token and cost values should not be negative' AS expected,
        sample_base || jsonb_build_object(
            'observed',
            jsonb_build_object(
                'input_tokens', input_tokens,
                'output_tokens', output_tokens,
                'total_tokens', total_tokens,
                'cache_read_input_tokens', cache_read_input_tokens,
                'cache_creation_input_tokens', cache_creation_input_tokens,
                'response_cost_usd', response_cost_usd
            )
        ) AS sample
    FROM base
    WHERE input_tokens < 0
       OR output_tokens < 0
       OR total_tokens < 0
       OR cache_read_input_tokens < 0
       OR cache_creation_input_tokens < 0
       OR response_cost_usd < 0

    UNION ALL

    SELECT
        id,
        observed_at,
        'token_total_less_than_parts' AS anomaly_class,
        'total_tokens should be at least input_tokens + output_tokens when all are reported' AS expected,
        sample_base || jsonb_build_object(
            'observed',
            jsonb_build_object(
                'input_tokens', input_tokens,
                'output_tokens', output_tokens,
                'total_tokens', total_tokens
            )
        ) AS sample
    FROM base
    WHERE total_tokens > 0
      AND input_tokens >= 0
      AND output_tokens >= 0
      AND (input_tokens + output_tokens) > total_tokens

    UNION ALL

    SELECT
        id,
        observed_at,
        'tool_activity_not_reflected_on_session_history' AS anomaly_class,
        'session_history.tool_call_count should reflect persisted tool activity for the same LiteLLM call' AS expected,
        sample_base || jsonb_build_object(
            'observed',
            jsonb_build_object(
                'tool_call_count', tool_call_count,
                'tool_activity_count', tool_activity_count
            )
        ) AS sample
    FROM base
    WHERE tool_activity_count > 0
      AND tool_call_count < tool_activity_count

    UNION ALL

    SELECT
        id,
        observed_at,
        'git_commit_activity_not_reflected' AS anomaly_class,
        'session_history.git_commit_count should reflect git commit tool activity' AS expected,
        sample_base || jsonb_build_object(
            'observed',
            jsonb_build_object(
                'git_commit_count', git_commit_count,
                'activity_git_commit_count', activity_git_commit_count,
                'activity_git_commit_command', activity_git_commit_command
            )
        ) AS sample
    FROM base
    WHERE git_commit_count = 0
      AND (
          activity_git_commit_count > 0
          OR activity_git_commit_command
      )

    UNION ALL

    SELECT
        id,
        observed_at,
        'git_push_activity_not_reflected' AS anomaly_class,
        'session_history.git_push_count should reflect git push tool activity' AS expected,
        sample_base || jsonb_build_object(
            'observed',
            jsonb_build_object(
                'git_push_count', git_push_count,
                'activity_git_push_count', activity_git_push_count,
                'activity_git_push_command', activity_git_push_command
            )
        ) AS sample
    FROM base
    WHERE git_push_count = 0
      AND (
          activity_git_push_count > 0
          OR activity_git_push_command
      )
),
ranked AS (
    SELECT
        *,
        ROW_NUMBER() OVER (
            PARTITION BY anomaly_class
            ORDER BY observed_at DESC, id DESC
        ) AS sample_rank
    FROM anomalies
)
SELECT
    anomaly_class,
    expected,
    COUNT(*)::int AS row_count,
    COALESCE(
        jsonb_agg(sample ORDER BY observed_at DESC, id DESC)
        FILTER (WHERE sample_rank <= %s::int),
        '[]'::jsonb
    ) AS examples
FROM ranked
GROUP BY anomaly_class, expected
ORDER BY row_count DESC, anomaly_class
"""

OBSERVABILITY_RATE_LIMIT_ANOMALY_SQL = """
WITH latest_observations AS (
    SELECT DISTINCT ON (
        provider,
        COALESCE(model, ''),
        quota_key,
        COALESCE(account_hash, ''),
        COALESCE(client, '')
    )
        id,
        observed_at,
        client,
        client_version,
        account_hash,
        provider,
        model,
        quota_key,
        quota_period,
        quota_type,
        expected_reset_at,
        source,
        session_id,
        trace_id,
        litellm_call_id
    FROM public.rate_limit_observations
    WHERE expected_reset_at IS NOT NULL
      AND observed_at >= (
          NOW() - (%s::double precision * INTERVAL '1 hour')
      )
      AND account_hash IS NULL
    ORDER BY
        provider,
        COALESCE(model, ''),
        quota_key,
        COALESCE(account_hash, ''),
        COALESCE(client, ''),
        observed_at DESC,
        id DESC
),
recent_traffic AS (
    SELECT
        provider,
        model,
        model_group,
        inbound_model_alias,
        MAX(COALESCE(end_time, start_time, created_at)) AS last_traffic_at,
        COUNT(*)::int AS traffic_count
    FROM public.session_history
    WHERE COALESCE(end_time, start_time, created_at) >= (
        NOW() - (%s::double precision * INTERVAL '1 hour')
    )
      AND NULLIF(BTRIM(COALESCE(provider, '')), '') IS NOT NULL
    GROUP BY provider, model, model_group, inbound_model_alias
),
matched AS (
    SELECT
        latest.id,
        latest.observed_at,
        latest.client,
        latest.client_version,
        latest.account_hash,
        latest.provider,
        latest.model,
        latest.quota_key,
        latest.quota_period,
        latest.quota_type,
        latest.expected_reset_at,
        latest.source,
        latest.session_id,
        latest.trace_id,
        latest.litellm_call_id,
        SUM(traffic.traffic_count)::int AS recent_traffic_count,
        MAX(traffic.last_traffic_at) AS last_traffic_at
    FROM latest_observations AS latest
    JOIN recent_traffic AS traffic
      ON LOWER(traffic.provider) = LOWER(latest.provider)
     AND (
          latest.model IS NULL
          OR LOWER(latest.model) = LOWER(traffic.model)
          OR LOWER(latest.model) = LOWER(traffic.model_group)
          OR LOWER(latest.model) = LOWER(traffic.inbound_model_alias)
      )
    WHERE latest.expected_reset_at < NOW()
      AND traffic.last_traffic_at > latest.expected_reset_at
    GROUP BY
        latest.id,
        latest.observed_at,
        latest.client,
        latest.client_version,
        latest.account_hash,
        latest.provider,
        latest.model,
        latest.quota_key,
        latest.quota_period,
        latest.quota_type,
        latest.expected_reset_at,
        latest.source,
        latest.session_id,
        latest.trace_id,
        latest.litellm_call_id
),
ranked AS (
    SELECT
        *,
        ROW_NUMBER() OVER (
            PARTITION BY provider, COALESCE(model, ''), quota_key
            ORDER BY last_traffic_at DESC, id DESC
        ) AS sample_rank
    FROM matched
)
SELECT
    'stale_rate_limit_reset_with_recent_traffic' AS anomaly_class,
    'active rate-limit reset timestamps should be in the future when matching provider/model traffic continues' AS expected,
    COUNT(*)::int AS row_count,
    COALESCE(
        jsonb_agg(
            jsonb_strip_nulls(
                jsonb_build_object(
                    'rate_limit_observation_id', id,
                    'observed_at', observed_at,
                    'client', client,
                    'client_version', client_version,
                    'account_hash', account_hash,
                    'provider', provider,
                    'model', model,
                    'quota_key', quota_key,
                    'quota_period', quota_period,
                    'quota_type', quota_type,
                    'expected_reset_at', expected_reset_at,
                    'last_traffic_at', last_traffic_at,
                    'recent_traffic_count', recent_traffic_count,
                    'source', source,
                    'session_id', session_id,
                    'trace_id', trace_id,
                    'litellm_call_id', litellm_call_id
                )
            )
            ORDER BY last_traffic_at DESC, id DESC
        ) FILTER (WHERE sample_rank <= %s::int),
        '[]'::jsonb
    ) AS examples
FROM ranked
"""


@dataclass(frozen=True)
class ProviderStatusLoopConfig:
    apply: bool
    dsn: Optional[str]
    environment: str
    interval_seconds: float
    timeout: float
    ping_count: int
    ping_timeout: int
    skip_icmp: bool
    once: bool
    setup_schema: bool
    db_lock_timeout_ms: int
    db_statement_timeout_ms: int
    schema_dsn: Optional[str] = None
    require_pgbouncer: bool = False
    grok_oidc_refresh_enabled: bool = False
    grok_oidc_auth_file: str = DEFAULT_GROK_OIDC_AUTH_FILE
    grok_oidc_auth_file_source: str = "default"
    grok_oidc_lock_file: str = DEFAULT_GROK_OIDC_LOCK_FILE
    grok_oidc_refresh_interval_seconds: float = (
        DEFAULT_GROK_OIDC_REFRESH_INTERVAL_SECONDS
    )
    grok_oidc_refresh_buffer_seconds: int = (
        grok_oidc_refresh.DEFAULT_GROK_OIDC_REFRESH_BUFFER_SECONDS
    )
    grok_oidc_force_refresh: bool = False
    grok_oidc_http_timeout_seconds: float = DEFAULT_GROK_OIDC_HTTP_TIMEOUT_SECONDS
    codex_oauth_refresh_enabled: bool = False
    codex_auth_file: str = DEFAULT_CODEX_AUTH_FILE
    codex_auth_file_source: str = "default"
    codex_lock_file: str = DEFAULT_CODEX_LOCK_FILE
    codex_refresh_interval_seconds: float = (
        DEFAULT_CODEX_OAUTH_REFRESH_INTERVAL_SECONDS
    )
    codex_refresh_buffer_seconds: int = (
        codex_oauth_refresh.DEFAULT_CODEX_REFRESH_BUFFER_SECONDS
    )
    codex_force_refresh: bool = False
    codex_http_timeout_seconds: float = DEFAULT_CODEX_OAUTH_HTTP_TIMEOUT_SECONDS
    xai_oauth_refresh_enabled: bool = False
    xai_oauth_auth_file: str = DEFAULT_XAI_OAUTH_AUTH_FILE
    xai_oauth_auth_file_source: str = "default"
    xai_oauth_lock_file: str = DEFAULT_XAI_OAUTH_LOCK_FILE
    xai_oauth_scope: str = xai_oauth_refresh.DEFAULT_XAI_OAUTH_SCOPE
    xai_oauth_refresh_interval_seconds: float = (
        DEFAULT_XAI_OAUTH_REFRESH_INTERVAL_SECONDS
    )
    xai_oauth_refresh_buffer_seconds: int = (
        xai_oauth_refresh.DEFAULT_XAI_OAUTH_REFRESH_BUFFER_SECONDS
    )
    xai_oauth_force_refresh: bool = False
    xai_oauth_http_timeout_seconds: float = DEFAULT_XAI_OAUTH_HTTP_TIMEOUT_SECONDS
    grok_billing_poll_enabled: bool = DEFAULT_GROK_BILLING_POLL_ENABLED
    grok_billing_poll_interval_seconds: float = (
        DEFAULT_GROK_BILLING_POLL_INTERVAL_SECONDS
    )
    grok_billing_poll_http_timeout_seconds: float = (
        DEFAULT_GROK_BILLING_POLL_HTTP_TIMEOUT_SECONDS
    )
    grok_billing_url: str = DEFAULT_GROK_BILLING_URL
    grok_billing_client_version: str = DEFAULT_GROK_BILLING_CLIENT_VERSION
    grok_billing_client_identifier: str = DEFAULT_GROK_BILLING_CLIENT_IDENTIFIER
    grok_billing_xai_token_auth: str = DEFAULT_GROK_BILLING_XAI_TOKEN_AUTH
    grok_billing_model: str = DEFAULT_GROK_BILLING_MODEL
    grok_billing_http_method: str = DEFAULT_GROK_BILLING_HTTP_METHOD
    grok_billing_include_model_override: bool = True
    grok_billing_poll_max_attempts: int = DEFAULT_GROK_BILLING_POLL_MAX_ATTEMPTS
    grok_billing_poll_retry_backoff_seconds: float = (
        DEFAULT_GROK_BILLING_POLL_RETRY_BACKOFF_SECONDS
    )
    codex_reset_credit_poll_enabled: bool = DEFAULT_CODEX_RESET_CREDIT_POLL_ENABLED
    codex_reset_credit_poll_interval_seconds: float = (
        DEFAULT_CODEX_RESET_CREDIT_POLL_INTERVAL_SECONDS
    )
    codex_reset_credit_poll_http_timeout_seconds: float = (
        DEFAULT_CODEX_RESET_CREDIT_POLL_HTTP_TIMEOUT_SECONDS
    )
    codex_usage_url: str = DEFAULT_CODEX_USAGE_URL
    codex_reset_credit_poll_max_attempts: int = DEFAULT_CODEX_RESET_CREDIT_POLL_MAX_ATTEMPTS
    codex_reset_credit_poll_retry_backoff_seconds: float = (
        DEFAULT_CODEX_RESET_CREDIT_POLL_RETRY_BACKOFF_SECONDS
    )
    observability_anomaly_scan_enabled: bool = (
        DEFAULT_OBSERVABILITY_ANOMALY_SCAN_ENABLED
    )
    observability_anomaly_scan_interval_seconds: float = (
        DEFAULT_OBSERVABILITY_ANOMALY_SCAN_INTERVAL_SECONDS
    )
    observability_anomaly_scan_lookback_hours: float = (
        DEFAULT_OBSERVABILITY_ANOMALY_SCAN_LOOKBACK_HOURS
    )
    observability_anomaly_scan_error_log_dir: str = (
        DEFAULT_OBSERVABILITY_ANOMALY_SCAN_ERROR_LOG_DIR
    )


@dataclass
class SidecarTaskState:
    grok_oidc_last_attempt_monotonic: Optional[float] = None
    codex_oauth_last_attempt_monotonic: Optional[float] = None
    xai_oauth_last_attempt_monotonic: Optional[float] = None
    grok_billing_last_attempt_monotonic: Optional[float] = None
    codex_reset_credit_last_attempt_monotonic: Optional[float] = None
    observability_anomaly_scan_last_attempt_monotonic: Optional[float] = None


def _env_bool(name: str, default: bool) -> bool:
    raw_value = os.getenv(name)
    if raw_value is None or raw_value == "":
        return default

    value = raw_value.strip().lower()
    if value in TRUE_VALUES:
        return True
    if value in FALSE_VALUES:
        return False
    raise ValueError(f"{name} must be one of {sorted(TRUE_VALUES | FALSE_VALUES)}")


def _first_non_empty_env(*names: str) -> Optional[str]:
    for name in names:
        raw_value = os.getenv(name)
        if isinstance(raw_value, str) and raw_value.strip():
            return raw_value.strip()
    return None


def _resolve_grok_sidecar_auth_file(
    explicit_auth_file: Optional[str],
) -> tuple[str, str]:
    explicit_value = (
        explicit_auth_file.strip()
        if isinstance(explicit_auth_file, str) and explicit_auth_file.strip()
        else None
    )

    aawm_auth_file = os.getenv("AAWM_GROK_OIDC_AUTH_FILE", "").strip()
    if aawm_auth_file:
        return str(Path(aawm_auth_file).expanduser()), "AAWM_GROK_OIDC_AUTH_FILE"

    if explicit_value and explicit_value != DEFAULT_GROK_OIDC_AUTH_FILE:
        return str(Path(explicit_value).expanduser()), "explicit"

    for env_name in GROK_SIDECAR_NATIVE_AUTH_FILE_ENV_VARS:
        env_value = os.getenv(env_name, "").strip()
        if env_value:
            return str(Path(env_value).expanduser()), env_name

    grok_home = os.getenv("GROK_HOME", "").strip()
    if grok_home:
        return str(Path(grok_home).expanduser() / "auth.json"), "GROK_HOME"

    return DEFAULT_GROK_OIDC_AUTH_FILE, "default"



def _resolve_codex_sidecar_auth_file(
    explicit_auth_file: Optional[str],
) -> tuple[str, str]:
    explicit_value = (
        explicit_auth_file.strip()
        if isinstance(explicit_auth_file, str) and explicit_auth_file.strip()
        else None
    )

    aawm_auth_file = os.getenv("AAWM_CODEX_AUTH_FILE", "").strip()
    if aawm_auth_file:
        return str(Path(aawm_auth_file).expanduser()), "AAWM_CODEX_AUTH_FILE"

    if explicit_value and explicit_value != DEFAULT_CODEX_AUTH_FILE:
        return str(Path(explicit_value).expanduser()), "explicit"

    for env_name in CODEX_SIDECAR_AUTH_FILE_ENV_VARS:
        env_value = os.getenv(env_name, "").strip()
        if env_value:
            return str(Path(env_value).expanduser()), env_name

    for env_name in CODEX_SIDECAR_TOKEN_DIR_ENV_VARS:
        token_dir = os.getenv(env_name, "").strip()
        if token_dir:
            return str(Path(token_dir).expanduser() / "auth.json"), env_name

    return DEFAULT_CODEX_AUTH_FILE, "default"


def _resolve_xai_oauth_sidecar_auth_file(
    explicit_auth_file: Optional[str],
) -> tuple[str, str]:
    explicit_value = (
        explicit_auth_file.strip()
        if isinstance(explicit_auth_file, str) and explicit_auth_file.strip()
        else None
    )

    aawm_auth_file = os.getenv("AAWM_XAI_OAUTH_AUTH_FILE", "").strip()
    if aawm_auth_file:
        return str(Path(aawm_auth_file).expanduser()), "AAWM_XAI_OAUTH_AUTH_FILE"

    if explicit_value and explicit_value != DEFAULT_XAI_OAUTH_AUTH_FILE:
        return str(Path(explicit_value).expanduser()), "explicit"

    for env_name in XAI_OAUTH_SIDECAR_AUTH_FILE_ENV_VARS:
        env_value = os.getenv(env_name, "").strip()
        if env_value:
            return str(Path(env_value).expanduser()), env_name

    return DEFAULT_XAI_OAUTH_AUTH_FILE, "default"


def _resolve_grok_billing_client_version() -> str:
    return (
        _first_non_empty_env("AAWM_GROK_BILLING_CLIENT_VERSION")
        or _first_non_empty_env("LITELLM_XAI_GROK_CLIENT_VERSION")
        or _first_non_empty_env("GROK_CLIENT_VERSION")
        or DEFAULT_GROK_BILLING_CLIENT_VERSION
    )


def _env_float(name: str, default: float) -> float:
    raw_value = os.getenv(name)
    if raw_value is None or raw_value == "":
        return default
    return float(raw_value)


def _env_int(name: str, default: int) -> int:
    raw_value = os.getenv(name)
    if raw_value is None or raw_value == "":
        return default
    return int(raw_value)


def _build_parser() -> argparse.ArgumentParser:  # noqa: PLR0915
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--apply",
        dest="apply",
        action="store_true",
        default=_env_bool("AAWM_PROVIDER_STATUS_APPLY", True),
        help="Insert observations into Postgres. Defaults to AAWM_PROVIDER_STATUS_APPLY or true.",
    )
    parser.add_argument(
        "--dry-run",
        dest="apply",
        action="store_false",
        help="Collect and log a summary without inserting observations.",
    )
    parser.add_argument("--dsn", default=os.getenv("AAWM_PROVIDER_STATUS_DSN"))
    parser.add_argument(
        "--schema-dsn",
        default=(
            os.getenv("AAWM_PROVIDER_STATUS_SCHEMA_DSN")
            or os.getenv("AAWM_DIRECT_DATABASE_URL")
        ),
        help=(
            "Direct Postgres DSN for explicit provider-status schema setup. "
            "Defaults to AAWM_PROVIDER_STATUS_SCHEMA_DSN or AAWM_DIRECT_DATABASE_URL."
        ),
    )
    parser.add_argument(
        "--environment",
        default=os.getenv("AAWM_LITELLM_ENVIRONMENT", "dev"),
    )
    parser.add_argument(
        "--interval-seconds",
        type=float,
        default=_env_float("AAWM_PROVIDER_STATUS_INTERVAL_SECONDS", 300.0),
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=_env_float("AAWM_PROVIDER_STATUS_TIMEOUT", 2.0),
    )
    parser.add_argument(
        "--ping-count",
        type=int,
        default=_env_int("AAWM_PROVIDER_STATUS_PING_COUNT", 1),
    )
    parser.add_argument(
        "--ping-timeout",
        type=int,
        default=_env_int("AAWM_PROVIDER_STATUS_PING_TIMEOUT", 2),
    )
    parser.add_argument(
        "--skip-icmp",
        action="store_true",
        default=_env_bool("AAWM_PROVIDER_STATUS_SKIP_ICMP", False),
    )
    parser.add_argument(
        "--once",
        action="store_true",
        default=_env_bool("AAWM_PROVIDER_STATUS_ONCE", False),
        help="Run exactly one cycle, then exit.",
    )
    schema_group = parser.add_mutually_exclusive_group()
    schema_group.add_argument(
        "--setup-schema",
        dest="setup_schema",
        action="store_true",
        default=_env_bool("AAWM_PROVIDER_STATUS_SETUP_SCHEMA_ON_START", False),
        help=(
            "Run provider-status schema setup once before the loop starts. "
            "Defaults to AAWM_PROVIDER_STATUS_SETUP_SCHEMA_ON_START or false."
        ),
    )
    schema_group.add_argument(
        "--no-setup-schema",
        dest="setup_schema",
        action="store_false",
        help="Skip startup schema setup; steady-state cycles never run DDL.",
    )
    parser.add_argument(
        "--db-lock-timeout-ms",
        type=int,
        default=_env_int(
            "AAWM_PROVIDER_STATUS_DB_LOCK_TIMEOUT_MS",
            probes.DEFAULT_DB_LOCK_TIMEOUT_MS,
        ),
    )
    parser.add_argument(
        "--db-statement-timeout-ms",
        type=int,
        default=_env_int(
            "AAWM_PROVIDER_STATUS_DB_STATEMENT_TIMEOUT_MS",
            probes.DEFAULT_DB_STATEMENT_TIMEOUT_MS,
        ),
    )
    parser.add_argument(
        "--require-pgbouncer",
        action="store_true",
        default=_env_bool("AAWM_PROVIDER_STATUS_REQUIRE_PGBOUNCER", False),
        help=(
            "Fail startup if steady-state writes do not resolve to the "
            "PgBouncer transaction pool."
        ),
    )
    grok_group = parser.add_mutually_exclusive_group()
    grok_group.add_argument(
        "--grok-oidc-refresh-enabled",
        dest="grok_oidc_refresh_enabled",
        action="store_true",
        default=_env_bool("AAWM_GROK_OIDC_REFRESH_ENABLED", False),
        help=(
            "Run the Grok OIDC auth-file refresh task from this sidecar loop. "
            "Defaults to AAWM_GROK_OIDC_REFRESH_ENABLED or false."
        ),
    )
    grok_group.add_argument(
        "--no-grok-oidc-refresh",
        dest="grok_oidc_refresh_enabled",
        action="store_false",
        help="Disable the Grok OIDC auth-file refresh task.",
    )
    parser.add_argument(
        "--grok-oidc-auth-file",
        default=os.getenv("AAWM_GROK_OIDC_AUTH_FILE", DEFAULT_GROK_OIDC_AUTH_FILE),
        help=(
            "Grok CLI auth JSON file maintained by this sidecar. Defaults to "
            "AAWM_GROK_OIDC_AUTH_FILE, then native Grok auth-file env vars, "
            "GROK_HOME/auth.json, or /home/zepfu/.grok/auth.json."
        ),
    )
    parser.add_argument(
        "--grok-oidc-lock-file",
        default=os.getenv("AAWM_GROK_OIDC_LOCK_FILE", DEFAULT_GROK_OIDC_LOCK_FILE),
        help=(
            "Lock file for sidecar Grok OIDC refresh writes. Defaults to "
            "AAWM_GROK_OIDC_LOCK_FILE or /home/zepfu/.grok/auth.json.lock."
        ),
    )
    parser.add_argument(
        "--grok-oidc-refresh-interval-seconds",
        type=float,
        default=_env_float(
            "AAWM_GROK_OIDC_REFRESH_INTERVAL_SECONDS",
            DEFAULT_GROK_OIDC_REFRESH_INTERVAL_SECONDS,
        ),
        help=(
            "Minimum seconds between Grok OIDC refresh attempts. Defaults to "
            "AAWM_GROK_OIDC_REFRESH_INTERVAL_SECONDS or 3600."
        ),
    )
    parser.add_argument(
        "--grok-oidc-refresh-buffer-seconds",
        type=int,
        default=_env_int(
            "AAWM_GROK_OIDC_REFRESH_BUFFER_SECONDS",
            grok_oidc_refresh.DEFAULT_GROK_OIDC_REFRESH_BUFFER_SECONDS,
        ),
        help=(
            "Refresh buffer for non-forced Grok OIDC refreshes. Defaults to "
            "AAWM_GROK_OIDC_REFRESH_BUFFER_SECONDS or 300."
        ),
    )
    parser.add_argument(
        "--grok-oidc-force-refresh",
        action="store_true",
        default=_env_bool("AAWM_GROK_OIDC_FORCE_REFRESH", False),
        help=(
            "Refresh the Grok OIDC credential on every scheduled attempt even "
            "when the current access token is still valid."
        ),
    )
    parser.add_argument(
        "--grok-oidc-http-timeout-seconds",
        type=float,
        default=_env_float(
            "AAWM_GROK_OIDC_HTTP_TIMEOUT_SECONDS",
            DEFAULT_GROK_OIDC_HTTP_TIMEOUT_SECONDS,
        ),
        help=(
            "HTTP timeout for Grok OIDC token endpoint calls. Defaults to "
            "AAWM_GROK_OIDC_HTTP_TIMEOUT_SECONDS or 30."
        ),
    )
    codex_group = parser.add_mutually_exclusive_group()
    codex_group.add_argument(
        "--codex-oauth-refresh-enabled",
        dest="codex_oauth_refresh_enabled",
        action="store_true",
        default=_env_bool("AAWM_CODEX_OAUTH_REFRESH_ENABLED", False),
        help=(
            "Run the Codex OAuth auth-file refresh task from this sidecar "
            "loop. Defaults to AAWM_CODEX_OAUTH_REFRESH_ENABLED or false."
        ),
    )
    codex_group.add_argument(
        "--no-codex-oauth-refresh",
        dest="codex_oauth_refresh_enabled",
        action="store_false",
        help="Disable the Codex OAuth auth-file refresh task.",
    )
    parser.add_argument(
        "--codex-auth-file",
        default=os.getenv("AAWM_CODEX_AUTH_FILE", DEFAULT_CODEX_AUTH_FILE),
        help=(
            "Codex OAuth auth JSON file maintained by this sidecar. Defaults "
            "to AAWM_CODEX_AUTH_FILE, then LiteLLM/Codex auth-file env vars, "
            "token-dir env vars, or /home/zepfu/.codex/auth.json."
        ),
    )
    parser.add_argument(
        "--codex-lock-file",
        default=os.getenv("AAWM_CODEX_LOCK_FILE", DEFAULT_CODEX_LOCK_FILE),
        help=(
            "Lock file for sidecar Codex OAuth refresh writes. Defaults to "
            "AAWM_CODEX_LOCK_FILE or /home/zepfu/.codex/auth.json.lock."
        ),
    )
    parser.add_argument(
        "--codex-refresh-interval-seconds",
        type=float,
        default=_env_float(
            "AAWM_CODEX_OAUTH_REFRESH_INTERVAL_SECONDS",
            DEFAULT_CODEX_OAUTH_REFRESH_INTERVAL_SECONDS,
        ),
        help=(
            "Minimum seconds between Codex OAuth refresh attempts. Defaults "
            "to AAWM_CODEX_OAUTH_REFRESH_INTERVAL_SECONDS or 3600."
        ),
    )
    parser.add_argument(
        "--codex-refresh-buffer-seconds",
        type=int,
        default=_env_int(
            "AAWM_CODEX_OAUTH_REFRESH_BUFFER_SECONDS",
            codex_oauth_refresh.DEFAULT_CODEX_REFRESH_BUFFER_SECONDS,
        ),
        help=(
            "Refresh buffer for non-forced Codex OAuth refreshes. Defaults to "
            "AAWM_CODEX_OAUTH_REFRESH_BUFFER_SECONDS or 300."
        ),
    )
    parser.add_argument(
        "--codex-force-refresh",
        action="store_true",
        default=_env_bool("AAWM_CODEX_OAUTH_FORCE_REFRESH", False),
        help=(
            "Refresh the Codex OAuth credential on every scheduled attempt "
            "even when the current access token is still valid."
        ),
    )
    parser.add_argument(
        "--codex-http-timeout-seconds",
        type=float,
        default=_env_float(
            "AAWM_CODEX_OAUTH_HTTP_TIMEOUT_SECONDS",
            DEFAULT_CODEX_OAUTH_HTTP_TIMEOUT_SECONDS,
        ),
        help=(
            "HTTP timeout for Codex OAuth token endpoint calls. Defaults to "
            "AAWM_CODEX_OAUTH_HTTP_TIMEOUT_SECONDS or 30."
        ),
    )
    xai_oauth_group = parser.add_mutually_exclusive_group()
    xai_oauth_group.add_argument(
        "--xai-oauth-refresh-enabled",
        dest="xai_oauth_refresh_enabled",
        action="store_true",
        default=_env_bool("AAWM_XAI_OAUTH_REFRESH_ENABLED", False),
        help=(
            "Run the managed xAI OAuth auth-file refresh task from this sidecar "
            "loop. Defaults to AAWM_XAI_OAUTH_REFRESH_ENABLED or false."
        ),
    )
    xai_oauth_group.add_argument(
        "--no-xai-oauth-refresh",
        dest="xai_oauth_refresh_enabled",
        action="store_false",
        help="Disable the managed xAI OAuth auth-file refresh task.",
    )
    parser.add_argument(
        "--xai-oauth-auth-file",
        default=os.getenv("AAWM_XAI_OAUTH_AUTH_FILE", DEFAULT_XAI_OAUTH_AUTH_FILE),
        help=(
            "Managed xAI OAuth auth JSON file maintained by this sidecar. "
            "Defaults to AAWM_XAI_OAUTH_AUTH_FILE, then LiteLLM managed xAI "
            "OAuth auth-file env vars, or /home/zepfu/.litellm/xai/oauth-auth.json."
        ),
    )
    parser.add_argument(
        "--xai-oauth-lock-file",
        default=os.getenv("AAWM_XAI_OAUTH_LOCK_FILE", DEFAULT_XAI_OAUTH_LOCK_FILE),
        help=(
            "Lock file for sidecar managed xAI OAuth refresh writes. Defaults "
            "to AAWM_XAI_OAUTH_LOCK_FILE or /home/zepfu/.litellm/xai/oauth-auth.json.lock."
        ),
    )
    parser.add_argument(
        "--xai-oauth-scope",
        default=(
            os.getenv("AAWM_XAI_OAUTH_SCOPE")
            or os.getenv("LITELLM_XAI_OAUTH_SCOPE")
            or xai_oauth_refresh.DEFAULT_XAI_OAUTH_SCOPE
        ),
        help=(
            "Managed xAI OAuth credential scope. Defaults to AAWM_XAI_OAUTH_SCOPE, "
            "LITELLM_XAI_OAUTH_SCOPE, or the Grok subscription scope."
        ),
    )
    parser.add_argument(
        "--xai-oauth-refresh-interval-seconds",
        type=float,
        default=_env_float(
            "AAWM_XAI_OAUTH_REFRESH_INTERVAL_SECONDS",
            DEFAULT_XAI_OAUTH_REFRESH_INTERVAL_SECONDS,
        ),
        help=(
            "Minimum seconds between managed xAI OAuth refresh attempts. "
            "Defaults to AAWM_XAI_OAUTH_REFRESH_INTERVAL_SECONDS or 3600."
        ),
    )
    parser.add_argument(
        "--xai-oauth-refresh-buffer-seconds",
        type=int,
        default=_env_int(
            "AAWM_XAI_OAUTH_REFRESH_BUFFER_SECONDS",
            xai_oauth_refresh.DEFAULT_XAI_OAUTH_REFRESH_BUFFER_SECONDS,
        ),
        help=(
            "Refresh buffer for non-forced managed xAI OAuth refreshes. "
            "Defaults to AAWM_XAI_OAUTH_REFRESH_BUFFER_SECONDS or 300."
        ),
    )
    parser.add_argument(
        "--xai-oauth-force-refresh",
        action="store_true",
        default=_env_bool("AAWM_XAI_OAUTH_FORCE_REFRESH", False),
        help=(
            "Refresh the managed xAI OAuth credential on every scheduled "
            "attempt even when the current access token is still valid."
        ),
    )
    parser.add_argument(
        "--xai-oauth-http-timeout-seconds",
        type=float,
        default=_env_float(
            "AAWM_XAI_OAUTH_HTTP_TIMEOUT_SECONDS",
            DEFAULT_XAI_OAUTH_HTTP_TIMEOUT_SECONDS,
        ),
        help=(
            "HTTP timeout for managed xAI OAuth token endpoint calls. Defaults "
            "to AAWM_XAI_OAUTH_HTTP_TIMEOUT_SECONDS or 30."
        ),
    )
    billing_group = parser.add_mutually_exclusive_group()
    billing_group.add_argument(
        "--grok-billing-poll-enabled",
        dest="grok_billing_poll_enabled",
        action="store_true",
        default=_env_bool(
            "AAWM_GROK_BILLING_POLL_ENABLED",
            DEFAULT_GROK_BILLING_POLL_ENABLED,
        ),
        help=(
            "Run the hourly Grok billing poll from this sidecar loop. "
            "Defaults to AAWM_GROK_BILLING_POLL_ENABLED or false."
        ),
    )
    billing_group.add_argument(
        "--no-grok-billing-poll",
        dest="grok_billing_poll_enabled",
        action="store_false",
        help="Disable the hourly Grok billing poll task.",
    )
    parser.add_argument(
        "--grok-billing-poll-interval-seconds",
        type=float,
        default=_env_float(
            "AAWM_GROK_BILLING_POLL_INTERVAL_SECONDS",
            DEFAULT_GROK_BILLING_POLL_INTERVAL_SECONDS,
        ),
        help=(
            "Minimum seconds between Grok billing poll attempts. Defaults to "
            "AAWM_GROK_BILLING_POLL_INTERVAL_SECONDS or 3600."
        ),
    )
    parser.add_argument(
        "--grok-billing-poll-http-timeout-seconds",
        type=float,
        default=_env_float(
            "AAWM_GROK_BILLING_POLL_HTTP_TIMEOUT_SECONDS",
            DEFAULT_GROK_BILLING_POLL_HTTP_TIMEOUT_SECONDS,
        ),
        help=(
            "HTTP timeout for Grok billing poll calls. Defaults to "
            "AAWM_GROK_BILLING_POLL_HTTP_TIMEOUT_SECONDS or 30."
        ),
    )
    parser.add_argument(
        "--grok-billing-url",
        default=os.getenv("AAWM_GROK_BILLING_URL", DEFAULT_GROK_BILLING_URL),
        help=(
            "Grok billing endpoint polled by the sidecar. Defaults to "
            "AAWM_GROK_BILLING_URL or the Grok CLI credits billing URL."
        ),
    )
    parser.add_argument(
        "--grok-billing-client-version",
        default=_resolve_grok_billing_client_version(),
        help=(
            "Grok CLI client version header for billing polls. Defaults to "
            "AAWM_GROK_BILLING_CLIENT_VERSION, LITELLM_XAI_GROK_CLIENT_VERSION, "
            "GROK_CLIENT_VERSION, or 0.2.55."
        ),
    )
    parser.add_argument(
        "--grok-billing-client-identifier",
        default=os.getenv(
            "AAWM_GROK_BILLING_CLIENT_IDENTIFIER",
            os.getenv(
                "LITELLM_XAI_GROK_CLIENT_IDENTIFIER",
                DEFAULT_GROK_BILLING_CLIENT_IDENTIFIER,
            ),
        ),
        help=(
            "Grok CLI client identifier header for billing polls. Defaults to "
            "AAWM_GROK_BILLING_CLIENT_IDENTIFIER, LITELLM_XAI_GROK_CLIENT_IDENTIFIER, "
            "or grok-cli."
        ),
    )
    parser.add_argument(
        "--grok-billing-xai-token-auth",
        default=os.getenv(
            "AAWM_GROK_BILLING_XAI_TOKEN_AUTH",
            os.getenv(
                "LITELLM_XAI_GROK_XAI_TOKEN_AUTH",
                DEFAULT_GROK_BILLING_XAI_TOKEN_AUTH,
            ),
        ),
        help=(
            "x-xai-token-auth header value for billing polls. Defaults to "
            "AAWM_GROK_BILLING_XAI_TOKEN_AUTH, LITELLM_XAI_GROK_XAI_TOKEN_AUTH, "
            "or xai-grok-cli."
        ),
    )
    parser.add_argument(
        "--grok-billing-model",
        default=os.getenv("AAWM_GROK_BILLING_MODEL", DEFAULT_GROK_BILLING_MODEL),
        help=(
            "Model label stored with Grok billing snapshots. Defaults to "
            "AAWM_GROK_BILLING_MODEL or grok-build."
        ),
    )
    parser.add_argument(
        "--grok-billing-http-method",
        default=os.getenv(
            "AAWM_GROK_BILLING_HTTP_METHOD",
            DEFAULT_GROK_BILLING_HTTP_METHOD,
        ),
        help=(
            "HTTP method used for Grok billing poll requests. Defaults to "
            "AAWM_GROK_BILLING_HTTP_METHOD or GET."
        ),
    )
    billing_override_group = parser.add_mutually_exclusive_group()
    billing_override_group.add_argument(
        "--grok-billing-include-model-override",
        dest="grok_billing_include_model_override",
        action="store_true",
        default=_env_bool(
            "AAWM_GROK_BILLING_INCLUDE_MODEL_OVERRIDE",
            True,
        ),
        help=(
            "Include native Grok billing request-shape headers on billing poll "
            "requests. Defaults to AAWM_GROK_BILLING_INCLUDE_MODEL_OVERRIDE or "
            "true."
        ),
    )
    billing_override_group.add_argument(
        "--no-grok-billing-include-model-override",
        dest="grok_billing_include_model_override",
        action="store_false",
        help="Omit x-grok-model-override from Grok billing poll requests.",
    )
    parser.add_argument(
        "--grok-billing-poll-max-attempts",
        type=int,
        default=_env_int(
            "AAWM_GROK_BILLING_POLL_MAX_ATTEMPTS",
            DEFAULT_GROK_BILLING_POLL_MAX_ATTEMPTS,
        ),
        help=(
            "Maximum Grok billing poll attempts per scheduled run, including "
            "retries. Defaults to AAWM_GROK_BILLING_POLL_MAX_ATTEMPTS or 3."
        ),
    )
    parser.add_argument(
        "--grok-billing-poll-retry-backoff-seconds",
        type=float,
        default=_env_float(
            "AAWM_GROK_BILLING_POLL_RETRY_BACKOFF_SECONDS",
            DEFAULT_GROK_BILLING_POLL_RETRY_BACKOFF_SECONDS,
        ),
        help=(
            "Base backoff seconds between retryable Grok billing poll failures. "
            "Defaults to AAWM_GROK_BILLING_POLL_RETRY_BACKOFF_SECONDS or 0.5."
        ),
    )

    codex_credit_group = parser.add_mutually_exclusive_group()
    codex_credit_group.add_argument(
        "--codex-reset-credit-poll-enabled",
        dest="codex_reset_credit_poll_enabled",
        action="store_true",
        default=_env_bool(
            "AAWM_CODEX_RESET_CREDIT_POLL_ENABLED",
            DEFAULT_CODEX_RESET_CREDIT_POLL_ENABLED,
        ),
        help=(
            "Run the hourly Codex reset-credit poll from this sidecar loop. "
            "Defaults to AAWM_CODEX_RESET_CREDIT_POLL_ENABLED or false."
        ),
    )
    codex_credit_group.add_argument(
        "--no-codex-reset-credit-poll",
        dest="codex_reset_credit_poll_enabled",
        action="store_false",
        help="Disable the hourly Codex reset-credit poll task.",
    )
    parser.add_argument(
        "--codex-reset-credit-poll-interval-seconds",
        type=float,
        default=_env_float(
            "AAWM_CODEX_RESET_CREDIT_POLL_INTERVAL_SECONDS",
            DEFAULT_CODEX_RESET_CREDIT_POLL_INTERVAL_SECONDS,
        ),
        help=(
            "Minimum seconds between Codex reset-credit poll attempts. Defaults "
            "to AAWM_CODEX_RESET_CREDIT_POLL_INTERVAL_SECONDS or 3600."
        ),
    )
    parser.add_argument(
        "--codex-reset-credit-poll-http-timeout-seconds",
        type=float,
        default=_env_float(
            "AAWM_CODEX_RESET_CREDIT_POLL_HTTP_TIMEOUT_SECONDS",
            DEFAULT_CODEX_RESET_CREDIT_POLL_HTTP_TIMEOUT_SECONDS,
        ),
        help=(
            "HTTP timeout for Codex reset-credit poll calls. Defaults to "
            "AAWM_CODEX_RESET_CREDIT_POLL_HTTP_TIMEOUT_SECONDS or 30."
        ),
    )
    parser.add_argument(
        "--codex-usage-url",
        default=os.getenv("AAWM_CODEX_USAGE_URL", DEFAULT_CODEX_USAGE_URL),
        help=(
            "Codex usage endpoint polled by the sidecar. Defaults to "
            "AAWM_CODEX_USAGE_URL or the native ChatGPT wham rate-limit-reset-credits URL."
        ),
    )
    parser.add_argument(
        "--codex-reset-credit-poll-max-attempts",
        type=int,
        default=_env_int(
            "AAWM_CODEX_RESET_CREDIT_POLL_MAX_ATTEMPTS",
            DEFAULT_CODEX_RESET_CREDIT_POLL_MAX_ATTEMPTS,
        ),
        help=(
            "Maximum Codex reset-credit poll attempts per scheduled run, including "
            "retries. Defaults to AAWM_CODEX_RESET_CREDIT_POLL_MAX_ATTEMPTS or 3."
        ),
    )
    parser.add_argument(
        "--codex-reset-credit-poll-retry-backoff-seconds",
        type=float,
        default=_env_float(
            "AAWM_CODEX_RESET_CREDIT_POLL_RETRY_BACKOFF_SECONDS",
            DEFAULT_CODEX_RESET_CREDIT_POLL_RETRY_BACKOFF_SECONDS,
        ),
        help=(
            "Base backoff seconds between retryable Codex reset-credit poll "
            "failures. Defaults to "
            "AAWM_CODEX_RESET_CREDIT_POLL_RETRY_BACKOFF_SECONDS or 0.5."
        ),
    )
    anomaly_group = parser.add_mutually_exclusive_group()
    anomaly_group.add_argument(
        "--observability-anomaly-scan-enabled",
        dest="observability_anomaly_scan_enabled",
        action="store_true",
        default=_env_bool(
            "AAWM_OBSERVABILITY_ANOMALY_SCAN_ENABLED",
            DEFAULT_OBSERVABILITY_ANOMALY_SCAN_ENABLED,
        ),
        help=(
            "Run the hourly session-history/rate-limit anomaly scan from this "
            "sidecar loop. Defaults to AAWM_OBSERVABILITY_ANOMALY_SCAN_ENABLED "
            "or false."
        ),
    )
    anomaly_group.add_argument(
        "--no-observability-anomaly-scan",
        dest="observability_anomaly_scan_enabled",
        action="store_false",
        help="Disable the observability anomaly scan task.",
    )
    parser.add_argument(
        "--observability-anomaly-scan-interval-seconds",
        type=float,
        default=_env_float(
            "AAWM_OBSERVABILITY_ANOMALY_SCAN_INTERVAL_SECONDS",
            DEFAULT_OBSERVABILITY_ANOMALY_SCAN_INTERVAL_SECONDS,
        ),
        help=(
            "Minimum seconds between observability anomaly scans. Defaults to "
            "AAWM_OBSERVABILITY_ANOMALY_SCAN_INTERVAL_SECONDS or 3600."
        ),
    )
    parser.add_argument(
        "--observability-anomaly-scan-lookback-hours",
        type=float,
        default=_env_float(
            "AAWM_OBSERVABILITY_ANOMALY_SCAN_LOOKBACK_HOURS",
            DEFAULT_OBSERVABILITY_ANOMALY_SCAN_LOOKBACK_HOURS,
        ),
        help=(
            "Recent database window scanned for telemetry anomalies. Defaults "
            "to AAWM_OBSERVABILITY_ANOMALY_SCAN_LOOKBACK_HOURS or 4."
        ),
    )
    parser.add_argument(
        "--observability-anomaly-scan-error-log-dir",
        default=os.getenv(
            "AAWM_OBSERVABILITY_ANOMALY_SCAN_ERROR_LOG_DIR",
            os.getenv(
                "LITELLM_AAWM_ERROR_LOG_DIR",
                DEFAULT_OBSERVABILITY_ANOMALY_SCAN_ERROR_LOG_DIR,
            ),
        ),
        help=(
            "Directory where detected anomalies are appended as "
            "<environment>-error.jsonl. Defaults to "
            "AAWM_OBSERVABILITY_ANOMALY_SCAN_ERROR_LOG_DIR, "
            "LITELLM_AAWM_ERROR_LOG_DIR, or /app/.analysis."
        ),
    )
    return parser


def _validate_config_args(args: argparse.Namespace) -> None:
    _validate_core_config_args(args)
    _validate_grok_oidc_config_args(args)
    _validate_codex_config_args(args)
    _validate_xai_oauth_config_args(args)
    _validate_grok_billing_config_args(args)
    _validate_observability_anomaly_scan_config_args(args)
    _validate_codex_reset_credit_poll_config_args(args)


def _validate_core_config_args(args: argparse.Namespace) -> None:
    if args.interval_seconds <= 0:
        raise SystemExit("--interval-seconds must be greater than 0")
    if args.timeout <= 0:
        raise SystemExit("--timeout must be greater than 0")
    if args.ping_count <= 0:
        raise SystemExit("--ping-count must be greater than 0")
    if args.ping_timeout <= 0:
        raise SystemExit("--ping-timeout must be greater than 0")
    if args.db_lock_timeout_ms <= 0:
        raise SystemExit("--db-lock-timeout-ms must be greater than 0")
    if args.db_statement_timeout_ms <= 0:
        raise SystemExit("--db-statement-timeout-ms must be greater than 0")


def _validate_grok_oidc_config_args(args: argparse.Namespace) -> None:
    if args.grok_oidc_refresh_interval_seconds <= 0:
        raise SystemExit("--grok-oidc-refresh-interval-seconds must be greater than 0")
    if args.grok_oidc_refresh_buffer_seconds < 0:
        raise SystemExit("--grok-oidc-refresh-buffer-seconds must be non-negative")
    if args.grok_oidc_http_timeout_seconds <= 0:
        raise SystemExit("--grok-oidc-http-timeout-seconds must be greater than 0")



def _validate_codex_config_args(args: argparse.Namespace) -> None:
    if args.codex_refresh_interval_seconds <= 0:
        raise SystemExit("--codex-refresh-interval-seconds must be greater than 0")
    if args.codex_refresh_buffer_seconds < 0:
        raise SystemExit("--codex-refresh-buffer-seconds must be non-negative")
    if args.codex_http_timeout_seconds <= 0:
        raise SystemExit("--codex-http-timeout-seconds must be greater than 0")


def _validate_xai_oauth_config_args(args: argparse.Namespace) -> None:
    if not str(args.xai_oauth_scope).strip():
        raise SystemExit("--xai-oauth-scope must not be empty")
    if args.xai_oauth_refresh_interval_seconds <= 0:
        raise SystemExit("--xai-oauth-refresh-interval-seconds must be greater than 0")
    if args.xai_oauth_refresh_buffer_seconds < 0:
        raise SystemExit("--xai-oauth-refresh-buffer-seconds must be non-negative")
    if args.xai_oauth_http_timeout_seconds <= 0:
        raise SystemExit("--xai-oauth-http-timeout-seconds must be greater than 0")


def _validate_grok_billing_config_args(args: argparse.Namespace) -> None:
    if args.grok_billing_poll_interval_seconds <= 0:
        raise SystemExit("--grok-billing-poll-interval-seconds must be greater than 0")
    if args.grok_billing_poll_http_timeout_seconds <= 0:
        raise SystemExit("--grok-billing-poll-http-timeout-seconds must be greater than 0")
    if not str(args.grok_billing_url).strip():
        raise SystemExit("--grok-billing-url must not be empty")
    if not str(args.grok_billing_client_version).strip():
        raise SystemExit("--grok-billing-client-version must not be empty")
    if not str(args.grok_billing_client_identifier).strip():
        raise SystemExit("--grok-billing-client-identifier must not be empty")
    if not str(args.grok_billing_xai_token_auth).strip():
        raise SystemExit("--grok-billing-xai-token-auth must not be empty")
    if not str(args.grok_billing_model).strip():
        raise SystemExit("--grok-billing-model must not be empty")
    grok_billing_http_method = str(args.grok_billing_http_method).strip().upper()
    if not grok_billing_http_method:
        raise SystemExit("--grok-billing-http-method must not be empty")
    if args.grok_billing_poll_max_attempts <= 0:
        raise SystemExit("--grok-billing-poll-max-attempts must be greater than 0")
    if args.grok_billing_poll_retry_backoff_seconds < 0:
        raise SystemExit(
            "--grok-billing-poll-retry-backoff-seconds must be non-negative"
        )


def _validate_observability_anomaly_scan_config_args(args: argparse.Namespace) -> None:
    if args.observability_anomaly_scan_interval_seconds <= 0:
        raise SystemExit(
            "--observability-anomaly-scan-interval-seconds must be greater than 0"
        )
    if args.observability_anomaly_scan_lookback_hours <= 0:
        raise SystemExit(
            "--observability-anomaly-scan-lookback-hours must be greater than 0"
        )
    if not str(args.observability_anomaly_scan_error_log_dir).strip():
        raise SystemExit("--observability-anomaly-scan-error-log-dir must not be empty")


def _validate_codex_reset_credit_poll_config_args(args: argparse.Namespace) -> None:
    if args.codex_reset_credit_poll_interval_seconds <= 0:
        raise SystemExit(
            "--codex-reset-credit-poll-interval-seconds must be greater than 0"
        )
    if args.codex_reset_credit_poll_http_timeout_seconds <= 0:
        raise SystemExit(
            "--codex-reset-credit-poll-http-timeout-seconds must be greater than 0"
        )
    if not str(args.codex_usage_url).strip():
        raise SystemExit("--codex-usage-url must not be empty")
    if args.codex_reset_credit_poll_max_attempts <= 0:
        raise SystemExit(
            "--codex-reset-credit-poll-max-attempts must be greater than 0"
        )
    if args.codex_reset_credit_poll_retry_backoff_seconds < 0:
        raise SystemExit(
            "--codex-reset-credit-poll-retry-backoff-seconds must be non-negative"
        )


def parse_config(argv: Optional[Sequence[str]] = None) -> ProviderStatusLoopConfig:
    args = _build_parser().parse_args(argv)
    _validate_config_args(args)
    grok_billing_http_method = str(args.grok_billing_http_method).strip().upper()

    resolved_grok_auth_file, resolved_grok_auth_file_source = (
        _resolve_grok_sidecar_auth_file(args.grok_oidc_auth_file)
    )
    resolved_codex_auth_file, resolved_codex_auth_file_source = (
        _resolve_codex_sidecar_auth_file(args.codex_auth_file)
    )
    resolved_xai_oauth_auth_file, resolved_xai_oauth_auth_file_source = (
        _resolve_xai_oauth_sidecar_auth_file(args.xai_oauth_auth_file)
    )

    return ProviderStatusLoopConfig(
        apply=args.apply,
        dsn=args.dsn,
        environment=args.environment,
        interval_seconds=args.interval_seconds,
        timeout=args.timeout,
        ping_count=args.ping_count,
        ping_timeout=args.ping_timeout,
        skip_icmp=args.skip_icmp,
        once=args.once,
        setup_schema=args.setup_schema,
        db_lock_timeout_ms=args.db_lock_timeout_ms,
        db_statement_timeout_ms=args.db_statement_timeout_ms,
        schema_dsn=args.schema_dsn,
        require_pgbouncer=args.require_pgbouncer,
        grok_oidc_refresh_enabled=args.grok_oidc_refresh_enabled,
        grok_oidc_auth_file=resolved_grok_auth_file,
        grok_oidc_auth_file_source=resolved_grok_auth_file_source,
        grok_oidc_lock_file=args.grok_oidc_lock_file,
        grok_oidc_refresh_interval_seconds=args.grok_oidc_refresh_interval_seconds,
        grok_oidc_refresh_buffer_seconds=args.grok_oidc_refresh_buffer_seconds,
        grok_oidc_force_refresh=args.grok_oidc_force_refresh,
        grok_oidc_http_timeout_seconds=args.grok_oidc_http_timeout_seconds,
        codex_oauth_refresh_enabled=args.codex_oauth_refresh_enabled,
        codex_auth_file=resolved_codex_auth_file,
        codex_auth_file_source=resolved_codex_auth_file_source,
        codex_lock_file=args.codex_lock_file,
        codex_refresh_interval_seconds=args.codex_refresh_interval_seconds,
        codex_refresh_buffer_seconds=args.codex_refresh_buffer_seconds,
        codex_force_refresh=args.codex_force_refresh,
        codex_http_timeout_seconds=args.codex_http_timeout_seconds,
        xai_oauth_refresh_enabled=args.xai_oauth_refresh_enabled,
        xai_oauth_auth_file=resolved_xai_oauth_auth_file,
        xai_oauth_auth_file_source=resolved_xai_oauth_auth_file_source,
        xai_oauth_lock_file=args.xai_oauth_lock_file,
        xai_oauth_scope=str(args.xai_oauth_scope).strip(),
        xai_oauth_refresh_interval_seconds=args.xai_oauth_refresh_interval_seconds,
        xai_oauth_refresh_buffer_seconds=args.xai_oauth_refresh_buffer_seconds,
        xai_oauth_force_refresh=args.xai_oauth_force_refresh,
        xai_oauth_http_timeout_seconds=args.xai_oauth_http_timeout_seconds,
        grok_billing_poll_enabled=args.grok_billing_poll_enabled,
        grok_billing_poll_interval_seconds=args.grok_billing_poll_interval_seconds,
        grok_billing_poll_http_timeout_seconds=args.grok_billing_poll_http_timeout_seconds,
        grok_billing_url=args.grok_billing_url,
        grok_billing_client_version=args.grok_billing_client_version,
        grok_billing_client_identifier=args.grok_billing_client_identifier,
        grok_billing_xai_token_auth=args.grok_billing_xai_token_auth,
        grok_billing_model=args.grok_billing_model,
        grok_billing_http_method=grok_billing_http_method,
        grok_billing_include_model_override=args.grok_billing_include_model_override,
        grok_billing_poll_max_attempts=args.grok_billing_poll_max_attempts,
        grok_billing_poll_retry_backoff_seconds=(
            args.grok_billing_poll_retry_backoff_seconds
        ),
        codex_reset_credit_poll_enabled=args.codex_reset_credit_poll_enabled,
        codex_reset_credit_poll_interval_seconds=(
            args.codex_reset_credit_poll_interval_seconds
        ),
        codex_reset_credit_poll_http_timeout_seconds=(
            args.codex_reset_credit_poll_http_timeout_seconds
        ),
        codex_usage_url=args.codex_usage_url,
        codex_reset_credit_poll_max_attempts=args.codex_reset_credit_poll_max_attempts,
        codex_reset_credit_poll_retry_backoff_seconds=(
            args.codex_reset_credit_poll_retry_backoff_seconds
        ),
        observability_anomaly_scan_enabled=(
            args.observability_anomaly_scan_enabled
        ),
        observability_anomaly_scan_interval_seconds=(
            args.observability_anomaly_scan_interval_seconds
        ),
        observability_anomaly_scan_lookback_hours=(
            args.observability_anomaly_scan_lookback_hours
        ),
        observability_anomaly_scan_error_log_dir=(
            args.observability_anomaly_scan_error_log_dir
        ),
    )


def _dsn_args(config: ProviderStatusLoopConfig) -> argparse.Namespace:
    return argparse.Namespace(
        dsn=config.dsn,
        pg_host=None,
        pg_port=None,
        pg_database=None,
        pg_user=None,
        pg_password=None,
        pg_sslmode=None,
    )


def _resolve_dsn(config: ProviderStatusLoopConfig) -> str:
    dsn = probes._build_dsn(_dsn_args(config))
    if not dsn:
        raise RuntimeError("No database DSN found. Set AAWM_DB_* or AAWM_PROVIDER_STATUS_DSN.")
    return dsn


def _resolve_schema_dsn(config: ProviderStatusLoopConfig) -> str:
    if config.schema_dsn:
        return probes._append_dsn_query_params(
            config.schema_dsn,
            {"application_name": probes._provider_status_db_application_name()},
        )
    if config.dsn:
        return probes._append_dsn_query_params(
            config.dsn,
            {"application_name": probes._provider_status_db_application_name()},
        )
    raise RuntimeError(
        "Provider-status schema setup requires AAWM_PROVIDER_STATUS_SCHEMA_DSN, "
        "AAWM_DIRECT_DATABASE_URL, or explicit --dsn. Steady-state PgBouncer "
        "DSNs are not used for schema setup by default."
    )


def _dsn_targets_pgbouncer(dsn: str) -> bool:
    try:
        parsed = urlsplit(dsn)
    except ValueError:
        return False
    return parsed.hostname == "pgbouncer" and parsed.port == 6432


def validate_runtime_guardrails(config: ProviderStatusLoopConfig) -> None:
    if config.setup_schema:
        _resolve_schema_dsn(config)
    if not config.apply or not config.require_pgbouncer:
        return
    dsn = _resolve_dsn(config)
    if not _dsn_targets_pgbouncer(dsn):
        raise RuntimeError(
            "Provider-status steady-state writes require "
            "pgbouncer:6432 when AAWM_PROVIDER_STATUS_REQUIRE_PGBOUNCER=1"
        )


def setup_schema_once(config: ProviderStatusLoopConfig) -> Dict[str, Any]:
    started = time.perf_counter()
    dsn = _resolve_schema_dsn(config)
    try:
        probes.setup_schema(
            dsn,
            lock_timeout_ms=config.db_lock_timeout_ms,
            statement_timeout_ms=config.db_statement_timeout_ms,
        )
    except probes.ProviderStatusDatabaseWriteSkipped as exc:
        return {
            "event": "provider_status_observations_schema_skipped",
            "observed_at": _utc_timestamp(),
            "environment": config.environment,
            "error_class": exc.error_class,
            "error_message": str(exc),
            "duration_ms": round((time.perf_counter() - started) * 1000, 3),
        }
    return {
        "event": "provider_status_observations_schema_ready",
        "observed_at": _utc_timestamp(),
        "environment": config.environment,
        "duration_ms": round((time.perf_counter() - started) * 1000, 3),
    }


def _bounded_summary_field(value: Any, *, limit: int = PROVIDER_FAILURE_FIELD_LIMIT) -> Optional[str]:
    if value is None:
        return None
    text = re.sub(r"\s+", " ", str(value)).strip()
    if not text:
        return None
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def _redacted_failure_message(value: Any) -> Optional[str]:
    text = _bounded_summary_field(value, limit=4096)
    if text is None:
        return None
    text = PROVIDER_FAILURE_SECRET_RE.sub("REDACTED", text)
    return _bounded_summary_field(text, limit=PROVIDER_FAILURE_MESSAGE_LIMIT)


def _redacted_summary_field(value: Any, *, limit: int = PROVIDER_FAILURE_FIELD_LIMIT) -> Optional[str]:
    text = _bounded_summary_field(value, limit=limit)
    if text is None:
        return None
    return PROVIDER_FAILURE_SECRET_RE.sub("REDACTED", text)


def _parse_sidecar_timestamp(value: Any) -> Optional[datetime]:
    if isinstance(value, datetime):
        return value
    if not isinstance(value, str) or not value.strip():
        return None
    normalized = value.strip()
    if normalized.endswith("Z"):
        normalized = normalized[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(normalized)
    except ValueError:
        return None


def _provider_auth_status_from_event(event: Mapping[str, Any]) -> str:
    if event.get("error_class"):
        return "failed"
    if event.get("refreshed"):
        return "refreshed"
    if event.get("skipped"):
        return "skipped"
    if event.get("attempted"):
        return "attempted"
    return "not_applicable"


def _build_grok_oidc_auth_observation(
    config: ProviderStatusLoopConfig,
    event: Mapping[str, Any],
) -> Dict[str, Any]:
    observed_at = _parse_sidecar_timestamp(event.get("observed_at")) or datetime.now(
        timezone.utc
    )
    expires_at = _parse_sidecar_timestamp(event.get("expires_at"))
    status = _provider_auth_status_from_event(event)
    successful_validation = status in {"refreshed", "skipped"} and not event.get(
        "error_class"
    )
    auth_file = event.get("auth_file") or config.grok_oidc_auth_file
    metadata = {
        "auth_file_hash_algorithm": "sha256",
        "auth_file_source": config.grok_oidc_auth_file_source,
        "refresh_buffer_seconds": config.grok_oidc_refresh_buffer_seconds,
        "refresh_interval_seconds": config.grok_oidc_refresh_interval_seconds,
        "force_refresh": config.grok_oidc_force_refresh,
        "raw_status_flags": {
            "attempted": bool(event.get("attempted")),
            "refreshed": bool(event.get("refreshed")),
            "skipped": bool(event.get("skipped")),
        },
    }
    return {
        "observed_at": observed_at,
        "environment": event.get("environment") or config.environment,
        "provider": "xai",
        "auth_family": "grok_oidc",
        "credential_scope": _redacted_summary_field(
            event.get("scope"),
            limit=512,
        ),
        "auth_file_hash": probes.auth_file_identity_hash(auth_file),
        "status": status,
        "attempted": bool(event.get("attempted")),
        "refreshed": bool(event.get("refreshed")),
        "skipped": bool(event.get("skipped")),
        "expires_at": expires_at,
        "last_success_at": observed_at if successful_validation else None,
        "source_task": "grok_oidc_refresh",
        "error_class": _redacted_summary_field(event.get("error_class")),
        "error_message": _redacted_failure_message(event.get("error_message")),
        "metadata": metadata,
    }


def _persist_grok_oidc_auth_observation(
    config: ProviderStatusLoopConfig,
    event: Mapping[str, Any],
) -> tuple[bool, int, Optional[str], Optional[str]]:
    if not config.apply:
        return False, 0, None, "apply_disabled"

    observation = _build_grok_oidc_auth_observation(config, event)
    dsn = _resolve_dsn(config)
    try:
        inserted_count = probes.insert_provider_auth_observations(
            dsn,
            [observation],
            lock_timeout_ms=config.db_lock_timeout_ms,
            statement_timeout_ms=config.db_statement_timeout_ms,
        )
    except probes.ProviderStatusDatabaseWriteSkipped as exc:
        return False, 0, exc.error_class, _redacted_failure_message(str(exc))
    except Exception as exc:
        return False, 0, exc.__class__.__name__, _redacted_failure_message(str(exc))
    return True, inserted_count, None, None



def _build_codex_auth_observation(
    config: ProviderStatusLoopConfig,
    event: Mapping[str, Any],
) -> Dict[str, Any]:
    observed_at = _parse_sidecar_timestamp(event.get("observed_at")) or datetime.now(
        timezone.utc
    )
    expires_at = _parse_sidecar_timestamp(event.get("expires_at"))
    status = _provider_auth_status_from_event(event)
    successful_validation = status in {"refreshed", "skipped"} and not event.get(
        "error_class"
    )
    auth_file = event.get("auth_file") or config.codex_auth_file
    metadata = {
        "auth_file_hash_algorithm": "sha256",
        "auth_file_source": config.codex_auth_file_source,
        "refresh_buffer_seconds": config.codex_refresh_buffer_seconds,
        "refresh_interval_seconds": config.codex_refresh_interval_seconds,
        "force_refresh": config.codex_force_refresh,
        "raw_status_flags": {
            "attempted": bool(event.get("attempted")),
            "refreshed": bool(event.get("refreshed")),
            "skipped": bool(event.get("skipped")),
        },
    }
    return {
        "observed_at": observed_at,
        "environment": event.get("environment") or config.environment,
        "provider": "openai",
        "auth_family": "codex_oauth",
        "credential_scope": _redacted_summary_field(
            event.get("account_id"),
            limit=512,
        ),
        "auth_file_hash": probes.auth_file_identity_hash(auth_file),
        "status": status,
        "attempted": bool(event.get("attempted")),
        "refreshed": bool(event.get("refreshed")),
        "skipped": bool(event.get("skipped")),
        "expires_at": expires_at,
        "last_success_at": observed_at if successful_validation else None,
        "source_task": "codex_oauth_refresh",
        "error_class": _redacted_summary_field(event.get("error_class")),
        "error_message": _redacted_failure_message(event.get("error_message")),
        "metadata": metadata,
    }


def _persist_codex_auth_observation(
    config: ProviderStatusLoopConfig,
    event: Mapping[str, Any],
) -> tuple[bool, int, Optional[str], Optional[str]]:
    if not config.apply:
        return False, 0, None, "apply_disabled"

    observation = _build_codex_auth_observation(config, event)
    dsn = _resolve_dsn(config)
    try:
        inserted_count = probes.insert_provider_auth_observations(
            dsn,
            [observation],
            lock_timeout_ms=config.db_lock_timeout_ms,
            statement_timeout_ms=config.db_statement_timeout_ms,
        )
    except probes.ProviderStatusDatabaseWriteSkipped as exc:
        return False, 0, exc.error_class, _redacted_failure_message(str(exc))
    except Exception as exc:
        return False, 0, exc.__class__.__name__, _redacted_failure_message(str(exc))
    return True, inserted_count, None, None


def _build_xai_oauth_auth_observation(
    config: ProviderStatusLoopConfig,
    event: Mapping[str, Any],
) -> Dict[str, Any]:
    observed_at = _parse_sidecar_timestamp(event.get("observed_at")) or datetime.now(
        timezone.utc
    )
    expires_at = _parse_sidecar_timestamp(event.get("expires_at"))
    status = _provider_auth_status_from_event(event)
    successful_validation = status in {"refreshed", "skipped"} and not event.get(
        "error_class"
    )
    auth_file = event.get("auth_file") or config.xai_oauth_auth_file
    metadata = {
        "auth_file_hash_algorithm": "sha256",
        "auth_file_source": config.xai_oauth_auth_file_source,
        "refresh_buffer_seconds": config.xai_oauth_refresh_buffer_seconds,
        "refresh_interval_seconds": config.xai_oauth_refresh_interval_seconds,
        "force_refresh": config.xai_oauth_force_refresh,
        "raw_status_flags": {
            "attempted": bool(event.get("attempted")),
            "refreshed": bool(event.get("refreshed")),
            "skipped": bool(event.get("skipped")),
        },
    }
    return {
        "observed_at": observed_at,
        "environment": event.get("environment") or config.environment,
        "provider": "xai",
        "auth_family": "xai_oauth",
        "credential_scope": _redacted_summary_field(
            event.get("scope") or config.xai_oauth_scope,
            limit=512,
        ),
        "auth_file_hash": probes.auth_file_identity_hash(auth_file),
        "status": status,
        "attempted": bool(event.get("attempted")),
        "refreshed": bool(event.get("refreshed")),
        "skipped": bool(event.get("skipped")),
        "expires_at": expires_at,
        "last_success_at": observed_at if successful_validation else None,
        "source_task": "xai_oauth_refresh",
        "error_class": _redacted_summary_field(event.get("error_class")),
        "error_message": _redacted_failure_message(event.get("error_message")),
        "metadata": metadata,
    }


def _persist_xai_oauth_auth_observation(
    config: ProviderStatusLoopConfig,
    event: Mapping[str, Any],
) -> tuple[bool, int, Optional[str], Optional[str]]:
    if not config.apply:
        return False, 0, None, "apply_disabled"

    observation = _build_xai_oauth_auth_observation(config, event)
    dsn = _resolve_dsn(config)
    try:
        inserted_count = probes.insert_provider_auth_observations(
            dsn,
            [observation],
            lock_timeout_ms=config.db_lock_timeout_ms,
            statement_timeout_ms=config.db_statement_timeout_ms,
        )
    except probes.ProviderStatusDatabaseWriteSkipped as exc:
        return False, 0, exc.error_class, _redacted_failure_message(str(exc))
    except Exception as exc:
        return False, 0, exc.__class__.__name__, _redacted_failure_message(str(exc))
    return True, inserted_count, None, None


def _provider_failure_summaries(rows: Sequence[Dict[str, Any]]) -> tuple[list[Dict[str, Any]], int]:
    failed_rows = [row for row in rows if not row.get("success")]
    summaries: list[Dict[str, Any]] = []
    for row in failed_rows[:PROVIDER_FAILURE_SUMMARY_LIMIT]:
        summaries.append(
            {
                "provider": _redacted_summary_field(row.get("provider")),
                "endpoint_key": _redacted_summary_field(row.get("endpoint_key")),
                "probe_type": _redacted_summary_field(row.get("probe_type")),
                "error_class": _redacted_summary_field(row.get("error_class")),
                "error_message": _redacted_failure_message(row.get("error_message")),
            }
        )
    return summaries, max(0, len(failed_rows) - len(summaries))


def run_cycle(config: ProviderStatusLoopConfig) -> Dict[str, Any]:
    started = time.perf_counter()
    rows = probes.collect_observations(
        probes.DEFAULT_ENDPOINTS,
        environment=config.environment,
        timeout=config.timeout,
        ping_count=config.ping_count,
        ping_timeout=config.ping_timeout,
        skip_icmp=config.skip_icmp,
    )
    inserted = False
    skipped = False
    skip_reason: Optional[str] = None
    skip_error_class: Optional[str] = None
    if config.apply:
        dsn = _resolve_dsn(config)
        try:
            probes.insert_observations(
                dsn,
                rows,
                lock_timeout_ms=config.db_lock_timeout_ms,
                statement_timeout_ms=config.db_statement_timeout_ms,
            )
        except probes.ProviderStatusDatabaseWriteSkipped as exc:
            skipped = True
            skip_reason = str(exc)
            skip_error_class = exc.error_class
        else:
            inserted = True

    elapsed_ms = round((time.perf_counter() - started) * 1000, 3)
    successes = sum(1 for row in rows if row.get("success"))
    failures = len(rows) - successes
    summary = {
        "event": "provider_status_observations_cycle",
        "observed_at": _utc_timestamp(),
        "apply": config.apply,
        "inserted": inserted,
        "skipped": skipped,
        "skip_error_class": skip_error_class,
        "skip_reason": skip_reason,
        "environment": config.environment,
        "row_count": len(rows),
        "success_count": successes,
        "failure_count": failures,
        "duration_ms": elapsed_ms,
    }
    if failures:
        failure_summaries, omitted_count = _provider_failure_summaries(rows)
        summary["failure_summaries"] = failure_summaries
        summary["failure_summaries_omitted_count"] = omitted_count
    return summary



CODEX_RESET_CREDIT_RETRYABLE_HTTP_STATUS_CODES = {408, 425, 500, 502, 503, 504}
CODEX_RESET_CREDIT_NON_RETRYABLE_HTTP_STATUS_CODES = {400, 401, 403, 404, 405, 409, 422, 429}
CODEX_RESET_CREDIT_RETRYABLE_ERROR_HINTS = (
    "operation was cancelled",
    "timeout expired",
)


def _codex_reset_credit_poll_sleep(seconds: float) -> None:
    if seconds <= 0:
        return
    CODEX_RESET_CREDIT_POLL_SLEEP_FN(seconds)


def _codex_reset_credit_poll_backoff_seconds(
    config: ProviderStatusLoopConfig,
    *,
    retry_count: int,
) -> float:
    if config.codex_reset_credit_poll_retry_backoff_seconds <= 0:
        return 0.0
    return config.codex_reset_credit_poll_retry_backoff_seconds * retry_count


def _codex_reset_credit_poll_failure_message(
    *,
    status_code: Optional[int],
    error_hint: Optional[str],
    fallback_message: str,
) -> str:
    message = "Codex reset-credit poll failed"
    if status_code is not None:
        message += f" with HTTP {status_code}"
    if error_hint:
        message += f" ({error_hint})"
    elif fallback_message:
        message += f" ({fallback_message})"
    return message + "."


def _load_codex_reset_credit_auth_context(auth_file: str) -> Dict[str, Any]:
    auth_data = codex_oauth_refresh._read_auth_data(Path(auth_file).expanduser())
    token_data = codex_oauth_refresh._get_token_data(auth_data)
    access_token = codex_oauth_refresh._clean_string(token_data.get("access_token"))
    if access_token is None:
        raise ValueError(
            "Codex OAuth auth file does not contain a usable access token for reset-credit poll."
        )
    account_id = codex_oauth_refresh._extract_account_id(token_data)
    return {
        "access_token": access_token,
        "account_id": account_id,
    }


def _build_codex_reset_credit_request_headers(
    auth_context: Mapping[str, Any],
) -> Dict[str, str]:
    headers = {
        "accept": "application/json",
        "authorization": f"Bearer {auth_context['access_token']}",
    }
    account_id = auth_context.get("account_id")
    if isinstance(account_id, str) and account_id.strip():
        headers["ChatGPT-Account-Id"] = account_id.strip()
    return headers


def _json_safe_codex_reset_credit_value(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _resolve_codex_reset_credit_poll_url(config: ProviderStatusLoopConfig) -> str:
    configured = str(config.codex_usage_url).strip()
    if not configured:
        return probes.DEFAULT_CODEX_RESET_CREDIT_DETAIL_URL
    legacy = getattr(probes, "LEGACY_CODEX_WHAM_USAGE_URL", "https://chatgpt.com/backend-api/wham/usage")
    if configured.rstrip("/") == legacy.rstrip("/"):
        return probes.DEFAULT_CODEX_RESET_CREDIT_DETAIL_URL
    return configured


def _parse_codex_reset_credit_available_count(response_body: Mapping[str, Any]) -> int:
    """Aggregate available count for poll events; prefers per-credit detail payloads."""
    try:
        credits = _parse_codex_reset_credit_detail_credits(response_body)
    except ValueError:
        credits = []
    if credits:
        return sum(1 for credit in credits if credit.get("status") == "available")
    legacy = response_body.get("rate_limit_reset_credits")
    if isinstance(legacy, dict):
        available = legacy.get("available_count")
        if isinstance(available, bool):
            raise ValueError("Codex reset-credit payload available_count was not an integer.")
        if isinstance(available, int):
            return available
        if isinstance(available, float) and available.is_integer():
            return int(available)
        if isinstance(available, str) and available.strip().isdigit():
            return int(available.strip())
    camel_credits = response_body.get("rateLimitResetCredits")
    if isinstance(camel_credits, dict):
        available = camel_credits.get("availableCount")
        if isinstance(available, bool):
            raise ValueError("Codex reset-credit payload availableCount was not an integer.")
        if isinstance(available, int):
            return available
        if isinstance(available, float) and available.is_integer():
            return int(available)
        if isinstance(available, str) and available.strip().isdigit():
            return int(available.strip())
    if isinstance(response_body.get("credits"), list):
        return 0
    raise ValueError(
        "Codex reset-credit payload did not include credits[] or rate_limit_reset_credits.available_count."
    )


def _parse_codex_reset_credit_field(
    entry: Mapping[str, Any],
    *keys: str,
) -> Any:
    for key in keys:
        if key in entry:
            return entry.get(key)
    return None


def _parse_codex_reset_credit_credit_entry(
    entry: Mapping[str, Any],
) -> Dict[str, Any]:
    provider_credit_id = _parse_codex_reset_credit_field(entry, "id", "credit_id", "creditId")
    if provider_credit_id is not None:
        provider_credit_id = str(provider_credit_id).strip() or None
    granted_at = probes._normalize_provider_credit_timestamp(
        _parse_codex_reset_credit_field(entry, "granted_at", "grantedAt")
    )
    expires_at = probes._normalize_provider_credit_timestamp(
        _parse_codex_reset_credit_field(entry, "expires_at", "expiresAt")
    )
    redeem_started_at = probes._normalize_provider_credit_timestamp(
        _parse_codex_reset_credit_field(entry, "redeem_started_at", "redeemStartedAt")
    )
    redeemed_at = probes._normalize_provider_credit_timestamp(
        _parse_codex_reset_credit_field(entry, "redeemed_at", "redeemedAt")
    )
    reset_type = _parse_codex_reset_credit_field(entry, "reset_type", "resetType")
    if reset_type is not None:
        reset_type = str(reset_type).strip() or None
    provider_status = _parse_codex_reset_credit_field(entry, "status")
    if provider_status is not None:
        provider_status = str(provider_status).strip().lower() or None
    status = _normalize_codex_reset_credit_status(
        provider_status,
        redeemed_at=redeemed_at,
        redeem_started_at=redeem_started_at,
        expires_at=expires_at,
        observed_at=None,
    )
    return {
        "provider_credit_id": provider_credit_id,
        "granted_at": granted_at,
        "expires_at": expires_at,
        "redeem_started_at": redeem_started_at,
        "redeemed_at": redeemed_at,
        "reset_type": reset_type,
        "status": status,
        "provider_status": provider_status,
    }


def _parse_codex_reset_credit_detail_credits(
    response_body: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    credits_value = response_body.get("credits")
    if credits_value is None:
        raise ValueError("Codex reset-credit detail payload did not include credits[].")
    if not isinstance(credits_value, list):
        raise ValueError("Codex reset-credit detail payload credits was not a list.")
    parsed: List[Dict[str, Any]] = []
    for item in credits_value:
        if not isinstance(item, dict):
            continue
        parsed.append(_parse_codex_reset_credit_credit_entry(item))
    return parsed


def _normalize_codex_reset_credit_status(
    provider_status: Optional[str],
    *,
    redeemed_at: Optional[datetime],
    redeem_started_at: Optional[datetime],
    expires_at: Optional[datetime],
    observed_at: Optional[datetime],
) -> str:
    if redeemed_at is not None:
        return "used"
    if redeem_started_at is not None:
        return "used"
    if provider_status in {"used", "redeemed", "consumed", "expired"}:
        return "used" if provider_status != "expired" else "expired"
    if (
        observed_at is not None
        and expires_at is not None
        and observed_at > expires_at
    ):
        return "expired"
    if provider_status in {"available", "active", "unused"}:
        return "available"
    if provider_status:
        return provider_status
    return "available"


def _sanitize_codex_reset_credit_raw_fields(entry: Mapping[str, Any]) -> Dict[str, Any]:
    sanitized: Dict[str, Any] = {}
    for key, value in entry.items():
        if key not in CODEX_RESET_CREDIT_SANITIZED_RAW_FIELD_KEYS:
            continue
        sanitized[key] = _json_safe_codex_reset_credit_value(value)
    return sanitized


def _parse_codex_reset_credit_expires_at(
    response_body: Mapping[str, Any],
) -> Optional[datetime]:
    try:
        credits = _parse_codex_reset_credit_detail_credits(response_body)
    except ValueError:
        credits = []
    if credits:
        expiries = [credit["expires_at"] for credit in credits if credit.get("expires_at")]
        return max(expiries) if expiries else None
    for credits_key, expires_key in (
        ("rate_limit_reset_credits", "expires_at"),
        ("rateLimitResetCredits", "expiresAt"),
    ):
        credits = response_body.get(credits_key)
        if not isinstance(credits, dict):
            continue
        if expires_key not in credits:
            continue
        value = credits.get(expires_key)
        return probes._normalize_provider_credit_timestamp(value)
    return None


def _build_codex_reset_credit_raw_provider_fields(
    response_body: Mapping[str, Any],
    *,
    credit_entry: Optional[Mapping[str, Any]] = None,
    available_count: Optional[int] = None,
) -> Dict[str, Any]:
    if credit_entry is not None:
        return {"credit": _sanitize_codex_reset_credit_raw_fields(credit_entry)}
    for credits_key, count_key, expires_key in (
        ("rate_limit_reset_credits", "available_count", "expires_at"),
        ("rateLimitResetCredits", "availableCount", "expiresAt"),
    ):
        credits = response_body.get(credits_key)
        if not isinstance(credits, dict) or count_key not in credits:
            continue
        fields: Dict[str, Any] = {
            credits_key: {
                count_key: available_count if available_count is not None else credits.get(count_key),
            }
        }
        if expires_key in credits:
            fields[credits_key][expires_key] = _json_safe_codex_reset_credit_value(
                credits.get(expires_key)
            )
        return fields
    if available_count is not None:
        return {
            "rate_limit_reset_credits": {
                "available_count": available_count,
            }
        }
    return {}


def _codex_reset_credit_poll_evidence(
    config: ProviderStatusLoopConfig,
    *,
    poll_url: str,
    status_code: int,
    attempt_count: int,
    retry_count: int,
    account_id: Any,
    detail_endpoint: bool,
    visible_credit_count: int,
) -> Dict[str, Any]:
    return {
        "signals": ["codex_reset_credit_poll"],
        "provider_fields": [
            "credits[].status",
            "credits[].reset_type",
            "credits[].granted_at",
            "credits[].expires_at",
            "credits[].redeem_started_at",
            "credits[].redeemed_at",
        ],
        "detail_endpoint": detail_endpoint,
        "poll_url": poll_url,
        "usage_url": poll_url,
        "status_code": status_code,
        "attempt_count": attempt_count,
        "retry_count": retry_count,
        "account_id_present": bool(account_id),
        "chatgpt_account_id_header_present": bool(account_id),
        "visible_credit_count": visible_credit_count,
    }


def _apply_codex_reset_credit_visible_source_url(
    observations: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    if not observations:
        return observations
    available_rows = [
        row for row in observations if row.get("status") == "available"
    ]
    if not available_rows:
        return observations
    newest = max(
        available_rows,
        key=lambda row: row.get("granted_at") or datetime.min.replace(tzinfo=timezone.utc),
    )
    updated: List[Dict[str, Any]] = []
    for row in observations:
        if row is newest and not row.get("source_url"):
            merged = dict(row)
            merged["source_url"] = DEFAULT_CODEX_RESET_CREDIT_LATEST_VISIBLE_SOURCE_URL
            updated.append(merged)
        else:
            updated.append(row)
    return updated


def _build_codex_reset_credit_observations(
    config: ProviderStatusLoopConfig,
    *,
    observed_at: datetime,
    response_body: Mapping[str, Any],
    auth_context: Mapping[str, Any],
    status_code: int,
    attempt_count: int,
    retry_count: int,
    poll_url: str,
) -> List[Dict[str, Any]]:
    account_id = auth_context.get("account_id")
    account_hash = probes.account_identity_hash(account_id)
    if account_hash is None:
        raise ValueError(
            "Codex reset-credit poll could not derive a stable hashed account identity."
        )
    detail_endpoint = True
    try:
        parsed_credits = _parse_codex_reset_credit_detail_credits(response_body)
    except ValueError:
        detail_endpoint = False
        parsed_credits = []

    observations: List[Dict[str, Any]] = []
    if parsed_credits:
        for parsed in parsed_credits:
            credit_identity = probes.derive_provider_credit_identity(
                account_hash=account_hash,
                credit_family=DEFAULT_CODEX_RESET_CREDIT_CREDIT_FAMILY,
                granted_at=parsed.get("granted_at"),
                expires_at=parsed.get("expires_at"),
                reset_type=parsed.get("reset_type"),
                provider_credit_id=parsed.get("provider_credit_id"),
            )
            status = _normalize_codex_reset_credit_status(
                parsed.get("provider_status"),
                redeemed_at=parsed.get("redeemed_at"),
                redeem_started_at=parsed.get("redeem_started_at"),
                expires_at=parsed.get("expires_at"),
                observed_at=observed_at,
            )
            available_count = 1 if status == "available" else 0
            raw_provider_fields = _build_codex_reset_credit_raw_provider_fields(
                response_body,
                credit_entry={
                    "id": parsed.get("provider_credit_id"),
                    "status": parsed.get("provider_status"),
                    "reset_type": parsed.get("reset_type"),
                    "granted_at": parsed.get("granted_at"),
                    "expires_at": parsed.get("expires_at"),
                    "redeem_started_at": parsed.get("redeem_started_at"),
                    "redeemed_at": parsed.get("redeemed_at"),
                },
            )
            evidence = _codex_reset_credit_poll_evidence(
                config,
                poll_url=poll_url,
                status_code=status_code,
                attempt_count=attempt_count,
                retry_count=retry_count,
                account_id=account_id,
                detail_endpoint=True,
                visible_credit_count=len(parsed_credits),
            )
            observations.append(
                {
                    "observed_at": observed_at,
                    "environment": config.environment,
                    "provider": "openai",
                    "account_hash": account_hash,
                    "credit_family": DEFAULT_CODEX_RESET_CREDIT_CREDIT_FAMILY,
                    "credit_type": DEFAULT_CODEX_RESET_CREDIT_CREDIT_TYPE,
                    "credit_identity": credit_identity,
                    "available_count": available_count,
                    "granted_at": parsed.get("granted_at"),
                    "expires_at": parsed.get("expires_at"),
                    "status": status,
                    "redeem_started_at": parsed.get("redeem_started_at"),
                    "redeemed_at": parsed.get("redeemed_at"),
                    "operator_annotation": None,
                    "source_url": None,
                    "raw_provider_fields": raw_provider_fields,
                    "evidence": evidence,
                    "source": DEFAULT_CODEX_RESET_CREDIT_SOURCE,
                }
            )
        observations = [
            probes.apply_provider_credit_seed_metadata(row) for row in observations
        ]
        return _apply_codex_reset_credit_visible_source_url(observations)
    if detail_endpoint:
        return []

    legacy = _build_codex_reset_credit_observation_legacy(
        config,
        observed_at=observed_at,
        response_body=response_body,
        auth_context=auth_context,
        status_code=status_code,
        attempt_count=attempt_count,
        retry_count=retry_count,
        poll_url=poll_url,
        detail_endpoint=False,
    )
    return [probes.apply_provider_credit_seed_metadata(legacy)]


def _build_codex_reset_credit_observation_legacy(
    config: ProviderStatusLoopConfig,
    *,
    observed_at: datetime,
    response_body: Mapping[str, Any],
    auth_context: Mapping[str, Any],
    status_code: int,
    attempt_count: int,
    retry_count: int,
    poll_url: str,
    detail_endpoint: bool,
) -> Dict[str, Any]:
    available_count = _parse_codex_reset_credit_available_count(response_body)
    expires_at = _parse_codex_reset_credit_expires_at(response_body)
    account_id = auth_context.get("account_id")
    account_hash = probes.account_identity_hash(account_id)
    if account_hash is None:
        raise ValueError(
            "Codex reset-credit poll could not derive a stable hashed account identity."
        )
    credit_identity = probes.derive_provider_credit_identity(
        account_hash=account_hash,
        credit_family=DEFAULT_CODEX_RESET_CREDIT_CREDIT_FAMILY,
        granted_at=None,
        expires_at=expires_at,
        reset_type="aggregate",
        provider_credit_id=None,
    )
    raw_provider_fields = _build_codex_reset_credit_raw_provider_fields(
        response_body,
        available_count=available_count,
    )
    evidence = _codex_reset_credit_poll_evidence(
        config,
        poll_url=poll_url,
        status_code=status_code,
        attempt_count=attempt_count,
        retry_count=retry_count,
        account_id=account_id,
        detail_endpoint=detail_endpoint,
        visible_credit_count=0,
    )
    return {
        "observed_at": observed_at,
        "environment": config.environment,
        "provider": "openai",
        "account_hash": account_hash,
        "credit_family": DEFAULT_CODEX_RESET_CREDIT_CREDIT_FAMILY,
        "credit_type": DEFAULT_CODEX_RESET_CREDIT_CREDIT_TYPE,
        "credit_identity": credit_identity,
        "available_count": available_count,
        "granted_at": None,
        "expires_at": expires_at,
        "status": "available" if available_count > 0 else "used",
        "redeem_started_at": None,
        "redeemed_at": None,
        "operator_annotation": None,
        "source_url": None,
        "raw_provider_fields": raw_provider_fields,
        "evidence": evidence,
        "source": DEFAULT_CODEX_RESET_CREDIT_SOURCE,
    }


def _build_codex_reset_credit_observation(
    config: ProviderStatusLoopConfig,
    *,
    observed_at: datetime,
    response_body: Mapping[str, Any],
    auth_context: Mapping[str, Any],
    status_code: int,
    attempt_count: int,
    retry_count: int,
) -> Dict[str, Any]:
    poll_url = _resolve_codex_reset_credit_poll_url(config)
    rows = _build_codex_reset_credit_observations(
        config,
        observed_at=observed_at,
        response_body=response_body,
        auth_context=auth_context,
        status_code=status_code,
        attempt_count=attempt_count,
        retry_count=retry_count,
        poll_url=poll_url,
    )
    if not rows:
        raise ValueError("Codex reset-credit poll produced no observations.")
    return rows[0]


def _synthesize_codex_reset_credit_lifecycle_observations(
    config: ProviderStatusLoopConfig,
    *,
    observed_at: datetime,
    account_hash: str,
    visible_identities: set[str],
    status_code: int,
    attempt_count: int,
    retry_count: int,
    poll_url: str,
) -> List[Dict[str, Any]]:
    dsn = _resolve_dsn(config)
    if not dsn:
        return []
    current_rows = probes.load_provider_credit_current_rows(
        dsn,
        environment=config.environment,
        provider="openai",
        account_hash=account_hash,
        credit_family=DEFAULT_CODEX_RESET_CREDIT_CREDIT_FAMILY,
        source=DEFAULT_CODEX_RESET_CREDIT_SOURCE,
        lock_timeout_ms=config.db_lock_timeout_ms,
        statement_timeout_ms=config.db_statement_timeout_ms,
    )
    synthesized: List[Dict[str, Any]] = []
    for current in current_rows:
        identity = str(current.get("credit_identity") or "").strip()
        if not identity or identity in visible_identities:
            continue
        if str(current.get("status") or "").lower() != "available":
            continue
        expires_at = current.get("expires_at")
        if isinstance(expires_at, str):
            expires_at = probes._normalize_provider_credit_timestamp(expires_at)
        if expires_at is not None and observed_at > expires_at:
            next_status = "expired"
        else:
            next_status = "used"
        evidence = dict(current.get("evidence") or {})
        evidence.update(
            {
                "lifecycle_inference": next_status,
                "lifecycle_reason": (
                    "credit_missing_before_expiry"
                    if next_status == "used"
                    else "credit_past_expiry"
                ),
                "detail_endpoint": True,
                "poll_url": poll_url,
                "status_code": status_code,
                "attempt_count": attempt_count,
                "retry_count": retry_count,
            }
        )
        synthesized.append(
            {
                "observed_at": observed_at,
                "environment": config.environment,
                "provider": "openai",
                "account_hash": account_hash,
                "credit_family": DEFAULT_CODEX_RESET_CREDIT_CREDIT_FAMILY,
                "credit_type": current.get("credit_type")
                or DEFAULT_CODEX_RESET_CREDIT_CREDIT_TYPE,
                "credit_identity": identity,
                "available_count": 0,
                "granted_at": current.get("granted_at"),
                "expires_at": expires_at,
                "status": next_status,
                "redeem_started_at": current.get("redeem_started_at"),
                "redeemed_at": current.get("redeemed_at"),
                "operator_annotation": current.get("operator_annotation"),
                "source_url": current.get("source_url"),
                "raw_provider_fields": current.get("raw_provider_fields") or {},
                "evidence": evidence,
                "source": DEFAULT_CODEX_RESET_CREDIT_SOURCE,
            }
        )
    return synthesized


def _build_codex_reset_credit_seed_observations(
    config: ProviderStatusLoopConfig,
    *,
    observed_at: datetime,
    account_hash: str,
    visible_identities: set[str],
    visible_credit_windows: Optional[
        set[tuple[Optional[datetime], Optional[datetime]]]
    ] = None,
    status_code: int,
    attempt_count: int,
    retry_count: int,
    poll_url: str,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for seed in probes.CODEX_RESET_CREDIT_SEED_METADATA:
        granted_at = probes._normalize_provider_credit_timestamp(seed.get("granted_at"))
        expires_at = probes._normalize_provider_credit_timestamp(seed.get("expires_at"))
        if granted_at is None:
            continue
        reset_type = str(seed.get("reset_type") or "codex_rate_limits").strip()
        credit_identity = probes.derive_provider_credit_identity(
            account_hash=account_hash,
            credit_family=DEFAULT_CODEX_RESET_CREDIT_CREDIT_FAMILY,
            granted_at=granted_at,
            expires_at=expires_at,
            reset_type=reset_type,
            provider_credit_id=None,
        )
        if credit_identity in visible_identities:
            continue
        if visible_credit_windows:
            if any(
                _codex_reset_credit_seed_matches_visible_window(
                    granted_at=granted_at,
                    expires_at=expires_at,
                    visible_granted_at=visible_granted_at,
                    visible_expires_at=visible_expires_at,
                )
                for visible_granted_at, visible_expires_at in visible_credit_windows
            ):
                continue
        status = "expired" if expires_at is not None and observed_at > expires_at else "used"
        raw_provider_fields = {
            "seed": {
                "granted_at": granted_at.isoformat().replace("+00:00", "Z"),
                "expires_at": (
                    expires_at.isoformat().replace("+00:00", "Z")
                    if expires_at is not None
                    else None
                ),
                "reset_type": reset_type,
                "operator_annotation": seed.get("operator_annotation"),
                "source_url": seed.get("source_url"),
            }
        }
        evidence = _codex_reset_credit_poll_evidence(
            config,
            poll_url=poll_url,
            status_code=status_code,
            attempt_count=attempt_count,
            retry_count=retry_count,
            account_id=True,
            detail_endpoint=True,
            visible_credit_count=len(visible_identities),
        )
        evidence.update(
            {
                "seed_backfill": True,
                "lifecycle_inference": status,
                "lifecycle_reason": (
                    "seed_credit_absent_before_expiry"
                    if status == "used"
                    else "seed_credit_past_expiry"
                ),
            }
        )
        rows.append(
            probes.apply_provider_credit_seed_metadata(
                {
                    "observed_at": observed_at,
                    "environment": config.environment,
                    "provider": "openai",
                    "account_hash": account_hash,
                    "credit_family": DEFAULT_CODEX_RESET_CREDIT_CREDIT_FAMILY,
                    "credit_type": DEFAULT_CODEX_RESET_CREDIT_CREDIT_TYPE,
                    "credit_identity": credit_identity,
                    "available_count": 0,
                    "granted_at": granted_at,
                    "expires_at": expires_at,
                    "status": status,
                    "redeem_started_at": None,
                    "redeemed_at": None,
                    "operator_annotation": seed.get("operator_annotation"),
                    "source_url": seed.get("source_url"),
                    "raw_provider_fields": raw_provider_fields,
                    "evidence": evidence,
                    "source": DEFAULT_CODEX_RESET_CREDIT_SOURCE,
                }
            )
        )
    return rows


def _codex_reset_credit_seed_matches_visible_window(
    *,
    granted_at: Optional[datetime],
    expires_at: Optional[datetime],
    visible_granted_at: Optional[datetime],
    visible_expires_at: Optional[datetime],
) -> bool:
    if granted_at is None or visible_granted_at is None:
        return False
    if abs((granted_at - visible_granted_at).total_seconds()) > 120:
        return False
    if expires_at is None or visible_expires_at is None:
        return True
    return abs((expires_at - visible_expires_at).total_seconds()) <= 120


def _codex_reset_credit_retryable_http_error(
    exc: urllib_error.HTTPError,
    *,
    error_hint: Optional[str],
) -> bool:
    status_code = exc.code
    if status_code in {401, 403, 429}:
        return False
    normalized_hint = (error_hint or "").strip().lower()
    if status_code == 400 and normalized_hint and any(
        hint in normalized_hint for hint in CODEX_RESET_CREDIT_RETRYABLE_ERROR_HINTS
    ):
        return True
    if status_code in CODEX_RESET_CREDIT_NON_RETRYABLE_HTTP_STATUS_CODES:
        return False
    if status_code in CODEX_RESET_CREDIT_RETRYABLE_HTTP_STATUS_CODES:
        return True
    return False


def _codex_reset_credit_retryable_url_error(exc: urllib_error.URLError) -> bool:
    reason = getattr(exc, "reason", None)
    if isinstance(reason, TimeoutError):
        return True
    if isinstance(reason, OSError) and getattr(reason, "errno", None) in {110, 111, 113}:
        return True
    message = str(reason or exc).strip().lower()
    return any(hint in message for hint in CODEX_RESET_CREDIT_RETRYABLE_ERROR_HINTS)


def _fetch_codex_reset_credit_payload(
    config: ProviderStatusLoopConfig,
) -> Dict[str, Any]:
    max_attempts = max(1, config.codex_reset_credit_poll_max_attempts)
    attempt_count = 0
    retry_count = 0
    last_status_code: Optional[int] = None
    last_error_hint: Optional[str] = None
    last_error_message = "Codex reset-credit poll failed."
    poll_url = _resolve_codex_reset_credit_poll_url(config)

    while attempt_count < max_attempts:
        attempt_count += 1
        try:
            auth_context = _load_codex_reset_credit_auth_context(config.codex_auth_file)
            request = urllib_request.Request(
                poll_url,
                headers=_build_codex_reset_credit_request_headers(auth_context),
                method="GET",
            )
            with urllib_request.urlopen(
                request,
                timeout=config.codex_reset_credit_poll_http_timeout_seconds,
            ) as response:
                status_code = getattr(response, "status", None) or response.getcode()
                response_body = response.read().decode("utf-8")
        except urllib_error.HTTPError as exc:
            last_status_code = exc.code
            last_error_hint = codex_oauth_refresh._extract_oauth_error_hint(
                exc.read().decode("utf-8", errors="replace")
            )
            last_error_message = _codex_reset_credit_poll_failure_message(
                status_code=last_status_code,
                error_hint=last_error_hint,
                fallback_message="",
            )
            if (
                attempt_count < max_attempts
                and _codex_reset_credit_retryable_http_error(
                    exc,
                    error_hint=last_error_hint,
                )
            ):
                retry_count += 1
                _codex_reset_credit_poll_sleep(
                    _codex_reset_credit_poll_backoff_seconds(
                        config,
                        retry_count=retry_count,
                    )
                )
                continue
            raise CodexResetCreditPollError(
                last_error_message,
                status_code=last_status_code,
                attempt_count=attempt_count,
                retry_count=retry_count,
            ) from exc
        except urllib_error.URLError as exc:
            last_status_code = None
            last_error_hint = None
            last_error_message = (
                "Codex reset-credit poll failed while contacting the reset-credit detail endpoint."
            )
            if attempt_count < max_attempts and _codex_reset_credit_retryable_url_error(exc):
                retry_count += 1
                _codex_reset_credit_poll_sleep(
                    _codex_reset_credit_poll_backoff_seconds(
                        config,
                        retry_count=retry_count,
                    )
                )
                continue
            raise CodexResetCreditPollError(
                last_error_message,
                status_code=last_status_code,
                attempt_count=attempt_count,
                retry_count=retry_count,
            ) from exc

        try:
            payload = json.loads(response_body)
        except json.JSONDecodeError as exc:
            raise CodexResetCreditPollError(
                "Codex reset-credit endpoint returned invalid JSON.",
                status_code=int(status_code),
                attempt_count=attempt_count,
                retry_count=retry_count,
            ) from exc
        if not isinstance(payload, dict):
            raise CodexResetCreditPollError(
                "Codex reset-credit endpoint returned a non-object payload.",
                status_code=int(status_code),
                attempt_count=attempt_count,
                retry_count=retry_count,
            )
        return {
            "status_code": int(status_code),
            "payload": payload,
            "auth_context": auth_context,
            "attempt_count": attempt_count,
            "retry_count": retry_count,
            "poll_url": poll_url,
        }

    raise CodexResetCreditPollError(
        last_error_message,
        status_code=last_status_code,
        attempt_count=attempt_count,
        retry_count=retry_count,
    )


def _persist_codex_reset_credit_observation(
    config: ProviderStatusLoopConfig,
    *,
    observed_at: datetime,
    response_body: Mapping[str, Any],
    auth_context: Mapping[str, Any],
    status_code: int,
    attempt_count: int,
    retry_count: int,
    poll_url: Optional[str] = None,
) -> tuple[int, int]:
    resolved_poll_url = poll_url or _resolve_codex_reset_credit_poll_url(config)
    observations = _build_codex_reset_credit_observations(
        config,
        observed_at=observed_at,
        response_body=response_body,
        auth_context=auth_context,
        status_code=status_code,
        attempt_count=attempt_count,
        retry_count=retry_count,
        poll_url=resolved_poll_url,
    )
    account_hash = (
        observations[0]["account_hash"]
        if observations
        else probes.account_identity_hash(auth_context.get("account_id"))
    )
    if account_hash is None:
        raise ValueError(
            "Codex reset-credit poll could not derive a stable hashed account identity."
        )
    visible_identities = {
        str(row.get("credit_identity") or "").strip()
        for row in observations
        if str(row.get("credit_identity") or "").strip()
    }
    visible_credit_windows = {
        (row.get("granted_at"), row.get("expires_at"))
        for row in observations
        if row.get("granted_at") is not None
    }
    seed_rows = _build_codex_reset_credit_seed_observations(
        config,
        observed_at=observed_at,
        account_hash=account_hash,
        visible_identities=visible_identities,
        visible_credit_windows=visible_credit_windows,
        status_code=status_code,
        attempt_count=attempt_count,
        retry_count=retry_count,
        poll_url=resolved_poll_url,
    )
    represented_identities = visible_identities | {
        str(row.get("credit_identity") or "").strip()
        for row in seed_rows
        if str(row.get("credit_identity") or "").strip()
    }
    lifecycle_rows = _synthesize_codex_reset_credit_lifecycle_observations(
        config,
        observed_at=observed_at,
        account_hash=account_hash,
        visible_identities=represented_identities,
        status_code=status_code,
        attempt_count=attempt_count,
        retry_count=retry_count,
        poll_url=resolved_poll_url,
    )
    rows = observations + seed_rows + lifecycle_rows
    dsn = _resolve_dsn(config)
    inserted_count = probes.insert_provider_credit_observations(
        dsn,
        rows,
        lock_timeout_ms=config.db_lock_timeout_ms,
        statement_timeout_ms=config.db_statement_timeout_ms,
    )
    return len(rows), inserted_count


def _run_codex_reset_credit_poll_task(
    config: ProviderStatusLoopConfig,
    state: SidecarTaskState,
    *,
    now_monotonic: float,
) -> Optional[Dict[str, Any]]:
    if not config.codex_reset_credit_poll_enabled:
        return None
    last_attempt = state.codex_reset_credit_last_attempt_monotonic
    if (
        last_attempt is not None
        and now_monotonic - last_attempt < config.codex_reset_credit_poll_interval_seconds
    ):
        return None

    state.codex_reset_credit_last_attempt_monotonic = now_monotonic
    observed_at = datetime.now(timezone.utc)
    summary: Dict[str, Any] = {
        "attempted": True,
        "persisted": False,
        "skipped": False,
        "auth_file": config.codex_auth_file,
        "resolved_auth_file": config.codex_auth_file,
        "auth_file_source": config.codex_auth_file_source,
        "usage_url": _resolve_codex_reset_credit_poll_url(config),
        "poll_url": _resolve_codex_reset_credit_poll_url(config),
        "available_count": None,
        "inserted_count": 0,
        "status_code": None,
        "attempt_count": 0,
        "retry_count": 0,
        "poll_max_attempts": max(1, config.codex_reset_credit_poll_max_attempts),
        "error_class": None,
        "error_message": None,
    }
    try:
        fetched = _fetch_codex_reset_credit_payload(config)
        summary["status_code"] = fetched["status_code"]
        summary["attempt_count"] = fetched.get("attempt_count", 1)
        summary["retry_count"] = fetched.get("retry_count", 0)
        available_count = _parse_codex_reset_credit_available_count(fetched["payload"])
        summary["available_count"] = available_count
        if config.apply:
            observation_count, inserted_count = _persist_codex_reset_credit_observation(
                config,
                observed_at=observed_at,
                response_body=fetched["payload"],
                auth_context=fetched["auth_context"],
                status_code=fetched["status_code"],
                attempt_count=summary["attempt_count"],
                retry_count=summary["retry_count"],
                poll_url=fetched.get("poll_url"),
            )
            summary["inserted_count"] = inserted_count
            summary["persisted"] = observation_count > 0 and inserted_count >= 0
        else:
            _build_codex_reset_credit_observations(
                config,
                observed_at=observed_at,
                response_body=fetched["payload"],
                auth_context=fetched["auth_context"],
                status_code=fetched["status_code"],
                attempt_count=summary["attempt_count"],
                retry_count=summary["retry_count"],
                poll_url=fetched.get("poll_url")
                or _resolve_codex_reset_credit_poll_url(config),
            )
            summary["persisted"] = False
    except Exception as exc:
        summary["error_class"] = exc.__class__.__name__
        summary["error_message"] = _redacted_failure_message(str(exc))
        if isinstance(exc, CodexResetCreditPollError):
            summary["status_code"] = exc.status_code
            summary["attempt_count"] = exc.attempt_count
            summary["retry_count"] = exc.retry_count
        elif summary["status_code"] is None:
            status_match = re.search(r"with HTTP (\d{3})", str(exc))
            if status_match is not None:
                summary["status_code"] = int(status_match.group(1))
        if summary["attempt_count"] == 0:
            summary["attempt_count"] = 1

    return {
        "event": "codex_reset_credit_poll",
        "observed_at": observed_at.isoformat().replace("+00:00", "Z"),
        "environment": config.environment,
        **summary,
    }


def _load_grok_billing_access_token(auth_file: str) -> str:
    return _load_grok_billing_auth_context(auth_file)["access_token"]


def _load_grok_billing_auth_context(auth_file: str) -> Dict[str, Any]:
    payload = grok_oidc_refresh._read_credential_payload(Path(auth_file).expanduser())
    scope = grok_oidc_refresh._resolve_scope(None)
    credential = grok_oidc_refresh._select_credential_record(payload, scope)
    for field_name in ("access_token", "key"):
        token = credential.get(field_name)
        if isinstance(token, str) and token.strip():
            return {
                "access_token": token.strip(),
                "identity_headers": _grok_billing_identity_headers(credential),
            }
    raise ValueError(
        "Grok OIDC auth file does not contain a usable access token for billing poll."
    )


def _grok_billing_identity_headers(
    credential: Mapping[str, Any],
) -> Dict[str, str]:
    headers: Dict[str, str] = {}
    missing_fields: list[str] = []
    for header_name, credential_field in GROK_BILLING_IDENTITY_HEADER_FIELDS:
        value = credential.get(credential_field)
        if isinstance(value, str) and value.strip():
            headers[header_name] = value.strip()
        elif credential_field not in missing_fields:
            missing_fields.append(credential_field)
    if missing_fields:
        joined_fields = ", ".join(sorted(missing_fields))
        raise ValueError(
            "Grok OIDC auth file does not contain required billing identity "
            f"fields: {joined_fields}."
        )
    return headers


def _build_grok_billing_request_headers(
    config: ProviderStatusLoopConfig,
    *,
    access_token: str,
    identity_headers: Optional[Mapping[str, str]] = None,
) -> Dict[str, str]:
    request_id = str(uuid.uuid4())
    headers = {
        "accept": "application/json",
        "authorization": f"Bearer {access_token}",
        "content-type": "application/json",
        "user-agent": f"grok/{config.grok_billing_client_version}",
        "x-grok-client-identifier": config.grok_billing_client_identifier,
        "x-grok-client-version": config.grok_billing_client_version,
        "x-grok-req-id": request_id,
        "x-request-id": request_id,
        "x-xai-token-auth": config.grok_billing_xai_token_auth,
    }
    model_override = str(config.grok_billing_model).strip()
    if config.grok_billing_include_model_override and model_override:
        headers["x-grok-model-override"] = model_override
    if identity_headers:
        for header_name, _credential_field in GROK_BILLING_IDENTITY_HEADER_FIELDS:
            value = identity_headers.get(header_name)
            if isinstance(value, str) and value.strip():
                headers[header_name] = value.strip()
    return headers


def _grok_billing_request_contract_summary(
    config: ProviderStatusLoopConfig,
    *,
    identity_headers: Optional[Mapping[str, str]] = None,
) -> Dict[str, Any]:
    parsed_url = urlsplit(config.grok_billing_url)
    query_keys = sorted(
        {
            key
            for key, _value in parse_qsl(
                parsed_url.query,
                keep_blank_values=True,
            )
            if key
        }
    )
    identity_header_names = [
        header_name
        for header_name, _credential_field in GROK_BILLING_IDENTITY_HEADER_FIELDS
        if isinstance((identity_headers or {}).get(header_name), str)
        and str((identity_headers or {}).get(header_name)).strip()
    ]
    request_headers = _build_grok_billing_request_headers(
        config,
        access_token="<redacted>",
        identity_headers={
            header_name: "<redacted>"
            for header_name in identity_header_names
        }
        if identity_header_names
        else None,
    )
    header_names = sorted(request_headers.keys())
    fingerprint_payload = {
        "billing_host": parsed_url.hostname,
        "billing_path": parsed_url.path or "/",
        "billing_query_keys": query_keys,
        "client_identifier": config.grok_billing_client_identifier,
        "client_version": config.grok_billing_client_version,
        "header_names": header_names,
        "http_client": "urllib",
        "include_model_override": config.grok_billing_include_model_override,
        "method": config.grok_billing_http_method,
        "model_override_configured": bool(
            config.grok_billing_include_model_override
            and str(config.grok_billing_model).strip()
        ),
        "x_xai_token_auth_configured": bool(config.grok_billing_xai_token_auth),
    }
    fingerprint = hashlib.sha256(
        json.dumps(fingerprint_payload, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()
    return {
        "http_client": "urllib",
        "request_method": config.grok_billing_http_method,
        "billing_host": parsed_url.hostname,
        "billing_path": parsed_url.path or "/",
        "billing_query_keys": query_keys,
        "billing_query_present": bool(parsed_url.query),
        "header_names": header_names,
        "include_model_override": config.grok_billing_include_model_override,
        "model_override_configured": bool(
            config.grok_billing_include_model_override
            and str(config.grok_billing_model).strip()
        ),
        "client_identifier": config.grok_billing_client_identifier,
        "client_version": config.grok_billing_client_version,
        "x_xai_token_auth_configured": bool(config.grok_billing_xai_token_auth),
        "resolved_auth_file": config.grok_oidc_auth_file,
        "auth_file_source": config.grok_oidc_auth_file_source,
        "poll_max_attempts": max(1, config.grok_billing_poll_max_attempts),
        "request_contract_fingerprint": fingerprint,
    }


def _grok_billing_poll_sleep(seconds: float) -> None:
    if seconds <= 0:
        return
    GROK_BILLING_POLL_SLEEP_FN(seconds)


def _grok_billing_http_error_hint(exc: urllib_error.HTTPError) -> Optional[str]:
    try:
        response_body = exc.read().decode("utf-8", errors="replace")
    except Exception:
        return None
    error_hint = grok_oidc_refresh._extract_oauth_error_hint(response_body)
    if error_hint:
        return error_hint
    try:
        payload = json.loads(response_body)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    code = payload.get("code")
    if isinstance(code, str) and code.strip():
        return grok_oidc_refresh._sanitize_error_message(code.strip())
    return None


def _grok_billing_retryable_http_error(
    exc: urllib_error.HTTPError,
    *,
    error_hint: Optional[str],
) -> bool:
    status_code = exc.code
    if status_code in {401, 403, 429}:
        return False
    normalized_hint = (error_hint or "").strip().lower()
    if status_code == 400 and normalized_hint and any(
        hint in normalized_hint for hint in GROK_BILLING_RETRYABLE_ERROR_HINTS
    ):
        return True
    if status_code in GROK_BILLING_NON_RETRYABLE_HTTP_STATUS_CODES:
        return False
    if status_code in GROK_BILLING_RETRYABLE_HTTP_STATUS_CODES:
        return True
    return False


def _grok_billing_retryable_url_error(exc: urllib_error.URLError) -> bool:
    reason = getattr(exc, "reason", None)
    if isinstance(reason, TimeoutError):
        return True
    if isinstance(reason, OSError) and getattr(reason, "errno", None) in {
        110,
        111,
        113,
    }:
        return True
    message = str(reason or exc).strip().lower()
    return any(hint in message for hint in GROK_BILLING_RETRYABLE_ERROR_HINTS)


def _grok_billing_poll_backoff_seconds(
    config: ProviderStatusLoopConfig,
    *,
    retry_count: int,
) -> float:
    if config.grok_billing_poll_retry_backoff_seconds <= 0:
        return 0.0
    return config.grok_billing_poll_retry_backoff_seconds * retry_count


def _grok_billing_poll_failure_message(
    *,
    status_code: Optional[int],
    error_hint: Optional[str],
    fallback_message: str,
) -> str:
    message = "Grok billing poll failed"
    if status_code is not None:
        message += f" with HTTP {status_code}"
    if error_hint:
        message += f" ({error_hint})"
    elif fallback_message:
        message += f" ({fallback_message})"
    return message + "."


def _fetch_grok_billing_payload(
    config: ProviderStatusLoopConfig,
) -> Dict[str, Any]:
    max_attempts = max(1, config.grok_billing_poll_max_attempts)
    attempt_count = 0
    retry_count = 0
    last_status_code: Optional[int] = None
    last_error_hint: Optional[str] = None
    last_error_message = "Grok billing poll failed."

    while attempt_count < max_attempts:
        attempt_count += 1
        try:
            auth_context = _load_grok_billing_auth_context(config.grok_oidc_auth_file)
            request = urllib_request.Request(
                config.grok_billing_url,
                headers=_build_grok_billing_request_headers(
                    config,
                    access_token=auth_context["access_token"],
                    identity_headers=auth_context["identity_headers"],
                ),
                method=config.grok_billing_http_method,
            )
            with urllib_request.urlopen(
                request,
                timeout=config.grok_billing_poll_http_timeout_seconds,
            ) as response:
                status_code = getattr(response, "status", None) or response.getcode()
                response_body = response.read().decode("utf-8")
        except urllib_error.HTTPError as exc:
            last_status_code = exc.code
            last_error_hint = _grok_billing_http_error_hint(exc)
            last_error_message = _grok_billing_poll_failure_message(
                status_code=last_status_code,
                error_hint=last_error_hint,
                fallback_message="",
            )
            if (
                attempt_count < max_attempts
                and _grok_billing_retryable_http_error(
                    exc,
                    error_hint=last_error_hint,
                )
            ):
                retry_count += 1
                _grok_billing_poll_sleep(
                    _grok_billing_poll_backoff_seconds(
                        config,
                        retry_count=retry_count,
                    )
                )
                continue
            raise GrokBillingPollError(
                last_error_message,
                status_code=last_status_code,
                attempt_count=attempt_count,
                retry_count=retry_count,
            ) from exc
        except urllib_error.URLError as exc:
            last_status_code = None
            last_error_hint = None
            last_error_message = (
                "Grok billing poll failed while contacting the billing endpoint."
            )
            if attempt_count < max_attempts and _grok_billing_retryable_url_error(exc):
                retry_count += 1
                _grok_billing_poll_sleep(
                    _grok_billing_poll_backoff_seconds(
                        config,
                        retry_count=retry_count,
                    )
                )
                continue
            raise GrokBillingPollError(
                last_error_message,
                status_code=last_status_code,
                attempt_count=attempt_count,
                retry_count=retry_count,
            ) from exc

        try:
            payload = json.loads(response_body)
        except json.JSONDecodeError as exc:
            raise GrokBillingPollError(
                "Grok billing endpoint returned invalid JSON.",
                status_code=int(status_code),
                attempt_count=attempt_count,
                retry_count=retry_count,
            ) from exc
        if not isinstance(payload, dict):
            raise GrokBillingPollError(
                "Grok billing endpoint returned a non-object payload.",
                status_code=int(status_code),
                attempt_count=attempt_count,
                retry_count=retry_count,
            )
        return {
            "status_code": int(status_code),
            "payload": payload,
            "attempt_count": attempt_count,
            "retry_count": retry_count,
        }

    raise GrokBillingPollError(
        last_error_message,
        status_code=last_status_code,
        attempt_count=attempt_count,
        retry_count=retry_count,
    )


def _safe_float(value: Any) -> Optional[float]:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _grok_billing_quota_value(value: Any) -> Optional[float]:
    if isinstance(value, dict):
        value = value.get("val")
    return _safe_float(value)


def _parse_grok_billing_timestamp(value: Any) -> Optional[datetime]:
    if not isinstance(value, str) or not value.strip():
        return None
    normalized = value.strip()
    if normalized.endswith("Z"):
        normalized = normalized[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _json_safe_grok_billing_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): _json_safe_grok_billing_value(nested)
            for key, nested in value.items()
            if str(key).lower()
            not in {"access_token", "authorization", "id_token", "refresh_token"}
        }
    if isinstance(value, list):
        return [_json_safe_grok_billing_value(item) for item in value[:50]]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _grok_billing_config(response_body: Dict[str, Any]) -> Dict[str, Any]:
    config = response_body.get("config")
    if isinstance(config, dict):
        return config
    return response_body


def _grok_billing_sidecar_request_contract_evidence(
    config: ProviderStatusLoopConfig,
    *,
    identity_headers: Optional[Mapping[str, str]] = None,
) -> Dict[str, Any]:
    summary = _grok_billing_request_contract_summary(
        config,
        identity_headers=identity_headers,
    )
    evidence: Dict[str, Any] = {
        "request_contract_fingerprint": summary["request_contract_fingerprint"],
        "request_contract_source": "grok_billing_sidecar_poll",
    }
    for summary_key, evidence_key in (
        ("http_client", "request_contract_http_client"),
        ("request_method", "request_contract_method"),
        ("billing_host", "request_contract_target_host"),
        ("billing_path", "request_contract_target_path"),
        ("billing_query_keys", "request_contract_query_keys"),
        ("header_names", "request_contract_header_names"),
        ("client_identifier", "request_contract_client_identifier"),
        ("client_version", "request_contract_client_version"),
    ):
        value = summary.get(summary_key)
        if value is not None and value != "" and value != []:
            evidence[evidence_key] = value
    for summary_key, evidence_key in (
        ("billing_query_present", "request_contract_query_present"),
        ("include_model_override", "request_contract_include_model_override"),
        ("model_override_configured", "request_contract_model_override_configured"),
        ("x_xai_token_auth_configured", "request_contract_x_xai_token_auth_configured"),
    ):
        value = summary.get(summary_key)
        if value is not None:
            evidence[evidence_key] = value
    user_agent = f"grok/{config.grok_billing_client_version}"
    if user_agent:
        evidence["request_contract_user_agent"] = user_agent
    return evidence


def _build_grok_billing_rate_limit_payload(
    config: ProviderStatusLoopConfig,
    *,
    observed_at: datetime,
    response_body: Dict[str, Any],
    identity_headers: Optional[Mapping[str, str]] = None,
) -> tuple[Any, ...]:
    billing_config = _grok_billing_config(response_body)
    billing_period_start_at = _parse_grok_billing_timestamp(
        billing_config.get("billingPeriodStart")
    )
    billing_period_end_at = _parse_grok_billing_timestamp(
        billing_config.get("billingPeriodEnd")
    )

    monthly_limit = _grok_billing_quota_value(billing_config.get("monthlyLimit"))
    used = _grok_billing_quota_value(billing_config.get("used"))
    if monthly_limit is not None and monthly_limit > 0 and used is not None and used >= 0:
        used_percentage = max(0.0, min(100.0, (used / monthly_limit) * 100.0))
        remaining_pct = int(max(0.0, min(100.0, 100.0 - used_percentage)) + 0.5)
        raw_provider_fields = {
            "monthlyLimit": _json_safe_grok_billing_value(
                billing_config.get("monthlyLimit")
            ),
            "used": _json_safe_grok_billing_value(billing_config.get("used")),
            "onDemandCap": _json_safe_grok_billing_value(
                billing_config.get("onDemandCap")
            ),
            "billingPeriodStart": billing_config.get("billingPeriodStart"),
            "billingPeriodEnd": billing_config.get("billingPeriodEnd"),
            "quota_unit": "grok_billing_used",
            "quota_unit_interpretation": "requests",
        }
        evidence = {
            "signals": ["grok_billing_payload"],
            "provider_fields": [
                "config.monthlyLimit.val",
                "config.used.val",
                "config.billingPeriodEnd",
            ],
            "rounding": "whole_remaining_percentage",
            "unit_note": (
                "Grok billing does not label used.val; observed tool traffic "
                "behaves request-like."
            ),
        }
        quota_key = "xai_grok_build_monthly_requests:requests"
        quota_type = "requests"
        quota_limit = monthly_limit
        quota_used = used
        quota_remaining = max(0.0, monthly_limit - used)
        remaining_pct_value: Optional[float] = float(remaining_pct)
    else:
        credit_usage_percent = _safe_float(billing_config.get("creditUsagePercent"))
        if credit_usage_percent is not None:
            used_percentage = max(0.0, min(100.0, credit_usage_percent))
            remaining_pct = max(0.0, min(100.0, 100.0 - used_percentage))
            raw_provider_fields = {
                "creditUsagePercent": _json_safe_grok_billing_value(
                    billing_config.get("creditUsagePercent")
                ),
                "productUsage": _json_safe_grok_billing_value(
                    billing_config.get("productUsage")
                ),
                "billingPeriodStart": billing_config.get("billingPeriodStart"),
                "billingPeriodEnd": billing_config.get("billingPeriodEnd"),
                "quota_unit": "grok_billing_credit_usage_percent",
                "quota_unit_interpretation": "percent_of_credit_quota",
            }
            evidence = {
                "signals": [
                    "grok_billing_payload",
                    "grok_billing_percentage_only",
                ],
                "provider_fields": [
                    "config.creditUsagePercent",
                    "config.productUsage",
                    "config.billingPeriodEnd",
                ],
                "rounding": "none",
                "unit_note": (
                    "Grok billing provided percentage-only credit usage; absolute "
                    "quota counts are intentionally left null."
                ),
            }
            quota_key = "xai_grok_build_monthly_credits:credits"
            quota_type = "credits"
            quota_limit = None
            quota_used = None
            quota_remaining = None
            remaining_pct_value = float(remaining_pct)
        elif billing_period_end_at is not None:
            raw_provider_fields = {
                "billingPeriodStart": billing_config.get("billingPeriodStart"),
                "billingPeriodEnd": billing_config.get("billingPeriodEnd"),
                "quota_unit": "grok_billing_period_only",
                "quota_unit_interpretation": "unknown_usage",
            }
            evidence = {
                "signals": [
                    "grok_billing_payload",
                    "grok_billing_period_only",
                ],
                "provider_fields": [
                    "config.billingPeriodStart",
                    "config.billingPeriodEnd",
                ],
                "rounding": "none",
                "unit_note": (
                    "Grok billing reported a billing-period boundary without "
                    "absolute or percentage usage fields; remaining quota is "
                    "unknown and left null."
                ),
            }
            quota_key = "xai_grok_build_monthly_credits:credits"
            quota_type = "credits"
            quota_limit = None
            quota_used = None
            quota_remaining = None
            remaining_pct_value = None
        else:
            raise ValueError(
                "Grok billing payload did not include absolute or percentage quota fields."
            )

    evidence.update(
        _grok_billing_sidecar_request_contract_evidence(
            config,
            identity_headers=identity_headers,
        )
    )

    signals = evidence.setdefault("signals", [])
    if "grok_billing_sidecar_request_contract" not in signals:
        signals.append("grok_billing_sidecar_request_contract")

    return (
        observed_at,
        "grok-build",
        config.grok_billing_client_version,
        None,
        "xai",
        config.grok_billing_model,
        quota_key,
        "monthly",
        quota_type,
        billing_period_end_at,
        remaining_pct_value,
        quota_limit,
        quota_used,
        quota_remaining,
        billing_period_start_at,
        billing_period_end_at,
        json.dumps(raw_provider_fields, sort_keys=True),
        json.dumps(evidence, sort_keys=True),
        "grok_billing",
        None,
        None,
        f"grok-billing-poll-{observed_at.strftime('%Y%m%d%H%M%S')}",
    )


def _persist_grok_billing_observations(
    config: ProviderStatusLoopConfig,
    *,
    observed_at: datetime,
    response_body: Dict[str, Any],
    identity_headers: Optional[Mapping[str, str]] = None,
) -> tuple[int, int]:
    payload = _build_grok_billing_rate_limit_payload(
        config,
        observed_at=observed_at,
        response_body=response_body,
        identity_headers=identity_headers,
    )
    dsn = _resolve_dsn(config)
    try:
        with probes.psycopg.connect(dsn) as conn:
            try:
                with conn.cursor() as cur:
                    _set_grok_billing_database_timeouts(
                        cur,
                        lock_timeout_ms=config.db_lock_timeout_ms,
                        statement_timeout_ms=config.db_statement_timeout_ms,
                    )
                    cur.execute(GROK_BILLING_RATE_LIMIT_INSERT_SQL, payload)
                    inserted_count = max(0, cur.rowcount)
            except (
                probes.psycopg.errors.LockNotAvailable,
                probes.psycopg.errors.QueryCanceled,
            ) as exc:
                conn.rollback()
                raise probes.ProviderStatusDatabaseWriteSkipped(
                    error_class=exc.__class__.__name__,
                    message=str(exc),
                ) from exc
    except probes.ProviderStatusDatabaseWriteSkipped:
        raise
    return 1, inserted_count


def _set_grok_billing_database_timeouts(
    cur: Any,
    *,
    lock_timeout_ms: int,
    statement_timeout_ms: int,
) -> None:
    cur.execute(
        "SELECT set_config('application_name', %s, false)",
        (f"{probes._provider_status_db_application_name()}-grok-billing",),
    )
    cur.execute("SELECT set_config('lock_timeout', %s, true)", (f"{lock_timeout_ms}ms",))
    cur.execute(
        "SELECT set_config('statement_timeout', %s, true)",
        (f"{statement_timeout_ms}ms",),
    )


def _set_observability_anomaly_scan_database_timeouts(
    cur: Any,
    *,
    lock_timeout_ms: int,
    statement_timeout_ms: int,
) -> None:
    cur.execute(
        "SELECT set_config('application_name', %s, false)",
        (f"{probes._provider_status_db_application_name()}-anomaly-scan",),
    )
    cur.execute("SELECT set_config('lock_timeout', %s, true)", (f"{lock_timeout_ms}ms",))
    cur.execute(
        "SELECT set_config('statement_timeout', %s, true)",
        (f"{statement_timeout_ms}ms",),
    )


def _rows_as_dicts(cur: Any) -> list[Dict[str, Any]]:
    columns = []
    for column in cur.description or ():
        name = getattr(column, "name", None)
        if name is None:
            name = column[0]
        columns.append(name)
    return [dict(zip(columns, row)) for row in cur.fetchall()]


def _collect_observability_anomalies(
    config: ProviderStatusLoopConfig,
) -> list[Dict[str, Any]]:
    dsn = _resolve_dsn(config)
    lookback_hours = config.observability_anomaly_scan_lookback_hours
    sample_limit = OBSERVABILITY_ANOMALY_SAMPLE_LIMIT
    anomalies: list[Dict[str, Any]] = []
    try:
        with probes.psycopg.connect(dsn) as conn:
            try:
                with conn.cursor() as cur:
                    _set_observability_anomaly_scan_database_timeouts(
                        cur,
                        lock_timeout_ms=config.db_lock_timeout_ms,
                        statement_timeout_ms=config.db_statement_timeout_ms,
                    )
                    cur.execute(
                        OBSERVABILITY_SESSION_HISTORY_ANOMALY_SQL,
                        (lookback_hours, lookback_hours, sample_limit),
                    )
                    anomalies.extend(_rows_as_dicts(cur))
                    cur.execute(
                        OBSERVABILITY_RATE_LIMIT_ANOMALY_SQL,
                        (lookback_hours, lookback_hours, sample_limit),
                    )
                    anomalies.extend(_rows_as_dicts(cur))
            except (
                probes.psycopg.errors.LockNotAvailable,
                probes.psycopg.errors.QueryCanceled,
            ) as exc:
                conn.rollback()
                raise probes.ProviderStatusDatabaseWriteSkipped(
                    error_class=exc.__class__.__name__,
                    message=str(exc),
                ) from exc
    except probes.ProviderStatusDatabaseWriteSkipped:
        raise

    return [
        {
            **anomaly,
            "row_count": int(anomaly.get("row_count") or 0),
            "examples": anomaly.get("examples") or [],
        }
        for anomaly in anomalies
        if int(anomaly.get("row_count") or 0) > 0
    ]


def _observability_anomaly_error_log_path(
    config: ProviderStatusLoopConfig,
) -> Path:
    directory = Path(
        config.observability_anomaly_scan_error_log_dir
    ).expanduser()
    environment = re.sub(
        r"[^A-Za-z0-9_.-]+",
        "_",
        (config.environment or "unknown").strip() or "unknown",
    )
    return directory / f"{environment}-error.jsonl"


def _parse_error_log_non_negative_int_env(name: str) -> Optional[int]:
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


def _parse_error_log_file_mode_env() -> Optional[int]:
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


def _normalize_error_log_file_metadata(path: Path) -> None:
    """Best-effort host bind-mount ownership repair for sidecar JSONL intake."""
    try:
        parent_stat = path.parent.stat()
        uid_raw = os.getenv("LITELLM_AAWM_ERROR_LOG_FILE_UID", "").strip()
        gid_raw = os.getenv("LITELLM_AAWM_ERROR_LOG_FILE_GID", "").strip()
        target_uid = (
            _parse_error_log_non_negative_int_env(
                "LITELLM_AAWM_ERROR_LOG_FILE_UID"
            )
            if uid_raw
            else parent_stat.st_uid
        )
        target_gid = (
            _parse_error_log_non_negative_int_env(
                "LITELLM_AAWM_ERROR_LOG_FILE_GID"
            )
            if gid_raw
            else parent_stat.st_gid
        )
        current_stat = path.stat()
        if (
            target_uid is not None
            and target_gid is not None
            and (current_stat.st_uid, current_stat.st_gid)
            != (target_uid, target_gid)
            and hasattr(os, "chown")
        ):
            try:
                os.chown(path, target_uid, target_gid)
            except OSError:
                pass

        target_mode = _parse_error_log_file_mode_env()
        if target_mode is not None and (current_stat.st_mode & 0o777) != target_mode:
            try:
                os.chmod(path, target_mode)
            except OSError:
                pass
    except Exception:
        # The anomaly scan should never fail because local intake repair failed.
        return


def _build_observability_anomaly_error_record(
    config: ProviderStatusLoopConfig,
    *,
    observed_at: datetime,
    anomaly: Mapping[str, Any],
) -> Dict[str, Any]:
    anomaly_class = str(anomaly.get("anomaly_class") or "observability_anomaly")
    row_count = int(anomaly.get("row_count") or 0)
    expected = str(anomaly.get("expected") or "observability data should be consistent")
    return {
        "event": "aawm_observability_anomaly",
        "observed_at": observed_at.isoformat().replace("+00:00", "Z"),
        "environment": config.environment,
        "error_class": "ObservabilityAnomaly",
        "error_message": (
            f"{anomaly_class}: {row_count} recent row(s) violated "
            "session-history or rate-limit telemetry expectations"
        ),
        "anomaly_class": anomaly_class,
        "anomaly_source": "provider_status_observations_sidecar",
        "lookback_hours": config.observability_anomaly_scan_lookback_hours,
        "row_count": row_count,
        "expected": expected,
        "examples": anomaly.get("examples") or [],
        "recommended_todo": (
            "Investigate the underlying telemetry mapping or persistence path, "
            "add/update the matching .analysis/todo.md item, verify healthy data, "
            "then remove or archive this error intake file."
        ),
        "cleanup_requirement": (
            "Clean up this environment error JSONL after the anomaly is resolved "
            "and represented in completed notes."
        ),
    }


def _write_observability_anomaly_error_records(
    config: ProviderStatusLoopConfig,
    *,
    observed_at: datetime,
    anomalies: Sequence[Mapping[str, Any]],
) -> tuple[int, Path]:
    path = _observability_anomaly_error_log_path(config)
    path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with path.open("a", encoding="utf-8") as handle:
        for anomaly in anomalies:
            record = _build_observability_anomaly_error_record(
                config,
                observed_at=observed_at,
                anomaly=anomaly,
            )
            handle.write(
                json.dumps(
                    _json_safe_grok_billing_value(record),
                    sort_keys=True,
                    separators=(",", ":"),
                )
            )
            handle.write("\n")
            written += 1
    _normalize_error_log_file_metadata(path)
    return written, path


def _build_grok_billing_observations_for_dry_run(
    config: ProviderStatusLoopConfig,
    *,
    observed_at: datetime,
    response_body: Dict[str, Any],
    identity_headers: Optional[Mapping[str, str]] = None,
) -> int:
    _build_grok_billing_rate_limit_payload(
        config,
        observed_at=observed_at,
        response_body=response_body,
        identity_headers=identity_headers,
    )
    return 1


def _run_grok_oidc_refresh_task(
    config: ProviderStatusLoopConfig,
    state: SidecarTaskState,
    *,
    now_monotonic: float,
) -> Optional[Dict[str, Any]]:
    if not config.grok_oidc_refresh_enabled:
        return None
    last_attempt = state.grok_oidc_last_attempt_monotonic
    if (
        last_attempt is not None
        and now_monotonic - last_attempt < config.grok_oidc_refresh_interval_seconds
    ):
        return None

    state.grok_oidc_last_attempt_monotonic = now_monotonic
    try:
        summary = grok_oidc_refresh.refresh_grok_oidc_auth_file(
            config.grok_oidc_auth_file,
            buffer_seconds=config.grok_oidc_refresh_buffer_seconds,
            force=config.grok_oidc_force_refresh,
            lock_file=config.grok_oidc_lock_file,
            http_timeout_seconds=config.grok_oidc_http_timeout_seconds,
        )
    except Exception as exc:
        summary = {
            "attempted": True,
            "refreshed": False,
            "skipped": False,
            "auth_file": config.grok_oidc_auth_file,
            "scope": None,
            "error_class": exc.__class__.__name__,
            "error_message": _redacted_failure_message(str(exc)),
        }

    event = {
        "event": "grok_oidc_refresh",
        "observed_at": _utc_timestamp(),
        "environment": config.environment,
        **summary,
    }
    (
        persisted,
        inserted_count,
        skip_error_class,
        skip_reason,
    ) = _persist_grok_oidc_auth_observation(config, event)
    event["auth_observation_status"] = _provider_auth_status_from_event(event)
    event["auth_observation_persisted"] = persisted
    event["auth_observation_inserted_count"] = inserted_count
    event["auth_observation_skip_error_class"] = skip_error_class
    event["auth_observation_skip_reason"] = skip_reason
    return event


def _run_grok_oidc_metadata_repair_task(
    config: ProviderStatusLoopConfig,
    state: SidecarTaskState,
    *,
    now_monotonic: float,
) -> Optional[Dict[str, Any]]:
    del state, now_monotonic
    if not config.grok_oidc_refresh_enabled:
        return None

    try:
        summary = grok_oidc_refresh.repair_grok_oidc_auth_file_metadata(
            config.grok_oidc_auth_file,
            lock_file=config.grok_oidc_lock_file,
        )
    except Exception as exc:
        summary = {
            "attempted": True,
            "repaired": False,
            "auth_file": config.grok_oidc_auth_file,
            "error_class": exc.__class__.__name__,
            "error_message": _redacted_failure_message(str(exc)),
        }

    if not summary.get("repaired") and not summary.get("error_class"):
        return None

    return {
        "event": "grok_oidc_metadata_repair",
        "observed_at": _utc_timestamp(),
        "environment": config.environment,
        **summary,
    }



def _run_codex_oauth_refresh_task(
    config: ProviderStatusLoopConfig,
    state: SidecarTaskState,
    *,
    now_monotonic: float,
) -> Optional[Dict[str, Any]]:
    if not config.codex_oauth_refresh_enabled:
        return None
    last_attempt = state.codex_oauth_last_attempt_monotonic
    if (
        last_attempt is not None
        and now_monotonic - last_attempt < config.codex_refresh_interval_seconds
    ):
        return None

    state.codex_oauth_last_attempt_monotonic = now_monotonic
    try:
        summary = codex_oauth_refresh.refresh_codex_oauth_auth_file(
            config.codex_auth_file,
            buffer_seconds=config.codex_refresh_buffer_seconds,
            force=config.codex_force_refresh,
            lock_file=config.codex_lock_file,
            http_timeout_seconds=config.codex_http_timeout_seconds,
        )
    except Exception as exc:
        summary = {
            "attempted": True,
            "refreshed": False,
            "skipped": False,
            "auth_file": config.codex_auth_file,
            "error_class": exc.__class__.__name__,
            "error_message": _redacted_failure_message(str(exc)),
        }

    event = {
        "event": "codex_oauth_refresh",
        "observed_at": _utc_timestamp(),
        "environment": config.environment,
        **summary,
    }
    (
        persisted,
        inserted_count,
        skip_error_class,
        skip_reason,
    ) = _persist_codex_auth_observation(config, event)
    event["auth_observation_status"] = _provider_auth_status_from_event(event)
    event["auth_observation_persisted"] = persisted
    event["auth_observation_inserted_count"] = inserted_count
    event["auth_observation_skip_error_class"] = skip_error_class
    event["auth_observation_skip_reason"] = skip_reason
    return event


def _run_xai_oauth_refresh_task(
    config: ProviderStatusLoopConfig,
    state: SidecarTaskState,
    *,
    now_monotonic: float,
) -> Optional[Dict[str, Any]]:
    if not config.xai_oauth_refresh_enabled:
        return None
    last_attempt = state.xai_oauth_last_attempt_monotonic
    if (
        last_attempt is not None
        and now_monotonic - last_attempt < config.xai_oauth_refresh_interval_seconds
    ):
        return None

    state.xai_oauth_last_attempt_monotonic = now_monotonic
    try:
        summary = xai_oauth_refresh.refresh_xai_oauth_auth_file(
            config.xai_oauth_auth_file,
            scope=config.xai_oauth_scope,
            buffer_seconds=config.xai_oauth_refresh_buffer_seconds,
            force=config.xai_oauth_force_refresh,
            lock_file=config.xai_oauth_lock_file,
            http_timeout_seconds=config.xai_oauth_http_timeout_seconds,
        )
    except Exception as exc:
        summary = {
            "attempted": True,
            "refreshed": False,
            "skipped": False,
            "auth_file": config.xai_oauth_auth_file,
            "scope": config.xai_oauth_scope,
            "error_class": exc.__class__.__name__,
            "error_message": _redacted_failure_message(str(exc)),
        }

    event = {
        "event": "xai_oauth_refresh",
        "observed_at": _utc_timestamp(),
        "environment": config.environment,
        **summary,
    }
    (
        persisted,
        inserted_count,
        skip_error_class,
        skip_reason,
    ) = _persist_xai_oauth_auth_observation(config, event)
    event["auth_observation_status"] = _provider_auth_status_from_event(event)
    event["auth_observation_persisted"] = persisted
    event["auth_observation_inserted_count"] = inserted_count
    event["auth_observation_skip_error_class"] = skip_error_class
    event["auth_observation_skip_reason"] = skip_reason
    return event


def _run_grok_billing_poll_task(
    config: ProviderStatusLoopConfig,
    state: SidecarTaskState,
    *,
    now_monotonic: float,
) -> Optional[Dict[str, Any]]:
    if not config.grok_billing_poll_enabled:
        return None
    last_attempt = state.grok_billing_last_attempt_monotonic
    if (
        last_attempt is not None
        and now_monotonic - last_attempt < config.grok_billing_poll_interval_seconds
    ):
        return None

    state.grok_billing_last_attempt_monotonic = now_monotonic
    observed_at = datetime.now(timezone.utc)
    summary: Dict[str, Any] = {
        "attempted": True,
        "persisted": False,
        "skipped": False,
        "auth_file": config.grok_oidc_auth_file,
        "resolved_auth_file": config.grok_oidc_auth_file,
        "auth_file_source": config.grok_oidc_auth_file_source,
        "billing_url": config.grok_billing_url,
        "client_version": config.grok_billing_client_version,
        "model": config.grok_billing_model,
        "observation_count": 0,
        "inserted_count": 0,
        "status_code": None,
        "attempt_count": 0,
        "retry_count": 0,
        "poll_max_attempts": max(1, config.grok_billing_poll_max_attempts),
        "error_class": None,
        "error_message": None,
    }
    summary.update(_grok_billing_request_contract_summary(config))
    try:
        auth_context = _load_grok_billing_auth_context(config.grok_oidc_auth_file)
        summary.update(
            _grok_billing_request_contract_summary(
                config,
                identity_headers=auth_context.get("identity_headers"),
            )
        )
        fetched = _fetch_grok_billing_payload(config)
        summary["status_code"] = fetched["status_code"]
        summary["attempt_count"] = fetched.get("attempt_count", 1)
        summary["retry_count"] = fetched.get("retry_count", 0)
        if config.apply:
            observation_count, inserted_count = _persist_grok_billing_observations(
                config,
                observed_at=observed_at,
                response_body=fetched["payload"],
                identity_headers=auth_context.get("identity_headers"),
            )
            summary["observation_count"] = observation_count
            summary["inserted_count"] = inserted_count
            summary["persisted"] = True
        else:
            summary["observation_count"] = _build_grok_billing_observations_for_dry_run(
                config,
                observed_at=observed_at,
                response_body=fetched["payload"],
                identity_headers=auth_context.get("identity_headers"),
            )
            summary["persisted"] = False
    except Exception as exc:
        summary["error_class"] = exc.__class__.__name__
        summary["error_message"] = _redacted_failure_message(str(exc))
        if isinstance(exc, GrokBillingPollError):
            summary["status_code"] = exc.status_code
            summary["attempt_count"] = exc.attempt_count
            summary["retry_count"] = exc.retry_count
        elif summary["status_code"] is None:
            status_match = re.search(r"with HTTP (\d{3})", str(exc))
            if status_match is not None:
                summary["status_code"] = int(status_match.group(1))
        if summary["attempt_count"] == 0:
            summary["attempt_count"] = 1

    return {
        "event": "grok_billing_poll",
        "observed_at": observed_at.isoformat().replace("+00:00", "Z"),
        "environment": config.environment,
        **summary,
    }


def _run_observability_anomaly_scan_task(
    config: ProviderStatusLoopConfig,
    state: SidecarTaskState,
    *,
    now_monotonic: float,
) -> Optional[Dict[str, Any]]:
    if not config.observability_anomaly_scan_enabled:
        return None
    last_attempt = state.observability_anomaly_scan_last_attempt_monotonic
    if (
        last_attempt is not None
        and now_monotonic - last_attempt
        < config.observability_anomaly_scan_interval_seconds
    ):
        return None

    state.observability_anomaly_scan_last_attempt_monotonic = now_monotonic
    observed_at = datetime.now(timezone.utc)
    summary: Dict[str, Any] = {
        "attempted": True,
        "status": "healthy",
        "lookback_hours": config.observability_anomaly_scan_lookback_hours,
        "anomaly_count": 0,
        "anomaly_classes": [],
        "error_log_record_count": 0,
        "error_log_path": str(_observability_anomaly_error_log_path(config)),
    }
    try:
        anomalies = _collect_observability_anomalies(config)
        if anomalies:
            written_count, path = _write_observability_anomaly_error_records(
                config,
                observed_at=observed_at,
                anomalies=anomalies,
            )
            summary.update(
                {
                    "status": "anomalies_found",
                    "anomaly_count": len(anomalies),
                    "anomaly_classes": [
                        anomaly.get("anomaly_class") for anomaly in anomalies
                    ],
                    "error_log_record_count": written_count,
                    "error_log_path": str(path),
                }
            )
    except Exception as exc:
        summary.update(
            {
                "status": "scan_failed",
                "error_class": exc.__class__.__name__,
                "error_message": _redacted_failure_message(str(exc)),
            }
        )

    return {
        "event": "observability_anomaly_scan",
        "observed_at": observed_at.isoformat().replace("+00:00", "Z"),
        "environment": config.environment,
        **summary,
    }


def run_due_sidecar_tasks(
    config: ProviderStatusLoopConfig,
    state: SidecarTaskState,
    *,
    now_monotonic: Optional[float] = None,
) -> list[Dict[str, Any]]:
    now = time.monotonic() if now_monotonic is None else now_monotonic
    events: list[Dict[str, Any]] = []
    for runner, event_name in (
        (_run_grok_oidc_metadata_repair_task, "grok_oidc_metadata_repair"),
        (_run_grok_oidc_refresh_task, "grok_oidc_refresh"),
        (_run_codex_oauth_refresh_task, "codex_oauth_refresh"),
        (_run_xai_oauth_refresh_task, "xai_oauth_refresh"),
        (_run_grok_billing_poll_task, "grok_billing_poll"),
        (_run_codex_reset_credit_poll_task, "codex_reset_credit_poll"),
        (_run_observability_anomaly_scan_task, "observability_anomaly_scan"),
    ):
        try:
            event = runner(config, state, now_monotonic=now)
        except Exception as exc:
            event = {
                "event": event_name,
                "observed_at": _utc_timestamp(),
                "environment": config.environment,
                "attempted": True,
                "error_class": exc.__class__.__name__,
                "error_message": _redacted_failure_message(str(exc)),
            }
        if event is not None:
            events.append(event)
    return events


def _emit(payload: Dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(payload, sort_keys=True))
    sys.stdout.write("\n")
    sys.stdout.flush()


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def main(argv: Optional[Sequence[str]] = None) -> int:
    config = parse_config(argv)
    try:
        validate_runtime_guardrails(config)
    except Exception as exc:
        _emit(
            {
                "event": "provider_status_observations_guardrail_error",
                "observed_at": _utc_timestamp(),
                "environment": config.environment,
                "error_class": exc.__class__.__name__,
                "error_message": str(exc),
            }
        )
        return 1
    stopping = False

    def _stop(_signum: int, _frame: object) -> None:
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    if config.apply and config.setup_schema:
        try:
            _emit(setup_schema_once(config))
        except Exception as exc:
            _emit(
                {
                    "event": "provider_status_observations_schema_error",
                    "observed_at": _utc_timestamp(),
                    "environment": config.environment,
                    "error_class": exc.__class__.__name__,
                    "error_message": str(exc),
                    "traceback": traceback.format_exc(limit=5),
                }
            )
            if config.once:
                return 1

    sidecar_state = SidecarTaskState()
    while not stopping:
        cycle_started = time.monotonic()
        try:
            _emit(run_cycle(config))
        except Exception as exc:
            _emit(
                {
                    "event": "provider_status_observations_error",
                    "observed_at": _utc_timestamp(),
                    "environment": config.environment,
                    "error_class": exc.__class__.__name__,
                    "error_message": str(exc),
                    "traceback": traceback.format_exc(limit=5),
                }
            )
            if config.once:
                return 1
        try:
            for event in run_due_sidecar_tasks(config, sidecar_state):
                _emit(event)
        except Exception as exc:
            _emit(
                {
                    "event": "provider_status_sidecar_task_error",
                    "observed_at": _utc_timestamp(),
                    "environment": config.environment,
                    "task": "sidecar_tasks",
                    "error_class": exc.__class__.__name__,
                    "error_message": _redacted_failure_message(str(exc)),
                    "traceback": traceback.format_exc(limit=5),
                }
            )
            if config.once:
                return 1

        if config.once:
            return 0

        remaining_seconds = config.interval_seconds - (time.monotonic() - cycle_started)
        while remaining_seconds > 0 and not stopping:
            time.sleep(min(remaining_seconds, 1.0))
            remaining_seconds = config.interval_seconds - (time.monotonic() - cycle_started)

    _emit(
        {
            "event": "provider_status_observations_stopped",
            "observed_at": _utc_timestamp(),
            "environment": config.environment,
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
