"""AAWM session_history SQL DDL/DML constants.

Owned by the AAWM session-history persistence subsystem. These strings are
shared by `litellm.integrations.aawm_agent_identity` (runtime writer) and
repair/backfill scripts. Schema evolution should land here rather than as
inline SQL in the identity callback module.

This module is intentionally SQL-only (no runtime side effects) so scripts can
import constants without loading the full agent-identity callback surface.
"""

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
    inbound_model_alias TEXT,
    model_group TEXT,
    agent_name TEXT,
    agent_id TEXT,
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
    invalid_tool_call_count INTEGER NOT NULL DEFAULT 0,
    structured_output_attempted BOOLEAN NOT NULL DEFAULT FALSE,
    structured_output_failed BOOLEAN NOT NULL DEFAULT FALSE,
    structured_output_mode TEXT,
    structured_output_schema_hash TEXT,
    structured_output_failure_reason TEXT,
    tool_names JSONB NOT NULL DEFAULT '[]'::jsonb,
    file_read_count INTEGER NOT NULL DEFAULT 0,
    file_modified_count INTEGER NOT NULL DEFAULT 0,
    changed_pre_commit_config BOOLEAN,
    changed_env_file BOOLEAN,
    changed_pyproject_toml BOOLEAN,
    changed_gitignore BOOLEAN,
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
    client_ip TEXT,
    host_name TEXT,
    token_permission_input INTEGER NOT NULL DEFAULT 0,
    token_permission_output INTEGER NOT NULL DEFAULT 0,
    permission_usd_cost DOUBLE PRECISION NOT NULL DEFAULT 0,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    repository TEXT,
    input_system_tokens_estimated INTEGER NOT NULL DEFAULT 0,
    input_tool_advertisement_tokens_estimated INTEGER NOT NULL DEFAULT 0,
    input_conversation_tokens_estimated INTEGER NOT NULL DEFAULT 0,
    input_other_tokens_estimated INTEGER NOT NULL DEFAULT 0,
    input_breakdown_residual_tokens INTEGER NOT NULL DEFAULT 0,
    system_behavior_tokens_estimated INTEGER NOT NULL DEFAULT 0,
    system_safety_tokens_estimated INTEGER NOT NULL DEFAULT 0,
    system_instructional_tokens_estimated INTEGER NOT NULL DEFAULT 0,
    system_unclassified_tokens_estimated INTEGER NOT NULL DEFAULT 0,
    litellm_processing_ms DOUBLE PRECISION,
    llm_upstream_elapsed_ms DOUBLE PRECISION,
    total_server_elapsed_ms DOUBLE PRECISION,
    ttft_ms DOUBLE PRECISION,
    litellm_pre_send_ms DOUBLE PRECISION,
    litellm_post_response_ms DOUBLE PRECISION,
    llm_upstream_time_to_first_byte_ms DOUBLE PRECISION,
    llm_upstream_stream_ms DOUBLE PRECISION,
    latency_unclassified_ms DOUBLE PRECISION,
    previous_response_to_current_request_ms DOUBLE PRECISION,
    trace_quality_score DOUBLE PRECISION,
    empty_completion_failure BOOLEAN,
    large_tool_result_payload_risk BOOLEAN,
    destructive_checkout_after_work BOOLEAN,
    invalid_tool_call_error BOOLEAN,
    read_only_policy_compliance_score DOUBLE PRECISION,
    read_only_policy_violation_count INTEGER,
    response_meaningfulness_score DOUBLE PRECISION,
    instruction_adherence_score DOUBLE PRECISION,
    answer_completeness_score DOUBLE PRECISION,
    evidence_fidelity_score DOUBLE PRECISION,
    tool_result_fidelity_score DOUBLE PRECISION,
    error_attribution_quality_score DOUBLE PRECISION,
    repetition_loop_risk_score DOUBLE PRECISION,
    context_retention_score DOUBLE PRECISION,
    tool_use_validity_score DOUBLE PRECISION,
    tool_error_recovery_score DOUBLE PRECISION,
    stall_risk_score DOUBLE PRECISION,
    output_contract_compliance_score DOUBLE PRECISION,
    task_progress_score DOUBLE PRECISION,
    scope_control_score DOUBLE PRECISION,
    destructive_action_policy_score DOUBLE PRECISION,
    ignored_path_tracking_policy_score DOUBLE PRECISION,
    ignored_path_tracking_violation_count INTEGER,
    baseline_deflection_attempted_score DOUBLE PRECISION,
    baseline_deflection_incident_score DOUBLE PRECISION,
    baseline_deflection_attempt_count INTEGER,
    baseline_deflection_tool_call_count INTEGER,
    baseline_deflection_input_tokens INTEGER,
    baseline_deflection_elapsed_ms DOUBLE PRECISION,
    quality_gate_trigger_count INTEGER,
    quality_gate_fix_attempt_count INTEGER,
    quality_gate_rerun_count INTEGER,
    sleep_wellness_interruption_attempted_score DOUBLE PRECISION,
    sleep_wellness_interruption_incident_score DOUBLE PRECISION,
    sleep_wellness_interruption_count INTEGER,
    sleep_wellness_interruption_output_tokens INTEGER,
    sleep_wellness_interruption_input_tokens INTEGER,
    sleep_wellness_interruption_elapsed_ms DOUBLE PRECISION,
    sleep_wellness_interruption_after_user_pushback_count INTEGER,
    sleep_wellness_interruption_repeated_count INTEGER,
    terminal_completion_score DOUBLE PRECISION,
    discovery_inventory_coverage_score DOUBLE PRECISION,
    discovery_inventory_missing_count INTEGER,
    agent_score_reasons JSONB NOT NULL DEFAULT '{}'::jsonb,
    is_compact_summary BOOLEAN NOT NULL DEFAULT FALSE,
    compact_summary_source TEXT,
    compact_summary_id TEXT,
    compact_summary_role TEXT
)
"""
_AAWM_SESSION_HISTORY_ALTER_STATEMENTS = (
    "ALTER TABLE public.session_history ADD COLUMN IF NOT EXISTS inbound_model_alias TEXT",
    "ALTER TABLE public.session_history ADD COLUMN IF NOT EXISTS agent_id TEXT",
    "ALTER TABLE public.session_history ADD COLUMN IF NOT EXISTS tenant_id TEXT",
    "ALTER TABLE public.session_history ADD COLUMN IF NOT EXISTS file_read_count INTEGER NOT NULL DEFAULT 0",
    "ALTER TABLE public.session_history ADD COLUMN IF NOT EXISTS file_modified_count INTEGER NOT NULL DEFAULT 0",
    "ALTER TABLE public.session_history ADD COLUMN IF NOT EXISTS changed_pre_commit_config BOOLEAN",
    "ALTER TABLE public.session_history ADD COLUMN IF NOT EXISTS changed_env_file BOOLEAN",
    "ALTER TABLE public.session_history ADD COLUMN IF NOT EXISTS changed_pyproject_toml BOOLEAN",
    "ALTER TABLE public.session_history ADD COLUMN IF NOT EXISTS changed_gitignore BOOLEAN",
    "ALTER TABLE public.session_history ADD COLUMN IF NOT EXISTS git_commit_count INTEGER NOT NULL DEFAULT 0",
    "ALTER TABLE public.session_history ADD COLUMN IF NOT EXISTS git_push_count INTEGER NOT NULL DEFAULT 0",
    "ALTER TABLE public.session_history ADD COLUMN IF NOT EXISTS provider_cache_attempted BOOLEAN NOT NULL DEFAULT FALSE",
    "ALTER TABLE public.session_history ADD COLUMN IF NOT EXISTS provider_cache_status TEXT",
    "ALTER TABLE public.session_history ADD COLUMN IF NOT EXISTS provider_cache_miss BOOLEAN NOT NULL DEFAULT FALSE",
    "ALTER TABLE public.session_history ADD COLUMN IF NOT EXISTS provider_cache_miss_reason TEXT",
    "ALTER TABLE public.session_history ADD COLUMN IF NOT EXISTS provider_cache_miss_token_count INTEGER",
    "ALTER TABLE public.session_history ADD COLUMN IF NOT EXISTS provider_cache_miss_cost_usd DOUBLE PRECISION",
    "ALTER TABLE public.session_history ADD COLUMN IF NOT EXISTS invalid_tool_call_count INTEGER NOT NULL DEFAULT 0",
    "ALTER TABLE public.session_history ADD COLUMN IF NOT EXISTS structured_output_attempted BOOLEAN NOT NULL DEFAULT FALSE",
    "ALTER TABLE public.session_history ADD COLUMN IF NOT EXISTS structured_output_failed BOOLEAN NOT NULL DEFAULT FALSE",
    "ALTER TABLE public.session_history ADD COLUMN IF NOT EXISTS structured_output_mode TEXT",
    "ALTER TABLE public.session_history ADD COLUMN IF NOT EXISTS structured_output_schema_hash TEXT",
    "ALTER TABLE public.session_history ADD COLUMN IF NOT EXISTS structured_output_failure_reason TEXT",
    "ALTER TABLE public.session_history ADD COLUMN IF NOT EXISTS litellm_environment TEXT",
    "ALTER TABLE public.session_history ADD COLUMN IF NOT EXISTS litellm_version TEXT",
    "ALTER TABLE public.session_history ADD COLUMN IF NOT EXISTS litellm_fork_version TEXT",
    "ALTER TABLE public.session_history ADD COLUMN IF NOT EXISTS litellm_wheel_versions JSONB NOT NULL DEFAULT '{}'::jsonb",
    "ALTER TABLE public.session_history ADD COLUMN IF NOT EXISTS client_name TEXT",
    "ALTER TABLE public.session_history ADD COLUMN IF NOT EXISTS client_version TEXT",
    "ALTER TABLE public.session_history ADD COLUMN IF NOT EXISTS client_user_agent TEXT",
    "ALTER TABLE public.session_history ADD COLUMN IF NOT EXISTS client_ip TEXT",
    "ALTER TABLE public.session_history ADD COLUMN IF NOT EXISTS host_name TEXT",
    "ALTER TABLE public.session_history ADD COLUMN IF NOT EXISTS token_permission_input INTEGER NOT NULL DEFAULT 0",
    "ALTER TABLE public.session_history ADD COLUMN IF NOT EXISTS token_permission_output INTEGER NOT NULL DEFAULT 0",
    "ALTER TABLE public.session_history ADD COLUMN IF NOT EXISTS permission_usd_cost DOUBLE PRECISION NOT NULL DEFAULT 0",
    "ALTER TABLE public.session_history ADD COLUMN IF NOT EXISTS repository TEXT",
    "ALTER TABLE public.session_history ADD COLUMN IF NOT EXISTS input_system_tokens_estimated INTEGER NOT NULL DEFAULT 0",
    "ALTER TABLE public.session_history ADD COLUMN IF NOT EXISTS input_tool_advertisement_tokens_estimated INTEGER NOT NULL DEFAULT 0",
    "ALTER TABLE public.session_history ADD COLUMN IF NOT EXISTS input_conversation_tokens_estimated INTEGER NOT NULL DEFAULT 0",
    "ALTER TABLE public.session_history ADD COLUMN IF NOT EXISTS input_other_tokens_estimated INTEGER NOT NULL DEFAULT 0",
    "ALTER TABLE public.session_history ADD COLUMN IF NOT EXISTS input_breakdown_residual_tokens INTEGER NOT NULL DEFAULT 0",
    "ALTER TABLE public.session_history ADD COLUMN IF NOT EXISTS system_behavior_tokens_estimated INTEGER NOT NULL DEFAULT 0",
    "ALTER TABLE public.session_history ADD COLUMN IF NOT EXISTS system_safety_tokens_estimated INTEGER NOT NULL DEFAULT 0",
    "ALTER TABLE public.session_history ADD COLUMN IF NOT EXISTS system_instructional_tokens_estimated INTEGER NOT NULL DEFAULT 0",
    "ALTER TABLE public.session_history ADD COLUMN IF NOT EXISTS system_unclassified_tokens_estimated INTEGER NOT NULL DEFAULT 0",
    "ALTER TABLE public.session_history ADD COLUMN IF NOT EXISTS litellm_processing_ms DOUBLE PRECISION",
    "ALTER TABLE public.session_history ADD COLUMN IF NOT EXISTS llm_upstream_elapsed_ms DOUBLE PRECISION",
    "ALTER TABLE public.session_history ADD COLUMN IF NOT EXISTS total_server_elapsed_ms DOUBLE PRECISION",
    "ALTER TABLE public.session_history ADD COLUMN IF NOT EXISTS ttft_ms DOUBLE PRECISION",
    "ALTER TABLE public.session_history ADD COLUMN IF NOT EXISTS litellm_pre_send_ms DOUBLE PRECISION",
    "ALTER TABLE public.session_history ADD COLUMN IF NOT EXISTS litellm_post_response_ms DOUBLE PRECISION",
    "ALTER TABLE public.session_history ADD COLUMN IF NOT EXISTS llm_upstream_time_to_first_byte_ms DOUBLE PRECISION",
    "ALTER TABLE public.session_history ADD COLUMN IF NOT EXISTS llm_upstream_stream_ms DOUBLE PRECISION",
    "ALTER TABLE public.session_history ADD COLUMN IF NOT EXISTS latency_unclassified_ms DOUBLE PRECISION",
    "ALTER TABLE public.session_history ADD COLUMN IF NOT EXISTS previous_response_to_current_request_ms DOUBLE PRECISION",
    "ALTER TABLE public.session_history ADD COLUMN IF NOT EXISTS trace_quality_score DOUBLE PRECISION",
    "ALTER TABLE public.session_history ADD COLUMN IF NOT EXISTS empty_completion_failure BOOLEAN",
    "ALTER TABLE public.session_history ADD COLUMN IF NOT EXISTS large_tool_result_payload_risk BOOLEAN",
    "ALTER TABLE public.session_history ADD COLUMN IF NOT EXISTS destructive_checkout_after_work BOOLEAN",
    "ALTER TABLE public.session_history ADD COLUMN IF NOT EXISTS invalid_tool_call_error BOOLEAN",
    "ALTER TABLE public.session_history ADD COLUMN IF NOT EXISTS "
    "read_only_policy_compliance_score DOUBLE PRECISION",
    "ALTER TABLE public.session_history ADD COLUMN IF NOT EXISTS read_only_policy_violation_count INTEGER",
    "ALTER TABLE public.session_history ADD COLUMN IF NOT EXISTS response_meaningfulness_score DOUBLE PRECISION",
    "ALTER TABLE public.session_history ADD COLUMN IF NOT EXISTS instruction_adherence_score DOUBLE PRECISION",
    "ALTER TABLE public.session_history ADD COLUMN IF NOT EXISTS answer_completeness_score DOUBLE PRECISION",
    "ALTER TABLE public.session_history ADD COLUMN IF NOT EXISTS evidence_fidelity_score DOUBLE PRECISION",
    "ALTER TABLE public.session_history ADD COLUMN IF NOT EXISTS tool_result_fidelity_score DOUBLE PRECISION",
    "ALTER TABLE public.session_history ADD COLUMN IF NOT EXISTS error_attribution_quality_score DOUBLE PRECISION",
    "ALTER TABLE public.session_history ADD COLUMN IF NOT EXISTS repetition_loop_risk_score DOUBLE PRECISION",
    "ALTER TABLE public.session_history ADD COLUMN IF NOT EXISTS context_retention_score DOUBLE PRECISION",
    "ALTER TABLE public.session_history ADD COLUMN IF NOT EXISTS tool_use_validity_score DOUBLE PRECISION",
    "ALTER TABLE public.session_history ADD COLUMN IF NOT EXISTS tool_error_recovery_score DOUBLE PRECISION",
    "ALTER TABLE public.session_history ADD COLUMN IF NOT EXISTS stall_risk_score DOUBLE PRECISION",
    "ALTER TABLE public.session_history ADD COLUMN IF NOT EXISTS "
    "output_contract_compliance_score DOUBLE PRECISION",
    "ALTER TABLE public.session_history ADD COLUMN IF NOT EXISTS task_progress_score DOUBLE PRECISION",
    "ALTER TABLE public.session_history ADD COLUMN IF NOT EXISTS scope_control_score DOUBLE PRECISION",
    "ALTER TABLE public.session_history ADD COLUMN IF NOT EXISTS "
    "destructive_action_policy_score DOUBLE PRECISION",
    "ALTER TABLE public.session_history ADD COLUMN IF NOT EXISTS "
    "ignored_path_tracking_policy_score DOUBLE PRECISION",
    "ALTER TABLE public.session_history ADD COLUMN IF NOT EXISTS "
    "ignored_path_tracking_violation_count INTEGER",
    "ALTER TABLE public.session_history ADD COLUMN IF NOT EXISTS "
    "baseline_deflection_attempted_score DOUBLE PRECISION",
    "ALTER TABLE public.session_history ADD COLUMN IF NOT EXISTS "
    "baseline_deflection_incident_score DOUBLE PRECISION",
    "ALTER TABLE public.session_history ADD COLUMN IF NOT EXISTS baseline_deflection_attempt_count INTEGER",
    "ALTER TABLE public.session_history ADD COLUMN IF NOT EXISTS "
    "baseline_deflection_tool_call_count INTEGER",
    "ALTER TABLE public.session_history ADD COLUMN IF NOT EXISTS baseline_deflection_input_tokens INTEGER",
    "ALTER TABLE public.session_history ADD COLUMN IF NOT EXISTS "
    "baseline_deflection_elapsed_ms DOUBLE PRECISION",
    "ALTER TABLE public.session_history ADD COLUMN IF NOT EXISTS quality_gate_trigger_count INTEGER",
    "ALTER TABLE public.session_history ADD COLUMN IF NOT EXISTS quality_gate_fix_attempt_count INTEGER",
    "ALTER TABLE public.session_history ADD COLUMN IF NOT EXISTS quality_gate_rerun_count INTEGER",
    "ALTER TABLE public.session_history ADD COLUMN IF NOT EXISTS "
    "sleep_wellness_interruption_attempted_score DOUBLE PRECISION",
    "ALTER TABLE public.session_history ADD COLUMN IF NOT EXISTS "
    "sleep_wellness_interruption_incident_score DOUBLE PRECISION",
    "ALTER TABLE public.session_history ADD COLUMN IF NOT EXISTS "
    "sleep_wellness_interruption_count INTEGER",
    "ALTER TABLE public.session_history ADD COLUMN IF NOT EXISTS "
    "sleep_wellness_interruption_output_tokens INTEGER",
    "ALTER TABLE public.session_history ADD COLUMN IF NOT EXISTS "
    "sleep_wellness_interruption_input_tokens INTEGER",
    "ALTER TABLE public.session_history ADD COLUMN IF NOT EXISTS "
    "sleep_wellness_interruption_elapsed_ms DOUBLE PRECISION",
    "ALTER TABLE public.session_history ADD COLUMN IF NOT EXISTS "
    "sleep_wellness_interruption_after_user_pushback_count INTEGER",
    "ALTER TABLE public.session_history ADD COLUMN IF NOT EXISTS "
    "sleep_wellness_interruption_repeated_count INTEGER",
    "ALTER TABLE public.session_history ADD COLUMN IF NOT EXISTS terminal_completion_score DOUBLE PRECISION",
    "ALTER TABLE public.session_history ADD COLUMN IF NOT EXISTS "
    "discovery_inventory_coverage_score DOUBLE PRECISION",
    "ALTER TABLE public.session_history ADD COLUMN IF NOT EXISTS "
    "discovery_inventory_missing_count INTEGER",
    "ALTER TABLE public.session_history ADD COLUMN IF NOT EXISTS agent_score_reasons "
    "JSONB NOT NULL DEFAULT '{}'::jsonb",
    "ALTER TABLE public.session_history ADD COLUMN IF NOT EXISTS is_compact_summary "
    "BOOLEAN NOT NULL DEFAULT FALSE",
    "ALTER TABLE public.session_history ADD COLUMN IF NOT EXISTS compact_summary_source TEXT",
    "ALTER TABLE public.session_history ADD COLUMN IF NOT EXISTS compact_summary_role TEXT",
    "ALTER TABLE public.session_history ADD COLUMN IF NOT EXISTS compact_summary_id TEXT",
)
_AAWM_SESSION_HISTORY_INDEX_STATEMENTS = (
    "CREATE INDEX IF NOT EXISTS session_history_created_at_idx ON public.session_history (created_at DESC)",
    "CREATE INDEX IF NOT EXISTS session_history_session_created_idx ON public.session_history (session_id, created_at DESC)",
    "CREATE INDEX IF NOT EXISTS session_history_session_model_created_idx ON public.session_history (session_id, model, created_at DESC)",
    "CREATE INDEX IF NOT EXISTS session_history_session_start_idx ON public.session_history (session_id, (COALESCE(start_time, created_at)), id)",
    "CREATE INDEX IF NOT EXISTS session_history_litellm_environment_created_idx ON public.session_history (litellm_environment, created_at DESC)",
    "CREATE INDEX IF NOT EXISTS session_history_client_created_idx ON public.session_history (client_name, client_version, created_at DESC)",
    "CREATE INDEX IF NOT EXISTS session_history_repository_created_idx ON public.session_history (repository, created_at DESC)",
    "CREATE INDEX IF NOT EXISTS session_history_compact_summary_idx ON public.session_history (session_id, compact_summary_id, created_at DESC) WHERE is_compact_summary",
    "CREATE INDEX IF NOT EXISTS session_history_openrouter_free_observed_idx ON public.session_history ((COALESCE(end_time, start_time, created_at)) DESC) WHERE provider = 'openrouter' AND lower(COALESCE(model, '')) LIKE '%:free'",
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
    agent_id TEXT,
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
_AAWM_TOOL_DEFINITION_SNAPSHOT_METADATA_KEY = "aawm_tool_definition_snapshot"
_AAWM_SESSION_HISTORY_TOOL_DEFINITION_SNAPSHOTS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS public.session_history_tool_definition_snapshots (
    id BIGSERIAL PRIMARY KEY,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    session_id TEXT NOT NULL,
    snapshot_hash TEXT NOT NULL,
    capture_version TEXT,
    capture_source TEXT,
    tool_definition_count INTEGER,
    captured_count INTEGER,
    tool_definition_sources JSONB NOT NULL DEFAULT '[]'::jsonb,
    tool_definition_names JSONB NOT NULL DEFAULT '[]'::jsonb,
    tool_definition_types JSONB NOT NULL DEFAULT '[]'::jsonb,
    snapshot_truncated BOOLEAN NOT NULL DEFAULT FALSE,
    sanitized_snapshot JSONB NOT NULL,
    first_litellm_call_id TEXT,
    first_trace_id TEXT,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    UNIQUE (session_id, snapshot_hash)
)
"""
_AAWM_SESSION_HISTORY_TOOL_DEFINITION_SNAPSHOTS_INDEX_STATEMENTS = (
    "CREATE INDEX IF NOT EXISTS session_history_tool_definition_snapshots_session_created_idx "
    "ON public.session_history_tool_definition_snapshots (session_id, created_at DESC)",
    "CREATE INDEX IF NOT EXISTS session_history_tool_definition_snapshots_hash_idx "
    "ON public.session_history_tool_definition_snapshots (snapshot_hash)",
)
_AAWM_RATE_LIMIT_OBSERVATIONS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS public.rate_limit_observations (
    id BIGSERIAL PRIMARY KEY,
    observed_at TIMESTAMPTZ NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    client TEXT,
    client_version TEXT,
    account_hash TEXT,
    provider TEXT NOT NULL,
    model TEXT,
    quota_key TEXT NOT NULL,
    quota_period TEXT,
    quota_type TEXT,
    expected_reset_at TIMESTAMPTZ,
    remaining_pct DOUBLE PRECISION,
    quota_limit DOUBLE PRECISION,
    quota_used DOUBLE PRECISION,
    quota_remaining DOUBLE PRECISION,
    billing_period_start_at TIMESTAMPTZ,
    billing_period_end_at TIMESTAMPTZ,
    raw_provider_fields JSONB NOT NULL DEFAULT '{}'::jsonb,
    evidence JSONB NOT NULL DEFAULT '{}'::jsonb,
    source TEXT,
    session_id TEXT,
    trace_id TEXT,
    litellm_call_id TEXT
)
"""
_AAWM_RATE_LIMIT_OBSERVATIONS_ALTER_STATEMENTS = (
    "DROP INDEX IF EXISTS public.rate_limit_observations_limit_observed_idx",
    "DROP INDEX IF EXISTS public.rate_limit_observations_provider_client_model_idx",
    "DROP INDEX IF EXISTS public.rate_limit_observations_reset_idx",
    "DROP INDEX IF EXISTS public.rate_limit_observations_session_idx",
    "DROP INDEX IF EXISTS public.rate_limit_observations_trace_call_idx",
    "DROP INDEX IF EXISTS public.rate_limit_observations_repository_idx",
    "ALTER TABLE public.rate_limit_observations ADD COLUMN IF NOT EXISTS client TEXT",
    "ALTER TABLE public.rate_limit_observations ADD COLUMN IF NOT EXISTS quota_key TEXT",
    "ALTER TABLE public.rate_limit_observations ADD COLUMN IF NOT EXISTS quota_type TEXT",
    "ALTER TABLE public.rate_limit_observations ADD COLUMN IF NOT EXISTS expected_reset_at TIMESTAMPTZ",
    "ALTER TABLE public.rate_limit_observations ADD COLUMN IF NOT EXISTS remaining_pct DOUBLE PRECISION",
    "ALTER TABLE public.rate_limit_observations ADD COLUMN IF NOT EXISTS quota_limit DOUBLE PRECISION",
    "ALTER TABLE public.rate_limit_observations ADD COLUMN IF NOT EXISTS quota_used DOUBLE PRECISION",
    "ALTER TABLE public.rate_limit_observations ADD COLUMN IF NOT EXISTS quota_remaining DOUBLE PRECISION",
    "ALTER TABLE public.rate_limit_observations ADD COLUMN IF NOT EXISTS billing_period_start_at TIMESTAMPTZ",
    "ALTER TABLE public.rate_limit_observations ADD COLUMN IF NOT EXISTS billing_period_end_at TIMESTAMPTZ",
    "ALTER TABLE public.rate_limit_observations ADD COLUMN IF NOT EXISTS raw_provider_fields JSONB DEFAULT '{}'::jsonb",
    "ALTER TABLE public.rate_limit_observations ADD COLUMN IF NOT EXISTS evidence JSONB DEFAULT '{}'::jsonb",
    "UPDATE public.rate_limit_observations SET raw_provider_fields = '{}'::jsonb WHERE raw_provider_fields IS NULL",
    "UPDATE public.rate_limit_observations SET evidence = '{}'::jsonb WHERE evidence IS NULL",
    "ALTER TABLE public.rate_limit_observations ALTER COLUMN raw_provider_fields SET DEFAULT '{}'::jsonb",
    "ALTER TABLE public.rate_limit_observations ALTER COLUMN evidence SET DEFAULT '{}'::jsonb",
    "ALTER TABLE public.rate_limit_observations ALTER COLUMN raw_provider_fields SET NOT NULL",
    "ALTER TABLE public.rate_limit_observations ALTER COLUMN evidence SET NOT NULL",
    """
DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_schema = 'public'
          AND table_name = 'rate_limit_observations'
          AND column_name = 'client_family'
    ) THEN
        UPDATE public.rate_limit_observations
        SET client = COALESCE(client, client_family);
    END IF;

    IF EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_schema = 'public'
          AND table_name = 'rate_limit_observations'
          AND column_name = 'client_name'
    ) THEN
        UPDATE public.rate_limit_observations
        SET client = COALESCE(client, client_name);
    END IF;

    UPDATE public.rate_limit_observations
    SET client = COALESCE(client, 'unknown');
END $$;
""",
    """
DO $$
BEGIN
    UPDATE public.rate_limit_observations
    SET provider = CASE
        WHEN lower(COALESCE(provider, '')) IN ('gemini', 'google_code_assist')
          OR lower(COALESCE(client, '')) IN ('gemini', 'google_code_assist')
          OR source LIKE 'google_%'
          OR source LIKE 'gemini_%'
            THEN 'google'
        WHEN provider IS NULL OR provider = ''
            THEN 'unknown'
        ELSE provider
    END;
END $$;
""",
    """
UPDATE public.rate_limit_observations
SET client = 'google_code_assist'
WHERE provider = 'google'
  AND source = 'google_retrieve_user_quota'
  AND client IN ('gemini', 'google');
""",
    """
DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_schema = 'public'
          AND table_name = 'rate_limit_observations'
          AND column_name = 'limit_id'
    ) AND EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_schema = 'public'
          AND table_name = 'rate_limit_observations'
          AND column_name = 'limit_scope'
    ) THEN
        UPDATE public.rate_limit_observations
        SET quota_key = COALESCE(
            quota_key,
            NULLIF(CONCAT_WS(':', NULLIF(limit_id, ''), NULLIF(limit_scope, '')), '')
        );
    END IF;

    IF EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_schema = 'public'
          AND table_name = 'rate_limit_observations'
          AND column_name = 'limit_key'
    ) THEN
        UPDATE public.rate_limit_observations
        SET quota_key = COALESCE(quota_key, limit_key);
    END IF;

    UPDATE public.rate_limit_observations
    SET quota_key = COALESCE(
        quota_key,
        CONCAT_WS(':', COALESCE(source, 'unknown_source'), COALESCE(model, 'unknown_model'))
    );
END $$;
""",
    """
DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_schema = 'public'
          AND table_name = 'rate_limit_observations'
          AND column_name = 'provider_resets_at'
    ) THEN
        UPDATE public.rate_limit_observations
        SET expected_reset_at = COALESCE(expected_reset_at, provider_resets_at);
    END IF;

    IF EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_schema = 'public'
          AND table_name = 'rate_limit_observations'
          AND column_name = 'used_percentage'
    ) THEN
        UPDATE public.rate_limit_observations
        SET remaining_pct = COALESCE(
            remaining_pct,
            GREATEST(0.0, LEAST(100.0, 100.0 - used_percentage))
        )
        WHERE used_percentage IS NOT NULL;
    END IF;
END $$;
""",
    """
DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_schema = 'public'
          AND table_name = 'rate_limit_observations'
          AND column_name = 'limit_scope'
    ) THEN
        UPDATE public.rate_limit_observations
        SET quota_type = COALESCE(
            quota_type,
            CASE
                WHEN limit_scope ILIKE '%request%' OR limit_scope = 'requests'
                    THEN 'requests'
                WHEN limit_scope ILIKE '%token%' OR limit_scope = 'tokens'
                    THEN 'tokens'
                WHEN limit_scope = 'model_capacity'
                    THEN 'capacity'
                ELSE NULL
            END
        );
    END IF;

    IF EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_schema = 'public'
          AND table_name = 'rate_limit_observations'
          AND column_name = 'raw_provider_fields'
    ) THEN
        UPDATE public.rate_limit_observations
        SET quota_type = COALESCE(
            quota_type,
            CASE
                WHEN raw_provider_fields->>'tokenType' ILIKE 'REQUESTS'
                    THEN 'requests'
                WHEN raw_provider_fields->>'tokenType' ILIKE 'TOKENS'
                    THEN 'tokens'
                ELSE NULL
            END
        );
    END IF;

    UPDATE public.rate_limit_observations
    SET quota_type = COALESCE(
        quota_type,
        CASE
            WHEN source = 'google_model_capacity_error' THEN 'capacity'
            WHEN provider = 'google' THEN 'requests'
            WHEN provider IN ('openai', 'anthropic') THEN 'tokens'
            ELSE 'unknown'
        END
    );
END $$;
""",
    "ALTER TABLE public.rate_limit_observations ALTER COLUMN provider SET NOT NULL",
    "ALTER TABLE public.rate_limit_observations ALTER COLUMN quota_key SET NOT NULL",
    "ALTER TABLE public.rate_limit_observations ALTER COLUMN source DROP NOT NULL",
    "ALTER TABLE public.rate_limit_observations DROP COLUMN IF EXISTS client_family",
    "ALTER TABLE public.rate_limit_observations DROP COLUMN IF EXISTS environment",
    "ALTER TABLE public.rate_limit_observations DROP COLUMN IF EXISTS tenant_id",
    "ALTER TABLE public.rate_limit_observations DROP COLUMN IF EXISTS repository",
    "ALTER TABLE public.rate_limit_observations DROP COLUMN IF EXISTS limit_key",
    "ALTER TABLE public.rate_limit_observations DROP COLUMN IF EXISTS limit_id",
    "ALTER TABLE public.rate_limit_observations DROP COLUMN IF EXISTS limit_name",
    "ALTER TABLE public.rate_limit_observations DROP COLUMN IF EXISTS limit_scope",
    "ALTER TABLE public.rate_limit_observations DROP COLUMN IF EXISTS window_minutes",
    "ALTER TABLE public.rate_limit_observations DROP COLUMN IF EXISTS provider_resets_at",
    "ALTER TABLE public.rate_limit_observations DROP COLUMN IF EXISTS inferred_window_start_at",
    "ALTER TABLE public.rate_limit_observations DROP COLUMN IF EXISTS used_percentage",
    "ALTER TABLE public.rate_limit_observations DROP COLUMN IF EXISTS remaining_requests",
    "ALTER TABLE public.rate_limit_observations DROP COLUMN IF EXISTS used_requests",
    "ALTER TABLE public.rate_limit_observations DROP COLUMN IF EXISTS total_requests",
    "ALTER TABLE public.rate_limit_observations DROP COLUMN IF EXISTS status",
    "ALTER TABLE public.rate_limit_observations DROP COLUMN IF EXISTS exhausted",
    "ALTER TABLE public.rate_limit_observations DROP COLUMN IF EXISTS exhaustion_kind",
    "ALTER TABLE public.rate_limit_observations DROP COLUMN IF EXISTS reset_hint_seconds",
    "ALTER TABLE public.rate_limit_observations DROP COLUMN IF EXISTS model_family",
    "ALTER TABLE public.rate_limit_observations DROP COLUMN IF EXISTS model_tier",
    "ALTER TABLE public.rate_limit_observations DROP COLUMN IF EXISTS parent_limit_key",
    "ALTER TABLE public.rate_limit_observations DROP COLUMN IF EXISTS route_family",
    "ALTER TABLE public.rate_limit_observations DROP COLUMN IF EXISTS request_model",
    "ALTER TABLE public.rate_limit_observations DROP COLUMN IF EXISTS response_model",
    "ALTER TABLE public.rate_limit_observations DROP COLUMN IF EXISTS client_name",
    "ALTER TABLE public.rate_limit_observations DROP COLUMN IF EXISTS client_user_agent",
    "ALTER TABLE public.rate_limit_observations DROP COLUMN IF EXISTS metadata",
    "DELETE FROM public.rate_limit_observations WHERE source LIKE 'claude_statusline%'",
    """
DELETE FROM public.rate_limit_observations AS doomed
USING (
    SELECT
        id,
        LAG(expected_reset_at) OVER identity_window AS previous_expected_reset_at,
        LAG(remaining_pct) OVER identity_window AS previous_remaining_pct
    FROM public.rate_limit_observations
    WINDOW identity_window AS (
        PARTITION BY
            provider,
            client,
            account_hash,
            quota_key,
            source,
            model,
            quota_period,
            quota_type
        ORDER BY observed_at ASC, id ASC
    )
) AS ranked
WHERE doomed.id = ranked.id
  AND (
      ranked.previous_expected_reset_at IS NOT DISTINCT FROM doomed.expected_reset_at
      OR (
          ranked.previous_expected_reset_at IS NOT NULL
          AND doomed.expected_reset_at IS NOT NULL
          AND ABS(EXTRACT(EPOCH FROM (doomed.expected_reset_at - ranked.previous_expected_reset_at))) < 900
      )
  )
  AND ranked.previous_remaining_pct IS NOT DISTINCT FROM doomed.remaining_pct;
""",
)
_AAWM_RATE_LIMIT_OBSERVATIONS_INDEX_STATEMENTS = (
    "CREATE INDEX IF NOT EXISTS rate_limit_observations_identity_latest_idx ON public.rate_limit_observations (provider, client, account_hash, quota_key, source, model, observed_at DESC)",
    "CREATE INDEX IF NOT EXISTS rate_limit_observations_quota_observed_idx ON public.rate_limit_observations (quota_key, observed_at DESC)",
    "CREATE INDEX IF NOT EXISTS rate_limit_observations_provider_quota_observed_idx ON public.rate_limit_observations (provider, quota_key, observed_at DESC) INCLUDE (expected_reset_at, remaining_pct, quota_type, model) WHERE remaining_pct >= 0",
    "CREATE INDEX IF NOT EXISTS rate_limit_observations_provider_model_observed_idx ON public.rate_limit_observations (provider, model, observed_at DESC)",
    "CREATE INDEX IF NOT EXISTS rate_limit_observations_provider_type_model_observed_idx ON public.rate_limit_observations (provider, quota_type, model, observed_at DESC) INCLUDE (expected_reset_at, remaining_pct, quota_key) WHERE remaining_pct >= 0",
    "CREATE INDEX IF NOT EXISTS rate_limit_observations_client_observed_idx ON public.rate_limit_observations (client, observed_at DESC)",
    "CREATE INDEX IF NOT EXISTS rate_limit_observations_reset_idx ON public.rate_limit_observations (expected_reset_at)",
    "CREATE INDEX IF NOT EXISTS rate_limit_observations_session_idx ON public.rate_limit_observations (session_id, observed_at DESC)",
    "CREATE INDEX IF NOT EXISTS rate_limit_observations_trace_call_idx ON public.rate_limit_observations (trace_id, litellm_call_id)",
)
_AAWM_OPENROUTER_FREE_DAILY_REQUEST_COUNT_SQL = """
SELECT COUNT(*)::integer
FROM public.session_history
WHERE provider = 'openrouter'
  AND lower(COALESCE(model, '')) LIKE '%:free'
  AND COALESCE(end_time, start_time, created_at) >= $1::timestamptz
  AND COALESCE(end_time, start_time, created_at) < $2::timestamptz
"""
_AAWM_RATE_LIMIT_TRANSITIONS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS public.rate_limit_transitions (
    id BIGSERIAL PRIMARY KEY,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    transition_key TEXT NOT NULL UNIQUE,
    limit_key TEXT NOT NULL,
    provider TEXT,
    client_family TEXT,
    account_hash TEXT,
    transition_type TEXT NOT NULL,
    confidence DOUBLE PRECISION NOT NULL DEFAULT 0,
    signals JSONB NOT NULL DEFAULT '[]'::jsonb,
    source TEXT,
    old_observed_at TIMESTAMPTZ,
    new_observed_at TIMESTAMPTZ NOT NULL,
    old_provider_resets_at TIMESTAMPTZ,
    new_provider_resets_at TIMESTAMPTZ,
    old_used_percentage DOUBLE PRECISION,
    new_used_percentage DOUBLE PRECISION,
    old_remaining_requests INTEGER,
    new_remaining_requests INTEGER,
    old_used_requests INTEGER,
    new_used_requests INTEGER,
    old_total_requests INTEGER,
    new_total_requests INTEGER,
    inferred_window_start_at TIMESTAMPTZ,
    detection_window_start_at TIMESTAMPTZ,
    detection_window_end_at TIMESTAMPTZ,
    session_usage_summary JSONB NOT NULL DEFAULT '{}'::jsonb,
    old_observation JSONB NOT NULL DEFAULT '{}'::jsonb,
    new_observation JSONB NOT NULL DEFAULT '{}'::jsonb,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb
)
"""
_AAWM_RATE_LIMIT_TRANSITIONS_ALTER_STATEMENTS = (
    "DELETE FROM public.rate_limit_transitions WHERE source LIKE 'claude_statusline%'",
)
_AAWM_RATE_LIMIT_TRANSITIONS_INDEX_STATEMENTS = (
    "CREATE INDEX IF NOT EXISTS rate_limit_transitions_limit_new_observed_idx ON public.rate_limit_transitions (limit_key, new_observed_at DESC)",
    "CREATE INDEX IF NOT EXISTS rate_limit_transitions_provider_client_idx ON public.rate_limit_transitions (provider, client_family, new_observed_at DESC)",
    "CREATE INDEX IF NOT EXISTS rate_limit_transitions_type_idx ON public.rate_limit_transitions (transition_type, new_observed_at DESC)",
)
_AAWM_PROVIDER_ERROR_OBSERVATIONS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS public.provider_error_observations (
    id BIGSERIAL PRIMARY KEY,
    observed_at TIMESTAMPTZ NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    environment TEXT,
    provider TEXT NOT NULL,
    model TEXT,
    model_group TEXT,
    route_family TEXT,
    status_code INTEGER,
    error_type TEXT,
    error_code TEXT,
    error_class TEXT NOT NULL,
    retry_after_seconds DOUBLE PRECISION,
    expected_reset_at TIMESTAMPTZ,
    session_id TEXT,
    trace_id TEXT,
    litellm_call_id TEXT,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb
)
"""
_AAWM_PROVIDER_ERROR_OBSERVATIONS_ALTER_STATEMENTS = (
    "ALTER TABLE public.provider_error_observations ADD COLUMN IF NOT EXISTS environment TEXT",
    "ALTER TABLE public.provider_error_observations ADD COLUMN IF NOT EXISTS model_group TEXT",
    "ALTER TABLE public.provider_error_observations ADD COLUMN IF NOT EXISTS route_family TEXT",
    "ALTER TABLE public.provider_error_observations ADD COLUMN IF NOT EXISTS retry_after_seconds DOUBLE PRECISION",
    "ALTER TABLE public.provider_error_observations ADD COLUMN IF NOT EXISTS expected_reset_at TIMESTAMPTZ",
    "ALTER TABLE public.provider_error_observations ADD COLUMN IF NOT EXISTS trace_id TEXT",
    "ALTER TABLE public.provider_error_observations ADD COLUMN IF NOT EXISTS litellm_call_id TEXT",
    "ALTER TABLE public.provider_error_observations ADD COLUMN IF NOT EXISTS metadata JSONB NOT NULL DEFAULT '{}'::jsonb",
)
_AAWM_PROVIDER_ERROR_OBSERVATIONS_INDEX_STATEMENTS = (
    "CREATE INDEX IF NOT EXISTS provider_error_observations_provider_time_idx ON public.provider_error_observations (provider, observed_at DESC)",
    "CREATE INDEX IF NOT EXISTS provider_error_observations_model_time_idx ON public.provider_error_observations (provider, model, observed_at DESC)",
    "CREATE INDEX IF NOT EXISTS provider_error_observations_class_time_idx ON public.provider_error_observations (error_class, observed_at DESC)",
    "CREATE INDEX IF NOT EXISTS provider_error_observations_trace_call_idx ON public.provider_error_observations (trace_id, litellm_call_id)",
)
_AAWM_PROVIDER_STATUS_OBSERVATIONS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS public.provider_status_observations (
    id BIGSERIAL PRIMARY KEY,
    observed_at TIMESTAMPTZ NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    environment TEXT NOT NULL,
    provider TEXT NOT NULL,
    endpoint_key TEXT NOT NULL,
    probe_type TEXT NOT NULL,
    success BOOLEAN NOT NULL,
    status_code INTEGER,
    address_family TEXT,
    resolved_ip TEXT,
    packet_loss_pct DOUBLE PRECISION,
    icmp_rtt_min_ms DOUBLE PRECISION,
    icmp_rtt_avg_ms DOUBLE PRECISION,
    icmp_rtt_max_ms DOUBLE PRECISION,
    icmp_rtt_mdev_ms DOUBLE PRECISION,
    dns_ms DOUBLE PRECISION,
    tcp_ms DOUBLE PRECISION,
    tls_ms DOUBLE PRECISION,
    ttfb_ms DOUBLE PRECISION,
    total_ms DOUBLE PRECISION,
    status_summary TEXT,
    error_class TEXT,
    error_message TEXT,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb
)
"""
_AAWM_PROVIDER_STATUS_OBSERVATIONS_ALTER_STATEMENTS = (
    "ALTER TABLE public.provider_status_observations ADD COLUMN IF NOT EXISTS address_family TEXT",
    "ALTER TABLE public.provider_status_observations ADD COLUMN IF NOT EXISTS resolved_ip TEXT",
    "ALTER TABLE public.provider_status_observations ADD COLUMN IF NOT EXISTS packet_loss_pct DOUBLE PRECISION",
    "ALTER TABLE public.provider_status_observations ADD COLUMN IF NOT EXISTS icmp_rtt_min_ms DOUBLE PRECISION",
    "ALTER TABLE public.provider_status_observations ADD COLUMN IF NOT EXISTS icmp_rtt_avg_ms DOUBLE PRECISION",
    "ALTER TABLE public.provider_status_observations ADD COLUMN IF NOT EXISTS icmp_rtt_max_ms DOUBLE PRECISION",
    "ALTER TABLE public.provider_status_observations ADD COLUMN IF NOT EXISTS icmp_rtt_mdev_ms DOUBLE PRECISION",
)
_AAWM_PROVIDER_STATUS_OBSERVATIONS_INDEX_STATEMENTS = (
    "CREATE INDEX IF NOT EXISTS provider_status_observations_provider_time_idx ON public.provider_status_observations (provider, observed_at DESC)",
    "CREATE INDEX IF NOT EXISTS provider_status_observations_endpoint_time_idx ON public.provider_status_observations (provider, endpoint_key, observed_at DESC)",
    "CREATE INDEX IF NOT EXISTS provider_status_observations_probe_time_idx ON public.provider_status_observations (probe_type, observed_at DESC)",
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
    created_at,
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
    invalid_tool_call_count,
    tool_names,
    file_read_count,
    file_modified_count,
    changed_pre_commit_config,
    changed_env_file,
    changed_pyproject_toml,
    changed_gitignore,
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
    client_ip,
    host_name,
    token_permission_input,
    token_permission_output,
    permission_usd_cost,
    metadata,
    repository,
    input_system_tokens_estimated,
    input_tool_advertisement_tokens_estimated,
    input_conversation_tokens_estimated,
    input_other_tokens_estimated,
    input_breakdown_residual_tokens,
    system_behavior_tokens_estimated,
    system_safety_tokens_estimated,
    system_instructional_tokens_estimated,
    system_unclassified_tokens_estimated,
    litellm_processing_ms,
    llm_upstream_elapsed_ms,
    total_server_elapsed_ms,
    ttft_ms,
    litellm_pre_send_ms,
    litellm_post_response_ms,
    llm_upstream_time_to_first_byte_ms,
    llm_upstream_stream_ms,
    latency_unclassified_ms,
    previous_response_to_current_request_ms,
    structured_output_attempted,
    structured_output_failed,
    structured_output_mode,
    structured_output_schema_hash,
    structured_output_failure_reason,
    trace_quality_score,
    empty_completion_failure,
    large_tool_result_payload_risk,
    destructive_checkout_after_work,
    invalid_tool_call_error,
    read_only_policy_compliance_score,
    read_only_policy_violation_count,
    response_meaningfulness_score,
    instruction_adherence_score,
    answer_completeness_score,
    evidence_fidelity_score,
    tool_result_fidelity_score,
    error_attribution_quality_score,
    repetition_loop_risk_score,
    context_retention_score,
    tool_use_validity_score,
    tool_error_recovery_score,
    stall_risk_score,
    output_contract_compliance_score,
    task_progress_score,
    scope_control_score,
    destructive_action_policy_score,
    ignored_path_tracking_policy_score,
    ignored_path_tracking_violation_count,
    baseline_deflection_attempted_score,
    baseline_deflection_incident_score,
    baseline_deflection_attempt_count,
    baseline_deflection_tool_call_count,
    baseline_deflection_input_tokens,
    baseline_deflection_elapsed_ms,
    quality_gate_trigger_count,
    quality_gate_fix_attempt_count,
    quality_gate_rerun_count,
    sleep_wellness_interruption_attempted_score,
    sleep_wellness_interruption_incident_score,
    sleep_wellness_interruption_count,
    sleep_wellness_interruption_output_tokens,
    sleep_wellness_interruption_input_tokens,
    sleep_wellness_interruption_elapsed_ms,
    sleep_wellness_interruption_after_user_pushback_count,
    sleep_wellness_interruption_repeated_count,
    terminal_completion_score,
    discovery_inventory_coverage_score,
    discovery_inventory_missing_count,
    agent_score_reasons,
    is_compact_summary,
    compact_summary_source,
    compact_summary_id,
    compact_summary_role,
    inbound_model_alias,
    agent_id
) VALUES (
    $1, $2, $3, $4, $5, $6, $7, $8, $9, $10,
    $11, COALESCE($11, $12, NOW()), $12, $13, $14, $15, $16, $17, $18, $19, $20,
    $21, $22, $23, $24, $25, $26, $27, $28, $29, $30, $31::jsonb,
    $32, $33, $34, $35, $36, $37, $38, $39, $40, $41, $42, $43, $44::jsonb, $45, $46, $47, $48, $49, $50, $51, $52, $53::jsonb, $54,
    $55, $56, $57, $58, $59, $60, $61, $62, $63,
    $64, $65, $66, $67, $68, $69, $70, $71, $72, $73,
    $74, $75, $76, $77, $78,
    $79, $80, $81, $82, $83, $84, $85, $86, $87, $88,
    $89, $90, $91, $92, $93, $94, $95, $96, $97, $98,
    $99, $100, $101, $102,
    $103, $104, $105, $106, $107, $108, $109, $110, $111, $112,
    $113, $114, $115, $116, $117, $118, $119, $120, $121, $122, $123::jsonb,
    $124, $125, $126, $127, $128, $129
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
    inbound_model_alias = COALESCE(
        NULLIF(EXCLUDED.inbound_model_alias, ''),
        session_history.inbound_model_alias
    ),
    model_group = COALESCE(NULLIF(EXCLUDED.model_group, ''), session_history.model_group),
    agent_name = COALESCE(NULLIF(EXCLUDED.agent_name, ''), session_history.agent_name),
    agent_id = COALESCE(NULLIF(EXCLUDED.agent_id, ''), session_history.agent_id),
    tenant_id = COALESCE(NULLIF(EXCLUDED.tenant_id, ''), session_history.tenant_id),
    call_type = COALESCE(NULLIF(EXCLUDED.call_type, ''), session_history.call_type),
    created_at = LEAST(session_history.created_at, EXCLUDED.created_at),
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
    invalid_tool_call_count = GREATEST(
        session_history.invalid_tool_call_count,
        EXCLUDED.invalid_tool_call_count
    ),
    structured_output_attempted = session_history.structured_output_attempted OR EXCLUDED.structured_output_attempted,
    structured_output_failed = session_history.structured_output_failed OR EXCLUDED.structured_output_failed,
    structured_output_mode = COALESCE(
        NULLIF(EXCLUDED.structured_output_mode, ''),
        session_history.structured_output_mode
    ),
    structured_output_schema_hash = COALESCE(
        NULLIF(EXCLUDED.structured_output_schema_hash, ''),
        session_history.structured_output_schema_hash
    ),
    structured_output_failure_reason = COALESCE(
        NULLIF(EXCLUDED.structured_output_failure_reason, ''),
        session_history.structured_output_failure_reason
    ),
    trace_quality_score = COALESCE(
        EXCLUDED.trace_quality_score,
        session_history.trace_quality_score
    ),
    empty_completion_failure = COALESCE(
        EXCLUDED.empty_completion_failure,
        session_history.empty_completion_failure
    ),
    large_tool_result_payload_risk = COALESCE(
        EXCLUDED.large_tool_result_payload_risk,
        session_history.large_tool_result_payload_risk
    ),
    destructive_checkout_after_work = COALESCE(
        EXCLUDED.destructive_checkout_after_work,
        session_history.destructive_checkout_after_work
    ),
    invalid_tool_call_error = COALESCE(
        EXCLUDED.invalid_tool_call_error,
        session_history.invalid_tool_call_error
    ),
    read_only_policy_compliance_score = COALESCE(
        EXCLUDED.read_only_policy_compliance_score,
        session_history.read_only_policy_compliance_score
    ),
    read_only_policy_violation_count = COALESCE(
        EXCLUDED.read_only_policy_violation_count,
        session_history.read_only_policy_violation_count
    ),
    response_meaningfulness_score = COALESCE(
        EXCLUDED.response_meaningfulness_score,
        session_history.response_meaningfulness_score
    ),
    instruction_adherence_score = COALESCE(
        EXCLUDED.instruction_adherence_score,
        session_history.instruction_adherence_score
    ),
    answer_completeness_score = COALESCE(
        EXCLUDED.answer_completeness_score,
        session_history.answer_completeness_score
    ),
    evidence_fidelity_score = COALESCE(
        EXCLUDED.evidence_fidelity_score,
        session_history.evidence_fidelity_score
    ),
    tool_result_fidelity_score = COALESCE(
        EXCLUDED.tool_result_fidelity_score,
        session_history.tool_result_fidelity_score
    ),
    error_attribution_quality_score = COALESCE(
        EXCLUDED.error_attribution_quality_score,
        session_history.error_attribution_quality_score
    ),
    repetition_loop_risk_score = COALESCE(
        EXCLUDED.repetition_loop_risk_score,
        session_history.repetition_loop_risk_score
    ),
    context_retention_score = COALESCE(
        EXCLUDED.context_retention_score,
        session_history.context_retention_score
    ),
    tool_use_validity_score = COALESCE(
        EXCLUDED.tool_use_validity_score,
        session_history.tool_use_validity_score
    ),
    tool_error_recovery_score = COALESCE(
        EXCLUDED.tool_error_recovery_score,
        session_history.tool_error_recovery_score
    ),
    stall_risk_score = COALESCE(
        EXCLUDED.stall_risk_score,
        session_history.stall_risk_score
    ),
    output_contract_compliance_score = COALESCE(
        EXCLUDED.output_contract_compliance_score,
        session_history.output_contract_compliance_score
    ),
    task_progress_score = COALESCE(
        EXCLUDED.task_progress_score,
        session_history.task_progress_score
    ),
    scope_control_score = COALESCE(
        EXCLUDED.scope_control_score,
        session_history.scope_control_score
    ),
    destructive_action_policy_score = COALESCE(
        EXCLUDED.destructive_action_policy_score,
        session_history.destructive_action_policy_score
    ),
    ignored_path_tracking_policy_score = COALESCE(
        EXCLUDED.ignored_path_tracking_policy_score,
        session_history.ignored_path_tracking_policy_score
    ),
    ignored_path_tracking_violation_count = COALESCE(
        EXCLUDED.ignored_path_tracking_violation_count,
        session_history.ignored_path_tracking_violation_count
    ),
    baseline_deflection_attempted_score = COALESCE(
        EXCLUDED.baseline_deflection_attempted_score,
        session_history.baseline_deflection_attempted_score
    ),
    baseline_deflection_incident_score = COALESCE(
        EXCLUDED.baseline_deflection_incident_score,
        session_history.baseline_deflection_incident_score
    ),
    baseline_deflection_attempt_count = COALESCE(
        EXCLUDED.baseline_deflection_attempt_count,
        session_history.baseline_deflection_attempt_count
    ),
    baseline_deflection_tool_call_count = COALESCE(
        EXCLUDED.baseline_deflection_tool_call_count,
        session_history.baseline_deflection_tool_call_count
    ),
    baseline_deflection_input_tokens = COALESCE(
        EXCLUDED.baseline_deflection_input_tokens,
        session_history.baseline_deflection_input_tokens
    ),
    baseline_deflection_elapsed_ms = COALESCE(
        EXCLUDED.baseline_deflection_elapsed_ms,
        session_history.baseline_deflection_elapsed_ms
    ),
    quality_gate_trigger_count = COALESCE(
        EXCLUDED.quality_gate_trigger_count,
        session_history.quality_gate_trigger_count
    ),
    quality_gate_fix_attempt_count = COALESCE(
        EXCLUDED.quality_gate_fix_attempt_count,
        session_history.quality_gate_fix_attempt_count
    ),
    quality_gate_rerun_count = COALESCE(
        EXCLUDED.quality_gate_rerun_count,
        session_history.quality_gate_rerun_count
    ),
    sleep_wellness_interruption_attempted_score = COALESCE(
        EXCLUDED.sleep_wellness_interruption_attempted_score,
        session_history.sleep_wellness_interruption_attempted_score
    ),
    sleep_wellness_interruption_incident_score = COALESCE(
        EXCLUDED.sleep_wellness_interruption_incident_score,
        session_history.sleep_wellness_interruption_incident_score
    ),
    sleep_wellness_interruption_count = COALESCE(
        EXCLUDED.sleep_wellness_interruption_count,
        session_history.sleep_wellness_interruption_count
    ),
    sleep_wellness_interruption_output_tokens = COALESCE(
        EXCLUDED.sleep_wellness_interruption_output_tokens,
        session_history.sleep_wellness_interruption_output_tokens
    ),
    sleep_wellness_interruption_input_tokens = COALESCE(
        EXCLUDED.sleep_wellness_interruption_input_tokens,
        session_history.sleep_wellness_interruption_input_tokens
    ),
    sleep_wellness_interruption_elapsed_ms = COALESCE(
        EXCLUDED.sleep_wellness_interruption_elapsed_ms,
        session_history.sleep_wellness_interruption_elapsed_ms
    ),
    sleep_wellness_interruption_after_user_pushback_count = COALESCE(
        EXCLUDED.sleep_wellness_interruption_after_user_pushback_count,
        session_history.sleep_wellness_interruption_after_user_pushback_count
    ),
    sleep_wellness_interruption_repeated_count = COALESCE(
        EXCLUDED.sleep_wellness_interruption_repeated_count,
        session_history.sleep_wellness_interruption_repeated_count
    ),
    terminal_completion_score = COALESCE(
        EXCLUDED.terminal_completion_score,
        session_history.terminal_completion_score
    ),
    discovery_inventory_coverage_score = COALESCE(
        EXCLUDED.discovery_inventory_coverage_score,
        session_history.discovery_inventory_coverage_score
    ),
    discovery_inventory_missing_count = COALESCE(
        EXCLUDED.discovery_inventory_missing_count,
        session_history.discovery_inventory_missing_count
    ),
    agent_score_reasons = COALESCE(
        session_history.agent_score_reasons,
        '{}'::jsonb
    ) || COALESCE(EXCLUDED.agent_score_reasons, '{}'::jsonb),
    is_compact_summary = session_history.is_compact_summary OR EXCLUDED.is_compact_summary,
    compact_summary_source = COALESCE(
        NULLIF(EXCLUDED.compact_summary_source, ''),
        session_history.compact_summary_source
    ),
    compact_summary_role = COALESCE(
        NULLIF(EXCLUDED.compact_summary_role, ''),
        session_history.compact_summary_role
    ),
    compact_summary_id = COALESCE(
        NULLIF(EXCLUDED.compact_summary_id, ''),
        session_history.compact_summary_id
    ),
    tool_names = CASE
        WHEN jsonb_array_length(
            CASE
                WHEN jsonb_typeof(EXCLUDED.tool_names) = 'array'
                    THEN EXCLUDED.tool_names
                ELSE '[]'::jsonb
            END
        ) > jsonb_array_length(
            CASE
                WHEN jsonb_typeof(session_history.tool_names) = 'array'
                    THEN session_history.tool_names
                ELSE '[]'::jsonb
            END
        )
            THEN EXCLUDED.tool_names
        ELSE session_history.tool_names
    END,
    file_read_count = GREATEST(session_history.file_read_count, EXCLUDED.file_read_count),
    file_modified_count = GREATEST(session_history.file_modified_count, EXCLUDED.file_modified_count),
    changed_pre_commit_config = CASE
        WHEN session_history.changed_pre_commit_config IS NULL
            AND EXCLUDED.changed_pre_commit_config IS NULL
            THEN NULL
        ELSE COALESCE(session_history.changed_pre_commit_config, FALSE)
            OR COALESCE(EXCLUDED.changed_pre_commit_config, FALSE)
    END,
    changed_env_file = CASE
        WHEN session_history.changed_env_file IS NULL
            AND EXCLUDED.changed_env_file IS NULL
            THEN NULL
        ELSE COALESCE(session_history.changed_env_file, FALSE)
            OR COALESCE(EXCLUDED.changed_env_file, FALSE)
    END,
    changed_pyproject_toml = CASE
        WHEN session_history.changed_pyproject_toml IS NULL
            AND EXCLUDED.changed_pyproject_toml IS NULL
            THEN NULL
        ELSE COALESCE(session_history.changed_pyproject_toml, FALSE)
            OR COALESCE(EXCLUDED.changed_pyproject_toml, FALSE)
    END,
    changed_gitignore = CASE
        WHEN session_history.changed_gitignore IS NULL
            AND EXCLUDED.changed_gitignore IS NULL
            THEN NULL
        ELSE COALESCE(session_history.changed_gitignore, FALSE)
            OR COALESCE(EXCLUDED.changed_gitignore, FALSE)
    END,
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
    client_ip = COALESCE(NULLIF(EXCLUDED.client_ip, ''), session_history.client_ip),
    host_name = COALESCE(NULLIF(EXCLUDED.host_name, ''), session_history.host_name),
    repository = COALESCE(NULLIF(EXCLUDED.repository, ''), session_history.repository),
    input_system_tokens_estimated = CASE
        WHEN EXCLUDED.input_tokens >= session_history.input_tokens THEN EXCLUDED.input_system_tokens_estimated
        ELSE session_history.input_system_tokens_estimated
    END,
    input_tool_advertisement_tokens_estimated = CASE
        WHEN EXCLUDED.input_tokens >= session_history.input_tokens THEN EXCLUDED.input_tool_advertisement_tokens_estimated
        ELSE session_history.input_tool_advertisement_tokens_estimated
    END,
    input_conversation_tokens_estimated = CASE
        WHEN EXCLUDED.input_tokens >= session_history.input_tokens THEN EXCLUDED.input_conversation_tokens_estimated
        ELSE session_history.input_conversation_tokens_estimated
    END,
    input_other_tokens_estimated = CASE
        WHEN EXCLUDED.input_tokens >= session_history.input_tokens THEN EXCLUDED.input_other_tokens_estimated
        ELSE session_history.input_other_tokens_estimated
    END,
    input_breakdown_residual_tokens = CASE
        WHEN EXCLUDED.input_tokens >= session_history.input_tokens THEN EXCLUDED.input_breakdown_residual_tokens
        ELSE session_history.input_breakdown_residual_tokens
    END,
    system_behavior_tokens_estimated = CASE
        WHEN EXCLUDED.input_tokens >= session_history.input_tokens THEN EXCLUDED.system_behavior_tokens_estimated
        ELSE session_history.system_behavior_tokens_estimated
    END,
    system_safety_tokens_estimated = CASE
        WHEN EXCLUDED.input_tokens >= session_history.input_tokens THEN EXCLUDED.system_safety_tokens_estimated
        ELSE session_history.system_safety_tokens_estimated
    END,
    system_instructional_tokens_estimated = CASE
        WHEN EXCLUDED.input_tokens >= session_history.input_tokens THEN EXCLUDED.system_instructional_tokens_estimated
        ELSE session_history.system_instructional_tokens_estimated
    END,
    system_unclassified_tokens_estimated = CASE
        WHEN EXCLUDED.input_tokens >= session_history.input_tokens THEN EXCLUDED.system_unclassified_tokens_estimated
        ELSE session_history.system_unclassified_tokens_estimated
    END,
    litellm_processing_ms = COALESCE(
        EXCLUDED.litellm_processing_ms,
        session_history.litellm_processing_ms
    ),
    llm_upstream_elapsed_ms = COALESCE(
        EXCLUDED.llm_upstream_elapsed_ms,
        session_history.llm_upstream_elapsed_ms
    ),
    total_server_elapsed_ms = COALESCE(
        EXCLUDED.total_server_elapsed_ms,
        session_history.total_server_elapsed_ms
    ),
    ttft_ms = COALESCE(EXCLUDED.ttft_ms, session_history.ttft_ms),
    litellm_pre_send_ms = COALESCE(
        EXCLUDED.litellm_pre_send_ms,
        session_history.litellm_pre_send_ms
    ),
    litellm_post_response_ms = COALESCE(
        EXCLUDED.litellm_post_response_ms,
        session_history.litellm_post_response_ms
    ),
    llm_upstream_time_to_first_byte_ms = COALESCE(
        EXCLUDED.llm_upstream_time_to_first_byte_ms,
        session_history.llm_upstream_time_to_first_byte_ms
    ),
    llm_upstream_stream_ms = COALESCE(
        EXCLUDED.llm_upstream_stream_ms,
        session_history.llm_upstream_stream_ms
    ),
    latency_unclassified_ms = COALESCE(
        EXCLUDED.latency_unclassified_ms,
        session_history.latency_unclassified_ms
    ),
    previous_response_to_current_request_ms = COALESCE(
        EXCLUDED.previous_response_to_current_request_ms,
        session_history.previous_response_to_current_request_ms
    ),
    metadata = COALESCE(session_history.metadata, '{}'::jsonb) || COALESCE(EXCLUDED.metadata, '{}'::jsonb)
"""
_AAWM_CLAUDE_AUTO_REVIEW_PARENT_IDENTITY_SQL = """
SELECT
    id,
    repository,
    tenant_id,
    agent_name,
    metadata,
    COALESCE(start_time, created_at) AS row_time
FROM public.session_history
WHERE session_id = $1
  AND COALESCE(start_time, created_at)
      BETWEEN COALESCE($2::timestamptz, NOW()) - INTERVAL '30 minutes'
          AND COALESCE($2::timestamptz, NOW()) + INTERVAL '5 minutes'
  AND provider IS NOT DISTINCT FROM 'anthropic'
  AND model IS DISTINCT FROM 'claude-auto-review'
  AND COALESCE(LOWER(metadata->>'claude_permission_check'), '') NOT IN ('1', 'true', 'yes', 'y')
  AND metadata::text NOT ILIKE '%claude-permission-check%'
ORDER BY
    CASE WHEN metadata::text ILIKE '%claude-project:%' THEN 0 ELSE 1 END,
    CASE WHEN repository IS NOT NULL THEN 0 ELSE 1 END,
    CASE WHEN COALESCE(metadata->>'trace_name', '') = 'claude-code.orchestrator' THEN 0 ELSE 1 END,
    COALESCE(start_time, created_at) DESC,
    id DESC
LIMIT 10
"""
_SESSION_HISTORY_PREVIOUS_GAP_FIELD = "previous_response_to_current_request_ms"
_AAWM_SESSION_HISTORY_PREVIOUS_GAP_UPDATE_SQL = f"""
WITH inserted AS (
    SELECT
        id,
        session_id,
        COALESCE(start_time, created_at) AS current_started_at
    FROM public.session_history
    WHERE litellm_call_id = ANY($1::text[])
),
affected AS (
    SELECT id
    FROM inserted
    UNION
    SELECT next_sh.id
    FROM inserted
    JOIN LATERAL (
        SELECT sh.id
        FROM public.session_history sh
        WHERE sh.session_id = inserted.session_id
          AND (COALESCE(sh.start_time, sh.created_at), sh.id)
              > (inserted.current_started_at, inserted.id)
        ORDER BY COALESCE(sh.start_time, sh.created_at) ASC, sh.id ASC
        LIMIT 1
    ) next_sh ON TRUE
),
target AS (
    SELECT
        sh.id,
        sh.session_id,
        COALESCE(sh.start_time, sh.created_at) AS current_started_at
    FROM public.session_history sh
    JOIN affected ON affected.id = sh.id
),
derived AS (
    SELECT
        target.id,
        CASE
            WHEN previous.end_time IS NULL THEN NULL
            WHEN target.current_started_at >= previous.end_time THEN
                EXTRACT(EPOCH FROM (target.current_started_at - previous.end_time)) * 1000.0
            ELSE NULL
        END AS {_SESSION_HISTORY_PREVIOUS_GAP_FIELD}
    FROM target
    LEFT JOIN LATERAL (
        SELECT sh.end_time
        FROM public.session_history sh
        WHERE sh.session_id = target.session_id
          AND (COALESCE(sh.start_time, sh.created_at), sh.id)
              < (target.current_started_at, target.id)
        ORDER BY COALESCE(sh.start_time, sh.created_at) DESC, sh.id DESC
        LIMIT 1
    ) previous ON TRUE
)
UPDATE public.session_history AS sh
SET {_SESSION_HISTORY_PREVIOUS_GAP_FIELD} = derived.{_SESSION_HISTORY_PREVIOUS_GAP_FIELD}
FROM derived
WHERE sh.id = derived.id
  AND sh.{_SESSION_HISTORY_PREVIOUS_GAP_FIELD}
      IS DISTINCT FROM derived.{_SESSION_HISTORY_PREVIOUS_GAP_FIELD}
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
    metadata,
    agent_id
) VALUES (
    $1, $2, $3, $4, $5, $6, $7, $8, $9, $10,
    $11::jsonb, $12::jsonb, $13, $14, $15, $16::jsonb, $17::jsonb, $18
)
ON CONFLICT (litellm_call_id, tool_index) DO UPDATE SET
    session_id = COALESCE(NULLIF(EXCLUDED.session_id, ''), session_history_tool_activity.session_id),
    trace_id = COALESCE(NULLIF(EXCLUDED.trace_id, ''), session_history_tool_activity.trace_id),
    provider = COALESCE(NULLIF(EXCLUDED.provider, ''), session_history_tool_activity.provider),
    model = COALESCE(NULLIF(EXCLUDED.model, ''), session_history_tool_activity.model),
    agent_name = COALESCE(NULLIF(EXCLUDED.agent_name, ''), session_history_tool_activity.agent_name),
    agent_id = COALESCE(NULLIF(EXCLUDED.agent_id, ''), session_history_tool_activity.agent_id),
    tool_call_id = COALESCE(NULLIF(EXCLUDED.tool_call_id, ''), session_history_tool_activity.tool_call_id),
    tool_name = COALESCE(NULLIF(EXCLUDED.tool_name, ''), session_history_tool_activity.tool_name),
    tool_kind = COALESCE(NULLIF(EXCLUDED.tool_kind, ''), session_history_tool_activity.tool_kind),
    file_paths_read = CASE
        WHEN jsonb_array_length(
            CASE
                WHEN jsonb_typeof(EXCLUDED.file_paths_read) = 'array'
                    THEN EXCLUDED.file_paths_read
                ELSE '[]'::jsonb
            END
        ) > jsonb_array_length(
            CASE
                WHEN jsonb_typeof(session_history_tool_activity.file_paths_read) = 'array'
                    THEN session_history_tool_activity.file_paths_read
                ELSE '[]'::jsonb
            END
        )
            THEN EXCLUDED.file_paths_read
        ELSE session_history_tool_activity.file_paths_read
    END,
    file_paths_modified = CASE
        WHEN jsonb_array_length(
            CASE
                WHEN jsonb_typeof(EXCLUDED.file_paths_modified) = 'array'
                    THEN EXCLUDED.file_paths_modified
                ELSE '[]'::jsonb
            END
        ) > jsonb_array_length(
            CASE
                WHEN jsonb_typeof(session_history_tool_activity.file_paths_modified) = 'array'
                    THEN session_history_tool_activity.file_paths_modified
                ELSE '[]'::jsonb
            END
        )
            THEN EXCLUDED.file_paths_modified
        ELSE session_history_tool_activity.file_paths_modified
    END,
    git_commit_count = GREATEST(session_history_tool_activity.git_commit_count, EXCLUDED.git_commit_count),
    git_push_count = GREATEST(session_history_tool_activity.git_push_count, EXCLUDED.git_push_count),
    command_text = COALESCE(NULLIF(EXCLUDED.command_text, ''), session_history_tool_activity.command_text),
    arguments = COALESCE(session_history_tool_activity.arguments, '{}'::jsonb) || COALESCE(EXCLUDED.arguments, '{}'::jsonb),
    metadata = COALESCE(session_history_tool_activity.metadata, '{}'::jsonb) || COALESCE(EXCLUDED.metadata, '{}'::jsonb)
"""
_AAWM_SESSION_HISTORY_TOOL_DEFINITION_SNAPSHOT_INSERT_SQL = """
INSERT INTO public.session_history_tool_definition_snapshots (
    session_id,
    snapshot_hash,
    capture_version,
    capture_source,
    tool_definition_count,
    captured_count,
    tool_definition_sources,
    tool_definition_names,
    tool_definition_types,
    snapshot_truncated,
    sanitized_snapshot,
    first_litellm_call_id,
    first_trace_id,
    metadata
) VALUES (
    $1::text, $2::text, $3::text, $4::text, $5::integer, $6::integer,
    $7::jsonb, $8::jsonb, $9::jsonb, $10::boolean, $11::jsonb,
    $12::text, $13::text, $14::jsonb
)
ON CONFLICT (session_id, snapshot_hash) DO UPDATE SET
    updated_at = NOW(),
    capture_version = COALESCE(
        NULLIF(EXCLUDED.capture_version, ''),
        session_history_tool_definition_snapshots.capture_version
    ),
    capture_source = COALESCE(
        NULLIF(EXCLUDED.capture_source, ''),
        session_history_tool_definition_snapshots.capture_source
    ),
    tool_definition_count = GREATEST(
        COALESCE(session_history_tool_definition_snapshots.tool_definition_count, 0),
        COALESCE(EXCLUDED.tool_definition_count, 0)
    ),
    captured_count = GREATEST(
        COALESCE(session_history_tool_definition_snapshots.captured_count, 0),
        COALESCE(EXCLUDED.captured_count, 0)
    ),
    tool_definition_sources = CASE
        WHEN jsonb_array_length(EXCLUDED.tool_definition_sources)
             > jsonb_array_length(session_history_tool_definition_snapshots.tool_definition_sources)
            THEN EXCLUDED.tool_definition_sources
        ELSE session_history_tool_definition_snapshots.tool_definition_sources
    END,
    tool_definition_names = CASE
        WHEN jsonb_array_length(EXCLUDED.tool_definition_names)
             > jsonb_array_length(session_history_tool_definition_snapshots.tool_definition_names)
            THEN EXCLUDED.tool_definition_names
        ELSE session_history_tool_definition_snapshots.tool_definition_names
    END,
    tool_definition_types = CASE
        WHEN jsonb_array_length(EXCLUDED.tool_definition_types)
             > jsonb_array_length(session_history_tool_definition_snapshots.tool_definition_types)
            THEN EXCLUDED.tool_definition_types
        ELSE session_history_tool_definition_snapshots.tool_definition_types
    END,
    snapshot_truncated = (
        session_history_tool_definition_snapshots.snapshot_truncated
        OR EXCLUDED.snapshot_truncated
    ),
    sanitized_snapshot = CASE
        WHEN jsonb_typeof(session_history_tool_definition_snapshots.sanitized_snapshot) = 'array'
             AND jsonb_array_length(session_history_tool_definition_snapshots.sanitized_snapshot) > 0
            THEN session_history_tool_definition_snapshots.sanitized_snapshot
        ELSE EXCLUDED.sanitized_snapshot
    END,
    first_litellm_call_id = COALESCE(
        NULLIF(session_history_tool_definition_snapshots.first_litellm_call_id, ''),
        NULLIF(EXCLUDED.first_litellm_call_id, '')
    ),
    first_trace_id = COALESCE(
        NULLIF(session_history_tool_definition_snapshots.first_trace_id, ''),
        NULLIF(EXCLUDED.first_trace_id, '')
    ),
    metadata = (
        COALESCE(session_history_tool_definition_snapshots.metadata, '{}'::jsonb)
        || COALESCE(EXCLUDED.metadata, '{}'::jsonb)
    )
"""
_AAWM_RATE_LIMIT_OBSERVATION_INSERT_SQL = """
WITH candidate AS (
    SELECT
        $1::timestamptz AS observed_at,
        $2::text AS client,
        $3::text AS client_version,
        $4::text AS account_hash,
        $5::text AS provider,
        $6::text AS model,
        $7::text AS quota_key,
        $8::text AS quota_period,
        $9::text AS quota_type,
        $10::timestamptz AS expected_reset_at,
        $11::double precision AS remaining_pct,
        $12::double precision AS quota_limit,
        $13::double precision AS quota_used,
        $14::double precision AS quota_remaining,
        $15::timestamptz AS billing_period_start_at,
        $16::timestamptz AS billing_period_end_at,
        $17::jsonb AS raw_provider_fields,
        $18::jsonb AS evidence,
        $19::text AS source,
        $20::text AS session_id,
        $21::text AS trace_id,
        $22::text AS litellm_call_id
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
            latest.raw_provider_fields
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
)
"""
_AAWM_RATE_LIMIT_PREVIOUS_OBSERVATION_SQL = """
SELECT
    observed_at,
    source,
    provider,
    client AS client_family,
    account_hash,
    NULL::text AS environment,
    NULL::text AS tenant_id,
    NULL::text AS repository,
    quota_key AS limit_key,
    quota_key AS limit_id,
    quota_key AS limit_name,
    quota_type AS limit_scope,
    NULL::integer AS window_minutes,
    quota_period,
    expected_reset_at AS provider_resets_at,
    NULL::timestamptz AS inferred_window_start_at,
    CASE
        WHEN remaining_pct IS NULL THEN NULL
        ELSE GREATEST(0.0, LEAST(100.0, 100.0 - remaining_pct))
    END AS used_percentage,
    NULL::integer AS remaining_requests,
    NULL::integer AS used_requests,
    NULL::integer AS total_requests,
    CASE WHEN remaining_pct <= 0 THEN 'exhausted' ELSE 'observed' END AS status,
    COALESCE(remaining_pct <= 0, FALSE) AS exhausted,
    NULL::text AS exhaustion_kind,
    NULL::integer AS reset_hint_seconds,
    model,
    quota_limit,
    quota_used,
    quota_remaining,
    billing_period_start_at,
    billing_period_end_at,
    NULL::text AS model_family,
    NULL::text AS model_tier,
    NULL::text AS parent_limit_key,
    session_id,
    trace_id,
    litellm_call_id,
    NULL::text AS route_family,
    NULL::text AS request_model,
    NULL::text AS response_model,
    client AS client_name,
    client_version,
    NULL::text AS client_user_agent,
    COALESCE(raw_provider_fields, '{}'::jsonb) AS raw_provider_fields,
    COALESCE(evidence, '{}'::jsonb) AS evidence,
    '{}'::jsonb AS metadata
FROM public.rate_limit_observations
WHERE quota_key = $1
  AND provider = $2
  AND client IS NOT DISTINCT FROM $3::text
  AND account_hash IS NOT DISTINCT FROM $4::text
  AND source IS NOT DISTINCT FROM $5::text
  AND observed_at < $6
ORDER BY observed_at DESC, id DESC
LIMIT 1
"""
_AAWM_RATE_LIMIT_PREVIOUS_OBSERVATIONS_BATCH_SQL = """
WITH candidate AS (
    SELECT
        input.ordinal::bigint AS ordinal,
        input.quota_key::text AS quota_key,
        input.provider::text AS provider,
        input.client::text AS client,
        input.account_hash::text AS account_hash,
        input.source::text AS source,
        input.observed_at::timestamptz AS observed_at
    FROM unnest(
        $1::text[],
        $2::text[],
        $3::text[],
        $4::text[],
        $5::text[],
        $6::timestamptz[]
    ) WITH ORDINALITY AS input(
        quota_key,
        provider,
        client,
        account_hash,
        source,
        observed_at,
        ordinal
    )
)
SELECT
    candidate.quota_key AS input_limit_key,
    latest.observed_at,
    latest.source,
    latest.provider,
    latest.client AS client_family,
    latest.account_hash,
    NULL::text AS environment,
    NULL::text AS tenant_id,
    NULL::text AS repository,
    latest.quota_key AS limit_key,
    latest.quota_key AS limit_id,
    latest.quota_key AS limit_name,
    latest.quota_type AS limit_scope,
    NULL::integer AS window_minutes,
    latest.quota_period,
    latest.expected_reset_at AS provider_resets_at,
    NULL::timestamptz AS inferred_window_start_at,
    CASE
        WHEN latest.remaining_pct IS NULL THEN NULL
        ELSE GREATEST(0.0, LEAST(100.0, 100.0 - latest.remaining_pct))
    END AS used_percentage,
    NULL::integer AS remaining_requests,
    NULL::integer AS used_requests,
    NULL::integer AS total_requests,
    CASE WHEN latest.remaining_pct <= 0 THEN 'exhausted' ELSE 'observed' END AS status,
    COALESCE(latest.remaining_pct <= 0, FALSE) AS exhausted,
    NULL::text AS exhaustion_kind,
    NULL::integer AS reset_hint_seconds,
    latest.model,
    latest.quota_limit,
    latest.quota_used,
    latest.quota_remaining,
    latest.billing_period_start_at,
    latest.billing_period_end_at,
    NULL::text AS model_family,
    NULL::text AS model_tier,
    NULL::text AS parent_limit_key,
    latest.session_id,
    latest.trace_id,
    latest.litellm_call_id,
    NULL::text AS route_family,
    NULL::text AS request_model,
    NULL::text AS response_model,
    latest.client AS client_name,
    latest.client_version,
    NULL::text AS client_user_agent,
    COALESCE(latest.raw_provider_fields, '{}'::jsonb) AS raw_provider_fields,
    COALESCE(latest.evidence, '{}'::jsonb) AS evidence,
    '{}'::jsonb AS metadata
FROM candidate
JOIN LATERAL (
    SELECT *
    FROM public.rate_limit_observations AS previous
    WHERE previous.quota_key = candidate.quota_key
      AND previous.provider = candidate.provider
      AND previous.client IS NOT DISTINCT FROM candidate.client
      AND previous.account_hash IS NOT DISTINCT FROM candidate.account_hash
      AND previous.source IS NOT DISTINCT FROM candidate.source
      AND previous.observed_at < candidate.observed_at
    ORDER BY previous.observed_at DESC, previous.id DESC
    LIMIT 1
) AS latest ON TRUE
ORDER BY candidate.ordinal
"""
_AAWM_RATE_LIMIT_TRANSITION_INSERT_SQL = """
INSERT INTO public.rate_limit_transitions (
    transition_key,
    limit_key,
    provider,
    client_family,
    account_hash,
    transition_type,
    confidence,
    signals,
    source,
    old_observed_at,
    new_observed_at,
    old_provider_resets_at,
    new_provider_resets_at,
    old_used_percentage,
    new_used_percentage,
    old_remaining_requests,
    new_remaining_requests,
    old_used_requests,
    new_used_requests,
    old_total_requests,
    new_total_requests,
    inferred_window_start_at,
    detection_window_start_at,
    detection_window_end_at,
    session_usage_summary,
    old_observation,
    new_observation,
    metadata
) VALUES (
    $1, $2, $3, $4, $5, $6, $7, $8::jsonb, $9, $10,
    $11, $12, $13, $14, $15, $16, $17, $18, $19, $20,
    $21, $22, $23, $24, $25::jsonb, $26::jsonb, $27::jsonb, $28::jsonb
)
ON CONFLICT (transition_key) DO NOTHING
"""
_AAWM_PROVIDER_ERROR_OBSERVATION_INSERT_SQL = """
INSERT INTO public.provider_error_observations (
    observed_at,
    environment,
    provider,
    model,
    model_group,
    route_family,
    status_code,
    error_type,
    error_code,
    error_class,
    retry_after_seconds,
    expected_reset_at,
    session_id,
    trace_id,
    litellm_call_id,
    metadata
) SELECT
    $1::timestamptz,
    $2::text,
    $3::text,
    $4::text,
    $5::text,
    $6::text,
    $7::integer,
    $8::text,
    $9::text,
    $10::text,
    $11::double precision,
    $12::timestamptz,
    $13::text,
    $14::text,
    $15::text,
    $16::jsonb
WHERE NULLIF($15::text, '') IS NULL
OR NOT EXISTS (
    SELECT 1
    FROM public.provider_error_observations existing
    WHERE existing.litellm_call_id = NULLIF($15::text, '')
      AND existing.provider IS NOT DISTINCT FROM $3::text
      AND existing.route_family IS NOT DISTINCT FROM $6::text
      AND existing.status_code IS NOT DISTINCT FROM $7::integer
)
"""
_AAWM_ALIAS_ROUTING_AUDIT_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS public.aawm_alias_routing_audit (
    id BIGSERIAL PRIMARY KEY,
    event_key TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    observed_at TIMESTAMPTZ NOT NULL,
    session_id TEXT,
    session_key TEXT,
    trace_id TEXT,
    litellm_call_id TEXT,
    alias_model TEXT NOT NULL,
    alias_family TEXT NOT NULL,
    route_family TEXT,
    provider TEXT,
    model TEXT,
    lane_key TEXT,
    cooldown_key TEXT,
    attempt_number INTEGER,
    event_type TEXT NOT NULL,
    selection_reason TEXT,
    candidate_status TEXT,
    failure_class TEXT,
    error_status_code INTEGER,
    cooldown_scope TEXT,
    cooldown_seconds DOUBLE PRECISION,
    cooldown_until TIMESTAMPTZ,
    selected BOOLEAN NOT NULL DEFAULT FALSE,
    skipped BOOLEAN NOT NULL DEFAULT FALSE,
    last_resort BOOLEAN NOT NULL DEFAULT FALSE,
    in_flight_session BOOLEAN NOT NULL DEFAULT FALSE,
    redispatch_required BOOLEAN NOT NULL DEFAULT FALSE,
    redispatch_threshold_crossed BOOLEAN NOT NULL DEFAULT FALSE,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb
)
"""
_AAWM_ALIAS_ROUTING_AUDIT_INDEX_STATEMENTS = (
    "CREATE UNIQUE INDEX IF NOT EXISTS aawm_alias_routing_audit_event_key_idx "
    "ON public.aawm_alias_routing_audit (event_key) WHERE event_key IS NOT NULL",
    "CREATE INDEX IF NOT EXISTS aawm_alias_routing_audit_session_observed_idx "
    "ON public.aawm_alias_routing_audit (session_id, observed_at DESC)",
    "CREATE INDEX IF NOT EXISTS aawm_alias_routing_audit_alias_observed_idx "
    "ON public.aawm_alias_routing_audit (alias_model, observed_at DESC)",
    "CREATE INDEX IF NOT EXISTS aawm_alias_routing_audit_provider_model_observed_idx "
    "ON public.aawm_alias_routing_audit (provider, model, observed_at DESC)",
    "CREATE INDEX IF NOT EXISTS aawm_alias_routing_audit_event_observed_idx "
    "ON public.aawm_alias_routing_audit (event_type, observed_at DESC)",
    "CREATE INDEX IF NOT EXISTS aawm_alias_routing_audit_cooldown_observed_idx "
    "ON public.aawm_alias_routing_audit (cooldown_key, observed_at DESC)",
)
_AAWM_ALIAS_ROUTING_AUDIT_INSERT_SQL = """
INSERT INTO public.aawm_alias_routing_audit (
    event_key,
    observed_at,
    session_id,
    session_key,
    trace_id,
    litellm_call_id,
    alias_model,
    alias_family,
    route_family,
    provider,
    model,
    lane_key,
    cooldown_key,
    attempt_number,
    event_type,
    selection_reason,
    candidate_status,
    failure_class,
    error_status_code,
    cooldown_scope,
    cooldown_seconds,
    cooldown_until,
    selected,
    skipped,
    last_resort,
    in_flight_session,
    redispatch_required,
    redispatch_threshold_crossed,
    metadata
) VALUES (
    $1::text, $2::timestamptz, $3::text, $4::text, $5::text,
    $6::text, $7::text, $8::text, $9::text, $10::text,
    $11::text, $12::text, $13::text, $14::integer, $15::text,
    $16::text, $17::text, $18::text, $19::integer, $20::text,
    $21::double precision, $22::timestamptz, $23::boolean, $24::boolean,
    $25::boolean, $26::boolean, $27::boolean, $28::boolean, $29::jsonb
)
ON CONFLICT (event_key) WHERE event_key IS NOT NULL DO UPDATE SET
    observed_at = LEAST(aawm_alias_routing_audit.observed_at, EXCLUDED.observed_at),
    session_id = COALESCE(NULLIF(EXCLUDED.session_id, ''), aawm_alias_routing_audit.session_id),
    session_key = COALESCE(NULLIF(EXCLUDED.session_key, ''), aawm_alias_routing_audit.session_key),
    trace_id = COALESCE(NULLIF(EXCLUDED.trace_id, ''), aawm_alias_routing_audit.trace_id),
    litellm_call_id = COALESCE(NULLIF(EXCLUDED.litellm_call_id, ''), aawm_alias_routing_audit.litellm_call_id),
    alias_model = COALESCE(NULLIF(EXCLUDED.alias_model, ''), aawm_alias_routing_audit.alias_model),
    alias_family = COALESCE(NULLIF(EXCLUDED.alias_family, ''), aawm_alias_routing_audit.alias_family),
    route_family = COALESCE(NULLIF(EXCLUDED.route_family, ''), aawm_alias_routing_audit.route_family),
    provider = COALESCE(NULLIF(EXCLUDED.provider, ''), aawm_alias_routing_audit.provider),
    model = COALESCE(NULLIF(EXCLUDED.model, ''), aawm_alias_routing_audit.model),
    lane_key = COALESCE(NULLIF(EXCLUDED.lane_key, ''), aawm_alias_routing_audit.lane_key),
    cooldown_key = COALESCE(NULLIF(EXCLUDED.cooldown_key, ''), aawm_alias_routing_audit.cooldown_key),
    attempt_number = COALESCE(EXCLUDED.attempt_number, aawm_alias_routing_audit.attempt_number),
    event_type = COALESCE(NULLIF(EXCLUDED.event_type, ''), aawm_alias_routing_audit.event_type),
    selection_reason = COALESCE(NULLIF(EXCLUDED.selection_reason, ''), aawm_alias_routing_audit.selection_reason),
    candidate_status = COALESCE(NULLIF(EXCLUDED.candidate_status, ''), aawm_alias_routing_audit.candidate_status),
    failure_class = COALESCE(NULLIF(EXCLUDED.failure_class, ''), aawm_alias_routing_audit.failure_class),
    error_status_code = COALESCE(EXCLUDED.error_status_code, aawm_alias_routing_audit.error_status_code),
    cooldown_scope = COALESCE(NULLIF(EXCLUDED.cooldown_scope, ''), aawm_alias_routing_audit.cooldown_scope),
    cooldown_seconds = COALESCE(EXCLUDED.cooldown_seconds, aawm_alias_routing_audit.cooldown_seconds),
    cooldown_until = COALESCE(EXCLUDED.cooldown_until, aawm_alias_routing_audit.cooldown_until),
    selected = aawm_alias_routing_audit.selected OR EXCLUDED.selected,
    skipped = aawm_alias_routing_audit.skipped OR EXCLUDED.skipped,
    last_resort = aawm_alias_routing_audit.last_resort OR EXCLUDED.last_resort,
    in_flight_session = aawm_alias_routing_audit.in_flight_session OR EXCLUDED.in_flight_session,
    redispatch_required = aawm_alias_routing_audit.redispatch_required OR EXCLUDED.redispatch_required,
    redispatch_threshold_crossed = aawm_alias_routing_audit.redispatch_threshold_crossed OR EXCLUDED.redispatch_threshold_crossed,
    metadata = COALESCE(aawm_alias_routing_audit.metadata, '{}'::jsonb) || COALESCE(EXCLUDED.metadata, '{}'::jsonb)
"""
