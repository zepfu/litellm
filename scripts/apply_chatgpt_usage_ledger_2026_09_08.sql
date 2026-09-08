BEGIN;

CREATE TABLE IF NOT EXISTS public.chatgpt_usage_scopes (
    scope_key TEXT PRIMARY KEY,
    collector_account_id TEXT NOT NULL UNIQUE,
    provider TEXT NOT NULL,
    provider_user_id TEXT,
    workspace_id TEXT,
    quota_owner_id TEXT,
    surface TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS public.chatgpt_usage_attempts (
    attempt_id TEXT NOT NULL,
    scope_key TEXT NOT NULL REFERENCES public.chatgpt_usage_scopes(scope_key)
        ON DELETE CASCADE,
    collector_account_id TEXT NOT NULL,
    conversation_id TEXT NOT NULL,
    identity_basis TEXT NOT NULL,
    time_basis TEXT NOT NULL,
    attempt_time TIMESTAMPTZ,
    earliest_possible_at TIMESTAMPTZ,
    latest_possible_at TIMESTAMPTZ,
    requested_model_raw TEXT,
    requested_mode_raw TEXT,
    requested_reasoning_effort_raw TEXT,
    recorded_final_model_raw TEXT,
    resolved_model_raw TEXT,
    requested_family TEXT,
    recorded_final_family TEXT,
    resolved_family TEXT,
    mapping_version TEXT NOT NULL,
    outcome TEXT NOT NULL,
    completed_answer BOOLEAN NOT NULL,
    generation_started BOOLEAN NOT NULL,
    surface TEXT NOT NULL,
    origin TEXT,
    revision INTEGER NOT NULL DEFAULT 1,
    projection_fingerprint TEXT NOT NULL,
    warnings JSONB NOT NULL DEFAULT '[]'::jsonb,
    tombstone BOOLEAN NOT NULL DEFAULT FALSE,
    observed_at TIMESTAMPTZ NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL,
    PRIMARY KEY (scope_key, attempt_id)
);

CREATE TABLE IF NOT EXISTS public.chatgpt_usage_attempt_revisions (
    scope_key TEXT NOT NULL,
    attempt_id TEXT NOT NULL,
    revision INTEGER NOT NULL,
    projection_fingerprint TEXT NOT NULL,
    payload JSONB NOT NULL,
    source_kind TEXT NOT NULL,
    source_id TEXT NOT NULL,
    run_id TEXT NOT NULL,
    schema_version TEXT NOT NULL,
    collector_account_id TEXT NOT NULL,
    recorded_at TIMESTAMPTZ NOT NULL,
    PRIMARY KEY (scope_key, attempt_id, revision),
    UNIQUE (scope_key, attempt_id, projection_fingerprint),
    FOREIGN KEY (scope_key, attempt_id)
        REFERENCES public.chatgpt_usage_attempts(scope_key, attempt_id)
        ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS public.chatgpt_usage_attempt_aliases (
    scope_key TEXT NOT NULL REFERENCES public.chatgpt_usage_scopes(scope_key)
        ON DELETE CASCADE,
    alias_kind TEXT NOT NULL,
    alias_value TEXT NOT NULL,
    attempt_id TEXT NOT NULL,
    first_seen_at TIMESTAMPTZ NOT NULL,
    last_seen_at TIMESTAMPTZ NOT NULL,
    PRIMARY KEY (scope_key, alias_kind, alias_value),
    FOREIGN KEY (scope_key, attempt_id)
        REFERENCES public.chatgpt_usage_attempts(scope_key, attempt_id)
        ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS public.chatgpt_usage_observations (
    observation_id TEXT NOT NULL,
    scope_key TEXT NOT NULL REFERENCES public.chatgpt_usage_scopes(scope_key)
        ON DELETE CASCADE,
    collector_account_id TEXT NOT NULL,
    provider TEXT NOT NULL,
    provider_user_id TEXT,
    workspace_id TEXT,
    quota_owner_id TEXT,
    source_kind TEXT NOT NULL,
    source_id TEXT NOT NULL,
    revision_fingerprint TEXT NOT NULL,
    revision_number INTEGER NOT NULL,
    surface TEXT NOT NULL,
    conversation_id TEXT,
    payload JSONB NOT NULL,
    observed_at TIMESTAMPTZ NOT NULL,
    run_id TEXT NOT NULL,
    schema_version TEXT NOT NULL,
    provenance JSONB NOT NULL DEFAULT '{}'::jsonb,
    supersedes_observation_id TEXT,
    PRIMARY KEY (scope_key, observation_id),
    UNIQUE (scope_key, collector_account_id, source_kind, source_id, revision_fingerprint)
);

CREATE TABLE IF NOT EXISTS public.chatgpt_usage_coverage_gaps (
    gap_id TEXT NOT NULL,
    scope_key TEXT NOT NULL REFERENCES public.chatgpt_usage_scopes(scope_key)
        ON DELETE CASCADE,
    collector_account_id TEXT NOT NULL,
    source_kind TEXT NOT NULL,
    source_id TEXT NOT NULL,
    reason TEXT NOT NULL,
    state TEXT NOT NULL,
    first_seen_at TIMESTAMPTZ NOT NULL,
    last_seen_at TIMESTAMPTZ NOT NULL,
    details JSONB NOT NULL DEFAULT '{}'::jsonb,
    PRIMARY KEY (scope_key, gap_id),
    UNIQUE (scope_key, source_kind, source_id, reason)
);

CREATE TABLE IF NOT EXISTS public.chatgpt_usage_activity_provenance (
    scope_key TEXT NOT NULL REFERENCES public.chatgpt_usage_scopes(scope_key)
        ON DELETE CASCADE,
    activity_kind TEXT NOT NULL,
    activity_id TEXT NOT NULL,
    collector_account_id TEXT NOT NULL,
    first_seen_at TIMESTAMPTZ NOT NULL,
    last_seen_at TIMESTAMPTZ NOT NULL,
    PRIMARY KEY (scope_key, activity_kind, activity_id, collector_account_id)
);

CREATE INDEX IF NOT EXISTS chatgpt_usage_attempts_time_idx
    ON public.chatgpt_usage_attempts (
        scope_key,
        attempt_time,
        earliest_possible_at,
        attempt_id
    );
CREATE INDEX IF NOT EXISTS chatgpt_usage_attempts_raw_model_idx
    ON public.chatgpt_usage_attempts (
        scope_key,
        requested_model_raw,
        recorded_final_model_raw,
        resolved_model_raw
    );
CREATE INDEX IF NOT EXISTS chatgpt_usage_attempt_revisions_lookup_idx
    ON public.chatgpt_usage_attempt_revisions (scope_key, attempt_id, revision);
CREATE INDEX IF NOT EXISTS chatgpt_usage_attempt_aliases_attempt_idx
    ON public.chatgpt_usage_attempt_aliases (scope_key, attempt_id);
CREATE INDEX IF NOT EXISTS chatgpt_usage_observations_source_idx
    ON public.chatgpt_usage_observations (
        scope_key,
        source_kind,
        source_id,
        revision_number
    );
CREATE INDEX IF NOT EXISTS chatgpt_usage_coverage_gaps_state_idx
    ON public.chatgpt_usage_coverage_gaps (scope_key, state, last_seen_at);
CREATE INDEX IF NOT EXISTS chatgpt_usage_activity_provenance_lookup_idx
    ON public.chatgpt_usage_activity_provenance (
        scope_key,
        activity_kind,
        activity_id,
        last_seen_at
    );

COMMIT;
