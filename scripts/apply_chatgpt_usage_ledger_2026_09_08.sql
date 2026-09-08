BEGIN;

CREATE TABLE IF NOT EXISTS public.chatgpt_usage_scopes (
    scope_key TEXT PRIMARY KEY,
    collector_account_id TEXT,
    provider TEXT NOT NULL,
    provider_user_id TEXT,
    workspace_id TEXT,
    quota_owner_id TEXT,
    surface TEXT NOT NULL,
    identity_state TEXT NOT NULL DEFAULT 'provisional',
    superseded_by_scope_key TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS public.chatgpt_usage_scope_bindings (
    scope_key TEXT NOT NULL REFERENCES public.chatgpt_usage_scopes(scope_key)
        ON DELETE CASCADE,
    collector_account_id TEXT NOT NULL,
    binding_generation INTEGER NOT NULL,
    binding_state TEXT NOT NULL,
    first_seen_at TIMESTAMPTZ NOT NULL,
    last_seen_at TIMESTAMPTZ NOT NULL,
    retired_at TIMESTAMPTZ,
    PRIMARY KEY (scope_key, collector_account_id, binding_generation)
);

CREATE UNIQUE INDEX IF NOT EXISTS chatgpt_usage_scope_bindings_active_collector_idx
    ON public.chatgpt_usage_scope_bindings (collector_account_id)
    WHERE binding_state = 'active';

CREATE INDEX IF NOT EXISTS chatgpt_usage_scope_bindings_scope_idx
    ON public.chatgpt_usage_scope_bindings (scope_key, binding_state, last_seen_at);

CREATE TABLE IF NOT EXISTS public.chatgpt_usage_scope_redirects (
    retired_scope_key TEXT PRIMARY KEY
        REFERENCES public.chatgpt_usage_scopes(scope_key) ON DELETE CASCADE,
    canonical_scope_key TEXT NOT NULL
        REFERENCES public.chatgpt_usage_scopes(scope_key) ON DELETE CASCADE,
    reason TEXT NOT NULL,
    first_seen_at TIMESTAMPTZ NOT NULL,
    last_seen_at TIMESTAMPTZ NOT NULL,
    CHECK (retired_scope_key <> canonical_scope_key)
);

CREATE INDEX IF NOT EXISTS chatgpt_usage_scope_redirects_canonical_idx
    ON public.chatgpt_usage_scope_redirects (canonical_scope_key);

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
    quarantine_state TEXT NOT NULL DEFAULT 'clear',
    tombstone BOOLEAN NOT NULL DEFAULT FALSE,
    observed_at TIMESTAMPTZ NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL,
    last_seen_at TIMESTAMPTZ NOT NULL,
    superseded_by_attempt_id TEXT,
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
    is_current_projection BOOLEAN NOT NULL DEFAULT FALSE,
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
    occurrence_number INTEGER NOT NULL,
    surface TEXT NOT NULL,
    conversation_id TEXT,
    payload JSONB NOT NULL,
    observed_at TIMESTAMPTZ NOT NULL,
    run_id TEXT NOT NULL,
    schema_version TEXT NOT NULL,
    provenance JSONB NOT NULL DEFAULT '{}'::jsonb,
    first_seen_at TIMESTAMPTZ NOT NULL,
    last_seen_at TIMESTAMPTZ NOT NULL,
    last_seen_run_id TEXT NOT NULL,
    last_seen_provenance JSONB NOT NULL DEFAULT '{}'::jsonb,
    is_current_projection BOOLEAN NOT NULL DEFAULT TRUE,
    supersedes_observation_id TEXT,
    PRIMARY KEY (scope_key, observation_id),
    UNIQUE (scope_key, source_kind, source_id, occurrence_number)
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

ALTER TABLE public.chatgpt_usage_scopes
    ALTER COLUMN collector_account_id DROP NOT NULL;
ALTER TABLE public.chatgpt_usage_scopes
    ADD COLUMN IF NOT EXISTS identity_state TEXT NOT NULL DEFAULT 'provisional';
ALTER TABLE public.chatgpt_usage_scopes
    ADD COLUMN IF NOT EXISTS superseded_by_scope_key TEXT;
ALTER TABLE public.chatgpt_usage_attempts
    ADD COLUMN IF NOT EXISTS last_seen_at TIMESTAMPTZ;
ALTER TABLE public.chatgpt_usage_attempts
    ADD COLUMN IF NOT EXISTS superseded_by_attempt_id TEXT;
ALTER TABLE public.chatgpt_usage_attempts
    ADD COLUMN IF NOT EXISTS quarantine_state TEXT NOT NULL DEFAULT 'clear';
ALTER TABLE public.chatgpt_usage_attempt_revisions
    ADD COLUMN IF NOT EXISTS is_current_projection BOOLEAN NOT NULL DEFAULT FALSE;
ALTER TABLE public.chatgpt_usage_observations
    ADD COLUMN IF NOT EXISTS occurrence_number INTEGER;
ALTER TABLE public.chatgpt_usage_observations
    ADD COLUMN IF NOT EXISTS first_seen_at TIMESTAMPTZ;
ALTER TABLE public.chatgpt_usage_observations
    ADD COLUMN IF NOT EXISTS last_seen_at TIMESTAMPTZ;
ALTER TABLE public.chatgpt_usage_observations
    ADD COLUMN IF NOT EXISTS last_seen_run_id TEXT;
ALTER TABLE public.chatgpt_usage_observations
    ADD COLUMN IF NOT EXISTS last_seen_provenance JSONB NOT NULL DEFAULT '{}'::jsonb;
ALTER TABLE public.chatgpt_usage_observations
    ADD COLUMN IF NOT EXISTS is_current_projection BOOLEAN NOT NULL DEFAULT TRUE;

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
        occurrence_number,
        observed_at
    );
CREATE INDEX IF NOT EXISTS chatgpt_usage_observations_current_idx
    ON public.chatgpt_usage_observations (
        scope_key,
        source_kind,
        source_id,
        is_current_projection,
        observed_at
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

ALTER TABLE public.chatgpt_usage_scopes
    DROP CONSTRAINT IF EXISTS chatgpt_usage_scopes_collector_account_id_key;
ALTER TABLE public.chatgpt_usage_observations
    DROP CONSTRAINT IF EXISTS
        chatgpt_usage_observations_scope_key_collector_account_id_source_kind_source_id_revision_fingerprint_key;
ALTER TABLE public.chatgpt_usage_observations
    DROP CONSTRAINT IF EXISTS
        chatgpt_usage_observations_scope_key_collector_account_id_sourc;
ALTER TABLE public.chatgpt_usage_attempt_revisions
    DROP CONSTRAINT IF EXISTS
        chatgpt_usage_attempt_revisions_scope_key_attempt_id_projection_fingerprint_key;
ALTER TABLE public.chatgpt_usage_attempt_revisions
    DROP CONSTRAINT IF EXISTS
        chatgpt_usage_attempt_revisions_scope_key_attempt_id_projection;

UPDATE public.chatgpt_usage_attempts
SET last_seen_at = COALESCE(last_seen_at, updated_at, observed_at)
WHERE last_seen_at IS NULL;
ALTER TABLE public.chatgpt_usage_attempts
    ALTER COLUMN last_seen_at SET NOT NULL;

WITH numbered AS (
    SELECT observation_id, scope_key,
           ROW_NUMBER() OVER (
               PARTITION BY scope_key, source_kind, source_id
               ORDER BY revision_number, observed_at, observation_id
           ) AS occurrence_number
    FROM public.chatgpt_usage_observations
)
UPDATE public.chatgpt_usage_observations AS observations
SET occurrence_number = COALESCE(
        observations.occurrence_number,
        numbered.occurrence_number
    ),
    first_seen_at = COALESCE(observations.first_seen_at, observations.observed_at),
    last_seen_at = COALESCE(observations.last_seen_at, observations.observed_at),
    last_seen_run_id = COALESCE(observations.last_seen_run_id, observations.run_id)
FROM numbered
WHERE observations.observation_id = numbered.observation_id
  AND observations.scope_key = numbered.scope_key;
ALTER TABLE public.chatgpt_usage_observations
    ALTER COLUMN occurrence_number SET NOT NULL;
ALTER TABLE public.chatgpt_usage_observations
    ALTER COLUMN first_seen_at SET NOT NULL;
ALTER TABLE public.chatgpt_usage_observations
    ALTER COLUMN last_seen_at SET NOT NULL;
ALTER TABLE public.chatgpt_usage_observations
    ALTER COLUMN last_seen_run_id SET NOT NULL;

UPDATE public.chatgpt_usage_attempt_revisions AS revisions
SET is_current_projection = FALSE
WHERE revisions.is_current_projection
  AND NOT EXISTS (
    SELECT 1
    FROM public.chatgpt_usage_attempts AS attempts
    WHERE attempts.scope_key = revisions.scope_key
      AND attempts.attempt_id = revisions.attempt_id
      AND NOT attempts.tombstone
      AND attempts.revision = revisions.revision
      AND attempts.projection_fingerprint = revisions.projection_fingerprint
);

UPDATE public.chatgpt_usage_attempt_revisions AS revisions
SET is_current_projection = TRUE
WHERE NOT revisions.is_current_projection
  AND EXISTS (
    SELECT 1
    FROM public.chatgpt_usage_attempts AS attempts
    WHERE attempts.scope_key = revisions.scope_key
      AND attempts.attempt_id = revisions.attempt_id
      AND NOT attempts.tombstone
      AND attempts.revision = revisions.revision
      AND attempts.projection_fingerprint = revisions.projection_fingerprint
);

WITH invalid_observation_groups AS (
    SELECT scope_key, source_kind, source_id
    FROM public.chatgpt_usage_observations
    GROUP BY scope_key, source_kind, source_id
    HAVING count(*) FILTER (WHERE is_current_projection) <> 1
),
ranked_observations AS (
    SELECT
        observations.observation_id,
        observations.scope_key,
        ROW_NUMBER() OVER (
            PARTITION BY observations.scope_key,
                         observations.source_kind,
                         observations.source_id
            ORDER BY observed_at DESC, occurrence_number DESC, observation_id DESC
        ) AS freshness_rank
    FROM public.chatgpt_usage_observations AS observations
    JOIN invalid_observation_groups AS invalid
      ON invalid.scope_key = observations.scope_key
     AND invalid.source_kind = observations.source_kind
     AND invalid.source_id = observations.source_id
)
UPDATE public.chatgpt_usage_observations AS observations
SET is_current_projection = ranked_observations.freshness_rank = 1
FROM ranked_observations
WHERE observations.observation_id = ranked_observations.observation_id
  AND observations.scope_key = ranked_observations.scope_key;

UPDATE public.chatgpt_usage_scopes
SET identity_state = CASE
    WHEN provider_user_id IS NOT NULL
     AND workspace_id IS NOT NULL
     AND quota_owner_id IS NOT NULL
    THEN 'verified'
    ELSE identity_state
END
WHERE identity_state = 'provisional';

CREATE UNIQUE INDEX IF NOT EXISTS chatgpt_usage_observations_occurrence_idx
    ON public.chatgpt_usage_observations (
        scope_key, source_kind, source_id, occurrence_number
    );
CREATE UNIQUE INDEX IF NOT EXISTS chatgpt_usage_attempt_revisions_current_idx
    ON public.chatgpt_usage_attempt_revisions (scope_key, attempt_id)
    WHERE is_current_projection;
CREATE UNIQUE INDEX IF NOT EXISTS chatgpt_usage_observations_current_unique_idx
    ON public.chatgpt_usage_observations (scope_key, source_kind, source_id)
    WHERE is_current_projection;

WITH bootstrap_scopes AS (
    SELECT DISTINCT ON (scopes.collector_account_id)
           scopes.scope_key,
           scopes.collector_account_id,
           scopes.created_at,
           scopes.updated_at
    FROM public.chatgpt_usage_scopes AS scopes
    WHERE scopes.collector_account_id IS NOT NULL
      AND NOT EXISTS (
          SELECT 1
          FROM public.chatgpt_usage_scope_bindings AS existing
          WHERE existing.collector_account_id = scopes.collector_account_id
      )
    ORDER BY scopes.collector_account_id,
             (scopes.identity_state = 'verified') DESC,
             scopes.updated_at DESC,
             scopes.scope_key
)
INSERT INTO public.chatgpt_usage_scope_bindings (
    scope_key, collector_account_id, binding_generation, binding_state,
    first_seen_at, last_seen_at
)
SELECT scope_key, collector_account_id, 1, 'active',
       COALESCE(created_at, NOW()), COALESCE(updated_at, created_at, NOW())
FROM bootstrap_scopes
ON CONFLICT (scope_key, collector_account_id, binding_generation) DO NOTHING;

COMMIT;
