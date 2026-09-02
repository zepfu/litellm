-- D1-679: define bounded rate-limit observation retention.
--
-- This file creates the archive table and the two database-native retention
-- functions. It does not install pg_cron, schedule a job, or run cleanup.
-- Apply only to the explicitly guarded target database.

\set ON_ERROR_STOP on

\if :{?expected_database}
SELECT CASE
    WHEN NULLIF(btrim(:'expected_database'), '') = current_database()
        THEN 'true'
    ELSE 'false'
END AS d1_679_database_matches \gset

\if :d1_679_database_matches
\set d1_679_database_guard_statement 'SELECT 1'
\else
\echo 'D1-679 abort: expected_database is empty or does not match current_database()'
\set d1_679_database_guard_statement 'SELECT 1 / 0 AS d1_679_database_guard_failure'
\endif
\else
\echo 'D1-679 abort: required psql variable expected_database is missing'
\set d1_679_database_guard_statement 'SELECT 1 / 0 AS d1_679_database_guard_failure'
\endif

:d1_679_database_guard_statement;

\if :{?owner_role}
SELECT CASE
    WHEN NULLIF(btrim(:'owner_role'), '') IS NOT NULL
         AND EXISTS (
             SELECT 1
             FROM pg_catalog.pg_roles
             WHERE rolname = :'owner_role'
         )
        THEN 'true'
    ELSE 'false'
END AS d1_679_owner_role_valid \gset

\if :d1_679_owner_role_valid
\set d1_679_owner_role_guard_statement 'SELECT 1'
\else
\echo 'D1-679 abort: owner_role is empty or does not resolve to an existing PostgreSQL role'
\set d1_679_owner_role_guard_statement 'SELECT 1 / 0 AS d1_679_owner_role_guard_failure'
\endif
\else
\echo 'D1-679 abort: required psql variable owner_role is missing'
\set d1_679_owner_role_guard_statement 'SELECT 1 / 0 AS d1_679_owner_role_guard_failure'
\endif

:d1_679_owner_role_guard_statement;

\if :{?job_role}
SELECT CASE
    WHEN NULLIF(btrim(:'job_role'), '') IS NOT NULL
         AND EXISTS (
             SELECT 1
             FROM pg_catalog.pg_roles
             WHERE rolname = :'job_role'
         )
        THEN 'true'
    ELSE 'false'
END AS d1_679_job_role_valid \gset

\if :d1_679_job_role_valid
\set d1_679_job_role_guard_statement 'SELECT 1'
\else
\echo 'D1-679 abort: job_role is empty or does not resolve to an existing PostgreSQL role'
\set d1_679_job_role_guard_statement 'SELECT 1 / 0 AS d1_679_job_role_guard_failure'
\endif
\else
\echo 'D1-679 abort: required psql variable job_role is missing'
\set d1_679_job_role_guard_statement 'SELECT 1 / 0 AS d1_679_job_role_guard_failure'
\endif

:d1_679_job_role_guard_statement;

SELECT CASE
    WHEN pg_catalog.to_regclass('public.rate_limit_observations') IS NOT NULL
        THEN 'true'
    ELSE 'false'
END AS d1_679_source_table_present \gset

\if :d1_679_source_table_present
\set d1_679_source_table_guard_statement 'SELECT 1'
\else
\echo 'D1-679 abort: public.rate_limit_observations is missing from the target database'
\set d1_679_source_table_guard_statement 'SELECT 1 / 0 AS d1_679_source_table_guard_failure'
\endif

:d1_679_source_table_guard_statement;

BEGIN;

CREATE TABLE IF NOT EXISTS public.rate_limit_observation_older (
    id BIGINT PRIMARY KEY,
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
);

CREATE INDEX IF NOT EXISTS rate_limit_observation_older_identity_latest_idx
    ON public.rate_limit_observation_older (
        provider,
        client,
        account_hash,
        quota_key,
        source,
        observed_at DESC,
        id DESC
    );

CREATE INDEX IF NOT EXISTS rate_limit_observation_older_observed_at_idx
    ON public.rate_limit_observation_older (observed_at ASC, id ASC);

CREATE OR REPLACE FUNCTION public.aawm_archive_rate_limit_observations_daily(
    p_batch_size INTEGER DEFAULT 1000
)
RETURNS INTEGER
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $function$
DECLARE
    v_batch_size INTEGER := LEAST(
        GREATEST(COALESCE(p_batch_size, 1000), 1),
        1000
    );
    v_cutoff TIMESTAMPTZ := NOW() - INTERVAL '45 days';
    v_moved INTEGER;
BEGIN
    PERFORM pg_advisory_xact_lock(
        hashtextextended('aawm.rate_limit_observation_retention', 0)
    );

    WITH candidates AS MATERIALIZED (
        SELECT
            id,
            observed_at,
            created_at,
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
        FROM public.rate_limit_observations
        WHERE observed_at < v_cutoff
        ORDER BY observed_at ASC, id ASC
        LIMIT v_batch_size
        FOR UPDATE SKIP LOCKED
    ),
    archived AS (
        INSERT INTO public.rate_limit_observation_older AS archive (
            id,
            observed_at,
            created_at,
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
            id,
            observed_at,
            created_at,
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
        FROM candidates
        ON CONFLICT (id) DO UPDATE
        SET id = EXCLUDED.id
        WHERE ROW(
            archive.id,
            archive.observed_at,
            archive.created_at,
            archive.client,
            archive.client_version,
            archive.account_hash,
            archive.provider,
            archive.model,
            archive.quota_key,
            archive.quota_period,
            archive.quota_type,
            archive.expected_reset_at,
            archive.remaining_pct,
            archive.quota_limit,
            archive.quota_used,
            archive.quota_remaining,
            archive.billing_period_start_at,
            archive.billing_period_end_at,
            archive.raw_provider_fields,
            archive.evidence,
            archive.source,
            archive.session_id,
            archive.trace_id,
            archive.litellm_call_id
        ) IS NOT DISTINCT FROM ROW(
            EXCLUDED.id,
            EXCLUDED.observed_at,
            EXCLUDED.created_at,
            EXCLUDED.client,
            EXCLUDED.client_version,
            EXCLUDED.account_hash,
            EXCLUDED.provider,
            EXCLUDED.model,
            EXCLUDED.quota_key,
            EXCLUDED.quota_period,
            EXCLUDED.quota_type,
            EXCLUDED.expected_reset_at,
            EXCLUDED.remaining_pct,
            EXCLUDED.quota_limit,
            EXCLUDED.quota_used,
            EXCLUDED.quota_remaining,
            EXCLUDED.billing_period_start_at,
            EXCLUDED.billing_period_end_at,
            EXCLUDED.raw_provider_fields,
            EXCLUDED.evidence,
            EXCLUDED.source,
            EXCLUDED.session_id,
            EXCLUDED.trace_id,
            EXCLUDED.litellm_call_id
        )
        RETURNING archive.id
    ),
    deleted AS (
        DELETE FROM public.rate_limit_observations AS source
        USING archived
        WHERE source.id = archived.id
          AND source.observed_at < v_cutoff
        RETURNING source.id
    )
    SELECT count(*)::INTEGER
    INTO v_moved
    FROM deleted;

    RETURN v_moved;
END
$function$;

CREATE OR REPLACE FUNCTION public.aawm_expire_rate_limit_observation_archive_weekly(
    p_batch_size INTEGER DEFAULT 5000
)
RETURNS INTEGER
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $function$
DECLARE
    v_batch_size INTEGER := LEAST(
        GREATEST(COALESCE(p_batch_size, 5000), 1),
        5000
    );
    v_cutoff TIMESTAMPTZ := NOW() - INTERVAL '6 months';
    v_deleted INTEGER;
BEGIN
    PERFORM pg_advisory_xact_lock(
        hashtextextended('aawm.rate_limit_observation_retention', 0)
    );

    WITH candidates AS MATERIALIZED (
        SELECT id
        FROM public.rate_limit_observation_older
        WHERE observed_at < v_cutoff
        ORDER BY observed_at ASC, id ASC
        LIMIT v_batch_size
        FOR UPDATE SKIP LOCKED
    ),
    deleted AS (
        DELETE FROM public.rate_limit_observation_older AS archive
        USING candidates
        WHERE archive.id = candidates.id
        RETURNING archive.id
    )
    SELECT count(*)::INTEGER
    INTO v_deleted
    FROM deleted;

    RETURN v_deleted;
END
$function$;

ALTER TABLE public.rate_limit_observation_older OWNER TO :"owner_role";
ALTER FUNCTION public.aawm_archive_rate_limit_observations_daily(INTEGER)
    OWNER TO :"owner_role";
ALTER FUNCTION public.aawm_expire_rate_limit_observation_archive_weekly(INTEGER)
    OWNER TO :"owner_role";

REVOKE ALL ON FUNCTION public.aawm_archive_rate_limit_observations_daily(INTEGER)
    FROM PUBLIC;
REVOKE ALL ON FUNCTION public.aawm_expire_rate_limit_observation_archive_weekly(INTEGER)
    FROM PUBLIC;
GRANT EXECUTE ON FUNCTION public.aawm_archive_rate_limit_observations_daily(INTEGER)
    TO :"job_role";
GRANT EXECUTE ON FUNCTION public.aawm_expire_rate_limit_observation_archive_weekly(INTEGER)
    TO :"job_role";

COMMIT;
