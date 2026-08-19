-- Generic locally counted accepted-call ledger.
-- COHERE-002 now uses this table instead of public.cohere_accepted_calls.
-- Intended target: aawm_tristore.public.locally_counted_accepted_calls.
-- Snapshots still go to public.rate_limit_observations as source=locally_counted.

\set ON_ERROR_STOP on

\if :{?expected_database}
SELECT CASE
    WHEN NULLIF(btrim(:'expected_database'), '') = 'aawm_tristore'
         AND current_database() = 'aawm_tristore'
        THEN 'true'
    ELSE 'false'
END AS locally_counted_database_matches \gset

\if :locally_counted_database_matches
\set locally_counted_database_guard_statement 'SELECT 1'
\else
\echo 'locally_counted abort: expected_database and current_database() must both be aawm_tristore'
\set locally_counted_database_guard_statement 'SELECT 1 / 0 AS locally_counted_database_guard_failure'
\endif
\else
\echo 'locally_counted abort: required psql variable expected_database is missing'
\set locally_counted_database_guard_statement 'SELECT 1 / 0 AS locally_counted_database_guard_failure'
\endif

:locally_counted_database_guard_statement;

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
END AS locally_counted_owner_role_valid \gset

\if :locally_counted_owner_role_valid
\set locally_counted_owner_role_guard_statement 'SELECT 1'
\else
\echo 'locally_counted abort: owner_role is empty or does not resolve to an existing PostgreSQL role'
\set locally_counted_owner_role_guard_statement 'SELECT 1 / 0 AS locally_counted_owner_role_guard_failure'
\endif
\else
\echo 'locally_counted abort: required psql variable owner_role is missing'
\set locally_counted_owner_role_guard_statement 'SELECT 1 / 0 AS locally_counted_owner_role_guard_failure'
\endif

:locally_counted_owner_role_guard_statement;

\if :{?runtime_role}
SELECT CASE
    WHEN NULLIF(btrim(:'runtime_role'), '') IS NOT NULL
         AND :'runtime_role' <> :'owner_role'
         AND EXISTS (
             SELECT 1
             FROM pg_catalog.pg_roles
             WHERE rolname = :'runtime_role'
         )
        THEN 'true'
    ELSE 'false'
END AS locally_counted_runtime_role_valid \gset

\if :locally_counted_runtime_role_valid
\set locally_counted_runtime_role_guard_statement 'SELECT 1'
\else
\echo 'locally_counted abort: runtime_role is empty, equals owner_role, or does not resolve to an existing PostgreSQL role'
\set locally_counted_runtime_role_guard_statement 'SELECT 1 / 0 AS locally_counted_runtime_role_guard_failure'
\endif
\else
\echo 'locally_counted abort: required psql variable runtime_role is missing'
\set locally_counted_runtime_role_guard_statement 'SELECT 1 / 0 AS locally_counted_runtime_role_guard_failure'
\endif

:locally_counted_runtime_role_guard_statement;

BEGIN;

CREATE TABLE IF NOT EXISTS public.locally_counted_accepted_calls (
    accepted_at TIMESTAMPTZ NOT NULL,
    provider TEXT NOT NULL,
    credential_scope TEXT NOT NULL,
    lane TEXT,
    model TEXT,
    litellm_call_id TEXT NOT NULL,
    session_id TEXT,
    trace_id TEXT,
    source TEXT NOT NULL,
    evidence JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (provider, credential_scope, litellm_call_id)
);

CREATE INDEX IF NOT EXISTS locally_counted_accepted_calls_provider_scope_accepted_at_idx
    ON public.locally_counted_accepted_calls (provider, credential_scope, accepted_at);

CREATE INDEX IF NOT EXISTS locally_counted_accepted_calls_provider_scope_model_accepted_at_idx
    ON public.locally_counted_accepted_calls (provider, credential_scope, model, accepted_at);

ALTER TABLE public.locally_counted_accepted_calls OWNER TO :"owner_role";

REVOKE ALL ON TABLE public.locally_counted_accepted_calls FROM PUBLIC;
REVOKE ALL ON TABLE public.locally_counted_accepted_calls FROM :"runtime_role";
GRANT USAGE ON SCHEMA public TO :"runtime_role";
GRANT SELECT, INSERT
    ON TABLE public.locally_counted_accepted_calls
    TO :"runtime_role";

COMMIT;
