-- COHERE-002: create the canonical direct Cohere accepted-call table.
-- Intended target: aawm_tristore.public.cohere_accepted_calls.

\set ON_ERROR_STOP on

\if :{?expected_database}
SELECT CASE
    WHEN NULLIF(btrim(:'expected_database'), '') = 'aawm_tristore'
         AND current_database() = 'aawm_tristore'
        THEN 'true'
    ELSE 'false'
END AS cohere_002_database_matches \gset

\if :cohere_002_database_matches
\set cohere_002_database_guard_statement 'SELECT 1'
\else
\echo 'COHERE-002 abort: expected_database and current_database() must both be aawm_tristore'
\set cohere_002_database_guard_statement 'SELECT 1 / 0 AS cohere_002_database_guard_failure'
\endif
\else
\echo 'COHERE-002 abort: required psql variable expected_database is missing'
\set cohere_002_database_guard_statement 'SELECT 1 / 0 AS cohere_002_database_guard_failure'
\endif

:cohere_002_database_guard_statement;

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
END AS cohere_002_owner_role_valid \gset

\if :cohere_002_owner_role_valid
\set cohere_002_owner_role_guard_statement 'SELECT 1'
\else
\echo 'COHERE-002 abort: owner_role is empty or does not resolve to an existing PostgreSQL role'
\set cohere_002_owner_role_guard_statement 'SELECT 1 / 0 AS cohere_002_owner_role_guard_failure'
\endif
\else
\echo 'COHERE-002 abort: required psql variable owner_role is missing'
\set cohere_002_owner_role_guard_statement 'SELECT 1 / 0 AS cohere_002_owner_role_guard_failure'
\endif

:cohere_002_owner_role_guard_statement;

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
END AS cohere_002_runtime_role_valid \gset

\if :cohere_002_runtime_role_valid
\set cohere_002_runtime_role_guard_statement 'SELECT 1'
\else
\echo 'COHERE-002 abort: runtime_role is empty, equals owner_role, or does not resolve to an existing PostgreSQL role'
\set cohere_002_runtime_role_guard_statement 'SELECT 1 / 0 AS cohere_002_runtime_role_guard_failure'
\endif
\else
\echo 'COHERE-002 abort: required psql variable runtime_role is missing'
\set cohere_002_runtime_role_guard_statement 'SELECT 1 / 0 AS cohere_002_runtime_role_guard_failure'
\endif

:cohere_002_runtime_role_guard_statement;

BEGIN;

CREATE TABLE IF NOT EXISTS public.cohere_accepted_calls (
    accepted_at TIMESTAMPTZ NOT NULL,
    month_start DATE NOT NULL,
    provider TEXT NOT NULL DEFAULT 'cohere'
        CHECK (provider = 'cohere'),
    credential_scope TEXT NOT NULL DEFAULT 'cohere_trial_default'
        CHECK (credential_scope = 'cohere_trial_default'),
    model TEXT,
    litellm_call_id TEXT NOT NULL UNIQUE,
    session_id TEXT,
    trace_id TEXT,
    source TEXT NOT NULL,
    evidence JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS cohere_accepted_calls_month_start_accepted_at_idx
    ON public.cohere_accepted_calls (month_start, accepted_at);

CREATE INDEX IF NOT EXISTS cohere_accepted_calls_model_accepted_at_idx
    ON public.cohere_accepted_calls (model, accepted_at);

ALTER TABLE public.cohere_accepted_calls OWNER TO :"owner_role";

REVOKE ALL ON TABLE public.cohere_accepted_calls FROM PUBLIC;
REVOKE ALL ON TABLE public.cohere_accepted_calls FROM :"runtime_role";
GRANT USAGE ON SCHEMA public TO :"runtime_role";
GRANT SELECT, INSERT
    ON TABLE public.cohere_accepted_calls
    TO :"runtime_role";

COMMIT;
