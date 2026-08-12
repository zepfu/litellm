-- D1-616: create the canonical Codex review-decision persistence table.
-- Intended target: <expected_database>.public.session_history_codex_review_decisions.

\set ON_ERROR_STOP on

\if :{?expected_database}
\else
\echo 'D1-616 abort: required psql variable expected_database is missing'
\quit 3
\endif

BEGIN;

DO $$
DECLARE
    expected_database_name text := NULLIF(btrim(:'expected_database'), '');
BEGIN
    IF expected_database_name IS NULL THEN
        RAISE EXCEPTION
            'D1-616 abort: required psql variable expected_database is empty';
    END IF;

    IF current_database() <> expected_database_name THEN
        RAISE EXCEPTION
            'D1-616 abort: expected database %, got %',
            expected_database_name,
            current_database();
    END IF;
END;
$$;

CREATE TABLE IF NOT EXISTS public.session_history_codex_review_decisions (
    id BIGSERIAL PRIMARY KEY,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    decision_key TEXT NOT NULL,
    reviewer_litellm_call_id TEXT NOT NULL,
    reviewer_session_id TEXT NOT NULL,
    reviewer_trace_id TEXT,
    reviewer_model TEXT,
    reviewer_agent_name TEXT,
    reviewer_agent_id TEXT,
    outcome TEXT NOT NULL CHECK (outcome IN ('allow', 'deny')),
    rationale TEXT,
    rationale_truncated BOOLEAN NOT NULL DEFAULT FALSE,
    risk_level TEXT,
    user_authorization TEXT,
    session_id TEXT,
    parent_litellm_call_id TEXT,
    parent_session_id TEXT,
    parent_thread_id TEXT,
    parent_agent_name TEXT,
    parent_agent_id TEXT,
    correlation_status TEXT NOT NULL DEFAULT 'unattributed'
        CHECK (correlation_status IN ('attributed', 'unattributed')),
    parser_version TEXT,
    contract_version TEXT,
    review_attempt_number INTEGER,
    review_attempt_key TEXT,
    governed_tool_call_id TEXT,
    governed_tool_activity_key TEXT,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    UNIQUE (decision_key)
);

CREATE INDEX IF NOT EXISTS session_history_codex_review_decisions_session_created_idx
    ON public.session_history_codex_review_decisions (session_id, created_at DESC);

CREATE INDEX IF NOT EXISTS session_history_codex_review_decisions_reviewer_call_idx
    ON public.session_history_codex_review_decisions (reviewer_litellm_call_id);

CREATE INDEX IF NOT EXISTS session_history_codex_review_decisions_reviewer_session_created_idx
    ON public.session_history_codex_review_decisions (reviewer_session_id, created_at DESC);

CREATE INDEX IF NOT EXISTS session_history_codex_review_decisions_parent_call_idx
    ON public.session_history_codex_review_decisions (parent_litellm_call_id)
    WHERE parent_litellm_call_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS session_history_codex_review_decisions_outcome_created_idx
    ON public.session_history_codex_review_decisions (outcome, created_at DESC);

CREATE INDEX IF NOT EXISTS session_history_codex_review_decisions_governed_tool_call_idx
    ON public.session_history_codex_review_decisions (governed_tool_call_id)
    WHERE governed_tool_call_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS session_history_codex_review_decisions_governed_activity_key_idx
    ON public.session_history_codex_review_decisions (governed_tool_activity_key)
    WHERE governed_tool_activity_key IS NOT NULL;

COMMIT;
