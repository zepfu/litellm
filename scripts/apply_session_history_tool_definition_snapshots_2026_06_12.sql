BEGIN;

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
);

CREATE INDEX IF NOT EXISTS session_history_tool_definition_snapshots_session_created_idx
    ON public.session_history_tool_definition_snapshots (session_id, created_at DESC);

CREATE INDEX IF NOT EXISTS session_history_tool_definition_snapshots_hash_idx
    ON public.session_history_tool_definition_snapshots (snapshot_hash);

COMMIT;
