BEGIN;

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
);

CREATE UNIQUE INDEX IF NOT EXISTS aawm_alias_routing_audit_event_key_idx
    ON public.aawm_alias_routing_audit (event_key)
    WHERE event_key IS NOT NULL;

CREATE INDEX IF NOT EXISTS aawm_alias_routing_audit_session_observed_idx
    ON public.aawm_alias_routing_audit (session_id, observed_at DESC);

CREATE INDEX IF NOT EXISTS aawm_alias_routing_audit_alias_observed_idx
    ON public.aawm_alias_routing_audit (alias_model, observed_at DESC);

CREATE INDEX IF NOT EXISTS aawm_alias_routing_audit_provider_model_observed_idx
    ON public.aawm_alias_routing_audit (provider, model, observed_at DESC);

CREATE INDEX IF NOT EXISTS aawm_alias_routing_audit_event_observed_idx
    ON public.aawm_alias_routing_audit (event_type, observed_at DESC);

CREATE INDEX IF NOT EXISTS aawm_alias_routing_audit_cooldown_observed_idx
    ON public.aawm_alias_routing_audit (cooldown_key, observed_at DESC);

COMMIT;
