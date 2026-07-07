-- Add first-class host attribution capture for session history reporting.
-- Intended target: aawm_tristore.public.session_history.

BEGIN;

ALTER TABLE public.session_history
    ADD COLUMN IF NOT EXISTS client_ip TEXT,
    ADD COLUMN IF NOT EXISTS host_name TEXT;

COMMIT;

-- Historical backfill should be run separately in bounded batches. A full-table
-- metadata backfill can block live session_history inserts on large dev/prod
-- tables.
