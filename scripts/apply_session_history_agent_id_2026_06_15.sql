-- Add first-class nullable agent_id capture for session history reporting.
-- Intended target: aawm_tristore.public.session_history and tool activity.

BEGIN;

ALTER TABLE public.session_history
    ADD COLUMN IF NOT EXISTS agent_id TEXT;

ALTER TABLE public.session_history_tool_activity
    ADD COLUMN IF NOT EXISTS agent_id TEXT;

COMMIT;
