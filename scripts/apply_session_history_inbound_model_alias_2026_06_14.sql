-- Add and backfill first-class inbound model-alias capture.
-- Intended target: aawm_tristore.public.session_history.

BEGIN;

ALTER TABLE public.session_history
    ADD COLUMN IF NOT EXISTS inbound_model_alias TEXT;

UPDATE public.session_history
SET inbound_model_alias = COALESCE(
    NULLIF(metadata->>'model_alias_label', ''),
    NULLIF(metadata->>'requested_model_alias', ''),
    NULLIF(metadata->>'codex_auto_agent_alias', ''),
    NULLIF(metadata->>'anthropic_auto_agent_alias', ''),
    NULLIF(metadata->>'aawm_auto_agent_alias', '')
)
WHERE inbound_model_alias IS NULL
  AND metadata IS NOT NULL
  AND jsonb_typeof(metadata) = 'object'
  AND COALESCE(
      NULLIF(metadata->>'model_alias_label', ''),
      NULLIF(metadata->>'requested_model_alias', ''),
      NULLIF(metadata->>'codex_auto_agent_alias', ''),
      NULLIF(metadata->>'anthropic_auto_agent_alias', ''),
      NULLIF(metadata->>'aawm_auto_agent_alias', '')
  ) IS NOT NULL;

COMMIT;
