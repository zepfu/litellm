BEGIN;

ALTER TABLE public.rate_limit_observations
    ADD COLUMN IF NOT EXISTS quota_limit DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS quota_used DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS quota_remaining DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS billing_period_start_at TIMESTAMPTZ,
    ADD COLUMN IF NOT EXISTS billing_period_end_at TIMESTAMPTZ,
    ADD COLUMN IF NOT EXISTS raw_provider_fields JSONB DEFAULT '{}'::jsonb,
    ADD COLUMN IF NOT EXISTS evidence JSONB DEFAULT '{}'::jsonb;

UPDATE public.rate_limit_observations
SET raw_provider_fields = '{}'::jsonb
WHERE raw_provider_fields IS NULL;

UPDATE public.rate_limit_observations
SET evidence = '{}'::jsonb
WHERE evidence IS NULL;

ALTER TABLE public.rate_limit_observations
    ALTER COLUMN raw_provider_fields SET DEFAULT '{}'::jsonb,
    ALTER COLUMN evidence SET DEFAULT '{}'::jsonb,
    ALTER COLUMN raw_provider_fields SET NOT NULL,
    ALTER COLUMN evidence SET NOT NULL;

COMMIT;
