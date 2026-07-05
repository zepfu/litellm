-- Rebuild public.rate_limit_intervals with Antigravity Code Assist pool rows.
-- This is intentionally an explicit database operation, not callback runtime DDL.

\set ON_ERROR_STOP on

BEGIN;

DROP MATERIALIZED VIEW IF EXISTS public.rate_limit_intervals;

CREATE MATERIALIZED VIEW public.rate_limit_intervals AS
WITH rate_limit_points AS (
    SELECT
        rate_limit_observations.id,
        rate_limit_observations.provider,
        CASE
            WHEN rate_limit_observations.provider = 'google'::text
                THEN regexp_replace(rate_limit_observations.quota_key, '^.*(gemini-\d+(?:\.\d+)?-[^-:]+(?:-[^-:]+)*).*'::text, '\1'::text, 'i'::text)
            WHEN rate_limit_observations.provider = 'xai'::text
                THEN COALESCE(NULLIF(rate_limit_observations.model, ''::text), 'grok-build'::text)
            WHEN rate_limit_observations.provider = 'openrouter'::text
                THEN COALESCE(NULLIF(rate_limit_observations.model, ''::text), ''::text)
            WHEN rate_limit_observations.provider = 'antigravity'::text
                THEN NULL::text
            ELSE ''::text
        END AS model,
        rate_limit_observations.quota_key,
        rate_limit_observations.quota_type,
        rate_limit_observations.expected_reset_at,
        rate_limit_observations.remaining_pct,
        rate_limit_observations.observed_at
    FROM public.rate_limit_observations
    WHERE provider = ANY (ARRAY['openai', 'anthropic', 'google', 'xai', 'openrouter', 'antigravity'])
      AND remaining_pct >= 0
      AND (
          remaining_pct < 100
          OR (
              provider = 'antigravity'
              AND quota_key = ANY (ARRAY[
                  'antigravity_code_assist:gemini_pool',
                  'antigravity_code_assist:vertex_pool'
              ])
          )
          OR (
              provider = 'xai'
              AND quota_key = 'xai_grok_build_weekly_credits:credits'
          )
      )
      AND (
          quota_key = ANY (ARRAY[
              'codex:secondary',
              'codex:primary',
              'codex_bengalfox:secondary',
              'codex_bengalfox:primary',
              'anthropic_unified_7d:7d',
              'anthropic_unified_7d_oi:7d_oi',
              'anthropic_unified_7d_sonnet:7d_sonnet',
              'anthropic_unified_5h:5h',
              'antigravity_code_assist:gemini_pool',
              'antigravity_code_assist:vertex_pool',
              'xai_grok_build_weekly_credits:credits'
          ])
          OR (
              provider <> 'antigravity'
              AND quota_type = 'requests'
          )
      )
),
rate_limit_changes AS (
    SELECT
        rate_limit_points.*,
        lag(expected_reset_at) OVER rate_limit_window AS previous_expected_reset_at,
        lag(remaining_pct) OVER rate_limit_window AS previous_remaining_pct
    FROM rate_limit_points
    WINDOW rate_limit_window AS (
        PARTITION BY provider, COALESCE(model, ''::text), quota_key, quota_type
        ORDER BY observed_at, id
    )
),
rate_limit_intervals AS (
    SELECT
        provider,
        model,
        quota_key,
        quota_type,
        expected_reset_at,
        remaining_pct,
        observed_at AS fromdate,
        lead(observed_at) OVER (
            PARTITION BY provider, COALESCE(model, ''::text), quota_key, quota_type
            ORDER BY observed_at, id
        ) AS next_fromdate
    FROM rate_limit_changes
    WHERE previous_remaining_pct IS NULL
       OR previous_remaining_pct IS DISTINCT FROM remaining_pct
       OR NOT (
           previous_expected_reset_at IS NOT DISTINCT FROM expected_reset_at
           OR (
               previous_expected_reset_at IS NOT NULL
               AND expected_reset_at IS NOT NULL
               AND abs(EXTRACT(epoch FROM expected_reset_at - previous_expected_reset_at)) < 900
           )
       )
)
SELECT DISTINCT
    provider,
    model,
    quota_key,
    expected_reset_at,
    remaining_pct,
    fromdate,
    COALESCE(next_fromdate, '9999-12-31 00:00:00+00'::timestamptz) AS todate,
    CASE
        WHEN quota_key = ANY (ARRAY['anthropic_unified_7d:7d', 'codex:secondary', 'xai_grok_build_weekly_credits:credits']) THEN 'weekly'
        WHEN quota_key = ANY (ARRAY['anthropic_unified_5h:5h', 'codex:primary']) THEN 'short'
        WHEN quota_key = ANY (ARRAY['codex_bengalfox:primary']) THEN 'short_special'
        WHEN quota_key = ANY (ARRAY['anthropic_unified_7d_oi:7d_oi']) THEN 'weekly_overage_included'
        WHEN quota_key = ANY (ARRAY['anthropic_unified_7d_sonnet:7d_sonnet', 'codex_bengalfox:secondary']) THEN 'weekly_special'
        ELSE quota_type
    END AS quota_type
FROM rate_limit_intervals;

CREATE INDEX rate_limit_intervals_requests_idx
    ON public.rate_limit_intervals (quota_type, provider, model, fromdate DESC);
CREATE INDEX rate_limit_intervals_type_provider_from_idx
    ON public.rate_limit_intervals (quota_type, provider, fromdate DESC);
CREATE UNIQUE INDEX rate_limit_intervals_unique_idx
    ON public.rate_limit_intervals (
        provider,
        COALESCE(model, ''::text),
        quota_key,
        quota_type,
        fromdate,
        expected_reset_at,
        remaining_pct
    );

ANALYZE public.rate_limit_intervals;

COMMIT;
