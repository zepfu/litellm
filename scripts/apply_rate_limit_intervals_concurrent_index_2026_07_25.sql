-- D1-491: repair rate_limit_intervals concurrent-refresh eligibility.
-- Replace the expression unique index with a column-only UNIQUE index
-- (NULLS NOT DISTINCT) without dropping/recreating the materialized view.
-- Safe for live apply: advisory-lock-serialized against dashboard maintenance,
-- preflight duplicate guards, refresh while both indexes coexist, then rename.

\set ON_ERROR_STOP on

-- Session-scoped lock matches dashboard_shell_maintain_materialized_view so
-- scheduled try-lock jobs skip during this repair. Explicit unlock at success;
-- ON_ERROR_STOP / session exit releases on failure.
SELECT pg_advisory_lock(
    hashtext('dashboard-shell'),
    hashtext('materialized-view-maintenance')
);

-- Fail closed if raw column-key duplicate groups already exist under the seven
-- MV uniqueness columns. Concurrent refresh eligibility cannot paper over these.
DO $$
DECLARE
    duplicate_groups integer;
BEGIN
    SELECT count(*)::integer
    INTO duplicate_groups
    FROM (
        SELECT 1
        FROM public.rate_limit_intervals
        GROUP BY
            provider,
            model,
            quota_key,
            quota_type,
            fromdate,
            expected_reset_at,
            remaining_pct
        HAVING count(*) > 1
    ) AS raw_duplicate_groups;

    IF duplicate_groups > 0 THEN
        RAISE EXCEPTION
            'D1-491 abort: found % raw column-key duplicate group(s) under (provider, model, quota_key, quota_type, fromdate, expected_reset_at, remaining_pct); resolve before concurrent-index repair',
            duplicate_groups;
    END IF;
END;
$$;

-- Fail closed on null-vs-empty model collisions. Window partitions coalesce
-- model NULL and '' via COALESCE(model, ''), so they must not coexist under the
-- remaining key fields.
DO $$
DECLARE
    collision_groups integer;
BEGIN
    SELECT count(*)::integer
    INTO collision_groups
    FROM (
        SELECT 1
        FROM public.rate_limit_intervals
        WHERE model IS NULL OR model = ''
        GROUP BY
            provider,
            quota_key,
            quota_type,
            fromdate,
            expected_reset_at,
            remaining_pct
        HAVING
            count(*) FILTER (WHERE model IS NULL) > 0
            AND count(*) FILTER (WHERE model = '') > 0
    ) AS null_empty_model_collision_groups;

    IF collision_groups > 0 THEN
        RAISE EXCEPTION
            'D1-491 abort: found % null-vs-empty model collision group(s) under (provider, quota_key, quota_type, fromdate, expected_reset_at, remaining_pct); window partitions coalesce NULL and empty model values',
            collision_groups;
    END IF;
END;
$$;

-- Drop only a prior temporary unique index from a partial D1-491 attempt.
DROP INDEX IF EXISTS public.rate_limit_intervals_unique_idx_d1_491;

-- Temporary concurrent-refresh-eligible unique index on the seven actual MV
-- columns. Column-only (no expression/WHERE), NULLS NOT DISTINCT.
CREATE UNIQUE INDEX rate_limit_intervals_unique_idx_d1_491
    ON public.rate_limit_intervals (
        provider,
        model,
        quota_key,
        quota_type,
        fromdate,
        expected_reset_at,
        remaining_pct
    ) NULLS NOT DISTINCT;

-- Refresh while both the old expression index and the new eligible index
-- coexist so readers keep a unique-index path and CONCURRENTLY can succeed.
REFRESH MATERIALIZED VIEW CONCURRENTLY public.rate_limit_intervals;

-- Only after successful refresh: drop canonical old unique index, promote the
-- temporary index name, and refresh planner stats.
DROP INDEX IF EXISTS public.rate_limit_intervals_unique_idx;

ALTER INDEX public.rate_limit_intervals_unique_idx_d1_491
    RENAME TO rate_limit_intervals_unique_idx;

ANALYZE public.rate_limit_intervals;

SELECT pg_advisory_unlock(
    hashtext('dashboard-shell'),
    hashtext('materialized-view-maintenance')
);

-- Final read-only index/freshness summary for operator evidence.
SELECT
    i.relname AS index_name,
    pg_get_indexdef(i.oid) AS index_def,
    ix.indisunique AS is_unique,
    (
        SELECT max(r.fromdate)
        FROM public.rate_limit_intervals AS r
    ) AS max_fromdate,
    (
        SELECT count(*)::bigint
        FROM public.rate_limit_intervals AS r
    ) AS row_count
FROM pg_class AS i
JOIN pg_index AS ix
    ON ix.indexrelid = i.oid
JOIN pg_class AS t
    ON t.oid = ix.indrelid
JOIN pg_namespace AS n
    ON n.oid = t.relnamespace
WHERE n.nspname = 'public'
  AND t.relname = 'rate_limit_intervals'
  AND i.relname = 'rate_limit_intervals_unique_idx'
ORDER BY i.relname;
