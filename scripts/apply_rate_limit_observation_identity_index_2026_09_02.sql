-- D1-678: install the selective latest-observation identity index.
-- Intended target: aawm_tristore.public.rate_limit_observations.
--
-- This script deliberately does not use BEGIN/COMMIT because PostgreSQL
-- requires CREATE/DROP INDEX CONCURRENTLY to run outside a transaction block.
-- The old canonical index remains available while the replacement builds.

\set ON_ERROR_STOP on

\if :{?expected_database}
SELECT CASE
    WHEN NULLIF(btrim(:'expected_database'), '') = 'aawm_tristore'
         AND current_database() = 'aawm_tristore'
        THEN 'true'
    ELSE 'false'
END AS d1_678_database_matches \gset

\if :d1_678_database_matches
\set d1_678_database_guard_statement 'SELECT 1'
\else
\echo 'D1-678 abort: expected_database and current_database() must both be aawm_tristore'
\set d1_678_database_guard_statement 'SELECT 1 / 0 AS d1_678_database_guard_failure'
\endif
\else
\echo 'D1-678 abort: required psql variable expected_database is missing'
\set d1_678_database_guard_statement 'SELECT 1 / 0 AS d1_678_database_guard_failure'
\endif

:d1_678_database_guard_statement;

SELECT CASE
    WHEN pg_catalog.to_regclass('public.rate_limit_observations') IS NOT NULL
         AND EXISTS (
             SELECT 1
             FROM pg_catalog.pg_class AS table_rel
             JOIN pg_catalog.pg_namespace AS table_ns
                 ON table_ns.oid = table_rel.relnamespace
             WHERE table_rel.oid =
                     pg_catalog.to_regclass('public.rate_limit_observations')
               AND table_ns.nspname = 'public'
               AND table_rel.relname = 'rate_limit_observations'
               AND table_rel.relkind = 'r'
         )
        THEN 'true'
    ELSE 'false'
END AS d1_678_table_exists \gset

\if :d1_678_table_exists
\set d1_678_table_guard_statement 'SELECT 1'
\else
\echo 'D1-678 abort: public.rate_limit_observations does not exist or is not an ordinary table'
\set d1_678_table_guard_statement 'SELECT 1 / 0 AS d1_678_table_guard_failure'
\endif

:d1_678_table_guard_statement;

-- Serialize operator retries. Session exit releases this lock after any
-- ON_ERROR_STOP failure, including an interrupted concurrent index build.
SELECT pg_advisory_lock(
    hashtext('aawm-tristore-schema'),
    hashtext('d1-678-rate-limit-observation-identity-index')
);

-- A completed prior run is a no-op. Validate the access method, readiness,
-- exact seven keys, normalized nullable expressions, and descending tie-break.
WITH target_index AS (
    SELECT ix.*, index_rel.relam
    FROM pg_catalog.pg_index AS ix
    JOIN pg_catalog.pg_class AS index_rel
        ON index_rel.oid = ix.indexrelid
    JOIN pg_catalog.pg_class AS table_rel
        ON table_rel.oid = ix.indrelid
    JOIN pg_catalog.pg_namespace AS table_ns
        ON table_ns.oid = table_rel.relnamespace
    WHERE table_rel.oid =
              pg_catalog.to_regclass('public.rate_limit_observations')
      AND table_ns.nspname = 'public'
      AND table_rel.relname = 'rate_limit_observations'
      AND table_rel.relkind = 'r'
      AND index_rel.relname = 'rate_limit_observations_identity_latest_idx'
      AND index_rel.relkind = 'i'
)
SELECT CASE
    WHEN EXISTS (
        SELECT 1
        FROM target_index AS target
        JOIN pg_catalog.pg_am AS access_method
            ON access_method.oid = target.relam
        WHERE target.indisvalid
          AND target.indisready
          AND target.indislive
          AND NOT target.indisunique
          AND target.indpred IS NULL
          AND target.indnkeyatts = 7
          AND target.indnatts = 7
          AND access_method.amname = 'btree'
          AND pg_get_indexdef(target.indexrelid, 1, false) = 'quota_key'
          AND pg_get_indexdef(target.indexrelid, 2, false) = 'provider'
          AND regexp_replace(
                  lower(pg_get_indexdef(target.indexrelid, 3, false)),
                  '(::text)|[[:space:]()]',
                  '',
                  'g'
              ) = 'casewhenclientisnullthen''n:''else''v:''||clientend'
          AND regexp_replace(
                  lower(pg_get_indexdef(target.indexrelid, 4, false)),
                  '(::text)|[[:space:]()]',
                  '',
                  'g'
              ) = 'casewhenaccount_hashisnullthen''n:''else''v:''||account_hashend'
          AND regexp_replace(
                  lower(pg_get_indexdef(target.indexrelid, 5, false)),
                  '(::text)|[[:space:]()]',
                  '',
                  'g'
              ) = 'casewhensourceisnullthen''n:''else''v:''||sourceend'
          AND regexp_replace(
                  lower(pg_get_indexdef(target.indexrelid, 6, false)),
                  '[[:space:]]',
                  '',
                  'g'
              ) = 'observed_atdesc'
          AND regexp_replace(
                  lower(pg_get_indexdef(target.indexrelid, 7, false)),
                  '[[:space:]]',
                  '',
                  'g'
              ) = 'iddesc'
    )
        THEN 'true'
    ELSE 'false'
END AS d1_678_canonical_ready \gset

\if :d1_678_canonical_ready
\echo 'D1-678: canonical identity index is already valid and current'
\else
-- Reuse a valid replacement left by an interruption after build. Invalid or
-- wrong-shape replacements are removed concurrently before retrying the build.
WITH target_index AS (
    SELECT ix.*, index_rel.relam
    FROM pg_catalog.pg_index AS ix
    JOIN pg_catalog.pg_class AS index_rel
        ON index_rel.oid = ix.indexrelid
    JOIN pg_catalog.pg_class AS table_rel
        ON table_rel.oid = ix.indrelid
    JOIN pg_catalog.pg_namespace AS table_ns
        ON table_ns.oid = table_rel.relnamespace
    WHERE table_rel.oid =
              pg_catalog.to_regclass('public.rate_limit_observations')
      AND table_ns.nspname = 'public'
      AND table_rel.relname = 'rate_limit_observations'
      AND table_rel.relkind = 'r'
      AND index_rel.relname = 'rate_limit_observations_identity_latest_d1_678_idx'
      AND index_rel.relkind = 'i'
)
SELECT CASE
    WHEN EXISTS (
        SELECT 1
        FROM target_index AS target
        JOIN pg_catalog.pg_am AS access_method
            ON access_method.oid = target.relam
        WHERE target.indisvalid
          AND target.indisready
          AND target.indislive
          AND NOT target.indisunique
          AND target.indpred IS NULL
          AND target.indnkeyatts = 7
          AND target.indnatts = 7
          AND access_method.amname = 'btree'
          AND pg_get_indexdef(target.indexrelid, 1, false) = 'quota_key'
          AND pg_get_indexdef(target.indexrelid, 2, false) = 'provider'
          AND regexp_replace(
                  lower(pg_get_indexdef(target.indexrelid, 3, false)),
                  '(::text)|[[:space:]()]',
                  '',
                  'g'
              ) = 'casewhenclientisnullthen''n:''else''v:''||clientend'
          AND regexp_replace(
                  lower(pg_get_indexdef(target.indexrelid, 4, false)),
                  '(::text)|[[:space:]()]',
                  '',
                  'g'
              ) = 'casewhenaccount_hashisnullthen''n:''else''v:''||account_hashend'
          AND regexp_replace(
                  lower(pg_get_indexdef(target.indexrelid, 5, false)),
                  '(::text)|[[:space:]()]',
                  '',
                  'g'
              ) = 'casewhensourceisnullthen''n:''else''v:''||sourceend'
          AND regexp_replace(
                  lower(pg_get_indexdef(target.indexrelid, 6, false)),
                  '[[:space:]]',
                  '',
                  'g'
              ) = 'observed_atdesc'
          AND regexp_replace(
                  lower(pg_get_indexdef(target.indexrelid, 7, false)),
                  '[[:space:]]',
                  '',
                  'g'
              ) = 'iddesc'
    )
        THEN 'true'
    ELSE 'false'
END AS d1_678_replacement_ready \gset

\if :d1_678_replacement_ready
\echo 'D1-678: reusing valid replacement index from an earlier attempt'
\else
DROP INDEX CONCURRENTLY IF EXISTS
    public.rate_limit_observations_identity_latest_d1_678_idx;

CREATE INDEX CONCURRENTLY rate_limit_observations_identity_latest_d1_678_idx
    ON public.rate_limit_observations (
        quota_key,
        provider,
        (CASE WHEN client IS NULL THEN 'n:' ELSE 'v:' || client END),
        (CASE
            WHEN account_hash IS NULL THEN 'n:'
            ELSE 'v:' || account_hash
        END),
        (CASE WHEN source IS NULL THEN 'n:' ELSE 'v:' || source END),
        observed_at DESC,
        id DESC
    );
\endif

-- Verify the replacement after either reuse or build. Never remove the old
-- canonical index unless the replacement is valid, ready, live, and exact.
WITH target_index AS (
    SELECT ix.*, index_rel.relam
    FROM pg_catalog.pg_index AS ix
    JOIN pg_catalog.pg_class AS index_rel
        ON index_rel.oid = ix.indexrelid
    JOIN pg_catalog.pg_class AS table_rel
        ON table_rel.oid = ix.indrelid
    JOIN pg_catalog.pg_namespace AS table_ns
        ON table_ns.oid = table_rel.relnamespace
    WHERE table_rel.oid =
              pg_catalog.to_regclass('public.rate_limit_observations')
      AND table_ns.nspname = 'public'
      AND table_rel.relname = 'rate_limit_observations'
      AND table_rel.relkind = 'r'
      AND index_rel.relname = 'rate_limit_observations_identity_latest_d1_678_idx'
      AND index_rel.relkind = 'i'
)
SELECT CASE
    WHEN EXISTS (
        SELECT 1
        FROM target_index AS target
        JOIN pg_catalog.pg_am AS access_method
            ON access_method.oid = target.relam
        WHERE target.indisvalid
          AND target.indisready
          AND target.indislive
          AND NOT target.indisunique
          AND target.indpred IS NULL
          AND target.indnkeyatts = 7
          AND target.indnatts = 7
          AND access_method.amname = 'btree'
          AND pg_get_indexdef(target.indexrelid, 1, false) = 'quota_key'
          AND pg_get_indexdef(target.indexrelid, 2, false) = 'provider'
          AND regexp_replace(
                  lower(pg_get_indexdef(target.indexrelid, 3, false)),
                  '(::text)|[[:space:]()]',
                  '',
                  'g'
              ) = 'casewhenclientisnullthen''n:''else''v:''||clientend'
          AND regexp_replace(
                  lower(pg_get_indexdef(target.indexrelid, 4, false)),
                  '(::text)|[[:space:]()]',
                  '',
                  'g'
              ) = 'casewhenaccount_hashisnullthen''n:''else''v:''||account_hashend'
          AND regexp_replace(
                  lower(pg_get_indexdef(target.indexrelid, 5, false)),
                  '(::text)|[[:space:]()]',
                  '',
                  'g'
              ) = 'casewhensourceisnullthen''n:''else''v:''||sourceend'
          AND regexp_replace(
                  lower(pg_get_indexdef(target.indexrelid, 6, false)),
                  '[[:space:]]',
                  '',
                  'g'
              ) = 'observed_atdesc'
          AND regexp_replace(
                  lower(pg_get_indexdef(target.indexrelid, 7, false)),
                  '[[:space:]]',
                  '',
                  'g'
              ) = 'iddesc'
    )
        THEN 'true'
    ELSE 'false'
END AS d1_678_replacement_ready \gset

\if :d1_678_replacement_ready
\set d1_678_replacement_guard_statement 'SELECT 1'
\else
\echo 'D1-678 abort: replacement index is missing, invalid, or has the wrong definition'
\set d1_678_replacement_guard_statement 'SELECT 1 / 0 AS d1_678_replacement_guard_failure'
\endif

:d1_678_replacement_guard_statement;

-- The validated replacement remains usable while the old canonical index is
-- removed. A crash after this drop leaves the replacement available; rerun
-- reuses it and completes the rename.
DROP INDEX CONCURRENTLY IF EXISTS
    public.rate_limit_observations_identity_latest_idx;

ALTER INDEX public.rate_limit_observations_identity_latest_d1_678_idx
    RENAME TO rate_limit_observations_identity_latest_idx;
\endif

-- Final postcondition. A failure exits psql and releases the session lock.
WITH target_index AS (
    SELECT ix.*, index_rel.relam
    FROM pg_catalog.pg_index AS ix
    JOIN pg_catalog.pg_class AS index_rel
        ON index_rel.oid = ix.indexrelid
    JOIN pg_catalog.pg_class AS table_rel
        ON table_rel.oid = ix.indrelid
    JOIN pg_catalog.pg_namespace AS table_ns
        ON table_ns.oid = table_rel.relnamespace
    WHERE table_rel.oid =
              pg_catalog.to_regclass('public.rate_limit_observations')
      AND table_ns.nspname = 'public'
      AND table_rel.relname = 'rate_limit_observations'
      AND table_rel.relkind = 'r'
      AND index_rel.relname = 'rate_limit_observations_identity_latest_idx'
      AND index_rel.relkind = 'i'
)
SELECT CASE
    WHEN EXISTS (
        SELECT 1
        FROM target_index AS target
        JOIN pg_catalog.pg_am AS access_method
            ON access_method.oid = target.relam
        WHERE target.indisvalid
          AND target.indisready
          AND target.indislive
          AND NOT target.indisunique
          AND target.indpred IS NULL
          AND target.indnkeyatts = 7
          AND target.indnatts = 7
          AND access_method.amname = 'btree'
          AND pg_get_indexdef(target.indexrelid, 1, false) = 'quota_key'
          AND pg_get_indexdef(target.indexrelid, 2, false) = 'provider'
          AND regexp_replace(
                  lower(pg_get_indexdef(target.indexrelid, 3, false)),
                  '(::text)|[[:space:]()]',
                  '',
                  'g'
              ) = 'casewhenclientisnullthen''n:''else''v:''||clientend'
          AND regexp_replace(
                  lower(pg_get_indexdef(target.indexrelid, 4, false)),
                  '(::text)|[[:space:]()]',
                  '',
                  'g'
              ) = 'casewhenaccount_hashisnullthen''n:''else''v:''||account_hashend'
          AND regexp_replace(
                  lower(pg_get_indexdef(target.indexrelid, 5, false)),
                  '(::text)|[[:space:]()]',
                  '',
                  'g'
              ) = 'casewhensourceisnullthen''n:''else''v:''||sourceend'
          AND regexp_replace(
                  lower(pg_get_indexdef(target.indexrelid, 6, false)),
                  '[[:space:]]',
                  '',
                  'g'
              ) = 'observed_atdesc'
          AND regexp_replace(
                  lower(pg_get_indexdef(target.indexrelid, 7, false)),
                  '[[:space:]]',
                  '',
                  'g'
              ) = 'iddesc'
    )
        THEN 'true'
    ELSE 'false'
END AS d1_678_canonical_ready \gset

\if :d1_678_canonical_ready
\set d1_678_final_guard_statement 'SELECT 1'
\else
\echo 'D1-678 abort: canonical index postcondition failed'
\set d1_678_final_guard_statement 'SELECT 1 / 0 AS d1_678_final_guard_failure'
\endif

:d1_678_final_guard_statement;

SELECT pg_advisory_unlock(
    hashtext('aawm-tristore-schema'),
    hashtext('d1-678-rate-limit-observation-identity-index')
);

SELECT
    current_database() AS database_name,
    index_rel.relname AS index_name,
    ix.indisvalid AS is_valid,
    ix.indisready AS is_ready,
    ix.indislive AS is_live,
    pg_get_indexdef(index_rel.oid) AS index_definition
FROM pg_catalog.pg_index AS ix
JOIN pg_catalog.pg_class AS index_rel
    ON index_rel.oid = ix.indexrelid
JOIN pg_catalog.pg_class AS table_rel
    ON table_rel.oid = ix.indrelid
JOIN pg_catalog.pg_namespace AS table_ns
    ON table_ns.oid = table_rel.relnamespace
WHERE table_rel.oid =
          pg_catalog.to_regclass('public.rate_limit_observations')
  AND table_ns.nspname = 'public'
  AND table_rel.relname = 'rate_limit_observations'
  AND table_rel.relkind = 'r'
  AND index_rel.relname = 'rate_limit_observations_identity_latest_idx'
  AND index_rel.relkind = 'i';
