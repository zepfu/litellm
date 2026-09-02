-- D1-679: define the pg_cron jobs for rate-limit observation retention.
--
-- This file only installs the two named schedules when an operator runs it.
-- pg_cron must already be installed in the configured cron control database,
-- with cron.timezone set to UTC. This file does not install or configure the
-- extension and does not run either retention function immediately.
--
-- Run this file from the pg_cron control database. The target database is
-- passed explicitly to cron.schedule_in_database().

\set ON_ERROR_STOP on

\if :{?cron_database}
SELECT CASE
    WHEN NULLIF(btrim(:'cron_database'), '') = current_database()
        THEN 'true'
    ELSE 'false'
END AS d1_679_cron_database_matches \gset

\if :d1_679_cron_database_matches
\set d1_679_cron_database_guard_statement 'SELECT 1'
\else
\echo 'D1-679 abort: cron_database is empty or does not match current_database()'
\set d1_679_cron_database_guard_statement 'SELECT 1 / 0 AS d1_679_cron_database_guard_failure'
\endif
\else
\echo 'D1-679 abort: required psql variable cron_database is missing'
\set d1_679_cron_database_guard_statement 'SELECT 1 / 0 AS d1_679_cron_database_guard_failure'
\endif

:d1_679_cron_database_guard_statement;

\if :{?target_database}
SELECT CASE
    WHEN NULLIF(btrim(:'target_database'), '') = 'aawm_tristore'
         AND EXISTS (
             SELECT 1
             FROM pg_catalog.pg_database
             WHERE datname = :'target_database'
         )
        THEN 'true'
    ELSE 'false'
END AS d1_679_target_database_valid \gset

\if :d1_679_target_database_valid
\set d1_679_target_database_guard_statement 'SELECT 1'
\else
\echo 'D1-679 abort: target_database must name the existing aawm_tristore database'
\set d1_679_target_database_guard_statement 'SELECT 1 / 0 AS d1_679_target_database_guard_failure'
\endif
\else
\echo 'D1-679 abort: required psql variable target_database is missing'
\set d1_679_target_database_guard_statement 'SELECT 1 / 0 AS d1_679_target_database_guard_failure'
\endif

:d1_679_target_database_guard_statement;

\if :{?job_role}
SELECT CASE
    WHEN NULLIF(btrim(:'job_role'), '') IS NOT NULL
         AND EXISTS (
             SELECT 1
             FROM pg_catalog.pg_roles
             WHERE rolname = :'job_role'
         )
        THEN 'true'
    ELSE 'false'
END AS d1_679_job_role_valid \gset

\if :d1_679_job_role_valid
\set d1_679_job_role_guard_statement 'SELECT 1'
\else
\echo 'D1-679 abort: job_role is empty or does not resolve to an existing PostgreSQL role'
\set d1_679_job_role_guard_statement 'SELECT 1 / 0 AS d1_679_job_role_guard_failure'
\endif
\else
\echo 'D1-679 abort: required psql variable job_role is missing'
\set d1_679_job_role_guard_statement 'SELECT 1 / 0 AS d1_679_job_role_guard_failure'
\endif

:d1_679_job_role_guard_statement;

SELECT CASE
    WHEN EXISTS (
        SELECT 1
        FROM pg_catalog.pg_extension
        WHERE extname = 'pg_cron'
    )
        THEN 'true'
    ELSE 'false'
END AS d1_679_pg_cron_present \gset

\if :d1_679_pg_cron_present
\set d1_679_pg_cron_guard_statement 'SELECT 1'
\else
\echo 'D1-679 abort: pg_cron is not installed in the configured cron database'
\set d1_679_pg_cron_guard_statement 'SELECT 1 / 0 AS d1_679_pg_cron_guard_failure'
\endif

:d1_679_pg_cron_guard_statement;

SELECT CASE
    WHEN current_setting('cron.timezone', true) = 'UTC'
        THEN 'true'
    ELSE 'false'
END AS d1_679_cron_timezone_utc \gset

\if :d1_679_cron_timezone_utc
\set d1_679_cron_timezone_guard_statement 'SELECT 1'
\else
\echo 'D1-679 abort: cron.timezone must already be UTC'
\set d1_679_cron_timezone_guard_statement 'SELECT 1 / 0 AS d1_679_cron_timezone_guard_failure'
\endif

:d1_679_cron_timezone_guard_statement;

SELECT CASE
    WHEN EXISTS (
        SELECT 1
        FROM cron.job
        WHERE jobname = 'aawm_rate_limit_observation_archive_daily'
          AND (
              database IS DISTINCT FROM :'target_database'
              OR username IS DISTINCT FROM :'job_role'
              OR schedule IS DISTINCT FROM '17 3 * * *'
          )
    )
        THEN 'false'
    ELSE 'true'
END AS d1_679_daily_job_compatible \gset

\if :d1_679_daily_job_compatible
\set d1_679_daily_job_guard_statement 'SELECT 1'
\else
\echo 'D1-679 abort: the named daily job has a different target, role, or schedule'
\set d1_679_daily_job_guard_statement 'SELECT 1 / 0 AS d1_679_daily_job_guard_failure'
\endif

:d1_679_daily_job_guard_statement;

SELECT CASE
    WHEN EXISTS (
        SELECT 1
        FROM cron.job
        WHERE jobname = 'aawm_rate_limit_observation_archive_weekly'
          AND (
              database IS DISTINCT FROM :'target_database'
              OR username IS DISTINCT FROM :'job_role'
              OR schedule IS DISTINCT FROM '29 4 * * 0'
          )
    )
        THEN 'false'
    ELSE 'true'
END AS d1_679_weekly_job_compatible \gset

\if :d1_679_weekly_job_compatible
\set d1_679_weekly_job_guard_statement 'SELECT 1'
\else
\echo 'D1-679 abort: the named weekly job has a different target, role, or schedule'
\set d1_679_weekly_job_guard_statement 'SELECT 1 / 0 AS d1_679_weekly_job_guard_failure'
\endif

:d1_679_weekly_job_guard_statement;

SELECT cron.schedule_in_database(
    'aawm_rate_limit_observation_archive_daily',
    '17 3 * * *',
    $job$SELECT public.aawm_archive_rate_limit_observations_daily(1000);$job$,
    :'target_database',
    :'job_role'
)
WHERE NOT EXISTS (
    SELECT 1
    FROM cron.job
    WHERE jobname = 'aawm_rate_limit_observation_archive_daily'
      AND database = :'target_database'
      AND username = :'job_role'
);

SELECT cron.schedule_in_database(
    'aawm_rate_limit_observation_archive_weekly',
    '29 4 * * 0',
    $job$SELECT public.aawm_expire_rate_limit_observation_archive_weekly(5000);$job$,
    :'target_database',
    :'job_role'
)
WHERE NOT EXISTS (
    SELECT 1
    FROM cron.job
    WHERE jobname = 'aawm_rate_limit_observation_archive_weekly'
      AND database = :'target_database'
      AND username = :'job_role'
);

-- Rollback is operator-controlled: unschedule these exact job names first.
-- Retain the archive table and its data. Only after both jobs are stopped may
-- an authorized operator revoke/drop the functions; dropping archive data is
-- destructive and is not part of rollback.
--
-- SELECT cron.unschedule(jobid)
-- FROM cron.job
-- WHERE jobname IN (
--     'aawm_rate_limit_observation_archive_daily',
--     'aawm_rate_limit_observation_archive_weekly'
-- )
--   AND database = :'target_database';
