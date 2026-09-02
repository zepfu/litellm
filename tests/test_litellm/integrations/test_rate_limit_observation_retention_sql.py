"""Focused D1-679 SQL retention-contract tests.

These tests inspect the unapplied migration and pg_cron job definitions only.
They do not connect to PostgreSQL, install extensions, schedule jobs, or run
retention cleanup.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
SOURCE_SQL = (
    REPO_ROOT / "litellm/integrations/aawm_session_history/sql.py"
)
RETENTION_MIGRATION = (
    REPO_ROOT / "scripts/apply_rate_limit_observation_retention_2026_09_02.sql"
)
RETENTION_JOBS = (
    REPO_ROOT
    / "scripts/install_rate_limit_observation_retention_jobs_2026_09_02.sql"
)
RETENTION_DOCS = REPO_ROOT / "docs/aawm-session-history.md"


def _table_body(sql: str, table_name: str) -> str:
    match = re.search(
        rf"CREATE TABLE IF NOT EXISTS public\.{re.escape(table_name)} \(\n"
        r"(?P<body>.*?)\n\)\s*;?",
        sql,
        re.DOTALL,
    )
    assert match is not None, f"missing table definition for {table_name}"
    return match.group("body")


def _column_definitions(sql: str, table_name: str) -> tuple[tuple[str, str], ...]:
    definitions = []
    for line in _table_body(sql, table_name).splitlines():
        line = line.strip().rstrip(",")
        if not line:
            continue
        name, definition = line.split(None, 1)
        definitions.append((name, " ".join(definition.split())))
    return tuple(definitions)


def _function_body(sql: str, function_name: str) -> str:
    match = re.search(
        rf"CREATE OR REPLACE FUNCTION public\.{re.escape(function_name)}\(.*?"
        r"AS \$function\$(?P<body>.*?)\$function\$;",
        sql,
        re.DOTALL,
    )
    assert match is not None, f"missing function definition for {function_name}"
    return match.group("body")


def _migration_sql() -> str:
    return RETENTION_MIGRATION.read_text(encoding="utf-8")


def _jobs_sql() -> str:
    return RETENTION_JOBS.read_text(encoding="utf-8")


def _docs_retention_section() -> str:
    document = RETENTION_DOCS.read_text(encoding="utf-8")
    return document.split("### Rate-limit observation retention (D1-679)", 1)[
        1
    ].split("\n### ", 1)[0]


def _source_rate_limit_sql() -> str:
    module = ast.parse(SOURCE_SQL.read_text(encoding="utf-8"))
    for statement in module.body:
        if not isinstance(statement, ast.Assign):
            continue
        if not any(
            isinstance(target, ast.Name)
            and target.id == "_AAWM_RATE_LIMIT_OBSERVATIONS_TABLE_SQL"
            for target in statement.targets
        ):
            continue
        value = ast.literal_eval(statement.value)
        assert isinstance(value, str)
        return value
    raise AssertionError(
        "_AAWM_RATE_LIMIT_OBSERVATIONS_TABLE_SQL assignment not found"
    )


def test_should_preserve_every_source_column_and_source_id() -> None:
    source_definitions = _column_definitions(
        _source_rate_limit_sql(),
        "rate_limit_observations",
    )
    archive_definitions = _column_definitions(
        _migration_sql(),
        "rate_limit_observation_older",
    )

    assert [name for name, _ in archive_definitions] == [
        name for name, _ in source_definitions
    ]
    assert archive_definitions[0] == ("id", "BIGINT PRIMARY KEY")
    assert source_definitions[0][0] == "id"
    assert archive_definitions[1:] == source_definitions[1:]
    assert "BIGSERIAL" not in _table_body(
        _migration_sql(), "rate_limit_observation_older"
    )


def test_should_add_only_archive_identity_and_age_indexes() -> None:
    migration = _migration_sql()
    assert migration.count("CREATE INDEX IF NOT EXISTS") == 2
    assert (
        "rate_limit_observation_older_identity_latest_idx" in migration
    )
    assert (
        "provider, client, account_hash, quota_key, source, observed_at DESC, id DESC"
        in " ".join(migration.split())
    )
    assert "rate_limit_observation_older_observed_at_idx" in migration
    assert "observed_at ASC, id ASC" in " ".join(migration.split())
    assert "rate_limit_observations_identity_latest_idx" not in migration


def test_should_use_strict_daily_and_weekly_age_cutoffs() -> None:
    migration = _migration_sql()
    daily = _function_body(
        migration, "aawm_archive_rate_limit_observations_daily"
    )
    weekly = _function_body(
        migration, "aawm_expire_rate_limit_observation_archive_weekly"
    )

    assert "v_cutoff TIMESTAMPTZ := NOW() - INTERVAL '45 days';" in daily
    assert "WHERE observed_at < v_cutoff" in daily
    assert "<= v_cutoff" not in daily
    assert "v_cutoff TIMESTAMPTZ := NOW() - INTERVAL '6 months';" in weekly
    assert "WHERE observed_at < v_cutoff" in weekly
    assert "<= v_cutoff" not in weekly


def test_should_cap_each_retention_batch() -> None:
    migration = _migration_sql()
    daily = _function_body(
        migration, "aawm_archive_rate_limit_observations_daily"
    )
    weekly = _function_body(
        migration, "aawm_expire_rate_limit_observation_archive_weekly"
    )

    assert (
        "LEAST( GREATEST(COALESCE(p_batch_size, 1000), 1), 1000 )"
        in " ".join(daily.split())
    )
    assert (
        "LEAST( GREATEST(COALESCE(p_batch_size, 5000), 1), 5000 )"
        in " ".join(weekly.split())
    )
    assert daily.count("LIMIT v_batch_size") == 1
    assert weekly.count("LIMIT v_batch_size") == 1
    assert "FOR UPDATE SKIP LOCKED" in daily
    assert "FOR UPDATE SKIP LOCKED" in weekly


def test_should_make_archive_retries_idempotent_and_exact() -> None:
    daily = _function_body(
        _migration_sql(), "aawm_archive_rate_limit_observations_daily"
    )

    assert "ON CONFLICT (id) DO UPDATE" in daily
    assert "SET id = EXCLUDED.id" in daily
    assert "IS NOT DISTINCT FROM ROW(" in daily
    assert "RETURNING archive.id" in daily
    assert "ON CONFLICT (id) DO NOTHING" not in daily


def test_should_guard_source_presence_and_assign_minimal_function_privileges() -> None:
    migration = _migration_sql()

    assert "pg_catalog.to_regclass('public.rate_limit_observations')" in migration
    assert migration.index("d1_679_source_table_present") < migration.index(
        "BEGIN;"
    )
    assert (
        'ALTER TABLE public.rate_limit_observation_older OWNER TO :"owner_role";'
        in migration
    )
    assert (
        'ALTER FUNCTION public.aawm_archive_rate_limit_observations_daily(INTEGER)'
        in migration
    )
    assert (
        'ALTER FUNCTION public.aawm_expire_rate_limit_observation_archive_weekly(INTEGER)'
        in migration
    )
    assert (
        'GRANT EXECUTE ON FUNCTION public.aawm_archive_rate_limit_observations_daily(INTEGER)'
        in migration
    )
    assert (
        'GRANT EXECUTE ON FUNCTION public.aawm_expire_rate_limit_observation_archive_weekly(INTEGER)'
        in migration
    )
    assert 'TO :"job_role";' in migration
    assert "GRANT ALL" not in migration
    assert "cron.schedule" not in migration
    assert "CREATE EXTENSION" not in migration.upper()


def test_should_insert_and_confirm_archive_before_source_delete() -> None:
    daily = _function_body(
        _migration_sql(), "aawm_archive_rate_limit_observations_daily"
    )
    archive_insert = daily.index(
        "INSERT INTO public.rate_limit_observation_older"
    )
    archive_return = daily.index("RETURNING archive.id")
    source_delete = daily.index(
        "DELETE FROM public.rate_limit_observations AS source"
    )

    assert archive_insert < archive_return < source_delete
    assert "deleted AS (" in daily
    assert "DELETE FROM public.rate_limit_observations AS source" in daily
    assert "USING archived" in daily
    assert "WHERE source.id = archived.id" in daily
    assert "AND source.observed_at < v_cutoff" in daily


def test_should_serialize_daily_and_weekly_runs_with_one_transaction_lock() -> None:
    migration = _migration_sql()
    daily = _function_body(
        migration, "aawm_archive_rate_limit_observations_daily"
    )
    weekly = _function_body(
        migration, "aawm_expire_rate_limit_observation_archive_weekly"
    )
    lock = (
        "hashtextextended('aawm.rate_limit_observation_retention', 0)"
    )

    assert daily.count("pg_advisory_xact_lock(") == 1
    assert weekly.count("pg_advisory_xact_lock(") == 1
    assert daily.count(lock) == 1
    assert weekly.count(lock) == 1
    assert migration.count(lock) == 2
    assert "pg_try_advisory" not in migration


def test_should_delete_weekly_rows_from_archive_only() -> None:
    weekly = _function_body(
        _migration_sql(), "aawm_expire_rate_limit_observation_archive_weekly"
    )

    assert "DELETE FROM public.rate_limit_observation_older AS archive" in weekly
    assert "FROM public.rate_limit_observation_older" in weekly
    assert "rate_limit_observations" not in weekly
    assert "session_history" not in weekly


def test_should_define_explicit_utc_low_traffic_targeted_jobs() -> None:
    jobs = _jobs_sql()

    assert r"\set ON_ERROR_STOP on" in jobs
    assert "cron.schedule_in_database(" in jobs
    assert "'17 3 * * *'" in jobs
    assert "'29 4 * * 0'" in jobs
    assert "aawm_rate_limit_observation_archive_daily" in jobs
    assert "aawm_rate_limit_observation_archive_weekly" in jobs
    assert ":'target_database'" in jobs
    assert ":'job_role'" in jobs
    assert "current_setting('cron.timezone', true) = 'UTC'" in jobs
    assert "CREATE EXTENSION" not in jobs.upper()
    assert "ALTER SYSTEM" not in jobs.upper()
    assert "schedule IS DISTINCT FROM" in jobs
    assert "SELECT public.aawm_archive_rate_limit_observations_daily(1000)" in jobs
    assert (
        "SELECT public.aawm_expire_rate_limit_observation_archive_weekly(5000)"
        in jobs
    )


def test_should_keep_job_definition_rollback_non_destructive_and_archive_only() -> None:
    jobs = _jobs_sql()
    assert "cron.unschedule(jobid)" in jobs
    assert "DROP TABLE" not in jobs.upper()
    assert "DELETE FROM public.rate_limit_observations" not in jobs
    assert "Retain the archive table and its data" in jobs


def test_should_document_unapplied_retention_install_contract() -> None:
    section = _docs_retention_section()
    normalized = " ".join(section.split())

    for fragment in (
        "`public.rate_limit_observation_older`",
        "`aawm.rate_limit_observation_retention`",
        "`17 3 * * *` UTC",
        "`29 4 * * 0` UTC",
        "`target_database=aawm_tristore`",
        "`cron.timezone` must already be `UTC`",
        "The definitions are prepared but unapplied.",
        "apply_rate_limit_observation_retention_2026_09_02.sql",
        "install_rate_limit_observation_retention_jobs_2026_09_02.sql",
        "D1-678 remains the owner",
        "it never deletes from `public.rate_limit_observations`",
    ):
        assert fragment in normalized, f"missing docs contract: {fragment}"
