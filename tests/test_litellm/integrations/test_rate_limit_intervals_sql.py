from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
LEGACY_SCRIPT = (
    REPO_ROOT / "scripts" / "apply_rate_limit_intervals_mview_2026_05_23.sql"
)

XAI_WEEKLY_100_PCT_EXCEPTION = (
    "provider = 'xai'\n"
    "              AND quota_key = 'xai_grok_build_weekly_credits:credits'"
)


def test_legacy_rate_limit_intervals_script_includes_weekly_credits_key() -> None:
    sql = LEGACY_SCRIPT.read_text(encoding="utf-8")

    assert "'xai_grok_build_weekly_credits:credits'" in sql
    assert "xai_grok_build_weekly_credits:credits']) THEN 'weekly'" in sql


def test_xai_grok_weekly_credits_allows_hundred_pct_remaining_legacy_script() -> None:
    sql = LEGACY_SCRIPT.read_text(encoding="utf-8")

    assert XAI_WEEKLY_100_PCT_EXCEPTION in sql
    assert "remaining_pct < 100" in sql


def test_anthropic_7d_oi_quota_key_allowed_in_legacy_script() -> None:
    sql = LEGACY_SCRIPT.read_text(encoding="utf-8")
    assert "'anthropic_unified_7d_oi:7d_oi'" in sql


def test_anthropic_7d_oi_quota_key_mapped_weekly_overage_included_legacy() -> None:
    sql = LEGACY_SCRIPT.read_text(encoding="utf-8")
    assert (
        "WHEN quota_key = ANY (ARRAY['anthropic_unified_7d_oi:7d_oi']) THEN 'weekly_overage_included'"
        in sql
    )


def test_anthropic_7d_sonnet_weekly_special_mapping_preserved() -> None:
    sql = LEGACY_SCRIPT.read_text(encoding="utf-8")
    assert "'anthropic_unified_7d_sonnet:7d_sonnet'" in sql
    assert "THEN 'weekly_special'" in sql


def _assert_nullable_model_partition_and_column_unique_index(sql: str) -> None:
    """Enforce concurrent-refresh eligibility and nullable-model uniqueness."""
    assert (
        "PARTITION BY provider, COALESCE(model, ''::text), quota_key, quota_type" in sql
    )
    assert (
        sql.count(
            "PARTITION BY provider, COALESCE(model, ''::text), quota_key, quota_type"
        )
        >= 2
    )

    unique_marker = "CREATE UNIQUE INDEX rate_limit_intervals_unique_idx"
    assert unique_marker in sql
    unique_block = sql.split(unique_marker, 1)[1]
    # Stop at next statement terminator after the index definition.
    unique_block = unique_block.split(";", 1)[0]

    # Column-only unique index (no expressions) over MV columns only.
    assert "provider," in unique_block
    assert "model," in unique_block
    assert "quota_key," in unique_block
    assert "quota_type," in unique_block
    assert "fromdate," in unique_block
    assert "expected_reset_at," in unique_block
    assert "remaining_pct" in unique_block
    assert "NULLS NOT DISTINCT" in unique_block
    assert "COALESCE" not in unique_block
    assert "WHERE" not in unique_block.upper()


def test_legacy_script_unique_index_is_column_only_with_nulls_not_distinct() -> None:
    sql = LEGACY_SCRIPT.read_text(encoding="utf-8")
    _assert_nullable_model_partition_and_column_unique_index(sql)


# --- D1-491 concurrent unique-index migration source contract ---

D1_491_SCRIPT = (
    REPO_ROOT / "scripts" / "apply_rate_limit_intervals_concurrent_index_2026_07_25.sql"
)

D1_491_TEMP_INDEX = "rate_limit_intervals_unique_idx_d1_491"
D1_491_CANONICAL_INDEX = "rate_limit_intervals_unique_idx"
D1_491_LOCK_KEYS = (
    "hashtext('dashboard-shell')",
    "hashtext('materialized-view-maintenance')",
)
D1_491_SEVEN_COLUMNS = (
    "provider",
    "model",
    "quota_key",
    "quota_type",
    "fromdate",
    "expected_reset_at",
    "remaining_pct",
)


def _d1_491_sql() -> str:
    return D1_491_SCRIPT.read_text(encoding="utf-8")


def _statement_index(sql: str, needle: str) -> int:
    idx = sql.find(needle)
    assert idx >= 0, f"missing required fragment: {needle!r}"
    return idx


def test_d1_491_migration_script_exists() -> None:
    assert D1_491_SCRIPT.is_file()


def test_d1_491_psql_fail_fast_on_error_stop() -> None:
    sql = _d1_491_sql()
    assert r"\set ON_ERROR_STOP on" in sql
    assert sql.index(r"\set ON_ERROR_STOP on") < sql.index("pg_advisory_lock")


def test_d1_491_acquires_dashboard_maintenance_advisory_lock_keys() -> None:
    sql = _d1_491_sql()
    assert "pg_advisory_lock(" in sql
    for key in D1_491_LOCK_KEYS:
        assert key in sql
    # Blocking lock (not try-lock) so scheduled try-lock jobs skip during repair.
    assert "pg_try_advisory_lock" not in sql
    lock_idx = sql.index("pg_advisory_lock(")
    unlock_idx = sql.index("pg_advisory_unlock(")
    assert lock_idx < unlock_idx
    # Unlock only after successful refresh + rename path.
    assert unlock_idx > sql.index("REFRESH MATERIALIZED VIEW CONCURRENTLY")
    assert unlock_idx > sql.index(
        f"RENAME TO {D1_491_CANONICAL_INDEX}"
    ) or unlock_idx > sql.rindex(f"RENAME TO {D1_491_CANONICAL_INDEX}")


def test_d1_491_duplicate_guards_before_mutation() -> None:
    sql = _d1_491_sql()
    raw_guard = "raw column-key duplicate"
    null_empty_guard = "null-vs-empty model collision"
    assert raw_guard in sql
    assert null_empty_guard in sql

    mutation_idx = min(
        _statement_index(sql, f"DROP INDEX IF EXISTS public.{D1_491_TEMP_INDEX};"),
        _statement_index(sql, f"CREATE UNIQUE INDEX {D1_491_TEMP_INDEX}"),
    )
    assert sql.index(raw_guard) < mutation_idx
    assert sql.index(null_empty_guard) < mutation_idx

    # Raw guard groups on the seven actual MV columns.
    raw_block = sql[sql.index(raw_guard) - 400 : sql.index(raw_guard) + 200]
    for column in D1_491_SEVEN_COLUMNS:
        assert column in raw_block or column in sql

    # Null-vs-empty collision uses remaining key fields and model null/empty.
    assert "model IS NULL" in sql
    assert "model = ''" in sql
    assert "window partitions coalesce" in sql.lower() or "coalesce" in sql.lower()


def test_d1_491_eligible_temp_unique_index_shape() -> None:
    sql = _d1_491_sql()
    create_marker = f"CREATE UNIQUE INDEX {D1_491_TEMP_INDEX}"
    assert create_marker in sql
    create_block = sql.split(create_marker, 1)[1].split(";", 1)[0]
    for column in D1_491_SEVEN_COLUMNS:
        assert column in create_block
    assert "NULLS NOT DISTINCT" in create_block
    assert "COALESCE" not in create_block
    assert "WHERE" not in create_block.upper()
    assert "ON public.rate_limit_intervals" in create_block


def test_d1_491_drops_only_prior_temporary_index_before_create() -> None:
    sql = _d1_491_sql()
    drop_temp = f"DROP INDEX IF EXISTS public.{D1_491_TEMP_INDEX};"
    create_temp = f"CREATE UNIQUE INDEX {D1_491_TEMP_INDEX}"
    assert drop_temp in sql
    assert sql.index(drop_temp) < sql.index(create_temp)


def test_d1_491_refresh_before_old_canonical_drop() -> None:
    sql = _d1_491_sql()
    refresh = "REFRESH MATERIALIZED VIEW CONCURRENTLY public.rate_limit_intervals;"
    # Trailing semicolon prevents matching the earlier temp-index drop
    # (...unique_idx_d1_491) as a prefix of the canonical index name.
    drop_old = f"DROP INDEX IF EXISTS public.{D1_491_CANONICAL_INDEX};"
    create_temp = f"CREATE UNIQUE INDEX {D1_491_TEMP_INDEX}"
    assert refresh in sql
    assert drop_old in sql
    assert sql.index(create_temp) < sql.index(refresh) < sql.index(drop_old)


def test_d1_491_canonical_rename_after_successful_refresh() -> None:
    sql = _d1_491_sql()
    refresh = "REFRESH MATERIALIZED VIEW CONCURRENTLY public.rate_limit_intervals"
    # Allow flexible whitespace around RENAME.
    assert f"RENAME TO {D1_491_CANONICAL_INDEX}" in sql
    assert sql.index(refresh) < sql.index(f"RENAME TO {D1_491_CANONICAL_INDEX}")
    assert "ANALYZE public.rate_limit_intervals" in sql
    assert sql.index(f"RENAME TO {D1_491_CANONICAL_INDEX}") < sql.index(
        "ANALYZE public.rate_limit_intervals"
    )


def test_d1_491_no_drop_materialized_view_or_cron_mutation() -> None:
    sql = _d1_491_sql()
    assert "DROP MATERIALIZED VIEW" not in sql.upper()
    assert "CREATE MATERIALIZED VIEW" not in sql.upper()
    assert "cron.schedule" not in sql
    assert "cron.unschedule" not in sql
    assert "cron.alter_job" not in sql
    assert "cron.job" not in sql


def test_d1_491_migration_ordering_contract() -> None:
    sql = _d1_491_sql()
    markers = [
        r"\set ON_ERROR_STOP on",
        "pg_advisory_lock(",
        "raw column-key duplicate",
        "null-vs-empty model collision",
        f"DROP INDEX IF EXISTS public.{D1_491_TEMP_INDEX};",
        f"CREATE UNIQUE INDEX {D1_491_TEMP_INDEX}",
        "REFRESH MATERIALIZED VIEW CONCURRENTLY public.rate_limit_intervals;",
        # Exact canonical drop (semicolon) avoids prefix match on temp index.
        f"DROP INDEX IF EXISTS public.{D1_491_CANONICAL_INDEX};",
        f"RENAME TO {D1_491_CANONICAL_INDEX}",
        "ANALYZE public.rate_limit_intervals;",
        "pg_advisory_unlock(",
    ]
    positions = [_statement_index(sql, marker) for marker in markers]
    assert positions == sorted(positions)


def test_d1_491_final_readonly_index_freshness_summary() -> None:
    sql = _d1_491_sql()
    unlock_idx = sql.index("pg_advisory_unlock(")
    # Summary after unlock path: index name + freshness signals.
    tail = sql[unlock_idx:]
    assert "rate_limit_intervals_unique_idx" in tail
    assert (
        "max_fromdate" in tail or "max(r.fromdate)" in tail or "max(fromdate)" in tail
    )
    assert "pg_get_indexdef" in tail or "index_def" in tail
