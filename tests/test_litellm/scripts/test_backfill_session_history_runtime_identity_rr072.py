"""RR-072: paginated ClickHouse fetch + single-resolved auth + batched inserts."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

_REPO = Path(__file__).resolve().parents[3]
_SCRIPT = _REPO / "scripts" / "backfill_session_history_runtime_identity.py"


def _load_module():
    name = "backfill_session_history_runtime_identity_rr072"
    if name in sys.modules:
        del sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, _SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_clickhouse_page_query_includes_limit_and_keyset_cursor() -> None:
    mod = _load_module()
    first = mod._build_observation_identity_page_query(limit=100)
    assert "LIMIT 100" in first
    assert "ORDER BY o.id ASC" in first
    assert "o.id >" not in first
    assert "FORMAT JSONEachRow" in first

    paged = mod._build_observation_identity_page_query(limit=50, cursor_id="obs-10")
    assert "LIMIT 50" in paged
    assert "o.id > 'obs-10'" in paged
    assert "ORDER BY o.id ASC" in paged


def test_sql_string_literal_escapes_quotes_and_backslashes() -> None:
    mod = _load_module()
    assert mod._sql_string_literal("plain") == "'plain'"
    assert mod._sql_string_literal("a'b") == r"'a\'b'"
    assert mod._sql_string_literal(r"a\b") == r"'a\\b'"


def test_request_clickhouse_rows_uses_passed_auth_without_reresolve() -> None:
    mod = _load_module()
    auth = {
        "url": "http://127.0.0.1:8123/",
        "user": "u",
        "password": "p",
        "url_input": None,
        "url_source": "test",
        "user_source": "test",
        "password_source": "test",
    }

    class _Resp:
        def read(self) -> bytes:
            return b'{"observation_id":"1"}\n'

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    with patch.object(
        mod, "_resolve_clickhouse_auth_sources"
    ) as resolve_mock, patch.object(
        mod, "urlopen", return_value=_Resp()
    ) as urlopen_mock:
        rows = mod._request_clickhouse_rows(auth, "SELECT 1", timeout_seconds=5)

    resolve_mock.assert_not_called()
    assert rows == [{"observation_id": "1"}]
    # Authorization header built from the provided auth dict
    request = urlopen_mock.call_args.args[0]
    assert "Authorization" in request.headers


def test_clickhouse_auth_header_accepts_auth_dict_not_args() -> None:
    mod = _load_module()
    header = mod._clickhouse_auth_header({"user": "alice", "password": "s3cret"})
    assert header.startswith("Basic ")
    # Must not re-resolve from args
    import inspect

    sig = inspect.signature(mod._clickhouse_auth_header)
    assert list(sig.parameters) == ["auth"]


def test_fetch_observation_identities_pages_until_short_page() -> None:
    mod = _load_module()
    args = SimpleNamespace(
        clickhouse_page_size=2,
        clickhouse_max_pages=None,
        clickhouse_resume_after_id=None,
        clickhouse_timeout_seconds=30,
        derive_environment_from_port=False,
        correct_default_environment_from_port=False,
    )
    auth = {"url": "http://127.0.0.1:8123/", "user": "u", "password": "p"}

    pages = [
        [
            {
                "observation_id": "obs-1",
                "trace_id": "tr-1",
                "observation_metadata": {"client_name": "claude-cli"},
                "requester_metadata": None,
                "trace_environment": "dev",
            },
            {
                "observation_id": "obs-2",
                "trace_id": "tr-1",
                "observation_metadata": {"client_name": "claude-cli"},
                "requester_metadata": None,
                "trace_environment": "dev",
            },
        ],
        [
            {
                "observation_id": "obs-3",
                "trace_id": "tr-2",
                "observation_metadata": {"client_name": "codex"},
                "requester_metadata": None,
                "trace_environment": "prod",
            },
        ],
    ]
    queries: list[str] = []

    def fake_request(auth_arg, query, *, timeout_seconds):  # noqa: ANN001
        assert auth_arg is auth
        queries.append(query)
        return pages[len(queries) - 1]

    with patch.object(mod, "_request_clickhouse_rows", side_effect=fake_request):
        identities = mod._fetch_observation_identities(args, auth)

    assert len(queries) == 2
    assert "LIMIT 2" in queries[0]
    assert "o.id >" not in queries[0]
    assert "o.id > 'obs-2'" in queries[1]
    assert len(identities) == 3
    assert {item.observation_id for item in identities} == {"obs-1", "obs-2", "obs-3"}
    assert args._clickhouse_pages_fetched == 2
    assert args._clickhouse_last_cursor_id == "obs-3"


def test_fetch_observation_identities_honors_max_pages_and_resume() -> None:
    mod = _load_module()
    args = SimpleNamespace(
        clickhouse_page_size=2,
        clickhouse_max_pages=1,
        clickhouse_resume_after_id="obs-0",
        clickhouse_timeout_seconds=30,
        derive_environment_from_port=False,
        correct_default_environment_from_port=False,
    )
    auth = {"url": "http://x/", "user": "u", "password": "p"}
    queries: list[str] = []

    def fake_request(auth_arg, query, *, timeout_seconds):  # noqa: ANN001
        queries.append(query)
        return [
            {
                "observation_id": "obs-1",
                "trace_id": "tr-1",
                "observation_metadata": {"client_name": "claude-cli"},
                "requester_metadata": None,
                "trace_environment": "dev",
            },
            {
                "observation_id": "obs-2",
                "trace_id": "tr-1",
                "observation_metadata": {"client_name": "claude-cli"},
                "requester_metadata": None,
                "trace_environment": "dev",
            },
        ]

    with patch.object(mod, "_request_clickhouse_rows", side_effect=fake_request):
        identities = mod._fetch_observation_identities(args, auth)

    assert len(queries) == 1
    assert "o.id > 'obs-0'" in queries[0]
    assert len(identities) == 2
    assert args._clickhouse_pages_fetched == 1
    assert args._clickhouse_last_cursor_id == "obs-2"


def test_iter_observation_identity_pages_keeps_peak_page_bound() -> None:
    """Production path must not retain O(total rows) in the page iterator."""
    mod = _load_module()
    args = SimpleNamespace(
        clickhouse_page_size=2,
        clickhouse_max_pages=None,
        clickhouse_resume_after_id=None,
        clickhouse_timeout_seconds=30,
        derive_environment_from_port=False,
        correct_default_environment_from_port=False,
    )
    auth = {"url": "http://127.0.0.1:8123/", "user": "u", "password": "p"}
    pages = [
        [
            {
                "observation_id": "obs-1",
                "trace_id": "tr-1",
                "observation_metadata": {"client_name": "claude-cli"},
                "requester_metadata": None,
                "trace_environment": "dev",
            },
            {
                "observation_id": "obs-2",
                "trace_id": "tr-1",
                "observation_metadata": {"client_name": "claude-cli"},
                "requester_metadata": None,
                "trace_environment": "dev",
            },
        ],
        [
            {
                "observation_id": "obs-3",
                "trace_id": "tr-2",
                "observation_metadata": {"client_name": "codex"},
                "requester_metadata": None,
                "trace_environment": "prod",
            },
            {
                "observation_id": "obs-4",
                "trace_id": "tr-2",
                "observation_metadata": {"client_name": "codex"},
                "requester_metadata": None,
                "trace_environment": "prod",
            },
        ],
        [
            {
                "observation_id": "obs-5",
                "trace_id": "tr-3",
                "observation_metadata": {"client_name": "gemini"},
                "requester_metadata": None,
                "trace_environment": "dev",
            },
        ],
    ]
    call_count = {"n": 0}

    def fake_request(auth_arg, query, *, timeout_seconds):  # noqa: ANN001
        idx = call_count["n"]
        call_count["n"] += 1
        return pages[idx]

    retained_pages: list[list[object]] = []
    peak = 0
    total = 0
    with patch.object(mod, "_request_clickhouse_rows", side_effect=fake_request):
        for page in mod._iter_observation_identity_pages(args, auth):
            # Emulate main(): insert-then-discard each page before the next yield.
            retained_pages.append(page)
            peak = max(peak, len(page))
            total += len(page)
            retained_pages.clear()

    assert call_count["n"] == 3
    assert total == 5
    assert peak == 2  # never larger than page_size
    assert args._clickhouse_pages_fetched == 3
    assert args._clickhouse_last_cursor_id == "obs-5"


def test_clickhouse_max_pages_zero_is_rejected() -> None:
    mod = _load_module()
    args = SimpleNamespace(
        clickhouse_page_size=2,
        clickhouse_max_pages=0,
        clickhouse_resume_after_id=None,
        clickhouse_timeout_seconds=30,
        derive_environment_from_port=False,
        correct_default_environment_from_port=False,
    )
    auth = {"url": "http://x/", "user": "u", "password": "p"}
    with pytest.raises(ValueError, match="clickhouse_max_pages must be >= 1"):
        list(mod._iter_observation_identity_pages(args, auth))


def test_clickhouse_max_pages_negative_is_rejected() -> None:
    mod = _load_module()
    args = SimpleNamespace(
        clickhouse_page_size=2,
        clickhouse_max_pages=-3,
        clickhouse_resume_after_id=None,
        clickhouse_timeout_seconds=30,
        derive_environment_from_port=False,
        correct_default_environment_from_port=False,
    )
    auth = {"url": "http://x/", "user": "u", "password": "p"}
    with pytest.raises(ValueError, match="clickhouse_max_pages must be >= 1"):
        list(mod._iter_observation_identity_pages(args, auth))


def test_main_rejects_non_positive_clickhouse_max_pages() -> None:
    mod = _load_module()
    with patch.object(
        sys,
        "argv",
        ["prog", "--clickhouse-max-pages", "0"],
    ):
        args = mod._parse_args()
    assert args.clickhouse_max_pages == 0

    with patch.object(mod, "_parse_args", return_value=args), patch.object(
        mod, "_resolve_clickhouse_auth_sources", return_value={"url": "u", "user": "u", "password": "p"}
    ), patch.object(mod, "_preflight_clickhouse_connection"), patch.object(
        mod, "_postgres_dsn_from_args", return_value="postgresql://example"
    ):
        with pytest.raises(SystemExit, match="--clickhouse-max-pages must be >= 1"):
            mod.main()


def test_insert_temp_rows_batches_executemany() -> None:
    mod = _load_module()
    cur = MagicMock()
    observations = [
        mod.ObservationIdentity(
            observation_id=f"obs-{i}",
            trace_id=f"tr-{i // 2}",
            litellm_environment="dev",
            litellm_version="1.0",
            litellm_fork_version=None,
            litellm_wheel_versions=None,
            client_name="claude-cli",
            client_version="1",
            client_user_agent=None,
            port_environment=None,
            port_host=None,
        )
        for i in range(5)
    ]
    traces = mod._derive_trace_identities(observations)

    mod._insert_temp_rows(cur, observations, traces, insert_batch_size=2)

    # CREATE TEMP x2 + batched inserts for obs and traces
    assert cur.execute.call_count == 2
    # observations: ceil(5/2)=3 batches; traces depend on unique traces
    assert cur.executemany.call_count >= 3
    obs_batch_sizes = [
        len(call.args[1])
        for call in cur.executemany.call_args_list
        if "tmp_session_history_runtime_identity_obs" in call.args[0]
    ]
    assert obs_batch_sizes == [2, 2, 1]


def test_populate_trace_identities_sql_matches_python_helper_semantics() -> None:
    """SQL aggregate path must keep only unanimous non-null field values."""
    mod = _load_module()
    observations = [
        mod.ObservationIdentity(
            observation_id="obs-1",
            trace_id="tr-agree",
            litellm_environment="dev",
            litellm_version="1.0",
            litellm_fork_version=None,
            litellm_wheel_versions=None,
            client_name="claude-cli",
            client_version="1",
            client_user_agent=None,
            port_environment=None,
            port_host=None,
        ),
        mod.ObservationIdentity(
            observation_id="obs-2",
            trace_id="tr-agree",
            litellm_environment="dev",
            litellm_version="1.0",
            litellm_fork_version=None,
            litellm_wheel_versions=None,
            client_name="claude-cli",
            client_version="1",
            client_user_agent=None,
            port_environment=None,
            port_host=None,
        ),
        mod.ObservationIdentity(
            observation_id="obs-3",
            trace_id="tr-conflict",
            litellm_environment="dev",
            litellm_version=None,
            litellm_fork_version=None,
            litellm_wheel_versions=None,
            client_name="claude-cli",
            client_version=None,
            client_user_agent=None,
            port_environment=None,
            port_host=None,
        ),
        mod.ObservationIdentity(
            observation_id="obs-4",
            trace_id="tr-conflict",
            litellm_environment="prod",
            litellm_version=None,
            litellm_fork_version=None,
            litellm_wheel_versions=None,
            client_name="codex",
            client_version=None,
            client_user_agent=None,
            port_environment=None,
            port_host=None,
        ),
    ]
    expected = {
        item.trace_id: item for item in mod._derive_trace_identities(observations)
    }
    assert set(expected) == {"tr-agree"}
    assert expected["tr-agree"].client_name == "claude-cli"
    assert expected["tr-agree"].litellm_environment == "dev"

    # Capture the SQL shape used by the production streaming path.
    sql = mod._populate_trace_identities_from_obs_temp.__code__.co_consts
    # Ensure the function body still references consensus filters.
    source = mod._populate_trace_identities_from_obs_temp.__doc__ or ""
    assert "unanimous" in source or "agree" in source
    assert any(
        isinstance(const, str) and "count(DISTINCT" in const for const in sql
    )


def test_iter_batches_splits_sequence() -> None:
    mod = _load_module()
    assert list(mod._iter_batches([1, 2, 3, 4, 5], 2)) == [[1, 2], [3, 4], [5]]
    assert list(mod._iter_batches([], 10)) == []


def test_parse_args_exposes_pagination_flags() -> None:
    mod = _load_module()
    with patch.object(
        sys,
        "argv",
        [
            "prog",
            "--clickhouse-page-size",
            "25",
            "--clickhouse-max-pages",
            "3",
            "--clickhouse-resume-after-id",
            "obs-9",
            "--insert-batch-size",
            "50",
        ],
    ):
        args = mod._parse_args()
    assert args.clickhouse_page_size == 25
    assert args.clickhouse_max_pages == 3
    assert args.clickhouse_resume_after_id == "obs-9"
    assert args.insert_batch_size == 50


# ---------------------------------------------------------------------------
# Fill-only apply SQL / parameter behavior
# ---------------------------------------------------------------------------


def test_create_temp_identity_tables_use_on_commit_drop() -> None:
    """Temp identity tables must be session-scoped and dropped on commit/rollback."""
    mod = _load_module()
    cur = MagicMock()
    mod._create_temp_identity_tables(cur)

    assert cur.execute.call_count == 2
    sqls = [call.args[0] for call in cur.execute.call_args_list]
    assert all("ON COMMIT DROP" in sql for sql in sqls)
    assert any("tmp_session_history_runtime_identity_obs" in sql for sql in sqls)
    assert any("tmp_session_history_runtime_identity_trace" in sql for sql in sqls)
    # Temporary tables, not permanent side tables.
    assert all("CREATE TEMPORARY TABLE" in sql for sql in sqls)


def test_observation_insert_param_order_matches_temp_table_columns() -> None:
    """executemany value tuples must follow the temp table column order."""
    mod = _load_module()
    cur = MagicMock()
    item = mod.ObservationIdentity(
        observation_id="obs-1",
        trace_id="tr-1",
        litellm_environment="dev",
        litellm_version="1.0",
        litellm_fork_version="f1",
        litellm_wheel_versions='{"a":"1"}',
        client_name="claude-cli",
        client_version="2",
        client_user_agent="ua",
        port_environment="dev",
        port_host="127.0.0.1:4001",
    )
    mod._insert_observation_identity_rows(cur, [item], insert_batch_size=10)

    assert cur.executemany.call_count == 1
    sql, rows = cur.executemany.call_args.args
    assert "tmp_session_history_runtime_identity_obs" in sql
    # 11 placeholders for 11 columns
    assert sql.count("%s") == 11
    assert rows == [
        (
            "obs-1",
            "tr-1",
            "dev",
            "1.0",
            "f1",
            '{"a":"1"}',
            "claude-cli",
            "2",
            "ua",
            "dev",
            "127.0.0.1:4001",
        )
    ]


def test_trace_insert_param_order_matches_temp_table_columns() -> None:
    mod = _load_module()
    cur = MagicMock()
    item = mod.TraceIdentity(
        trace_id="tr-1",
        litellm_environment="prod",
        litellm_version="2.0",
        litellm_fork_version="f2",
        litellm_wheel_versions='{"b":"2"}',
        client_name="codex",
        client_version="3",
        client_user_agent="ua2",
        port_environment="prod",
    )
    mod._insert_trace_identity_rows(cur, [item], insert_batch_size=10)

    assert cur.executemany.call_count == 1
    sql, rows = cur.executemany.call_args.args
    assert "tmp_session_history_runtime_identity_trace" in sql
    assert sql.count("%s") == 9
    assert rows == [
        (
            "tr-1",
            "prod",
            "2.0",
            "f2",
            '{"b":"2"}',
            "codex",
            "3",
            "ua2",
            "prod",
        )
    ]


def test_insert_helpers_skip_empty_lists_without_executemany() -> None:
    mod = _load_module()
    cur = MagicMock()
    mod._insert_observation_identity_rows(cur, [], insert_batch_size=10)
    mod._insert_trace_identity_rows(cur, [], insert_batch_size=10)
    cur.executemany.assert_not_called()


def test_matched_cte_is_fill_only_and_does_not_overwrite_existing() -> None:
    """Apply SQL must only fill empty/null columns (fill-only), not overwrite."""
    mod = _load_module()
    sql = mod._matched_cte(False)

    # Fill-only predicates: empty/null current value AND non-null candidate.
    for field in (
        "litellm_environment",
        "litellm_version",
        "litellm_fork_version",
        "client_name",
        "client_version",
        "client_user_agent",
    ):
        assert f"coalesce({field}, '') = ''" in sql
        assert f"n_{field} IS NOT NULL" in sql
        assert f"change_{field}" in sql

    # Wheel versions: null or empty jsonb object only.
    assert (
        "(litellm_wheel_versions IS NULL OR litellm_wheel_versions = '{}'::jsonb)"
        in sql
        or "(litellm_wheel_versions IS NULL OR litellm_wheel_versions = '{{}}'::jsonb)"
        in sql
    )

    # Must not use unconditional overwrite predicates like always-true change flags.
    assert "change_litellm_environment = true" not in sql.lower()
    # Default-env correction clause is opt-in only.
    assert "litellm_environment, '') = 'default'" not in sql

    sql_with_default = mod._matched_cte(True)
    assert "litellm_environment, '') = 'default'" in sql_with_default
    assert "n_port_environment IS NOT NULL" in sql_with_default


def test_apply_sql_uses_case_when_change_flags_only() -> None:
    """_apply must gate every column write on the fill-only change_* flags."""
    mod = _load_module()
    cur = MagicMock()
    cur.description = [SimpleNamespace(name="count")]
    cur.fetchone.return_value = (3,)

    updated = mod._apply(cur, correct_default_environment_from_port=False)
    assert updated == 3
    assert cur.execute.call_count == 1
    sql = cur.execute.call_args.args[0]

    assert "UPDATE public.session_history" in sql
    # Every identity column assignment is gated.
    for field in (
        "litellm_environment",
        "litellm_version",
        "litellm_fork_version",
        "litellm_wheel_versions",
        "client_name",
        "client_version",
        "client_user_agent",
    ):
        assert f"change_{field}" in sql
        assert f"CASE WHEN changes.change_{field}" in sql

    # WHERE requires at least one fill-only change flag.
    assert "changes.change_litellm_environment OR" in sql
    assert "changes.change_client_user_agent" in sql
    # Metadata merge is also gated (jsonb_strip_nulls of change-flagged keys).
    assert "jsonb_strip_nulls" in sql
    assert "jsonb_build_object" in sql


def test_dry_run_sql_counts_fill_only_change_flags() -> None:
    mod = _load_module()
    cur = MagicMock()
    cur.description = [
        SimpleNamespace(name=name)
        for name in (
            "matched_rows",
            "litellm_environment",
            "litellm_version",
            "litellm_fork_version",
            "litellm_wheel_versions",
            "client_name",
            "client_version",
            "client_user_agent",
        )
    ]
    cur.fetchone.return_value = (10, 1, 2, 0, 0, 3, 0, 1)

    result = mod._dry_run(cur, correct_default_environment_from_port=True)
    assert result["matched_rows"] == 10
    assert result["client_name"] == 3
    sql = cur.execute.call_args.args[0]
    assert "count(*) FILTER (WHERE change_litellm_environment)" in sql
    assert "FROM changes" in sql
    # Opt-in default correction is threaded through.
    assert "litellm_environment, '') = 'default'" in sql


# ---------------------------------------------------------------------------
# Dry-run rollback / ON COMMIT DROP semantics (mockable)
# ---------------------------------------------------------------------------


def test_main_dry_run_rolls_back_and_skips_apply() -> None:
    """Default mode is dry-run: compute would_fill, rollback, never _apply."""
    mod = _load_module()

    args = SimpleNamespace(
        apply=False,
        target_db_name="aawm_tristore",
        correct_default_environment_from_port=False,
        derive_environment_from_port=False,
        clickhouse_page_size=1000,
        clickhouse_max_pages=None,
        clickhouse_resume_after_id=None,
        clickhouse_timeout_seconds=30,
        insert_batch_size=1000,
        pg_dsn=None,
        pg_host=None,
        pg_port=None,
        pg_user=None,
        pg_password=None,
        clickhouse_url=None,
        clickhouse_user=None,
        clickhouse_password=None,
    )

    cur = MagicMock()
    cur.fetchone.side_effect = [
        ("aawm_tristore",),  # current_database
    ]
    # _dry_run will call execute/fetchone; we patch helpers below.

    conn = MagicMock()
    conn.cursor.return_value.__enter__.return_value = cur
    conn.cursor.return_value.__exit__.return_value = False
    conn.__enter__.return_value = conn
    conn.__exit__.return_value = False

    auth = {
        "url": "http://127.0.0.1:8123/",
        "user": "u",
        "password": "p",
        "url_input": None,
        "url_source": "test",
        "user_source": "test",
        "password_source": "test",
    }

    dry_payload = {
        "matched_rows": 2,
        "litellm_environment": 1,
        "litellm_version": 0,
        "litellm_fork_version": 0,
        "litellm_wheel_versions": 0,
        "client_name": 1,
        "client_version": 0,
        "client_user_agent": 0,
    }

    with patch.object(mod, "_parse_args", return_value=args), patch.object(
        mod, "_resolve_clickhouse_auth_sources", return_value=auth
    ), patch.object(mod, "_preflight_clickhouse_connection"), patch.object(
        mod, "_postgres_dsn_from_args", return_value="postgresql://example/aawm_tristore"
    ), patch.object(mod.psycopg, "connect", return_value=conn), patch.object(
        mod, "_create_temp_identity_tables"
    ) as create_tmp, patch.object(
        mod, "_iter_observation_identity_pages", return_value=iter([])
    ), patch.object(
        mod, "_populate_trace_identities_from_obs_temp", return_value=0
    ), patch.object(
        mod, "_dry_run", return_value=dry_payload
    ) as dry_run_mock, patch.object(
        mod, "_apply"
    ) as apply_mock, patch.object(
        mod, "_final_counts"
    ) as final_counts_mock, patch(
        "builtins.print"
    ) as print_mock:
        rc = mod.main()

    assert rc == 0
    create_tmp.assert_called_once()
    dry_run_mock.assert_called_once()
    apply_mock.assert_not_called()
    final_counts_mock.assert_not_called()
    conn.rollback.assert_called_once()
    printed = print_mock.call_args.args[0]
    assert '"mode": "dry-run"' in printed
    assert '"would_fill"' in printed


def test_main_apply_writes_without_rollback() -> None:
    """--apply path runs _apply and does not dry-run-rollback the transaction."""
    mod = _load_module()

    args = SimpleNamespace(
        apply=True,
        target_db_name="aawm_tristore",
        correct_default_environment_from_port=False,
        derive_environment_from_port=False,
        clickhouse_page_size=1000,
        clickhouse_max_pages=None,
        clickhouse_resume_after_id=None,
        clickhouse_timeout_seconds=30,
        insert_batch_size=1000,
        pg_dsn=None,
        pg_host=None,
        pg_port=None,
        pg_user=None,
        pg_password=None,
        clickhouse_url=None,
        clickhouse_user=None,
        clickhouse_password=None,
    )

    cur = MagicMock()
    cur.fetchone.side_effect = [("aawm_tristore",)]

    conn = MagicMock()
    conn.cursor.return_value.__enter__.return_value = cur
    conn.cursor.return_value.__exit__.return_value = False
    conn.__enter__.return_value = conn
    conn.__exit__.return_value = False

    auth = {
        "url": "http://127.0.0.1:8123/",
        "user": "u",
        "password": "p",
        "url_input": None,
        "url_source": "test",
        "user_source": "test",
        "password_source": "test",
    }
    dry_payload = {"matched_rows": 1}
    final_payload = {"rows": 5}

    with patch.object(mod, "_parse_args", return_value=args), patch.object(
        mod, "_resolve_clickhouse_auth_sources", return_value=auth
    ), patch.object(mod, "_preflight_clickhouse_connection"), patch.object(
        mod, "_postgres_dsn_from_args", return_value="postgresql://example/aawm_tristore"
    ), patch.object(mod.psycopg, "connect", return_value=conn), patch.object(
        mod, "_create_temp_identity_tables"
    ), patch.object(
        mod, "_iter_observation_identity_pages", return_value=iter([])
    ), patch.object(
        mod, "_populate_trace_identities_from_obs_temp", return_value=1
    ), patch.object(
        mod, "_dry_run", return_value=dry_payload
    ), patch.object(
        mod, "_apply", return_value=4
    ) as apply_mock, patch.object(
        mod, "_final_counts", return_value=final_payload
    ), patch("builtins.print") as print_mock:
        rc = mod.main()

    assert rc == 0
    apply_mock.assert_called_once()
    conn.rollback.assert_not_called()
    printed = print_mock.call_args.args[0]
    assert '"mode": "apply"' in printed
    assert '"updated_rows": 4' in printed


def test_main_streams_pages_into_obs_temp_before_sql_trace_populate() -> None:
    """Each CH page is inserted before SQL consensus runs (O(page) retention)."""
    mod = _load_module()

    args = SimpleNamespace(
        apply=False,
        target_db_name="aawm_tristore",
        correct_default_environment_from_port=False,
        derive_environment_from_port=False,
        clickhouse_page_size=2,
        clickhouse_max_pages=None,
        clickhouse_resume_after_id=None,
        clickhouse_timeout_seconds=30,
        insert_batch_size=2,
        pg_dsn=None,
        pg_host=None,
        pg_port=None,
        pg_user=None,
        pg_password=None,
        clickhouse_url=None,
        clickhouse_user=None,
        clickhouse_password=None,
    )

    page1 = [
        mod.ObservationIdentity(
            observation_id="obs-1",
            trace_id="tr-1",
            litellm_environment="dev",
            litellm_version=None,
            litellm_fork_version=None,
            litellm_wheel_versions=None,
            client_name="claude-cli",
            client_version=None,
            client_user_agent=None,
            port_environment=None,
            port_host=None,
        )
    ]
    page2 = [
        mod.ObservationIdentity(
            observation_id="obs-2",
            trace_id="tr-1",
            litellm_environment="dev",
            litellm_version=None,
            litellm_fork_version=None,
            litellm_wheel_versions=None,
            client_name="claude-cli",
            client_version=None,
            client_user_agent=None,
            port_environment=None,
            port_host=None,
        )
    ]

    cur = MagicMock()
    cur.fetchone.side_effect = [("aawm_tristore",)]
    conn = MagicMock()
    conn.cursor.return_value.__enter__.return_value = cur
    conn.cursor.return_value.__exit__.return_value = False
    conn.__enter__.return_value = conn
    conn.__exit__.return_value = False

    auth = {
        "url": "http://127.0.0.1:8123/",
        "user": "u",
        "password": "p",
        "url_input": None,
        "url_source": "test",
        "user_source": "test",
        "password_source": "test",
    }

    call_order: list[str] = []

    def track_insert(cur_arg, page, *, insert_batch_size):  # noqa: ANN001
        call_order.append(f"insert:{len(page)}:{insert_batch_size}")

    def track_populate(cur_arg):  # noqa: ANN001
        call_order.append("populate")
        return 1

    with patch.object(mod, "_parse_args", return_value=args), patch.object(
        mod, "_resolve_clickhouse_auth_sources", return_value=auth
    ), patch.object(mod, "_preflight_clickhouse_connection"), patch.object(
        mod, "_postgres_dsn_from_args", return_value="postgresql://example/aawm_tristore"
    ), patch.object(mod.psycopg, "connect", return_value=conn), patch.object(
        mod, "_create_temp_identity_tables"
    ), patch.object(
        mod,
        "_iter_observation_identity_pages",
        return_value=iter([page1, page2]),
    ), patch.object(
        mod, "_insert_observation_identity_rows", side_effect=track_insert
    ), patch.object(
        mod, "_populate_trace_identities_from_obs_temp", side_effect=track_populate
    ), patch.object(
        mod, "_dry_run", return_value={"matched_rows": 0}
    ), patch("builtins.print"):
        rc = mod.main()

    assert rc == 0
    assert call_order == ["insert:1:2", "insert:1:2", "populate"]
    conn.rollback.assert_called_once()


# ---------------------------------------------------------------------------
# SQL trace-identity consensus parity with Python _derive_trace_identities
# ---------------------------------------------------------------------------


def _sql_style_consensus_from_observations(mod, observations):  # noqa: ANN001
    """Pure-Python mirror of the SQL COUNT(DISTINCT)/HAVING consensus rules.

    Matches ``_populate_trace_identities_from_obs_temp`` semantics so tests can
    assert SQL/Python parity without a live PostgreSQL.
    """
    fields = (*mod.RUNTIME_IDENTITY_FIELDS, "port_environment")
    by_trace: dict[str, dict[str, set[str]]] = {}
    for obs in observations:
        if not obs.trace_id:
            continue
        bucket = by_trace.setdefault(obs.trace_id, {f: set() for f in fields})
        for field in fields:
            value = getattr(obs, field)
            if value is not None and value != "":
                bucket[field].add(value)

    traces = []
    for trace_id, values_by_field in by_trace.items():
        kwargs = {"trace_id": trace_id}
        for field in fields:
            values = values_by_field.get(field) or set()
            # COUNT(DISTINCT ...) FILTER (WHERE x IS NOT NULL) = 1 -> min(x)
            kwargs[field] = next(iter(values)) if len(values) == 1 else None
        # HAVING: at least one RUNTIME_IDENTITY_FIELDS consensus value present
        if any(kwargs.get(field) for field in mod.RUNTIME_IDENTITY_FIELDS):
            traces.append(mod.TraceIdentity(**kwargs))
    return traces


def _obs(
    mod,  # noqa: ANN001
    *,
    observation_id: str,
    trace_id: str | None,
    litellm_environment=None,  # noqa: ANN001
    litellm_version=None,  # noqa: ANN001
    litellm_fork_version=None,  # noqa: ANN001
    litellm_wheel_versions=None,  # noqa: ANN001
    client_name=None,  # noqa: ANN001
    client_version=None,  # noqa: ANN001
    client_user_agent=None,  # noqa: ANN001
    port_environment=None,  # noqa: ANN001
    port_host=None,  # noqa: ANN001
):
    return mod.ObservationIdentity(
        observation_id=observation_id,
        trace_id=trace_id,
        litellm_environment=litellm_environment,
        litellm_version=litellm_version,
        litellm_fork_version=litellm_fork_version,
        litellm_wheel_versions=litellm_wheel_versions,
        client_name=client_name,
        client_version=client_version,
        client_user_agent=client_user_agent,
        port_environment=port_environment,
        port_host=port_host,
    )


def test_sql_trace_identity_parity_unanimous_and_conflict() -> None:
    """SQL-style consensus must match Python helper on agree/conflict cases."""
    mod = _load_module()
    observations = [
        _obs(
            mod,
            observation_id="obs-1",
            trace_id="tr-agree",
            litellm_environment="dev",
            litellm_version="1.0",
            client_name="claude-cli",
            client_version="1",
            port_environment="dev",
        ),
        _obs(
            mod,
            observation_id="obs-2",
            trace_id="tr-agree",
            litellm_environment="dev",
            litellm_version="1.0",
            client_name="claude-cli",
            client_version="1",
            # null port_environment ignored for consensus
            port_environment=None,
        ),
        _obs(
            mod,
            observation_id="obs-3",
            trace_id="tr-conflict",
            litellm_environment="dev",
            client_name="claude-cli",
        ),
        _obs(
            mod,
            observation_id="obs-4",
            trace_id="tr-conflict",
            litellm_environment="prod",
            client_name="codex",
        ),
        # Null/empty trace ids must never produce a row.
        _obs(
            mod,
            observation_id="obs-5",
            trace_id=None,
            client_name="ghost",
        ),
        _obs(
            mod,
            observation_id="obs-6",
            trace_id="",
            client_name="ghost",
        ),
    ]

    python_rows = {
        item.trace_id: item for item in mod._derive_trace_identities(observations)
    }
    sql_rows = {
        item.trace_id: item
        for item in _sql_style_consensus_from_observations(mod, observations)
    }

    assert set(python_rows) == {"tr-agree"}
    assert set(sql_rows) == set(python_rows)

    agree_py = python_rows["tr-agree"]
    agree_sql = sql_rows["tr-agree"]
    assert agree_py.litellm_environment == agree_sql.litellm_environment == "dev"
    assert agree_py.litellm_version == agree_sql.litellm_version == "1.0"
    assert agree_py.client_name == agree_sql.client_name == "claude-cli"
    assert agree_py.client_version == agree_sql.client_version == "1"
    assert agree_py.port_environment == agree_sql.port_environment == "dev"


def test_sql_trace_identity_parity_partial_field_consensus() -> None:
    """Per-field consensus: conflict nulls one field, keeps unanimous others."""
    mod = _load_module()
    observations = [
        _obs(
            mod,
            observation_id="obs-1",
            trace_id="tr-partial",
            litellm_environment="dev",
            client_name="claude-cli",
            client_version="1",
        ),
        _obs(
            mod,
            observation_id="obs-2",
            trace_id="tr-partial",
            litellm_environment="dev",
            client_name="codex",  # conflict
            client_version="1",
        ),
        # port-only consensus with no RUNTIME fields must be dropped (HAVING).
        _obs(
            mod,
            observation_id="obs-3",
            trace_id="tr-port-only",
            port_environment="prod",
        ),
        _obs(
            mod,
            observation_id="obs-4",
            trace_id="tr-port-only",
            port_environment="prod",
        ),
    ]

    python_rows = {
        item.trace_id: item for item in mod._derive_trace_identities(observations)
    }
    sql_rows = {
        item.trace_id: item
        for item in _sql_style_consensus_from_observations(mod, observations)
    }

    assert set(python_rows) == {"tr-partial"}
    assert set(sql_rows) == set(python_rows)

    partial_py = python_rows["tr-partial"]
    partial_sql = sql_rows["tr-partial"]
    assert partial_py.litellm_environment == partial_sql.litellm_environment == "dev"
    assert partial_py.client_version == partial_sql.client_version == "1"
    assert partial_py.client_name is None
    assert partial_sql.client_name is None


def test_populate_trace_identities_sql_shape_mirrors_python_consensus_rules() -> None:
    """Production SQL string must encode the same consensus filters as Python."""
    mod = _load_module()
    cur = MagicMock()
    # First execute: INSERT...SELECT; second: SELECT count(*)
    cur.fetchone.return_value = (7,)

    count = mod._populate_trace_identities_from_obs_temp(cur)
    assert count == 7
    assert cur.execute.call_count == 2

    insert_sql = cur.execute.call_args_list[0].args[0]
    count_sql = cur.execute.call_args_list[1].args[0]

    assert "INSERT INTO tmp_session_history_runtime_identity_trace" in insert_sql
    assert "FROM tmp_session_history_runtime_identity_obs" in insert_sql
    assert "GROUP BY o.trace_id" in insert_sql
    assert "count(DISTINCT" in insert_sql
    assert "FILTER (WHERE" in insert_sql
    assert "HAVING" in insert_sql
    # Empty/null traces excluded
    assert "o.trace_id IS NOT NULL" in insert_sql
    assert "o.trace_id <> ''" in insert_sql
    # Unanimous-only: DISTINCT count must equal 1
    assert "= 1" in insert_sql
    # Every runtime identity field is projected via consensus CASE.
    for field in mod.RUNTIME_IDENTITY_FIELDS:
        assert f"o.{field}" in insert_sql
    # port_environment is derived too (used by fill path), but not required alone.
    assert "o.port_environment" in insert_sql
    assert "SELECT count(*) FROM tmp_session_history_runtime_identity_trace" in count_sql


def test_populate_trace_identities_and_python_share_field_set() -> None:
    """Guard against SQL/Python field-set drift on runtime identity columns."""
    mod = _load_module()
    # Capture SQL once
    cur = MagicMock()
    cur.fetchone.return_value = (0,)
    mod._populate_trace_identities_from_obs_temp(cur)
    insert_sql = cur.execute.call_args_list[0].args[0]

    # Python helper iterates RUNTIME_IDENTITY_FIELDS (+ port_environment).
    source = Path(mod.__file__).read_text(encoding="utf-8")
    derive_start = source.index("def _derive_trace_identities")
    derive_end = source.index("def _iter_batches", derive_start)
    derive_body = source[derive_start:derive_end]
    assert "RUNTIME_IDENTITY_FIELDS" in derive_body
    assert "port_environment" in derive_body

    # SQL must name every runtime identity field explicitly (cannot rely on the
    # Python constant expansion), and must include port_environment.
    for field in (*mod.RUNTIME_IDENTITY_FIELDS, "port_environment"):
        assert f"o.{field}" in insert_sql

    # Module-level constant is the shared field set the Python path iterates.
    assert mod.RUNTIME_IDENTITY_FIELDS == (
        "litellm_environment",
        "litellm_version",
        "litellm_fork_version",
        "litellm_wheel_versions",
        "client_name",
        "client_version",
        "client_user_agent",
    )
