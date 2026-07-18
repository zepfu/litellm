"""Focused RR-067 residual tests for scripts/backfill_local_cli_session_history.py."""

from __future__ import annotations

import json
from datetime import date, datetime, timezone
from pathlib import Path

import pytest

import scripts.backfill_local_cli_session_history as backfill


def test_rr067_existing_days_query_scopes_client_and_source_import() -> None:
    sql, params = backfill._existing_days_query(
        tz_name="America/New_York",
        client_names=["claude-code", "codex_tui"],
        source_import=backfill.IMPORT_MARKER,
    )
    assert "client_name IN" in sql
    assert "metadata->>'source_import'" in sql
    assert params[0] == "America/New_York"
    assert "claude-code" in params
    assert "codex_tui" in params
    assert params[-1] == backfill.IMPORT_MARKER

    empty_sql, empty_params = backfill._existing_days_query(
        tz_name="UTC",
        client_names=[],
    )
    assert "WHERE FALSE" in empty_sql
    assert empty_params == ("UTC",)


def test_rr067_safe_int_matches_shared_helper_no_float_truncation() -> None:
    from litellm.integrations.aawm_agent_identity import _safe_int as shared_safe_int

    assert backfill._safe_int("3") == 3
    assert backfill._safe_int(3.0) == 3  # int(3.0) works directly
    # Shared helper does not accept non-integral strings via float truncation.
    assert shared_safe_int("3.9") is None
    assert backfill._safe_int("3.9") is None
    assert backfill._safe_int(None) is None
    assert backfill._safe_float("1.5") == 1.5


def test_rr067_redact_text_preserves_api_key_and_token_prefixes() -> None:
    assert (
        backfill._redact_text('curl -H "api_key=sk-abc123"')
        == 'curl -H "api_key=<redacted>"'
    )
    assert backfill._redact_text("export TOKEN=ghp_abc123") == "export TOKEN=<redacted>"
    assert (
        backfill._redact_text("Authorization: Bearer secret-token-value")
        == "Authorization: Bearer <redacted>"
    )
    # Secret value itself must never remain.
    assert "sk-abc123" not in backfill._redact_text("api_key=sk-abc123")
    assert "None<redacted>" not in backfill._redact_text("api_key=sk-abc123")


def test_rr067_source_path_metadata_omits_absolute_home_path(tmp_path: Path) -> None:
    path = tmp_path / ".claude" / "projects" / "session.jsonl"
    path.parent.mkdir(parents=True)
    path.write_text("{}\n", encoding="utf-8")
    meta = backfill._source_path_metadata(path)
    assert set(meta) == {"source_path_hash", "source_path_basename"}
    assert meta["source_path_basename"] == "session.jsonl"
    assert "source_path" not in meta
    assert str(tmp_path) not in json.dumps(meta)


def test_rr067_claude_records_do_not_persist_absolute_source_path(
    tmp_path: Path,
) -> None:
    transcript = (
        tmp_path
        / ".claude"
        / "projects"
        / "-home-zepfu-projects-litellm"
        / "session.jsonl"
    )
    transcript.parent.mkdir(parents=True)
    transcript.write_text(
        json.dumps(
            {
                "type": "assistant",
                "timestamp": "2026-03-01T12:00:00.000Z",
                "sessionId": "claude-session",
                "requestId": "req-1",
                "uuid": "uuid-1",
                "cwd": "/home/zepfu/projects/litellm",
                "version": "2.1.119",
                "message": {
                    "id": "msg-1",
                    "model": "claude-opus-4-6",
                    "usage": {
                        "input_tokens": 10,
                        "output_tokens": 4,
                    },
                    "content": [{"type": "text", "text": "hi"}],
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    stats = backfill.ScanStats()
    records = list(backfill._iter_claude_records(tmp_path, stats))
    assert records
    metadata = records[0]["metadata"]
    assert "source_path" not in metadata
    assert "source_path_hash" in metadata
    assert metadata["source_path_basename"] == "session.jsonl"
    assert str(tmp_path) not in json.dumps(metadata)


def test_rr067_grok_records_mark_approximate_timestamps(tmp_path: Path) -> None:
    session_dir = (
        tmp_path
        / ".grok"
        / "sessions"
        / "%2Fhome%2Fzepfu%2Fprojects%2Flitellm"
        / "sess-1"
    )
    session_dir.mkdir(parents=True)
    (session_dir / "summary.json").write_text(
        json.dumps(
            {
                "created_at": "2026-03-01T12:00:00.000Z",
                "current_model_id": "grok-build",
                "info": {"id": "sess-1", "cwd": "/home/zepfu/projects/litellm"},
            }
        ),
        encoding="utf-8",
    )
    (session_dir / "chat_history.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"type": "user", "content": "hi"}),
                json.dumps(
                    {
                        "type": "assistant",
                        "content": "hello",
                        "tool_calls": [],
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    stats = backfill.ScanStats()
    records = list(backfill._iter_grok_records(tmp_path, stats))
    assert len(records) == 1
    record = records[0]
    assert record["metadata"]["timestamp_precision"] == "approximate_session_day"
    assert record["metadata"]["timestamp_source"] == "session_summary_plus_line_offset"
    assert "approximate" in record["metadata"]["timestamp_note"].lower()
    assert "source_path" not in record["metadata"]
    # Synthetic ms offset still orders within the session day of the summary.
    assert record["created_at"] >= datetime(2026, 3, 1, 12, 0, tzinfo=timezone.utc)


def test_rr067_dry_run_summary_surfaces_approximate_timestamps(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    """Operator dry-run must call out approximate Grok day-bucketing."""

    def fake_iter(root, clients, stats):
        record = _minimal_record(
            client_name="grok_build",
            litellm_call_id="g1",
            session_id="gs",
            created_at=datetime(2026, 3, 1, 12, 0, tzinfo=timezone.utc),
            provider="xai",
        )
        record["metadata"] = {
            "source_import": backfill.IMPORT_MARKER,
            "timestamp_precision": "approximate_session_day",
            "timestamp_source": "session_summary_plus_line_offset",
            "timestamp_note": "approximate for day-bucketing only",
        }
        yield record

    monkeypatch.setattr(backfill, "_iter_records", fake_iter)
    monkeypatch.setattr(backfill, "_fetch_existing_days", lambda *a, **k: {})
    rc = backfill.main(
        [
            "--home",
            str(tmp_path),
            "--clients",
            "grok",
            "--day-timezone",
            "UTC",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "approximate_timestamp_records: 1" in out
    assert "day buckets for those rows are approximate" in out
    assert '"timestamp_precision": "approximate_session_day"' in out


def test_rr067_session_history_sql_uses_merge_not_do_nothing() -> None:
    sql = backfill.SESSION_HISTORY_INSERT_SQL
    assert "ON CONFLICT (litellm_call_id) DO UPDATE SET" in sql
    # Active conflict action must be DO UPDATE (comment text may mention DO NOTHING).
    assert "ON CONFLICT (litellm_call_id) DO NOTHING" not in sql
    # JSONB || alone would preserve historical keys only on the left side; rescans
    # must strip deprecated absolute source_path after the merge.
    assert "|| COALESCE(EXCLUDED.metadata, '{}'::jsonb)" in sql
    assert ") - 'source_path'" in sql or "- 'source_path'" in sql
    assert "source_path" in sql


def test_rr067_session_history_metadata_merge_strips_source_path_key() -> None:
    """Document/assert the SQL expression that removes source_path on rescan."""
    sql = backfill.SESSION_HISTORY_INSERT_SQL
    # Collapsed form of the merge expression for regression locking.
    collapsed = " ".join(sql.split())
    assert (
        "metadata = ( COALESCE(session_history.metadata, '{}'::jsonb) "
        "|| COALESCE(EXCLUDED.metadata, '{}'::jsonb) ) - 'source_path'"
    ) in collapsed


def test_rr067_tool_activity_conflict_is_scoped_to_call_and_index() -> None:
    """Distinct tool events must not be suppressed by an over-broad conflict key."""
    sql = backfill.TOOL_ACTIVITY_INSERT_SQL
    assert "ON CONFLICT (litellm_call_id, tool_index) DO NOTHING" in sql
    # Must not conflict on litellm_call_id alone (would drop tools 1..N).
    assert "ON CONFLICT (litellm_call_id) DO NOTHING" not in sql
    assert "ON CONFLICT (tool_call_id)" not in sql
    assert "ON CONFLICT (tool_name)" not in sql


def test_rr067_tool_payloads_preserve_distinct_tool_indexes() -> None:
    created = datetime(2026, 3, 1, 12, 0, tzinfo=timezone.utc)
    record = _minimal_record(
        client_name="claude-code",
        litellm_call_id="call-tools",
        session_id="s-tools",
        created_at=created,
        provider="anthropic",
    )
    record["tool_activity"] = [
        {
            "tool_index": 0,
            "tool_call_id": "t0",
            "tool_name": "Read",
            "tool_kind": "read",
            "file_paths_read": ["a.py"],
            "file_paths_modified": [],
            "git_commit_count": 0,
            "git_push_count": 0,
            "command_text": None,
            "arguments": {},
            "metadata": {"source": "unit"},
        },
        {
            "tool_index": 1,
            "tool_call_id": "t1",
            "tool_name": "Edit",
            "tool_kind": "modify",
            "file_paths_read": [],
            "file_paths_modified": ["a.py"],
            "git_commit_count": 0,
            "git_push_count": 0,
            "command_text": None,
            "arguments": {},
            "metadata": {"source": "unit"},
        },
    ]
    payloads = backfill._tool_payloads(record)
    assert len(payloads) == 2
    # tool_index is column index 8 in the payload tuple (after created_at..agent_id)
    indexes = {payload[8] for payload in payloads}
    assert indexes == {0, 1}
    call_ids = {payload[1] for payload in payloads}
    assert call_ids == {"call-tools"}


def test_rr067_postgres_dsn_require_explicit_fails_closed(monkeypatch) -> None:
    monkeypatch.delenv("AAWM_DIRECT_DATABASE_URL", raising=False)
    monkeypatch.delenv("AAWM_DB_HOST", raising=False)
    monkeypatch.delenv("AAWM_DB_PORT", raising=False)
    monkeypatch.delenv("AAWM_DB_USER", raising=False)
    monkeypatch.delenv("AAWM_DB_PASSWORD", raising=False)
    args = backfill.argparse.Namespace(
        pg_dsn=None,
        pg_host=None,
        pg_port=None,
        pg_user=None,
        pg_password=None,
        target_db_name="aawm_tristore",
    )
    with pytest.raises(RuntimeError, match="explicit database configuration"):
        backfill._postgres_dsn_from_args(args, require_explicit=True)

    monkeypatch.setenv(
        "AAWM_DIRECT_DATABASE_URL",
        "postgresql://aawm:secret@127.0.0.1:5434/aawm_tristore",
    )
    assert backfill._postgres_dsn_from_args(args, require_explicit=True).startswith(
        "postgresql://aawm:secret@"
    )


def test_rr067_postgres_dsn_partial_components_fail_closed(monkeypatch) -> None:
    """Partial --pg-* must not fill missing fields from hardcoded defaults."""
    monkeypatch.delenv("AAWM_DIRECT_DATABASE_URL", raising=False)
    monkeypatch.delenv("AAWM_DB_HOST", raising=False)
    monkeypatch.delenv("AAWM_DB_PORT", raising=False)
    monkeypatch.delenv("AAWM_DB_USER", raising=False)
    monkeypatch.delenv("AAWM_DB_PASSWORD", raising=False)
    # Only host provided via CLI; remaining fields missing.
    args = backfill.argparse.Namespace(
        pg_dsn=None,
        pg_host="127.0.0.1",
        pg_port=None,
        pg_user=None,
        pg_password=None,
        target_db_name="aawm_tristore",
    )
    with pytest.raises(RuntimeError, match="explicit database configuration"):
        backfill._postgres_dsn_from_args(args, require_explicit=True)

    # Even if a direct DSN exists, component override path must be complete.
    monkeypatch.setenv(
        "AAWM_DIRECT_DATABASE_URL",
        "postgresql://aawm:secret@127.0.0.1:5434/aawm_tristore",
    )
    with pytest.raises(RuntimeError, match="Missing"):
        backfill._postgres_dsn_from_args(args, require_explicit=True)


def test_rr067_postgres_dsn_prefers_direct_when_no_component_override(
    monkeypatch,
) -> None:
    args = backfill.argparse.Namespace(
        pg_dsn=None,
        pg_host=None,
        pg_port=None,
        pg_user=None,
        pg_password=None,
        target_db_name="aawm_tristore",
    )
    monkeypatch.setenv(
        "AAWM_DIRECT_DATABASE_URL",
        "postgresql://aawm:aawm_dev@127.0.0.1:5434/aawm_tristore",
    )
    monkeypatch.setenv("AAWM_DB_HOST", "127.0.0.1")
    monkeypatch.setenv("AAWM_DB_PORT", "6432")
    assert backfill._postgres_dsn_from_args(args, require_explicit=True) == (
        "postgresql://aawm:aawm_dev@127.0.0.1:5434/aawm_tristore"
    )


def _minimal_record(
    *,
    client_name: str,
    litellm_call_id: str,
    session_id: str,
    created_at: datetime,
    provider: str = "openai",
) -> dict:
    return {
        "client_name": client_name,
        "created_at": created_at,
        "input_tokens": 1,
        "output_tokens": 1,
        "cache_read_input_tokens": 0,
        "cache_creation_input_tokens": 0,
        "reasoning_tokens_reported": None,
        "reasoning_tokens_estimated": None,
        "response_cost_usd": 0.0,
        "provider_cache_miss_token_count": 0,
        "provider_cache_miss_cost_usd": 0.0,
        "file_read_count": 0,
        "file_modified_count": 0,
        "git_commit_count": 0,
        "git_push_count": 0,
        "tool_activity": [],
        "model": "m",
        "provider": provider,
        "metadata": {"source_import": backfill.IMPORT_MARKER},
        "litellm_call_id": litellm_call_id,
        "session_id": session_id,
        "tool_names": [],
        "tool_call_count": 0,
        "invalid_tool_call_count": 0,
        "start_time": created_at,
        "end_time": created_at,
        "total_tokens": 2,
    }


def test_rr067_day_skip_is_per_client(monkeypatch, tmp_path: Path, capsys) -> None:
    """A day present for claude-code must not skip codex_tui records."""
    existing = {"claude-code": {date(2026, 3, 1)}}
    monkeypatch.setattr(
        backfill,
        "_fetch_existing_days",
        lambda *a, **k: existing,
    )

    def fake_iter(root, clients, stats):
        yield _minimal_record(
            client_name="claude-code",
            litellm_call_id="a",
            session_id="s",
            created_at=datetime(2026, 3, 1, 15, 0, tzinfo=timezone.utc),
            provider="anthropic",
        )
        yield _minimal_record(
            client_name="codex_tui",
            litellm_call_id="b",
            session_id="s2",
            created_at=datetime(2026, 3, 1, 16, 0, tzinfo=timezone.utc),
            provider="openai",
        )

    monkeypatch.setattr(backfill, "_iter_records", fake_iter)
    rc = backfill.main(
        [
            "--home",
            str(tmp_path),
            "--clients",
            "claude,codex",
            "--day-timezone",
            "UTC",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "records_that_would_be_added: 1" in out
    assert "Records skipped because day already exists" in out
    assert "claude-code: 1" in out
    assert "codex_tui: records=1" in out


def test_rr067_rescan_existing_days_reprocesses_skipped_client(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    """--rescan-existing-days must ignore prior source_import day skip."""
    monkeypatch.setattr(
        backfill,
        "_fetch_existing_days",
        lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("fetch should not run under --rescan-existing-days")
        ),
    )

    def fake_iter(root, clients, stats):
        yield _minimal_record(
            client_name="claude-code",
            litellm_call_id="a",
            session_id="s",
            created_at=datetime(2026, 3, 1, 15, 0, tzinfo=timezone.utc),
            provider="anthropic",
        )

    monkeypatch.setattr(backfill, "_iter_records", fake_iter)
    rc = backfill.main(
        [
            "--home",
            str(tmp_path),
            "--clients",
            "claude",
            "--day-timezone",
            "UTC",
            "--rescan-existing-days",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "rescan_existing_days: True" in out
    assert "records_that_would_be_added: 1" in out
    assert "existing_days_check_skipped: --rescan-existing-days" in out


def test_rr067_assert_target_database_rejects_mismatch() -> None:
    class _Cur:
        def __init__(self, name: str):
            self._name = name

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def execute(self, sql):
            assert "current_database" in sql

        def fetchone(self):
            return (self._name,)

    class _Conn:
        def __init__(self, name: str):
            self._name = name

        def cursor(self):
            return _Cur(self._name)

    with pytest.raises(RuntimeError, match="aawm_dev"):
        backfill._assert_target_database(_Conn("aawm_dev"), "aawm_tristore")
    assert (
        backfill._assert_target_database(_Conn("aawm_tristore"), "aawm_tristore")
        == "aawm_tristore"
    )


def test_rr067_apply_batch_commits_at_client_day_boundaries(
    monkeypatch, tmp_path: Path
) -> None:
    """Within-day batch-size and client/day boundary commits both fire."""
    commits: list[str] = []
    applied_batches: list[list[str]] = []

    class _Conn:
        autocommit = False

        def commit(self):
            commits.append("commit")

        def close(self):
            commits.append("close")

    def fake_connect(dsn, connect_timeout=5):
        assert "aawm_tristore" in dsn or dsn  # explicit config path used
        return _Conn()

    def fake_apply(conn, records):
        applied_batches.append([r["litellm_call_id"] for r in records])

    monkeypatch.setattr(
        backfill, "psycopg", type("P", (), {"connect": staticmethod(fake_connect)})
    )
    monkeypatch.setattr(backfill, "_assert_target_database", lambda conn, name: name)
    monkeypatch.setattr(backfill, "_fetch_existing_days", lambda *a, **k: {})
    monkeypatch.setattr(backfill, "_apply_batch", fake_apply)
    monkeypatch.setenv(
        "AAWM_DIRECT_DATABASE_URL",
        "postgresql://aawm:secret@127.0.0.1:5434/aawm_tristore",
    )

    def fake_iter(root, clients, stats):
        # day1 client A: 2 rows, batch-size 1 => two within-day commits
        yield _minimal_record(
            client_name="claude-code",
            litellm_call_id="d1a",
            session_id="s1",
            created_at=datetime(2026, 3, 1, 10, 0, tzinfo=timezone.utc),
            provider="anthropic",
        )
        yield _minimal_record(
            client_name="claude-code",
            litellm_call_id="d1b",
            session_id="s1",
            created_at=datetime(2026, 3, 1, 11, 0, tzinfo=timezone.utc),
            provider="anthropic",
        )
        # day2 client A: day boundary flush before append
        yield _minimal_record(
            client_name="claude-code",
            litellm_call_id="d2a",
            session_id="s2",
            created_at=datetime(2026, 3, 2, 10, 0, tzinfo=timezone.utc),
            provider="anthropic",
        )
        # different client same day2: client boundary flush
        yield _minimal_record(
            client_name="codex_tui",
            litellm_call_id="d2c",
            session_id="s3",
            created_at=datetime(2026, 3, 2, 12, 0, tzinfo=timezone.utc),
            provider="openai",
        )

    monkeypatch.setattr(backfill, "_iter_records", fake_iter)
    rc = backfill.main(
        [
            "--home",
            str(tmp_path),
            "--clients",
            "claude,codex",
            "--day-timezone",
            "UTC",
            "--apply",
            "--batch-size",
            "1",
            "--target-db-name",
            "aawm_tristore",
        ]
    )
    assert rc == 0
    assert applied_batches == [["d1a"], ["d1b"], ["d2a"], ["d2c"]]
    assert commits.count("commit") == 4
    assert commits[-1] == "close"


def test_rr067_batch_size_must_be_positive(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="batch-size"):
        backfill.main(
            [
                "--home",
                str(tmp_path),
                "--clients",
                "claude",
                "--batch-size",
                "0",
            ]
        )


def test_rr067_safe_str_matches_shared_helper() -> None:
    from litellm.integrations.aawm_agent_identity import _safe_str as shared_safe_str

    assert backfill._safe_str("  x  ") == "x"
    assert backfill._safe_str("") is None
    assert backfill._safe_str(None) is None
    assert backfill._safe_str("  x  ") == shared_safe_str("  x  ")


def test_rr067_dry_run_skips_existing_days_on_connection_error(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    """Dry-run remains usable when day-existence DB probe fails."""

    def boom(*a, **k):
        raise OSError("connection is bad: could not create socket")

    monkeypatch.setattr(backfill, "_fetch_existing_days", boom)
    monkeypatch.setattr(backfill, "_iter_records", lambda *a, **k: iter(()))
    rc = backfill.main(
        [
            "--home",
            str(tmp_path),
            "--clients",
            "claude",
            "--day-timezone",
            "UTC",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "existing_days_check_skipped:" in out
    assert "connection is bad" in out
