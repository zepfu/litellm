from __future__ import annotations

import argparse
from datetime import datetime, timezone
from unittest.mock import patch

from scripts import backfill_session_history as backfill


def test_should_exclude_clickhouse_input_output_columns_by_default() -> None:
    query = backfill.build_langfuse_clickhouse_generation_batch_query(limit=25)

    assert "NULL AS observation_input" in query
    assert "NULL AS observation_output" in query
    assert "o.input AS observation_input" not in query
    assert "o.output AS observation_output" not in query
    assert "ORDER BY o.start_time DESC, o.id DESC" in query


def test_should_include_clickhouse_input_output_columns_when_enabled() -> None:
    query = backfill.build_langfuse_clickhouse_generation_batch_query(
        limit=10,
        include_payloads=True,
    )

    assert "o.input AS observation_input" in query
    assert "o.output AS observation_output" in query
    assert "NULL AS observation_input" not in query
    assert "NULL AS observation_output" not in query


def test_should_preserve_clickhouse_cursor_predicate_and_ordering() -> None:
    cursor_time = datetime(2026, 7, 1, 12, 0, 0, tzinfo=timezone.utc)
    query = backfill.build_langfuse_clickhouse_generation_batch_query(
        limit=5,
        cursor_start_time=cursor_time,
        cursor_id="obs-cursor-1",
    )

    assert "o.start_time < toDateTime64" in query
    assert "AND o.id <" in query
    assert "LIMIT 5" in query
    assert "{cursor_id:String}" in query
    assert "obs-cursor-1" not in str(query)
    assert query.params["cursor_id"] == "obs-cursor-1"


def test_should_filter_clickhouse_session_id_with_trace_subquery() -> None:
    query = backfill.build_langfuse_clickhouse_generation_batch_query(
        limit=5,
        session_id="session-123",
    )

    assert "t.session_id" not in query
    assert "o.trace_id IN (" in query
    assert "SELECT id FROM traces" in query
    assert "session_id = {session_id:String}" in query
    assert query.params["session_id"] == "session-123"
    assert "session-123" not in str(query)  # bound via params, not SQL text


class _CapturingClickHouseSource(backfill.LangfuseClickHouseSource):
    def __init__(self) -> None:
        super().__init__(url="http://127.0.0.1:8123", user="u", password="p")
        self.queries: list[str] = []
        self.params_seen: list = []

    def _request_rows(self, query: str, params=None):  # type: ignore[override]
        self.queries.append(query)
        self.params_seen.append(params)
        return []


async def test_should_pass_include_payloads_flag_through_fetch_generation_batch() -> None:
    source = _CapturingClickHouseSource()

    # Avoid asyncio.to_thread under restrictive sandboxes; exercise the same
    # query construction path fetch_generation_batch uses.
    async def immediate_to_thread(fn, *args, **kwargs):
        return fn(*args, **kwargs)

    with patch.object(backfill.asyncio, "to_thread", side_effect=immediate_to_thread):
        await source.fetch_generation_batch(limit=3, include_payloads=False)
        await source.fetch_generation_batch(limit=3, include_payloads=True)

    assert len(source.queries) == 2
    assert "NULL AS observation_input" in source.queries[0]
    assert "o.input AS observation_input" in source.queries[1]
    assert source.params_seen[0] is not None or source.params_seen[0] == {}


def test_should_report_builtin_clickhouse_auth_sources_when_unconfigured() -> None:
    args = argparse.Namespace(clickhouse_url=None, clickhouse_user=None, clickhouse_password=None)
    with patch.object(backfill, "get_secret_str", return_value=None):
        auth = backfill._resolve_clickhouse_auth_sources(args)

    assert auth["url"] == "http://127.0.0.1:8123"
    assert auth["user"] == "clickhouse"
    assert auth["url_source"] == "default:local_http_8123"
    assert auth["user_source"] == "default:clickhouse_builtin"
    assert auth["password_source"] == "default:clickhouse_builtin"

    diagnostics = backfill._clickhouse_auth_diagnostics(auth)
    assert diagnostics["clickhouse_url_normalized"] == "http://127.0.0.1:8123"
    assert diagnostics["using_builtin_local_url_default"] is True
    assert diagnostics["using_builtin_clickhouse_credentials"] is True
    assert "password" not in diagnostics


def test_should_report_env_clickhouse_auth_sources() -> None:
    args = argparse.Namespace(clickhouse_url=None, clickhouse_user=None, clickhouse_password=None)

    def fake_secret(name: str):
        return {
            "LANGFUSE_CLICKHOUSE_URL": "http://clickhouse:8123",
            "LANGFUSE_CLICKHOUSE_USER": "lf_user",
            "LANGFUSE_CLICKHOUSE_PASSWORD": "lf_secret",
        }.get(name)

    with patch.object(backfill, "get_secret_str", side_effect=fake_secret):
        auth = backfill._resolve_clickhouse_auth_sources(args)

    assert auth["url"] == "http://127.0.0.1:8123"
    assert auth["user"] == "lf_user"
    assert auth["url_source"] == "env:LANGFUSE_CLICKHOUSE_URL"
    assert auth["user_source"] == "env:LANGFUSE_CLICKHOUSE_USER"
    assert auth["password_source"] == "env:LANGFUSE_CLICKHOUSE_PASSWORD"

    diagnostics = backfill._clickhouse_auth_diagnostics(auth)
    assert diagnostics["clickhouse_url_normalized"] == "http://127.0.0.1:8123"
    assert diagnostics["clickhouse_url_raw"] == "http://clickhouse:8123"
    assert "lf_secret" not in str(diagnostics)


def test_should_redact_userinfo_in_clickhouse_auth_diagnostics() -> None:
    args = argparse.Namespace(
        clickhouse_url="http://chuser:supersecret@clickhouse:8123/path",
        clickhouse_user=None,
        clickhouse_password=None,
    )
    with patch.object(backfill, "get_secret_str", return_value=None):
        auth = backfill._resolve_clickhouse_auth_sources(args)

    diagnostics = backfill._clickhouse_auth_diagnostics(auth)
    assert diagnostics["clickhouse_url_normalized"] == "http://127.0.0.1:8123/path"
    assert diagnostics["clickhouse_url_raw"] == "http://chuser:***@clickhouse:8123/path"
    assert "supersecret" not in str(diagnostics)
    assert "password" not in diagnostics


def test_should_preflight_clickhouse_before_query() -> None:
    args = argparse.Namespace(clickhouse_url=None, clickhouse_user=None, clickhouse_password=None)
    with patch.object(backfill, "get_secret_str", return_value=None):
        auth = backfill._resolve_clickhouse_auth_sources(args)
    with patch.object(backfill.LangfuseClickHouseSource, "_request_rows") as mock_rows:
        backfill._preflight_clickhouse_connection(auth)
        mock_rows.assert_called_once()
        assert "SELECT 1" in mock_rows.call_args[0][0]


def test_should_skip_preflight_for_non_clickhouse_source_modes() -> None:
    assert backfill._should_preflight_clickhouse_for_source_mode("langfuse_clickhouse") is True
    assert backfill._should_preflight_clickhouse_for_source_mode("langfuse") is False
    assert backfill._should_preflight_clickhouse_for_source_mode("spendlogs") is False
    assert backfill._should_preflight_clickhouse_for_source_mode("langfuse_db") is False
