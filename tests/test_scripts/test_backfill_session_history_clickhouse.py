from __future__ import annotations

from datetime import datetime, timezone

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


def test_should_filter_clickhouse_session_id_with_trace_subquery() -> None:
    query = backfill.build_langfuse_clickhouse_generation_batch_query(
        limit=5,
        session_id="session-123",
    )

    assert "t.session_id" not in query
    assert "o.trace_id IN (" in query
    assert "SELECT id FROM traces" in query
    assert "session_id = 'session-123'" in query


class _CapturingClickHouseSource(backfill.LangfuseClickHouseSource):
    def __init__(self) -> None:
        super().__init__(url="http://127.0.0.1:8123", user="u", password="p")
        self.queries: list[str] = []

    def _request_rows(self, query: str):  # type: ignore[override]
        self.queries.append(query)
        return []


async def test_should_pass_include_payloads_flag_through_fetch_generation_batch() -> None:
    source = _CapturingClickHouseSource()

    await source.fetch_generation_batch(limit=3, include_payloads=False)
    await source.fetch_generation_batch(limit=3, include_payloads=True)

    assert len(source.queries) == 2
    assert "NULL AS observation_input" in source.queries[0]
    assert "o.input AS observation_input" in source.queries[1]
