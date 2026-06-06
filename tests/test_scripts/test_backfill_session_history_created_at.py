from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock

import pytest

from scripts import backfill_session_history as backfill


class _FakePoolAcquire:
    def __init__(self, conn):
        self.conn = conn

    async def __aenter__(self):
        return self.conn

    async def __aexit__(self, exc_type, exc, tb):
        return None


class _FakePool:
    def __init__(self, conn):
        self.conn = conn

    def acquire(self):
        return _FakePoolAcquire(self.conn)


def test_should_build_created_at_updates_from_event_time() -> None:
    updates = backfill._session_history_created_at_updates(
        [
            {
                "litellm_call_id": "call-1",
                "start_time": datetime(2026, 6, 5, 19, 12, 24),
            },
            {
                "litellm_call_id": "call-1",
                "start_time": datetime(2026, 6, 5, 19, 13, tzinfo=timezone.utc),
            },
            {
                "litellm_call_id": "call-2",
                "end_time": datetime(2026, 6, 5, 19, 14, tzinfo=timezone.utc),
            },
            {"litellm_call_id": "call-no-time"},
        ]
    )

    assert updates == [
        (datetime(2026, 6, 5, 19, 12, 24, tzinfo=timezone.utc), "call-1"),
        (datetime(2026, 6, 5, 19, 14, tzinfo=timezone.utc), "call-2"),
    ]


def test_should_record_trace_tag_patch_result_counts() -> None:
    stats = backfill.BackfillStats()

    backfill._record_trace_tag_patch_result(
        stats,
        {"status": "patched", "added": 2},
    )
    backfill._record_trace_tag_patch_result(
        stats,
        {"status": "unchanged", "added": 0},
    )
    backfill._record_trace_tag_patch_result(
        stats,
        {"status": "missing_trace", "added": 0},
    )

    assert stats.traces_patched == 1
    assert stats.trace_tags_added == 2
    assert stats.traces_unchanged == 1
    assert stats.traces_missing == 1


@pytest.mark.asyncio
async def test_should_align_session_history_created_at_to_event_time(monkeypatch) -> None:
    mock_conn = AsyncMock()
    fake_pool = _FakePool(mock_conn)
    monkeypatch.setattr(
        backfill,
        "_get_session_history_pool",
        AsyncMock(return_value=fake_pool),
    )
    monkeypatch.setattr(backfill, "_ensure_session_history_schema_with_pool", AsyncMock())

    aligned = await backfill._align_session_history_created_at_to_event_time(
        [
            {
                "litellm_call_id": "call-1",
                "start_time": datetime(2026, 6, 5, 19, 12, 24, tzinfo=timezone.utc),
            }
        ]
    )

    assert aligned == 1
    mock_conn.executemany.assert_awaited_once()
    sql, params = mock_conn.executemany.await_args.args
    assert "UPDATE public.session_history" in sql
    assert "SET created_at = $1" in sql
    assert params == [
        (datetime(2026, 6, 5, 19, 12, 24, tzinfo=timezone.utc), "call-1")
    ]
