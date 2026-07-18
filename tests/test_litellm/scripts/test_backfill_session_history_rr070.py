"""RR-070: session_history backfill resume cursors, keyset repair, ClickHouse params."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import scripts.backfill_session_history as backfill


def test_parse_resume_cursor_round_trip() -> None:
    raw = json.dumps(
        {
            "skip": 150,
            "page": 3,
            "cursor_id": "obs-9",
            "cursor_start_time": "2026-07-01T12:00:00Z",
        }
    )
    parsed = backfill._parse_resume_cursor(raw)
    assert parsed["skip"] == 150
    assert parsed["page"] == 3
    assert parsed["cursor_id"] == "obs-9"
    assert isinstance(parsed["cursor_start_time"], datetime)

    serialized = backfill._serialize_resume_cursor(parsed)
    assert serialized is not None
    assert serialized["skip"] == 150
    assert serialized["cursor_id"] == "obs-9"
    assert "2026-07-01" in serialized["cursor_start_time"]


def test_parse_resume_cursor_rejects_invalid_json() -> None:
    with pytest.raises(ValueError, match="Invalid --resume-cursor"):
        backfill._parse_resume_cursor("{not-json")


def test_parser_accepts_resume_cursor_flag() -> None:
    parser = backfill._build_arg_parser()
    args = parser.parse_args(
        ["--resume-cursor", '{"skip": 10}', "--source-mode", "spendlogs"]
    )
    assert args.resume_cursor == '{"skip": 10}'
    assert backfill._resume_cursor_from_args(args)["skip"] == 10


def test_clickhouse_query_uses_http_params_not_string_interpolation() -> None:
    query = backfill.build_langfuse_clickhouse_generation_batch_query(
        limit=5,
        session_id="session-123",
        trace_id="trace-abc",
        model="gpt-4o",
        cursor_start_time=datetime(2026, 7, 1, 12, 0, 0, tzinfo=timezone.utc),
        cursor_id="obs-cursor-1",
    )
    params = query.params
    assert isinstance(query, str)
    assert isinstance(params, dict)
    assert "{session_id:String}" in query
    assert "{trace_id:String}" in query
    assert "{model:String}" in query
    assert "{cursor_id:String}" in query
    assert "session_id = 'session-123'" not in query
    assert "obs-cursor-1" not in query  # bound, not interpolated
    assert params["session_id"] == "session-123"
    assert params["trace_id"] == "trace-abc"
    assert params["model"] == "gpt-4o"
    assert params["cursor_id"] == "obs-cursor-1"
    assert "cursor_start_time" in params
    assert "LIMIT 5" in query
    assert "ORDER BY o.start_time DESC, o.id DESC" in query


def test_clickhouse_payload_flags_still_work() -> None:
    q_off = backfill.build_langfuse_clickhouse_generation_batch_query(
        limit=2, include_payloads=False
    )
    q_on = backfill.build_langfuse_clickhouse_generation_batch_query(
        limit=2, include_payloads=True
    )
    assert "NULL AS observation_input" in q_off
    assert "o.input AS observation_input" in q_on


def test_request_rows_sends_param_query_string() -> None:
    source = backfill.LangfuseClickHouseSource(
        url="http://127.0.0.1:8123", user="u", password="p"
    )
    captured: Dict[str, Any] = {}

    class _Resp:
        def read(self) -> bytes:
            return b""

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    def fake_urlopen(request, timeout=60):  # noqa: ARG001
        captured["url"] = request.full_url
        captured["data"] = request.data
        return _Resp()

    with patch.object(backfill, "urlopen", side_effect=fake_urlopen):
        source._request_rows("SELECT 1", {"session_id": "sess-1"})

    assert "param_session_id=sess-1" in captured["url"]
    assert captured["data"] == b"SELECT 1"


def test_fetch_trace_retries_with_backoff_sleep() -> None:
    backfiller = object.__new__(backfill.LangfuseTraceTagBackfiller)
    backfiller.host = "http://127.0.0.1:3000"
    backfiller._auth_header = "Basic abc"

    sleeps: List[float] = []

    def boom(*_a, **_k):
        raise backfill.URLError("temporary")

    with patch.object(backfill, "urlopen", side_effect=boom), patch.object(
        backfill.time, "sleep", side_effect=lambda s: sleeps.append(s)
    ):
        with pytest.raises(backfill.URLError):
            backfiller._fetch_trace("trace-1")

    assert sleeps == [0.5, 1.0]


@pytest.mark.asyncio
async def test_fetch_trace_page_forwards_session_id() -> None:
    source = backfill.LangfuseTraceSource(
        host="http://127.0.0.1:3000", public_key="pk", secret_key="sk"
    )
    captured: Dict[str, Any] = {}

    def fake_request(path, params):
        captured["path"] = path
        captured["params"] = params
        return {"data": [], "meta": {}}

    with patch.object(source, "_request_json", side_effect=fake_request):
        await source.fetch_trace_page(page=1, limit=10, fields="core", session_id="s1")
    assert captured["params"]["sessionId"] == "s1"


def _repair_row(call_id: str, start: datetime | None) -> Dict[str, Any]:
    return {
        "litellm_call_id": call_id,
        "tenant_id": None,
        "provider": "anthropic",
        "model": "claude-sonnet-4-6",
        "inbound_model_alias": "claude-sonnet-4-6",
        "input_tokens": 10,
        "output_tokens": 5,
        "cache_read_input_tokens": 0,
        "cache_creation_input_tokens": 0,
        "response_cost_usd": 0.01,
        "repository": None,
        "metadata": {
            "custom_llm_provider": "anthropic",
            "passthrough_route_family": "anthropic_messages",
        },
        "start_time": start,
    }


@pytest.mark.asyncio
async def test_session_history_repair_uses_keyset_not_offset() -> None:
    queries: List[str] = []
    args_seen: List[tuple] = []
    page1 = [
        _repair_row("call-1", datetime(2026, 1, 1, tzinfo=timezone.utc)),
    ]
    page2 = [
        _repair_row("call-2", datetime(2026, 1, 2, tzinfo=timezone.utc)),
    ]

    async def fake_fetch(query: str, *params: Any) -> List[Dict[str, Any]]:
        queries.append(query)
        args_seen.append(params)
        if len(queries) == 1:
            return page1
        if len(queries) == 2:
            return page2
        return []

    mock_connection = MagicMock()
    mock_connection.fetch = AsyncMock(side_effect=fake_fetch)
    mock_connection.execute = AsyncMock()
    mock_connection.executemany = AsyncMock()

    mock_pool = MagicMock()
    mock_acquire = MagicMock()
    mock_acquire.__aenter__ = AsyncMock(return_value=mock_connection)
    mock_acquire.__aexit__ = AsyncMock(return_value=None)
    mock_pool.acquire.return_value = mock_acquire

    args = argparse.Namespace(
        repair_gemini_control_plane=None,
        repair_costs=False,
        repair_tenant_ids=False,
        repair_anthropic_context_window=True,
        request_id=None,
        trace_id=None,
        session_id=None,
        provider=None,
        model=None,
        from_start_time=None,
        to_start_time=None,
        limit=None,
        batch_size=1,
        apply=False,
    )

    with patch.object(
        backfill, "_get_session_history_pool", AsyncMock(return_value=mock_pool)
    ), patch.object(backfill, "_ensure_session_history_schema_with_pool", AsyncMock()):
        result = await backfill._run_session_history_repair(args)

    assert result["stats"]["scanned_rows"] == 2
    assert result["stats"]["anthropic_context_window_updates"] == 2
    assert all("OFFSET" not in q for q in queries)
    assert len(queries) >= 2
    assert "ORDER BY start_time ASC NULLS LAST, litellm_call_id ASC" in queries[0]
    assert "start_time >" in queries[1]
    assert "litellm_call_id >" in queries[1]
    assert "start_time IS NULL" in queries[1]
    # second page continues after first row's (start_time, call_id)
    assert args_seen[1][0] == page1[0]["start_time"]
    assert args_seen[1][1] == "call-1"
    mock_connection.execute.assert_not_called()


@pytest.mark.asyncio
async def test_session_history_repair_keyset_nulls_last_partition() -> None:
    queries: List[str] = []
    args_seen: List[tuple] = []

    async def fake_fetch(query: str, *params: Any) -> List[Dict[str, Any]]:
        queries.append(query)
        args_seen.append(params)
        if len(queries) == 1:
            return [_repair_row("null-1", None)]
        if len(queries) == 2:
            return [_repair_row("null-2", None)]
        return []

    mock_connection = MagicMock()
    mock_connection.fetch = AsyncMock(side_effect=fake_fetch)
    mock_connection.executemany = AsyncMock()

    mock_pool = MagicMock()
    mock_acquire = MagicMock()
    mock_acquire.__aenter__ = AsyncMock(return_value=mock_connection)
    mock_acquire.__aexit__ = AsyncMock(return_value=None)
    mock_pool.acquire.return_value = mock_acquire

    args = argparse.Namespace(
        repair_gemini_control_plane=None,
        repair_costs=False,
        repair_tenant_ids=False,
        repair_anthropic_context_window=True,
        request_id=None,
        trace_id=None,
        session_id=None,
        provider=None,
        model=None,
        from_start_time=None,
        to_start_time=None,
        limit=None,
        batch_size=1,
        apply=False,
    )

    with patch.object(
        backfill, "_get_session_history_pool", AsyncMock(return_value=mock_pool)
    ), patch.object(backfill, "_ensure_session_history_schema_with_pool", AsyncMock()):
        result = await backfill._run_session_history_repair(args)

    assert result["stats"]["scanned_rows"] == 2
    assert "start_time IS NULL AND litellm_call_id >" in queries[1]
    assert args_seen[1][0] == "null-1"


@pytest.mark.asyncio
async def test_session_history_repair_batches_updates_with_executemany() -> None:
    row = _repair_row("call-1", datetime(2026, 1, 1, tzinfo=timezone.utc))

    async def fake_fetch(query: str, *params: Any) -> List[Dict[str, Any]]:
        if "litellm_call_id >" in query or "start_time >" in query:
            return []
        return [row]

    mock_connection = MagicMock()
    mock_connection.fetch = AsyncMock(side_effect=fake_fetch)
    mock_connection.executemany = AsyncMock()

    mock_pool = MagicMock()
    mock_acquire = MagicMock()
    mock_acquire.__aenter__ = AsyncMock(return_value=mock_connection)
    mock_acquire.__aexit__ = AsyncMock(return_value=None)
    mock_pool.acquire.return_value = mock_acquire

    args = argparse.Namespace(
        repair_gemini_control_plane=None,
        repair_costs=False,
        repair_tenant_ids=False,
        repair_anthropic_context_window=True,
        request_id=None,
        trace_id=None,
        session_id=None,
        provider=None,
        model=None,
        from_start_time=None,
        to_start_time=None,
        limit=None,
        batch_size=50,
        apply=True,
    )

    with patch.object(
        backfill, "_get_session_history_pool", AsyncMock(return_value=mock_pool)
    ), patch.object(backfill, "_ensure_session_history_schema_with_pool", AsyncMock()):
        result = await backfill._run_session_history_repair(args)

    assert result["stats"]["rows_with_updates"] == 1
    mock_connection.executemany.assert_called_once()
    # One acquire for the page (fetch + executemany), not per-row.
    assert mock_pool.acquire.call_count == 1


@pytest.mark.asyncio
async def test_gemini_control_plane_repair_reuses_page_connection() -> None:
    row = {
        "id": 10,
        "litellm_call_id": "g-1",
        "provider": "gemini",
        "model": "models/gemini-2.5-flash",
        "model_group": None,
        "call_type": "generateContent",
        "start_time": datetime(2026, 1, 1, tzinfo=timezone.utc),
        "metadata": {"passthrough_route_family": "gemini"},
    }

    mock_connection = MagicMock()
    mock_connection.fetch = AsyncMock(side_effect=[[row], []])
    mock_connection.execute = AsyncMock(return_value="UPDATE 1")
    mock_connection.executemany = AsyncMock()
    mock_connection.transaction = MagicMock()
    mock_tx = MagicMock()
    mock_tx.__aenter__ = AsyncMock(return_value=None)
    mock_tx.__aexit__ = AsyncMock(return_value=None)
    mock_connection.transaction.return_value = mock_tx

    mock_pool = MagicMock()
    mock_acquire = MagicMock()
    mock_acquire.__aenter__ = AsyncMock(return_value=mock_connection)
    mock_acquire.__aexit__ = AsyncMock(return_value=None)
    mock_pool.acquire.return_value = mock_acquire

    args = argparse.Namespace(
        repair_gemini_control_plane="mark",
        request_id=None,
        trace_id=None,
        session_id=None,
        provider=None,
        model=None,
        from_start_time=None,
        to_start_time=None,
        limit=None,
        batch_size=50,
        apply=True,
    )

    with patch.object(
        backfill, "_get_session_history_pool", AsyncMock(return_value=mock_pool)
    ), patch.object(
        backfill, "_ensure_session_history_schema_with_pool", AsyncMock()
    ), patch.object(
        backfill,
        "_is_gemini_control_plane_session_history_row",
        return_value=(True, "generateContent"),
    ):
        result = await backfill._run_gemini_control_plane_session_history_repair(args)

    assert result["stats"]["matched_rows"] == 1
    assert result["stats"]["updated_rows"] == 1
    # Page select + page writes share one acquire (no per-row reacquire).
    assert mock_pool.acquire.call_count == 1
    mock_connection.executemany.assert_called_once()
    mock_connection.execute.assert_not_called()


@pytest.mark.asyncio
async def test_get_existing_call_ids_used_in_dry_run_path_helper() -> None:
    """Issue #5: dry-run must query existing ids when target DSN is available."""
    with patch.object(
        backfill, "_build_aawm_admin_dsn", return_value="postgres://x"
    ), patch.object(
        backfill, "_get_session_history_pool", AsyncMock()
    ) as mock_pool_fn, patch.object(
        backfill, "_ensure_session_history_schema_with_pool", AsyncMock()
    ) as mock_schema:
        mock_conn = MagicMock()
        mock_conn.fetch = AsyncMock(return_value=[{"litellm_call_id": "already-there"}])
        mock_pool = MagicMock()
        mock_acquire = MagicMock()
        mock_acquire.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_acquire.__aexit__ = AsyncMock(return_value=None)
        mock_pool.acquire.return_value = mock_acquire
        mock_pool_fn.return_value = mock_pool

        existing = await backfill._get_existing_call_ids(["already-there", "new-one"])
        assert existing == {"already-there"}
        mock_schema.assert_not_called()


@pytest.mark.asyncio
async def test_iter_spend_logs_respects_skip_start_and_yields_skip() -> None:
    class _DB:
        def __init__(self) -> None:
            self.calls: List[Dict[str, Any]] = []

        async def find_many(self, **kwargs):
            self.calls.append(kwargs)
            if kwargs["skip"] >= 4:
                return []
            return [{"id": f"r{kwargs['skip'] + i}"} for i in range(kwargs["take"])]

    class _Prisma:
        def __init__(self) -> None:
            self.db = argparse.Namespace(litellm_spendlogs=_DB())

    prisma = _Prisma()
    with patch.object(backfill, "_coerce_row_to_dict", side_effect=lambda r: r):
        batches = []
        async for batch, skip in backfill._iter_spend_logs(
            prisma,  # type: ignore[arg-type]
            where={},
            batch_size=2,
            limit=3,
            skip_start=2,
        ):
            batches.append((batch, skip))

    assert batches[0][1] == 4  # started at 2, took 2
    assert prisma.db.litellm_spendlogs.calls[0]["skip"] == 2


def test_attach_resume_cursor_omits_empty() -> None:
    result = backfill._attach_resume_cursor({"ok": True}, {})
    assert "resume_cursor" not in result
    result2 = backfill._attach_resume_cursor({"ok": True}, {"skip": 5})
    assert result2["resume_cursor"] == {"skip": 5}


@pytest.mark.asyncio
async def test_gemini_control_plane_repair_batches_deletes() -> None:
    row = {
        "id": 11,
        "litellm_call_id": "g-del",
        "provider": "gemini",
        "model": "models/gemini-2.5-flash",
        "model_group": None,
        "call_type": "generateContent",
        "start_time": datetime(2026, 1, 1, tzinfo=timezone.utc),
        "metadata": {"passthrough_route_family": "gemini"},
    }

    mock_connection = MagicMock()
    mock_connection.fetch = AsyncMock(side_effect=[[row], []])
    mock_connection.execute = AsyncMock(return_value="DELETE 1")
    mock_connection.executemany = AsyncMock()
    mock_connection.transaction = MagicMock()
    mock_tx = MagicMock()
    mock_tx.__aenter__ = AsyncMock(return_value=None)
    mock_tx.__aexit__ = AsyncMock(return_value=None)
    mock_connection.transaction.return_value = mock_tx

    mock_pool = MagicMock()
    mock_acquire = MagicMock()
    mock_acquire.__aenter__ = AsyncMock(return_value=mock_connection)
    mock_acquire.__aexit__ = AsyncMock(return_value=None)
    mock_pool.acquire.return_value = mock_acquire

    args = argparse.Namespace(
        repair_gemini_control_plane="delete",
        request_id=None,
        trace_id=None,
        session_id=None,
        provider=None,
        model=None,
        from_start_time=None,
        to_start_time=None,
        limit=None,
        batch_size=50,
        apply=True,
    )

    with patch.object(
        backfill, "_get_session_history_pool", AsyncMock(return_value=mock_pool)
    ), patch.object(
        backfill, "_ensure_session_history_schema_with_pool", AsyncMock()
    ), patch.object(
        backfill,
        "_is_gemini_control_plane_session_history_row",
        return_value=(True, "generateContent"),
    ):
        result = await backfill._run_gemini_control_plane_session_history_repair(args)

    assert result["stats"]["matched_rows"] == 1
    assert result["stats"]["deleted_rows"] == 1
    assert mock_pool.acquire.call_count == 1
    # Two batched executes: tool_activity then session_history
    assert mock_connection.execute.await_count == 2
    first_sql = mock_connection.execute.await_args_list[0].args[0]
    second_sql = mock_connection.execute.await_args_list[1].args[0]
    assert "session_history_tool_activity" in first_sql
    assert "ANY($1::text[])" in first_sql
    assert "session_history WHERE id = ANY" in second_sql


@pytest.mark.asyncio
async def test_langfuse_trace_backfill_skips_unbounded_prefetch() -> None:
    """Issue #2: full history must not preload all traces into memory."""
    source = MagicMock()
    source.fetch_trace_page = AsyncMock(
        return_value={
            "data": [{"id": "t1", "sessionId": "s1"}],
            "meta": {"totalPages": 1},
        }
    )
    source.fetch_observation_page = AsyncMock(
        return_value={"data": [], "meta": {"totalPages": 0}}
    )
    source.fetch_trace_by_id = AsyncMock()

    args = argparse.Namespace(
        apply=False,
        session_id=None,
        trace_id=None,
        batch_size=10,
        limit=None,
        request_id=None,
        provider=None,
        model=None,
        status=None,
        from_start_time=None,
        to_start_time=None,
        patch_langfuse_tags=False,
        resume_cursor=None,
    )

    with patch.object(backfill, "_build_aawm_admin_dsn", return_value=None):
        result = await backfill._run_langfuse_trace_backfill(
            args,
            langfuse_source=source,
            langfuse_backfiller=None,
            run_id="run-1",
        )

    source.fetch_trace_page.assert_not_awaited()
    source.fetch_observation_page.assert_awaited()
    assert result["stats"]["scanned_rows"] == 0
    assert "resume_cursor" in result


@pytest.mark.asyncio
async def test_clickhouse_backfill_emits_resume_cursor_on_limit() -> None:
    """Issue #1: limit early-exit must still emit a resume_cursor."""
    row = {
        "observation_id": "obs-limit-1",
        "observation_start_time": "2026-07-01 12:00:00.000",
        "observation_trace_id": "trace-1",
        "trace_id": "trace-1",
        "session_id": "sess-1",
        "observation_name": "generation",
        "provided_model_name": "gpt-4o",
        "observation_input": None,
        "observation_output": None,
        "usage_details": {},
        "metadata": {},
        "trace_metadata": {},
        "trace_user_id": None,
        "trace_name": "t",
        "trace_tags": [],
    }

    class _Source:
        async def fetch_generation_batch(self, **kwargs):  # noqa: ANN003
            return [row]

    args = argparse.Namespace(
        apply=False,
        batch_size=10,
        limit=1,
        trace_id=None,
        session_id=None,
        from_start_time=None,
        to_start_time=None,
        provider=None,
        model=None,
        status=None,
        request_id=None,
        patch_langfuse_tags=False,
        resume_cursor=None,
        clickhouse_include_payloads=False,
    )

    fake_record = {
        "litellm_call_id": "call-limit-1",
        "metadata": {
            "session_id_source": "trace.sessionId",
            "trace_id_source": "trace.id",
        },
    }

    with patch.object(
        backfill, "_build_aawm_admin_dsn", return_value=None
    ), patch.object(
        backfill, "LangfuseClickHouseSource", return_value=_Source()
    ), patch.object(
        backfill,
        "_build_session_history_record_from_langfuse_trace_observation",
        return_value=fake_record,
    ), patch.object(
        backfill, "_record_matches_filters", return_value=True
    ), patch.object(
        backfill, "_get_existing_call_ids", AsyncMock(return_value=set())
    ):
        result = await backfill._run_langfuse_clickhouse_backfill(
            args,
            clickhouse_config={
                "url": "http://127.0.0.1:8123",
                "user": "u",
                "password": "p",
            },
            langfuse_backfiller=None,
            run_id="run-limit",
        )

    assert result["stats"]["scanned_rows"] == 1
    assert result["resume_cursor"]["cursor_id"] == "obs-limit-1"
    assert "2026-07-01" in str(result["resume_cursor"]["cursor_start_time"])

def test_parse_resume_cursor_rejects_incomplete_keyset() -> None:
    with pytest.raises(ValueError, match="cursor_start_time and cursor_id"):
        backfill._parse_resume_cursor('{"cursor_id": "obs-only"}')
    with pytest.raises(ValueError, match="cursor_start_time and cursor_id"):
        backfill._parse_resume_cursor(
            '{"cursor_start_time": "2026-07-01T12:00:00Z"}'
        )


def test_keyset_resume_cursor_from_observation_row_normalizes_clickhouse_time() -> None:
    cursor = backfill._keyset_resume_cursor_from_observation_row(
        {
            "observation_id": "obs-9",
            "observation_start_time": "2026-07-01 12:00:00.000",
        }
    )
    assert cursor is not None
    assert cursor["cursor_id"] == "obs-9"
    assert isinstance(cursor["cursor_start_time"], datetime)
    assert cursor["cursor_start_time"].year == 2026


def _multi_row_observation_batch() -> List[Dict[str, Any]]:
    # DESC order as produced by langfuse_db / clickhouse fetch helpers.
    return [
        {
            "observation_id": "obs-newer",
            "observation_start_time": "2026-07-02 12:00:00.000",
            "observation_trace_id": "trace-newer",
            "trace_id": "trace-newer",
            "session_id": "sess-1",
            "observation_name": "generation",
            "provided_model_name": "gpt-4o",
            "observation_input": None,
            "observation_output": None,
            "usage_details": {},
            "metadata": {},
            "trace_metadata": {},
            "trace_user_id": None,
            "trace_name": "t",
            "trace_tags": [],
        },
        {
            "observation_id": "obs-older",
            "observation_start_time": "2026-07-01 12:00:00.000",
            "observation_trace_id": "trace-older",
            "trace_id": "trace-older",
            "session_id": "sess-1",
            "observation_name": "generation",
            "provided_model_name": "gpt-4o",
            "observation_input": None,
            "observation_output": None,
            "usage_details": {},
            "metadata": {},
            "trace_metadata": {},
            "trace_user_id": None,
            "trace_name": "t",
            "trace_tags": [],
        },
    ]


@pytest.mark.asyncio
async def test_clickhouse_mid_batch_limit_resume_uses_last_processed_row() -> None:
    """Multi-row batch + --limit=1 must not advance past unprocessed rows."""
    batch = _multi_row_observation_batch()
    captured_cursors: List[Dict[str, Any]] = []

    class _Source:
        async def fetch_generation_batch(self, **kwargs):  # noqa: ANN003
            captured_cursors.append(
                {
                    "cursor_id": kwargs.get("cursor_id"),
                    "cursor_start_time": kwargs.get("cursor_start_time"),
                }
            )
            # First page returns multi-row batch; resume should not need a second page.
            if kwargs.get("cursor_id") is None:
                return list(batch)
            return []

    args = argparse.Namespace(
        apply=False,
        batch_size=10,
        limit=1,
        trace_id=None,
        session_id=None,
        from_start_time=None,
        to_start_time=None,
        provider=None,
        model=None,
        status=None,
        request_id=None,
        patch_langfuse_tags=False,
        resume_cursor=None,
        clickhouse_include_payloads=False,
    )

    def fake_record(trace, observation, backfill_run_id=None):  # noqa: ANN001
        obs_id = observation.get("id") or observation.get("observation_id")
        return {
            "litellm_call_id": f"call-{obs_id}",
            "metadata": {
                "session_id_source": "trace.sessionId",
                "trace_id_source": "trace.id",
            },
        }

    with patch.object(
        backfill, "_build_aawm_admin_dsn", return_value=None
    ), patch.object(
        backfill, "LangfuseClickHouseSource", return_value=_Source()
    ), patch.object(
        backfill,
        "_build_session_history_record_from_langfuse_trace_observation",
        side_effect=fake_record,
    ), patch.object(
        backfill, "_record_matches_filters", return_value=True
    ), patch.object(
        backfill, "_get_existing_call_ids", AsyncMock(return_value=set())
    ), patch.object(
        backfill,
        "_build_langfuse_trace_from_clickhouse_row",
        side_effect=lambda row: {"id": row["observation_trace_id"]},
    ), patch.object(
        backfill,
        "_build_langfuse_observation_from_clickhouse_row",
        side_effect=lambda row: {"id": row["observation_id"], "metadata": {}},
    ):
        result = await backfill._run_langfuse_clickhouse_backfill(
            args,
            clickhouse_config={
                "url": "http://127.0.0.1:8123",
                "user": "u",
                "password": "p",
            },
            langfuse_backfiller=None,
            run_id="run-mid-batch",
        )

    assert result["stats"]["scanned_rows"] == 1
    assert result["stats"]["reconstructable_rows"] == 1
    # Cursor must be the first/newer processed row, not the unprocessed older tail.
    assert result["resume_cursor"]["cursor_id"] == "obs-newer"
    assert "2026-07-02" in str(result["resume_cursor"]["cursor_start_time"])
    assert result["resume_cursor"]["cursor_id"] != "obs-older"


@pytest.mark.asyncio
async def test_langfuse_db_mid_batch_limit_resume_uses_last_processed_row() -> None:
    """langfuse_db multi-row batch + --limit=1 uses last processed observation."""
    batch = _multi_row_observation_batch()

    class _Source:
        def __init__(self) -> None:
            self.closed = False

        async def connect(self) -> None:
            return None

        async def close(self) -> None:
            self.closed = True

        async def fetch_generation_batch(self, **kwargs):  # noqa: ANN003
            if kwargs.get("cursor_id") is None:
                return list(batch)
            return []

    source = _Source()
    args = argparse.Namespace(
        apply=False,
        batch_size=10,
        limit=1,
        trace_id=None,
        session_id=None,
        from_start_time=None,
        to_start_time=None,
        provider=None,
        model=None,
        status=None,
        request_id=None,
        patch_langfuse_tags=False,
        resume_cursor=None,
    )

    def fake_record(trace, observation, backfill_run_id=None):  # noqa: ANN001
        obs_id = observation.get("id") or observation.get("observation_id")
        return {
            "litellm_call_id": f"call-{obs_id}",
            "metadata": {
                "session_id_source": "trace.sessionId",
                "trace_id_source": "trace.id",
            },
        }

    with patch.object(
        backfill, "_build_aawm_admin_dsn", return_value=None
    ), patch.object(
        backfill, "LangfuseDatabaseSource", return_value=source
    ), patch.object(
        backfill,
        "_build_session_history_record_from_langfuse_trace_observation",
        side_effect=fake_record,
    ), patch.object(
        backfill, "_record_matches_filters", return_value=True
    ), patch.object(
        backfill, "_get_existing_call_ids", AsyncMock(return_value=set())
    ), patch.object(
        backfill,
        "_build_langfuse_trace_from_db_row",
        side_effect=lambda row: {"id": row["observation_trace_id"]},
    ), patch.object(
        backfill,
        "_build_langfuse_observation_from_db_row",
        side_effect=lambda row: {"id": row["observation_id"], "metadata": {}},
    ):
        result = await backfill._run_langfuse_db_backfill(
            args,
            langfuse_database_url="postgres://langfuse",
            run_id="run-db-mid-batch",
        )

    assert result["stats"]["scanned_rows"] == 1
    assert result["stats"]["reconstructable_rows"] == 1
    assert result["resume_cursor"]["cursor_id"] == "obs-newer"
    assert "2026-07-02" in str(result["resume_cursor"]["cursor_start_time"])
    assert source.closed is True
