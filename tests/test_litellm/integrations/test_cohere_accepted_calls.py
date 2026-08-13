import json
from dataclasses import FrozenInstanceError
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from litellm.integrations.aawm_session_history import sql as session_history_sql
from litellm.integrations.aawm_session_history import cohere_accepted_calls as accepted_calls
from litellm.integrations import aawm_session_history_sql as compatibility_sql


class _Transaction:
    def __init__(self) -> None:
        self.entered = False
        self.committed = False
        self.rolled_back = False

    async def __aenter__(self) -> "_Transaction":
        self.entered = True
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        self.rolled_back = exc_type is not None
        self.committed = exc_type is None
        return False


class _Acquire:
    def __init__(self, conn: "_Connection") -> None:
        self.conn = conn

    async def __aenter__(self) -> "_Connection":
        return self.conn

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        return False


class _Pool:
    def __init__(self, conn: "_Connection") -> None:
        self.conn = conn

    def acquire(self) -> _Acquire:
        return _Acquire(self.conn)


class _Connection:
    def __init__(
        self,
        *,
        database_name: str = "aawm_tristore",
        inserted_row: Any = {"litellm_call_id": "call-1"},
        monthly_used: int = 1,
        rpm_used: int = 1,
    ) -> None:
        self.database_name = database_name
        self.inserted_row = inserted_row
        self.monthly_used = monthly_used
        self.rpm_used = rpm_used
        self.transaction_context = _Transaction()
        self.fetchval_calls: list[tuple[str, tuple[Any, ...]]] = []
        self.fetchrow_calls: list[tuple[str, tuple[Any, ...]]] = []
        self.operation_calls: list[str] = []
        self.fail_on_insert = False

    def transaction(self) -> _Transaction:
        return self.transaction_context

    async def fetchval(self, query: str, *args: Any) -> Any:
        self.fetchval_calls.append((query, args))
        self.operation_calls.append(query)
        if query == accepted_calls._COHERE_CURRENT_DATABASE_SQL:
            return self.database_name
        if query == accepted_calls._COHERE_ACCEPTED_CALL_ADVISORY_LOCK_SQL:
            return None
        if query == accepted_calls._COHERE_ACCEPTED_CALL_MONTHLY_COUNT_SQL:
            return self.monthly_used
        if query == accepted_calls._COHERE_ACCEPTED_CALL_RPM_COUNT_SQL:
            return self.rpm_used
        raise AssertionError(f"unexpected fetchval query: {query}")

    async def fetchrow(self, query: str, *args: Any) -> Any:
        self.fetchrow_calls.append((query, args))
        self.operation_calls.append(query)
        if query != accepted_calls._COHERE_ACCEPTED_CALL_INSERT_SQL:
            raise AssertionError(f"unexpected fetchrow query: {query}")
        if self.fail_on_insert:
            raise RuntimeError("insert failed")
        return self.inserted_row


async def _get_pool(conn: _Connection) -> _Pool:
    return _Pool(conn)


def _call_kwargs(**overrides: Any) -> dict[str, Any]:
    values: dict[str, Any] = {
        "litellm_call_id": "call-1",
        "accepted_at": datetime(2026, 8, 13, 3, 4, 5, tzinfo=timezone.utc),
        "model": "cohere/north-mini-code-1-0",
        "session_id": "session-1",
        "trace_id": "trace-1",
        "source": "codex_cohere_chat_completions_adapter",
    }
    values.update(overrides)
    return values


def test_should_define_idempotent_cohere_accepted_call_schema() -> None:
    ddl = session_history_sql._AAWM_COHERE_ACCEPTED_CALLS_TABLE_SQL
    assert "CREATE TABLE IF NOT EXISTS public.cohere_accepted_calls" in ddl
    assert "accepted_at TIMESTAMPTZ NOT NULL" in ddl
    assert "month_start DATE NOT NULL" in ddl
    assert "provider TEXT NOT NULL DEFAULT 'cohere'" in ddl
    assert "credential_scope TEXT NOT NULL DEFAULT 'cohere_trial_default'" in ddl
    assert "litellm_call_id TEXT NOT NULL UNIQUE" in ddl
    assert "evidence JSONB NOT NULL DEFAULT '{}'::jsonb" in ddl
    assert "created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()" in ddl
    indexes = " ".join(session_history_sql._AAWM_COHERE_ACCEPTED_CALLS_INDEX_STATEMENTS)
    assert "CREATE INDEX IF NOT EXISTS" in indexes
    assert "(month_start, accepted_at)" in indexes
    assert "(model, accepted_at)" in indexes
    assert compatibility_sql._AAWM_COHERE_ACCEPTED_CALLS_TABLE_SQL == ddl
    assert (
        compatibility_sql._AAWM_COHERE_ACCEPTED_CALLS_INDEX_STATEMENTS
        == session_history_sql._AAWM_COHERE_ACCEPTED_CALLS_INDEX_STATEMENTS
    )


def test_should_reject_cohere_accepted_call_state_mutation() -> None:
    state = accepted_calls.CohereAcceptedCallState(
        counted=True,
        monthly_used=1,
        monthly_remaining=999,
        monthly_limit=1000,
        rpm_used=1,
        rpm_remaining=19,
        rpm_limit=20,
        month_start=datetime(2026, 8, 1, tzinfo=timezone.utc),
        month_end=datetime(2026, 9, 1, tzinfo=timezone.utc),
    )

    with pytest.raises(FrozenInstanceError):
        state.counted = False


@pytest.mark.asyncio
async def test_should_persist_redacted_payload_and_model_specific_state(monkeypatch) -> None:
    conn = _Connection(monthly_used=4, rpm_used=2)
    monkeypatch.setattr(
        accepted_calls,
        "_get_aawm_session_history_pool",
        lambda: _get_pool(conn),
    )
    monkeypatch.setattr(
        accepted_calls,
        "get_model_info",
        lambda **kwargs: {"rpm": 20},
    )

    state = await accepted_calls.record_cohere_accepted_call(
        **_call_kwargs(session_id=None, trace_id=None)
    )

    assert state.counted is True
    assert state.monthly_used == 4
    assert state.monthly_remaining == 996
    assert state.monthly_limit == 1000
    assert state.rpm_used == 2
    assert state.rpm_remaining == 18
    assert state.rpm_limit == 20
    assert state.observation_source == "locally_counted"
    assert state.month_start == datetime(2026, 8, 1, tzinfo=timezone.utc)
    assert state.month_end == datetime(2026, 9, 1, tzinfo=timezone.utc)
    assert conn.operation_calls == [
        accepted_calls._COHERE_CURRENT_DATABASE_SQL,
        accepted_calls._COHERE_ACCEPTED_CALL_ADVISORY_LOCK_SQL,
        accepted_calls._COHERE_ACCEPTED_CALL_INSERT_SQL,
        accepted_calls._COHERE_ACCEPTED_CALL_MONTHLY_COUNT_SQL,
        accepted_calls._COHERE_ACCEPTED_CALL_RPM_COUNT_SQL,
    ]
    lock_query, lock_args = conn.fetchval_calls[1]
    assert "pg_advisory_xact_lock" in lock_query
    assert "hashtext($1::text)" in lock_query
    assert "COALESCE($2::text, '<null-model>')" not in lock_query
    assert lock_args == ("cohere_trial_default",)

    query, params = conn.fetchrow_calls[0]
    assert "ON CONFLICT (litellm_call_id) DO NOTHING" in query
    assert "RETURNING litellm_call_id" in query
    assert params[0] == datetime(2026, 8, 13, 3, 4, 5, tzinfo=timezone.utc)
    assert params[1] == datetime(2026, 8, 1, tzinfo=timezone.utc).date()
    assert params[2] == "cohere/north-mini-code-1-0"
    assert params[3] == "call-1"
    evidence = json.loads(params[7])
    assert evidence == {"observation_source": "locally_counted"}
    assert "api_key" not in params[7]
    assert "secret" not in params[7]
    assert "hash" not in params[7]


@pytest.mark.asyncio
async def test_should_reject_wrong_database_before_write(monkeypatch) -> None:
    conn = _Connection(database_name="wrong_database")
    monkeypatch.setattr(
        accepted_calls,
        "_get_aawm_session_history_pool",
        lambda: _get_pool(conn),
    )

    with pytest.raises(RuntimeError, match="aawm_tristore"):
        await accepted_calls.record_cohere_accepted_call(**_call_kwargs())

    assert conn.fetchrow_calls == []
    assert conn.transaction_context.entered is True
    assert conn.transaction_context.rolled_back is True


@pytest.mark.asyncio
async def test_should_roll_back_insert_errors(monkeypatch) -> None:
    conn = _Connection()
    conn.fail_on_insert = True
    monkeypatch.setattr(
        accepted_calls,
        "_get_aawm_session_history_pool",
        lambda: _get_pool(conn),
    )

    with pytest.raises(RuntimeError, match="insert failed"):
        await accepted_calls.record_cohere_accepted_call(**_call_kwargs())

    assert conn.transaction_context.rolled_back is True


@pytest.mark.asyncio
async def test_should_leave_counts_unchanged_for_duplicate_insert(monkeypatch) -> None:
    conn = _Connection(inserted_row=None, monthly_used=9, rpm_used=3)
    monkeypatch.setattr(
        accepted_calls,
        "_get_aawm_session_history_pool",
        lambda: _get_pool(conn),
    )
    monkeypatch.setattr(accepted_calls, "get_model_info", lambda **kwargs: {"rpm": 20})

    state = await accepted_calls.record_cohere_accepted_call(**_call_kwargs())

    assert state.counted is False
    assert state.monthly_used == 9
    assert state.rpm_used == 3
    assert len(conn.fetchval_calls) == 4
    assert conn.transaction_context.committed is True


@pytest.mark.asyncio
async def test_should_share_monthly_usage_and_lock_but_isolate_rolling_counts_by_model(
    monkeypatch,
) -> None:
    conn = _Connection(monthly_used=2, rpm_used=1)
    monkeypatch.setattr(
        accepted_calls,
        "_get_aawm_session_history_pool",
        lambda: _get_pool(conn),
    )
    model_info_calls: list[dict[str, Any]] = []

    def model_info(**kwargs: Any) -> dict[str, int]:
        model_info_calls.append(kwargs)
        return {"rpm": 20 if kwargs["model"] == "cohere/north-mini-code-1-0" else 7}

    monkeypatch.setattr(accepted_calls, "get_model_info", model_info)

    first_state = await accepted_calls.record_cohere_accepted_call(
        **_call_kwargs(litellm_call_id="call-north")
    )
    second_state = await accepted_calls.record_cohere_accepted_call(
        **_call_kwargs(
            litellm_call_id="call-other",
            model="cohere/other-model",
        )
    )

    assert first_state.monthly_used == second_state.monthly_used == 2
    assert first_state.monthly_limit == second_state.monthly_limit == 1000
    assert first_state.rpm_used == second_state.rpm_used == 1
    assert first_state.rpm_limit == 20
    assert second_state.rpm_limit == 7
    first_lock_query, first_lock_args = conn.fetchval_calls[1]
    second_lock_query, second_lock_args = conn.fetchval_calls[5]
    assert first_lock_query == second_lock_query
    assert first_lock_args == second_lock_args == ("cohere_trial_default",)
    first_monthly_query, first_monthly_args = conn.fetchval_calls[2]
    second_monthly_query, second_monthly_args = conn.fetchval_calls[6]
    assert first_monthly_query == second_monthly_query
    assert first_monthly_args == second_monthly_args == (datetime(2026, 8, 1).date(),)
    first_rpm_query, first_rpm_args = conn.fetchval_calls[3]
    second_rpm_query, second_rpm_args = conn.fetchval_calls[7]
    assert first_rpm_query == second_rpm_query
    assert "AND model IS NOT DISTINCT FROM $2::text" in first_rpm_query
    assert first_rpm_args[1] == "cohere/north-mini-code-1-0"
    assert second_rpm_args[1] == "cohere/other-model"
    assert model_info_calls == [
        {"model": "cohere/north-mini-code-1-0", "custom_llm_provider": "cohere"},
        {"model": "cohere/other-model", "custom_llm_provider": "cohere"}
    ]


@pytest.mark.asyncio
async def test_should_preserve_unknown_rpm_metadata(monkeypatch) -> None:
    conn = _Connection(monthly_used=2, rpm_used=1)
    monkeypatch.setattr(
        accepted_calls,
        "_get_aawm_session_history_pool",
        lambda: _get_pool(conn),
    )
    monkeypatch.setattr(accepted_calls, "get_model_info", lambda **kwargs: {})

    state = await accepted_calls.record_cohere_accepted_call(**_call_kwargs())

    assert state.rpm_used == 1
    assert state.rpm_limit is None
    assert state.rpm_remaining is None


def test_should_guard_cohere_migration_to_exact_database_and_writer_privileges() -> None:
    migration = (
        Path(__file__).resolve().parents[3]
        / "scripts"
        / "apply_cohere_accepted_calls_2026_08_13.sql"
    ).read_text(encoding="utf-8")

    assert r"\set ON_ERROR_STOP on" in migration
    assert "current_database() = 'aawm_tristore'" in migration
    assert "NULLIF(btrim(:'expected_database'), '') = 'aawm_tristore'" in migration
    assert "CREATE TABLE IF NOT EXISTS public.cohere_accepted_calls" in migration
    assert migration.count("CREATE INDEX IF NOT EXISTS") == 2
    assert 'ALTER TABLE public.cohere_accepted_calls OWNER TO :"owner_role";' in migration
    assert (
        "GRANT SELECT, INSERT\n"
        "    ON TABLE public.cohere_accepted_calls\n"
        '    TO :"runtime_role";' in migration
    )
    assert "GRANT UPDATE" not in migration
    assert "GRANT DELETE" not in migration
    assert "CREATE TABLE" not in accepted_calls.__file__


@pytest.mark.asyncio
async def test_should_use_utc_month_and_strict_rolling_minute_boundary(monkeypatch) -> None:
    conn = _Connection(monthly_used=12, rpm_used=5)
    monkeypatch.setattr(
        accepted_calls,
        "_get_aawm_session_history_pool",
        lambda: _get_pool(conn),
    )
    monkeypatch.setattr(accepted_calls, "get_model_info", lambda **kwargs: {"rpm": 20})

    state = await accepted_calls.record_cohere_accepted_call(
        **_call_kwargs(
            accepted_at=datetime(2026, 8, 31, 23, 59, 59),
        )
    )

    assert state.month_start == datetime(2026, 8, 1, tzinfo=timezone.utc)
    assert state.month_end == datetime(2026, 9, 1, tzinfo=timezone.utc)
    monthly_query, monthly_args = conn.fetchval_calls[2]
    rpm_query, rpm_args = conn.fetchval_calls[3]
    assert "month_start = $1::date" in monthly_query
    assert monthly_args == (datetime(2026, 8, 1, tzinfo=timezone.utc).date(),)
    assert "accepted_at > $1::timestamptz - INTERVAL '60 seconds'" in rpm_query
    assert rpm_args[0] == datetime(2026, 8, 31, 23, 59, 59, tzinfo=timezone.utc)
