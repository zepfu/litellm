import json
from dataclasses import FrozenInstanceError
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from litellm.integrations.aawm_session_history import sql as session_history_sql
from litellm.integrations.aawm_session_history import cohere_accepted_calls as accepted_calls
from litellm.integrations.aawm_session_history import (
    locally_counted_accepted_calls as ledger,
)
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
        if query == ledger._CURRENT_DATABASE_SQL:
            return self.database_name
        if query == ledger._ADVISORY_LOCK_SQL:
            return None
        if query == ledger._RANGE_COUNT_SQL:
            return self.monthly_used
        if query == ledger._ROLLING_MODEL_COUNT_SQL:
            return self.rpm_used
        raise AssertionError(f"unexpected fetchval query: {query}")

    async def fetchrow(self, query: str, *args: Any) -> Any:
        self.fetchrow_calls.append((query, args))
        self.operation_calls.append(query)
        if query != ledger._INSERT_SQL:
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


def _patch_pool(monkeypatch, conn: _Connection) -> None:
    monkeypatch.setattr(
        ledger,
        "_get_aawm_session_history_pool",
        lambda: _get_pool(conn),
    )


def test_should_define_idempotent_locally_counted_accepted_call_schema() -> None:
    ddl = session_history_sql._AAWM_LOCALLY_COUNTED_ACCEPTED_CALLS_TABLE_SQL
    assert "CREATE TABLE IF NOT EXISTS public.locally_counted_accepted_calls" in ddl
    assert "accepted_at TIMESTAMPTZ NOT NULL" in ddl
    assert "month_start" not in ddl
    assert "provider TEXT NOT NULL" in ddl
    assert "credential_scope TEXT NOT NULL" in ddl
    assert "lane TEXT" in ddl
    assert "CHECK (provider" not in ddl
    assert "CHECK (credential_scope" not in ddl
    assert "UNIQUE (provider, credential_scope, litellm_call_id)" in ddl
    assert "evidence JSONB NOT NULL DEFAULT '{}'::jsonb" in ddl
    assert "created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()" in ddl
    indexes = " ".join(
        session_history_sql._AAWM_LOCALLY_COUNTED_ACCEPTED_CALLS_INDEX_STATEMENTS
    )
    assert "CREATE INDEX IF NOT EXISTS" in indexes
    assert "(provider, credential_scope, accepted_at)" in indexes
    assert "(provider, credential_scope, model, accepted_at)" in indexes
    assert session_history_sql._AAWM_COHERE_ACCEPTED_CALLS_TABLE_SQL == ddl
    assert (
        session_history_sql._AAWM_COHERE_ACCEPTED_CALLS_INDEX_STATEMENTS
        == session_history_sql._AAWM_LOCALLY_COUNTED_ACCEPTED_CALLS_INDEX_STATEMENTS
    )
    assert compatibility_sql._AAWM_LOCALLY_COUNTED_ACCEPTED_CALLS_TABLE_SQL == ddl
    assert (
        compatibility_sql._AAWM_LOCALLY_COUNTED_ACCEPTED_CALLS_INDEX_STATEMENTS
        == session_history_sql._AAWM_LOCALLY_COUNTED_ACCEPTED_CALLS_INDEX_STATEMENTS
    )
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
    _patch_pool(monkeypatch, conn)
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
        ledger._CURRENT_DATABASE_SQL,
        ledger._ADVISORY_LOCK_SQL,
        ledger._INSERT_SQL,
        ledger._RANGE_COUNT_SQL,
        ledger._ROLLING_MODEL_COUNT_SQL,
    ]
    lock_query, lock_args = conn.fetchval_calls[1]
    assert "pg_advisory_xact_lock" in lock_query
    assert "hashtext($1::text)" in lock_query
    assert "hashtext($2::text)" in lock_query
    assert lock_args == ("cohere", "cohere_trial_default")

    query, params = conn.fetchrow_calls[0]
    assert "INSERT INTO public.locally_counted_accepted_calls" in query
    assert "ON CONFLICT (provider, credential_scope, litellm_call_id) DO NOTHING" in query
    assert "RETURNING litellm_call_id" in query
    assert params[0] == datetime(2026, 8, 13, 3, 4, 5, tzinfo=timezone.utc)
    assert params[1] == "cohere"
    assert params[2] == "cohere_trial_default"
    assert params[3] == "cohere_native"
    assert params[4] == "cohere/north-mini-code-1-0"
    assert params[5] == "call-1"
    evidence = json.loads(params[9])
    assert evidence == {"observation_source": "locally_counted"}
    assert "api_key" not in params[9]
    assert "secret" not in params[9]
    assert "hash" not in params[9]


@pytest.mark.asyncio
async def test_should_reject_wrong_database_before_write(monkeypatch) -> None:
    conn = _Connection(database_name="wrong_database")
    _patch_pool(monkeypatch, conn)

    with pytest.raises(RuntimeError, match="aawm_tristore"):
        await accepted_calls.record_cohere_accepted_call(**_call_kwargs())

    assert conn.fetchrow_calls == []
    assert conn.transaction_context.entered is True
    assert conn.transaction_context.rolled_back is True


@pytest.mark.asyncio
async def test_should_roll_back_insert_errors(monkeypatch) -> None:
    conn = _Connection()
    conn.fail_on_insert = True
    _patch_pool(monkeypatch, conn)

    with pytest.raises(RuntimeError, match="insert failed"):
        await accepted_calls.record_cohere_accepted_call(**_call_kwargs())

    assert conn.transaction_context.rolled_back is True


@pytest.mark.asyncio
async def test_should_leave_counts_unchanged_for_duplicate_insert(monkeypatch) -> None:
    conn = _Connection(inserted_row=None, monthly_used=9, rpm_used=3)
    _patch_pool(monkeypatch, conn)
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
    _patch_pool(monkeypatch, conn)
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
    assert first_lock_args == second_lock_args == ("cohere", "cohere_trial_default")
    first_monthly_query, first_monthly_args = conn.fetchval_calls[2]
    second_monthly_query, second_monthly_args = conn.fetchval_calls[6]
    assert first_monthly_query == second_monthly_query
    assert first_monthly_args == second_monthly_args == (
        "cohere",
        "cohere_trial_default",
        datetime(2026, 8, 1, tzinfo=timezone.utc),
        datetime(2026, 9, 1, tzinfo=timezone.utc),
    )
    first_rpm_query, first_rpm_args = conn.fetchval_calls[3]
    second_rpm_query, second_rpm_args = conn.fetchval_calls[7]
    assert first_rpm_query == second_rpm_query
    assert "AND model IS NOT DISTINCT FROM $5::text" in first_rpm_query
    assert first_rpm_args[4] == "cohere/north-mini-code-1-0"
    assert second_rpm_args[4] == "cohere/other-model"
    assert model_info_calls == [
        {"model": "cohere/north-mini-code-1-0", "custom_llm_provider": "cohere"},
        {"model": "cohere/other-model", "custom_llm_provider": "cohere"},
    ]


@pytest.mark.asyncio
async def test_should_preserve_unknown_rpm_metadata(monkeypatch) -> None:
    conn = _Connection(monthly_used=2, rpm_used=1)
    _patch_pool(monkeypatch, conn)
    monkeypatch.setattr(accepted_calls, "get_model_info", lambda **kwargs: {})

    state = await accepted_calls.record_cohere_accepted_call(**_call_kwargs())

    assert state.rpm_used == 1
    assert state.rpm_limit is None
    assert state.rpm_remaining is None


def test_should_guard_cohere_migration_to_exact_database_and_writer_privileges() -> None:
    scripts = Path(__file__).resolve().parents[3] / "scripts"
    wrapper = (scripts / "apply_cohere_accepted_calls_2026_08_13.sql").read_text(
        encoding="utf-8"
    )
    migration = (
        scripts / "apply_locally_counted_accepted_calls_2026_08_19.sql"
    ).read_text(encoding="utf-8")

    assert "public.locally_counted_accepted_calls" in wrapper
    assert r"\ir apply_locally_counted_accepted_calls_2026_08_19.sql" in wrapper
    assert r"\set ON_ERROR_STOP on" in migration
    assert "current_database() = 'aawm_tristore'" in migration
    assert "NULLIF(btrim(:'expected_database'), '') = 'aawm_tristore'" in migration
    assert "CREATE TABLE IF NOT EXISTS public.locally_counted_accepted_calls" in migration
    assert "UNIQUE (provider, credential_scope, litellm_call_id)" in migration
    assert "CHECK (provider" not in migration
    assert "CHECK (credential_scope" not in migration
    assert "month_start" not in migration
    assert migration.count("CREATE INDEX IF NOT EXISTS") == 2
    assert (
        'ALTER TABLE public.locally_counted_accepted_calls OWNER TO :"owner_role";'
        in migration
    )
    assert (
        "GRANT SELECT, INSERT\n"
        "    ON TABLE public.locally_counted_accepted_calls\n"
        '    TO :"runtime_role";' in migration
    )
    assert "GRANT UPDATE" not in migration
    assert "GRANT DELETE" not in migration
    assert "CREATE TABLE" not in Path(accepted_calls.__file__).read_text(encoding="utf-8")
    assert "CREATE TABLE" not in Path(ledger.__file__).read_text(encoding="utf-8")


@pytest.mark.asyncio
async def test_should_use_utc_month_and_strict_rolling_minute_boundary(monkeypatch) -> None:
    conn = _Connection(monthly_used=12, rpm_used=5)
    _patch_pool(monkeypatch, conn)
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
    assert "month_start" not in monthly_query
    assert "accepted_at >= $3::timestamptz" in monthly_query
    assert "accepted_at < $4::timestamptz" in monthly_query
    assert monthly_args == (
        "cohere",
        "cohere_trial_default",
        datetime(2026, 8, 1, tzinfo=timezone.utc),
        datetime(2026, 9, 1, tzinfo=timezone.utc),
    )
    assert "accepted_at > $3::timestamptz - ($4::integer * INTERVAL '1 second')" in rpm_query
    assert rpm_args[0] == "cohere"
    assert rpm_args[1] == "cohere_trial_default"
    assert rpm_args[2] == datetime(2026, 8, 31, 23, 59, 59, tzinfo=timezone.utc)
    assert rpm_args[3] == 60
