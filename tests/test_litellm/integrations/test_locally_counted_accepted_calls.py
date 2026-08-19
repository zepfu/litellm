from dataclasses import FrozenInstanceError
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

import pytest

from litellm.integrations.aawm_session_history import (
    locally_counted_accepted_calls as ledger,
)


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
    def __init__(self, conn: "_LedgerConnection") -> None:
        self.conn = conn

    async def __aenter__(self) -> "_LedgerConnection":
        return self.conn

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        return False


class _Pool:
    def __init__(self, conn: "_LedgerConnection") -> None:
        self.conn = conn

    def acquire(self) -> _Acquire:
        return _Acquire(self.conn)


class _LedgerConnection:
    def __init__(self, *, database_name: str = "aawm_tristore") -> None:
        self.database_name = database_name
        self.rows: list[dict[str, Any]] = []
        self.transaction_context = _Transaction()
        self.fetchval_calls: list[tuple[str, tuple[Any, ...]]] = []
        self.fetchrow_calls: list[tuple[str, tuple[Any, ...]]] = []
        self.lock_args: list[tuple[Any, ...]] = []

    def transaction(self) -> _Transaction:
        return self.transaction_context

    def _count_range(
        self,
        *,
        provider: str,
        credential_scope: str,
        start: datetime,
        end: datetime,
        model: Optional[str] = None,
        model_scoped: bool = False,
    ) -> int:
        used = 0
        for row in self.rows:
            if row["provider"] != provider or row["credential_scope"] != credential_scope:
                continue
            if not (start <= row["accepted_at"] < end):
                continue
            if model_scoped and row["model"] != model:
                continue
            used += 1
        return used

    def _count_rolling(
        self,
        *,
        provider: str,
        credential_scope: str,
        accepted_at: datetime,
        window_seconds: int,
        model: Optional[str] = None,
        model_scoped: bool = False,
    ) -> int:
        start_exclusive = accepted_at - timedelta(seconds=window_seconds)
        used = 0
        for row in self.rows:
            if row["provider"] != provider or row["credential_scope"] != credential_scope:
                continue
            if not (start_exclusive < row["accepted_at"] <= accepted_at):
                continue
            if model_scoped and row["model"] != model:
                continue
            used += 1
        return used

    async def fetchval(self, query: str, *args: Any) -> Any:
        self.fetchval_calls.append((query, args))
        if query == ledger._CURRENT_DATABASE_SQL:
            return self.database_name
        if query == ledger._ADVISORY_LOCK_SQL:
            self.lock_args.append(args)
            return None
        if query == ledger._RANGE_COUNT_SQL:
            return self._count_range(
                provider=args[0],
                credential_scope=args[1],
                start=args[2],
                end=args[3],
            )
        if query == ledger._RANGE_MODEL_COUNT_SQL:
            return self._count_range(
                provider=args[0],
                credential_scope=args[1],
                start=args[2],
                end=args[3],
                model=args[4],
                model_scoped=True,
            )
        if query == ledger._ROLLING_COUNT_SQL:
            return self._count_rolling(
                provider=args[0],
                credential_scope=args[1],
                accepted_at=args[2],
                window_seconds=args[3],
            )
        if query == ledger._ROLLING_MODEL_COUNT_SQL:
            return self._count_rolling(
                provider=args[0],
                credential_scope=args[1],
                accepted_at=args[2],
                window_seconds=args[3],
                model=args[4],
                model_scoped=True,
            )
        raise AssertionError(f"unexpected fetchval query: {query}")

    async def fetchrow(self, query: str, *args: Any) -> Any:
        self.fetchrow_calls.append((query, args))
        if query != ledger._INSERT_SQL:
            raise AssertionError(f"unexpected fetchrow query: {query}")
        row = {
            "accepted_at": args[0],
            "provider": args[1],
            "credential_scope": args[2],
            "lane": args[3],
            "model": args[4],
            "litellm_call_id": args[5],
            "session_id": args[6],
            "trace_id": args[7],
            "source": args[8],
            "evidence": args[9],
        }
        key = (row["provider"], row["credential_scope"], row["litellm_call_id"])
        for existing in self.rows:
            existing_key = (
                existing["provider"],
                existing["credential_scope"],
                existing["litellm_call_id"],
            )
            if existing_key == key:
                return None
        self.rows.append(row)
        return {"litellm_call_id": row["litellm_call_id"]}


async def _get_pool(conn: _LedgerConnection) -> _Pool:
    return _Pool(conn)


def _patch_pool(monkeypatch, conn: _LedgerConnection) -> None:
    monkeypatch.setattr(
        ledger,
        "_get_aawm_session_history_pool",
        lambda: _get_pool(conn),
    )


async def _record(
    conn: _LedgerConnection,
    monkeypatch,
    **overrides: Any,
) -> ledger.LocallyCountedAcceptedCallState:
    _patch_pool(monkeypatch, conn)
    values: dict[str, Any] = {
        "litellm_call_id": "call-1",
        "accepted_at": datetime(2026, 8, 19, 12, 0, tzinfo=timezone.utc),
        "provider": "openrouter",
        "credential_scope": ledger.OPENROUTER_FREE_DAILY_CREDENTIAL_SCOPE,
        "source": "test",
        "windows": ledger.openrouter_free_daily_windows(limit=1000),
    }
    values.update(overrides)
    return await ledger.record_locally_counted_accepted_call(**values)


def test_should_reject_state_and_window_mutation() -> None:
    window = ledger.CountWindow(name="daily", period="daily", limit=1000)
    state = ledger.LocallyCountedAcceptedCallState(
        counted=True,
        windows={"daily": ledger.WindowCount("daily", "daily", 1, 999, 1000)},
    )
    with pytest.raises(FrozenInstanceError):
        window.limit = 1  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        state.counted = False  # type: ignore[misc]


def test_should_expose_named_window_helpers_without_inventing_numeric_limits() -> None:
    openrouter_windows = ledger.openrouter_free_daily_windows()
    assert openrouter_windows == (
        ledger.CountWindow(
            name="daily",
            period="daily",
            limit=ledger.openrouter_free_daily_request_limit(),
            model_scoped=False,
        ),
    )
    assert ledger.OPENROUTER_PROVIDER == "openrouter"
    assert (
        ledger.OPENROUTER_FREE_DAILY_CREDENTIAL_SCOPE == "openrouter_free_daily_shared"
    )
    assert ledger.opencode_zen_windows() == (
        ledger.CountWindow(name="usage", period="daily", limit=None, model_scoped=False),
    )
    assert ledger.nvidia_nim_windows() == (
        ledger.CountWindow(name="usage", period="daily", limit=None, model_scoped=False),
    )
    assert ledger.OPENCODE_ZEN_PROVIDER == "opencode_zen"
    assert ledger.NVIDIA_NIM_PROVIDER == "nvidia_nim"


def test_should_read_openrouter_daily_limit_from_env(monkeypatch) -> None:
    monkeypatch.setattr(ledger, "get_secret_str", lambda name: "250")
    assert ledger.openrouter_free_daily_request_limit() == 250
    assert ledger.openrouter_free_daily_windows()[0].limit == 250


@pytest.mark.asyncio
async def test_should_allow_same_call_id_across_providers_and_scopes(monkeypatch) -> None:
    conn = _LedgerConnection()
    first = await _record(
        conn,
        monkeypatch,
        provider="openrouter",
        credential_scope="openrouter_free_daily_shared",
        windows=ledger.openrouter_free_daily_windows(limit=1000),
    )
    second = await _record(
        conn,
        monkeypatch,
        provider="cohere",
        credential_scope="cohere_trial_default",
        lane="cohere_native",
        windows=ledger.cohere_trial_windows(rpm_limit=20),
    )
    third = await _record(
        conn,
        monkeypatch,
        provider="openrouter",
        credential_scope="other-scope",
        windows=ledger.openrouter_free_daily_windows(limit=1000),
    )

    assert first.counted is True
    assert second.counted is True
    assert third.counted is True
    assert [row["provider"] for row in conn.rows] == [
        "openrouter",
        "cohere",
        "openrouter",
    ]
    assert {row["credential_scope"] for row in conn.rows} == {
        "openrouter_free_daily_shared",
        "cohere_trial_default",
        "other-scope",
    }
    assert {row["litellm_call_id"] for row in conn.rows} == {"call-1"}
    assert conn.lock_args == [
        ("openrouter", "openrouter_free_daily_shared"),
        ("cohere", "cohere_trial_default"),
        ("openrouter", "other-scope"),
    ]


@pytest.mark.asyncio
async def test_should_ignore_replayed_call_id_for_same_provider_and_scope(
    monkeypatch,
) -> None:
    conn = _LedgerConnection()
    first = await _record(conn, monkeypatch)
    replay = await _record(conn, monkeypatch)

    assert first.counted is True
    assert replay.counted is False
    assert first.windows["daily"].used == replay.windows["daily"].used == 1
    assert len(conn.rows) == 1


@pytest.mark.asyncio
async def test_should_count_daily_window_only_for_utc_day(monkeypatch) -> None:
    conn = _LedgerConnection()
    day = datetime(2026, 8, 19, 23, 59, tzinfo=timezone.utc)
    previous_day = datetime(2026, 8, 18, 23, 59, tzinfo=timezone.utc)
    next_day = datetime(2026, 8, 20, 0, 0, tzinfo=timezone.utc)
    windows = ledger.openrouter_free_daily_windows(limit=1000)

    await _record(
        conn,
        monkeypatch,
        litellm_call_id="same-day",
        accepted_at=day,
        windows=windows,
    )
    await _record(
        conn,
        monkeypatch,
        litellm_call_id="previous-day",
        accepted_at=previous_day,
        windows=windows,
    )
    later = await _record(
        conn,
        monkeypatch,
        litellm_call_id="next-day",
        accepted_at=next_day,
        windows=windows,
    )
    same_day = await _record(
        conn,
        monkeypatch,
        litellm_call_id="same-day-late",
        accepted_at=datetime(2026, 8, 19, 0, 1, tzinfo=timezone.utc),
        windows=windows,
    )

    assert same_day.windows["daily"].used == 2
    assert same_day.windows["daily"].remaining == 998
    assert later.windows["daily"].used == 1
    assert later.windows["daily"].remaining == 999
    _, same_day_args = conn.fetchval_calls[-1]
    assert same_day_args[2] == datetime(2026, 8, 19, tzinfo=timezone.utc)
    assert same_day_args[3] == datetime(2026, 8, 20, tzinfo=timezone.utc)


@pytest.mark.asyncio
async def test_should_scope_rolling_window_to_model_when_requested(monkeypatch) -> None:
    conn = _LedgerConnection()
    accepted_at = datetime(2026, 8, 19, 12, 0, tzinfo=timezone.utc)
    windows = (
        ledger.CountWindow(
            name="rpm",
            period="rolling_seconds",
            limit=20,
            model_scoped=True,
            window_seconds=60,
        ),
    )
    await _record(
        conn,
        monkeypatch,
        litellm_call_id="north",
        accepted_at=accepted_at - timedelta(seconds=10),
        provider="cohere",
        credential_scope="cohere_trial_default",
        model="cohere/north-mini-code-1-0",
        windows=windows,
    )
    await _record(
        conn,
        monkeypatch,
        litellm_call_id="other",
        accepted_at=accepted_at - timedelta(seconds=5),
        provider="cohere",
        credential_scope="cohere_trial_default",
        model="cohere/other-model",
        windows=windows,
    )
    stale = await _record(
        conn,
        monkeypatch,
        litellm_call_id="stale-north",
        accepted_at=accepted_at - timedelta(seconds=60),
        provider="cohere",
        credential_scope="cohere_trial_default",
        model="cohere/north-mini-code-1-0",
        windows=windows,
    )
    current = await _record(
        conn,
        monkeypatch,
        litellm_call_id="current-north",
        accepted_at=accepted_at,
        provider="cohere",
        credential_scope="cohere_trial_default",
        model="cohere/north-mini-code-1-0",
        windows=windows,
    )

    assert stale.windows["rpm"].used == 1
    assert current.windows["rpm"].used == 2
    assert current.windows["rpm"].remaining == 18
    assert "AND model IS NOT DISTINCT FROM $5::text" in conn.fetchval_calls[-1][0]


@pytest.mark.asyncio
async def test_should_return_used_count_and_unknown_remaining_without_numeric_limit(
    monkeypatch,
) -> None:
    conn = _LedgerConnection()
    first = await _record(
        conn,
        monkeypatch,
        provider=ledger.OPENCODE_ZEN_PROVIDER,
        credential_scope="opencode_zen_default",
        windows=ledger.opencode_zen_windows(),
    )
    second = await _record(
        conn,
        monkeypatch,
        litellm_call_id="call-2",
        provider=ledger.NVIDIA_NIM_PROVIDER,
        credential_scope="nvidia_nim_default",
        windows=ledger.nvidia_nim_windows(),
    )
    replay = await _record(
        conn,
        monkeypatch,
        provider=ledger.OPENCODE_ZEN_PROVIDER,
        credential_scope="opencode_zen_default",
        windows=ledger.opencode_zen_windows(),
    )

    assert first.counted is True
    assert first.windows["usage"].used == 1
    assert first.windows["usage"].limit is None
    assert first.windows["usage"].remaining is None
    assert second.counted is True
    assert second.windows["usage"].used == 1
    assert second.windows["usage"].remaining is None
    assert replay.counted is False
    assert replay.windows["usage"].used == 1
    assert replay.windows["usage"].remaining is None
