"""RR-069: rate_limit observation signature parity + shared target connection."""

from __future__ import annotations

import asyncio
import importlib.util
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, List

import pytest

_REPO = Path(__file__).resolve().parents[3]
_SCRIPT = _REPO / "scripts" / "backfill_rate_limit_observations.py"


def _load_module() -> ModuleType:
    name = "backfill_rate_limit_observations_rr069"
    # Always reload so residual patches are visible during iterative runs.
    sys.modules.pop(name, None)
    spec = importlib.util.spec_from_file_location(name, _SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def mod() -> ModuleType:
    return _load_module()


def _dt(minute: int = 0) -> datetime:
    return datetime(2026, 7, 17, 12, minute, 0, tzinfo=timezone.utc)


def _observation(**overrides: Any) -> Dict[str, Any]:
    base: Dict[str, Any] = {
        "source": "langfuse",
        "provider": "openai",
        "model": "gpt-5",
        "quota_key": "requests",
        "observed_at": _dt(0),
        "provider_resets_at": _dt(30),
        "remaining_pct": 12.5,
        "quota_limit": 1000,
        "quota_used": 875,
        "quota_remaining": 125,
        "billing_period_start_at": _dt(0),
        "billing_period_end_at": _dt(59),
        "trace_id": "trace-1",
        "litellm_call_id": "call-1",
        # storage extractors may also read limit_id; keep explicit storage fields.
        "limit_id": "requests",
        "limit_scope": "requests",
    }
    base.update(overrides)
    return base


def test_observation_signature_is_13_fields_and_matches_db_row_path(
    mod: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    """RR-069 #1: write path and DB path share one canonical signature."""
    # Bypass storage helpers so the test isolates signature shape, not extractor
    # semantics. When extractors return the raw field, both builders must agree.
    for name in (
        "_rate_limit_storage_provider",
        "_rate_limit_storage_quota_key",
        "_rate_limit_storage_remaining_pct",
        "_rate_limit_storage_quota_limit",
        "_rate_limit_storage_quota_used",
        "_rate_limit_storage_quota_remaining",
        "_rate_limit_storage_billing_period_start_at",
        "_rate_limit_storage_billing_period_end_at",
    ):
        monkeypatch.setattr(
            mod,
            name,
            lambda observation, key=None, _n=name: observation.get(
                {
                    "_rate_limit_storage_provider": "provider",
                    "_rate_limit_storage_quota_key": "quota_key",
                    "_rate_limit_storage_remaining_pct": "remaining_pct",
                    "_rate_limit_storage_quota_limit": "quota_limit",
                    "_rate_limit_storage_quota_used": "quota_used",
                    "_rate_limit_storage_quota_remaining": "quota_remaining",
                    "_rate_limit_storage_billing_period_start_at": "billing_period_start_at",
                    "_rate_limit_storage_billing_period_end_at": "billing_period_end_at",
                }[_n]
            ),
        )

    obs = _observation()
    write_sig = mod._observation_signature(obs)
    db_row = {
        "source": obs["source"],
        "provider": obs["provider"],
        "model": obs["model"],
        "quota_key": obs["quota_key"],
        "observed_at": obs["observed_at"],
        "provider_resets_at": obs["provider_resets_at"],
        "remaining_pct": obs["remaining_pct"],
        "quota_limit": obs["quota_limit"],
        "quota_used": obs["quota_used"],
        "quota_remaining": obs["quota_remaining"],
        "billing_period_start_at": obs["billing_period_start_at"],
        "billing_period_end_at": obs["billing_period_end_at"],
        "trace_id": obs["trace_id"],
    }
    db_sig = mod._observation_signature_from_db_row(db_row)
    assert len(write_sig) == 13
    assert len(db_sig) == 13
    assert write_sig == db_sig
    # Drifted 8-tuple must never match.
    drifted = write_sig[:7] + (write_sig[-1],)
    assert write_sig not in {drifted}


def test_write_signature_matches_db_row_with_real_storage_extractors(
    mod: ModuleType,
) -> None:
    """RR-069 #1 residual: real extractors must agree with stored columns.

    Runtime observations carry ``limit_id``/``limit_scope``; the DB stores the
    composite ``quota_key`` (and other storage-normalized columns). Skip-existing
    must still treat those as the same signature without monkeypatched helpers.
    """
    obs = {
        "source": "langfuse",
        "provider": "openai",
        "model": "gpt-5",
        "limit_id": "requests",
        "limit_scope": "requests",
        "observed_at": _dt(0),
        "provider_resets_at": _dt(30),
        "remaining_pct": 12.5,
        "quota_limit": 1000,
        "quota_used": 875,
        "quota_remaining": 125,
        "billing_period_start_at": _dt(0),
        "billing_period_end_at": _dt(59),
        "trace_id": "trace-real",
        "litellm_call_id": "call-real",
    }
    write_sig = mod._observation_signature(obs)
    db_row = {
        "source": obs["source"],
        "provider": mod._rate_limit_storage_provider(obs),
        "model": obs["model"],
        "quota_key": mod._rate_limit_storage_quota_key(obs),
        "observed_at": obs["observed_at"],
        "provider_resets_at": obs["provider_resets_at"],
        "remaining_pct": mod._rate_limit_storage_remaining_pct(obs),
        "quota_limit": mod._rate_limit_storage_quota_limit(obs),
        "quota_used": mod._rate_limit_storage_quota_used(obs),
        "quota_remaining": mod._rate_limit_storage_quota_remaining(obs),
        "billing_period_start_at": mod._rate_limit_storage_billing_period_start_at(obs),
        "billing_period_end_at": mod._rate_limit_storage_billing_period_end_at(obs),
        "trace_id": obs["trace_id"],
    }
    assert write_sig == mod._observation_signature_from_db_row(db_row)
    assert write_sig[3] == "requests:requests"


def test_filter_existing_uses_shared_signature_and_optional_conn(
    mod: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name in (
        "_rate_limit_storage_provider",
        "_rate_limit_storage_quota_key",
        "_rate_limit_storage_remaining_pct",
        "_rate_limit_storage_quota_limit",
        "_rate_limit_storage_quota_used",
        "_rate_limit_storage_quota_remaining",
        "_rate_limit_storage_billing_period_start_at",
        "_rate_limit_storage_billing_period_end_at",
    ):
        field = {
            "_rate_limit_storage_provider": "provider",
            "_rate_limit_storage_quota_key": "quota_key",
            "_rate_limit_storage_remaining_pct": "remaining_pct",
            "_rate_limit_storage_quota_limit": "quota_limit",
            "_rate_limit_storage_quota_used": "quota_used",
            "_rate_limit_storage_quota_remaining": "quota_remaining",
            "_rate_limit_storage_billing_period_start_at": "billing_period_start_at",
            "_rate_limit_storage_billing_period_end_at": "billing_period_end_at",
        }[name]
        monkeypatch.setattr(mod, name, lambda observation, f=field: observation.get(f))

    existing_obs = _observation()
    new_obs = _observation(
        trace_id="trace-2", litellm_call_id="call-2", remaining_pct=1.0
    )

    class FakeConn:
        def __init__(self) -> None:
            self.fetch_calls = 0
            self.closed = False

        async def fetch(self, sql: str, *args: Any) -> List[Dict[str, Any]]:
            self.fetch_calls += 1
            return [
                {
                    "source": existing_obs["source"],
                    "provider": existing_obs["provider"],
                    "model": existing_obs["model"],
                    "quota_key": existing_obs["quota_key"],
                    "observed_at": existing_obs["observed_at"],
                    "provider_resets_at": existing_obs["provider_resets_at"],
                    "remaining_pct": existing_obs["remaining_pct"],
                    "quota_limit": existing_obs["quota_limit"],
                    "quota_used": existing_obs["quota_used"],
                    "quota_remaining": existing_obs["quota_remaining"],
                    "billing_period_start_at": existing_obs["billing_period_start_at"],
                    "billing_period_end_at": existing_obs["billing_period_end_at"],
                    "trace_id": existing_obs["trace_id"],
                }
            ]

        async def close(self) -> None:
            self.closed = True

    conn = FakeConn()
    records = [
        {
            "rate_limit_observations": [existing_obs, new_obs],
            "_skip_session_history": True,
        }
    ]
    filtered = asyncio.run(mod._filter_existing_observations(records, conn=conn))
    assert conn.fetch_calls == 1
    assert conn.closed is False  # shared conn must not be closed by filter
    assert len(filtered) == 1
    assert len(filtered[0]["rate_limit_observations"]) == 1
    assert filtered[0]["rate_limit_observations"][0]["trace_id"] == "trace-2"


def test_run_reuses_one_target_connection_across_batches(
    mod: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    """RR-069 #2: one acquire/schema for the whole apply run, not per batch."""
    connects: List[str] = []
    ensure_calls = {"n": 0}
    filter_conns: List[int] = []

    class FakeConn:
        def __init__(self) -> None:
            self.id = id(self)
            self.closed = False

        async def fetchval(self, sql: str, *args: Any) -> str:
            return "aawm_tristore"

        async def fetch(self, sql: str, *args: Any) -> list[Any]:
            return []

        async def close(self) -> None:
            self.closed = True

    async def fake_connect(dsn: str, *args: Any, **kwargs: Any) -> FakeConn:
        connects.append(dsn)
        return FakeConn()

    async def fake_ensure(conn: Any) -> None:
        ensure_calls["n"] += 1

    async def fake_filter(
        records: List[Dict[str, Any]], *, conn: Any = None
    ) -> List[Dict[str, Any]]:
        filter_conns.append(id(conn) if conn is not None else -1)
        return records

    async def fake_persist_session(records: List[Dict[str, Any]]) -> None:
        return None

    class FakeCH:
        def __init__(self, **kwargs: Any) -> None:
            self._batches = [
                [
                    {
                        "observation_id": "o1",
                        "observation_start_time": "2026-07-17 12:00:00",
                    }
                ],
                [
                    {
                        "observation_id": "o2",
                        "observation_start_time": "2026-07-17 12:01:00",
                    }
                ],
                [],
            ]

        async def fetch_observation_batch(self, **kwargs: Any) -> list[Any]:
            return self._batches.pop(0)

    def fake_build_record(
        row: Dict[str, Any], include_input: bool = False
    ) -> Dict[str, Any]:
        return {
            "rate_limit_observations": [
                _observation(
                    litellm_call_id=str(row["observation_id"]),
                    trace_id=str(row["observation_id"]),
                )
            ],
            "_skip_session_history": True,
        }

    monkeypatch.setattr(mod.asyncpg, "connect", fake_connect)
    monkeypatch.setattr(mod, "_ensure_session_history_schema", fake_ensure)
    monkeypatch.setattr(
        mod, "_build_aawm_admin_dsn", lambda: "postgresql://x/aawm_tristore"
    )
    monkeypatch.setattr(mod, "_filter_existing_observations", fake_filter)
    monkeypatch.setattr(mod, "_persist_session_history_records", fake_persist_session)
    monkeypatch.setattr(mod, "ClickHouseClient", FakeCH)
    monkeypatch.setattr(mod, "build_record_from_clickhouse_row", fake_build_record)
    monkeypatch.setattr(
        mod, "_should_preflight_clickhouse_for_source_mode", lambda _m: False
    )
    monkeypatch.setattr(
        mod,
        "_resolve_clickhouse_auth_sources",
        lambda args: {
            "url": "http://ch",
            "user": "u",
            "password": "p",
            "timeout_seconds": 1,
            "url_source": "t",
            "user_source": "t",
            "password_source": "t",
            "url_input": None,
        },
    )
    monkeypatch.setattr(
        mod,
        "_resolve_clickhouse_config",
        lambda args: {
            "url": "http://ch",
            "user": "u",
            "password": "p",
            "timeout_seconds": 1,
        },
    )
    monkeypatch.setattr(mod, "_clickhouse_auth_diagnostics", lambda auth: {"ok": True})

    args = mod.build_parser().parse_args(
        ["--source-mode", "clickhouse", "--apply", "--batch-size", "1"]
    )
    result = asyncio.run(mod._run(args))

    assert result["applied"] is True
    assert len(connects) == 1
    assert ensure_calls["n"] == 1
    assert len(filter_conns) == 2  # two non-empty batches
    assert len(set(filter_conns)) == 1  # same conn object id reused
    assert filter_conns[0] != -1


def test_canonical_signature_length_constant(mod: ModuleType) -> None:
    sig = mod._canonical_observation_signature(
        source="s",
        provider="p",
        model="m",
        quota_key="q",
        observed_at=_dt(0),
        provider_resets_at=None,
        remaining_pct=1,
        quota_limit=2,
        quota_used=3,
        quota_remaining=4,
        billing_period_start_at=None,
        billing_period_end_at=None,
        trace_id="t",
    )
    assert len(sig) == 13


def test_signature_datetime_normalizes_naive_and_aware(mod: ModuleType) -> None:
    """DB/asyncpg may return naive UTC; write path is usually aware."""
    aware = datetime(2026, 7, 17, 12, 0, 0, 123000, tzinfo=timezone.utc)
    naive = datetime(2026, 7, 17, 12, 0, 0, 123000)
    assert mod._signature_datetime_millis(aware) == mod._signature_datetime_millis(
        naive
    )
    assert mod._signature_datetime_millis(aware).endswith("+00:00")


def test_signature_numeric_normalizes_decimal_and_int(mod: ModuleType) -> None:
    from decimal import Decimal

    assert mod._signature_numeric(Decimal("12.5")) == 12.5
    assert mod._signature_numeric(12) == 12.0
    assert mod._signature_numeric("12.5") == 12.5
    assert mod._signature_numeric(None) is None
    assert mod._signature_numeric("nope") is None


def test_observation_signature_matches_naive_db_row(
    mod: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name in (
        "_rate_limit_storage_provider",
        "_rate_limit_storage_quota_key",
        "_rate_limit_storage_remaining_pct",
        "_rate_limit_storage_quota_limit",
        "_rate_limit_storage_quota_used",
        "_rate_limit_storage_quota_remaining",
        "_rate_limit_storage_billing_period_start_at",
        "_rate_limit_storage_billing_period_end_at",
    ):
        field = {
            "_rate_limit_storage_provider": "provider",
            "_rate_limit_storage_quota_key": "quota_key",
            "_rate_limit_storage_remaining_pct": "remaining_pct",
            "_rate_limit_storage_quota_limit": "quota_limit",
            "_rate_limit_storage_quota_used": "quota_used",
            "_rate_limit_storage_quota_remaining": "quota_remaining",
            "_rate_limit_storage_billing_period_start_at": "billing_period_start_at",
            "_rate_limit_storage_billing_period_end_at": "billing_period_end_at",
        }[name]
        monkeypatch.setattr(mod, name, lambda observation, f=field: observation.get(f))

    obs = _observation()
    write_sig = mod._observation_signature(obs)
    from decimal import Decimal

    db_row = {
        "source": obs["source"],
        "provider": obs["provider"],
        "model": obs["model"],
        "quota_key": obs["quota_key"],
        "observed_at": obs["observed_at"].replace(tzinfo=None),
        "provider_resets_at": obs["provider_resets_at"].replace(tzinfo=None),
        "remaining_pct": Decimal("12.5"),
        "quota_limit": Decimal("1000"),
        "quota_used": Decimal("875"),
        "quota_remaining": Decimal("125"),
        "billing_period_start_at": obs["billing_period_start_at"].replace(tzinfo=None),
        "billing_period_end_at": obs["billing_period_end_at"].replace(tzinfo=None),
        "trace_id": obs["trace_id"],
    }
    assert write_sig == mod._observation_signature_from_db_row(db_row)


def test_parse_optional_datetime_accepts_space_and_rejects_none_string(
    mod: ModuleType,
) -> None:
    parsed = mod._parse_optional_datetime("2026-05-05 15:00:01")
    assert parsed is not None
    assert parsed.tzinfo is not None
    assert parsed.year == 2026 and parsed.hour == 15
    assert mod._parse_optional_datetime(None) is None
    assert mod._parse_optional_datetime("None") is None
    assert mod._parse_optional_datetime("") is None
    # Cursor path must tolerate raw ClickHouse nulls without raising.
    assert mod._parse_optional_datetime(mod._parse_optional_datetime(None)) is None


def test_filter_existing_one_shot_conn_fallback_closes(
    mod: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When no shared conn is passed, filter still owns/closes its one-shot conn."""
    for name in (
        "_rate_limit_storage_provider",
        "_rate_limit_storage_quota_key",
        "_rate_limit_storage_remaining_pct",
        "_rate_limit_storage_quota_limit",
        "_rate_limit_storage_quota_used",
        "_rate_limit_storage_quota_remaining",
        "_rate_limit_storage_billing_period_start_at",
        "_rate_limit_storage_billing_period_end_at",
    ):
        field = {
            "_rate_limit_storage_provider": "provider",
            "_rate_limit_storage_quota_key": "quota_key",
            "_rate_limit_storage_remaining_pct": "remaining_pct",
            "_rate_limit_storage_quota_limit": "quota_limit",
            "_rate_limit_storage_quota_used": "quota_used",
            "_rate_limit_storage_quota_remaining": "quota_remaining",
            "_rate_limit_storage_billing_period_start_at": "billing_period_start_at",
            "_rate_limit_storage_billing_period_end_at": "billing_period_end_at",
        }[name]
        monkeypatch.setattr(mod, name, lambda observation, f=field: observation.get(f))

    class FakeConn:
        def __init__(self) -> None:
            self.closed = False
            self.ensure = 0

        async def fetch(self, sql: str, *args: Any) -> List[Dict[str, Any]]:
            return []

        async def close(self) -> None:
            self.closed = True

    conn = FakeConn()

    async def fake_connect(dsn: str, *args: Any, **kwargs: Any) -> FakeConn:
        return conn

    async def fake_ensure(c: Any) -> None:
        conn.ensure += 1

    monkeypatch.setattr(mod.asyncpg, "connect", fake_connect)
    monkeypatch.setattr(mod, "_ensure_session_history_schema", fake_ensure)
    monkeypatch.setattr(
        mod, "_build_aawm_admin_dsn", lambda: "postgresql://x/aawm_tristore"
    )

    records = [
        {"rate_limit_observations": [_observation()], "_skip_session_history": True}
    ]
    filtered = asyncio.run(mod._filter_existing_observations(records))
    assert filtered  # nothing existing
    assert conn.closed is True
    assert conn.ensure == 1


def test_keyset_cursor_from_row_rejects_null_time(mod: ModuleType) -> None:
    """Bounded keyset: null/missing page cursors must fail closed, not loop."""
    with pytest.raises(RuntimeError, match="keyset cursor cannot advance"):
        mod._keyset_cursor_from_row(
            {"observation_start_time": None, "observation_id": "o1"},
            time_field="observation_start_time",
            id_field="observation_id",
        )
    with pytest.raises(RuntimeError, match="keyset cursor cannot advance"):
        mod._keyset_cursor_from_row(
            {"observation_start_time": "2026-07-17 12:00:00", "observation_id": ""},
            time_field="observation_start_time",
            id_field="observation_id",
        )
    with pytest.raises(RuntimeError, match="keyset cursor cannot advance"):
        mod._keyset_cursor_from_row(
            {"observation_start_time": "None", "observation_id": "o1"},
            time_field="observation_start_time",
            id_field="observation_id",
        )


def test_keyset_cursor_from_row_parses_clickhouse_space_timestamp(
    mod: ModuleType,
) -> None:
    cursor_time, cursor_id = mod._keyset_cursor_from_row(
        {
            "observation_start_time": "2026-07-17 12:00:01.123",
            "observation_id": "obs-abc",
        },
        time_field="observation_start_time",
        id_field="observation_id",
    )
    assert cursor_id == "obs-abc"
    assert cursor_time == datetime(2026, 7, 17, 12, 0, 1, 123000, tzinfo=timezone.utc)


def test_assert_keyset_progress_rejects_stalled_cursor(mod: ModuleType) -> None:
    current = (
        datetime(2026, 7, 17, 12, 0, 0, tzinfo=timezone.utc),
        "obs-1",
    )
    with pytest.raises(RuntimeError, match="did not progress"):
        mod._assert_keyset_progress(current, current)
    # First page (no previous) always allowed.
    mod._assert_keyset_progress(None, current)
    next_cursor = (
        datetime(2026, 7, 17, 11, 59, 0, tzinfo=timezone.utc),
        "obs-0",
    )
    mod._assert_keyset_progress(current, next_cursor)


def test_run_clickhouse_backfill_fails_closed_on_stalled_keyset(
    mod: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If a page cannot advance the keyset, stop instead of rescanning forever."""

    class FakeCH:
        def __init__(self) -> None:
            self.calls = 0

        async def fetch_observation_batch(self, **kwargs: Any) -> list[Any]:
            self.calls += 1
            # Same last-row identity every page: classic non-progressing keyset.
            return [
                {
                    "observation_id": "stuck-id",
                    "observation_start_time": "2026-07-17 12:00:00",
                }
            ]

    async def fake_persist(*args: Any, **kwargs: Any) -> None:
        return None

    monkeypatch.setattr(mod, "build_record_from_clickhouse_row", lambda row, include_input=False: None)
    monkeypatch.setattr(mod, "_persist_records", fake_persist)

    client = FakeCH()
    args = mod.build_parser().parse_args(
        ["--source-mode", "clickhouse", "--batch-size", "1"]
    )
    with pytest.raises(RuntimeError, match="did not progress"):
        asyncio.run(mod._run_clickhouse_backfill(args, client))
    # First page succeeds; second page detects stalled cursor and aborts.
    assert client.calls == 2


def test_run_clickhouse_backfill_fails_closed_on_unusable_last_row_cursor(
    mod: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    class FakeCH:
        async def fetch_observation_batch(self, **kwargs: Any) -> list[Any]:
            return [
                {
                    "observation_id": "ok-id",
                    "observation_start_time": None,
                }
            ]

    async def fake_persist(*args: Any, **kwargs: Any) -> None:
        return None

    monkeypatch.setattr(mod, "build_record_from_clickhouse_row", lambda row, include_input=False: None)
    monkeypatch.setattr(mod, "_persist_records", fake_persist)
    args = mod.build_parser().parse_args(
        ["--source-mode", "clickhouse", "--batch-size", "1"]
    )
    with pytest.raises(RuntimeError, match="keyset cursor cannot advance"):
        asyncio.run(mod._run_clickhouse_backfill(args, FakeCH()))


def test_build_record_does_not_mutate_source_metadata(mod: ModuleType) -> None:
    """Safe metadata updates: provenance tags must not write into caller maps."""
    source_metadata = {
        "session_id": "sess-1",
        "custom_llm_provider": "openai",
        "codex_response_headers": {
            "source": "codex_response_headers",
            "x-codex-primary-used-percent": "10",
            "x-codex-primary-window-minutes": "300",
            "x-codex-primary-reset-at": "1778000000",
            "x-codex-secondary-used-percent": "20",
            "x-codex-secondary-window-minutes": "10080",
            "x-codex-secondary-reset-at": "1778018910",
            "x-codex-limit-id": "codex_test",
            "x-codex-limit-name": "test-model",
        },
    }
    body = {
        "id": "obs-meta-safe",
        "traceId": "trace-meta-safe",
        "name": "litellm-pass_through_endpoint",
        "startTime": "2026-05-05T15:00:00Z",
        "endTime": "2026-05-05T15:00:01Z",
        "metadata": source_metadata,
        "output": None,
        "model": "gpt-5.5",
        "environment": "dev",
    }
    snapshot = dict(source_metadata)
    record = mod._build_record_from_langfuse_body(
        body,
        backfill_source="langfuse_clickhouse",
        source_locator="obs-meta-safe",
    )
    assert record is not None
    # Caller-owned map stays free of backfill provenance tags.
    assert "rate_limit_backfill_source" not in source_metadata
    assert "rate_limit_backfill_locator" not in source_metadata
    assert source_metadata == snapshot
    # Body still points at the original metadata object.
    assert body["metadata"] is source_metadata
    assert body["metadata"].get("rate_limit_backfill_source") is None


def test_parse_clickhouse_map_returns_fresh_dict_for_json_string(
    mod: ModuleType,
) -> None:
    import json

    # Top-level JSON string map, with a nested JSON-encoded value.
    raw = json.dumps({"session_id": "s1", "nested": json.dumps({"k": 1})})
    first = mod._parse_clickhouse_map(raw)
    second = mod._parse_clickhouse_map(raw)
    assert first == second
    assert first is not second
    first["rate_limit_backfill_source"] = "mutated"
    assert "rate_limit_backfill_source" not in second
    # Nested JSON values are expanded once into independent structures.
    assert first["nested"] == {"k": 1}
    first["nested"]["k"] = 99
    third = mod._parse_clickhouse_map(raw)
    assert third["nested"] == {"k": 1}

    # Plain dict input also returns a fresh copy (no source mutation).
    source = {"session_id": "s2", "nested": json.dumps({"k": 2})}
    parsed = mod._parse_clickhouse_map(source)
    parsed["rate_limit_backfill_source"] = "x"
    assert "rate_limit_backfill_source" not in source
    assert parsed["nested"] == {"k": 2}



def test_build_record_from_clickhouse_row_keeps_row_metadata_pristine(
    mod: ModuleType,
) -> None:
    row_metadata = {
        "client_name": "codex",
        "passthrough_route_family": "codex_responses",
    }
    row = {
        "observation_id": "obs-row-meta",
        "observation_trace_id": "trace-row-meta",
        "observation_start_time": "2026-05-05T15:00:00Z",
        "observation_end_time": "2026-05-05T15:00:01Z",
        "observation_name": "litellm-pass_through_endpoint",
        "observation_metadata": row_metadata,
        "observation_input": None,
        "observation_output": (
            '{"rate_limits":{"limit_id":"codex","limit_name":"m",'
            '"primary":{"used_percent":10.0,"window_minutes":5,"resets_at":1778000000},'
            '"secondary":{"used_percent":20.0,"window_minutes":60,"resets_at":1778018910}}}'
        ),
        "observation_model": "gpt-5.5",
        "observation_environment": "dev",
    }
    before = dict(row_metadata)
    record = mod.build_record_from_clickhouse_row(row)
    assert record is not None
    assert row["observation_metadata"] is row_metadata
    assert row_metadata == before
    assert "rate_limit_backfill_source" not in row_metadata
