"""D1-610: dry-run-first session_history reference-pricing repair.

Uses a fake connection only. Must not open production Postgres.
"""

from __future__ import annotations

import argparse
import re
from typing import Any, Optional

import pytest

GPT_4O_MINI_INPUT_COST_PER_TOKEN = 1.5e-07
GPT_4O_MINI_OUTPUT_COST_PER_TOKEN = 6e-07
GPT_4O_MINI_REFERENCE_COST_USD = (
    1000 * GPT_4O_MINI_INPUT_COST_PER_TOKEN + 500 * GPT_4O_MINI_OUTPUT_COST_PER_TOKEN
)


def _load_repair():
    """Import is inside the test body so collection still runs on a missing module."""
    from scripts import repair_session_history_reference_pricing as repair

    return repair


class _Cursor:
    def __init__(self, *, database_name: str, rows: Optional[list[dict[str, Any]]] = None) -> None:
        self.database_name = database_name
        self.rows = [dict(row) for row in (rows or [])]
        self.executed: list[tuple[str, Any]] = []
        self.executemany_calls: list[tuple[str, Any]] = []
        self._last_result: list[dict[str, Any]] = []

    def __enter__(self) -> "_Cursor":
        return self

    def __exit__(self, *args: Any) -> None:
        return None

    def _row_by_id(self, row_id: Any) -> Optional[dict[str, Any]]:
        for row in self.rows:
            if row.get("id") == row_id:
                return row
        return None

    def _apply_update_params(self, params: Any) -> None:
        if params is None:
            return
        values = list(params)
        row_id = None
        reference_cost = None
        for value in values:
            if isinstance(value, int) and self._row_by_id(value) is not None:
                row_id = value
            elif isinstance(value, float):
                reference_cost = value
        row = self._row_by_id(row_id)
        if row is None:
            return
        if reference_cost is not None:
            row["reference_cost_usd"] = reference_cost
            row["actual_invoice_cost_known"] = False

    def execute(self, statement: str, params: Any = None) -> None:
        self.executed.append((statement, params))
        sql = " ".join(str(statement).split())
        lowered = sql.lower()
        if "current_database()" in lowered:
            self._last_result = []
            return
        if lowered.lstrip().startswith("select") and "session_history" in lowered:
            rows = [dict(row) for row in self.rows]
            limit_match = re.search(r"\blimit\s+%s", lowered) or re.search(
                r"\blimit\s+(\d+)", lowered
            )
            limit_value: Optional[int] = None
            if limit_match and limit_match.lastindex:
                limit_value = int(limit_match.group(1))
            elif params:
                for value in reversed(list(params) if not isinstance(params, dict) else params.values()):
                    if isinstance(value, int) and value > 0:
                        limit_value = value
                        break
            if limit_value is not None:
                rows = rows[:limit_value]
            self._last_result = rows
            return
        if "update" in lowered and "session_history" in lowered:
            self._apply_update_params(params)
        self._last_result = []

    def executemany(self, statement: str, params: Any = None) -> None:
        self.executemany_calls.append((statement, params))
        for item in params or []:
            self._apply_update_params(item)

    def fetchone(self) -> Any:
        last = self.executed[-1][0] if self.executed else ""
        if "current_database()" in last:
            return {"current_database": self.database_name}
        if self._last_result:
            return self._last_result[0]
        return None

    def fetchall(self) -> list[dict[str, Any]]:
        if self._last_result:
            return list(self._last_result)
        last = self.executed[-1][0] if self.executed else ""
        if "session_history" in str(last).lower():
            return [dict(row) for row in self.rows]
        return []


class _Conn:
    def __init__(self, cursor: _Cursor) -> None:
        self.cursor_instance = cursor
        self.committed = False
        self.rolled_back = False

    def __enter__(self) -> "_Conn":
        return self

    def __exit__(self, *args: Any) -> None:
        return None

    def cursor(self, *, row_factory=None) -> _Cursor:
        return self.cursor_instance

    def commit(self) -> None:
        self.committed = True

    def rollback(self) -> None:
        self.rolled_back = True


def _args(**overrides: Any) -> argparse.Namespace:
    payload = dict(
        apply=False,
        target_db_name="aawm_tristore",
        batch_size=50,
        limit=None,
        ensure_schema=False,
        provider=None,
        session_id=None,
        preview_limit=5,
    )
    payload.update(overrides)
    return argparse.Namespace(**payload)


def test_module_exists_and_defaults_to_dry_run() -> None:
    repair = _load_repair()
    assert repair.__name__.endswith("repair_session_history_reference_pricing")
    parser = repair._build_arg_parser()
    args = parser.parse_args([])
    assert args.apply is False
    assert args.target_db_name == "aawm_tristore"
    assert getattr(args, "ensure_schema", False) is False


def test_apply_requires_current_database_to_match_target(monkeypatch) -> None:
    repair = _load_repair()
    cursor = _Cursor(database_name="wrong_db")
    conn = _Conn(cursor)
    monkeypatch.setattr(repair.psycopg, "connect", lambda *a, **k: conn)
    monkeypatch.setattr(repair, "_build_aawm_admin_dsn", lambda: "postgresql://unused")

    with pytest.raises(SystemExit, match="Refusing"):
        repair._run_repair(_args(apply=True, target_db_name="aawm_tristore"))
    assert conn.committed is False
    assert cursor.executemany_calls == []


def test_dry_run_does_not_write_when_database_name_differs(monkeypatch) -> None:
    repair = _load_repair()
    cursor = _Cursor(database_name="wrong_db", rows=[])
    conn = _Conn(cursor)
    monkeypatch.setattr(repair.psycopg, "connect", lambda *a, **k: conn)
    monkeypatch.setattr(repair, "_build_aawm_admin_dsn", lambda: "postgresql://unused")

    result = repair._run_repair(_args(apply=False, target_db_name="aawm_tristore"))
    assert result["mode"] == "dry_run"
    assert conn.committed is False
    assert cursor.executemany_calls == []


def test_does_not_guess_missing_token_breakdowns() -> None:
    repair = _load_repair()
    row = {
        "id": 11,
        "provider": "alibaba",
        "model": "qwen-plus",
        "alias": "alibaba-token-plan",
        "input_tokens": None,
        "output_tokens": None,
        "cache_read_input_tokens": None,
        "cache_creation_input_tokens": None,
        "response_cost_usd": 0.0,
        "actual_invoice_cost_known": False,
        "metadata": {},
    }
    repaired = repair._build_repaired_row(row)
    assert repaired is not None
    assert repaired.get("actual_invoice_cost_known") is False
    assert repaired.get("guessed_token_breakdown") is not True
    # Leave unknown token/cache/tier fields unknown rather than inventing counts.
    for field in (
        "input_tokens",
        "output_tokens",
        "cache_read_input_tokens",
        "cache_creation_input_tokens",
    ):
        assert repaired.get(field) in (None, row.get(field))
    assert repaired.get("reference_cost_usd") in (None, 0, 0.0) or repaired.get(
        "skip_reason"
    ) in {"missing_token_breakdown", "unknown_token_breakdown"}


def test_run_repair_uses_fake_connection_only(monkeypatch) -> None:
    repair = _load_repair()
    opened: list[str] = []
    cursor = _Cursor(database_name="aawm_tristore", rows=[])
    conn = _Conn(cursor)

    def _connect(dsn: str, **kwargs: Any) -> _Conn:
        opened.append(dsn)
        assert "production" not in dsn
        assert "aawm-litellm" not in dsn
        return conn

    monkeypatch.setattr(repair.psycopg, "connect", _connect)
    monkeypatch.setattr(repair, "_build_aawm_admin_dsn", lambda: "postgresql://fake-local/aawm_tristore")

    result = repair._run_repair(_args(apply=False))
    assert opened == ["postgresql://fake-local/aawm_tristore"]
    assert result["mode"] == "dry_run"
    assert conn.committed is False


def _eligible_pricing_row(*, row_id: int, reference_cost_usd: float = 0.0) -> dict[str, Any]:
    return {
        "id": row_id,
        "provider": "openai",
        "model": "gpt-4o-mini",
        "alias": "openai-gpt-4o-mini",
        "input_tokens": 1000,
        "output_tokens": 500,
        "cache_read_input_tokens": None,
        "cache_creation_input_tokens": None,
        "response_cost_usd": 0.0,
        "reference_cost_usd": reference_cost_usd,
        "actual_invoice_cost_known": False,
        "metadata": {},
    }


def _ineligible_pricing_row(*, row_id: int) -> dict[str, Any]:
    return {
        "id": row_id,
        "provider": "alibaba",
        "model": "qwen-plus",
        "alias": "alibaba-token-plan",
        "input_tokens": None,
        "output_tokens": None,
        "cache_read_input_tokens": None,
        "cache_creation_input_tokens": None,
        "response_cost_usd": 0.0,
        "reference_cost_usd": 0.0,
        "actual_invoice_cost_known": False,
        "metadata": {},
    }


def _patch_fake_db(monkeypatch, repair, cursor: _Cursor, conn: _Conn) -> None:
    def _connect(dsn: str, **kwargs: Any) -> _Conn:
        assert "production" not in dsn
        assert "aawm-litellm" not in dsn
        return conn

    monkeypatch.setattr(repair.psycopg, "connect", _connect)
    monkeypatch.setattr(
        repair,
        "_build_aawm_admin_dsn",
        lambda: "postgresql://fake-local/aawm_tristore_rr099",
    )


def _write_count(result: dict[str, Any]) -> int:
    for key in ("written", "repaired_rows", "rows_written"):
        if key in result and result[key] is not None:
            return int(result[key])
    return 0


def _assert_mutated_via_sql(cursor: _Cursor) -> None:
    sql_blobs = [str(statement) for statement, _params in cursor.executed]
    sql_blobs.extend(str(statement) for statement, _params in cursor.executemany_calls)
    joined = "\n".join(sql_blobs).lower()
    assert "update" in joined
    assert "session_history" in joined


def test_build_repaired_row_calculates_provider_equivalent_reference_cost() -> None:
    repair = _load_repair()
    repaired = repair._build_repaired_row(_eligible_pricing_row(row_id=21, reference_cost_usd=0.0))

    assert repaired is not None
    assert repaired.get("skip_reason") in {None, ""}
    assert repaired.get("actual_invoice_cost_known") is False
    assert repaired.get("guessed_token_breakdown") is not True
    assert repaired.get("input_tokens") == 1000
    assert repaired.get("output_tokens") == 500
    assert repaired.get("reference_cost_usd") == pytest.approx(GPT_4O_MINI_REFERENCE_COST_USD)
    assert repaired.get("reference_cost_usd") != 0.0


def test_apply_persists_calculated_repair_on_named_disposable_database(monkeypatch) -> None:
    repair = _load_repair()
    eligible = _eligible_pricing_row(row_id=21, reference_cost_usd=0.0)
    ineligible = _ineligible_pricing_row(row_id=22)
    cursor = _Cursor(
        database_name="aawm_tristore_rr099",
        rows=[eligible, ineligible],
    )
    conn = _Conn(cursor)
    _patch_fake_db(monkeypatch, repair, cursor, conn)

    result = repair._run_repair(
        _args(
            apply=True,
            target_db_name="aawm_tristore_rr099",
            limit=10,
            batch_size=10,
        )
    )

    assert result["mode"] == "apply"
    assert _write_count(result) == 1
    assert conn.committed is True
    _assert_mutated_via_sql(cursor)
    persisted = cursor._row_by_id(21)
    skipped = cursor._row_by_id(22)
    assert persisted is not None
    assert persisted["reference_cost_usd"] == pytest.approx(GPT_4O_MINI_REFERENCE_COST_USD)
    assert skipped is not None
    assert skipped["reference_cost_usd"] == 0.0
    assert skipped["input_tokens"] is None


def test_apply_honors_limit_and_batch_size_and_reports_real_write_count(monkeypatch) -> None:
    repair = _load_repair()
    cursor = _Cursor(
        database_name="aawm_tristore_rr099",
        rows=[
            _eligible_pricing_row(row_id=31, reference_cost_usd=0.0),
            _eligible_pricing_row(row_id=32, reference_cost_usd=0.0),
            _eligible_pricing_row(row_id=33, reference_cost_usd=0.0),
        ],
    )
    conn = _Conn(cursor)
    _patch_fake_db(monkeypatch, repair, cursor, conn)

    result = repair._run_repair(
        _args(
            apply=True,
            target_db_name="aawm_tristore_rr099",
            limit=2,
            batch_size=1,
        )
    )

    assert result["mode"] == "apply"
    assert _write_count(result) == 2
    assert conn.committed is True
    _assert_mutated_via_sql(cursor)
    sql_blob = "\n".join(str(statement) for statement, _params in cursor.executed).lower()
    sql_blob += "\n".join(str(statement) for statement, _params in cursor.executemany_calls).lower()
    assert "limit" in sql_blob
    assert any(
        params is not None
        and (
            params == (1,)
            or params == [1]
            or (isinstance(params, (list, tuple)) and 1 in params)
        )
        for _statement, params in cursor.executed
    ) or "limit 1" in sql_blob or "limit %s" in sql_blob
    updated_ids = {
        row["id"]
        for row in cursor.rows
        if row.get("reference_cost_usd") == pytest.approx(GPT_4O_MINI_REFERENCE_COST_USD)
    }
    assert len(updated_ids) == 2
    assert 33 not in updated_ids


def test_dry_run_with_eligible_rows_does_not_write(monkeypatch) -> None:
    repair = _load_repair()
    cursor = _Cursor(
        database_name="aawm_tristore_rr099",
        rows=[_eligible_pricing_row(row_id=41, reference_cost_usd=0.0)],
    )
    conn = _Conn(cursor)
    _patch_fake_db(monkeypatch, repair, cursor, conn)

    result = repair._run_repair(
        _args(
            apply=False,
            target_db_name="aawm_tristore_rr099",
            limit=10,
            batch_size=2,
        )
    )

    assert result["mode"] == "dry_run"
    assert _write_count(result) == 0
    assert conn.committed is False
    assert cursor.executemany_calls == []
    assert all("update" not in str(statement).lower() for statement, _params in cursor.executed)
    assert cursor._row_by_id(41)["reference_cost_usd"] == 0.0
