"""D1-610: dry-run-first session_history reference-pricing repair.

Uses a fake connection only. Must not open production Postgres.
"""

from __future__ import annotations

import argparse
from typing import Any, Optional

import pytest


def _load_repair():
    """Import is inside the test body so collection still runs on a missing module."""
    from scripts import repair_session_history_reference_pricing as repair

    return repair


class _Cursor:
    def __init__(self, *, database_name: str, rows: Optional[list[dict[str, Any]]] = None) -> None:
        self.database_name = database_name
        self.rows = list(rows or [])
        self.executed: list[tuple[str, Any]] = []
        self.executemany_calls: list[tuple[str, Any]] = []

    def __enter__(self) -> "_Cursor":
        return self

    def __exit__(self, *args: Any) -> None:
        return None

    def execute(self, statement: str, params: Any = None) -> None:
        self.executed.append((statement, params))

    def executemany(self, statement: str, params: Any = None) -> None:
        self.executemany_calls.append((statement, params))

    def fetchone(self) -> Any:
        last = self.executed[-1][0] if self.executed else ""
        if "current_database()" in last:
            return {"current_database": self.database_name}
        return None

    def fetchall(self) -> list[dict[str, Any]]:
        return list(self.rows)


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
