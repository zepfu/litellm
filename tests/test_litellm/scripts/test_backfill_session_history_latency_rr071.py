"""RR-071: latency backfill must not own unconditional schema DDL."""

from __future__ import annotations

import importlib.util
import inspect
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, List

import pytest

_REPO = Path(__file__).resolve().parents[3]
_SCRIPT = _REPO / "scripts" / "backfill_session_history_latency.py"


def _load_module() -> ModuleType:
    module_name = "backfill_session_history_latency_rr071"
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def mod() -> ModuleType:
    return _load_module()


def test_no_local_unconditional_ensure_schema_helper(mod: ModuleType) -> None:
    """RR-071 #1: drop the local copy-pasted ensure that always ran DDL."""
    assert not hasattr(mod, "_ensure_session_history_schema")
    # Shared policy entrypoint is imported (migration-owned readiness helper).
    assert hasattr(mod, "_shared_ensure_session_history_schema")
    assert callable(mod._shared_ensure_session_history_schema)
    assert inspect.iscoroutinefunction(mod._shared_ensure_session_history_schema)


def test_ensure_schema_is_opt_in_cli_flag(mod: ModuleType) -> None:
    default_args = mod._parse_args([])
    assert default_args.ensure_schema is False
    assert default_args.apply is False

    opted_in = mod._parse_args(["--ensure-schema"])
    assert opted_in.ensure_schema is True

    source = Path(mod.__file__).read_text(encoding="utf-8")
    assert "--ensure-schema" in source
    assert "migration-owned" in source
    assert "ACCESS EXCLUSIVE" in source


def test_run_backfill_skips_ddl_by_default(mod: ModuleType, monkeypatch: pytest.MonkeyPatch) -> None:
    executed_sql: List[str] = []
    bootstrap_calls = {"n": 0}

    class FakeCursor:
        def __enter__(self) -> "FakeCursor":
            return self

        def __exit__(self, *args: Any) -> None:
            return None

        def execute(self, sql: str, params: Any = None) -> None:
            executed_sql.append(str(sql))

        def executemany(self, sql: str, params: Any) -> None:
            executed_sql.append(f"executemany:{sql}")

        def fetchone(self) -> Dict[str, Any]:
            return {"current_database": "aawm_tristore"}

        def fetchall(self) -> list[Any]:
            return []

    class FakeConn:
        def __enter__(self) -> "FakeConn":
            return self

        def __exit__(self, *args: Any) -> None:
            return None

        def cursor(self) -> FakeCursor:
            return FakeCursor()

        def commit(self) -> None:
            return None

        def rollback(self) -> None:
            return None

    def fake_connect(*args: Any, **kwargs: Any) -> FakeConn:
        return FakeConn()

    def boom_bootstrap(conn: Any) -> None:
        bootstrap_calls["n"] += 1
        raise AssertionError("bootstrap must not run without --ensure-schema")

    monkeypatch.setattr(mod.psycopg, "connect", fake_connect)
    monkeypatch.setattr(mod, "_build_aawm_admin_dsn", lambda: "postgresql://example/aawm_tristore")
    monkeypatch.setattr(mod, "_bootstrap_session_history_schema_from_shared_sql", boom_bootstrap)

    args = mod._parse_args(["--limit", "1"])
    summary = mod._run_backfill(args)

    assert summary["ensure_schema"] is False
    assert summary["schema_bootstrapped"] is False
    assert bootstrap_calls["n"] == 0
    # Only the database guard SELECT should run, not CREATE/ALTER DDL.
    assert any("current_database" in sql for sql in executed_sql)
    joined = "\n".join(executed_sql)
    assert "CREATE TABLE" not in joined.upper()
    assert "ALTER TABLE" not in joined.upper()


def test_run_backfill_opt_in_bootstrap_uses_shared_sql_constants(
    mod: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    bootstrap_calls: List[Any] = []

    class FakeCursor:
        def __enter__(self) -> "FakeCursor":
            return self

        def __exit__(self, *args: Any) -> None:
            return None

        def execute(self, sql: str, params: Any = None) -> None:
            return None

        def fetchone(self) -> Dict[str, Any]:
            return {"current_database": "aawm_tristore"}

        def fetchall(self) -> list[Any]:
            return []

    class FakeConn:
        def __enter__(self) -> "FakeConn":
            return self

        def __exit__(self, *args: Any) -> None:
            return None

        def cursor(self) -> FakeCursor:
            return FakeCursor()

        def commit(self) -> None:
            return None

        def rollback(self) -> None:
            return None

    def fake_connect(*args: Any, **kwargs: Any) -> FakeConn:
        return FakeConn()

    def record_bootstrap(conn: Any) -> None:
        bootstrap_calls.append(conn)

    monkeypatch.setattr(mod.psycopg, "connect", fake_connect)
    monkeypatch.setattr(mod, "_build_aawm_admin_dsn", lambda: "postgresql://example/aawm_tristore")
    monkeypatch.setattr(mod, "_bootstrap_session_history_schema_from_shared_sql", record_bootstrap)

    args = mod._parse_args(["--ensure-schema", "--limit", "1"])
    summary = mod._run_backfill(args)

    assert summary["ensure_schema"] is True
    assert summary["schema_bootstrapped"] is True
    assert len(bootstrap_calls) == 1


def test_bootstrap_helper_executes_only_shared_constants(
    mod: ModuleType,
) -> None:
    """Opt-in bootstrap must use aawm_agent_identity SQL, not a local catalog."""
    executed: List[str] = []

    class FakeCursor:
        def __enter__(self) -> "FakeCursor":
            return self

        def __exit__(self, *args: Any) -> None:
            return None

        def execute(self, sql: str, params: Any = None) -> None:
            executed.append(str(sql).strip())

    class FakeConn:
        def cursor(self) -> FakeCursor:
            return FakeCursor()

        def commit(self) -> None:
            return None

    mod._bootstrap_session_history_schema_from_shared_sql(FakeConn())

    # Table create + each alter + each index from the shared constants.
    expected = (
        1
        + len(mod._AAWM_SESSION_HISTORY_ALTER_STATEMENTS)
        + len(mod._AAWM_SESSION_HISTORY_INDEX_STATEMENTS)
    )
    assert len(executed) == expected
    assert executed[0] == mod._AAWM_SESSION_HISTORY_TABLE_SQL.strip()
    # Must not pull in tool_activity DDL (that belongs to other repair scripts).
    joined = "\n".join(executed).lower()
    assert "session_history_tool_activity" not in joined


def test_same_float_tolerance(mod: ModuleType) -> None:
    assert mod._same_float(None, None) is True
    assert mod._same_float(1.0, None) is False
    assert mod._same_float(1.0, 1.0001) is True
    assert mod._same_float(1.0, 1.01) is False
