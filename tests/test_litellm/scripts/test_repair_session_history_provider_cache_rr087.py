"""RR-087: provider-cache repair guardrails and reconciliation."""


# RR-087 issue matrix (no residual/deferred exit):
# 1 Medium/operational: --target-db-name / current_database() guard on --apply and --ensure-schema
# 2 Medium/correctness: git counts trust complete tool_activity only (can lower; incomplete leaves stored)
# 3 Low/maintainability: no local unconditional ensure; shared aawm_agent_identity helper + opt-in bootstrap
# 4 Low/maintainability: metadata cache keys from single _PROVIDER_CACHE_METADATA_FIELDS catalog
# 5 Low/operational: schema DDL opt-in via --ensure-schema only (default no ACCESS EXCLUSIVE DDL)

from __future__ import annotations

import argparse
import importlib.util
import inspect
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, List, Optional

import pytest

_REPO = Path(__file__).resolve().parents[3]
_SCRIPT = _REPO / "scripts" / "repair_session_history_provider_cache.py"


def _load_module() -> ModuleType:
    module_name = "repair_session_history_provider_cache_rr087"
    if module_name in sys.modules:
        del sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def mod() -> ModuleType:
    return _load_module()


def _parse(mod: ModuleType, argv: Optional[List[str]] = None) -> argparse.Namespace:
    return mod._build_arg_parser().parse_args(argv or [])


def test_no_local_unconditional_ensure_schema_helper(mod: ModuleType) -> None:
    """RR-087 #3/#5: drop unconditional local ensure that always ran DDL."""
    assert not hasattr(mod, "_ensure_session_history_schema")
    assert hasattr(mod, "_shared_ensure_session_history_schema")
    assert callable(mod._shared_ensure_session_history_schema)
    assert inspect.iscoroutinefunction(mod._shared_ensure_session_history_schema)
    assert hasattr(mod, "_bootstrap_session_history_schema_from_shared_sql")
    assert callable(mod._bootstrap_session_history_schema_from_shared_sql)


def test_ensure_schema_is_opt_in_cli_flag(mod: ModuleType) -> None:
    default_args = _parse(mod, [])
    assert default_args.ensure_schema is False
    assert default_args.apply is False
    assert default_args.target_db_name == "aawm_tristore"

    opted_in = _parse(mod, ["--ensure-schema"])
    assert opted_in.ensure_schema is True

    source = Path(mod.__file__).read_text(encoding="utf-8")
    assert "--ensure-schema" in source
    assert "--target-db-name" in source
    assert "migration-owned" in source
    assert "ACCESS EXCLUSIVE" in source


def test_reconcile_git_counts_uses_tool_activity_only_when_complete(
    mod: ModuleType,
) -> None:
    """RR-087 #2: incomplete join must not force stored counts to max/0."""
    incomplete = {
        "git_commit_count": 5,
        "git_push_count": 2,
        "tool_git_commit_count": None,
        "tool_git_push_count": None,
        "has_tool_activity": False,
    }
    assert mod._reconcile_git_counts(incomplete) == (5, 2)

    # COALESCE-style zeros without has_tool_activity still incomplete if nulls
    # already handled above; explicit zero tool evidence with complete join
    # must lower over-counted stored values.
    complete_lower = {
        "git_commit_count": 5,
        "git_push_count": 2,
        "tool_git_commit_count": 1,
        "tool_git_push_count": 0,
        "has_tool_activity": True,
    }
    assert mod._reconcile_git_counts(complete_lower) == (1, 0)

    complete_raise = {
        "git_commit_count": 1,
        "git_push_count": 0,
        "tool_git_commit_count": 4,
        "tool_git_push_count": 3,
        "has_tool_activity": True,
    }
    assert mod._reconcile_git_counts(complete_raise) == (4, 3)

    # No flag but non-null tool counts implies join produced a row.
    join_row_without_flag = {
        "git_commit_count": 9,
        "git_push_count": 9,
        "tool_git_commit_count": 2,
        "tool_git_push_count": 1,
    }
    assert mod._reconcile_git_counts(join_row_without_flag) == (2, 1)


def test_apply_cache_state_metadata_keys_from_single_source(mod: ModuleType) -> None:
    """RR-087 #4: generic and per-family keys come from one field catalog."""
    cache_state = {
        "attempted": True,
        "status": "miss",
        "miss": True,
        "miss_reason": "cache_write_only",
        "miss_token_count": 10,
        "miss_cost_usd": 0.01,
        "miss_cost_basis": "write_vs_read_delta",
        "source": "usage",
    }
    updated = mod._apply_cache_state_to_metadata(
        {"keep": 1},
        provider_family="anthropic",
        cache_state=cache_state,
    )
    assert updated["keep"] == 1
    for field in mod._PROVIDER_CACHE_METADATA_FIELDS:
        generic = f"usage_provider_cache_{field}"
        family = f"anthropic_provider_cache_{field}"
        assert generic in updated
        assert family in updated
        assert updated[generic] == updated[family] == cache_state[field]

    # None / empty values are removed, not stored as null placeholders.
    cleared = mod._apply_cache_state_to_metadata(
        updated,
        provider_family="anthropic",
        cache_state={
            "attempted": True,
            "status": "hit",
            "miss": False,
            "miss_reason": None,
            "miss_token_count": None,
            "miss_cost_usd": None,
            "miss_cost_basis": "",
            "source": None,
        },
    )
    assert cleared["usage_provider_cache_attempted"] is True
    assert cleared["anthropic_provider_cache_status"] == "hit"
    assert "usage_provider_cache_miss_reason" not in cleared
    assert "anthropic_provider_cache_miss_reason" not in cleared
    assert "usage_provider_cache_source" not in cleared

    pairs = mod._provider_cache_metadata_key_pairs(
        provider_family="openai",
        cache_state=cache_state,
    )
    # One generic + one family key per catalog field.
    assert len(pairs) == 2 * len(mod._PROVIDER_CACHE_METADATA_FIELDS)
    fields_from_pairs = {key.split("_provider_cache_", 1)[1] for key, _ in pairs}
    assert fields_from_pairs == set(mod._PROVIDER_CACHE_METADATA_FIELDS)


def test_run_repair_apply_aborts_on_target_db_mismatch(
    mod: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    """RR-087 #1: --apply requires exact current_database() match."""
    executed_sql: List[str] = []

    class FakeCursor:
        def __enter__(self) -> "FakeCursor":
            return self

        def __exit__(self, *args: Any) -> None:
            return None

        def execute(self, sql: str, params: Any = None) -> None:
            executed_sql.append(str(sql))

        def executemany(self, sql: str, params: Any) -> None:
            raise AssertionError("apply must not write when target db mismatches")

        def fetchone(self) -> Dict[str, Any]:
            return {"current_database": "xx_aawm_dev"}

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

    def fake_connect(*args: Any, **kwargs: Any) -> FakeConn:
        return FakeConn()

    monkeypatch.setattr(mod.psycopg, "connect", fake_connect)
    monkeypatch.setattr(
        mod, "_build_aawm_admin_dsn", lambda: "postgresql://example/xx_aawm_dev"
    )

    args = _parse(mod, ["--apply"])
    with pytest.raises(SystemExit, match="xx_aawm_dev"):
        mod._run_repair(args)

    assert any("current_database" in sql for sql in executed_sql)
    joined = "\n".join(executed_sql).upper()
    assert "CREATE TABLE" not in joined
    assert "ALTER TABLE" not in joined


def test_run_repair_skips_ddl_by_default(
    mod: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
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

    def boom_bootstrap(conn: Any) -> None:
        bootstrap_calls["n"] += 1
        raise AssertionError("bootstrap must not run without --ensure-schema")

    monkeypatch.setattr(mod.psycopg, "connect", lambda *a, **k: FakeConn())
    monkeypatch.setattr(
        mod, "_build_aawm_admin_dsn", lambda: "postgresql://example/aawm_tristore"
    )
    monkeypatch.setattr(
        mod, "_bootstrap_session_history_schema_from_shared_sql", boom_bootstrap
    )

    summary = mod._run_repair(_parse(mod, ["--limit", "1"]))
    assert summary["mode"] == "dry_run"
    assert summary["ensure_schema"] is False
    assert summary["schema_bootstrapped"] is False
    assert summary["database"] == "aawm_tristore"
    assert bootstrap_calls["n"] == 0
    joined = "\n".join(executed_sql).upper()
    assert "CREATE TABLE" not in joined
    assert "ALTER TABLE" not in joined


def test_run_repair_opt_in_bootstrap_uses_shared_sql_constants(
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

    def record_bootstrap(conn: Any) -> None:
        bootstrap_calls.append(conn)

    monkeypatch.setattr(mod.psycopg, "connect", lambda *a, **k: FakeConn())
    monkeypatch.setattr(
        mod, "_build_aawm_admin_dsn", lambda: "postgresql://example/aawm_tristore"
    )
    monkeypatch.setattr(
        mod, "_bootstrap_session_history_schema_from_shared_sql", record_bootstrap
    )

    summary = mod._run_repair(_parse(mod, ["--ensure-schema", "--limit", "1"]))
    assert summary["ensure_schema"] is True
    assert summary["schema_bootstrapped"] is True
    assert len(bootstrap_calls) == 1


def test_bootstrap_helper_executes_shared_constants_including_tool_activity(
    mod: ModuleType,
) -> None:
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

    expected = (
        2  # session_history + tool_activity CREATE TABLE
        + len(mod._AAWM_SESSION_HISTORY_ALTER_STATEMENTS)
        + len(mod._AAWM_SESSION_HISTORY_INDEX_STATEMENTS)
        + len(mod._AAWM_SESSION_HISTORY_TOOL_ACTIVITY_INDEX_STATEMENTS)
    )
    assert len(executed) == expected
    assert executed[0] == mod._AAWM_SESSION_HISTORY_TABLE_SQL.strip()
    assert executed[1] == mod._AAWM_SESSION_HISTORY_TOOL_ACTIVITY_TABLE_SQL.strip()
    joined = "\n".join(executed).lower()
    assert "session_history_tool_activity" in joined


def test_build_repair_row_update_lowers_overcounted_git_when_tool_activity_complete(
    mod: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        mod, "_normalize_session_history_provider", lambda *a, **k: "anthropic"
    )
    monkeypatch.setattr(mod, "_normalize_provider_cache_family", lambda *a, **k: None)

    row = {
        "id": 42,
        "provider": "anthropic",
        "model": "claude-sonnet-4",
        "input_tokens": 1,
        "output_tokens": 1,
        "total_tokens": 2,
        "cache_read_input_tokens": 0,
        "cache_creation_input_tokens": 0,
        "provider_cache_attempted": False,
        "provider_cache_status": None,
        "provider_cache_miss": False,
        "provider_cache_miss_reason": None,
        "provider_cache_miss_token_count": None,
        "provider_cache_miss_cost_usd": None,
        "response_cost_usd": None,
        "reasoning_tokens_reported": None,
        "reasoning_tokens_estimated": None,
        "reasoning_tokens_source": "not_applicable",
        "reasoning_present": False,
        "thinking_signature_present": False,
        "git_commit_count": 7,
        "git_push_count": 3,
        "tool_git_commit_count": 2,
        "tool_git_push_count": 1,
        "has_tool_activity": True,
        "metadata": {},
    }
    update = mod._build_repair_row_update(row)
    assert update is not None
    # update tuple: ... git_commit, git_push, metadata_json, id
    assert update[10] == 2
    assert update[11] == 1
    assert update[13] == 42
    assert isinstance(update[12], str)
    metadata = json.loads(update[12])
    assert metadata["usage_reasoning_tokens_source"] == "not_applicable"


def test_run_repair_dry_run_allows_mismatched_db_without_writes(
    mod: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    class FakeCursor:
        def __enter__(self) -> "FakeCursor":
            return self

        def __exit__(self, *args: Any) -> None:
            return None

        def execute(self, sql: str, params: Any = None) -> None:
            return None

        def executemany(self, sql: str, params: Any) -> None:
            raise AssertionError("dry-run must not write")

        def fetchone(self) -> Dict[str, Any]:
            return {"current_database": "xx_aawm_dev"}

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
            raise AssertionError("dry-run must not commit writes")

    monkeypatch.setattr(mod.psycopg, "connect", lambda *a, **k: FakeConn())
    monkeypatch.setattr(
        mod, "_build_aawm_admin_dsn", lambda: "postgresql://example/xx_aawm_dev"
    )

    summary = mod._run_repair(_parse(mod, ["--limit", "1"]))
    assert summary["mode"] == "dry_run"
    assert summary["database"] == "xx_aawm_dev"
    assert summary["repaired_rows"] == 0


def test_run_repair_ensure_schema_aborts_on_target_db_mismatch(
    mod: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    """RR-087 #1/#5: --ensure-schema is also a mutation path requiring target DB."""
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
            raise AssertionError("must not write when target db mismatches")

        def fetchone(self) -> Dict[str, Any]:
            return {"current_database": "xx_aawm_dev"}

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
            raise AssertionError("must not commit when target db mismatches")

    def boom_bootstrap(conn: Any) -> None:
        bootstrap_calls["n"] += 1
        raise AssertionError("bootstrap must not run when target db mismatches")

    monkeypatch.setattr(mod.psycopg, "connect", lambda *a, **k: FakeConn())
    monkeypatch.setattr(
        mod, "_build_aawm_admin_dsn", lambda: "postgresql://example/xx_aawm_dev"
    )
    monkeypatch.setattr(
        mod, "_bootstrap_session_history_schema_from_shared_sql", boom_bootstrap
    )

    with pytest.raises(SystemExit, match="xx_aawm_dev"):
        mod._run_repair(_parse(mod, ["--ensure-schema"]))

    assert bootstrap_calls["n"] == 0
    assert any("current_database" in sql for sql in executed_sql)


def test_run_repair_rejects_non_positive_batch_size(
    mod: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
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

    monkeypatch.setattr(mod.psycopg, "connect", lambda *a, **k: FakeConn())
    monkeypatch.setattr(
        mod, "_build_aawm_admin_dsn", lambda: "postgresql://example/aawm_tristore"
    )

    with pytest.raises(SystemExit, match="batch-size"):
        mod._run_repair(_parse(mod, ["--batch-size", "0"]))


def test_run_repair_apply_commits_updates_when_target_matches(
    mod: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Bounded unit path: apply writes only after target-db match."""
    commits = {"n": 0}
    executemany_payloads: List[Any] = []
    batches = {"n": 0}

    class FakeCursor:
        def __enter__(self) -> "FakeCursor":
            return self

        def __exit__(self, *args: Any) -> None:
            return None

        def execute(self, sql: str, params: Any = None) -> None:
            return None

        def executemany(self, sql: str, params: Any) -> None:
            executemany_payloads.append(list(params))

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
            commits["n"] += 1

    row = {
        "id": 7,
        "provider": "anthropic",
        "model": "claude-sonnet-4",
        "input_tokens": 1,
        "output_tokens": 1,
        "total_tokens": 2,
        "cache_read_input_tokens": 0,
        "cache_creation_input_tokens": 0,
        "provider_cache_attempted": False,
        "provider_cache_status": None,
        "provider_cache_miss": False,
        "provider_cache_miss_reason": None,
        "provider_cache_miss_token_count": None,
        "provider_cache_miss_cost_usd": None,
        "response_cost_usd": None,
        "reasoning_tokens_reported": None,
        "reasoning_tokens_estimated": None,
        "reasoning_tokens_source": "not_applicable",
        "reasoning_present": False,
        "thinking_signature_present": False,
        "git_commit_count": 5,
        "git_push_count": 2,
        "tool_git_commit_count": 1,
        "tool_git_push_count": 0,
        "has_tool_activity": True,
        "metadata": {},
        "litellm_call_id": "call-1",
        "session_id": "sess-1",
    }

    def fake_fetch(conn: Any, *, cursor_id: int, args: Any) -> list[Dict[str, Any]]:
        batches["n"] += 1
        if batches["n"] == 1:
            return [row]
        return []

    monkeypatch.setattr(mod.psycopg, "connect", lambda *a, **k: FakeConn())
    monkeypatch.setattr(
        mod, "_build_aawm_admin_dsn", lambda: "postgresql://example/aawm_tristore"
    )
    monkeypatch.setattr(mod, "_fetch_repair_batch", fake_fetch)
    monkeypatch.setattr(
        mod, "_normalize_session_history_provider", lambda *a, **k: "anthropic"
    )
    monkeypatch.setattr(mod, "_normalize_provider_cache_family", lambda *a, **k: None)
    monkeypatch.setattr(mod, "_record_provider_status", lambda *a, **k: None)

    summary = mod._run_repair(_parse(mod, ["--apply", "--limit", "10"]))
    assert summary["mode"] == "apply"
    assert summary["database"] == "aawm_tristore"
    assert summary["repaired_rows"] == 1
    assert commits["n"] == 1
    assert len(executemany_payloads) == 1
    update = executemany_payloads[0][0]
    assert update[10] == 1
    assert update[11] == 0
    assert update[13] == 7


def test_reconcile_git_counts_incomplete_with_flag_false_and_zero_tools(
    mod: ModuleType,
) -> None:
    """has_tool_activity=False must leave stored counts even if tool zeros are present."""
    incomplete_zeros = {
        "git_commit_count": 4,
        "git_push_count": 1,
        "tool_git_commit_count": 0,
        "tool_git_push_count": 0,
        "has_tool_activity": False,
    }
    assert mod._reconcile_git_counts(incomplete_zeros) == (4, 1)


def test_acknowledge_shared_schema_policy_requires_shared_helper(
    mod: ModuleType,
) -> None:
    """RR-087 #3: script stays coupled to shared migration-owned ensure helper."""
    mod._acknowledge_shared_schema_policy()
    assert callable(mod._shared_ensure_session_history_schema)


def test_issue_matrix_cli_and_helpers_cover_all_five(mod: ModuleType) -> None:
    """Final RR-087 audit: each numbered issue has a concrete code surface."""
    import inspect

    source = Path(mod.__file__).read_text(encoding="utf-8")
    # #1 target-db guard
    assert "target_db_name" in source
    assert "current_database" in source
    assert "Refusing to run against" in source
    # #2 git reconcile without max(); complete tool_activity only
    reconcile_src = inspect.getsource(mod._reconcile_git_counts)
    assert "max(" not in reconcile_src
    assert "_row_has_complete_tool_activity" in reconcile_src
    # #3 shared ensure, no local _ensure_session_history_schema
    assert "_shared_ensure_session_history_schema" in source
    assert "def _ensure_session_history_schema" not in source
    # #4 single field catalog for metadata keys
    assert "_PROVIDER_CACHE_METADATA_FIELDS" in source
    assert "_provider_cache_metadata_key_pairs" in source
    # #5 opt-in ensure-schema only
    assert "--ensure-schema" in source
    assert "migration-owned" in source
    assert "ACCESS EXCLUSIVE" in source
    default_args = mod._build_arg_parser().parse_args([])
    assert default_args.ensure_schema is False
    assert default_args.apply is False
    assert default_args.target_db_name == "aawm_tristore"
