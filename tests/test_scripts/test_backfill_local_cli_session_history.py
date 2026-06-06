from __future__ import annotations

import json
import sqlite3
from datetime import date
from pathlib import Path

import scripts.backfill_local_cli_session_history as backfill


def test_should_prefer_direct_dsn_when_no_pg_component_override(monkeypatch) -> None:
    args = backfill.argparse.Namespace(
        pg_dsn=None,
        pg_host=None,
        pg_port=None,
        pg_user=None,
        pg_password=None,
        target_db_name="aawm_tristore",
    )
    monkeypatch.setenv(
        "AAWM_DIRECT_DATABASE_URL",
        "postgresql://aawm:aawm_dev@127.0.0.1:5434/aawm_tristore",
    )
    monkeypatch.setenv("AAWM_DB_HOST", "127.0.0.1")
    monkeypatch.setenv("AAWM_DB_PORT", "6432")

    assert backfill._postgres_dsn_from_args(args) == (
        "postgresql://aawm:aawm_dev@127.0.0.1:5434/aawm_tristore"
    )


def test_should_prefer_pg_component_override_over_direct_dsn(monkeypatch) -> None:
    args = backfill.argparse.Namespace(
        pg_dsn=None,
        pg_host="127.0.0.1",
        pg_port="6432",
        pg_user="aawm",
        pg_password="aawm_dev",
        target_db_name="aawm_tristore",
    )
    monkeypatch.setenv(
        "AAWM_DIRECT_DATABASE_URL",
        "postgresql://aawm:aawm_dev@127.0.0.1:5434/aawm_tristore",
    )

    assert backfill._postgres_dsn_from_args(args) == (
        "postgresql://aawm:aawm_dev@127.0.0.1:6432/aawm_tristore"
    )


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )


def test_should_parse_claude_usage_and_tool_activity(tmp_path: Path) -> None:
    transcript = (
        tmp_path
        / ".claude"
        / "projects"
        / "-home-zepfu-projects-litellm"
        / "session.jsonl"
    )
    _write_jsonl(
        transcript,
        [
            {
                "type": "assistant",
                "timestamp": "2026-03-01T12:00:00.000Z",
                "sessionId": "claude-session",
                "requestId": "req-1",
                "uuid": "uuid-1",
                "cwd": "/home/zepfu/projects/litellm",
                "version": "2.1.119",
                "message": {
                    "id": "msg-1",
                    "model": "claude-opus-4-6",
                    "usage": {
                        "input_tokens": 10,
                        "output_tokens": 5,
                        "cache_read_input_tokens": 7,
                        "cache_creation_input_tokens": 11,
                    },
                    "content": [
                        {"type": "thinking", "thinking": "short reasoning"},
                        {
                            "type": "tool_use",
                            "id": "tool-1",
                            "name": "Read",
                            "input": {"file_path": "README.md"},
                        },
                    ],
                },
            }
        ],
    )
    stats = backfill.ScanStats()

    [record] = list(backfill._iter_claude_records(tmp_path, stats))

    assert record["client_name"] == "claude-code"
    assert record["client_version"] is None
    assert record["client_user_agent"] == "backfill"
    assert record["provider"] == "anthropic"
    assert record["model"] == "claude-opus-4-6"
    assert record["model_group"] == "claude-opus-4-6"
    assert record["input_tokens"] == 10
    assert record["output_tokens"] == 5
    assert record["cache_read_input_tokens"] == 7
    assert record["cache_creation_input_tokens"] == 11
    assert record["provider_cache_status"] == "hit"
    assert record["provider_cache_miss"] is False
    assert record["response_cost_usd"] is not None
    assert record["tool_call_count"] == 1
    assert record["file_read_count"] == 1
    assert record["changed_env_file"] is False
    assert record["changed_pyproject_toml"] is False
    assert record["reasoning_present"] is True


def test_should_flag_sensitive_config_changes_from_local_tool_activity() -> None:
    record = backfill._base_record(
        source_client_name="codex",
        created_at=backfill._parse_datetime("2026-03-01T12:00:00Z"),
        litellm_call_id="call-sensitive-config",
        session_id="session-sensitive-config",
        provider="openai",
        model="gpt-5.4-mini",
        repository="/repo",
        metadata={},
        tool_activity=[
            {
                "tool_index": 0,
                "tool_name": "apply_patch",
                "tool_kind": "modify",
                "file_paths_modified": [
                    ".pre-commit-config.yaml",
                    ".env.local",
                    "pyproject.toml",
                    "./.gitignore",
                ],
            }
        ],
    )

    assert record["changed_pre_commit_config"] is True
    assert record["changed_env_file"] is True
    assert record["changed_pyproject_toml"] is True
    assert record["changed_gitignore"] is True


def test_should_match_nested_sensitive_config_paths_and_ignore_lookalikes() -> None:
    flags = backfill._sensitive_config_change_flags_from_paths(
        [
            "services/api/.pre-commit-config.yml",
            "packages/backend/.env.production",
            "src/python/pyproject.toml",
            "worktrees/example/.gitignore",
            "notes.env.example.txt",
            "pyproject.toml.bak",
            "not-.gitignore",
            ".pre-commit-config.yaml.bak",
        ]
    )

    assert flags == {
        "changed_pre_commit_config": True,
        "changed_env_file": True,
        "changed_pyproject_toml": True,
        "changed_gitignore": True,
    }

    lookalike_flags = backfill._sensitive_config_change_flags_from_paths(
        [
            "notes.env.example.txt",
            "pyproject.toml.bak",
            "not-.gitignore",
            ".pre-commit-config.yaml.bak",
        ]
    )

    assert lookalike_flags == {
        "changed_pre_commit_config": False,
        "changed_env_file": False,
        "changed_pyproject_toml": False,
        "changed_gitignore": False,
    }


def test_should_coalesce_claude_split_rows_by_message_id(tmp_path: Path) -> None:
    transcript = (
        tmp_path
        / ".claude"
        / "projects"
        / "-home-zepfu-projects-litellm"
        / "session.jsonl"
    )
    usage = {
        "input_tokens": 1,
        "output_tokens": 354,
        "cache_read_input_tokens": 77225,
        "cache_creation_input_tokens": 605,
    }
    _write_jsonl(
        transcript,
        [
            {
                "type": "assistant",
                "timestamp": "2026-03-01T12:00:00.000Z",
                "sessionId": "claude-session",
                "requestId": "req-1",
                "uuid": "uuid-text",
                "cwd": "/home/zepfu/projects/litellm",
                "message": {
                    "id": "msg-1",
                    "model": "claude-opus-4-6",
                    "usage": usage,
                    "content": [{"type": "text", "text": "working"}],
                },
            },
            {
                "type": "assistant",
                "timestamp": "2026-03-01T12:00:01.000Z",
                "sessionId": "claude-session",
                "requestId": "req-1",
                "uuid": "uuid-tool",
                "cwd": "/home/zepfu/projects/litellm",
                "message": {
                    "id": "msg-1",
                    "model": "claude-opus-4-6",
                    "usage": usage,
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "tool-1",
                            "name": "Read",
                            "input": {"file_path": "README.md"},
                        }
                    ],
                },
            },
        ],
    )
    stats = backfill.ScanStats()

    [record] = list(backfill._iter_claude_records(tmp_path, stats))

    assert record["input_tokens"] == 1
    assert record["output_tokens"] == 354
    assert record["cache_read_input_tokens"] == 77225
    assert record["cache_creation_input_tokens"] == 605
    assert record["tool_call_count"] == 1
    assert record["metadata"]["coalesced_claude_rows"] == 2
    assert record["metadata"]["source_lines"] == [0, 1]


def test_should_parse_codex_token_count_and_attach_pending_tool(tmp_path: Path) -> None:
    codex_root = tmp_path / ".codex"
    rollout = (
        codex_root
        / "sessions"
        / "2026"
        / "03"
        / "01"
        / "rollout-2026-03-01T12-00-00-thread-1.jsonl"
    )
    _write_jsonl(
        rollout,
        [
            {"type": "session_meta", "timestamp": "2026-03-01T12:00:00.000Z", "payload": {"type": "session_meta"}},
            {
                "type": "response_item",
                "timestamp": "2026-03-01T12:00:01.000Z",
                "payload": {
                    "type": "function_call",
                    "name": "exec_command",
                    "call_id": "call-1",
                    "arguments": json.dumps(
                        {"cmd": "git commit -m test", "workdir": "/repo"}
                    ),
                },
            },
            {
                "type": "event_msg",
                "timestamp": "2026-03-01T12:00:02.000Z",
                "payload": {
                    "type": "token_count",
                    "info": {
                        "last_token_usage": {
                            "input_tokens": 100,
                            "cached_input_tokens": 40,
                            "output_tokens": 20,
                            "reasoning_output_tokens": 9,
                            "total_tokens": 120,
                        },
                        "total_token_usage": {
                            "input_tokens": 100,
                            "cached_input_tokens": 40,
                            "output_tokens": 20,
                            "reasoning_output_tokens": 9,
                            "total_tokens": 120,
                        },
                    },
                },
            },
        ],
    )
    db_path = codex_root / "state_5.sqlite"
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE threads (
            id TEXT PRIMARY KEY,
            rollout_path TEXT NOT NULL,
            created_at INTEGER NOT NULL,
            updated_at INTEGER NOT NULL,
            source TEXT NOT NULL,
            model_provider TEXT NOT NULL,
            cwd TEXT NOT NULL,
            title TEXT NOT NULL,
            sandbox_policy TEXT NOT NULL,
            approval_mode TEXT NOT NULL,
            tokens_used INTEGER NOT NULL,
            has_user_event INTEGER NOT NULL,
            archived INTEGER NOT NULL,
            cli_version TEXT NOT NULL,
            first_user_message TEXT NOT NULL,
            model TEXT
        )
        """
    )
    conn.execute(
        """
        INSERT INTO threads (
            id, rollout_path, created_at, updated_at, source, model_provider,
            cwd, title, sandbox_policy, approval_mode, tokens_used,
            has_user_event, archived, cli_version, first_user_message, model
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "thread-1",
            str(rollout),
            1772366400,
            1772366402,
            "codex",
            "openai",
            "/home/zepfu/projects/litellm",
            "test",
            "workspace-write",
            "on-request",
            120,
            1,
            0,
            "0.125.0",
            "hello",
            "gpt-5.5",
        ),
    )
    conn.commit()
    conn.close()
    stats = backfill.ScanStats()

    [record] = list(backfill._iter_codex_records(tmp_path, stats))

    assert record["client_name"] == "codex_tui"
    assert record["provider"] == "openai"
    assert record["input_tokens"] == 100
    assert record["cache_read_input_tokens"] == 40
    assert record["output_tokens"] == 20
    assert record["reasoning_tokens_reported"] == 9
    assert record["tool_call_count"] == 1
    assert record["git_commit_count"] == 1


def test_should_parse_gemini_jsonl_tokens_and_dedupe(tmp_path: Path) -> None:
    projects = tmp_path / ".gemini" / "projects.json"
    projects.parent.mkdir(parents=True)
    projects.write_text(
        json.dumps({"projects": {"/home/zepfu/projects/litellm": "litellm"}}),
        encoding="utf-8",
    )
    chat = tmp_path / ".gemini" / "tmp" / "litellm" / "chats" / "session.jsonl"
    row = {
        "id": "message-1",
        "timestamp": "2026-03-01T12:00:00.000Z",
        "type": "gemini",
        "model": "gemini-3-flash-preview",
        "tokens": {
            "input": 10,
            "output": 2,
            "cached": 3,
            "thoughts": 4,
            "total": 16,
        },
        "toolCalls": [
            {"id": "read-1", "name": "read_file", "args": {"file_path": "a.py"}}
        ],
    }
    _write_jsonl(
        chat,
        [
            {
                "sessionId": "gemini-session",
                "startTime": "2026-03-01T11:59:00.000Z",
                "projectHash": "hash",
            },
            row,
            row,
        ],
    )
    stats = backfill.ScanStats()

    [record] = list(backfill._iter_gemini_records(tmp_path, stats))

    assert record["client_name"] == "gemini_cli"
    assert record["repository"] == "/home/zepfu/projects/litellm"
    assert record["input_tokens"] == 10
    assert record["output_tokens"] == 2
    assert record["cache_read_input_tokens"] == 3
    assert record["reasoning_tokens_reported"] == 4
    assert record["total_tokens"] == 16
    assert record["tool_call_count"] == 1


def test_should_bucket_summary_by_day() -> None:
    summary = backfill.DryRunSummary()
    record = backfill._base_record(
        source_client_name="grok",
        created_at=backfill._parse_datetime("2026-03-01T12:00:00Z"),
        litellm_call_id="call",
        session_id="session",
        provider="xai",
        model="grok-build",
        repository="/repo",
        metadata={},
    )

    summary.add(record, date(2026, 3, 1))

    assert summary.records == 1
    assert summary.by_client["grok_build"]["no_token_records"] == 1


def test_should_mark_cache_creation_only_rows_as_cache_write_misses() -> None:
    record = backfill._base_record(
        source_client_name="claude",
        created_at=backfill._parse_datetime("2026-03-01T12:00:00Z"),
        litellm_call_id="call",
        session_id="session",
        provider="anthropic",
        model="claude-opus-4-6",
        repository="/repo",
        metadata={},
        input_tokens=1,
        output_tokens=2,
        cache_creation_input_tokens=10,
        usage_obj={"cache_creation_input_tokens": 10},
    )

    assert record["provider_cache_attempted"] is True
    assert record["provider_cache_status"] == "write"
    assert record["provider_cache_miss"] is True
    assert record["provider_cache_miss_reason"] == "cache_write_only"
    assert record["provider_cache_miss_token_count"] == 10
    assert abs(record["provider_cache_miss_cost_usd"] - 0.0000575) < 0.000000001
    assert record["metadata"]["usage_provider_cache_miss_cost_basis"] == "write_vs_read_delta"
    assert record["response_cost_usd"] is not None


def test_history_payload_should_include_cache_miss_detail() -> None:
    record = backfill._base_record(
        source_client_name="claude",
        created_at=backfill._parse_datetime("2026-03-01T12:00:00Z"),
        litellm_call_id="call",
        session_id="session",
        provider="anthropic",
        model="claude-opus-4-6",
        repository="/repo",
        metadata={},
        cache_creation_input_tokens=10,
        usage_obj={"cache_creation_input_tokens": 10},
    )

    payload = backfill._history_payload(record)

    assert backfill.SESSION_HISTORY_INSERT_SQL.count("%s") == len(payload)
    assert len(payload) == 50
    assert payload[24] == "write"
    assert payload[25] is True
    assert payload[26] == "cache_write_only"
    assert payload[27] == 10
    assert abs(payload[28] - 0.0000575) < 0.000000001


def test_tool_payload_should_include_backfill_identity() -> None:
    record = backfill._base_record(
        source_client_name="claude",
        created_at=backfill._parse_datetime("2026-03-01T12:00:00Z"),
        litellm_call_id="call",
        session_id="session",
        provider="anthropic",
        model="claude-opus-4-6",
        repository="/repo",
        metadata={},
        tool_activity=[
            {
                "tool_index": 0,
                "tool_name": "Read",
                "tool_kind": "read",
                "metadata": {"source": "unit"},
            }
        ],
    )

    [payload] = backfill._tool_payloads(record)
    metadata = json.loads(payload[-1])

    assert metadata["source"] == "unit"
    assert metadata["source_import"] == backfill.IMPORT_MARKER
    assert metadata["parser_version"] == backfill.PARSER_VERSION
    assert metadata["source_client"] == "claude"
    assert metadata["client_name"] == "claude-code"
    assert metadata["client_user_agent"] == "backfill"
