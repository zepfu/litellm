"""Focused RR-090 coverage for score_agent_trace_quality maintainability/ops fixes."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import scripts.score_agent_trace_quality as scorer


def _write_jsonl(path: Path, events: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(event) + "\n" for event in events),
        encoding="utf-8",
    )


def _payload(messages: list[dict[str, object]], output: object) -> scorer.ObservationPayload:
    anthropic_request = {"model": "claude-sonnet-4-6", "messages": messages}
    langfuse_input = {
        "messages": [
            {
                "role": "user",
                "content": json.dumps(anthropic_request),
            }
        ]
    }
    return scorer.ObservationPayload(
        observation_id="obs-1",
        trace_id="trace-1",
        body={
            "id": "obs-1",
            "traceId": "trace-1",
            "input": json.dumps(langfuse_input),
            "output": json.dumps(output),
        },
        source="minio",
        source_locator="observation/obs-1/blob.json",
    )


def _candidate(**overrides: object) -> scorer.SessionCandidate:
    values: dict[str, object] = {
        "row_id": 1,
        "created_at": "2026-05-17T19:50:00Z",
        "trace_id": "trace-1",
        "session_id": "session-1",
        "litellm_call_id": "obs-1",
        "source_observation_id": "obs-1",
        "provider": "anthropic",
        "model": "claude-sonnet-4-6",
        "agent_name": "principal",
        "agent_id": None,
        "repository": "dashboard-shell",
        "tenant_id": "dashboard-shell",
        "input_tokens": 100,
        "output_tokens": 8,
        "tool_call_count": 1,
        "invalid_tool_call_count": 0,
        "llm_upstream_elapsed_ms": None,
        "total_server_elapsed_ms": None,
        "ttft_ms": None,
        "metadata": {},
    }
    values.update(overrides)
    return scorer.SessionCandidate(**values)  # type: ignore[arg-type]


def _init_ignored_path_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "ignored-path-repo"
    repo.mkdir()
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
    (repo / ".gitignore").write_text(".analysis/\n", encoding="utf-8")
    analysis_dir = repo / ".analysis"
    analysis_dir.mkdir()
    (analysis_dir / "todo.md").write_text("local planning\n", encoding="utf-8")
    subprocess.run(
        ["git", "add", "-f", ".analysis/todo.md"],
        cwd=repo,
        check=True,
        capture_output=True,
    )
    return repo


def test_should_not_own_session_history_score_ddl() -> None:
    assert not hasattr(scorer, "_SESSION_HISTORY_SCORE_ALTER_STATEMENTS")
    assert "agent_score_reasons" in scorer._SESSION_HISTORY_SCORE_REQUIRED_COLUMNS
    assert "trace_quality_score" in scorer._SESSION_HISTORY_SCORE_REQUIRED_COLUMNS


def test_should_fail_fast_when_score_schema_columns_missing() -> None:
    class FakeCursor:
        def __enter__(self) -> "FakeCursor":
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def execute(self, sql: str, params: dict[str, object] | None = None) -> None:
            self.sql = sql
            self.params = params

        def fetchall(self) -> list[tuple[str]]:
            return [("trace_quality_score",)]

    class FakeConnection:
        def cursor(self) -> FakeCursor:
            return FakeCursor()

    with pytest.raises(RuntimeError, match="missing columns"):
        scorer._verify_session_history_score_schema(FakeConnection())  # type: ignore[arg-type]


def test_should_pass_score_schema_verification_when_all_columns_present() -> None:
    required = list(scorer._SESSION_HISTORY_SCORE_REQUIRED_COLUMNS)

    class FakeCursor:
        def __enter__(self) -> "FakeCursor":
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def execute(self, sql: str, params: dict[str, object] | None = None) -> None:
            return None

        def fetchall(self) -> list[tuple[str]]:
            return [(column,) for column in required]

    class FakeConnection:
        def cursor(self) -> FakeCursor:
            return FakeCursor()

    scorer._verify_session_history_score_schema(FakeConnection())  # type: ignore[arg-type]


def test_should_verify_not_alter_when_ensure_schema_flag_set(monkeypatch) -> None:
    evidence = scorer.score_candidate(
        _candidate(invalid_tool_call_count=0),
        None,
        provider_error_present=False,
        max_output_tokens=5,
        large_base64_threshold=100_000,
    )
    executed_sql: list[str] = []
    verified: list[bool] = []

    class FakeCursor:
        rowcount = 1

        def __enter__(self) -> "FakeCursor":
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def execute(self, sql: str, params: dict[str, object] | None = None) -> None:
            executed_sql.append(sql)

        def executemany(self, sql: str, params_seq: list[dict[str, object]]) -> None:
            executed_sql.append(sql)
            self.rowcount = len(params_seq)

        def fetchone(self) -> tuple[str]:
            return ("aawm_tristore",)

    class FakeConnection:
        def __enter__(self) -> "FakeConnection":
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def cursor(self) -> FakeCursor:
            return FakeCursor()

        def commit(self) -> None:
            return None

    def fake_verify(conn: object) -> None:
        verified.append(True)

    monkeypatch.setattr(scorer.psycopg, "connect", lambda dsn: FakeConnection())
    monkeypatch.setattr(scorer, "_verify_session_history_score_schema", fake_verify)
    args = SimpleNamespace(
        pg_dsn="postgresql://example/aawm_tristore",
        pg_host=None,
        pg_port=None,
        pg_user=None,
        pg_password=None,
        target_db_name="aawm_tristore",
        require_target_database="aawm_tristore",
        ensure_session_history_score_schema=True,
    )

    assert scorer._update_session_history_scores(args, [evidence]) == 1
    assert verified == [True]
    assert all("ALTER TABLE" not in sql for sql in executed_sql)


def test_should_resolve_projects_root_from_env(monkeypatch, tmp_path: Path) -> None:
    projects = tmp_path / "workspaces"
    projects.mkdir()
    monkeypatch.setenv("AAWM_PROJECTS_ROOT", str(projects))

    assert scorer._aawm_projects_root() == projects.expanduser()
    marker = scorer._aawm_projects_root_marker()
    assert marker.endswith("/")
    assert str(projects).replace("\\", "/") in marker
    assert (
        scorer._repository_from_cwd(str(projects / "litellm" / "scripts")) == "litellm"
    )
    assert scorer._path_has_common_ignored_prefix(
        str(projects / "litellm" / ".analysis" / "todo.md")
    )
    # candidate repo root uses projects root / repo_name when cwd missing
    repo = projects / "demo-repo"
    repo.mkdir()
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
    root = scorer._candidate_repo_root(
        _candidate(repository="demo-repo", metadata={})
    )
    assert root is not None
    assert root.resolve() == repo.resolve()


def test_should_surface_git_check_failed_distinctly(monkeypatch, tmp_path: Path) -> None:
    repo = _init_ignored_path_repo(tmp_path)

    def boom(*_args: object, **_kwargs: object) -> None:
        raise TimeoutError("git timed out")

    monkeypatch.setattr(scorer.subprocess, "run", boom)
    payload = _payload(
        [
            {"role": "user", "content": "Update the planning notes."},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "name": "Bash",
                        "input": {"cmd": "git add -f .analysis/todo.md"},
                    },
                ],
            },
        ],
        {"role": "assistant", "content": "Done.", "tool_calls": None},
    )
    evidence = scorer.score_candidate(
        _candidate(
            output_tokens=8,
            repository=repo.name,
            metadata={"cwd": str(repo)},
        ),
        payload,
        provider_error_present=False,
        max_output_tokens=5,
        large_base64_threshold=100_000,
    )

    assert "git_check_failed" in evidence.errors
    # Git failure makes the git-backed policy check unevaluable rather than a
    # clean "not ignored" pass. Command-only heuristics may still report a
    # violation independently; that is intentional and separate from git failure.
    assert evidence.ignored_path_tracking_policy_score is None
    assert any(
        item.get("ignored_check") == "git_check_failed"
        for item in evidence.ignored_path_tracking_evidence
    )


def test_should_cache_parent_transcript_spawn_lookups(
    monkeypatch, tmp_path: Path
) -> None:
    parent = tmp_path / "parent.jsonl"
    child_a = tmp_path / "child-a.jsonl"
    child_b = tmp_path / "child-b.jsonl"
    _write_jsonl(
        parent,
        [
            {
                "type": "response_item",
                "payload": {
                    "type": "function_call",
                    "name": "spawn_agent",
                    "call_id": "call-a",
                    "arguments": json.dumps({"message": "Investigate A only."}),
                },
            },
            {
                "type": "response_item",
                "payload": {
                    "type": "function_call_output",
                    "call_id": "call-a",
                    "output": json.dumps({"agent_id": "child-a"}),
                },
            },
            {
                "type": "response_item",
                "payload": {
                    "type": "function_call",
                    "name": "spawn_agent",
                    "call_id": "call-b",
                    "arguments": json.dumps({"message": "Investigate B only."}),
                },
            },
            {
                "type": "response_item",
                "payload": {
                    "type": "function_call_output",
                    "call_id": "call-b",
                    "output": json.dumps({"agent_id": "child-b"}),
                },
            },
        ],
    )
    for path, session_id in ((child_a, "child-a"), (child_b, "child-b")):
        _write_jsonl(
            path,
            [
                {
                    "type": "session_meta",
                    "payload": {
                        "id": session_id,
                        "timestamp": "2026-05-27T01:04:21Z",
                        "cwd": "/home/zepfu/projects/litellm",
                        "originator": "codex-tui",
                        "model_provider": "litellm",
                    },
                },
                {
                    "type": "turn_context",
                    "payload": {"model": "aawm-codex-agent-auto"},
                },
                {
                    "type": "event_msg",
                    "payload": {
                        "type": "task_complete",
                        "completed_at": 1779843984,
                        "duration_ms": 10,
                        "last_agent_message": "done",
                    },
                },
            ],
        )

    parse_calls = {"count": 0}
    real = scorer._spawn_messages_by_agent_id

    def counted(path: Path) -> dict[str, str]:
        parse_calls["count"] += 1
        return real(path)

    monkeypatch.setattr(scorer, "_spawn_messages_by_agent_id", counted)
    args = SimpleNamespace(
        codex_transcript=[str(child_a), str(child_b)],
        codex_parent_transcript=[str(parent)],
    )
    bundles = scorer._resolve_codex_transcript_bundles(args)

    assert parse_calls["count"] == 1
    assert len(bundles) == 2
    assert bundles[0].payload is not None
    assert bundles[1].payload is not None
    # Parent spawn messages become the first user message in the derived payload.
    messages_a = scorer._extract_anthropic_request_from_langfuse_input(
        bundles[0].payload.body.get("input")
    ).get("messages")
    messages_b = scorer._extract_anthropic_request_from_langfuse_input(
        bundles[1].payload.body.get("input")
    ).get("messages")
    assert isinstance(messages_a, list) and messages_a
    assert isinstance(messages_b, list) and messages_b
    assert messages_a[0]["content"] == "Investigate A only."
    assert messages_b[0]["content"] == "Investigate B only."


def test_should_batch_ignored_path_git_checks(monkeypatch, tmp_path: Path) -> None:
    repo = _init_ignored_path_repo(tmp_path)
    git_commands: list[list[str]] = []
    real_run = subprocess.run

    def tracking_run(cmd: list[str] | tuple[str, ...], **kwargs: Any):  # type: ignore[no-untyped-def]
        command = list(cmd)
        git_commands.append(command)
        return real_run(command, **kwargs)

    monkeypatch.setattr(scorer.subprocess, "run", tracking_run)
    payload = _payload(
        [
            {"role": "user", "content": "Update the planning notes."},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "name": "Bash",
                        "input": {
                            "cmd": (
                                "git add -f .analysis/todo.md "
                                "&& git add -f .analysis/other.md"
                            )
                        },
                    },
                ],
            },
        ],
        {"role": "assistant", "content": "Done.", "tool_calls": None},
    )
    evidence = scorer.score_candidate(
        _candidate(
            output_tokens=8,
            repository=repo.name,
            metadata={"cwd": str(repo)},
        ),
        payload,
        provider_error_present=False,
        max_output_tokens=5,
        large_base64_threshold=100_000,
    )

    ls_files_calls = [
        cmd
        for cmd in git_commands
        if len(cmd) >= 4 and cmd[0] == "git" and "ls-files" in cmd
    ]
    assert ls_files_calls, "expected batched ls-files invocation"
    # One multi-pathspec ls-files call rather than one process per path.
    multi_path_calls = [
        cmd
        for cmd in ls_files_calls
        if cmd.count("--") >= 1 and len(cmd[cmd.index("--") + 1 :]) >= 2
    ]
    assert multi_path_calls, f"expected multi-pathspec ls-files, saw {ls_files_calls}"
    assert evidence.ignored_path_tracking_policy_score == 0.0
    assert evidence.ignored_path_tracking_violation_count >= 1
