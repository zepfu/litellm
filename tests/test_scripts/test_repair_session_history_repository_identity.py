from scripts import repair_session_history_repository_identity as repair


def test_should_repair_rollout_memory_filename_from_memory_registry(tmp_path) -> None:
    memories_dir = tmp_path / "memories"
    summaries_dir = memories_dir / "rollout_summaries"
    summaries_dir.mkdir(parents=True)
    (memories_dir / "MEMORY.md").write_text(
        "- rollout_summaries/example.md "
        "(cwd=/home/zepfu/projects/litellm, "
        "rollout_path=/home/zepfu/.codex/sessions/2026/05/21/"
        "rollout-2026-05-21T19-34-36-019e4ce4-136c-78f2-bf86-0e3f7a0d95db.jsonl, "
        "thread_id=019e4ce4-136c-78f2-bf86-0e3f7a0d95db)\n"
    )
    rollout_map = repair._load_rollout_repository_map(memories_dir, {"litellm"})

    repaired = repair._build_repaired_row(
        {
            "id": 50,
            "session_id": "memory-session",
            "provider": "openai",
            "model": "gpt-5.4",
            "repository": "rollout-2026-05-21T19-34-36-019e4ce4-136c-78f2-bf86-0e3f7a0d95db.json (memory)",
            "tenant_id": "rollout-2026-05-21T19-34-36-019e4ce4-136c-78f2-bf86-0e3f7a0d95db.json (memory)",
            "metadata": {
                "repository": "rollout-2026-05-21T19-34-36-019e4ce4-136c-78f2-bf86-0e3f7a0d95db.json (memory)",
                "source_repository": "rollout-2026-05-21T19-34-36-019e4ce4-136c-78f2-bf86-0e3f7a0d95db.json",
                "tenant_id": "rollout-2026-05-21T19-34-36-019e4ce4-136c-78f2-bf86-0e3f7a0d95db.json (memory)",
                "trace_user_id": "rollout-2026-05-21T19-34-36-019e4ce4-136c-78f2-bf86-0e3f7a0d95db.json (memory)",
            },
        },
        {"litellm"},
        {},
        None,
        rollout_map,
    )

    assert repaired is not None
    assert repaired["repository"] == "litellm (memory)"
    assert repaired["tenant_id"] == "litellm (memory)"
    assert repaired["repair_source"] == "rollout_memory_registry"
    assert repaired["metadata"]["repository"] == "litellm (memory)"
    assert repaired["metadata"]["source_repository"] == "litellm"
    assert repaired["metadata"]["trace_user_id"] == "litellm (memory)"


def test_should_null_placeholder_repository_without_guessing() -> None:
    repaired = repair._build_repaired_row(
        {
            "id": 51,
            "session_id": "placeholder-session",
            "provider": "openai",
            "model": "gpt-5.5",
            "repository": "...",
            "tenant_id": "...",
            "metadata": {
                "repository": "...",
                "tenant_id": "...",
                "trace_user_id": "...",
            },
        },
        {"litellm"},
        {},
    )

    assert repaired is not None
    assert repaired["repository"] is None
    assert repaired["tenant_id"] is None
    assert "repository" not in repaired["metadata"]
    assert "tenant_id" not in repaired["metadata"]
    assert "trace_user_id" not in repaired["metadata"]


def test_should_null_non_project_memory_fragment_without_guessing() -> None:
    repaired = repair._build_repaired_row(
        {
            "id": 52,
            "session_id": "memory-fragment-session",
            "provider": "openai",
            "model": "gpt-5.4",
            "repository": "memories (memory)",
            "tenant_id": "memories (memory)",
            "metadata": {
                "repository": "memories (memory)",
                "source_repository": "memories",
                "tenant_id": "memories (memory)",
                "trace_user_id": "memories (memory)",
            },
        },
        {"litellm", "ts-testable"},
        {},
    )

    assert repaired is not None
    assert repaired["repository"] is None
    assert repaired["tenant_id"] is None
    assert "repository" not in repaired["metadata"]
    assert "source_repository" not in repaired["metadata"]
    assert "tenant_id" not in repaired["metadata"]
    assert "trace_user_id" not in repaired["metadata"]


def test_should_not_override_valid_memory_repository_from_rollout_metadata(
    tmp_path,
) -> None:
    memories_dir = tmp_path / "memories"
    memories_dir.mkdir()
    (memories_dir / "MEMORY.md").write_text(
        "- rollout_summaries/example.md "
        "(cwd=/home/zepfu/projects/litellm, "
        "rollout_path=/home/zepfu/.codex/sessions/2026/05/21/"
        "rollout-2026-05-21T19-34-36-019e4ce4-136c-78f2-bf86-0e3f7a0d95db.jsonl, "
        "thread_id=019e4ce4-136c-78f2-bf86-0e3f7a0d95db)\n"
    )
    rollout_map = repair._load_rollout_repository_map(
        memories_dir,
        {"aawm-tap", "litellm"},
    )

    repaired = repair._build_repaired_row(
        {
            "id": 53,
            "session_id": "memory-session",
            "provider": "openai",
            "model": "gpt-5.4",
            "repository": "aawm-tap (memory)",
            "tenant_id": "aawm-tap (memory)",
            "metadata": {
                "repository": "aawm-tap (memory)",
                "source_repository": "rollout-2026-05-21T19-34-36-019e4ce4-136c-78f2-bf86-0e3f7a0d95db.json",
                "tenant_id": "aawm-tap (memory)",
            },
        },
        {"aawm-tap", "litellm"},
        {},
        None,
        rollout_map,
    )

    assert repaired is not None
    assert repaired["repository"] == "aawm-tap (memory)"
    assert repaired["tenant_id"] == "aawm-tap (memory)"
    assert repaired["metadata"]["source_repository"] == "aawm-tap"


def test_should_repair_agent_repository_from_same_session_project() -> None:
    known_repositories = {"dashboard-shell", "litellm"}
    session_repositories = {
        "session-1": {
            "repository": "dashboard-shell",
            "source": "session_metadata.aawm_claude_project",
            "priority": 0,
            "source_row_id": 10,
        }
    }

    repaired = repair._build_repaired_row(
        {
            "id": 11,
            "session_id": "session-1",
            "provider": "anthropic",
            "model": "claude-opus-4-7",
            "repository": "agent-ac357ffbc895e51d4",
            "tenant_id": "agent-ac357ffbc895e51d4",
            "metadata": {
                "repository": "agent-ac357ffbc895e51d4",
                "tenant_id": "agent-ac357ffbc895e51d4",
            },
        },
        known_repositories,
        session_repositories,
    )

    assert repaired is not None
    assert repaired["repository"] == "dashboard-shell"
    assert repaired["tenant_id"] == "dashboard-shell"
    assert repaired["repair_source"] == "same_session.session_metadata.aawm_claude_project"
    assert repaired["metadata"]["repository_identity_previous_repository"] == (
        "agent-ac357ffbc895e51d4"
    )


def test_should_build_session_repository_map_from_known_tenant() -> None:
    session_repositories = repair._build_session_repository_map(
        [
            {
                "id": 20,
                "session_id": "session-2",
                "repository": None,
                "tenant_id": "aawm-tap-dashboard",
                "metadata": {},
            }
        ],
        {"aawm-tap-dashboard"},
    )

    repaired = repair._build_repaired_row(
        {
            "id": 21,
            "session_id": "session-2",
            "provider": "anthropic",
            "model": "claude-opus-4-7",
            "repository": "path",
            "tenant_id": "project",
            "metadata": {"repository": "path", "tenant_id": "project"},
        },
        {"aawm-tap-dashboard"},
        session_repositories,
    )

    assert repaired is not None
    assert repaired["repository"] == "aawm-tap-dashboard"
    assert repaired["tenant_id"] == "aawm-tap-dashboard"


def test_should_exclude_mixed_sessions_from_unique_repository_map() -> None:
    session_repositories = repair._build_unique_session_repository_map(
        [
            {
                "id": 20,
                "session_id": "mixed-session",
                "repository": "litellm",
                "tenant_id": "litellm",
                "metadata": {"repository": "litellm"},
            },
            {
                "id": 21,
                "session_id": "mixed-session",
                "repository": "aawm-tap",
                "tenant_id": "aawm-tap",
                "metadata": {"repository": "aawm-tap"},
            },
            {
                "id": 22,
                "session_id": "single-session",
                "repository": None,
                "tenant_id": "pytest-testable",
                "metadata": {"tenant_id": "pytest-testable"},
            },
        ],
        {"aawm-tap", "litellm", "pytest-testable"},
    )

    assert "mixed-session" not in session_repositories
    assert session_repositories["single-session"]["repository"] == "pytest-testable"


def test_should_not_repair_null_repository_from_mixed_session() -> None:
    session_repositories = repair._build_unique_session_repository_map(
        [
            {
                "id": 20,
                "session_id": "mixed-session",
                "repository": "litellm",
                "tenant_id": "litellm",
                "metadata": {"repository": "litellm"},
            },
            {
                "id": 21,
                "session_id": "mixed-session",
                "repository": "aawm-tap",
                "tenant_id": "aawm-tap",
                "metadata": {"repository": "aawm-tap"},
            },
        ],
        {"aawm-tap", "litellm"},
    )

    repaired = repair._build_repaired_row(
        {
            "id": 22,
            "session_id": "mixed-session",
            "provider": "gemini",
            "model": "gemini-3-flash-preview",
            "repository": None,
            "tenant_id": None,
            "metadata": {"codex_auto_agent_alias": "aawm-codex-agent-auto"},
        },
        {"aawm-tap", "litellm"},
        session_repositories,
    )

    assert repaired is None


def test_should_stamp_grok_rows_with_explicit_repository_override() -> None:
    repaired = repair._build_repaired_row(
        {
            "id": 30,
            "session_id": "grok-session",
            "provider": "xai",
            "model": "grok-build",
            "repository": "aawm-tap-dashboard",
            "tenant_id": None,
            "metadata": {"repository": "aawm-tap-dashboard"},
        },
        {"aawm-tap", "aawm-tap-dashboard"},
        {},
        "aawm-tap",
    )

    assert repaired is not None
    assert repaired["repository"] == "aawm-tap"
    assert repaired["tenant_id"] == "aawm-tap"
    assert repaired["repair_source"] == "grok_repository_override"
    assert repaired["metadata"]["repository"] == "aawm-tap"


def test_should_not_guess_grok_repository_without_override_or_evidence() -> None:
    repaired = repair._build_repaired_row(
        {
            "id": 31,
            "session_id": "grok-session",
            "provider": "xai",
            "model": "grok-build",
            "repository": None,
            "tenant_id": None,
            "metadata": {},
        },
        {"aawm-tap"},
        {},
    )

    assert repaired is None


def test_should_preserve_good_repository_and_tenant() -> None:
    repaired = repair._build_repaired_row(
        {
            "id": 40,
            "session_id": "session-4",
            "provider": "anthropic",
            "model": "claude-opus-4-7",
            "repository": "dashboard-shell",
            "tenant_id": "dashboard-shell",
            "metadata": {"repository": "dashboard-shell", "tenant_id": "dashboard-shell"},
        },
        {"dashboard-shell"},
        {},
    )

    assert repaired is None
