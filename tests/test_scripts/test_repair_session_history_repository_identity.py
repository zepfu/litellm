from pathlib import Path

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
    assert repaired["repository"] == "litellm"
    assert repaired["tenant_id"] == "litellm"
    assert repaired["metadata"]["memory_workload_label"] == "litellm (memory)"
    assert repaired["repair_source"] == "rollout_memory_registry"
    assert repaired["metadata"]["repository"] == "litellm"
    assert repaired["metadata"]["source_repository"] == "litellm"
    assert repaired["metadata"].get("trace_user_id") in (None, "litellm")


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
    assert repaired["repository"] == "aawm-tap"
    assert repaired["tenant_id"] == "aawm-tap"
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


def test_should_repair_null_repository_from_referenced_artifact_owner() -> None:
    repaired = repair._build_repaired_row(
        {
            "id": 23,
            "session_id": "artifact-session",
            "provider": "openai",
            "model": "gpt-5.5",
            "repository": None,
            "tenant_id": None,
            "metadata": {
                "aawm_d1_452_original_repository": (
                    "proposal-focused-verification-profile-registry.md"
                ),
                "aawm_d1_452_referenced_artifact_owner": "aawm-devtools",
                "session_history_repository_status": "unresolved",
                "session_history_repository_unresolved": True,
            },
        },
        {"aawm-devtools", "litellm"},
        {},
    )

    assert repaired is not None
    assert repaired["repository"] == "aawm-devtools"
    assert repaired["tenant_id"] == "aawm-devtools"
    assert repaired["repair_source"] == "row_metadata.aawm_d1_452_referenced_artifact_owner"
    assert repaired["metadata"]["session_history_repository_status"] == "repaired"
    assert "session_history_repository_unresolved" not in repaired["metadata"]


def test_should_classify_unrepairable_file_like_null_repository() -> None:
    classified = repair._build_unresolved_classification_row(
        {
            "id": 24,
            "session_id": "file-like-session",
            "provider": "openai",
            "model": "gpt-5.5",
            "repository": None,
            "tenant_id": None,
            "metadata": {
                "aawm_d1_452_original_repository": "ci.yml",
                "tenant_id_source": "repository_untrusted",
            },
        },
        {"litellm"},
    )

    assert classified is not None
    assert classified["repository"] is None
    assert classified["tenant_id"] is None
    assert classified["classification_reason"] == (
        "untrusted_file_like_repository_candidate"
    )
    metadata = classified["metadata"]
    assert metadata["session_history_repository_status"] == "unresolved"
    assert metadata["session_history_repository_unresolved"] is True
    assert (
        metadata["session_history_repository_unresolved_reason"]
        == "untrusted_file_like_repository_candidate"
    )


def test_should_clear_untrusted_generic_repository_and_tenant_literals() -> None:
    repaired = repair._build_repaired_row(
        {
            "id": 26,
            "session_id": "generic-owner-session",
            "provider": "xai",
            "model": "grok-composer-2.5-fast",
            "repository": "zepfu",
            "tenant_id": "zepfu",
            "metadata": {
                "repository": "zepfu",
                "tenant_id": "zepfu",
                "tenant_id_source": "repository_untrusted",
            },
        },
        {"litellm"},
        {},
    )

    assert repaired is not None
    assert repaired["repository"] is None
    assert repaired["tenant_id"] is None
    metadata = repaired["metadata"]
    assert "repository" not in metadata
    assert "tenant_id" not in metadata
    assert "tenant_id_source" not in metadata
    assert metadata["session_history_repository_status"] == "unresolved"
    assert metadata["session_history_repository_unresolved"] is True
    assert metadata["session_history_repository_unresolved_reason"] == (
        "no_trusted_grok_project_signal"
    )


def test_should_repair_known_repository_untrusted_tenant_source() -> None:
    repaired = repair._build_repaired_row(
        {
            "id": 27,
            "session_id": "known-repo-session",
            "provider": "openai",
            "model": "gpt-5.5",
            "repository": "aawm-tap",
            "tenant_id": None,
            "metadata": {
                "repository": "aawm-tap",
                "tenant_id_source": "repository_untrusted",
                "repository_tenant_fallback_skipped": True,
                "session_history_repository_unresolved": True,
            },
        },
        {"aawm-tap", "litellm"},
        {},
    )

    assert repaired is not None
    assert repaired["repository"] == "aawm-tap"
    assert repaired["tenant_id"] == "aawm-tap"
    assert repaired["metadata"]["tenant_id_source"] == "repository_repair"
    assert repaired["metadata"]["session_history_repository_status"] == "repaired"
    assert "repository_tenant_fallback_skipped" not in repaired["metadata"]
    assert "session_history_repository_unresolved" not in repaired["metadata"]


def test_should_repair_trace_user_untrusted_known_repository() -> None:
    repaired = repair._build_repaired_row(
        {
            "id": 28,
            "session_id": "trace-untrusted-session",
            "provider": "openai",
            "model": "gpt-5.5",
            "repository": "litellm",
            "tenant_id": None,
            "metadata": {
                "repository": "litellm",
                "tenant_id_source": "trace_user_untrusted",
                "trace_user_tenant_fallback_skipped": True,
            },
        },
        {"litellm"},
        {},
    )

    assert repaired is not None
    assert repaired["repository"] == "litellm"
    assert repaired["tenant_id"] == "litellm"
    assert repaired["repair_source"] == "known_repository_untrusted_tenant_repair"
    assert repaired["metadata"]["tenant_id_source"] == "repository_repair"
    assert repaired["metadata"]["session_history_repository_status"] == "repaired"


def test_should_not_classify_reporting_excluded_null_repository() -> None:
    classified = repair._build_unresolved_classification_row(
        {
            "id": 25,
            "session_id": "excluded-grok-session",
            "provider": "xai",
            "model": "grok-build",
            "repository": None,
            "tenant_id": None,
            "metadata": {"session_history_reporting_excluded": True},
        },
        {"litellm"},
    )

    assert classified is None


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


class _FakeRepairCursor:
    def __init__(self) -> None:
        self.execute_calls: list[tuple[str, tuple | None]] = []
        self._rows: list[dict] = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def execute(self, statement, params=None) -> None:
        self.execute_calls.append((statement, params))

    def fetchall(self):
        return self._rows


class _FakeRepairConnection:
    def __init__(self) -> None:
        self.cursor_instance = _FakeRepairCursor()

    def cursor(self, *, row_factory=None):
        return self.cursor_instance


def test_should_include_max_id_in_repository_value_candidate_fetch() -> None:
    conn = _FakeRepairConnection()
    repair._fetch_candidate_rows(
        conn,
        cursor_id=10,
        batch_size=25,
        repository_values=["litellm", "aawm-tap"],
        max_id=500,
    )
    assert len(conn.cursor_instance.execute_calls) == 1
    statement, params = conn.cursor_instance.execute_calls[0]
    assert "id <= %s" in statement
    assert "repository = ANY(%s::text[])" in statement
    assert params == (10, 500, ["litellm", "aawm-tap"], ["litellm", "aawm-tap"], 25)


def test_should_include_max_id_in_null_repository_since_candidate_fetch() -> None:
    conn = _FakeRepairConnection()
    repair._fetch_candidate_rows(
        conn,
        cursor_id=0,
        batch_size=100,
        null_repository_since="2026-01-01T00:00:00+00:00",
        max_id=999,
    )
    statement, params = conn.cursor_instance.execute_calls[0]
    assert "repository IS NULL" in statement
    assert "created_at >= %s::timestamptz" in statement
    assert "id <= %s" in statement
    assert params == (0, 999, "2026-01-01T00:00:00+00:00", 100)


def test_should_omit_max_id_predicate_when_not_provided() -> None:
    conn = _FakeRepairConnection()
    repair._fetch_candidate_rows(
        conn,
        cursor_id=5,
        batch_size=10,
        repository_values=["zepfu"],
    )
    statement, params = conn.cursor_instance.execute_calls[0]
    assert "id <= %s" not in statement
    assert params[0] == 5
    assert params[-1] == 10
    assert 999 not in params


def test_should_include_max_id_in_default_candidate_fetch() -> None:
    conn = _FakeRepairConnection()
    repair._fetch_candidate_rows(
        conn,
        cursor_id=1,
        batch_size=50,
        max_id=42,
    )
    statement, params = conn.cursor_instance.execute_calls[0]
    assert "id <= %s" in statement
    assert params[0] == 1
    assert 42 in params
    assert params[-1] == 50
    assert "metadata->>'tenant_id_source'" in statement
    assert "repository_untrusted" in statement
    assert "trace_user_untrusted" in statement
    assert "repository LIKE '%% (memory)'" in statement
    assert "tenant_id LIKE '%% (memory)'" in statement

def test_should_default_projects_and_memories_dirs_to_home(monkeypatch) -> None:
    import sys

    monkeypatch.setattr(sys, "argv", ["repair_session_history_repository_identity.py"])
    parsed = repair._parse_args()
    assert parsed.projects_dir == str(Path.home() / "projects")
    assert parsed.memories_dir == str(Path.home() / ".codex" / "memories")
    assert parsed.target_db_name == "aawm_tristore"
    assert parsed.session_evidence_limit_per_session == 50
    # Defaults must be portable Path.home()-relative, not a hardcoded operator path.
    assert parsed.projects_dir == repair.DEFAULT_PROJECTS_DIR
    assert parsed.memories_dir == repair.DEFAULT_MEMORIES_DIR
    assert not parsed.projects_dir.startswith("/home/zepfu/") or Path.home() == Path(
        "/home/zepfu"
    )


def test_should_warn_when_projects_dir_missing(tmp_path) -> None:
    missing = tmp_path / "does-not-exist"
    with __import__("pytest").warns(UserWarning, match="does not exist"):
        known = repair._load_known_repositories(missing)
    assert known == {repair.REPO_ROOT.name}


def test_should_use_shared_priority_list_for_best_and_build() -> None:
    extractors = repair._row_repository_candidate_extractors(
        known_repositories={"litellm", "aawm-tap"},
        session_repositories={
            "s1": {
                "repository": "litellm",
                "source": "session_metadata.repository",
                "priority": 0,
            }
        },
        rollout_repository_map={"rollout.json": "litellm"},
        grok_repository="aawm-tap",
    )
    source_names = [name for name, _ in extractors]
    assert source_names[0] == "grok_repository_override"
    assert "rollout_memory_registry" in source_names
    assert "session_metadata.aawm_d1_452_referenced_artifact_owner" in source_names
    assert "same_session" in source_names
    assert "session_history.tenant_id" in source_names

    # Best-candidate ranking uses the same helper without repair-only overrides.
    best = repair._best_row_repository_candidate(
        {
            "id": 1,
            "session_id": "s1",
            "repository": None,
            "tenant_id": None,
            "metadata": {
                "aawm_d1_452_referenced_artifact_owner": "aawm-tap",
                "repository": "litellm",
            },
        },
        {"litellm", "aawm-tap"},
    )
    assert best is not None
    assert best[0] == "aawm-tap"
    assert best[1] == "session_metadata.aawm_d1_452_referenced_artifact_owner"


def test_should_preserve_rollout_memory_registry_priority_for_repeated_extractors() -> None:
    resolved = repair._resolve_repository_candidate(
        {
            "id": 10,
            "session_id": "session-1",
            "provider": "openai",
            "model": "gpt-5.5",
            "repository": "rollout-alpha.json",
            "tenant_id": "agent-ac357ffbc895e51d4",
            "metadata": {
                "repository": "rollout-beta.json",
                "source_repository": "rollout-gamma.json",
            },
        },
        {"litellm", "aawm-tap"},
        rollout_repository_map={
            "rollout-alpha.json": "litellm",
            "rollout-beta.json": "aawm-tap",
            "rollout-gamma.json": "dashboard-shell",
        },
    )
    assert resolved is not None
    assert resolved[0] == "litellm"
    assert resolved[1] == "rollout_memory_registry"
    assert resolved[2] == 0


def test_should_not_evaluate_following_rollout_extractors_after_first_hit(
    monkeypatch,
) -> None:
    calls: list[str] = []

    def fake_rollout_candidate(
        value: object,
        _rollout_map: dict[str, str],
    ) -> str | None:
        calls.append(str(value))
        if value == "rollout-alpha.json":
            return "litellm"
        raise AssertionError(f"unexpected rollout extractor call after first hit: {value!r}")

    monkeypatch.setattr(repair, "_rollout_repository_candidate", fake_rollout_candidate)

    resolved = repair._resolve_repository_candidate(
        {
            "id": 11,
            "session_id": "session-1",
            "provider": "openai",
            "model": "gpt-5.5",
            "repository": "rollout-alpha.json",
            "tenant_id": "agent-ac357ffbc895e51d4",
            "metadata": {
                "repository": "rollout-beta.json",
                "source_repository": "rollout-gamma.json",
            },
        },
        {"litellm", "aawm-tap"},
        rollout_repository_map={
            "rollout-alpha.json": "litellm",
            "rollout-beta.json": "aawm-tap",
            "rollout-gamma.json": "dashboard-shell",
        },
    )
    assert resolved is not None
    assert resolved[1] == "rollout_memory_registry"
    assert calls == ["rollout-alpha.json"]


def test_should_report_same_session_and_tenant_id_source_labels_through_selector() -> None:
    same_session = repair._resolve_repository_candidate(
        {
            "id": 12,
            "session_id": "session-1",
            "provider": "anthropic",
            "model": "claude-opus-4-7",
            "repository": "agent-unknown",
            "tenant_id": "agent-unknown",
            "metadata": {},
        },
        {"litellm", "aawm-tap"},
        session_repositories={
            "session-1": {
                "repository": "litellm",
                "source": "session_metadata.aawm_claude_project",
                "priority": 0,
                "source_row_id": 1,
            },
        },
    )
    assert same_session is not None
    assert same_session[1] == "same_session.session_metadata.aawm_claude_project"

    tenant_id = repair._resolve_repository_candidate(
        {
            "id": 13,
            "session_id": "session-2",
            "provider": "openai",
            "model": "gpt-5.5",
            "repository": "agent-unknown",
            "tenant_id": "aawm-tap",
            "metadata": {},
        },
        {"aawm-tap"},
    )
    assert tenant_id is not None
    assert tenant_id[0] == "aawm-tap"
    assert tenant_id[1] == "row_tenant_id"


def test_should_merge_only_owned_metadata_keys_on_apply() -> None:
    class _Cursor:
        def __init__(self) -> None:
            self.params = None

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def executemany(self, statement, params):
            self.statement = statement
            self.params = list(params)

    class _Conn:
        def __init__(self):
            self.cursor_instance = _Cursor()

        def cursor(self):
            return self.cursor_instance

    metadata = {
        "repository": "litellm",
        "tenant_id": "litellm",
        "tenant_id_source": "repository_repair",
        "session_history_repository_status": "repaired",
        "session_history_repository_status_source": "row_identity_normalization",
        "repository_identity_repaired_at": "2026-07-17T00:00:00+00:00",
        "repository_identity_repair_source": "row_identity_normalization",
        "unrelated_sibling_key": "must-not-be-written",
        "provider_cache_marker": "leave-me",
    }
    # Build repaired metadata without the keys that should be cleared.
    patch = repair._owned_metadata_patch(metadata)
    assert "unrelated_sibling_key" not in patch
    assert "provider_cache_marker" not in patch
    assert patch["repository"] == "litellm"

    clear_keys = repair._owned_metadata_null_clear_keys(
        {
            "repository": "litellm",
            "tenant_id": "litellm",
            "session_history_repository_status": "repaired",
            "session_history_repository_status_source": "row_identity_normalization",
            "repository_identity_repaired_at": "t",
            "repository_identity_repair_source": "row_identity_normalization",
            "tenant_id_source": "repository_repair",
        }
    )
    # Unresolved keys not present in the repair payload should be cleared.
    assert "session_history_repository_unresolved" in clear_keys
    assert "repository_tenant_fallback_skipped" in clear_keys
    # Clear list must be unique (no duplicated key deletes).
    assert len(repair._OWNED_METADATA_CLEAR_KEYS) == len(
        set(repair._OWNED_METADATA_CLEAR_KEYS)
    )
    # Intentional absence of optional owned keys (e.g. previous_*) is clearable,
    # but unrelated sibling keys are never listed for deletion.
    assert "repository_identity_previous_repository" in clear_keys
    assert "provider_cache_marker" not in clear_keys

    conn = _Conn()
    repair._apply_repairs(
        conn,
        [
            {
                "id": 9,
                "repository": "litellm",
                "tenant_id": "litellm",
                "metadata": {
                    "repository": "litellm",
                    "tenant_id": "litellm",
                    "tenant_id_source": "repository_repair",
                    "session_history_repository_status": "repaired",
                    "session_history_repository_status_source": "x",
                    "repository_identity_repaired_at": "t",
                    "repository_identity_repair_source": "x",
                },
            }
        ],
    )
    statement = conn.cursor_instance.statement
    assert "||" in statement
    assert "%s::jsonb" in statement
    assert "metadata = %s::jsonb" not in statement.replace(
        "|| COALESCE(%s::jsonb, '{}'::jsonb)", ""
    )
    params = conn.cursor_instance.params[0]
    assert params[0] == "litellm"
    assert params[1] == "litellm"
    assert isinstance(params[2], list)
    assert "session_history_repository_unresolved" in params[2]
    assert '"repository": "litellm"' in params[3]
    assert "unrelated_sibling_key" not in params[3]
    assert params[4] == 9


def test_should_bound_session_identity_fetch_per_session() -> None:
    conn = repair._FakeRepairConnection() if hasattr(repair, "_FakeRepairConnection") else None
    # Use the local fake classes from this module via duck typing recreation.
    class _Cursor:
        def __init__(self):
            self.execute_calls = []
            self._rows = []

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return None

        def execute(self, statement, params=None):
            self.execute_calls.append((statement, params))

        def fetchall(self):
            return self._rows

    class _Conn:
        def __init__(self):
            self.cursor_instance = _Cursor()

        def cursor(self, *, row_factory=None):
            return self.cursor_instance

    conn = _Conn()
    repair._fetch_session_identity_rows(
        conn,
        {"sess-a", "sess-b"},
        limit_per_session=7,
    )
    statement, params = conn.cursor_instance.execute_calls[0]
    assert "row_number()" in statement
    assert "PARTITION BY session_id" in statement
    assert "session_row_rank <=" in statement
    assert params[2] == 7
    assert set(params[0]) == {"sess-a", "sess-b"}


def test_should_refuse_apply_when_current_database_mismatches_target(monkeypatch) -> None:
    class _Cursor:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return None

        def execute(self, statement, params=None):
            self.statement = statement

        def fetchone(self):
            return ("wrong_db",)

        def fetchall(self):
            return []

    class _Conn:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return None

        def cursor(self, *, row_factory=None):
            return _Cursor()

        def rollback(self):
            return None

        def commit(self):
            return None

    monkeypatch.setattr(repair.psycopg, "connect", lambda dsn: _Conn())
    monkeypatch.setattr(repair, "_load_known_repositories", lambda path: {"litellm"})
    monkeypatch.setattr(
        repair,
        "_load_rollout_repository_map",
        lambda memories_dir, known: {},
    )
    monkeypatch.setattr(repair, "_fetch_candidate_rows", lambda *a, **k: [])
    args = repair.argparse.Namespace(
        dsn="postgresql://unused",
        apply=True,
        target_db_name="aawm_tristore",
        projects_dir="/tmp",
        memories_dir="/tmp",
        grok_repository=None,
        batch_size=10,
        cursor_id=0,
        null_repository_since=None,
        repository_value=None,
        max_id=None,
        skip_session_evidence=True,
        session_evidence_limit_per_session=50,
        classify_unresolved=False,
        preview_limit=5,
    )
    import pytest

    with pytest.raises(SystemExit, match="Refusing to apply"):
        repair.repair_repository_identities(args)


def test_should_allow_dry_run_when_current_database_mismatches_target(
    monkeypatch, capsys
) -> None:
    class _Cursor:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return None

        def execute(self, statement, params=None):
            return None

        def fetchone(self):
            return ("wrong_db",)

        def fetchall(self):
            return []

    class _Conn:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return None

        def cursor(self, *, row_factory=None):
            return _Cursor()

        def rollback(self):
            self.rolled_back = True

        def commit(self):
            return None

    monkeypatch.setattr(repair.psycopg, "connect", lambda dsn: _Conn())
    monkeypatch.setattr(repair, "_load_known_repositories", lambda path: {"litellm"})
    monkeypatch.setattr(
        repair,
        "_load_rollout_repository_map",
        lambda memories_dir, known: {},
    )
    monkeypatch.setattr(repair, "_fetch_candidate_rows", lambda *a, **k: [])
    args = repair.argparse.Namespace(
        dsn="postgresql://unused",
        apply=False,
        target_db_name="aawm_tristore",
        projects_dir="/tmp",
        memories_dir="/tmp",
        grok_repository=None,
        batch_size=10,
        cursor_id=0,
        null_repository_since=None,
        repository_value=None,
        max_id=None,
        skip_session_evidence=True,
        session_evidence_limit_per_session=50,
        classify_unresolved=False,
        preview_limit=5,
    )
    assert repair.repair_repository_identities(args) == 0
    out = capsys.readouterr().out
    assert "database=wrong_db" in out
    assert "target_db_name=aawm_tristore" in out
    assert "applied=false" in out


def test_should_reject_non_positive_session_evidence_limit() -> None:
    import pytest

    class _Conn:
        def cursor(self, *, row_factory=None):
            raise AssertionError("cursor should not be opened")

    with pytest.raises(ValueError, match="limit_per_session"):
        repair._fetch_session_identity_rows(_Conn(), {"s"}, limit_per_session=0)
