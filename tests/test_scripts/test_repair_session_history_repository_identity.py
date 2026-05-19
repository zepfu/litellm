from scripts import repair_session_history_repository_identity as repair


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
