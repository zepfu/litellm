from datetime import datetime, timezone

from scripts import backfill_claude_auto_review_session_history as backfill


def test_should_repair_permission_row_to_auto_review_identity() -> None:
    rows = [
        {
            "id": 10,
            "created_at": datetime(2026, 5, 19, 12, 0, tzinfo=timezone.utc),
            "session_id": "session-1",
            "provider": "anthropic",
            "model": "claude-opus-4-7",
            "agent_name": "orchestrator",
            "repository": "dashboard-shell",
            "tenant_id": "dashboard-shell",
            "metadata": {
                "request_tags": ["claude-project:dashboard-shell"],
                "trace_name": "claude-code.orchestrator",
            },
        },
        {
            "id": 11,
            "created_at": datetime(2026, 5, 19, 12, 1, tzinfo=timezone.utc),
            "session_id": "session-1",
            "provider": "anthropic",
            "model": "claude-opus-4-7",
            "agent_name": None,
            "repository": "agent-a3ee0f55d7cda22ec",
            "tenant_id": "agent-a3ee0f55d7cda22ec",
            "metadata": {
                "claude_permission_check": True,
                "claude_permission_check_request_model": "claude-opus-4-7[1m]",
                "claude_permission_check_response_model": "claude-opus-4-7",
                "request_tags": ["claude-permission-check"],
            },
        },
    ]

    candidates = backfill._build_session_identity_candidates(rows)
    repaired = backfill._build_repaired_row(rows[1], candidates)

    assert repaired is not None
    assert repaired["model"] == "claude-auto-review"
    assert repaired["agent_name"] == "auto-reviewer"
    assert repaired["repository"] == "dashboard-shell"
    assert repaired["tenant_id"] == "dashboard-shell"
    assert repaired["metadata"]["source_model"] == "claude-opus-4-7"
    assert repaired["metadata"]["logical_model"] == "claude-auto-review"
    assert repaired["metadata"]["trace_name"] == "claude-code.auto-reviewer"
    assert repaired["metadata"]["trace_user_id"] == "dashboard-shell"
    assert "claude-agent:auto-reviewer" in repaired["metadata"]["request_tags"]
    assert "claude-project:dashboard-shell" in repaired["metadata"]["request_tags"]
    assert repaired["metadata"]["auto_review_parent_identity_source_row_id"] == 10
    assert repaired["metadata"]["auto_review_backfill_source"]


def test_should_leave_permission_repository_null_without_parent_identity() -> None:
    row = {
        "id": 20,
        "created_at": datetime(2026, 5, 19, 12, 0, tzinfo=timezone.utc),
        "session_id": "session-2",
        "provider": "anthropic",
        "model": "claude-opus-4-7[1m]",
        "agent_name": None,
        "repository": "agent-a3ee0f55d7cda22ec",
        "tenant_id": "agent-a3ee0f55d7cda22ec",
        "metadata": {
            "claude_permission_check": True,
            "request_tags": ["claude-permission-check"],
        },
    }

    repaired = backfill._build_repaired_row(row, {})

    assert repaired is not None
    assert repaired["model"] == "claude-auto-review"
    assert repaired["repository"] is None
    assert repaired["tenant_id"] is None
    assert "repository" not in repaired["metadata"]
    assert repaired["metadata"]["source_model"] == "claude-opus-4-7[1m]"


def test_should_skip_non_permission_rows() -> None:
    row = {
        "id": 30,
        "created_at": datetime(2026, 5, 19, 12, 0, tzinfo=timezone.utc),
        "session_id": "session-3",
        "provider": "anthropic",
        "model": "claude-opus-4-7",
        "metadata": {"request_tags": ["claude-project:dashboard-shell"]},
    }

    assert backfill._build_repaired_row(row, {}) is None
