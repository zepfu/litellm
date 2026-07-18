"""RR-066: backfill_claude_auto_review_session_history correctness fixes."""

from __future__ import annotations

import asyncio
import importlib.util
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, List, Sequence

import pytest

_REPO = Path(__file__).resolve().parents[3]
_SCRIPT = _REPO / "scripts" / "backfill_claude_auto_review_session_history.py"


def _load_module() -> ModuleType:
    module_name = "backfill_claude_auto_review_session_history_rr066"
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


def _dt(minute: int, second: int = 0) -> datetime:
    return datetime(2026, 5, 19, 12, minute, second, tzinfo=timezone.utc)


def _permission_metadata(**extra: Any) -> Dict[str, Any]:
    metadata: Dict[str, Any] = {
        "claude_permission_check": "true",
        "request_tags": ["claude-permission-check"],
        "tags": ["claude-permission-check"],
    }
    metadata.update(extra)
    return metadata


def _work_metadata(repository: str, **extra: Any) -> Dict[str, Any]:
    metadata: Dict[str, Any] = {
        "repository": repository,
        "aawm_claude_project": repository,
        "tenant_id": repository,
        "request_tags": [f"claude-project:{repository}"],
        "tags": [f"claude-project:{repository}"],
    }
    metadata.update(extra)
    return metadata


def test_project_identity_cascade_is_single_ordered_shape(mod: ModuleType) -> None:
    """Cascade policy is local; first-match uses shared identity_selection."""
    assert mod._PROJECT_IDENTITY_CASCADE == (
        "row.repository",
        "tag:claude-project",
        "metadata.aawm_claude_project",
        "metadata.repository",
        "row.tenant_id",
        "metadata.tenant_id",
    )

    row = {
        "repository": None,
        "tenant_id": "tenant-only",
        "metadata": {
            "repository": "meta-repo",
            "aawm_claude_project": None,
            "tenant_id": "meta-tenant",
            "request_tags": ["claude-project:tag-repo"],
            "tags": ["claude-project:tag-repo"],
        },
    }
    candidates = mod._project_identity_candidates(row)
    labels = [label for label, _value in candidates]
    assert labels == list(mod._PROJECT_IDENTITY_CASCADE)
    # First non-empty after normalize should prefer tag over lower metadata.
    assert mod._row_project_identity(row) == "tag-repo"

    row_with_column = {
        "repository": "column-repo",
        "tenant_id": "tenant-only",
        "metadata": row["metadata"],
    }
    assert mod._row_project_identity(row_with_column) == "column-repo"

    # Cross-script reuse: selection must go through the shared package helper.
    from litellm.integrations.aawm_session_history.identity_selection import (
        select_first_identity as shared_select,
    )

    assert mod.select_first_identity is shared_select


def test_nearest_session_identity_is_timestamp_aware(mod: ModuleType) -> None:
    """Permission rows pick nearest-in-time identity, not last-write-wins."""
    session_id = "shared-parent-session"
    rows: List[Dict[str, Any]] = [
        {
            "id": 1,
            "created_at": _dt(0),
            "session_id": session_id,
            "provider": "anthropic",
            "model": "claude-opus-4-7",
            "agent_name": "coder",
            "tenant_id": "repo-a",
            "repository": "repo-a",
            "metadata": _work_metadata("repo-a"),
        },
        {
            "id": 2,
            "created_at": _dt(1),
            "session_id": session_id,
            "provider": "anthropic",
            "model": "claude-opus-4-7",
            "agent_name": None,
            "tenant_id": None,
            "repository": None,
            "metadata": _permission_metadata(),
        },
        {
            "id": 3,
            "created_at": _dt(10),
            "session_id": session_id,
            "provider": "anthropic",
            "model": "claude-opus-4-7",
            "agent_name": "coder",
            "tenant_id": "repo-b",
            "repository": "repo-b",
            "metadata": _work_metadata("repo-b"),
        },
        {
            "id": 4,
            "created_at": _dt(11),
            "session_id": session_id,
            "provider": "anthropic",
            "model": "claude-opus-4-7",
            "agent_name": None,
            "tenant_id": None,
            "repository": None,
            "metadata": _permission_metadata(),
        },
    ]

    candidates = mod._build_session_identity_candidates(rows)
    assert set(candidates) == {session_id}
    assert [c.repository for c in candidates[session_id]] == ["repo-a", "repo-b"]

    early = mod._build_repaired_row(rows[1], candidates)
    late = mod._build_repaired_row(rows[3], candidates)
    assert early is not None
    assert late is not None
    assert early["repository"] == "repo-a"
    assert early["tenant_id"] == "repo-a"
    assert early["metadata"]["auto_review_parent_identity_source_row_id"] == 1
    assert late["repository"] == "repo-b"
    assert late["tenant_id"] == "repo-b"
    assert late["metadata"]["auto_review_parent_identity_source_row_id"] == 3

    # Equidistant before/after: prefer the earlier (at-or-before) candidate.
    midpoint = _dt(5)
    tie = mod._resolve_nearest_session_identity(candidates[session_id], midpoint)
    assert tie is not None
    assert tie.repository == "repo-a"
    assert tie.source_row_id == 1


def test_select_sql_uses_single_percent_wildcard_and_session_limit(
    mod: ModuleType,
) -> None:
    captured: Dict[str, Any] = {}

    class FakeConn:
        async def fetch(self, sql: str, *args: Any) -> list[Any]:
            captured["sql"] = sql
            captured["args"] = args
            return []

    since = _dt(0)
    asyncio.run(mod._select_rows(FakeConn(), since, session_limit=7))

    sql = captured["sql"]
    assert "ILIKE '%claude-permission-check%'" in sql
    assert "%%claude-permission-check%%" not in sql
    assert "LIMIT $3" in sql
    assert captured["args"][0] == since
    assert set(captured["args"][1]) == set(mod._AFFECTED_MODELS)
    assert captured["args"][2] == 7


def test_apply_repairs_chunks_executemany(mod: ModuleType) -> None:
    batch_sizes: List[int] = []

    class FakeConn:
        async def executemany(self, sql: str, args: Sequence[Any]) -> None:
            batch_sizes.append(len(args))
            assert "UPDATE public.session_history" in sql

    repaired_rows = [
        {
            "id": index,
            "model": mod._CLAUDE_AUTO_REVIEW_LOGICAL_MODEL,
            "agent_name": mod._CLAUDE_AUTO_REVIEW_AGENT_NAME,
            "repository": f"repo-{index}",
            "tenant_id": f"repo-{index}",
            "metadata": {"n": index},
        }
        for index in range(5)
    ]

    applied = asyncio.run(mod._apply_repairs(FakeConn(), repaired_rows, batch_size=2))
    assert applied == 5
    assert batch_sizes == [2, 2, 1]


def test_arg_parser_documents_session_limit_semantics(mod: ModuleType) -> None:
    parser = mod._build_arg_parser()
    help_text = parser.format_help()

    # argparse may wrap long help lines; normalize whitespace for substring checks.
    help_compact = " ".join(help_text.split())
    assert "--session-limit" in help_text
    assert "session cap, not a row cap" in help_compact
    assert "Legacy alias: --limit" in help_compact
    assert "--apply-batch-size" in help_text

    # Primary flag and legacy alias share dest.
    ns_primary = parser.parse_args(["--session-limit", "42"])
    assert ns_primary.session_limit == 42
    ns_alias = parser.parse_args(["--limit", "17"])
    assert ns_alias.session_limit == 17
    ns_default = parser.parse_args([])
    assert ns_default.session_limit == mod._DEFAULT_SESSION_LIMIT
    assert ns_default.apply_batch_size == mod._DEFAULT_APPLY_BATCH_SIZE


def test_repaired_row_sets_auto_review_fields_without_identity(mod: ModuleType) -> None:
    row = {
        "id": 99,
        "created_at": _dt(0),
        "session_id": "lonely",
        "provider": "anthropic",
        "model": "claude-opus-4-7[1m]",
        "agent_name": None,
        "tenant_id": None,
        "repository": None,
        "metadata": _permission_metadata(source_model="claude-opus-4-7[1m]"),
    }
    repaired = mod._build_repaired_row(row, {})
    assert repaired is not None
    assert repaired["model"] == mod._CLAUDE_AUTO_REVIEW_LOGICAL_MODEL
    assert repaired["agent_name"] == mod._CLAUDE_AUTO_REVIEW_AGENT_NAME
    assert repaired["repository"] is None
    assert (
        repaired["metadata"]["logical_model"] == mod._CLAUDE_AUTO_REVIEW_LOGICAL_MODEL
    )
    assert repaired["metadata"]["source_model"] == "claude-opus-4-7[1m]"
    assert "auto_review_parent_identity_source_row_id" not in repaired["metadata"]


def test_non_permission_rows_are_not_repaired(mod: ModuleType) -> None:
    row = {
        "id": 1,
        "created_at": _dt(0),
        "session_id": "s1",
        "provider": "anthropic",
        "model": "claude-opus-4-7",
        "repository": "repo-a",
        "tenant_id": "repo-a",
        "metadata": _work_metadata("repo-a"),
    }
    assert mod._build_repaired_row(row, {}) is None


def test_already_repaired_permission_rows_do_not_seed_identity(mod: ModuleType) -> None:
    """Re-run safety: rewritten model must not make permission rows identity sources."""
    session_id = "rerun-session"
    rows: List[Dict[str, Any]] = [
        {
            "id": 1,
            "created_at": _dt(0),
            "session_id": session_id,
            "provider": "anthropic",
            # Already repaired: model is the logical alias, not the raw Opus id.
            "model": mod._CLAUDE_AUTO_REVIEW_LOGICAL_MODEL,
            "agent_name": mod._CLAUDE_AUTO_REVIEW_AGENT_NAME,
            "tenant_id": "poison-repo",
            "repository": "poison-repo",
            "metadata": _permission_metadata(
                repository="poison-repo",
                aawm_claude_project="poison-repo",
                auto_review_backfill_source=mod._BACKFILL_SOURCE,
                request_tags=[
                    "claude-permission-check",
                    "claude-project:poison-repo",
                    f"claude-agent:{mod._CLAUDE_AUTO_REVIEW_AGENT_NAME}",
                ],
                tags=[
                    "claude-permission-check",
                    "claude-project:poison-repo",
                ],
            ),
        },
        {
            "id": 2,
            "created_at": _dt(1),
            "session_id": session_id,
            "provider": "anthropic",
            "model": "claude-opus-4-7",
            "agent_name": None,
            "tenant_id": None,
            "repository": None,
            "metadata": _permission_metadata(),
        },
    ]
    candidates = mod._build_session_identity_candidates(rows)
    assert candidates == {}
    repaired = mod._build_repaired_row(rows[1], candidates)
    assert repaired is not None
    assert repaired["repository"] is None
    assert "auto_review_parent_identity_source_row_id" not in repaired["metadata"]


def test_nearest_prefers_closer_after_over_far_before(mod: ModuleType) -> None:
    session_id = "near-after-session"
    rows: List[Dict[str, Any]] = [
        {
            "id": 1,
            "created_at": _dt(0),
            "session_id": session_id,
            "provider": "anthropic",
            "model": "claude-opus-4-7",
            "repository": "far-before",
            "tenant_id": "far-before",
            "metadata": _work_metadata("far-before"),
        },
        {
            "id": 2,
            "created_at": _dt(9),
            "session_id": session_id,
            "provider": "anthropic",
            "model": "claude-opus-4-7",
            "repository": None,
            "tenant_id": None,
            "metadata": _permission_metadata(),
        },
        {
            "id": 3,
            "created_at": _dt(10),
            "session_id": session_id,
            "provider": "anthropic",
            "model": "claude-opus-4-7",
            "repository": "near-after",
            "tenant_id": "near-after",
            "metadata": _work_metadata("near-after"),
        },
    ]
    candidates = mod._build_session_identity_candidates(rows)
    repaired = mod._build_repaired_row(rows[1], candidates)
    assert repaired is not None
    assert repaired["repository"] == "near-after"
    assert repaired["metadata"]["auto_review_parent_identity_source_row_id"] == 3


def test_nearest_resolves_after_only_identity(mod: ModuleType) -> None:
    session_id = "after-only"
    rows: List[Dict[str, Any]] = [
        {
            "id": 10,
            "created_at": _dt(0),
            "session_id": session_id,
            "provider": "anthropic",
            "model": "claude-opus-4-7",
            "repository": None,
            "tenant_id": None,
            "metadata": _permission_metadata(),
        },
        {
            "id": 11,
            "created_at": _dt(5),
            "session_id": session_id,
            "provider": "anthropic",
            "model": "claude-opus-4-7",
            "repository": "later-repo",
            "tenant_id": "later-repo",
            "metadata": _work_metadata("later-repo"),
        },
    ]
    candidates = mod._build_session_identity_candidates(rows)
    repaired = mod._build_repaired_row(rows[0], candidates)
    assert repaired is not None
    assert repaired["repository"] == "later-repo"
    assert repaired["metadata"]["auto_review_parent_identity_source_row_id"] == 11
