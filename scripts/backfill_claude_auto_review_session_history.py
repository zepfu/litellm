#!/usr/bin/env python3
"""Backfill Claude permission-check rows to the claude-auto-review alias."""

from __future__ import annotations

import argparse
import asyncio
from collections import Counter
from datetime import datetime, timedelta, timezone
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import asyncpg

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _load_repo_dotenv() -> None:
    dotenv_path = REPO_ROOT / ".env"
    if not dotenv_path.exists():
        return

    for raw_line in dotenv_path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key or key in os.environ:
            continue
        cleaned_value = value.strip()
        if (
            len(cleaned_value) >= 2
            and cleaned_value[0] == cleaned_value[-1]
            and cleaned_value[0] in {'"', "'"}
        ):
            cleaned_value = cleaned_value[1:-1]
        os.environ[key] = cleaned_value


_load_repo_dotenv()

from litellm.integrations.aawm_agent_identity import (  # noqa: E402
    _CLAUDE_AUTO_REVIEW_AGENT_NAME,
    _CLAUDE_AUTO_REVIEW_LOGICAL_MODEL,
    _CLAUDE_AUTO_REVIEW_TRACE_NAME,
    _build_aawm_dsn,
    _is_claude_permission_check_metadata,
    _normalize_repository_identity,
)

_AFFECTED_MODELS = {
    "claude-opus-4-7",
    "claude-opus-4-7[1m]",
}
_BACKFILL_SOURCE = "claude_auto_review_session_history_2026_05_19"


def _parse_datetime(value: Optional[str]) -> datetime:
    if not value:
        return datetime.now(timezone.utc) - timedelta(days=7)
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _safe_json_metadata(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return dict(parsed) if isinstance(parsed, dict) else {}
    return {}


def _metadata_tags(metadata: Dict[str, Any]) -> list[str]:
    tags: list[str] = []
    for key in ("request_tags", "tags"):
        value = metadata.get(key)
        if not isinstance(value, list):
            continue
        for tag in value:
            if isinstance(tag, str) and tag.strip() and tag not in tags:
                tags.append(tag)
    return tags


def _merge_tags(metadata: Dict[str, Any], tags_to_add: Iterable[str]) -> None:
    for key in ("request_tags", "tags"):
        existing = metadata.get(key) or []
        if not isinstance(existing, list):
            existing = []
        merged = list(existing)
        for tag in tags_to_add:
            if tag and tag not in merged:
                merged.append(tag)
        metadata[key] = merged


def _extract_project_from_tags(metadata: Dict[str, Any]) -> Optional[str]:
    for tag in _metadata_tags(metadata):
        if not tag.startswith("claude-project:"):
            continue
        repository = _normalize_repository_identity(tag.split(":", 1)[1])
        if repository:
            return repository
    return None


def _row_project_identity(row: Dict[str, Any]) -> Optional[str]:
    metadata = _safe_json_metadata(row.get("metadata"))
    return (
        _normalize_repository_identity(row.get("repository"))
        or _extract_project_from_tags(metadata)
        or _normalize_repository_identity(metadata.get("aawm_claude_project"))
        or _normalize_repository_identity(metadata.get("repository"))
        or _normalize_repository_identity(row.get("tenant_id"))
        or _normalize_repository_identity(metadata.get("tenant_id"))
    )


def _is_permission_row(row: Dict[str, Any]) -> bool:
    metadata = _safe_json_metadata(row.get("metadata"))
    return (
        row.get("provider") == "anthropic"
        and row.get("model") in _AFFECTED_MODELS
        and _is_claude_permission_check_metadata(metadata)
    )


def _build_session_identity_map(rows: list[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    identity_by_session: Dict[str, Dict[str, Any]] = {}
    for row in sorted(rows, key=lambda item: item.get("created_at") or datetime.min):
        session_id = row.get("session_id")
        if not session_id or _is_permission_row(row):
            continue
        repository = _row_project_identity(row)
        if not repository:
            continue
        identity_by_session[session_id] = {
            "repository": repository,
            "tenant_id": repository,
            "source_row_id": row.get("id"),
        }
    return identity_by_session


def _build_repaired_row(
    row: Dict[str, Any],
    session_identities: Dict[str, Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    if not _is_permission_row(row):
        return None

    metadata = _safe_json_metadata(row.get("metadata"))
    source_model = metadata.get("source_model") or row.get("model")
    session_identity = session_identities.get(row.get("session_id") or "")
    repository = _normalize_repository_identity(
        (session_identity or {}).get("repository")
    )
    tenant_id = repository

    metadata["source_model"] = source_model
    metadata["logical_model"] = _CLAUDE_AUTO_REVIEW_LOGICAL_MODEL
    metadata["trace_name"] = _CLAUDE_AUTO_REVIEW_TRACE_NAME
    metadata["agent_name"] = _CLAUDE_AUTO_REVIEW_AGENT_NAME
    metadata["aawm_claude_agent_name"] = _CLAUDE_AUTO_REVIEW_AGENT_NAME
    metadata["auto_review_backfill_source"] = _BACKFILL_SOURCE
    if repository:
        metadata["repository"] = repository
        metadata["tenant_id"] = repository
        metadata["aawm_tenant_id"] = repository
        metadata["aawm_claude_project"] = repository
        metadata["trace_user_id"] = repository
        metadata["auto_review_parent_identity_source_row_id"] = session_identity.get(
            "source_row_id"
        )

    tags_to_add = [
        "claude-internal-check",
        "claude-permission-check",
        f"claude-agent:{_CLAUDE_AUTO_REVIEW_AGENT_NAME}",
    ]
    if repository:
        tags_to_add.append(f"claude-project:{repository}")
    _merge_tags(metadata, tags_to_add)

    return {
        "id": row["id"],
        "model": _CLAUDE_AUTO_REVIEW_LOGICAL_MODEL,
        "agent_name": _CLAUDE_AUTO_REVIEW_AGENT_NAME,
        "repository": repository,
        "tenant_id": tenant_id,
        "metadata": metadata,
    }


async def _select_rows(conn: Any, since: datetime, limit: int) -> list[Dict[str, Any]]:
    rows = await conn.fetch(
        """
        WITH affected_sessions AS (
            SELECT DISTINCT session_id
            FROM public.session_history
            WHERE created_at >= $1
              AND provider = 'anthropic'
              AND model = ANY($2::text[])
              AND (
                  COALESCE(LOWER(metadata->>'claude_permission_check'), '') IN ('1', 'true', 'yes', 'y')
                  OR metadata::text ILIKE '%%claude-permission-check%%'
              )
            LIMIT $3
        )
        SELECT id, created_at, session_id, provider, model, agent_name,
               tenant_id, repository, response_cost_usd, metadata
        FROM public.session_history
        WHERE session_id IN (SELECT session_id FROM affected_sessions)
          AND created_at >= $1 - INTERVAL '30 minutes'
        ORDER BY session_id, created_at, id
        """,
        since,
        list(_AFFECTED_MODELS),
        limit,
    )
    return [dict(row) for row in rows]


async def _apply_repairs(conn: Any, repaired_rows: list[Dict[str, Any]]) -> None:
    await conn.executemany(
        """
        UPDATE public.session_history
        SET model = $2,
            agent_name = $3,
            repository = $4,
            tenant_id = $5,
            metadata = $6::jsonb
        WHERE id = $1
        """,
        [
            (
                row["id"],
                row["model"],
                row["agent_name"],
                row["repository"],
                row["tenant_id"],
                json.dumps(row["metadata"], sort_keys=True),
            )
            for row in repaired_rows
        ],
    )


async def _async_main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--since", help="ISO timestamp lower bound; default 7 days ago")
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    dsn = _build_aawm_dsn()
    if not dsn:
        raise SystemExit("AAWM database DSN is not configured")

    since = _parse_datetime(args.since)
    conn = await asyncpg.connect(dsn=dsn, command_timeout=30)
    try:
        rows = await _select_rows(conn, since, args.limit)
        session_identities = _build_session_identity_map(rows)
        repaired_rows = [
            repaired
            for row in rows
            if (repaired := _build_repaired_row(row, session_identities)) is not None
        ]
        counts = Counter(
            "with_identity" if row.get("repository") else "without_identity"
            for row in repaired_rows
        )
        sys.stdout.write(
            json.dumps(
                {
                    "selected_rows": len(rows),
                    "repaired_rows": len(repaired_rows),
                    "counts": dict(counts),
                    "apply": bool(args.apply),
                },
                sort_keys=True,
            )
            + "\n"
        )
        if args.apply and repaired_rows:
            await _apply_repairs(conn, repaired_rows)
    finally:
        await conn.close()

    return 0


def main() -> int:
    return asyncio.run(_async_main())


if __name__ == "__main__":
    raise SystemExit(main())
