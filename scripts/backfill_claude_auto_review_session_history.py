#!/usr/bin/env python3
"""Backfill Claude permission-check rows to the claude-auto-review alias.

Repairs historical session_history rows for Claude permission-check traffic that
were stored under the raw Anthropic model id instead of the logical
``claude-auto-review`` alias. Project/repository identity for each permission
row is resolved from the nearest-in-time non-permission row in the same
session (not a single last-write-wins identity for the whole session).
"""

from __future__ import annotations

import argparse
import asyncio
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

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
from litellm.integrations.aawm_session_history.identity_selection import (  # noqa: E402
    select_first_identity,
)

_AFFECTED_MODELS = {
    "claude-opus-4-7",
    "claude-opus-4-7[1m]",
}
_BACKFILL_SOURCE = "claude_auto_review_session_history_2026_05_19"
_DEFAULT_APPLY_BATCH_SIZE = 200
_DEFAULT_SESSION_LIMIT = 1000
_DEFAULT_COMMAND_TIMEOUT_SECONDS = 30

# Project/repository identity source labels for this backfill, in priority order.
# Selection uses the shared ordered helper in
# ``litellm.integrations.aawm_session_history.identity_selection`` so first-match
# behavior stays consistent with sibling repair/backfill scripts. Values are
# normalized via ``_normalize_repository_identity`` (tags are pre-normalized).
_PROJECT_IDENTITY_CASCADE: Tuple[str, ...] = (
    "row.repository",
    "tag:claude-project",
    "metadata.aawm_claude_project",
    "metadata.repository",
    "row.tenant_id",
    "metadata.tenant_id",
)


@dataclass(frozen=True)
class SessionIdentityCandidate:
    """A non-permission session_history row that can supply project identity."""

    created_at: datetime
    repository: str
    source_row_id: Any


def _parse_datetime(value: Optional[str]) -> datetime:
    if not value:
        return datetime.now(timezone.utc) - timedelta(days=7)
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _as_utc_datetime(value: Any) -> datetime:
    """Normalize created_at-like values for nearest-in-time comparison."""
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)
    if isinstance(value, str) and value.strip():
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return datetime.min.replace(tzinfo=timezone.utc)
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    return datetime.min.replace(tzinfo=timezone.utc)


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


def _project_identity_sources(
    row: Dict[str, Any],
) -> List[Tuple[str, Any]]:
    """Return ordered (source_label, zero-arg extractor) pairs for a row.

    Policy (which fields / order) is local to this backfill; first-match
    selection is delegated to the shared identity_selection helper.
    """
    metadata = _safe_json_metadata(row.get("metadata"))
    extractors = {
        "row.repository": lambda: row.get("repository"),
        "tag:claude-project": lambda: _extract_project_from_tags(metadata),
        "metadata.aawm_claude_project": lambda: metadata.get("aawm_claude_project"),
        "metadata.repository": lambda: metadata.get("repository"),
        "row.tenant_id": lambda: row.get("tenant_id"),
        "metadata.tenant_id": lambda: metadata.get("tenant_id"),
    }
    return [(source, extractors[source]) for source in _PROJECT_IDENTITY_CASCADE]


def _normalize_project_identity_source(source: str, raw: Any) -> Optional[str]:
    """Normalize one project-identity candidate for this backfill."""
    if source == "tag:claude-project":
        # Already normalized by _extract_project_from_tags.
        return raw if isinstance(raw, str) and raw.strip() else None
    return _normalize_repository_identity(raw)


def _project_identity_candidates(
    row: Dict[str, Any]
) -> List[Tuple[str, Optional[str]]]:
    """Return ordered (source_label, normalized_value) pairs for a row.

    Uses the shared ordered-selection contract so sibling scripts can share
    first-match semantics while keeping this cascade's field policy local.
    """
    ordered: List[Tuple[str, Optional[str]]] = []
    for source, extractor in _project_identity_sources(row):
        ordered.append(
            (source, _normalize_project_identity_source(source, extractor()))
        )
    return ordered


def _row_project_identity(row: Dict[str, Any]) -> Optional[str]:
    """Resolve project/repository identity for a session_history row."""
    # Pre-normalize per source so tag vs repository rules stay correct, then
    # delegate first-match to the shared helper (identity_selection).
    selected = select_first_identity(
        [
            (
                source,
                (
                    lambda _source=source, _extractor=extractor: (
                        _normalize_project_identity_source(_source, _extractor())
                    )
                ),
            )
            for source, extractor in _project_identity_sources(row)
        ]
    )
    if selected is None:
        return None
    return selected[1]


def _is_permission_row(row: Dict[str, Any]) -> bool:
    metadata = _safe_json_metadata(row.get("metadata"))
    return (
        row.get("provider") == "anthropic"
        and row.get("model") in _AFFECTED_MODELS
        and _is_claude_permission_check_metadata(metadata)
    )


def _is_permission_check_metadata_row(row: Dict[str, Any]) -> bool:
    """True when metadata marks a Claude permission-check row.

    Used when building identity sources so already-repaired rows (model rewritten
    to ``claude-auto-review``) never seed project identity for other permission
    checks in the same session on re-run or partial apply.
    """
    return _is_claude_permission_check_metadata(
        _safe_json_metadata(row.get("metadata"))
    )


def _build_session_identity_candidates(
    rows: Sequence[Dict[str, Any]],
) -> Dict[str, List[SessionIdentityCandidate]]:
    """Index non-permission rows by session for nearest-in-time lookup.

    Each session may legitimately span more than one repository/project (e.g.
    shared parent session_id across subagent dispatches). Callers must resolve
    identity per permission-check row by timestamp rather than last-write-wins.
    """
    candidates_by_session: Dict[str, List[SessionIdentityCandidate]] = defaultdict(list)
    for row in rows:
        session_id = row.get("session_id")
        # Exclude permission-check metadata rows even after model rewrite so
        # re-runs / partial applies never seed identity from repaired checks.
        if not session_id or _is_permission_check_metadata_row(row):
            continue
        repository = _row_project_identity(row)
        if not repository:
            continue
        candidates_by_session[str(session_id)].append(
            SessionIdentityCandidate(
                created_at=_as_utc_datetime(row.get("created_at")),
                repository=repository,
                source_row_id=row.get("id"),
            )
        )
    for session_id, candidates in candidates_by_session.items():
        candidates_by_session[session_id] = sorted(
            candidates,
            key=lambda item: (item.created_at, str(item.source_row_id)),
        )
    return dict(candidates_by_session)


def _resolve_nearest_session_identity(
    candidates: Sequence[SessionIdentityCandidate],
    target_created_at: Any,
) -> Optional[SessionIdentityCandidate]:
    """Pick the nearest-in-time identity candidate for a permission-check row.

    Tie-break: when two candidates are equidistant, prefer the earlier
    (before-or-at) candidate so earlier project context wins over later
    overwrites within the same absolute delta.
    """
    if not candidates:
        return None

    target = _as_utc_datetime(target_created_at)
    best: Optional[SessionIdentityCandidate] = None
    best_key: Optional[Tuple[float, int, datetime, str]] = None
    for candidate in candidates:
        delta = abs((candidate.created_at - target).total_seconds())
        # 0 = at-or-before target preferred; 1 = after target.
        after_target = 0 if candidate.created_at <= target else 1
        key = (delta, after_target, candidate.created_at, str(candidate.source_row_id))
        if best is None or best_key is None or key < best_key:
            best = candidate
            best_key = key
    return best


def _build_repaired_row(
    row: Dict[str, Any],
    session_identity_candidates: Dict[str, List[SessionIdentityCandidate]],
) -> Optional[Dict[str, Any]]:
    if not _is_permission_row(row):
        return None

    metadata = _safe_json_metadata(row.get("metadata"))
    source_model = metadata.get("source_model") or row.get("model")
    session_id = row.get("session_id") or ""
    nearest = _resolve_nearest_session_identity(
        session_identity_candidates.get(str(session_id), []),
        row.get("created_at"),
    )
    repository = _normalize_repository_identity(
        nearest.repository if nearest is not None else None
    )
    tenant_id = repository

    metadata["source_model"] = source_model
    metadata["logical_model"] = _CLAUDE_AUTO_REVIEW_LOGICAL_MODEL
    metadata["trace_name"] = _CLAUDE_AUTO_REVIEW_TRACE_NAME
    metadata["agent_name"] = _CLAUDE_AUTO_REVIEW_AGENT_NAME
    metadata["aawm_claude_agent_name"] = _CLAUDE_AUTO_REVIEW_AGENT_NAME
    metadata["auto_review_backfill_source"] = _BACKFILL_SOURCE
    if repository and nearest is not None:
        metadata["repository"] = repository
        metadata["tenant_id"] = repository
        metadata["aawm_tenant_id"] = repository
        metadata["aawm_claude_project"] = repository
        metadata["trace_user_id"] = repository
        metadata["auto_review_parent_identity_source_row_id"] = nearest.source_row_id

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


async def _select_rows(
    conn: Any,
    since: datetime,
    session_limit: int,
) -> list[Dict[str, Any]]:
    """Fetch affected sessions and every session_history row in their window.

    ``session_limit`` bounds distinct ``session_id`` values with permission-check
    traffic, not the number of returned rows. The outer SELECT returns all rows
    for each selected session within ``created_at >= since - 30 minutes``.
    """
    # asyncpg uses $n placeholders; LIKE wildcards are plain SQL, so use a
    # single '%' (not '%%' from %-style DBAPI escaping).
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
                  OR metadata::text ILIKE '%claude-permission-check%'
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
        session_limit,
    )
    return [dict(row) for row in rows]


def _chunked(
    items: Sequence[Dict[str, Any]], batch_size: int
) -> Iterable[list[Dict[str, Any]]]:
    size = max(1, int(batch_size))
    for index in range(0, len(items), size):
        yield list(items[index : index + size])


async def _apply_repairs(
    conn: Any,
    repaired_rows: list[Dict[str, Any]],
    *,
    batch_size: int = _DEFAULT_APPLY_BATCH_SIZE,
) -> int:
    """Apply repairs in chunks to avoid one giant executemany under command_timeout."""
    if not repaired_rows:
        return 0

    applied = 0
    sql = """
        UPDATE public.session_history
        SET model = $2,
            agent_name = $3,
            repository = $4,
            tenant_id = $5,
            metadata = $6::jsonb
        WHERE id = $1
        """
    for batch in _chunked(repaired_rows, batch_size):
        await conn.executemany(
            sql,
            [
                (
                    row["id"],
                    row["model"],
                    row["agent_name"],
                    row["repository"],
                    row["tenant_id"],
                    json.dumps(row["metadata"], sort_keys=True),
                )
                for row in batch
            ],
        )
        applied += len(batch)
    return applied


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--since", help="ISO timestamp lower bound; default 7 days ago")
    parser.add_argument(
        "--session-limit",
        "--limit",
        dest="session_limit",
        type=int,
        default=_DEFAULT_SESSION_LIMIT,
        metavar="N",
        help=(
            "Maximum number of distinct affected session_id values to select "
            f"(default: {_DEFAULT_SESSION_LIMIT}). This is a session cap, not a "
            "row cap: every session_history row for each selected session in the "
            "widened time window is still loaded for identity resolution and "
            "repair. Legacy alias: --limit."
        ),
    )
    parser.add_argument(
        "--apply-batch-size",
        type=int,
        default=_DEFAULT_APPLY_BATCH_SIZE,
        metavar="N",
        help=(
            "Chunk size for UPDATE executemany batches when --apply is set "
            f"(default: {_DEFAULT_APPLY_BATCH_SIZE})."
        ),
    )
    parser.add_argument("--apply", action="store_true")
    return parser


async def _async_main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    dsn = _build_aawm_dsn()
    if not dsn:
        raise SystemExit("AAWM database DSN is not configured")

    since = _parse_datetime(args.since)
    session_limit = max(1, int(args.session_limit))
    apply_batch_size = max(1, int(args.apply_batch_size))

    conn = await asyncpg.connect(
        dsn=dsn,
        command_timeout=_DEFAULT_COMMAND_TIMEOUT_SECONDS,
    )
    try:
        rows = await _select_rows(conn, since, session_limit)
        session_identity_candidates = _build_session_identity_candidates(rows)
        repaired_rows = [
            repaired
            for row in rows
            if (repaired := _build_repaired_row(row, session_identity_candidates))
            is not None
        ]
        counts = Counter(
            "with_identity" if row.get("repository") else "without_identity"
            for row in repaired_rows
        )
        applied_rows = 0
        if args.apply and repaired_rows:
            applied_rows = await _apply_repairs(
                conn,
                repaired_rows,
                batch_size=apply_batch_size,
            )
        sys.stdout.write(
            json.dumps(
                {
                    "selected_rows": len(rows),
                    "repaired_rows": len(repaired_rows),
                    "applied_rows": applied_rows,
                    "session_limit": session_limit,
                    "apply_batch_size": apply_batch_size,
                    "counts": dict(counts),
                    "apply": bool(args.apply),
                },
                sort_keys=True,
            )
            + "\n"
        )
    finally:
        await conn.close()

    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    return asyncio.run(_async_main(argv))


if __name__ == "__main__":
    raise SystemExit(main())
