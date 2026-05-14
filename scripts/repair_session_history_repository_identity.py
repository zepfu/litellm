#!/usr/bin/env python3
"""Repair malformed repository identity values in public.session_history."""

import argparse
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import psycopg
import psycopg.rows

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
    _CODEX_MEMORY_REPOSITORY_SUFFIX,
    _build_aawm_dsn,
    _normalize_repository_identity,
)


_VALID_IDENTITY_SQL = r"^[A-Za-z0-9_.-]+(/[A-Za-z0-9_.-]+)?( \(memory\))?$"
_NOISY_REPO_PREFIX_PATTERNS = (
    re.compile(r"^(?P<repo>[A-Za-z0-9_.-]+)\s+(?:all|commits|files)="),
    re.compile(r"^(?P<repo>[A-Za-z0-9_.-]+)(?:\\\\n|\\n|\n)+"),
)


def _safe_json_metadata(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            return {}
    return {}


def _load_known_repositories(projects_dir: Path) -> set[str]:
    if not projects_dir.exists():
        return {REPO_ROOT.name}
    return {
        path.name
        for path in projects_dir.iterdir()
        if path.is_dir() and not path.name.startswith(".")
    }


def _repo_base(identity: str) -> str:
    if identity.endswith(_CODEX_MEMORY_REPOSITORY_SUFFIX):
        return identity[: -len(_CODEX_MEMORY_REPOSITORY_SUFFIX)]
    return identity


def _is_known_repository(identity: Optional[str], known_repositories: set[str]) -> bool:
    if not identity:
        return False
    base = _repo_base(identity)
    if "/" in base:
        base = base.rsplit("/", 1)[-1]
    return base in known_repositories


def _repair_noisy_identity(
    value: Any, known_repositories: set[str]
) -> Optional[str]:
    normalized = _normalize_repository_identity(value)
    if normalized:
        return normalized
    if not isinstance(value, str):
        return None

    text = value.strip()
    memory_suffix = _CODEX_MEMORY_REPOSITORY_SUFFIX in text
    for pattern in _NOISY_REPO_PREFIX_PATTERNS:
        match = pattern.match(text)
        if not match:
            continue
        candidate = match.group("repo")
        if candidate in known_repositories:
            if memory_suffix:
                return f"{candidate}{_CODEX_MEMORY_REPOSITORY_SUFFIX}"
            return candidate
    return None


def _build_repaired_row(
    row: Dict[str, Any],
    known_repositories: set[str],
) -> Optional[Dict[str, Any]]:
    metadata = _safe_json_metadata(row.get("metadata"))
    original_repository = row.get("repository")
    original_tenant_id = row.get("tenant_id")

    repository = _normalize_repository_identity(original_repository)
    metadata_repository = _normalize_repository_identity(metadata.get("repository"))
    source_repository = _normalize_repository_identity(metadata.get("source_repository"))
    tenant_id = _normalize_repository_identity(original_tenant_id)
    repaired_tenant_id = _repair_noisy_identity(
        original_tenant_id, known_repositories
    )

    if repository is None:
        if metadata_repository is not None:
            repository = metadata_repository
        elif source_repository is not None:
            repository = source_repository
        elif _is_known_repository(tenant_id, known_repositories):
            repository = tenant_id
        elif _is_known_repository(repaired_tenant_id, known_repositories):
            repository = repaired_tenant_id

    if tenant_id is None:
        tenant_id = repaired_tenant_id or repository

    if repository is not None:
        metadata["repository"] = repository
    else:
        metadata.pop("repository", None)

    if tenant_id is not None:
        metadata["tenant_id"] = tenant_id
    else:
        metadata.pop("tenant_id", None)
        metadata.pop("tenant_id_source", None)

    changed = (
        repository != original_repository
        or tenant_id != original_tenant_id
        or metadata != _safe_json_metadata(row.get("metadata"))
    )
    if not changed:
        return None

    metadata["repository_identity_repaired_at"] = datetime.now(timezone.utc).isoformat()
    metadata["repository_identity_repair_source"] = (
        "repair_session_history_repository_identity.py"
    )
    if tenant_id == repository and tenant_id is not None:
        metadata["tenant_id_source"] = "repository_repair"

    return {
        "id": row["id"],
        "repository": repository,
        "tenant_id": tenant_id,
        "metadata": metadata,
        "previous_repository": original_repository,
        "previous_tenant_id": original_tenant_id,
    }


def _fetch_candidate_rows(
    conn: psycopg.Connection,
    *,
    cursor_id: int,
    batch_size: int,
) -> list[Dict[str, Any]]:
    with conn.cursor(row_factory=psycopg.rows.dict_row) as cur:
        cur.execute(
            """
            SELECT id, repository, tenant_id, metadata
            FROM public.session_history
            WHERE id > %s
              AND (
                    (repository IS NOT NULL AND repository !~ %s)
                 OR (tenant_id IS NOT NULL AND tenant_id !~ %s)
                 OR (
                        metadata ? 'repository'
                    AND (metadata->>'repository') !~ %s
                    )
              )
            ORDER BY id ASC
            LIMIT %s
            """,
            (
                cursor_id,
                _VALID_IDENTITY_SQL,
                _VALID_IDENTITY_SQL,
                _VALID_IDENTITY_SQL,
                batch_size,
            ),
        )
        return [dict(row) for row in cur.fetchall()]


def _apply_repairs(
    conn: psycopg.Connection,
    repairs: list[Dict[str, Any]],
) -> None:
    with conn.cursor() as cur:
        cur.executemany(
            """
            UPDATE public.session_history
            SET repository = %s,
                tenant_id = %s,
                metadata = %s::jsonb
            WHERE id = %s
            """,
            [
                (
                    repair["repository"],
                    repair["tenant_id"],
                    json.dumps(repair["metadata"], sort_keys=True),
                    repair["id"],
                )
                for repair in repairs
            ],
        )


def repair_repository_identities(args: argparse.Namespace) -> int:
    dsn = args.dsn or _build_aawm_dsn()
    if not dsn:
        raise SystemExit("No database DSN found. Set AAWM_DB_* or pass --dsn.")

    known_repositories = _load_known_repositories(Path(args.projects_dir))
    total_seen = 0
    total_repaired = 0
    preview: list[Dict[str, Any]] = []
    cursor_id = args.cursor_id

    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT current_database()")
            database_name = cur.fetchone()[0]

        while True:
            rows = _fetch_candidate_rows(
                conn,
                cursor_id=cursor_id,
                batch_size=args.batch_size,
            )
            if not rows:
                break
            total_seen += len(rows)
            cursor_id = max(int(row["id"]) for row in rows)
            repairs = [
                repair
                for row in rows
                if (
                    repair := _build_repaired_row(
                        row,
                        known_repositories,
                    )
                )
                is not None
            ]
            if repairs:
                total_repaired += len(repairs)
                preview.extend(repairs[: max(0, args.preview_limit - len(preview))])
                if args.apply:
                    _apply_repairs(conn, repairs)
                    conn.commit()
            elif args.apply:
                conn.commit()

        if not args.apply:
            conn.rollback()

    print(f"database={database_name}")
    print(f"candidate_rows={total_seen}")
    print(f"repairable_rows={total_repaired}")
    print(f"applied={str(bool(args.apply)).lower()}")
    for repair in preview:
        print(
            "preview "
            f"id={repair['id']} "
            f"repository={repair['previous_repository']!r}->{repair['repository']!r} "
            f"tenant_id={repair['previous_tenant_id']!r}->{repair['tenant_id']!r}"
        )
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Repair malformed public.session_history repository identities."
    )
    parser.add_argument("--dsn", help="PostgreSQL DSN. Defaults to AAWM_DB_* env.")
    parser.add_argument("--apply", action="store_true", help="Apply repairs.")
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument("--cursor-id", type=int, default=0)
    parser.add_argument("--preview-limit", type=int, default=20)
    parser.add_argument("--projects-dir", default="/home/zepfu/projects")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(repair_repository_identities(_parse_args()))
