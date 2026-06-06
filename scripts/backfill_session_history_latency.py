#!/usr/bin/env python3
"""Backfill first-class latency columns on public.session_history."""

import argparse
import json
import os
import sys
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
    _AAWM_SESSION_HISTORY_ALTER_STATEMENTS,
    _AAWM_SESSION_HISTORY_INDEX_STATEMENTS,
    _AAWM_SESSION_HISTORY_TABLE_SQL,
    _SESSION_HISTORY_PREVIOUS_GAP_FIELD,
    _SESSION_HISTORY_LATENCY_FIELDS,
    _build_aawm_dsn,
    _build_session_history_latency_breakdown,
)


def _build_aawm_admin_dsn() -> Optional[str]:
    direct_dsn = os.getenv("AAWM_DIRECT_DATABASE_URL")
    if direct_dsn and direct_dsn.strip():
        return direct_dsn.strip()
    return _build_aawm_dsn()


_LATENCY_METADATA_KEYS = (
    "aawm_local_prepare_ms",
    "aawm_upstream_wait_ms",
    "aawm_time_to_first_token_ms",
    "aawm_upstream_first_chunk_ms",
    "aawm_first_emitted_chunk_ms",
    "aawm_upstream_stream_complete_ms",
    "aawm_local_stream_finalize_ms",
    "aawm_local_finalize_ms",
    "aawm_total_proxy_overhead_ms",
    "aawm_total_proxy_duration_ms",
)

_UPDATE_SQL = f"""
UPDATE public.session_history
SET
    {", ".join(f"{field} = %s" for field in _SESSION_HISTORY_LATENCY_FIELDS)}
WHERE id = %s
"""
_GAP_UPDATE_SQL = f"""
UPDATE public.session_history
SET {_SESSION_HISTORY_PREVIOUS_GAP_FIELD} = %s
WHERE id = %s
"""


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


def _ensure_session_history_schema(conn: psycopg.Connection) -> None:
    with conn.cursor() as cur:
        cur.execute(_AAWM_SESSION_HISTORY_TABLE_SQL)
        for statement in _AAWM_SESSION_HISTORY_ALTER_STATEMENTS:
            cur.execute(statement)
        for statement in _AAWM_SESSION_HISTORY_INDEX_STATEMENTS:
            cur.execute(statement)
    conn.commit()


def _same_float(left: Any, right: Any) -> bool:
    if left is None and right is None:
        return True
    if left is None or right is None:
        return False
    try:
        return abs(float(left) - float(right)) < 0.0005
    except (TypeError, ValueError):
        return False


def _build_where_sql(args: argparse.Namespace) -> tuple[str, list[Any]]:
    where_clauses = [
        "sh.id > %s",
        """
        (
            COALESCE(sh.metadata, '{}'::jsonb) ?| %s::text[]
            OR (sh.start_time IS NOT NULL AND sh.end_time IS NOT NULL)
        )
        """,
    ]
    params: list[Any] = [args.cursor_id, list(_LATENCY_METADATA_KEYS)]
    if args.provider:
        where_clauses.append("sh.provider = %s")
        params.append(args.provider)
    if args.session_id:
        where_clauses.append("sh.session_id = %s")
        params.append(args.session_id)
    if not args.recompute_existing:
        where_clauses.append(
            "("
            + " OR ".join(
                f"sh.{field} IS NULL" for field in _SESSION_HISTORY_LATENCY_FIELDS
            )
            + ")"
        )
    return " AND ".join(f"({clause})" for clause in where_clauses), params


def _build_gap_where_sql(args: argparse.Namespace) -> tuple[str, list[Any]]:
    where_clauses = ["sh.id > %s"]
    params: list[Any] = [args.cursor_id]
    if args.provider:
        where_clauses.append("sh.provider = %s")
        params.append(args.provider)
    if args.session_id:
        where_clauses.append("sh.session_id = %s")
        params.append(args.session_id)
    if not args.recompute_existing:
        where_clauses.append(f"sh.{_SESSION_HISTORY_PREVIOUS_GAP_FIELD} IS NULL")
    return " AND ".join(f"({clause})" for clause in where_clauses), params


def _run_latency_backfill_pass(
    conn: psycopg.Connection,
    args: argparse.Namespace,
    initial_cursor_id: int,
) -> Dict[str, Any]:
    scanned_rows = 0
    derivable_rows = 0
    changed_rows = 0
    updated_rows = 0
    cursor_id = initial_cursor_id

    while True:
        args.cursor_id = cursor_id
        where_sql, params = _build_where_sql(args)
        query = f"""
            SELECT
                sh.id,
                sh.start_time,
                sh.end_time,
                sh.metadata,
                {", ".join(f"sh.{field}" for field in _SESSION_HISTORY_LATENCY_FIELDS)}
            FROM public.session_history sh
            WHERE {where_sql}
            ORDER BY sh.id ASC
            LIMIT {int(args.batch_size)}
        """
        with conn.cursor() as cur:
            cur.execute(query, params)
            rows = cur.fetchall()
        if not rows:
            break

        updates: list[tuple[Any, ...]] = []
        for row in rows:
            scanned_rows += 1
            cursor_id = int(row["id"])
            derived = _build_session_history_latency_breakdown(
                metadata=_safe_json_metadata(row.get("metadata")),
                start_time=row.get("start_time"),
                end_time=row.get("end_time"),
            )
            if not any(value is not None for value in derived.values()):
                continue
            derivable_rows += 1

            new_values: Dict[str, Optional[float]] = {}
            row_changed = False
            for field in _SESSION_HISTORY_LATENCY_FIELDS:
                candidate = derived.get(field)
                current = row.get(field)
                new_value = candidate if candidate is not None else current
                new_values[field] = new_value
                if not _same_float(current, new_value):
                    row_changed = True

            if not row_changed:
                continue
            changed_rows += 1
            updates.append(
                tuple(new_values[field] for field in _SESSION_HISTORY_LATENCY_FIELDS)
                + (row["id"],)
            )

        if args.apply and updates:
            with conn.cursor() as cur:
                cur.executemany(_UPDATE_SQL, updates)
            conn.commit()
            updated_rows += len(updates)
        elif updates:
            conn.rollback()

        if args.limit and scanned_rows >= args.limit:
            break
        if len(rows) < args.batch_size:
            break

    return {
        "scanned_rows": scanned_rows,
        "derivable_rows": derivable_rows,
        "changed_rows": changed_rows,
        "updated_rows": updated_rows,
        "last_id": cursor_id,
    }


def _run_gap_backfill_pass(
    conn: psycopg.Connection,
    args: argparse.Namespace,
    initial_cursor_id: int,
) -> Dict[str, Any]:
    scanned_rows = 0
    derivable_rows = 0
    changed_rows = 0
    updated_rows = 0
    cursor_id = initial_cursor_id

    while True:
        args.cursor_id = cursor_id
        where_sql, params = _build_gap_where_sql(args)
        query = f"""
            SELECT
                sh.id,
                sh.{_SESSION_HISTORY_PREVIOUS_GAP_FIELD},
                CASE
                    WHEN previous.end_time IS NULL THEN NULL
                    WHEN COALESCE(sh.start_time, sh.created_at) >= previous.end_time THEN
                        EXTRACT(
                            EPOCH FROM (
                                COALESCE(sh.start_time, sh.created_at)
                                - previous.end_time
                            )
                        ) * 1000.0
                    ELSE NULL
                END AS derived_gap_ms
            FROM public.session_history sh
            LEFT JOIN LATERAL (
                SELECT previous_sh.end_time
                FROM public.session_history previous_sh
                WHERE previous_sh.session_id = sh.session_id
                  AND (
                      COALESCE(previous_sh.start_time, previous_sh.created_at),
                      previous_sh.id
                  ) < (COALESCE(sh.start_time, sh.created_at), sh.id)
                ORDER BY
                    COALESCE(previous_sh.start_time, previous_sh.created_at) DESC,
                    previous_sh.id DESC
                LIMIT 1
            ) previous ON TRUE
            WHERE {where_sql}
            ORDER BY sh.id ASC
            LIMIT {int(args.batch_size)}
        """
        with conn.cursor() as cur:
            cur.execute(query, params)
            rows = cur.fetchall()
        if not rows:
            break

        updates: list[tuple[Any, ...]] = []
        for row in rows:
            scanned_rows += 1
            cursor_id = int(row["id"])
            derived_gap_ms = row.get("derived_gap_ms")
            if derived_gap_ms is not None:
                derivable_rows += 1
            if _same_float(row.get(_SESSION_HISTORY_PREVIOUS_GAP_FIELD), derived_gap_ms):
                continue
            changed_rows += 1
            updates.append((derived_gap_ms, row["id"]))

        if args.apply and updates:
            with conn.cursor() as cur:
                cur.executemany(_GAP_UPDATE_SQL, updates)
            conn.commit()
            updated_rows += len(updates)
        elif updates:
            conn.rollback()

        if args.limit and scanned_rows >= args.limit:
            break
        if len(rows) < args.batch_size:
            break

    return {
        "gap_scanned_rows": scanned_rows,
        "gap_derivable_rows": derivable_rows,
        "gap_changed_rows": changed_rows,
        "gap_updated_rows": updated_rows,
        "gap_last_id": cursor_id,
    }


def _run_backfill(args: argparse.Namespace) -> Dict[str, Any]:
    dsn = _build_aawm_admin_dsn()
    if not dsn:
        raise RuntimeError("AAWM/tristore database configuration is missing")

    initial_cursor_id = int(args.cursor_id or 0)
    with psycopg.connect(dsn, row_factory=psycopg.rows.dict_row) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT current_database()")
            database_name = cur.fetchone()["current_database"]
        if args.expected_database and database_name != args.expected_database:
            raise RuntimeError(
                f"Refusing to backfill database {database_name!r}; "
                f"expected {args.expected_database!r}"
            )

        _ensure_session_history_schema(conn)
        latency_summary = _run_latency_backfill_pass(conn, args, initial_cursor_id)
        gap_summary = _run_gap_backfill_pass(conn, args, initial_cursor_id)

    return {
        "applied": bool(args.apply),
        "database": args.expected_database,
        **latency_summary,
        **gap_summary,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backfill session_history latency columns from stored metadata.",
    )
    parser.add_argument("--apply", action="store_true", help="Write updates.")
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--cursor-id", type=int, default=0)
    parser.add_argument("--provider")
    parser.add_argument("--session-id")
    parser.add_argument(
        "--recompute-existing",
        action="store_true",
        help="Recompute rows even when all latency columns are already populated.",
    )
    parser.add_argument(
        "--expected-database",
        default="aawm_tristore",
        help="Refuse to run unless current_database() matches this value.",
    )
    return parser.parse_args()


def main() -> int:
    summary = _run_backfill(_parse_args())
    print(json.dumps(summary, sort_keys=True))  # noqa: T201
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
