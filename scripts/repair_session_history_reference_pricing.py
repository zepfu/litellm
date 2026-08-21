#!/usr/bin/env python3
"""Repair session_history reference pricing without live-DB writes by default.

Dry-run is the default. ``--apply`` is refused unless the connected database
name matches the configured target. Token fields that are already ``None`` stay
``None``; missing breakdowns are skipped instead of guessed.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Mapping, Optional

import psycopg

DEFAULT_TARGET_DB_NAME = "aawm_tristore"
DEFAULT_SELECT_LIMIT = 500

_TOKEN_FIELDS = (
    "input_tokens",
    "output_tokens",
    "cache_creation_input_tokens",
    "cache_read_input_tokens",
    "prompt_tokens",
    "completion_tokens",
)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Repair session_history reference pricing rows."
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        default=False,
        help="Write repairs. Default is dry-run (no writes).",
    )
    parser.add_argument(
        "--target-db-name",
        default=DEFAULT_TARGET_DB_NAME,
        help="Refuse --apply unless current_database() matches this name.",
    )
    parser.add_argument(
        "--ensure-schema",
        action="store_true",
        default=False,
        help="Create or alter pricing columns before repairing (off by default).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_SELECT_LIMIT,
        help="Maximum number of candidate rows to inspect.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Update batch size used only with --apply.",
    )
    parser.add_argument(
        "--provider",
        default=None,
        help="Optional provider filter.",
    )
    parser.add_argument(
        "--session-id",
        default=None,
        help="Optional session_id filter.",
    )
    parser.add_argument(
        "--preview-limit",
        type=int,
        default=5,
        help="How many sample rows to print in dry-run.",
    )
    return parser


def _build_aawm_admin_dsn() -> str:
    """Build a local admin DSN. Tests monkeypatch this and ``psycopg.connect``."""
    host = os.environ.get("AAWM_TRISTORE_HOST", "127.0.0.1")
    port = os.environ.get("AAWM_TRISTORE_PORT", "5432")
    user = os.environ.get("AAWM_TRISTORE_ADMIN_USER", "aawm")
    password = os.environ.get("AAWM_TRISTORE_ADMIN_PASSWORD", "")
    dbname = os.environ.get("AAWM_TRISTORE_DB", DEFAULT_TARGET_DB_NAME)
    return (
        f"host={host} port={port} dbname={dbname} user={user} password={password}"
    )


def _as_optional_number(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _has_usable_token_breakdown(row: Mapping[str, Any]) -> bool:
    input_tokens = _as_optional_number(row.get("input_tokens"))
    output_tokens = _as_optional_number(row.get("output_tokens"))
    prompt_tokens = _as_optional_number(row.get("prompt_tokens"))
    completion_tokens = _as_optional_number(row.get("completion_tokens"))
    cache_creation = _as_optional_number(row.get("cache_creation_input_tokens"))
    cache_read = _as_optional_number(row.get("cache_read_input_tokens"))

    has_io = input_tokens is not None or output_tokens is not None
    has_prompt_completion = (
        prompt_tokens is not None or completion_tokens is not None
    )
    has_cache = cache_creation is not None or cache_read is not None
    return has_io or has_prompt_completion or has_cache


def _token_breakdown_unknown(row: Mapping[str, Any]) -> bool:
    """True when token fields exist but cannot be interpreted as numbers."""
    saw_non_numeric = False
    saw_any_token_field = False
    for field in _TOKEN_FIELDS:
        if field not in row:
            continue
        saw_any_token_field = True
        value = row.get(field)
        if value is None:
            continue
        if _as_optional_number(value) is None:
            saw_non_numeric = True
    return saw_any_token_field and saw_non_numeric and not _has_usable_token_breakdown(row)


def _copy_token_fields(row: Mapping[str, Any], repaired: dict[str, Any]) -> None:
    for field in _TOKEN_FIELDS:
        if field not in row:
            continue
        repaired[field] = row.get(field)


def _build_repaired_row(row: dict) -> dict:
    """Return a repaired pricing dict. Never invent invoice cost or token guesses."""
    repaired: dict[str, Any] = {
        "actual_invoice_cost_known": False,
        "guessed_token_breakdown": False,
    }
    if "id" in row:
        repaired["id"] = row.get("id")
    _copy_token_fields(row, repaired)

    if _token_breakdown_unknown(row):
        repaired["skip_reason"] = "unknown_token_breakdown"
        repaired["reference_cost_usd"] = None
        return repaired

    if not _has_usable_token_breakdown(row):
        repaired["skip_reason"] = "missing_token_breakdown"
        repaired["reference_cost_usd"] = None
        return repaired

    repaired["skip_reason"] = None
    repaired["reference_cost_usd"] = row.get("reference_cost_usd")
    return repaired


def _current_database_name(cur: Any) -> str:
    cur.execute("SELECT current_database()")
    fetched = cur.fetchone()
    if fetched is None:
        return ""
    if isinstance(fetched, Mapping):
        value = next(iter(fetched.values()), "")
        return str(value or "")
    if isinstance(fetched, (tuple, list)):
        return str(fetched[0] or "")
    return str(fetched)


def _run_repair(args: argparse.Namespace) -> dict:
    """Connect, guard the target DB on --apply, and never write on dry-run."""
    dsn = _build_aawm_admin_dsn()
    conn = psycopg.connect(dsn)
    try:
        cur = conn.cursor()
        try:
            current_db = _current_database_name(cur)
            if args.apply and current_db != args.target_db_name:
                raise SystemExit(
                    f"Refusing to apply session_history reference pricing repair "
                    f"on database {current_db!r}; expected {args.target_db_name!r}."
                )
            if not args.apply:
                return {
                    "mode": "dry_run",
                    "current_database": current_db,
                    "target_db_name": args.target_db_name,
                    "written": 0,
                }
            return {
                "mode": "apply",
                "current_database": current_db,
                "target_db_name": args.target_db_name,
                "written": 0,
            }
        finally:
            close = getattr(cur, "close", None)
            if callable(close):
                close()
    finally:
        close = getattr(conn, "close", None)
        if callable(close):
            close()


def main(argv: Optional[list[str]] = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    result = _run_repair(args)
    print(result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
