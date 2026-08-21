#!/usr/bin/env python3
"""Repair session_history reference pricing without live-DB writes by default.

Dry-run is the default. ``--apply`` is refused unless the connected database
name matches the configured target. Token fields that are already ``None`` stay
``None``; missing breakdowns are skipped instead of guessed.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import psycopg
import psycopg.rows

DEFAULT_TARGET_DB_NAME = "aawm_tristore"
DEFAULT_SELECT_LIMIT = 500
DEFAULT_BATCH_SIZE = 50
REPO_ROOT = Path(__file__).resolve().parents[1]
_COST_MAP_PATH = REPO_ROOT / "model_prices_and_context_window.json"

_TOKEN_FIELDS = (
    "input_tokens",
    "output_tokens",
    "cache_creation_input_tokens",
    "cache_read_input_tokens",
    "prompt_tokens",
    "completion_tokens",
)

_SELECT_CANDIDATE_SQL = """
SELECT
    id,
    provider,
    model,
    inbound_model_alias,
    input_tokens,
    output_tokens,
    cache_read_input_tokens,
    cache_creation_input_tokens,
    response_cost_usd,
    metadata
FROM public.session_history
WHERE 1=1
"""

_UPDATE_SQL = """
UPDATE public.session_history
SET reference_cost_usd = %s,
    actual_invoice_cost_known = FALSE
WHERE id = %s
"""

_MODEL_COST_MAP: Optional[dict[str, Any]] = None


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
        default=DEFAULT_BATCH_SIZE,
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


def _load_model_cost_map() -> dict[str, Any]:
    global _MODEL_COST_MAP
    if _MODEL_COST_MAP is not None:
        return _MODEL_COST_MAP
    try:
        payload = json.loads(_COST_MAP_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        payload = {}
    _MODEL_COST_MAP = payload if isinstance(payload, dict) else {}
    return _MODEL_COST_MAP


def _lookup_model_info(row: Mapping[str, Any]) -> Optional[Mapping[str, Any]]:
    model = str(row.get("model") or "").strip()
    provider = str(row.get("provider") or "").strip()
    if not model:
        return None
    cost_map = _load_model_cost_map()
    if not cost_map:
        return None

    candidates: list[str] = []
    if provider:
        provider_prefix = f"{provider}/"
        if model.startswith(provider_prefix):
            candidates.append(model)
            stripped = model[len(provider_prefix) :]
            if stripped:
                candidates.append(stripped)
        else:
            candidates.append(f"{provider_prefix}{model}")
            candidates.append(model)
    else:
        candidates.append(model)

    lookup = {key.lower(): key for key in cost_map if isinstance(key, str)}
    for candidate in candidates:
        info = cost_map.get(candidate)
        if isinstance(info, dict):
            return info
        matched = lookup.get(candidate.lower())
        matched_info = cost_map.get(matched) if matched is not None else None
        if isinstance(matched_info, dict):
            return matched_info
    return None


def _first_present_token_count(row: Mapping[str, Any], *fields: str) -> Optional[float]:
    for field in fields:
        if field not in row:
            continue
        value = _as_optional_number(row.get(field))
        if value is not None:
            return value
    return None


def _calculate_reference_cost_usd(row: Mapping[str, Any]) -> Optional[float]:
    """Provider-equivalent reference cost from stored tokens and the cost map."""
    model_info = _lookup_model_info(row)
    if not model_info:
        return None
    if (
        "input_cost_per_token" not in model_info
        and "output_cost_per_token" not in model_info
    ):
        return None

    prompt_tokens = _first_present_token_count(row, "input_tokens", "prompt_tokens")
    completion_tokens = _first_present_token_count(
        row, "output_tokens", "completion_tokens"
    )
    if prompt_tokens is None and completion_tokens is None:
        return None

    input_rate = _as_optional_number(model_info.get("input_cost_per_token")) or 0.0
    output_rate = _as_optional_number(model_info.get("output_cost_per_token")) or 0.0
    return (prompt_tokens or 0.0) * input_rate + (completion_tokens or 0.0) * output_rate


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

    reference_cost = _calculate_reference_cost_usd(row)
    if reference_cost is None:
        repaired["skip_reason"] = "unpriced"
        repaired["reference_cost_usd"] = None
        return repaired

    repaired["skip_reason"] = None
    repaired["reference_cost_usd"] = reference_cost
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


def _positive_int(value: Any, default: int) -> int:
    if value is None:
        return default
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed


def _inspect_limit(args: argparse.Namespace) -> int:
    limit = _positive_int(getattr(args, "limit", None), DEFAULT_SELECT_LIMIT)
    return max(limit, 0)


def _update_batch_size(args: argparse.Namespace) -> int:
    batch_size = _positive_int(getattr(args, "batch_size", None), DEFAULT_BATCH_SIZE)
    return batch_size if batch_size > 0 else DEFAULT_BATCH_SIZE


def _select_candidate_rows(cur: Any, args: argparse.Namespace) -> list[dict[str, Any]]:
    sql = _SELECT_CANDIDATE_SQL
    params: list[Any] = []
    if args.provider:
        sql += " AND provider = %s"
        params.append(args.provider)
    if args.session_id:
        sql += " AND session_id = %s"
        params.append(args.session_id)
    sql += " ORDER BY id ASC LIMIT %s"
    params.append(_inspect_limit(args))
    cur.execute(sql, tuple(params))
    fetched = cur.fetchall() or []
    rows: list[dict[str, Any]] = []
    for item in fetched:
        if isinstance(item, Mapping):
            rows.append(dict(item))
        elif isinstance(item, (tuple, list)):
            rows.append({"id": item[0] if item else None})
    return rows


def _repair_update_params(row: Mapping[str, Any]) -> Optional[tuple[float, Any]]:
    repaired = _build_repaired_row(dict(row))
    if repaired.get("skip_reason"):
        return None
    reference_cost = _as_optional_number(repaired.get("reference_cost_usd"))
    row_id = repaired.get("id", row.get("id"))
    if reference_cost is None or row_id is None:
        return None
    return (float(reference_cost), row_id)


def _batched(items: Sequence[tuple[float, Any]], batch_size: int) -> list[list[tuple[float, Any]]]:
    size = batch_size if batch_size > 0 else DEFAULT_BATCH_SIZE
    return [list(items[index : index + size]) for index in range(0, len(items), size)]


def _persist_repairs(
    cur: Any,
    updates: Sequence[tuple[float, Any]],
    *,
    batch_size: int,
) -> int:
    written = 0
    for batch in _batched(updates, batch_size):
        if not batch:
            continue
        cur.executemany(_UPDATE_SQL, batch)
        written += len(batch)
    return written


def _run_repair(args: argparse.Namespace) -> dict:
    """Connect, guard the target DB on --apply, and never write on dry-run."""
    dsn = _build_aawm_admin_dsn()
    conn = psycopg.connect(dsn, row_factory=psycopg.rows.dict_row)
    try:
        cur = conn.cursor()
        try:
            current_db = _current_database_name(cur)
            if args.apply and current_db != args.target_db_name:
                raise SystemExit(
                    f"Refusing to apply session_history reference pricing repair "
                    f"on database {current_db!r}; expected {args.target_db_name!r}."
                )
            rows = _select_candidate_rows(cur, args)
            updates = [
                params
                for params in (_repair_update_params(row) for row in rows)
                if params is not None
            ]
            written = 0
            if args.apply:
                written = _persist_repairs(
                    cur,
                    updates,
                    batch_size=_update_batch_size(args),
                )
                commit = getattr(conn, "commit", None)
                if callable(commit):
                    commit()
            return {
                "mode": "apply" if args.apply else "dry_run",
                "current_database": current_db,
                "target_db_name": args.target_db_name,
                "scanned": len(rows),
                "eligible": len(updates),
                "written": written,
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
