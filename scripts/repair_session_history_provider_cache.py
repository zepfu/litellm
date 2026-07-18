#!/usr/bin/env python3
"""
Repair observability telemetry on existing session_history rows.

This is a best-effort repair for historical rows when the original proxy
spend-log source is unavailable locally. It derives provider, reasoning source,
provider-cache state, and git rollups from stored session_history rows plus
session_history_tool_activity, then writes the normalized fields back to the same
table.

Schema ownership note (RR-087):
  ``session_history`` DDL is migration-owned. This script does not run
  CREATE/ALTER/INDEX on every invocation. The shared
  ``litellm.integrations.aawm_agent_identity._ensure_session_history_schema``
  helper is a readiness gate only (no DDL). Operators who need an emergency
  bootstrap can pass ``--ensure-schema``, which applies the *same* SQL
  constants already defined in ``aawm_agent_identity`` rather than a second
  diverging catalog.

Target-database safety (RR-087):
  Before any mutating path (``--apply`` writes or ``--ensure-schema`` DDL),
  refuse to continue unless ``current_database()`` matches ``--target-db-name``
  (default ``aawm_tristore``). Inherited environments have previously pointed at
  the wrong database. Pure dry-run scans may still report the connected database
  without aborting.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

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
    _AAWM_SESSION_HISTORY_TOOL_ACTIVITY_INDEX_STATEMENTS,
    _AAWM_SESSION_HISTORY_TOOL_ACTIVITY_TABLE_SQL,
    _build_aawm_dsn,
    _compute_provider_cache_miss_cost_state,
    # Shared readiness helper (async, migration-owned no-op). Imported so this
    # script does not maintain a second "ensure schema" policy surface.
    _ensure_session_history_schema as _shared_ensure_session_history_schema,
    _normalize_provider_cache_family,
    _normalize_session_history_provider,
    _resolve_provider_cache_state,
    _safe_int,
)

DEFAULT_TARGET_DB_NAME = "aawm_tristore"

# Single source for cache-state fields written under both the generic
# ``usage_provider_cache_*`` namespace and the per-family
# ``{provider_family}_provider_cache_*`` namespace (RR-087 #4).
_PROVIDER_CACHE_METADATA_FIELDS: Tuple[str, ...] = (
    "attempted",
    "status",
    "miss",
    "miss_reason",
    "miss_token_count",
    "miss_cost_usd",
    "miss_cost_basis",
    "source",
)


def _build_aawm_admin_dsn() -> Optional[str]:
    direct_dsn = os.getenv("AAWM_DIRECT_DATABASE_URL")
    if direct_dsn and direct_dsn.strip():
        return direct_dsn.strip()
    return _build_aawm_dsn()


_REPAIR_UPDATE_SQL = """
UPDATE public.session_history
SET
    provider = %s,
    reasoning_tokens_reported = %s,
    reasoning_tokens_estimated = %s,
    reasoning_tokens_source = %s,
    provider_cache_attempted = %s,
    provider_cache_status = %s,
    provider_cache_miss = %s,
    provider_cache_miss_reason = %s,
    provider_cache_miss_token_count = %s,
    provider_cache_miss_cost_usd = %s,
    git_commit_count = %s,
    git_push_count = %s,
    metadata = %s::jsonb
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


def _provider_cache_metadata_key_pairs(
    *,
    provider_family: str,
    cache_state: Dict[str, Any],
) -> list[tuple[str, Any]]:
    """Build generic + per-family metadata keys from one field catalog."""
    pairs: list[tuple[str, Any]] = []
    for field_name in _PROVIDER_CACHE_METADATA_FIELDS:
        value = cache_state.get(field_name)
        pairs.append((f"usage_provider_cache_{field_name}", value))
        pairs.append((f"{provider_family}_provider_cache_{field_name}", value))
    return pairs


def _apply_cache_state_to_metadata(
    metadata: Dict[str, Any],
    *,
    provider_family: str,
    cache_state: Dict[str, Any],
) -> Dict[str, Any]:
    updated = dict(metadata)
    for key, value in _provider_cache_metadata_key_pairs(
        provider_family=provider_family,
        cache_state=cache_state,
    ):
        if value is None or value == "":
            updated.pop(key, None)
        else:
            updated[key] = value
    return updated


def _positive_int_or_none(value: Any) -> Optional[int]:
    normalized = _safe_int(value)
    if normalized is not None and normalized > 0:
        return normalized
    return None


def _resolve_reasoning_state(
    row: Dict[str, Any]
) -> tuple[Optional[int], Optional[int], str]:
    reported = _positive_int_or_none(row.get("reasoning_tokens_reported"))
    estimated = _positive_int_or_none(row.get("reasoning_tokens_estimated"))
    source = row.get("reasoning_tokens_source")
    reasoning_present = bool(
        row.get("reasoning_present") or row.get("thinking_signature_present")
    )

    if source == "provider_signature_present" and reported is not None:
        return reported, estimated, "provider_signature_present"
    if source == "provider_reported" and reported is not None:
        return reported, estimated, "provider_reported"
    if estimated is not None:
        return reported, estimated, "estimated_from_reasoning_text"
    if reasoning_present:
        return reported, estimated, "not_available"
    return reported, estimated, "not_applicable"


def _apply_reasoning_state_to_metadata(
    metadata: Dict[str, Any],
    *,
    reported: Optional[int],
    estimated: Optional[int],
    source: str,
) -> Dict[str, Any]:
    updated = dict(metadata)
    updated["usage_reasoning_tokens_source"] = source
    if reported is not None:
        updated["usage_reasoning_tokens_reported"] = reported
    else:
        updated.pop("usage_reasoning_tokens_reported", None)
    if estimated is not None:
        updated["usage_reasoning_tokens_estimated"] = estimated
    else:
        updated.pop("usage_reasoning_tokens_estimated", None)
    return updated


def _row_usage_object(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "prompt_tokens": int(row.get("input_tokens") or 0),
        "completion_tokens": int(row.get("output_tokens") or 0),
        "total_tokens": int(row.get("total_tokens") or 0),
        "cache_read_input_tokens": int(row.get("cache_read_input_tokens") or 0),
        "cache_creation_input_tokens": int(row.get("cache_creation_input_tokens") or 0),
        "cost": row.get("response_cost_usd"),
    }


def _row_has_complete_tool_activity(row: Dict[str, Any]) -> bool:
    """True when the tool_activity join produced complete rollup evidence."""
    if "has_tool_activity" in row:
        return bool(row.get("has_tool_activity"))
    # Nullable tool counts without the flag mean the join produced a row.
    if (
        row.get("tool_git_commit_count") is not None
        or row.get("tool_git_push_count") is not None
    ):
        return True
    return False


def _reconcile_git_counts(row: Dict[str, Any]) -> Tuple[int, int]:
    """Reconcile git counts only from complete tool_activity evidence.

    When tool_activity rows exist for the call, those aggregates are the
    authoritative source (so previously over-counted stored values can be
    lowered). When the join is empty/incomplete, leave the stored counts
    untouched rather than treating COALESCE(0) as real evidence.
    """
    stored_commit = int(row.get("git_commit_count") or 0)
    stored_push = int(row.get("git_push_count") or 0)
    if not _row_has_complete_tool_activity(row):
        return stored_commit, stored_push
    return (
        int(row.get("tool_git_commit_count") or 0),
        int(row.get("tool_git_push_count") or 0),
    )


def _bootstrap_session_history_schema_from_shared_sql(
    conn: psycopg.Connection,
) -> None:
    """Optional operator bootstrap using shared SQL constants only.

    Prefer migrations. This exists solely for ``--ensure-schema`` so the
    script does not own a diverging DDL catalog; statements come from
    ``aawm_agent_identity``.
    """
    with conn.cursor() as cur:
        cur.execute(_AAWM_SESSION_HISTORY_TABLE_SQL)
        cur.execute(_AAWM_SESSION_HISTORY_TOOL_ACTIVITY_TABLE_SQL)
        for statement in _AAWM_SESSION_HISTORY_ALTER_STATEMENTS:
            cur.execute(statement)
        for statement in _AAWM_SESSION_HISTORY_INDEX_STATEMENTS:
            cur.execute(statement)
        for statement in _AAWM_SESSION_HISTORY_TOOL_ACTIVITY_INDEX_STATEMENTS:
            cur.execute(statement)
    conn.commit()


def _acknowledge_shared_schema_policy() -> None:
    """Document dependency on the shared ensure helper without running DDL.

    The shared async helper is intentionally a no-op readiness gate
    (schema changes are migration-owned). Referencing it keeps this script
    coupled to that policy (RR-087 #3) instead of reintroducing a local
    ensure that mutates schema on every run (RR-087 #5). Emergency bootstrap
    remains opt-in via ``--ensure-schema``.
    """
    if _shared_ensure_session_history_schema is None:  # pragma: no cover
        raise RuntimeError("shared session_history schema helper is unavailable")


def _build_repair_row_update(row: Dict[str, Any]) -> Optional[Tuple[Any, ...]]:
    """Compute one repair UPDATE payload, or None when the row is already correct."""
    model = str(row.get("model") or "")
    metadata = _safe_json_metadata(row.get("metadata"))
    provider = _normalize_session_history_provider(
        row.get("provider"),
        model,
        metadata,
    )
    reported, estimated, reasoning_source = _resolve_reasoning_state(row)
    provider_family = _normalize_provider_cache_family(provider, model, metadata)

    cache_state = None
    if provider_family is not None:
        cache_state = _resolve_provider_cache_state(
            provider=provider,
            model=model,
            usage_obj=_row_usage_object(row),
            metadata=metadata,
            request_body=None,
        )
        if cache_state is not None:
            cache_state = dict(cache_state)
            cache_state.update(
                _compute_provider_cache_miss_cost_state(
                    provider_family=provider_family,
                    model=model,
                    usage_obj=_row_usage_object(row),
                    cache_state=cache_state,
                    metadata=metadata,
                    response_cost_usd=row.get("response_cost_usd"),
                )
            )

    updated_metadata = _apply_reasoning_state_to_metadata(
        metadata,
        reported=reported,
        estimated=estimated,
        source=reasoning_source,
    )
    if provider_family is not None and cache_state is not None:
        updated_metadata = _apply_cache_state_to_metadata(
            updated_metadata,
            provider_family=provider_family,
            cache_state=cache_state,
        )

    git_commit_count, git_push_count = _reconcile_git_counts(row)

    cache_attempted = bool(cache_state and cache_state.get("attempted"))
    cache_status = (
        cache_state.get("status") if cache_state else row.get("provider_cache_status")
    )
    cache_miss = bool(cache_state and cache_state.get("miss"))
    cache_miss_reason = (
        cache_state.get("miss_reason")
        if cache_state
        else row.get("provider_cache_miss_reason")
    )
    cache_miss_token_count = (
        cache_state.get("miss_token_count")
        if cache_state
        else row.get("provider_cache_miss_token_count")
    )
    cache_miss_cost_usd = (
        cache_state.get("miss_cost_usd")
        if cache_state
        else row.get("provider_cache_miss_cost_usd")
    )

    current_state = (
        row.get("provider"),
        _positive_int_or_none(row.get("reasoning_tokens_reported")),
        _positive_int_or_none(row.get("reasoning_tokens_estimated")),
        row.get("reasoning_tokens_source"),
        bool(row.get("provider_cache_attempted")),
        row.get("provider_cache_status"),
        bool(row.get("provider_cache_miss")),
        row.get("provider_cache_miss_reason"),
        row.get("provider_cache_miss_token_count"),
        row.get("provider_cache_miss_cost_usd"),
        int(row.get("git_commit_count") or 0),
        int(row.get("git_push_count") or 0),
        metadata,
    )
    new_state = (
        provider,
        reported,
        estimated,
        reasoning_source,
        cache_attempted,
        cache_status,
        cache_miss,
        cache_miss_reason,
        cache_miss_token_count,
        cache_miss_cost_usd,
        git_commit_count,
        git_push_count,
        updated_metadata,
    )
    if current_state == new_state:
        return None

    return (
        provider,
        reported,
        estimated,
        reasoning_source,
        cache_attempted,
        cache_status,
        cache_miss,
        cache_miss_reason,
        cache_miss_token_count,
        cache_miss_cost_usd,
        git_commit_count,
        git_push_count,
        json.dumps(updated_metadata),
        int(row["id"]),
    )


def _current_database_name(conn: psycopg.Connection) -> Optional[str]:
    with conn.cursor() as cur:
        cur.execute("SELECT current_database()")
        db_row = cur.fetchone()
    if isinstance(db_row, dict):
        return db_row.get("current_database")
    if db_row:
        return db_row[0]
    return None


def _fetch_repair_batch(
    conn: psycopg.Connection,
    *,
    cursor_id: int,
    args: argparse.Namespace,
) -> list[Dict[str, Any]]:
    params: list[Any] = [cursor_id]
    where_clauses = ["sh.id > %s"]
    if args.provider:
        where_clauses.append("sh.provider = %s")
        params.append(args.provider)
    if args.session_id:
        where_clauses.append("sh.session_id = %s")
        params.append(args.session_id)
    if args.cache_misses_only:
        where_clauses.append("sh.provider_cache_miss = TRUE")
    if args.missing_cache_miss_fields_only:
        where_clauses.append(
            """
            sh.provider_cache_miss = TRUE
            AND (
                sh.provider_cache_miss_token_count IS NULL
                OR sh.provider_cache_miss_cost_usd IS NULL
            )
            """
        )
    where_sql = " AND ".join(f"({clause})" for clause in where_clauses)
    query = f"""
        SELECT
            sh.id,
            sh.litellm_call_id,
            sh.session_id,
            sh.provider,
            sh.model,
            sh.input_tokens,
            sh.output_tokens,
            sh.total_tokens,
            sh.cache_read_input_tokens,
            sh.cache_creation_input_tokens,
            sh.provider_cache_attempted,
            sh.provider_cache_status,
            sh.provider_cache_miss,
            sh.provider_cache_miss_reason,
            sh.provider_cache_miss_token_count,
            sh.provider_cache_miss_cost_usd,
            sh.response_cost_usd,
            sh.reasoning_tokens_reported,
            sh.reasoning_tokens_estimated,
            sh.reasoning_tokens_source,
            sh.reasoning_present,
            sh.thinking_signature_present,
            sh.git_commit_count,
            sh.git_push_count,
            tool_activity.git_commit_count AS tool_git_commit_count,
            tool_activity.git_push_count AS tool_git_push_count,
            (tool_activity.litellm_call_id IS NOT NULL) AS has_tool_activity,
            sh.metadata
        FROM public.session_history sh
        LEFT JOIN (
            SELECT
                litellm_call_id,
                SUM(git_commit_count)::integer AS git_commit_count,
                SUM(git_push_count)::integer AS git_push_count
            FROM public.session_history_tool_activity
            GROUP BY litellm_call_id
        ) tool_activity ON tool_activity.litellm_call_id = sh.litellm_call_id
        WHERE {where_sql}
        ORDER BY sh.id ASC
        LIMIT {int(args.batch_size)}
    """
    with conn.cursor() as cur:
        cur.execute(query, params)
        return list(cur.fetchall())


def _record_provider_status(
    provider_status_counts: Dict[str, Dict[str, int]],
    row: Dict[str, Any],
) -> None:
    model = str(row.get("model") or "")
    metadata = _safe_json_metadata(row.get("metadata"))
    provider = _normalize_session_history_provider(row.get("provider"), model, metadata)
    provider_family = _normalize_provider_cache_family(provider, model, metadata)
    if provider_family is None:
        return
    provisional_cache = _resolve_provider_cache_state(
        provider=provider,
        model=model,
        usage_obj=_row_usage_object(row),
        metadata=metadata,
        request_body=None,
    )
    if provisional_cache is None:
        return
    provisional_cache = dict(provisional_cache)
    provisional_cache.update(
        _compute_provider_cache_miss_cost_state(
            provider_family=provider_family,
            model=model,
            usage_obj=_row_usage_object(row),
            cache_state=provisional_cache,
            metadata=metadata,
            response_cost_usd=row.get("response_cost_usd"),
        )
    )
    provider_status_counts.setdefault(provider_family, {})
    status_key = str(provisional_cache.get("status") or "unknown")
    provider_status_counts[provider_family][status_key] = (
        provider_status_counts[provider_family].get(status_key, 0) + 1
    )


def _process_repair_batch(
    rows: list[Dict[str, Any]],
    *,
    args: argparse.Namespace,
    repaired_rows: int,
    provider_status_counts: Dict[str, Dict[str, int]],
) -> tuple[int, int, list[Tuple[Any, ...]]]:
    updates: list[Tuple[Any, ...]] = []
    scanned = 0
    cursor_id = 0
    for row in rows:
        scanned += 1
        cursor_id = int(row["id"])
        _record_provider_status(provider_status_counts, row)
        update = _build_repair_row_update(row)
        if update is None:
            continue
        updates.append(update)
        if args.limit is not None and repaired_rows + len(updates) >= args.limit:
            break
    return scanned, cursor_id, updates


def _run_repair(args: argparse.Namespace) -> Dict[str, Any]:
    dsn = _build_aawm_admin_dsn()
    if not dsn:
        raise RuntimeError("AAWM/tristore database configuration is missing")

    scanned_rows = 0
    repaired_rows = 0
    provider_status_counts: Dict[str, Dict[str, int]] = {}
    cursor_id = 0
    schema_bootstrapped = False
    database_name: Optional[str] = None

    with psycopg.connect(dsn, row_factory=psycopg.rows.dict_row) as conn:
        if int(args.batch_size) <= 0:
            raise SystemExit("--batch-size must be a positive integer")
        if args.limit is not None and int(args.limit) < 0:
            raise SystemExit("--limit must be >= 0 when provided")

        # Exact target-DB guard for any mutation path (RR-087 #1). Pure dry-run
        # scans may still report the connected database without aborting.
        database_name = _current_database_name(conn)
        mutates = bool(args.apply or args.ensure_schema)
        if mutates and database_name != args.target_db_name:
            raise SystemExit(
                f"Refusing to run against {database_name!r}; "
                f"expected {args.target_db_name!r}."
            )

        # Default: no DDL. Align with shared migration-owned schema policy.
        _acknowledge_shared_schema_policy()
        if args.ensure_schema:
            _bootstrap_session_history_schema_from_shared_sql(conn)
            schema_bootstrapped = True

        while True:
            rows = _fetch_repair_batch(conn, cursor_id=cursor_id, args=args)
            if not rows:
                break

            scanned, cursor_id, updates = _process_repair_batch(
                rows,
                args=args,
                repaired_rows=repaired_rows,
                provider_status_counts=provider_status_counts,
            )
            scanned_rows += scanned

            if args.apply and updates:
                with conn.cursor() as cur:
                    cur.executemany(_REPAIR_UPDATE_SQL, updates)
                conn.commit()
            repaired_rows += len(updates)

            if args.limit is not None and repaired_rows >= args.limit:
                break

    return {
        "mode": "apply" if args.apply else "dry_run",
        "database": database_name,
        "target_db_name": args.target_db_name,
        "ensure_schema": bool(args.ensure_schema),
        "schema_bootstrapped": schema_bootstrapped,
        "scanned_rows": scanned_rows,
        "repaired_rows": repaired_rows,
        "provider_status_counts": provider_status_counts,
    }


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Repair provider-cache telemetry on existing session_history rows. "
            "Schema DDL is migration-owned; this script does not alter schema "
            "unless --ensure-schema is explicitly passed. --apply and "
            "--ensure-schema both require current_database() to match "
            "--target-db-name (default dry-run is scan-only)."
        )
    )
    parser.add_argument("--apply", action="store_true", help="Persist repaired values.")
    parser.add_argument("--provider", help="Restrict to a single provider.")
    parser.add_argument("--session-id", help="Restrict to a single session_id.")
    parser.add_argument(
        "--cache-misses-only",
        action="store_true",
        help="Restrict to rows already marked as provider cache misses.",
    )
    parser.add_argument(
        "--missing-cache-miss-fields-only",
        action="store_true",
        help="Restrict to provider cache misses missing token count or miss cost.",
    )
    parser.add_argument("--batch-size", type=int, default=500, help="Rows per batch.")
    parser.add_argument(
        "--limit",
        type=int,
        help="Maximum number of rows to update after scanning.",
    )
    parser.add_argument(
        "--target-db-name",
        default=DEFAULT_TARGET_DB_NAME,
        help=(
            "Database name that must be returned by current_database() before any "
            "--apply writes or --ensure-schema DDL. Default: aawm_tristore."
        ),
    )
    parser.add_argument(
        "--ensure-schema",
        action="store_true",
        help=(
            "Opt-in emergency bootstrap: apply shared session_history / "
            "session_history_tool_activity SQL constants from aawm_agent_identity "
            "(CREATE/ALTER/INDEX). Default is off because schema is migration-owned "
            "and unconditional DDL takes ACCESS EXCLUSIVE lock windows on every run."
        ),
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    result = _run_repair(args)
    print(json.dumps(result, indent=2, default=str))  # noqa: T201
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
