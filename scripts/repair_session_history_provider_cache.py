#!/usr/bin/env python3
"""
Repair provider-cache telemetry on existing session_history rows.

This is a best-effort repair for historical rows when the original proxy spend-log
source is unavailable locally. It derives provider cache state from the stored
session_history provider/model/cache counters plus persisted metadata and writes the
normalized fields back to the same table.
"""

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
    _build_aawm_dsn,
    _compute_provider_cache_miss_cost_state,
    _normalize_provider_cache_family,
    _resolve_provider_cache_state,
)


_REPAIR_UPDATE_SQL = """
UPDATE public.session_history
SET
    provider_cache_attempted = %s,
    provider_cache_status = %s,
    provider_cache_miss = %s,
    provider_cache_miss_reason = %s,
    provider_cache_miss_token_count = %s,
    provider_cache_miss_cost_usd = %s,
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


def _apply_cache_state_to_metadata(
    metadata: Dict[str, Any],
    *,
    provider_family: str,
    cache_state: Dict[str, Any],
) -> Dict[str, Any]:
    updated = dict(metadata)
    updated["usage_provider_cache_attempted"] = cache_state["attempted"]
    updated["usage_provider_cache_status"] = cache_state["status"]
    updated["usage_provider_cache_miss"] = cache_state["miss"]

    if cache_state.get("miss_reason"):
        updated["usage_provider_cache_miss_reason"] = cache_state["miss_reason"]
    else:
        updated.pop("usage_provider_cache_miss_reason", None)

    if cache_state.get("miss_token_count") is not None:
        updated["usage_provider_cache_miss_token_count"] = cache_state["miss_token_count"]
    else:
        updated.pop("usage_provider_cache_miss_token_count", None)

    if cache_state.get("miss_cost_usd") is not None:
        updated["usage_provider_cache_miss_cost_usd"] = cache_state["miss_cost_usd"]
    else:
        updated.pop("usage_provider_cache_miss_cost_usd", None)

    if cache_state.get("miss_cost_basis"):
        updated["usage_provider_cache_miss_cost_basis"] = cache_state["miss_cost_basis"]
    else:
        updated.pop("usage_provider_cache_miss_cost_basis", None)

    if cache_state.get("source"):
        updated["usage_provider_cache_source"] = cache_state["source"]
    else:
        updated.pop("usage_provider_cache_source", None)

    for key, value in (
        (f"{provider_family}_provider_cache_attempted", cache_state["attempted"]),
        (f"{provider_family}_provider_cache_status", cache_state["status"]),
        (f"{provider_family}_provider_cache_miss", cache_state["miss"]),
        (f"{provider_family}_provider_cache_miss_reason", cache_state.get("miss_reason")),
        (
            f"{provider_family}_provider_cache_miss_token_count",
            cache_state.get("miss_token_count"),
        ),
        (
            f"{provider_family}_provider_cache_miss_cost_usd",
            cache_state.get("miss_cost_usd"),
        ),
        (
            f"{provider_family}_provider_cache_miss_cost_basis",
            cache_state.get("miss_cost_basis"),
        ),
        (f"{provider_family}_provider_cache_source", cache_state.get("source")),
    ):
        if value is None or value == "":
            updated.pop(key, None)
        else:
            updated[key] = value

    return updated


def _row_usage_object(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "prompt_tokens": int(row.get("input_tokens") or 0),
        "completion_tokens": int(row.get("output_tokens") or 0),
        "total_tokens": int(row.get("total_tokens") or 0),
        "cache_read_input_tokens": int(row.get("cache_read_input_tokens") or 0),
        "cache_creation_input_tokens": int(row.get("cache_creation_input_tokens") or 0),
    }


def _ensure_session_history_schema(conn: psycopg.Connection) -> None:
    with conn.cursor() as cur:
        cur.execute(_AAWM_SESSION_HISTORY_TABLE_SQL)
        for statement in _AAWM_SESSION_HISTORY_ALTER_STATEMENTS:
            cur.execute(statement)
        for statement in _AAWM_SESSION_HISTORY_INDEX_STATEMENTS:
            cur.execute(statement)
    conn.commit()


def _run_repair(args: argparse.Namespace) -> Dict[str, Any]:
    dsn = _build_aawm_dsn()
    if not dsn:
        raise RuntimeError("AAWM/tristore database configuration is missing")

    scanned_rows = 0
    repaired_rows = 0
    provider_status_counts: Dict[str, Dict[str, int]] = {}
    cursor_id = 0

    with psycopg.connect(dsn, row_factory=psycopg.rows.dict_row) as conn:
        _ensure_session_history_schema(conn)
        while True:
            params: list[Any] = [cursor_id]
            if args.provider:
                params.append(args.provider)
            if args.session_id:
                params.append(args.session_id)

            query = f"""
                SELECT
                    id,
                    litellm_call_id,
                    session_id,
                    provider,
                    model,
                    input_tokens,
                    output_tokens,
                    total_tokens,
                    cache_read_input_tokens,
                    cache_creation_input_tokens,
                    provider_cache_attempted,
                    provider_cache_status,
                    provider_cache_miss,
                    provider_cache_miss_reason,
                    provider_cache_miss_token_count,
                    provider_cache_miss_cost_usd,
                    metadata
                FROM public.session_history
                WHERE id > %s{' AND provider = %s' if args.provider else ''}{' AND session_id = %s' if args.session_id else ''}
                ORDER BY id ASC
                LIMIT {int(args.batch_size)}
            """
            with conn.cursor() as cur:
                cur.execute(query, params)
                rows = cur.fetchall()
            if not rows:
                break

            updates = []
            for row in rows:
                scanned_rows += 1
                cursor_id = int(row["id"])
                provider = row.get("provider")
                model = str(row.get("model") or "")
                metadata = _safe_json_metadata(row.get("metadata"))
                provider_family = _normalize_provider_cache_family(provider, model, metadata)
                if provider_family is None:
                    continue

                cache_state = _resolve_provider_cache_state(
                    provider=provider,
                    model=model,
                    usage_obj=_row_usage_object(row),
                    metadata=metadata,
                    request_body=None,
                )
                if cache_state is None:
                    continue
                cache_state = dict(cache_state)
                cache_state.update(
                    _compute_provider_cache_miss_cost_state(
                        provider_family=provider_family,
                        model=model,
                        usage_obj=_row_usage_object(row),
                        cache_state=cache_state,
                        metadata=metadata,
                    )
                )

                provider_status_counts.setdefault(provider_family, {})
                status_key = str(cache_state.get("status") or "unknown")
                provider_status_counts[provider_family][status_key] = (
                    provider_status_counts[provider_family].get(status_key, 0) + 1
                )

                updated_metadata = _apply_cache_state_to_metadata(
                    metadata,
                    provider_family=provider_family,
                    cache_state=cache_state,
                )

                current_state = (
                    bool(row.get("provider_cache_attempted")),
                    row.get("provider_cache_status"),
                    bool(row.get("provider_cache_miss")),
                    row.get("provider_cache_miss_reason"),
                    row.get("provider_cache_miss_token_count"),
                    row.get("provider_cache_miss_cost_usd"),
                    metadata,
                )
                new_state = (
                    bool(cache_state.get("attempted")),
                    cache_state.get("status"),
                    bool(cache_state.get("miss")),
                    cache_state.get("miss_reason"),
                    cache_state.get("miss_token_count"),
                    cache_state.get("miss_cost_usd"),
                    updated_metadata,
                )
                if current_state == new_state:
                    continue

                updates.append(
                    (
                        bool(cache_state.get("attempted")),
                        cache_state.get("status"),
                        bool(cache_state.get("miss")),
                        cache_state.get("miss_reason"),
                        cache_state.get("miss_token_count"),
                        cache_state.get("miss_cost_usd"),
                        json.dumps(updated_metadata),
                        int(row["id"]),
                    )
                )

                if args.limit is not None and repaired_rows + len(updates) >= args.limit:
                    break

            if args.apply and updates:
                with conn.cursor() as cur:
                    cur.executemany(_REPAIR_UPDATE_SQL, updates)
                conn.commit()
            repaired_rows += len(updates)

            if args.limit is not None and repaired_rows >= args.limit:
                break

        return {
            "mode": "apply" if args.apply else "dry_run",
            "scanned_rows": scanned_rows,
            "repaired_rows": repaired_rows,
            "provider_status_counts": provider_status_counts,
        }


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Repair provider-cache telemetry on existing session_history rows."
    )
    parser.add_argument("--apply", action="store_true", help="Persist repaired values.")
    parser.add_argument("--provider", help="Restrict to a single provider.")
    parser.add_argument("--session-id", help="Restrict to a single session_id.")
    parser.add_argument("--batch-size", type=int, default=500, help="Rows per batch.")
    parser.add_argument(
        "--limit",
        type=int,
        help="Maximum number of rows to update after scanning.",
    )
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()
    result = _run_repair(args)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
