#!/usr/bin/env python3
"""
Repair observability telemetry on existing session_history rows.

This is a best-effort repair for historical rows when the original proxy
spend-log source is unavailable locally. It derives provider, reasoning source,
provider-cache state, and git rollups from stored session_history rows plus
session_history_tool_activity, then writes the normalized fields back to the same
table.
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
    _AAWM_SESSION_HISTORY_TOOL_ACTIVITY_INDEX_STATEMENTS,
    _AAWM_SESSION_HISTORY_TOOL_ACTIVITY_TABLE_SQL,
    _build_aawm_dsn,
    _compute_provider_cache_miss_cost_state,
    _normalize_provider_cache_family,
    _normalize_session_history_provider,
    _resolve_provider_cache_state,
    _safe_int,
)


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


def _positive_int_or_none(value: Any) -> Optional[int]:
    normalized = _safe_int(value)
    if normalized is not None and normalized > 0:
        return normalized
    return None


def _resolve_reasoning_state(row: Dict[str, Any]) -> tuple[Optional[int], Optional[int], str]:
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
    }


def _ensure_session_history_schema(conn: psycopg.Connection) -> None:
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
                    sh.reasoning_tokens_reported,
                    sh.reasoning_tokens_estimated,
                    sh.reasoning_tokens_source,
                    sh.reasoning_present,
                    sh.thinking_signature_present,
                    sh.git_commit_count,
                    sh.git_push_count,
                    COALESCE(tool_activity.git_commit_count, 0) AS tool_git_commit_count,
                    COALESCE(tool_activity.git_push_count, 0) AS tool_git_push_count,
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
                WHERE sh.id > %s{' AND sh.provider = %s' if args.provider else ''}{' AND sh.session_id = %s' if args.session_id else ''}
                ORDER BY sh.id ASC
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
                            )
                        )

                        provider_status_counts.setdefault(provider_family, {})
                        status_key = str(cache_state.get("status") or "unknown")
                        provider_status_counts[provider_family][status_key] = (
                            provider_status_counts[provider_family].get(status_key, 0) + 1
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

                git_commit_count = max(
                    int(row.get("git_commit_count") or 0),
                    int(row.get("tool_git_commit_count") or 0),
                )
                git_push_count = max(
                    int(row.get("git_push_count") or 0),
                    int(row.get("tool_git_push_count") or 0),
                )

                cache_attempted = bool(
                    cache_state and cache_state.get("attempted")
                )
                cache_status = cache_state.get("status") if cache_state else row.get("provider_cache_status")
                cache_miss = bool(cache_state and cache_state.get("miss"))
                cache_miss_reason = (
                    cache_state.get("miss_reason") if cache_state else row.get("provider_cache_miss_reason")
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
                    continue

                updates.append(
                    (
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
