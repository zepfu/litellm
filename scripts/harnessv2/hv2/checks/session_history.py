"""Read-only session_history export. enabled:false stays a skip."""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any, Mapping
from urllib.parse import unquote, urlparse

_DEFAULT_LIMIT = 50
_MAX_LIMIT = 200
_COLUMNS = (
    "id",
    "session_id",
    "trace_id",
    "litellm_call_id",
    "inbound_model_alias",
    "provider",
    "model",
    "start_time",
    "end_time",
    "metadata",
)
_QUERY = """
SELECT
    id,
    session_id,
    trace_id,
    litellm_call_id,
    inbound_model_alias,
    provider,
    model,
    start_time,
    end_time,
    metadata
FROM public.session_history
WHERE
    (%(session_id)s::text IS NULL OR session_id = %(session_id)s)
    AND (%(trace_id)s::text IS NULL OR trace_id = %(trace_id)s)
    AND (%(litellm_call_id)s::text IS NULL OR litellm_call_id = %(litellm_call_id)s)
    AND (%(inbound_model_alias)s::text IS NULL OR inbound_model_alias = %(inbound_model_alias)s)
ORDER BY id DESC
LIMIT %(limit)s
"""


def session_history_result(
    config: Mapping[str, Any], query: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """Export matching session_history rows. One-argument callers stay skipped."""

    checks = config.get("checks") if isinstance(config.get("checks"), dict) else {}
    spec = checks.get("session_history")
    if not isinstance(spec, dict) or not spec.get("enabled"):
        return {"enabled": False, "skipped": True}
    dsn, database = _resolve_dsn(spec)
    if not dsn:
        return {
            "enabled": True,
            "status": "unavailable",
            "database": None,
            "skipped": False,
        }
    try:
        rows = _read_rows(dsn, spec, query)
    except Exception as exc:
        return {
            "enabled": True,
            "status": "unavailable",
            "database": database,
            "skipped": False,
            "error": str(exc),
        }
    return {
        "enabled": True,
        "status": "ok",
        "database": database,
        "skipped": False,
        "rows": rows,
        "wait_accounting": _latest_wait_accounting(rows),
    }


def _resolve_dsn(spec: Mapping[str, Any]) -> tuple[str | None, str | None]:
    direct = spec.get("database_url")
    if isinstance(direct, str) and direct.strip():
        dsn = direct.strip()
        return dsn, _database_name(dsn)
    env_name = spec.get("database_url_env")
    if isinstance(env_name, str) and env_name.strip():
        raw = os.environ.get(env_name.strip())
        if isinstance(raw, str) and raw.strip():
            dsn = raw.strip()
            return dsn, _database_name(dsn)
    return None, None


def _database_name(dsn: str) -> str | None:
    if "://" in dsn:
        name = unquote(urlparse(dsn).path.lstrip("/"))
        return name or None
    for part in dsn.replace(";", " ").split():
        if part.startswith("dbname="):
            return part.split("=", 1)[1] or None
    return None


def _limit(spec: Mapping[str, Any]) -> int:
    raw = spec.get("limit", spec.get("row_limit", _DEFAULT_LIMIT))
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return _DEFAULT_LIMIT
    if value < 1:
        return 1
    return min(value, _MAX_LIMIT)


def _filters(query: Mapping[str, Any] | None) -> dict[str, str | None]:
    source = query if isinstance(query, Mapping) else {}
    filters: dict[str, str | None] = {}
    for key in ("session_id", "trace_id", "litellm_call_id", "inbound_model_alias"):
        value = source.get(key)
        filters[key] = str(value) if value not in (None, "") else None
    return filters


def _read_rows(
    dsn: str,
    spec: Mapping[str, Any],
    query: Mapping[str, Any] | None,
) -> list[dict[str, Any]]:
    import psycopg
    import psycopg.rows

    timeout = spec.get("timeout_seconds")
    try:
        timeout_s = float(timeout) if timeout is not None else None
    except (TypeError, ValueError):
        timeout_s = None
    params = {**_filters(query), "limit": _limit(spec)}
    connect_kwargs: dict[str, Any] = {
        "row_factory": psycopg.rows.dict_row,
        "autocommit": True,
    }
    if timeout_s is not None and timeout_s > 0:
        connect_kwargs["connect_timeout"] = max(1, int(timeout_s))
    with psycopg.connect(dsn, **connect_kwargs) as conn:
        conn.read_only = True
        with conn.cursor() as cur:
            if timeout_s is not None and timeout_s > 0:
                cur.execute("SET statement_timeout = %s", (f"{int(timeout_s * 1000)}ms",))
            cur.execute(_QUERY, params)
            fetched = cur.fetchall()
    return [_project_row(dict(row)) for row in fetched]


def _project_row(row: Mapping[str, Any]) -> dict[str, Any]:
    metadata = row.get("metadata") if isinstance(row.get("metadata"), Mapping) else {}
    start = row.get("start_time")
    end = row.get("end_time")
    projected = {column: _jsonable(row.get(column)) for column in _COLUMNS if column != "metadata"}
    projected["metadata"] = _jsonable(dict(metadata))
    projected["completion_start_time"] = _jsonable(
        metadata.get("completion_start_time") or metadata.get("completion_start")
    )
    projected["duration_seconds"] = _duration_seconds(start, end)
    return projected


def _duration_seconds(start: Any, end: Any) -> float | None:
    if start is None or end is None:
        return None
    if isinstance(start, datetime) and isinstance(end, datetime):
        start_dt = start if start.tzinfo else start.replace(tzinfo=timezone.utc)
        end_dt = end if end.tzinfo else end.replace(tzinfo=timezone.utc)
        seconds = (end_dt - start_dt).total_seconds()
        return seconds if seconds >= 0 else None
    return None


def _latest_wait_accounting(rows: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """One snapshot per call: the highest metadata.wait_accounting sequence."""

    latest: dict[str, dict[str, Any]] = {}
    for row in rows:
        metadata = row.get("metadata") if isinstance(row.get("metadata"), Mapping) else {}
        snapshot = metadata.get("wait_accounting")
        if not isinstance(snapshot, Mapping):
            continue
        call_id = str(row.get("litellm_call_id") or "")
        if not call_id:
            continue
        try:
            sequence = int(snapshot.get("sequence"))
        except (TypeError, ValueError):
            continue
        previous = latest.get(call_id)
        if previous is None or sequence >= int(previous["sequence"]):
            latest[call_id] = {
                "litellm_call_id": call_id,
                "sequence": sequence,
                "started_at": snapshot.get("started_at"),
                "updated_at": snapshot.get("updated_at"),
                "status": snapshot.get("status"),
                "durations_ms": dict(snapshot.get("durations_ms") or {}),
                "counts": dict(snapshot.get("counts") or {}),
                "current_waits": list(snapshot.get("current_waits") or []),
            }
    return list(latest.values())


def _jsonable(value: Any) -> Any:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc).isoformat()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    return value
