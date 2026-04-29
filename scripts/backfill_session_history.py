#!/usr/bin/env python3
"""
Backfill session_history rows from historical LiteLLM_SpendLogs.

This reconstructs session-history records using the same normalization logic as the
AAWM callback and can optionally patch historical Langfuse trace tags from the same
derived tag set.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
import sys
from decimal import Decimal
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from http.client import RemoteDisconnected
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple
from urllib.parse import urlencode, urlsplit, urlunsplit
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

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

# Historical repairs should be reproducible against the bundled/promoted cost map
# unless an operator explicitly opts into a different source.
os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")

from litellm.caching import DualCache
from litellm.integrations.aawm_agent_identity import (
    _build_aawm_dsn,
    _build_session_history_record_from_langfuse_trace_observation,
    _build_session_history_record_from_spend_log_row,
    _derive_request_tags_from_langfuse_metadata,
    _derive_langfuse_trace_tags_from_spend_log_row,
    _ensure_session_history_schema,
    _extract_repository_identity_from_metadata_sources,
    _extract_tenant_identity_from_metadata_sources,
    _persist_session_history_records,
    _safe_float,
    _safe_int,
)
from litellm.proxy.utils import PrismaClient, ProxyLogging
from litellm.secret_managers.main import get_secret_str

_SOURCE_DB_ENV_VARS = ("DATABASE_URL", "DIRECT_URL")
_LANGFUSE_DB_ENV_VARS = ("LANGFUSE_DATABASE_URL", "AAWM_LANGFUSE_DATABASE_URL")
_CLICKHOUSE_URL_ENV_VARS = ("CLICKHOUSE_URL", "LANGFUSE_CLICKHOUSE_URL")
_CLICKHOUSE_USER_ENV_VARS = ("CLICKHOUSE_USER", "LANGFUSE_CLICKHOUSE_USER")
_CLICKHOUSE_PASSWORD_ENV_VARS = ("CLICKHOUSE_PASSWORD", "LANGFUSE_CLICKHOUSE_PASSWORD")
_SESSION_HISTORY_POOL: Optional[asyncpg.Pool] = None
_GEMINI_CONTROL_PLANE_METHODS = frozenset(
    (
        "loadCodeAssist",
        "listExperiments",
        "retrieveUserQuota",
        "fetchAdminControls",
    )
)
_GEMINI_CONTROL_PLANE_METHODS_BY_LOWER = {
    method.lower(): method for method in _GEMINI_CONTROL_PLANE_METHODS
}


def _clean_secret(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    cleaned = value.strip()
    if len(cleaned) >= 2 and cleaned[0] == cleaned[-1] and cleaned[0] in {'"', "'"}:
        cleaned = cleaned[1:-1].strip()
    return cleaned or None


def _get_first_secret(names: Sequence[str]) -> Optional[str]:
    for name in names:
        value = _clean_secret(get_secret_str(name))
        if value:
            return value
    return None


def _parse_optional_datetime(value: Optional[str]) -> Optional[datetime]:
    if value is None:
        return None
    normalized = value.strip().replace("Z", "+00:00")
    return datetime.fromisoformat(normalized)


def _normalize_langfuse_host(host: Optional[str]) -> str:
    cleaned = _clean_secret(host) or "http://127.0.0.1:3000"
    normalized = cleaned.rstrip("/")
    if not normalized.startswith(("http://", "https://")):
        normalized = f"http://{normalized}"

    for internal_host in ("aawm-langfuse-web", "langfuse-web"):
        normalized = normalized.replace(internal_host, "127.0.0.1")

    return normalized


def _derive_langfuse_database_url(
    explicit_url: Optional[str],
    *,
    target_dsn: Optional[str],
) -> Optional[str]:
    explicit = _clean_secret(explicit_url) or _get_first_secret(_LANGFUSE_DB_ENV_VARS)
    if explicit:
        return explicit
    if not target_dsn:
        return None

    parsed = urlsplit(target_dsn)
    if not parsed.scheme or not parsed.netloc:
        return None
    return urlunsplit((parsed.scheme, parsed.netloc, "/langfuse", parsed.query, parsed.fragment))


def _maybe_decimal_to_float(value: Any) -> Any:
    if isinstance(value, Decimal):
        return float(value)
    return value


def _normalize_clickhouse_http_url(url: Optional[str]) -> str:
    cleaned = _clean_secret(url) or "http://127.0.0.1:8123"
    normalized = cleaned.rstrip("/")
    if not normalized.startswith(("http://", "https://")):
        normalized = f"http://{normalized}"
    normalized = normalized.replace("clickhouse", "127.0.0.1")
    return normalized


def _resolve_clickhouse_config(args: argparse.Namespace) -> Dict[str, str]:
    url = _normalize_clickhouse_http_url(
        getattr(args, "clickhouse_url", None)
        or _get_first_secret(_CLICKHOUSE_URL_ENV_VARS)
    )
    user = (
        _clean_secret(getattr(args, "clickhouse_user", None))
        or _get_first_secret(_CLICKHOUSE_USER_ENV_VARS)
        or "clickhouse"
    )
    password = (
        _clean_secret(getattr(args, "clickhouse_password", None))
        or _get_first_secret(_CLICKHOUSE_PASSWORD_ENV_VARS)
        or "clickhouse"
    )
    return {"url": url, "user": user, "password": password}


def _parse_clickhouse_value(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    if not stripped:
        return value
    try:
        return json.loads(stripped)
    except (json.JSONDecodeError, TypeError, ValueError):
        return value


def _coerce_row_to_dict(row: Any) -> Dict[str, Any]:
    if isinstance(row, dict):
        return row
    if hasattr(row, "model_dump"):
        return row.model_dump()
    if hasattr(row, "dict"):
        return row.dict()
    if hasattr(row, "__dict__"):
        return {
            key: value
            for key, value in vars(row).items()
            if not key.startswith("_")
        }
    raise TypeError(f"Unsupported spend log row type: {type(row)!r}")


def _build_source_where(args: argparse.Namespace) -> Dict[str, Any]:
    where: Dict[str, Any] = {}

    if args.request_id:
        where["request_id"] = args.request_id

    if args.trace_id:
        where["session_id"] = args.trace_id

    if args.provider:
        where["custom_llm_provider"] = args.provider

    if args.model:
        where["model"] = args.model

    if args.status:
        where["status"] = args.status

    if args.from_start_time or args.to_start_time:
        start_time_filter: Dict[str, Any] = {}
        if args.from_start_time:
            start_time_filter["gte"] = _parse_optional_datetime(args.from_start_time)
        if args.to_start_time:
            start_time_filter["lte"] = _parse_optional_datetime(args.to_start_time)
        where["startTime"] = start_time_filter

    return where


async def _get_session_history_pool() -> asyncpg.Pool:
    global _SESSION_HISTORY_POOL

    if _SESSION_HISTORY_POOL is not None:
        return _SESSION_HISTORY_POOL

    dsn = _build_aawm_dsn()
    if not dsn:
        raise RuntimeError("AAWM/tristore database configuration is missing")

    _SESSION_HISTORY_POOL = await asyncpg.create_pool(
        dsn=dsn,
        min_size=1,
        max_size=4,
        command_timeout=30,
    )
    return _SESSION_HISTORY_POOL


async def _ensure_session_history_schema_with_pool(pool: asyncpg.Pool) -> None:
    async with pool.acquire() as conn:
        await _ensure_session_history_schema(conn)


async def _iter_spend_logs(
    prisma_client: PrismaClient,
    *,
    where: Dict[str, Any],
    batch_size: int,
    limit: Optional[int],
) -> Iterable[List[Dict[str, Any]]]:
    fetched = 0
    skip = 0

    while True:
        take = batch_size
        if limit is not None:
            remaining = limit - fetched
            if remaining <= 0:
                break
            take = min(take, remaining)

        rows = await prisma_client.db.litellm_spendlogs.find_many(
            where=where or None,
            order={"startTime": "asc"},
            skip=skip,
            take=take,
        )
        if not rows:
            break

        batch = [_coerce_row_to_dict(row) for row in rows]
        fetched += len(batch)
        skip += len(batch)
        yield batch


async def _get_existing_call_ids(call_ids: Sequence[str]) -> Set[str]:
    if not call_ids:
        return set()

    pool = await _get_session_history_pool()
    await _ensure_session_history_schema_with_pool(pool)
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT litellm_call_id FROM public.session_history WHERE litellm_call_id = ANY($1::text[])",
            list(call_ids),
        )
    return {
        str(row["litellm_call_id"])
        for row in rows
        if row.get("litellm_call_id") is not None
    }


class LangfuseTraceTagBackfiller:
    def __init__(self, *, host: str, public_key: str, secret_key: str) -> None:
        from langfuse import Langfuse

        normalized_host = _normalize_langfuse_host(host)

        self.host = normalized_host
        self.public_key = public_key
        self.secret_key = secret_key
        self._auth_header = "Basic " + base64.b64encode(
            f"{public_key}:{secret_key}".encode("utf-8")
        ).decode("ascii")
        self._client = Langfuse(
            public_key=public_key,
            secret_key=secret_key,
            host=normalized_host,
        )

    def _fetch_trace(self, trace_id: str) -> Optional[Dict[str, Any]]:
        request = Request(
            url=f"{self.host}/api/public/traces/{trace_id}",
            headers={"Authorization": self._auth_header},
            method="GET",
        )
        last_error: Optional[Exception] = None
        for _attempt in range(3):
            try:
                with urlopen(request, timeout=10) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                break
            except HTTPError as exc:
                if exc.code == 404:
                    return None
                last_error = exc
            except (URLError, RemoteDisconnected) as exc:
                last_error = exc
        else:
            if last_error is not None:
                raise last_error
            return None

        if isinstance(payload, dict):
            return payload
        return None

    def patch_tags(self, trace_id: str, tags: Sequence[str]) -> Dict[str, Any]:
        existing_trace = self._fetch_trace(trace_id)
        if existing_trace is None:
            return {"status": "missing_trace", "added": 0}

        existing_tags = existing_trace.get("tags") or []
        if not isinstance(existing_tags, list):
            existing_tags = []

        merged_tags = sorted(
            {
                str(tag)
                for tag in [*existing_tags, *tags]
                if isinstance(tag, str) and tag.strip()
            }
        )
        existing_tag_set = {
            str(tag) for tag in existing_tags if isinstance(tag, str) and tag.strip()
        }
        added_count = len(set(merged_tags) - existing_tag_set)

        if added_count == 0:
            return {"status": "unchanged", "added": 0}

        trace_kwargs: Dict[str, Any] = {"id": trace_id, "tags": merged_tags}
        trace_name = existing_trace.get("name")
        if isinstance(trace_name, str) and trace_name.strip():
            trace_kwargs["name"] = trace_name
        session_id = existing_trace.get("sessionId") or existing_trace.get("session_id")
        if isinstance(session_id, str) and session_id.strip():
            trace_kwargs["session_id"] = session_id

        self._client.trace(**trace_kwargs)
        return {"status": "patched", "added": added_count}

    def flush(self) -> None:
        self._client.flush()


class LangfuseTraceSource:
    def __init__(self, *, host: str, public_key: str, secret_key: str) -> None:
        self.host = _normalize_langfuse_host(host)
        self._auth_header = "Basic " + base64.b64encode(
            f"{public_key}:{secret_key}".encode("utf-8")
        ).decode("ascii")

    def _request_json(self, path: str, params: Dict[str, Any]) -> Dict[str, Any]:
        query = urlencode(
            {
                key: value
                for key, value in params.items()
                if value is not None and value != ""
            }
        )
        url = f"{self.host}{path}"
        if query:
            url = f"{url}?{query}"

        request = Request(
            url=url,
            headers={"Authorization": self._auth_header},
            method="GET",
        )
        last_error: Optional[BaseException] = None
        for attempt in range(3):
            try:
                with urlopen(request, timeout=20) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                break
            except (
                URLError,
                TimeoutError,
                RemoteDisconnected,
                ConnectionResetError,
                ConnectionRefusedError,
            ) as exc:
                last_error = exc
                if attempt == 2:
                    raise
                import time

                time.sleep(0.5 * (attempt + 1))
        else:
            if last_error is not None:
                raise last_error
            raise RuntimeError(f"Langfuse request failed without a captured error for {url}")
        if not isinstance(payload, dict):
            raise RuntimeError(f"Unexpected Langfuse payload type for {url}")
        return payload

    async def fetch_trace_page(
        self,
        *,
        page: int,
        limit: int,
        fields: Optional[str] = None,
    ) -> Dict[str, Any]:
        return await asyncio.to_thread(
            self._request_json,
            "/api/public/traces",
            {"page": page, "limit": limit, "fields": fields},
        )

    async def fetch_trace_by_id(self, trace_id: str) -> Dict[str, Any]:
        return await asyncio.to_thread(
            self._request_json,
            f"/api/public/traces/{trace_id}",
            {"fields": "core"},
        )

    async def fetch_observation_page(
        self,
        *,
        page: int,
        limit: int,
        observation_type: str = "GENERATION",
    ) -> Dict[str, Any]:
        return await asyncio.to_thread(
            self._request_json,
            "/api/public/observations",
            {
                "page": page,
                "limit": limit,
                "type": observation_type,
                "orderBy": "startTime.desc",
            },
        )


class LangfuseDatabaseSource:
    def __init__(self, *, database_url: str) -> None:
        self.database_url = database_url
        self._pool: Optional[asyncpg.Pool] = None

    async def connect(self) -> None:
        if self._pool is None:
            self._pool = await asyncpg.create_pool(
                dsn=self.database_url,
                min_size=1,
                max_size=2,
                command_timeout=60,
            )

    async def close(self) -> None:
        if self._pool is not None:
            await self._pool.close()
            self._pool = None

    async def fetch_generation_batch(
        self,
        *,
        limit: int,
        cursor_start_time: Optional[datetime] = None,
        cursor_id: Optional[str] = None,
        trace_id: Optional[str] = None,
        session_id: Optional[str] = None,
        from_start_time: Optional[str] = None,
        to_start_time: Optional[str] = None,
    ) -> List[asyncpg.Record]:
        if self._pool is None:
            raise RuntimeError("Langfuse database pool is not connected")

        predicates = ["o.type = 'GENERATION'"]
        params: List[Any] = []

        if trace_id:
            params.append(trace_id)
            predicates.append(f"o.trace_id = ${len(params)}")
        if session_id:
            params.append(session_id)
            predicates.append(f"t.session_id = ${len(params)}")

        from_dt = _parse_optional_datetime(from_start_time)
        to_dt = _parse_optional_datetime(to_start_time)
        if from_dt is not None:
            params.append(from_dt)
            predicates.append(f"o.start_time >= ${len(params)}")
        if to_dt is not None:
            params.append(to_dt)
            predicates.append(f"o.start_time <= ${len(params)}")

        if cursor_start_time is not None and cursor_id is not None:
            params.extend([cursor_start_time, cursor_id])
            predicates.append(
                f"(o.start_time, o.id) < (${len(params) - 1}, ${len(params)})"
            )

        params.append(limit)
        query = f"""
            SELECT
                o.id AS observation_id,
                o.name AS observation_name,
                o.start_time AS observation_start_time,
                o.end_time AS observation_end_time,
                o.parent_observation_id AS observation_parent_observation_id,
                o.type::text AS observation_type,
                o.trace_id AS observation_trace_id,
                o.metadata AS observation_metadata,
                o.model AS observation_model,
                o."modelParameters" AS observation_model_parameters,
                o.input AS observation_input,
                o.output AS observation_output,
                o.level::text AS observation_level,
                o.status_message AS observation_status_message,
                o.completion_start_time AS observation_completion_start_time,
                o.completion_tokens AS observation_completion_tokens,
                o.prompt_tokens AS observation_prompt_tokens,
                o.total_tokens AS observation_total_tokens,
                o.version AS observation_version,
                o.project_id AS observation_project_id,
                o.created_at AS observation_created_at,
                o.unit AS observation_unit,
                o.prompt_id AS observation_prompt_id,
                o.input_cost AS observation_input_cost,
                o.output_cost AS observation_output_cost,
                o.total_cost AS observation_total_cost,
                o.internal_model AS observation_internal_model,
                o.updated_at AS observation_updated_at,
                o.calculated_input_cost AS observation_calculated_input_cost,
                o.calculated_output_cost AS observation_calculated_output_cost,
                o.calculated_total_cost AS observation_calculated_total_cost,
                o.internal_model_id AS observation_internal_model_id,
                t.id AS trace_id,
                t.timestamp AS trace_timestamp,
                t.name AS trace_name,
                t.project_id AS trace_project_id,
                t.metadata AS trace_metadata,
                t.external_id AS trace_external_id,
                t.user_id AS trace_user_id,
                t.release AS trace_release,
                t.version AS trace_version,
                t.public AS trace_public,
                t.bookmarked AS trace_bookmarked,
                t.input AS trace_input,
                t.output AS trace_output,
                t.session_id AS trace_session_id,
                t.tags AS trace_tags,
                t.created_at AS trace_created_at,
                t.updated_at AS trace_updated_at
            FROM observations o
            JOIN traces t ON t.id = o.trace_id
            WHERE {' AND '.join(predicates)}
            ORDER BY o.start_time DESC, o.id DESC
            LIMIT ${len(params)}
        """

        async with self._pool.acquire() as connection:
            return await connection.fetch(query, *params)

    async def patch_tags(self, trace_id: str, tags: Sequence[str]) -> Dict[str, Any]:
        if self._pool is None:
            raise RuntimeError("Langfuse database pool is not connected")

        normalized_new_tags = sorted(
            {str(tag) for tag in tags if isinstance(tag, str) and tag.strip()}
        )
        if not normalized_new_tags:
            return {"status": "unchanged", "added": 0}

        async with self._pool.acquire() as connection:
            row = await connection.fetchrow(
                "SELECT tags FROM traces WHERE id = $1",
                trace_id,
            )
            if row is None:
                return {"status": "missing_trace", "added": 0}

            existing_tags = [
                str(tag)
                for tag in (row["tags"] or [])
                if isinstance(tag, str) and tag.strip()
            ]
            merged_tags = sorted(set(existing_tags) | set(normalized_new_tags))
            added_count = len(set(merged_tags) - set(existing_tags))
            if added_count == 0:
                return {"status": "unchanged", "added": 0}

            await connection.execute(
                """
                UPDATE traces
                SET tags = $2::text[], updated_at = CURRENT_TIMESTAMP
                WHERE id = $1
                """,
                trace_id,
                merged_tags,
            )
            return {"status": "patched", "added": added_count}


def _build_langfuse_trace_from_db_row(row: asyncpg.Record) -> Dict[str, Any]:
    return {
        "id": row["trace_id"],
        "timestamp": row["trace_timestamp"],
        "name": row["trace_name"],
        "projectId": row["trace_project_id"],
        "metadata": row["trace_metadata"] or {},
        "externalId": row["trace_external_id"],
        "userId": row["trace_user_id"],
        "release": row["trace_release"],
        "version": row["trace_version"],
        "public": row["trace_public"],
        "bookmarked": row["trace_bookmarked"],
        "input": row["trace_input"],
        "output": row["trace_output"],
        "sessionId": row["trace_session_id"],
        "tags": list(row["trace_tags"] or []),
        "createdAt": row["trace_created_at"],
        "updatedAt": row["trace_updated_at"],
    }


def _build_langfuse_observation_from_db_row(row: asyncpg.Record) -> Dict[str, Any]:
    return {
        "id": row["observation_id"],
        "name": row["observation_name"],
        "startTime": row["observation_start_time"],
        "endTime": row["observation_end_time"],
        "parentObservationId": row["observation_parent_observation_id"],
        "type": row["observation_type"],
        "traceId": row["observation_trace_id"],
        "metadata": row["observation_metadata"] or {},
        "model": row["observation_model"],
        "modelParameters": row["observation_model_parameters"],
        "input": row["observation_input"],
        "output": row["observation_output"],
        "level": row["observation_level"],
        "statusMessage": row["observation_status_message"],
        "completionStartTime": row["observation_completion_start_time"],
        "completionTokens": row["observation_completion_tokens"],
        "promptTokens": row["observation_prompt_tokens"],
        "totalTokens": row["observation_total_tokens"],
        "version": row["observation_version"],
        "projectId": row["observation_project_id"],
        "createdAt": row["observation_created_at"],
        "unit": row["observation_unit"],
        "promptId": row["observation_prompt_id"],
        "inputCost": _maybe_decimal_to_float(row["observation_input_cost"]),
        "outputCost": _maybe_decimal_to_float(row["observation_output_cost"]),
        "totalCost": _maybe_decimal_to_float(row["observation_total_cost"]),
        "internalModel": row["observation_internal_model"],
        "updatedAt": row["observation_updated_at"],
        "calculatedInputCost": _maybe_decimal_to_float(
            row["observation_calculated_input_cost"]
        ),
        "calculatedOutputCost": _maybe_decimal_to_float(
            row["observation_calculated_output_cost"]
        ),
        "calculatedTotalCost": _maybe_decimal_to_float(
            row["observation_calculated_total_cost"]
        ),
        "internalModelId": row["observation_internal_model_id"],
    }


class LangfuseClickHouseSource:
    def __init__(self, *, url: str, user: str, password: str) -> None:
        self.url = url
        self.user = user
        self.password = password
        self._auth_header = "Basic " + base64.b64encode(
            f"{self.user}:{self.password}".encode("utf-8")
        ).decode("ascii")

    def _request_rows(self, query: str) -> List[Dict[str, Any]]:
        request = Request(
            url=f"{self.url}/?default_format=JSONEachRow",
            headers={
                "Authorization": self._auth_header,
                "Content-Type": "text/plain; charset=utf-8",
            },
            data=query.encode("utf-8"),
            method="POST",
        )
        try:
            with urlopen(request, timeout=60) as response:
                payload = response.read().decode("utf-8")
        except HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"ClickHouse query failed with HTTP {exc.code}: {error_body}"
            ) from exc
        if not payload.strip():
            return []
        return [json.loads(line) for line in payload.splitlines() if line.strip()]

    async def fetch_generation_batch(
        self,
        *,
        limit: int,
        cursor_start_time: Optional[datetime] = None,
        cursor_id: Optional[str] = None,
        trace_id: Optional[str] = None,
        session_id: Optional[str] = None,
        from_start_time: Optional[str] = None,
        to_start_time: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        predicates = ["o.type = 'GENERATION'", "o.is_deleted = 0"]

        def q(value: str) -> str:
            return "'" + value.replace("\\", "\\\\").replace("'", "\\'") + "'"

        if trace_id:
            predicates.append(f"o.trace_id = {q(trace_id)}")
        if session_id:
            predicates.append(f"t.session_id = {q(session_id)}")

        from_dt = _parse_optional_datetime(from_start_time)
        to_dt = _parse_optional_datetime(to_start_time)
        if from_dt is not None:
            predicates.append(f"o.start_time >= toDateTime64({q(from_dt.isoformat(sep=' '))}, 3)")
        if to_dt is not None:
            predicates.append(f"o.start_time <= toDateTime64({q(to_dt.isoformat(sep=' '))}, 3)")

        if cursor_start_time is not None and cursor_id is not None:
            predicates.append(
                "("
                f"o.start_time < toDateTime64({q(cursor_start_time.isoformat(sep=' '))}, 3)"
                f" OR (o.start_time = toDateTime64({q(cursor_start_time.isoformat(sep=' '))}, 3)"
                f" AND o.id < {q(cursor_id)}))"
            )

        observation_query = f"""
            SELECT
                o.id AS observation_id,
                o.trace_id AS observation_trace_id,
                o.start_time AS observation_start_time,
                o.end_time AS observation_end_time,
                o.parent_observation_id AS observation_parent_observation_id,
                o.type AS observation_type,
                o.name AS observation_name,
                o.metadata AS observation_metadata,
                o.level AS observation_level,
                o.status_message AS observation_status_message,
                o.version AS observation_version,
                o.output AS observation_output,
                o.provided_model_name AS observation_model,
                o.internal_model_id AS observation_internal_model_id,
                o.model_parameters AS observation_model_parameters,
                o.provided_usage_details AS observation_provided_usage_details,
                o.usage_details AS observation_usage_details,
                o.provided_cost_details AS observation_provided_cost_details,
                o.cost_details AS observation_cost_details,
                o.total_cost AS observation_total_cost,
                o.completion_start_time AS observation_completion_start_time,
                o.prompt_id AS observation_prompt_id,
                o.prompt_name AS observation_prompt_name,
                o.prompt_version AS observation_prompt_version,
                o.created_at AS observation_created_at,
                o.updated_at AS observation_updated_at,
                o.project_id AS observation_project_id,
                o.environment AS observation_environment,
                o.tool_calls AS observation_tool_calls,
                o.tool_call_names AS observation_tool_call_names
            FROM observations AS o
            WHERE {' AND '.join(predicates)}
            ORDER BY o.start_time DESC, o.id DESC
            LIMIT {int(limit)}
            FORMAT JSONEachRow
        """
        observation_rows = await asyncio.to_thread(self._request_rows, observation_query)
        if not observation_rows:
            return []

        trace_ids = sorted(
            {
                str(row["observation_trace_id"])
                for row in observation_rows
                if isinstance(row.get("observation_trace_id"), str)
                and row["observation_trace_id"].strip()
            }
        )
        if not trace_ids:
            return observation_rows

        trace_query = f"""
            SELECT
                id AS trace_id,
                timestamp AS trace_timestamp,
                name AS trace_name,
                user_id AS trace_user_id,
                metadata AS trace_metadata,
                release AS trace_release,
                version AS trace_version,
                project_id AS trace_project_id,
                environment AS trace_environment,
                public AS trace_public,
                bookmarked AS trace_bookmarked,
                tags AS trace_tags,
                session_id AS trace_session_id,
                created_at AS trace_created_at,
                updated_at AS trace_updated_at
            FROM traces
            WHERE is_deleted = 0
              AND id IN ({', '.join(q(trace_id_value) for trace_id_value in trace_ids)})
            FORMAT JSONEachRow
        """
        trace_rows = await asyncio.to_thread(self._request_rows, trace_query)
        trace_by_id = {
            str(trace_row["trace_id"]): trace_row for trace_row in trace_rows if trace_row.get("trace_id")
        }

        if session_id:
            observation_rows = [
                row
                for row in observation_rows
                if trace_by_id.get(str(row.get("observation_trace_id")), {}).get("trace_session_id")
                == session_id
            ]

        merged_rows: List[Dict[str, Any]] = []
        for observation_row in observation_rows:
            trace_row = trace_by_id.get(str(observation_row.get("observation_trace_id")))
            if trace_row is None:
                continue
            merged_rows.append({**observation_row, **trace_row})
        return merged_rows


def _build_langfuse_trace_from_clickhouse_row(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": row.get("trace_id"),
        "timestamp": row.get("trace_timestamp"),
        "name": row.get("trace_name"),
        "projectId": row.get("trace_project_id"),
        "metadata": {
            key: _parse_clickhouse_value(value)
            for key, value in (row.get("trace_metadata") or {}).items()
        },
        "userId": row.get("trace_user_id"),
        "release": row.get("trace_release"),
        "version": row.get("trace_version"),
        "environment": row.get("trace_environment"),
        "public": row.get("trace_public"),
        "bookmarked": row.get("trace_bookmarked"),
        "tags": list(row.get("trace_tags") or []),
        "input": None,
        "output": None,
        "sessionId": row.get("trace_session_id"),
        "createdAt": row.get("trace_created_at"),
        "updatedAt": row.get("trace_updated_at"),
    }


def _build_langfuse_observation_from_clickhouse_row(row: Dict[str, Any]) -> Dict[str, Any]:
    usage_details = {
        key: int(value)
        for key, value in (row.get("observation_usage_details") or {}).items()
    }
    provided_usage_details = {
        key: int(value)
        for key, value in (row.get("observation_provided_usage_details") or {}).items()
    }
    cost_details = {
        key: _maybe_decimal_to_float(value)
        for key, value in (row.get("observation_cost_details") or {}).items()
    }
    provided_cost_details = {
        key: _maybe_decimal_to_float(value)
        for key, value in (row.get("observation_provided_cost_details") or {}).items()
    }

    tool_calls = row.get("observation_tool_calls") or []
    output_payload = _parse_clickhouse_value(row.get("observation_output"))
    if not isinstance(output_payload, dict):
        output_payload = {"raw_output": output_payload} if output_payload else None
    if tool_calls:
        if output_payload is None:
            output_payload = {"tool_calls": tool_calls}
        elif isinstance(output_payload, dict) and not isinstance(
            output_payload.get("tool_calls"), list
        ):
            output_payload["tool_calls"] = tool_calls
    tool_call_names = row.get("observation_tool_call_names") or []
    if tool_call_names and isinstance(output_payload, dict):
        output_payload.setdefault("tool_call_names", tool_call_names)

    return {
        "id": row.get("observation_id"),
        "traceId": row.get("observation_trace_id"),
        "startTime": row.get("observation_start_time"),
        "endTime": row.get("observation_end_time"),
        "parentObservationId": row.get("observation_parent_observation_id"),
        "type": row.get("observation_type"),
        "name": row.get("observation_name"),
        "metadata": {
            key: _parse_clickhouse_value(value)
            for key, value in (row.get("observation_metadata") or {}).items()
        },
        "level": row.get("observation_level"),
        "statusMessage": row.get("observation_status_message"),
        "version": row.get("observation_version"),
        "input": None,
        "output": output_payload,
        "model": row.get("observation_model"),
        "internalModelId": row.get("observation_internal_model_id"),
        "modelParameters": _parse_clickhouse_value(row.get("observation_model_parameters")),
        "usage": provided_usage_details or usage_details,
        "usageDetails": usage_details or provided_usage_details,
        "costDetails": cost_details or provided_cost_details,
        "calculatedTotalCost": _maybe_decimal_to_float(row.get("observation_total_cost")),
        "completionStartTime": row.get("observation_completion_start_time"),
        "promptId": row.get("observation_prompt_id"),
        "promptName": row.get("observation_prompt_name"),
        "promptVersion": row.get("observation_prompt_version"),
        "projectId": row.get("observation_project_id"),
        "environment": row.get("observation_environment"),
        "createdAt": row.get("observation_created_at"),
        "updatedAt": row.get("observation_updated_at"),
        "toolCallNames": tool_call_names,
    }


@dataclass
class BackfillStats:
    scanned_rows: int = 0
    reconstructable_rows: int = 0
    inserted_rows: int = 0
    updated_rows: int = 0
    existing_rows: int = 0
    skipped_rows: int = 0
    trace_tag_candidates: int = 0
    traces_patched: int = 0
    traces_unchanged: int = 0
    traces_missing: int = 0
    trace_tags_added: int = 0
    trace_write_errors: int = 0


async def _create_source_prisma_client(database_url: str) -> PrismaClient:
    proxy_logging = ProxyLogging(user_api_key_cache=DualCache())
    prisma_client = PrismaClient(
        database_url=database_url,
        proxy_logging_obj=proxy_logging,
    )
    await prisma_client.connect()
    return prisma_client


def _resolve_langfuse_credentials(args: argparse.Namespace) -> Dict[str, str]:
    host = _normalize_langfuse_host(
        _clean_secret(args.langfuse_host) or _clean_secret(get_secret_str("LANGFUSE_HOST"))
    )
    public_key = _clean_secret(args.langfuse_public_key) or _clean_secret(
        get_secret_str("LANGFUSE_PUBLIC_KEY")
    )
    secret_key = _clean_secret(args.langfuse_secret_key) or _clean_secret(
        get_secret_str("LANGFUSE_SECRET_KEY")
    )
    if not public_key or not secret_key:
        raise RuntimeError("Langfuse public/secret keys are required for Langfuse backfill")
    return {
        "host": host,
        "public_key": public_key,
        "secret_key": secret_key,
    }


def _record_matches_filters(record: Dict[str, Any], args: argparse.Namespace) -> bool:
    if args.request_id and record.get("litellm_call_id") != args.request_id:
        return False
    if args.trace_id and record.get("trace_id") != args.trace_id:
        return False
    if args.session_id and record.get("session_id") != args.session_id:
        return False
    if args.provider and record.get("provider") != args.provider:
        return False
    if args.model and record.get("model") != args.model:
        return False
    if args.status and record.get("metadata", {}).get("source_status") != args.status:
        return False

    start_time = record.get("start_time")
    from_start = _parse_optional_datetime(args.from_start_time)
    to_start = _parse_optional_datetime(args.to_start_time)
    if from_start and isinstance(start_time, datetime) and start_time < from_start:
        return False
    if to_start and isinstance(start_time, datetime) and start_time > to_start:
        return False

    return True


async def _run_spend_log_backfill(
    args: argparse.Namespace,
    *,
    source_database_url: str,
    langfuse_backfiller: Optional[LangfuseTraceTagBackfiller],
    run_id: str,
) -> Dict[str, Any]:
    prisma_client = await _create_source_prisma_client(source_database_url)
    target_dsn = _build_aawm_dsn()
    if args.apply:
        if not target_dsn:
            raise RuntimeError("AAWM/tristore database configuration is missing")
        pool = await _get_session_history_pool()
        await _ensure_session_history_schema_with_pool(pool)
    else:
        target_dsn = target_dsn or "unresolved"

    where = _build_source_where(args)
    stats = BackfillStats()
    session_source_counts: Counter[str] = Counter()
    trace_source_counts: Counter[str] = Counter()

    try:
        async for batch in _iter_spend_logs(
            prisma_client,
            where=where,
            batch_size=args.batch_size,
            limit=args.limit,
        ):
            stats.scanned_rows += len(batch)

            records: List[Dict[str, Any]] = []
            trace_tag_map: Dict[str, Set[str]] = defaultdict(set)

            for row in batch:
                record = _build_session_history_record_from_spend_log_row(
                    row, backfill_run_id=run_id
                )
                if record is None:
                    stats.skipped_rows += 1
                    continue

                stats.reconstructable_rows += 1
                records.append(record)
                session_source = record["metadata"].get("session_id_source")
                trace_source = record["metadata"].get("trace_id_source")
                if isinstance(session_source, str):
                    session_source_counts[session_source] += 1
                if isinstance(trace_source, str):
                    trace_source_counts[trace_source] += 1

                trace_id, trace_tags = _derive_langfuse_trace_tags_from_spend_log_row(row)
                if trace_id:
                    trace_tag_map[trace_id].update(trace_tags)

            if args.apply:
                existing_ids = await _get_existing_call_ids(
                    [record["litellm_call_id"] for record in records]
                )
                stats.existing_rows += len(existing_ids)
            else:
                existing_ids = set()

            new_records = [
                record for record in records if record["litellm_call_id"] not in existing_ids
            ]
            existing_records = [
                record for record in records if record["litellm_call_id"] in existing_ids
            ]

            if args.apply and records:
                await _persist_session_history_records(records)
                stats.inserted_rows += len(new_records)
                stats.updated_rows += len(existing_records)

            stats.trace_tag_candidates += len(trace_tag_map)
            if langfuse_backfiller is not None and args.apply:
                for trace_id, trace_tags in trace_tag_map.items():
                    if not trace_tags:
                        continue
                    try:
                        result = langfuse_backfiller.patch_tags(
                            trace_id, sorted(trace_tags)
                        )
                    except Exception:
                        stats.trace_write_errors += 1
                        continue
                    status = result["status"]
                    if status == "patched":
                        stats.traces_patched += 1
                        stats.trace_tags_added += int(result["added"])
                    elif status == "unchanged":
                        stats.traces_unchanged += 1
                    elif status == "missing_trace":
                        stats.traces_missing += 1
    finally:
        await prisma_client.disconnect()

    if langfuse_backfiller is not None and args.apply:
        langfuse_backfiller.flush()

    return {
        "source_mode": "spendlogs",
        "source_where": where,
        "target_dsn_redacted": target_dsn.split("@", 1)[-1] if "@" in target_dsn else target_dsn,
        "stats": {
            "scanned_rows": stats.scanned_rows,
            "reconstructable_rows": stats.reconstructable_rows,
            "inserted_rows": stats.inserted_rows,
            "updated_rows": stats.updated_rows,
            "existing_rows": stats.existing_rows,
            "skipped_rows": stats.skipped_rows,
            "trace_tag_candidates": stats.trace_tag_candidates,
            "traces_patched": stats.traces_patched,
            "traces_unchanged": stats.traces_unchanged,
            "traces_missing": stats.traces_missing,
            "trace_tags_added": stats.trace_tags_added,
            "trace_write_errors": stats.trace_write_errors,
        },
        "session_id_sources": dict(session_source_counts),
        "trace_id_sources": dict(trace_source_counts),
    }


async def _run_langfuse_trace_backfill(
    args: argparse.Namespace,
    *,
    langfuse_source: LangfuseTraceSource,
    langfuse_backfiller: Optional[LangfuseTraceTagBackfiller],
    run_id: str,
) -> Dict[str, Any]:
    target_dsn = _build_aawm_dsn()
    if args.apply:
        if not target_dsn:
            raise RuntimeError("AAWM/tristore database configuration is missing")
        pool = await _get_session_history_pool()
        await _ensure_session_history_schema_with_pool(pool)
    else:
        target_dsn = target_dsn or "unresolved"

    stats = BackfillStats()
    session_source_counts: Counter[str] = Counter()
    trace_source_counts: Counter[str] = Counter()
    trace_core_lookup: Dict[str, Dict[str, Any]] = {}
    trace_page = 1
    total_trace_pages: Optional[int] = None
    processed_records = 0

    trace_page_size = max(50, min(args.batch_size * 4, 200))
    while True:
        response = await langfuse_source.fetch_trace_page(
            page=trace_page,
            limit=trace_page_size,
            fields="core",
        )
        traces = response.get("data")
        meta = response.get("meta") or {}
        if not isinstance(traces, list) or not traces:
            break
        if total_trace_pages is None and isinstance(meta, dict):
            total_trace_pages = _safe_int(meta.get("totalPages"))

        for trace in traces:
            if not isinstance(trace, dict):
                continue
            trace_id = trace.get("id")
            if not isinstance(trace_id, str) or not trace_id.strip():
                continue
            if args.trace_id and trace_id != args.trace_id:
                continue
            if args.session_id and trace.get("sessionId") != args.session_id:
                continue
            trace_core_lookup[trace_id] = trace

        if args.trace_id and args.trace_id in trace_core_lookup:
            break
        if total_trace_pages is not None and trace_page >= total_trace_pages:
            break
        trace_page += 1

    page = 1
    total_pages: Optional[int] = None
    observation_page_size = max(25, min(args.batch_size * 5, 100))
    while True:
        response = await langfuse_source.fetch_observation_page(
            page=page,
            limit=observation_page_size,
            observation_type="GENERATION",
        )
        observations = response.get("data")
        meta = response.get("meta") or {}
        if not isinstance(observations, list) or not observations:
            break
        if total_pages is None and isinstance(meta, dict):
            total_pages = _safe_int(meta.get("totalPages"))

        records: List[Dict[str, Any]] = []
        trace_tag_map: Dict[str, Set[str]] = defaultdict(set)

        for observation in observations:
            if not isinstance(observation, dict):
                continue
            stats.scanned_rows += 1
            trace_id = observation.get("traceId")
            if not isinstance(trace_id, str) or not trace_id.strip():
                stats.skipped_rows += 1
                continue
            if args.trace_id and trace_id != args.trace_id:
                continue

            trace = trace_core_lookup.get(trace_id)
            if trace is None:
                try:
                    trace = await langfuse_source.fetch_trace_by_id(trace_id)
                except (HTTPError, URLError, TimeoutError):
                    trace = None
                if isinstance(trace, dict):
                    trace_core_lookup[trace_id] = trace
            if not isinstance(trace, dict):
                stats.skipped_rows += 1
                continue
            if args.session_id and trace.get("sessionId") != args.session_id:
                continue

            record = _build_session_history_record_from_langfuse_trace_observation(
                trace,
                observation,
                backfill_run_id=run_id,
            )
            if record is None:
                stats.skipped_rows += 1
                continue
            if not _record_matches_filters(record, args):
                continue

            stats.reconstructable_rows += 1
            records.append(record)

            session_source = record["metadata"].get("session_id_source")
            trace_source = record["metadata"].get("trace_id_source")
            if isinstance(session_source, str):
                session_source_counts[session_source] += 1
            if isinstance(trace_source, str):
                trace_source_counts[trace_source] += 1

            metadata = observation.get("metadata")
            if isinstance(metadata, dict):
                trace_tag_map[trace_id].update(
                    _derive_request_tags_from_langfuse_metadata(metadata)
                )

            processed_records += 1
            if args.limit is not None and processed_records >= args.limit:
                break

        if args.apply:
            existing_ids = await _get_existing_call_ids(
                [record["litellm_call_id"] for record in records]
            )
            stats.existing_rows += len(existing_ids)
        else:
            existing_ids = set()

        new_records = [
            record for record in records if record["litellm_call_id"] not in existing_ids
        ]
        existing_records = [
            record for record in records if record["litellm_call_id"] in existing_ids
        ]

        if args.apply and records:
            await _persist_session_history_records(records)
            stats.inserted_rows += len(new_records)
            stats.updated_rows += len(existing_records)

        stats.trace_tag_candidates += len(trace_tag_map)
        if langfuse_backfiller is not None and args.apply:
                for trace_id, trace_tags in trace_tag_map.items():
                    if not trace_tags:
                        continue
                    try:
                        result = langfuse_backfiller.patch_tags(
                            trace_id, sorted(trace_tags)
                        )
                    except Exception:
                        stats.trace_write_errors += 1
                        continue
                    status = result["status"]
                if status == "patched":
                    stats.traces_patched += 1
                    stats.trace_tags_added += int(result["added"])
                elif status == "unchanged":
                    stats.traces_unchanged += 1
                elif status == "missing_trace":
                    stats.traces_missing += 1

        if args.limit is not None and processed_records >= args.limit:
            break

        if total_pages is not None and page >= total_pages:
            break
        page += 1

    if langfuse_backfiller is not None and args.apply:
        langfuse_backfiller.flush()

    return {
        "source_mode": "langfuse",
        "source_where": {
            "trace_id": args.trace_id,
            "session_id": args.session_id,
            "provider": args.provider,
            "status": args.status,
            "from_start_time": args.from_start_time,
            "to_start_time": args.to_start_time,
            "request_id": args.request_id,
        },
        "target_dsn_redacted": target_dsn.split("@", 1)[-1] if "@" in target_dsn else target_dsn,
        "stats": {
            "scanned_rows": stats.scanned_rows,
            "reconstructable_rows": stats.reconstructable_rows,
            "inserted_rows": stats.inserted_rows,
            "updated_rows": stats.updated_rows,
            "existing_rows": stats.existing_rows,
            "skipped_rows": stats.skipped_rows,
            "trace_tag_candidates": stats.trace_tag_candidates,
            "traces_patched": stats.traces_patched,
            "traces_unchanged": stats.traces_unchanged,
            "traces_missing": stats.traces_missing,
            "trace_tags_added": stats.trace_tags_added,
            "trace_write_errors": stats.trace_write_errors,
        },
        "session_id_sources": dict(session_source_counts),
        "trace_id_sources": dict(trace_source_counts),
    }


async def _run_langfuse_db_backfill(
    args: argparse.Namespace,
    *,
    langfuse_database_url: str,
    run_id: str,
) -> Dict[str, Any]:
    target_dsn = _build_aawm_dsn()
    if args.apply:
        if not target_dsn:
            raise RuntimeError("AAWM/tristore database configuration is missing")
        pool = await _get_session_history_pool()
        await _ensure_session_history_schema_with_pool(pool)
    else:
        target_dsn = target_dsn or "unresolved"

    stats = BackfillStats()
    session_source_counts: Counter[str] = Counter()
    trace_source_counts: Counter[str] = Counter()
    cursor_start_time: Optional[datetime] = None
    cursor_id: Optional[str] = None
    processed_records = 0

    langfuse_source = LangfuseDatabaseSource(database_url=langfuse_database_url)
    await langfuse_source.connect()
    try:
        while True:
            rows = await langfuse_source.fetch_generation_batch(
                limit=max(1, args.batch_size),
                cursor_start_time=cursor_start_time,
                cursor_id=cursor_id,
                trace_id=args.trace_id,
                session_id=args.session_id,
                from_start_time=args.from_start_time,
                to_start_time=args.to_start_time,
            )
            if not rows:
                break

            records: List[Dict[str, Any]] = []
            trace_tag_map: Dict[str, Set[str]] = defaultdict(set)

            for row in rows:
                stats.scanned_rows += 1
                trace = _build_langfuse_trace_from_db_row(row)
                observation = _build_langfuse_observation_from_db_row(row)

                record = _build_session_history_record_from_langfuse_trace_observation(
                    trace,
                    observation,
                    backfill_run_id=run_id,
                )
                if record is None:
                    stats.skipped_rows += 1
                    continue
                if not _record_matches_filters(record, args):
                    continue

                stats.reconstructable_rows += 1
                records.append(record)

                session_source = record["metadata"].get("session_id_source")
                trace_source = record["metadata"].get("trace_id_source")
                if isinstance(session_source, str):
                    session_source_counts[session_source] += 1
                if isinstance(trace_source, str):
                    trace_source_counts[trace_source] += 1

                metadata = observation.get("metadata")
                if isinstance(metadata, dict):
                    trace_tag_map[trace["id"]].update(
                        _derive_request_tags_from_langfuse_metadata(metadata)
                    )

                processed_records += 1
                if args.limit is not None and processed_records >= args.limit:
                    break

            if args.apply:
                existing_ids = await _get_existing_call_ids(
                    [record["litellm_call_id"] for record in records]
                )
                stats.existing_rows += len(existing_ids)
            else:
                existing_ids = set()

            new_records = [
                record for record in records if record["litellm_call_id"] not in existing_ids
            ]
            existing_records = [
                record for record in records if record["litellm_call_id"] in existing_ids
            ]

            if args.apply and records:
                await _persist_session_history_records(records)
                stats.inserted_rows += len(new_records)
                stats.updated_rows += len(existing_records)

            stats.trace_tag_candidates += len(trace_tag_map)
            if args.apply and args.patch_langfuse_tags:
                for trace_id, trace_tags in trace_tag_map.items():
                    if not trace_tags:
                        continue
                    try:
                        result = await langfuse_source.patch_tags(
                            trace_id, sorted(trace_tags)
                        )
                    except Exception:
                        stats.trace_write_errors += 1
                        continue
                    status = result["status"]
                    if status == "patched":
                        stats.traces_patched += 1
                        stats.trace_tags_added += int(result["added"])
                    elif status == "unchanged":
                        stats.traces_unchanged += 1
                    elif status == "missing_trace":
                        stats.traces_missing += 1

            if args.limit is not None and processed_records >= args.limit:
                break

            last_row = rows[-1]
            cursor_start_time = last_row["observation_start_time"]
            cursor_id = last_row["observation_id"]

    finally:
        await langfuse_source.close()

    return {
        "source_mode": "langfuse_db",
        "source_where": {
            "trace_id": args.trace_id,
            "session_id": args.session_id,
            "provider": args.provider,
            "status": args.status,
            "from_start_time": args.from_start_time,
            "to_start_time": args.to_start_time,
            "request_id": args.request_id,
        },
        "target_dsn_redacted": target_dsn.split("@", 1)[-1] if "@" in target_dsn else target_dsn,
        "langfuse_database_url_redacted": langfuse_database_url.split("@", 1)[-1]
        if "@" in langfuse_database_url
        else langfuse_database_url,
        "stats": {
            "scanned_rows": stats.scanned_rows,
            "reconstructable_rows": stats.reconstructable_rows,
            "inserted_rows": stats.inserted_rows,
            "updated_rows": stats.updated_rows,
            "existing_rows": stats.existing_rows,
            "skipped_rows": stats.skipped_rows,
            "trace_tag_candidates": stats.trace_tag_candidates,
            "traces_patched": stats.traces_patched,
            "traces_unchanged": stats.traces_unchanged,
            "traces_missing": stats.traces_missing,
            "trace_tags_added": stats.trace_tags_added,
            "trace_write_errors": stats.trace_write_errors,
        },
        "session_id_sources": dict(session_source_counts),
        "trace_id_sources": dict(trace_source_counts),
    }


async def _run_langfuse_clickhouse_backfill(
    args: argparse.Namespace,
    *,
    clickhouse_config: Dict[str, str],
    langfuse_backfiller: Optional[LangfuseTraceTagBackfiller],
    run_id: str,
) -> Dict[str, Any]:
    target_dsn = _build_aawm_dsn()
    if args.apply:
        if not target_dsn:
            raise RuntimeError("AAWM/tristore database configuration is missing")
        pool = await _get_session_history_pool()
        await _ensure_session_history_schema_with_pool(pool)
    else:
        target_dsn = target_dsn or "unresolved"

    stats = BackfillStats()
    session_source_counts: Counter[str] = Counter()
    trace_source_counts: Counter[str] = Counter()
    cursor_start_time: Optional[datetime] = None
    cursor_id: Optional[str] = None
    processed_records = 0

    clickhouse_source = LangfuseClickHouseSource(**clickhouse_config)

    while True:
        rows = await clickhouse_source.fetch_generation_batch(
            limit=max(1, args.batch_size),
            cursor_start_time=cursor_start_time,
            cursor_id=cursor_id,
            trace_id=args.trace_id,
            session_id=args.session_id,
            from_start_time=args.from_start_time,
            to_start_time=args.to_start_time,
        )
        if not rows:
            break

        records: List[Dict[str, Any]] = []
        trace_tag_map: Dict[str, Set[str]] = defaultdict(set)

        for row in rows:
            stats.scanned_rows += 1
            trace = _build_langfuse_trace_from_clickhouse_row(row)
            observation = _build_langfuse_observation_from_clickhouse_row(row)

            record = _build_session_history_record_from_langfuse_trace_observation(
                trace,
                observation,
                backfill_run_id=run_id,
            )
            if record is None:
                stats.skipped_rows += 1
                continue
            if not _record_matches_filters(record, args):
                continue

            stats.reconstructable_rows += 1
            records.append(record)

            session_source = record["metadata"].get("session_id_source")
            trace_source = record["metadata"].get("trace_id_source")
            if isinstance(session_source, str):
                session_source_counts[session_source] += 1
            if isinstance(trace_source, str):
                trace_source_counts[trace_source] += 1

            metadata = observation.get("metadata")
            if isinstance(metadata, dict):
                trace_tag_map[trace["id"]].update(
                    _derive_request_tags_from_langfuse_metadata(metadata)
                )

            processed_records += 1
            if args.limit is not None and processed_records >= args.limit:
                break

        if args.apply:
            existing_ids = await _get_existing_call_ids(
                [record["litellm_call_id"] for record in records]
            )
            stats.existing_rows += len(existing_ids)
        else:
            existing_ids = set()

        new_records = [
            record for record in records if record["litellm_call_id"] not in existing_ids
        ]
        existing_records = [
            record for record in records if record["litellm_call_id"] in existing_ids
        ]

        if args.apply and records:
            await _persist_session_history_records(records)
            stats.inserted_rows += len(new_records)
            stats.updated_rows += len(existing_records)

        stats.trace_tag_candidates += len(trace_tag_map)
        if langfuse_backfiller is not None and args.apply:
            for trace_id, trace_tags in trace_tag_map.items():
                if not trace_tags:
                    continue
                try:
                    result = langfuse_backfiller.patch_tags(trace_id, sorted(trace_tags))
                except Exception:
                    stats.trace_write_errors += 1
                    continue
                status = result["status"]
                if status == "patched":
                    stats.traces_patched += 1
                    stats.trace_tags_added += int(result["added"])
                elif status == "unchanged":
                    stats.traces_unchanged += 1
                elif status == "missing_trace":
                    stats.traces_missing += 1

        if args.limit is not None and processed_records >= args.limit:
            break

        last_row = rows[-1]
        last_start_time = last_row.get("observation_start_time")
        cursor_start_time = (
            _parse_optional_datetime(str(last_start_time).replace(" ", "T"))
            if isinstance(last_start_time, str)
            else last_start_time
        )
        cursor_id = last_row.get("observation_id")

    if langfuse_backfiller is not None and args.apply:
        langfuse_backfiller.flush()

    return {
        "source_mode": "langfuse_clickhouse",
        "source_where": {
            "trace_id": args.trace_id,
            "session_id": args.session_id,
            "provider": args.provider,
            "status": args.status,
            "from_start_time": args.from_start_time,
            "to_start_time": args.to_start_time,
            "request_id": args.request_id,
        },
        "target_dsn_redacted": target_dsn.split("@", 1)[-1] if "@" in target_dsn else target_dsn,
        "clickhouse_url": clickhouse_config["url"],
        "stats": {
            "scanned_rows": stats.scanned_rows,
            "reconstructable_rows": stats.reconstructable_rows,
            "inserted_rows": stats.inserted_rows,
            "updated_rows": stats.updated_rows,
            "existing_rows": stats.existing_rows,
            "skipped_rows": stats.skipped_rows,
            "trace_tag_candidates": stats.trace_tag_candidates,
            "traces_patched": stats.traces_patched,
            "traces_unchanged": stats.traces_unchanged,
            "traces_missing": stats.traces_missing,
            "trace_tags_added": stats.trace_tags_added,
            "trace_write_errors": stats.trace_write_errors,
        },
        "session_id_sources": dict(session_source_counts),
        "trace_id_sources": dict(trace_source_counts),
    }


async def _load_trace_tag_candidates_from_session_history(
    args: argparse.Namespace,
) -> Dict[str, Set[str]]:
    pool = await _get_session_history_pool()
    where_clauses = ["trace_id IS NOT NULL"]
    params: List[Any] = []

    if args.trace_id:
        params.append(args.trace_id)
        where_clauses.append(f"trace_id = ${len(params)}")
    if args.session_id:
        params.append(args.session_id)
        where_clauses.append(f"session_id = ${len(params)}")

    query = f"""
        SELECT trace_id, metadata->'request_tags' AS request_tags
        FROM public.session_history
        WHERE {' AND '.join(where_clauses)}
    """

    async with pool.acquire() as connection:
        rows = await connection.fetch(query, *params)

    trace_tag_map: Dict[str, Set[str]] = defaultdict(set)
    for row in rows:
        trace_id = row["trace_id"]
        request_tags = row["request_tags"]
        if not isinstance(trace_id, str) or not trace_id.strip():
            continue
        if isinstance(request_tags, str):
            try:
                request_tags = json.loads(request_tags)
            except json.JSONDecodeError:
                request_tags = None
        if not isinstance(request_tags, list):
            continue
        for tag in request_tags:
            if isinstance(tag, str) and tag.strip():
                trace_tag_map[trace_id].add(tag)
    return trace_tag_map


async def _run_trace_tag_writeback_from_session_history(
    args: argparse.Namespace,
    *,
    langfuse_backfiller: Optional[LangfuseTraceTagBackfiller],
) -> Dict[str, Any]:
    trace_tag_map = await _load_trace_tag_candidates_from_session_history(args)
    stats = BackfillStats(trace_tag_candidates=len(trace_tag_map))

    if args.apply:
        if langfuse_backfiller is None:
            raise RuntimeError("Langfuse tag writeback requires Langfuse API credentials")
        for trace_id, trace_tags in trace_tag_map.items():
            if not trace_tags:
                continue
            try:
                result = langfuse_backfiller.patch_tags(trace_id, sorted(trace_tags))
            except Exception:
                stats.trace_write_errors += 1
                continue
            status = result["status"]
            if status == "patched":
                stats.traces_patched += 1
                stats.trace_tags_added += int(result["added"])
            elif status == "unchanged":
                stats.traces_unchanged += 1
            elif status == "missing_trace":
                stats.traces_missing += 1
        langfuse_backfiller.flush()

    return {
        "source_mode": "session_history_trace_tags",
        "source_where": {
            "trace_id": args.trace_id,
            "session_id": args.session_id,
        },
        "stats": {
            "trace_tag_candidates": stats.trace_tag_candidates,
            "traces_patched": stats.traces_patched,
            "traces_unchanged": stats.traces_unchanged,
            "traces_missing": stats.traces_missing,
            "trace_tags_added": stats.trace_tags_added,
            "trace_write_errors": stats.trace_write_errors,
        },
    }


def _recalculate_session_history_response_cost(row: Dict[str, Any]) -> Optional[float]:
    model = str(row.get("model") or "").strip()
    if not model or model.lower() == "unknown":
        return None

    prompt_tokens = _safe_int(row.get("input_tokens")) or 0
    completion_tokens = _safe_int(row.get("output_tokens")) or 0
    if prompt_tokens <= 0 and completion_tokens <= 0:
        return None

    cache_creation_input_tokens = _safe_int(row.get("cache_creation_input_tokens")) or 0
    cache_read_input_tokens = _safe_int(row.get("cache_read_input_tokens")) or 0
    cache_input_tokens = cache_creation_input_tokens + cache_read_input_tokens
    prompt_tokens_for_cost = prompt_tokens
    if cache_input_tokens > prompt_tokens:
        # Older rows stored only non-cache input in input_tokens while keeping
        # cache counters separately. LiteLLM cost calculation expects total
        # prompt tokens when cache details are present.
        prompt_tokens_for_cost = prompt_tokens + cache_input_tokens
    provider = str(row.get("provider") or "").strip() or None
    cost_model = model
    cost_provider = provider
    if (provider or "").lower() == "chatgpt" or model.lower().startswith("chatgpt/"):
        # Avoid the ChatGPT auth-backed cost path during offline historical repairs.
        # The stored ChatGPT rows here use OpenAI model ids and static pricing.
        cost_provider = "openai"
        if model.lower().startswith("chatgpt/"):
            cost_model = model.split("/", 1)[1]

    try:
        import litellm
        from litellm.types.utils import (
            CacheCreationTokenDetails,
            PromptTokensDetailsWrapper,
            Usage,
        )

        usage_object = None
        if (
            (cost_provider or "").lower() == "anthropic"
            and cache_creation_input_tokens > 0
        ):
            # Claude Code cache writes use Anthropic's 1-hour ephemeral cache tier.
            # session_history only stores aggregate creation tokens, so rebuild the
            # detail object needed for LiteLLM to price the same way live logging did.
            usage_object = Usage(
                prompt_tokens=prompt_tokens_for_cost,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens_for_cost + completion_tokens,
                prompt_tokens_details=PromptTokensDetailsWrapper(
                    text_tokens=max(
                        prompt_tokens_for_cost
                        - cache_read_input_tokens
                        - cache_creation_input_tokens,
                        0,
                    ),
                    cached_tokens=cache_read_input_tokens,
                    cache_creation_tokens=cache_creation_input_tokens,
                    cache_creation_token_details=CacheCreationTokenDetails(
                        ephemeral_1h_input_tokens=cache_creation_input_tokens
                    ),
                ),
            )

        prompt_cost, completion_cost = litellm.cost_per_token(
            model=cost_model,
            prompt_tokens=prompt_tokens_for_cost,
            completion_tokens=completion_tokens,
            custom_llm_provider=cost_provider,
            cache_creation_input_tokens=cache_creation_input_tokens,
            cache_read_input_tokens=cache_read_input_tokens,
            usage_object=usage_object,
            call_type="responses",
        )
    except Exception:
        try:
            import litellm

            prompt_cost, completion_cost = litellm.cost_per_token(
                model=cost_model,
                prompt_tokens=prompt_tokens_for_cost,
                completion_tokens=completion_tokens,
                cache_creation_input_tokens=cache_creation_input_tokens,
                cache_read_input_tokens=cache_read_input_tokens,
            )
        except Exception:
            return None

    return float(prompt_cost + completion_cost)


def _derive_session_history_tenant_identity(
    row: Dict[str, Any],
) -> Tuple[Optional[str], Optional[str]]:
    metadata = row.get("metadata")
    if not isinstance(metadata, dict):
        metadata = _parse_clickhouse_value(metadata)
    tenant_id, source = _extract_tenant_identity_from_metadata_sources(
        ("session_history.metadata", metadata),
    )
    if tenant_id:
        return tenant_id, source

    if str(row.get("tenant_id") or "").strip():
        return None, None

    repository = _extract_repository_identity_from_metadata_sources(
        ("session_history.metadata", metadata),
    )
    if repository:
        return repository, "session_history.metadata.repository"

    repository = _extract_repository_identity_from_metadata_sources(
        ("session_history", {"repository": row.get("repository")}),
    )
    if repository:
        return repository, "session_history.repository"

    return None, None


def _extract_gemini_control_plane_method(value: Any) -> Optional[str]:
    if isinstance(value, str):
        value_lower = value.lower()
        for method_lower, method in _GEMINI_CONTROL_PLANE_METHODS_BY_LOWER.items():
            if method_lower in value_lower:
                return method
        return None
    if isinstance(value, dict):
        for nested_value in value.values():
            method = _extract_gemini_control_plane_method(nested_value)
            if method is not None:
                return method
        return None
    if isinstance(value, list):
        for nested_value in value:
            method = _extract_gemini_control_plane_method(nested_value)
            if method is not None:
                return method
    return None


def _is_gemini_control_plane_session_history_row(
    row: Dict[str, Any],
) -> Tuple[bool, Optional[str]]:
    metadata = row.get("metadata")
    if not isinstance(metadata, dict):
        metadata = _parse_clickhouse_value(metadata)
    if not isinstance(metadata, dict):
        metadata = {}

    method = _extract_gemini_control_plane_method(
        {
            "call_type": row.get("call_type"),
            "model": row.get("model"),
            "model_group": row.get("model_group"),
            "metadata": metadata,
        }
    )
    if method is None:
        return False, None

    provider_candidates = (
        row.get("provider"),
        metadata.get("custom_llm_provider"),
        metadata.get("provider"),
        metadata.get("litellm_provider"),
    )
    provider_is_gemini = any(
        isinstance(candidate, str)
        and candidate.strip().lower() in {"gemini", "google"}
        for candidate in provider_candidates
    )

    hidden_params = metadata.get("hidden_params")
    hidden_api_base = (
        hidden_params.get("api_base") if isinstance(hidden_params, dict) else None
    )
    route_candidates = (
        row.get("call_type"),
        metadata.get("user_api_key_request_route"),
        metadata.get("passthrough_route_family"),
        metadata.get("api_base"),
        hidden_api_base,
    )
    route_is_gemini = any(
        isinstance(candidate, str)
        and (
            "gemini" in candidate.lower()
            or "googleapis.com" in candidate.lower()
            or "generativelanguage" in candidate.lower()
            or "v1internal:" in candidate.lower()
        )
        for candidate in route_candidates
    )
    method_on_route = _extract_gemini_control_plane_method(route_candidates) is not None

    # Historical bad rows may have provider/model gaps. The four Code Assist
    # method names are only accepted when the row also looks like a Gemini route.
    if provider_is_gemini or route_is_gemini or method_on_route:
        return True, method
    return False, None


async def _run_gemini_control_plane_session_history_repair(  # noqa: PLR0915
    args: argparse.Namespace,
) -> Dict[str, Any]:
    pool = await _get_session_history_pool()
    await _ensure_session_history_schema_with_pool(pool)

    action = args.repair_gemini_control_plane
    where_clauses = ["id > $1"]
    params: List[Any] = []
    if args.request_id:
        params.append(args.request_id)
        where_clauses.append(f"litellm_call_id = ${len(params) + 1}")
    if args.trace_id:
        params.append(args.trace_id)
        where_clauses.append(f"trace_id = ${len(params) + 1}")
    if args.session_id:
        params.append(args.session_id)
        where_clauses.append(f"session_id = ${len(params) + 1}")
    if args.provider:
        params.append(args.provider)
        where_clauses.append(f"provider = ${len(params) + 1}")
    if args.model:
        params.append(args.model)
        where_clauses.append(f"model = ${len(params) + 1}")
    from_start = _parse_optional_datetime(args.from_start_time)
    if from_start is not None:
        params.append(from_start)
        where_clauses.append(f"start_time >= ${len(params) + 1}")
    to_start = _parse_optional_datetime(args.to_start_time)
    if to_start is not None:
        params.append(to_start)
        where_clauses.append(f"start_time <= ${len(params) + 1}")

    method_predicates = []
    for method in sorted(_GEMINI_CONTROL_PLANE_METHODS):
        params.append(f"%{method}%")
        placeholder = f"${len(params) + 1}"
        method_predicates.append(
            f"(call_type ILIKE {placeholder} OR model ILIKE {placeholder} OR metadata::text ILIKE {placeholder})"
        )
    where_clauses.append(f"({' OR '.join(method_predicates)})")

    limit = args.limit
    batch_size = max(1, args.batch_size)
    scanned_rows = 0
    matched_rows = 0
    deleted_rows = 0
    updated_rows = 0
    deleted_tool_activity_rows = 0
    matched_methods: Counter[str] = Counter()
    cursor_id = 0

    while True:
        page_params = [cursor_id, *params, batch_size]
        query = f"""
            SELECT
                id,
                litellm_call_id,
                provider,
                model,
                model_group,
                call_type,
                start_time,
                metadata
            FROM public.session_history
            WHERE {' AND '.join(where_clauses)}
            ORDER BY id ASC
            LIMIT ${len(page_params)}
        """
        async with pool.acquire() as connection:
            rows = await connection.fetch(query, *page_params)
        if not rows:
            break

        for row in rows:
            cursor_id = max(cursor_id, int(row["id"]))
            if limit is not None and scanned_rows >= limit:
                break
            scanned_rows += 1
            row_dict = dict(row)
            is_match, method = _is_gemini_control_plane_session_history_row(row_dict)
            if not is_match or method is None:
                continue
            matched_rows += 1
            matched_methods[method] += 1

            if not args.apply:
                continue

            async with pool.acquire() as connection:
                async with connection.transaction():
                    if action == "delete":
                        litellm_call_id = row_dict.get("litellm_call_id")
                        if litellm_call_id:
                            tool_result = await connection.execute(
                                """
                                DELETE FROM public.session_history_tool_activity
                                WHERE litellm_call_id = $1
                                """,
                                litellm_call_id,
                            )
                            try:
                                deleted_tool_activity_rows += int(
                                    tool_result.rsplit(" ", 1)[-1]
                                )
                            except Exception:
                                pass
                        result = await connection.execute(
                            "DELETE FROM public.session_history WHERE id = $1",
                            row_dict["id"],
                        )
                        try:
                            deleted_rows += int(result.rsplit(" ", 1)[-1])
                        except Exception:
                            deleted_rows += 1
                    elif action == "mark":
                        metadata = row_dict.get("metadata")
                        if not isinstance(metadata, dict):
                            metadata = _parse_clickhouse_value(metadata)
                        if not isinstance(metadata, dict):
                            metadata = {}
                        metadata.update(
                            {
                                "session_history_repair": "gemini_control_plane",
                                "gemini_control_plane_repair_action": "mark",
                                "gemini_control_plane_excluded": True,
                                "gemini_control_plane_method": method,
                            }
                        )
                        result = await connection.execute(
                            """
                            UPDATE public.session_history
                            SET metadata = $1::jsonb
                            WHERE id = $2
                            """,
                            json.dumps(metadata),
                            row_dict["id"],
                        )
                        try:
                            updated_rows += int(result.rsplit(" ", 1)[-1])
                        except Exception:
                            updated_rows += 1
                    else:
                        raise RuntimeError(f"Unsupported Gemini repair action: {action}")

        if limit is not None and scanned_rows >= limit:
            break
        if len(rows) < batch_size:
            break

    return {
        "source_mode": "session_history_repair",
        "repair": "gemini_control_plane",
        "repair_action": action,
        "source_where": {
            "request_id": args.request_id,
            "trace_id": args.trace_id,
            "session_id": args.session_id,
            "provider": args.provider,
            "model": args.model,
            "from_start_time": args.from_start_time,
            "to_start_time": args.to_start_time,
        },
        "stats": {
            "scanned_rows": scanned_rows,
            "matched_rows": matched_rows,
            "deleted_rows": deleted_rows,
            "updated_rows": updated_rows,
            "deleted_tool_activity_rows": deleted_tool_activity_rows,
            "matched_methods": dict(sorted(matched_methods.items())),
        },
    }


async def _run_session_history_repair(args: argparse.Namespace) -> Dict[str, Any]:
    if args.repair_gemini_control_plane:
        return await _run_gemini_control_plane_session_history_repair(args)

    pool = await _get_session_history_pool()
    await _ensure_session_history_schema_with_pool(pool)

    repair_costs = args.repair_costs or not args.repair_tenant_ids
    repair_tenant_ids = args.repair_tenant_ids or not args.repair_costs

    where_clauses = ["litellm_call_id IS NOT NULL"]
    params: List[Any] = []
    if args.request_id:
        params.append(args.request_id)
        where_clauses.append(f"litellm_call_id = ${len(params)}")
    if args.trace_id:
        params.append(args.trace_id)
        where_clauses.append(f"trace_id = ${len(params)}")
    if args.session_id:
        params.append(args.session_id)
        where_clauses.append(f"session_id = ${len(params)}")
    if args.provider:
        params.append(args.provider)
        where_clauses.append(f"provider = ${len(params)}")
    if args.model:
        params.append(args.model)
        where_clauses.append(f"model = ${len(params)}")
    from_start = _parse_optional_datetime(args.from_start_time)
    if from_start is not None:
        params.append(from_start)
        where_clauses.append(f"start_time >= ${len(params)}")
    to_start = _parse_optional_datetime(args.to_start_time)
    if to_start is not None:
        params.append(to_start)
        where_clauses.append(f"start_time <= ${len(params)}")

    limit = args.limit
    batch_size = max(1, args.batch_size)
    scanned_rows = 0
    rows_with_updates = 0
    tenant_updates = 0
    cost_candidates = 0
    cost_updates = 0
    cost_repair_errors = 0
    offset = 0

    while True:
        page_params = [*params, batch_size, offset]
        query = f"""
            SELECT
                litellm_call_id,
                tenant_id,
                provider,
                model,
                input_tokens,
                output_tokens,
                cache_read_input_tokens,
                cache_creation_input_tokens,
                response_cost_usd,
                repository,
                metadata
            FROM public.session_history
            WHERE {' AND '.join(where_clauses)}
            ORDER BY start_time ASC NULLS LAST, litellm_call_id ASC
            LIMIT ${len(params) + 1}
            OFFSET ${len(params) + 2}
        """
        async with pool.acquire() as connection:
            rows = await connection.fetch(query, *page_params)
        if not rows:
            break

        for row in rows:
            if limit is not None and scanned_rows >= limit:
                break
            scanned_rows += 1

            row_dict = dict(row)
            metadata = row_dict.get("metadata")
            if not isinstance(metadata, dict):
                metadata = _parse_clickhouse_value(metadata)
            if not isinstance(metadata, dict):
                metadata = {}

            next_tenant_id = row_dict.get("tenant_id")
            next_cost = _safe_float(row_dict.get("response_cost_usd"))
            should_update = False

            if repair_tenant_ids:
                tenant_id, tenant_source = _derive_session_history_tenant_identity(
                    {**row_dict, "metadata": metadata}
                )
                if tenant_id:
                    if str(next_tenant_id or "").strip() != tenant_id:
                        next_tenant_id = tenant_id
                        tenant_updates += 1
                        should_update = True
                    if metadata.get("tenant_id") != tenant_id:
                        metadata["tenant_id"] = tenant_id
                        should_update = True
                    if tenant_source and metadata.get("tenant_id_source") != tenant_source:
                        metadata["tenant_id_source"] = tenant_source
                        should_update = True
                elif next_tenant_id and metadata.get("tenant_id") != next_tenant_id:
                    metadata["tenant_id"] = next_tenant_id
                    metadata.setdefault("tenant_id_source", "session_history.tenant_id")
                    should_update = True

            if repair_costs:
                recalculated_cost = _recalculate_session_history_response_cost(row_dict)
                if recalculated_cost is None:
                    cost_repair_errors += 1
                else:
                    cost_candidates += 1
                    if (
                        next_cost is None
                        or abs(float(next_cost) - recalculated_cost) > 0.000000001
                    ):
                        next_cost = recalculated_cost
                        metadata["response_cost_source"] = "session_history_repair"
                        cost_updates += 1
                        should_update = True

            if should_update:
                rows_with_updates += 1
                if args.apply:
                    async with pool.acquire() as connection:
                        await connection.execute(
                            """
                            UPDATE public.session_history
                            SET
                                tenant_id = $1,
                                response_cost_usd = $2,
                                metadata = $3::jsonb
                            WHERE litellm_call_id = $4
                            """,
                            next_tenant_id,
                            next_cost,
                            json.dumps(metadata),
                            row_dict["litellm_call_id"],
                        )

        if limit is not None and scanned_rows >= limit:
            break
        if len(rows) < batch_size:
            break
        offset += len(rows)

    return {
        "source_mode": "session_history_repair",
        "source_where": {
            "request_id": args.request_id,
            "trace_id": args.trace_id,
            "session_id": args.session_id,
            "provider": args.provider,
            "model": args.model,
            "from_start_time": args.from_start_time,
            "to_start_time": args.to_start_time,
        },
        "stats": {
            "scanned_rows": scanned_rows,
            "rows_with_updates": rows_with_updates,
            "tenant_updates": tenant_updates,
            "cost_candidates": cost_candidates,
            "cost_updates": cost_updates,
            "cost_repair_errors": cost_repair_errors,
        },
    }


async def _run_backfill(args: argparse.Namespace) -> Dict[str, Any]:
    if args.repair_session_history:
        run_id = datetime.now(timezone.utc).strftime("repair-%Y%m%dT%H%M%SZ")
        result = await _run_session_history_repair(args)
        return {
            "mode": "apply" if args.apply else "dry_run",
            "run_id": run_id,
            **result,
        }

    source_database_url = _clean_secret(args.source_database_url) or _get_first_secret(
        _SOURCE_DB_ENV_VARS
    )
    target_dsn = _build_aawm_dsn()
    langfuse_database_url = _derive_langfuse_database_url(
        getattr(args, "langfuse_database_url", None),
        target_dsn=target_dsn,
    )
    clickhouse_config = _resolve_clickhouse_config(args)

    langfuse_backfiller: Optional[LangfuseTraceTagBackfiller] = None
    langfuse_creds: Optional[Dict[str, str]] = None
    needs_langfuse_writeback = args.patch_langfuse_tags and (
        args.source_mode in {"auto", "langfuse", "langfuse_clickhouse", "langfuse_db"}
        or (args.source_mode == "spendlogs")
    )
    use_langfuse_http = bool(args.source_mode == "langfuse") or (
        args.source_mode == "auto"
        and not source_database_url
        and not langfuse_database_url
    ) or needs_langfuse_writeback

    if use_langfuse_http:
        langfuse_creds = _resolve_langfuse_credentials(args)
        if args.patch_langfuse_tags:
            langfuse_backfiller = LangfuseTraceTagBackfiller(
                host=langfuse_creds["host"],
                public_key=langfuse_creds["public_key"],
                secret_key=langfuse_creds["secret_key"],
            )

    run_id = datetime.now(timezone.utc).strftime("backfill-%Y%m%dT%H%M%SZ")
    source_mode = args.source_mode
    if source_mode == "auto":
        if source_database_url:
            source_mode = "spendlogs"
        elif clickhouse_config.get("url"):
            source_mode = "langfuse_clickhouse"
        elif langfuse_database_url:
            source_mode = "langfuse_db"
        else:
            source_mode = "langfuse"

    if args.tags_only:
        result = await _run_trace_tag_writeback_from_session_history(
            args,
            langfuse_backfiller=langfuse_backfiller,
        )
    elif source_mode == "spendlogs":
        if not source_database_url:
            raise RuntimeError("Source proxy DATABASE_URL is missing")
        result = await _run_spend_log_backfill(
            args,
            source_database_url=source_database_url,
            langfuse_backfiller=langfuse_backfiller,
            run_id=run_id,
        )
    elif source_mode == "langfuse_db":
        if not langfuse_database_url:
            raise RuntimeError("Langfuse database URL is missing")
        result = await _run_langfuse_db_backfill(
            args,
            langfuse_database_url=langfuse_database_url,
            run_id=run_id,
        )
    elif source_mode == "langfuse_clickhouse":
        result = await _run_langfuse_clickhouse_backfill(
            args,
            clickhouse_config=clickhouse_config,
            langfuse_backfiller=langfuse_backfiller,
            run_id=run_id,
        )
    elif source_mode == "langfuse":
        langfuse_source = LangfuseTraceSource(
            host=langfuse_creds["host"],
            public_key=langfuse_creds["public_key"],
            secret_key=langfuse_creds["secret_key"],
        )
        result = await _run_langfuse_trace_backfill(
            args,
            langfuse_source=langfuse_source,
            langfuse_backfiller=langfuse_backfiller,
            run_id=run_id,
        )
    else:
        raise RuntimeError(f"Unsupported source mode: {source_mode}")

    return {
        "mode": "apply" if args.apply else "dry_run",
        "run_id": run_id,
        **result,
    }


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Backfill AAWM session_history from historical LiteLLM_SpendLogs."
    )
    parser.add_argument(
        "--source-mode",
        choices=("auto", "spendlogs", "langfuse", "langfuse_db", "langfuse_clickhouse"),
        default="auto",
        help="Backfill source: auto prefers proxy spend logs when DATABASE_URL is available, otherwise Langfuse ClickHouse, then fallback Langfuse APIs.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Persist session_history rows and patch Langfuse tags. Default is dry-run.",
    )
    parser.add_argument(
        "--tags-only",
        action="store_true",
        help="Skip event reconstruction and only write back historical trace tags derived from session_history.",
    )
    parser.add_argument(
        "--repair-session-history",
        action="store_true",
        help="Repair existing session_history rows in place. By default repairs both tenant ids and response costs.",
    )
    parser.add_argument(
        "--repair-costs",
        action="store_true",
        help="With --repair-session-history, only recalculate response_cost_usd from current model pricing.",
    )
    parser.add_argument(
        "--repair-tenant-ids",
        action="store_true",
        help="With --repair-session-history, only fill tenant_id from stored session metadata.",
    )
    parser.add_argument(
        "--repair-gemini-control-plane",
        choices=("delete", "mark"),
        help=(
            "With --repair-session-history, repair historical Gemini Code Assist "
            "control-plane rows for loadCodeAssist/listExperiments/"
            "retrieveUserQuota/fetchAdminControls. Choose delete or mark; "
            "requires --apply to persist changes."
        ),
    )
    parser.add_argument(
        "--patch-langfuse-tags",
        action="store_true",
        help="Patch historical Langfuse trace tags from derived request tags.",
    )
    parser.add_argument("--request-id", help="Restrict to a single spend-log request_id.")
    parser.add_argument(
        "--trace-id",
        help="Restrict to a single trace id.",
    )
    parser.add_argument("--session-id", help="Restrict to a single session id.")
    parser.add_argument("--provider", help="Restrict to a custom_llm_provider value.")
    parser.add_argument("--model", help="Restrict to a single session_history model value.")
    parser.add_argument(
        "--status",
        choices=("success", "failure"),
        help="Restrict to a spend-log status.",
    )
    parser.add_argument(
        "--from-start-time",
        help="Only include spend logs with startTime >= this ISO timestamp.",
    )
    parser.add_argument(
        "--to-start-time",
        help="Only include spend logs with startTime <= this ISO timestamp.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Number of source rows/traces to fetch per batch or page.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Maximum number of spend logs to scan.",
    )
    parser.add_argument(
        "--source-database-url",
        help="Override the proxy DATABASE_URL used to read LiteLLM_SpendLogs.",
    )
    parser.add_argument("--langfuse-host", help="Override LANGFUSE_HOST.")
    parser.add_argument("--clickhouse-url", help="Override CLICKHOUSE_URL.")
    parser.add_argument("--clickhouse-user", help="Override CLICKHOUSE_USER.")
    parser.add_argument("--clickhouse-password", help="Override CLICKHOUSE_PASSWORD.")
    parser.add_argument(
        "--langfuse-database-url",
        help="Override the Langfuse Postgres DATABASE_URL used for direct historical reads.",
    )
    parser.add_argument(
        "--langfuse-public-key", help="Override LANGFUSE_PUBLIC_KEY."
    )
    parser.add_argument(
        "--langfuse-secret-key", help="Override LANGFUSE_SECRET_KEY."
    )
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()
    if args.repair_gemini_control_plane and not args.repair_session_history:
        parser.error("--repair-gemini-control-plane requires --repair-session-history")
    if args.repair_gemini_control_plane and (
        args.repair_costs or args.repair_tenant_ids
    ):
        parser.error(
            "--repair-gemini-control-plane cannot be combined with --repair-costs "
            "or --repair-tenant-ids"
        )
    result = asyncio.run(_run_backfill(args))
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
