#!/usr/bin/env python3
"""
Backfill public.rate_limit_observations from historical Langfuse data.

The backfill is intentionally conservative. It only feeds structured Langfuse
metadata/input/output objects through the same runtime quota extractors used by
the AAWM callback; arbitrary prose matches are treated as candidate rows only
and do not become observations unless they parse as a known provider payload.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Set, Tuple
from urllib.error import HTTPError
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

from litellm.integrations.aawm_agent_identity import (  # noqa: E402
    _build_aawm_dsn,
    _build_rate_limit_observations,
    _ensure_session_history_schema,
    _persist_session_history_records,
    _rate_limit_storage_quota_key,
    _rate_limit_storage_remaining_pct,
)
from litellm.secret_managers.main import get_secret_str  # noqa: E402

_CLICKHOUSE_URL_ENV_VARS = ("CLICKHOUSE_URL", "LANGFUSE_CLICKHOUSE_URL")
_CLICKHOUSE_USER_ENV_VARS = ("CLICKHOUSE_USER", "LANGFUSE_CLICKHOUSE_USER")
_CLICKHOUSE_PASSWORD_ENV_VARS = ("CLICKHOUSE_PASSWORD", "LANGFUSE_CLICKHOUSE_PASSWORD")
_MINIO_ENDPOINT_ENV_VARS = (
    "LANGFUSE_S3_EVENT_UPLOAD_ENDPOINT",
    "MINIO_ENDPOINT",
    "AWS_ENDPOINT_URL",
)
_MINIO_ACCESS_KEY_ENV_VARS = (
    "LANGFUSE_S3_EVENT_UPLOAD_ACCESS_KEY_ID",
    "MINIO_ROOT_USER",
    "AWS_ACCESS_KEY_ID",
)
_MINIO_SECRET_KEY_ENV_VARS = (
    "LANGFUSE_S3_EVENT_UPLOAD_SECRET_ACCESS_KEY",
    "MINIO_ROOT_PASSWORD",
    "AWS_SECRET_ACCESS_KEY",
)
_TARGET_DB_ARG_ENV_MAP = {
    "aawm_db_host": "AAWM_DB_HOST",
    "aawm_db_port": "AAWM_DB_PORT",
    "aawm_db_name": "AAWM_DB_NAME",
    "aawm_db_user": "AAWM_DB_USER",
    "aawm_db_password": "AAWM_DB_PASSWORD",
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


def _normalize_clickhouse_url(value: Optional[str]) -> str:
    cleaned = _clean_secret(value) or "http://127.0.0.1:8123"
    normalized = cleaned.rstrip("/")
    if not normalized.startswith(("http://", "https://")):
        normalized = f"http://{normalized}"
    return normalized.replace("clickhouse", "127.0.0.1")


def _parse_optional_datetime(value: Optional[str]) -> Optional[datetime]:
    if value is None:
        return None
    normalized = value.strip().replace("Z", "+00:00")
    if not normalized:
        return None
    return datetime.fromisoformat(normalized)


def _parse_json_value(value: Any) -> Any:
    if isinstance(value, (dict, list)) or value is None:
        return value
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    if not stripped:
        return None
    if stripped[0] not in "[{":
        return value
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        return value


def _parse_clickhouse_map(raw_map: Any) -> Dict[str, Any]:
    if not isinstance(raw_map, dict):
        return {}
    return {str(key): _parse_json_value(value) for key, value in raw_map.items()}


def _quote_clickhouse_string(value: str) -> str:
    return "'" + value.replace("\\", "\\\\").replace("'", "\\'") + "'"


def _format_clickhouse_datetime(value: datetime) -> str:
    if value.tzinfo is not None:
        value = value.astimezone(timezone.utc).replace(tzinfo=None)
    return value.isoformat(sep=" ")


class ClickHouseClient:
    def __init__(self, *, url: str, user: str, password: str, timeout_seconds: int) -> None:
        self.url = url
        self.timeout_seconds = timeout_seconds
        self._auth_header = "Basic " + base64.b64encode(
            f"{user}:{password}".encode("utf-8")
        ).decode("ascii")

    def request_rows(self, query: str) -> List[Dict[str, Any]]:
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
            with urlopen(request, timeout=self.timeout_seconds) as response:
                payload = response.read().decode("utf-8")
        except HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"ClickHouse query failed with HTTP {exc.code}: {error_body}"
            ) from exc
        if not payload.strip():
            return []
        return [json.loads(line) for line in payload.splitlines() if line.strip()]

    async def fetch_observation_batch(
        self,
        *,
        limit: int,
        cursor_start_time: Optional[datetime],
        cursor_id: Optional[str],
        from_start_time: Optional[str],
        to_start_time: Optional[str],
        trace_id: Optional[str],
        include_input: bool,
    ) -> List[Dict[str, Any]]:
        predicates = ["o.type = 'GENERATION'", "o.is_deleted = 0"]
        markers = (
            "rate_limits",
            "usage_limit_reached",
            "resets_at",
            "used_percent",
            "used_percentage",
            "retrieveUserQuota",
            "MODEL_CAPACITY_EXHAUSTED",
            "RATE_LIMIT_EXCEEDED",
        )
        marker_predicates = [
            f"positionUTF8(toString(o.metadata), {_quote_clickhouse_string(marker)}) > 0"
            for marker in markers
        ]
        marker_predicates.extend(
            f"positionUTF8(ifNull(o.output, ''), {_quote_clickhouse_string(marker)}) > 0"
            for marker in markers
        )
        if include_input:
            marker_predicates.extend(
                f"positionUTF8(ifNull(o.input, ''), {_quote_clickhouse_string(marker)}) > 0"
                for marker in markers
            )
        predicates.append("(" + " OR ".join(marker_predicates) + ")")

        if trace_id:
            predicates.append(f"o.trace_id = {_quote_clickhouse_string(trace_id)}")

        from_dt = _parse_optional_datetime(from_start_time)
        to_dt = _parse_optional_datetime(to_start_time)
        if from_dt is not None:
            predicates.append(
                "o.start_time >= "
                f"toDateTime64({_quote_clickhouse_string(_format_clickhouse_datetime(from_dt))}, 3)"
            )
        if to_dt is not None:
            predicates.append(
                "o.start_time <= "
                f"toDateTime64({_quote_clickhouse_string(_format_clickhouse_datetime(to_dt))}, 3)"
            )
        if cursor_start_time is not None and cursor_id is not None:
            cursor_value = _quote_clickhouse_string(
                _format_clickhouse_datetime(cursor_start_time)
            )
            predicates.append(
                "("
                f"o.start_time < toDateTime64({cursor_value}, 3)"
                f" OR (o.start_time = toDateTime64({cursor_value}, 3)"
                f" AND o.id < {_quote_clickhouse_string(cursor_id)}))"
            )

        input_select = "o.input AS observation_input," if include_input else "NULL AS observation_input,"
        query = f"""
            SELECT
                o.id AS observation_id,
                o.trace_id AS observation_trace_id,
                o.start_time AS observation_start_time,
                o.end_time AS observation_end_time,
                o.name AS observation_name,
                o.metadata AS observation_metadata,
                {input_select}
                o.output AS observation_output,
                o.provided_model_name AS observation_model,
                o.environment AS observation_environment
            FROM observations AS o
            WHERE {' AND '.join(predicates)}
            ORDER BY o.start_time DESC, o.id DESC
            LIMIT {int(limit)}
            FORMAT JSONEachRow
        """
        return await asyncio.to_thread(self.request_rows, query)

    async def fetch_blob_log_batch(
        self,
        *,
        limit: int,
        cursor_created_at: Optional[datetime],
        cursor_id: Optional[str],
        from_created_at: Optional[str],
        to_created_at: Optional[str],
        entity_type: str,
    ) -> List[Dict[str, Any]]:
        predicates = ["is_deleted = 0", f"entity_type = {_quote_clickhouse_string(entity_type)}"]
        from_dt = _parse_optional_datetime(from_created_at)
        to_dt = _parse_optional_datetime(to_created_at)
        if from_dt is not None:
            predicates.append(
                "created_at >= "
                f"toDateTime64({_quote_clickhouse_string(_format_clickhouse_datetime(from_dt))}, 3)"
            )
        if to_dt is not None:
            predicates.append(
                "created_at <= "
                f"toDateTime64({_quote_clickhouse_string(_format_clickhouse_datetime(to_dt))}, 3)"
            )
        if cursor_created_at is not None and cursor_id is not None:
            cursor_value = _quote_clickhouse_string(
                _format_clickhouse_datetime(cursor_created_at)
            )
            predicates.append(
                "("
                f"created_at < toDateTime64({cursor_value}, 3)"
                f" OR (created_at = toDateTime64({cursor_value}, 3)"
                f" AND id < {_quote_clickhouse_string(cursor_id)}))"
            )
        query = f"""
            SELECT
                id,
                entity_type,
                entity_id,
                event_id,
                bucket_name,
                bucket_path,
                created_at,
                updated_at,
                event_ts
            FROM blob_storage_file_log
            WHERE {' AND '.join(predicates)}
            ORDER BY created_at DESC, id DESC
            LIMIT {int(limit)}
            FORMAT JSONEachRow
        """
        return await asyncio.to_thread(self.request_rows, query)


class MinioEventBlobClient:
    def __init__(
        self,
        *,
        endpoint_url: str,
        access_key_id: str,
        secret_access_key: str,
    ) -> None:
        try:
            import boto3
            from botocore.config import Config
        except Exception as exc:  # pragma: no cover - depends on local extras
            raise RuntimeError(
                "boto3/botocore are required for --source-mode minio"
            ) from exc

        self._client = boto3.client(
            "s3",
            endpoint_url=endpoint_url,
            aws_access_key_id=access_key_id,
            aws_secret_access_key=secret_access_key,
            config=Config(signature_version="s3v4"),
        )

    def read_event_blob(self, bucket: str, key: str) -> Any:
        response = self._client.get_object(Bucket=bucket, Key=key)
        with response["Body"] as body:
            payload = body.read().decode("utf-8", errors="replace")
        return json.loads(payload)


def _infer_provider(model: Optional[str], metadata: Dict[str, Any]) -> Optional[str]:
    explicit = metadata.get("custom_llm_provider") or metadata.get("provider")
    if isinstance(explicit, str) and explicit.strip():
        return "gemini" if explicit.strip().lower() == "google" else explicit.strip().lower()
    text = " ".join(
        str(value)
        for value in (
            model,
            metadata.get("passthrough_route_family"),
            metadata.get("trace_name"),
        )
        if value
    ).lower()
    if "claude" in text or "anthropic" in text:
        return "anthropic"
    if "gemini" in text or "google" in text:
        return "gemini"
    if "codex" in text or "gpt" in text or "openai" in text:
        return "openai"
    return None


_FREEFORM_TEXT_KEYS = {
    "content",
    "raw_output",
    "result",
    "stderr",
    "stdout",
    "text",
}
_STRUCTURED_RATE_LIMIT_KEYS = {
    "details",
    "error",
    "google_retrieve_user_quota",
    "google_user_quota",
    "metadata",
    "rate_limits",
}


def _sanitize_structured_quota_payload(value: Any, *, parent_key: str = "") -> Any:
    parsed = _parse_json_value(value)
    if isinstance(parsed, list):
        sanitized_items = [
            _sanitize_structured_quota_payload(item, parent_key=parent_key)
            for item in parsed
        ]
        return [item for item in sanitized_items if item is not None]
    if not isinstance(parsed, dict):
        return None if parent_key in _FREEFORM_TEXT_KEYS else parsed

    sanitized: Dict[str, Any] = {}
    for key, nested_value in parsed.items():
        key_text = str(key)
        key_lower = key_text.lower()
        if key_lower in _FREEFORM_TEXT_KEYS:
            continue
        if key_lower == "message" and parent_key not in {"error", "details"}:
            continue
        sanitized_value = _sanitize_structured_quota_payload(
            nested_value,
            parent_key=key_lower,
        )
        if sanitized_value is not None:
            sanitized[key_text] = sanitized_value

    if sanitized:
        return sanitized
    if _STRUCTURED_RATE_LIMIT_KEYS.intersection(str(key).lower() for key in parsed):
        return sanitized
    return None


def _coerce_result_payload(*values: Any) -> Any:
    for value in values:
        parsed = _parse_json_value(value)
        if isinstance(parsed, (dict, list)):
            return _sanitize_structured_quota_payload(parsed)
    return None


def _build_record_from_langfuse_body(
    body: Dict[str, Any],
    *,
    backfill_source: str,
    source_locator: Optional[str] = None,
    include_input: bool = False,
) -> Optional[Dict[str, Any]]:
    if not isinstance(body, dict):
        return None
    metadata = _parse_clickhouse_map(body.get("metadata"))
    if body.get("environment") is not None:
        metadata.setdefault("litellm_environment", body.get("environment"))
    if body.get("name") is not None:
        metadata.setdefault("trace_name", body.get("name"))
    if source_locator:
        metadata["rate_limit_backfill_locator"] = source_locator
    metadata["rate_limit_backfill_source"] = backfill_source

    model = body.get("model") or metadata.get("model") or "unknown"
    result_payload = _coerce_result_payload(
        body.get("output"),
        body.get("input") if include_input else None,
    )
    kwargs = {
        "custom_llm_provider": _infer_provider(str(model), metadata),
        "model": str(model) if model is not None else "unknown",
        "litellm_call_id": body.get("id"),
        "litellm_session_id": metadata.get("session_id"),
        "litellm_trace_id": body.get("traceId") or metadata.get("trace_id"),
        "standard_logging_object": {
            "metadata": metadata,
            "response": result_payload,
            "output": result_payload,
            "trace_id": body.get("traceId"),
            "session_id": metadata.get("session_id"),
        },
        "litellm_params": {
            "metadata": metadata,
            "litellm_session_id": metadata.get("session_id"),
            "litellm_trace_id": body.get("traceId") or metadata.get("trace_id"),
        },
        "passthrough_logging_payload": {
            "request_body": (
                _sanitize_structured_quota_payload(body.get("input"))
                if include_input
                else None
            ),
        },
    }
    observations = _build_rate_limit_observations(
        kwargs,
        result_payload,
        body.get("startTime") or body.get("timestamp"),
        body.get("endTime") or body.get("timestamp"),
    )
    if not observations:
        return None
    for observation in observations:
        evidence = observation.setdefault("evidence", {})
        if isinstance(evidence, dict):
            evidence["historical_backfill"] = True
            evidence["backfill_source"] = backfill_source
            if source_locator:
                evidence["source_locator"] = source_locator
    return {
        "_skip_session_history": True,
        "litellm_call_id": body.get("id"),
        "session_id": metadata.get("session_id"),
        "model": str(model) if model is not None else "unknown",
        "rate_limit_observations": observations,
    }


def _build_body_from_clickhouse_row(row: Dict[str, Any], *, include_input: bool) -> Dict[str, Any]:
    metadata = _parse_clickhouse_map(row.get("observation_metadata"))
    if row.get("observation_environment") is not None:
        metadata.setdefault("litellm_environment", row.get("observation_environment"))
    return {
        "id": row.get("observation_id"),
        "traceId": row.get("observation_trace_id"),
        "name": row.get("observation_name"),
        "startTime": row.get("observation_start_time"),
        "endTime": row.get("observation_end_time"),
        "metadata": metadata,
        "input": row.get("observation_input") if include_input else None,
        "output": row.get("observation_output"),
        "model": row.get("observation_model"),
        "environment": row.get("observation_environment"),
    }


def build_record_from_clickhouse_row(
    row: Dict[str, Any],
    *,
    include_input: bool = False,
) -> Optional[Dict[str, Any]]:
    body = _build_body_from_clickhouse_row(row, include_input=include_input)
    return _build_record_from_langfuse_body(
        body,
        backfill_source="langfuse_clickhouse",
        source_locator=str(row.get("observation_id") or ""),
        include_input=include_input,
    )


def iter_langfuse_events(blob_payload: Any) -> Iterator[Dict[str, Any]]:
    payload = _parse_json_value(blob_payload)
    if isinstance(payload, dict):
        payload = [payload]
    if not isinstance(payload, list):
        return
    for event in payload:
        if isinstance(event, dict):
            yield event


def build_record_from_langfuse_event(
    event: Dict[str, Any],
    *,
    source_locator: Optional[str] = None,
    include_input: bool = False,
) -> Optional[Dict[str, Any]]:
    if not isinstance(event, dict):
        return None
    event_type = str(event.get("type") or "")
    if not (
        event_type.startswith("generation-")
        or event_type.startswith("observation-")
        or event_type.startswith("span-")
    ):
        return None
    body = event.get("body")
    if not isinstance(body, dict):
        return None
    body = dict(body)
    body.setdefault("timestamp", event.get("timestamp"))
    return _build_record_from_langfuse_body(
        body,
        backfill_source="langfuse_minio",
        source_locator=source_locator,
        include_input=include_input,
    )


def _observation_signature(observation: Dict[str, Any]) -> Tuple[Any, ...]:
    return (
        observation.get("source"),
        _rate_limit_storage_quota_key(observation),
        str(observation.get("observed_at")),
        str(observation.get("provider_resets_at")),
        _rate_limit_storage_remaining_pct(observation),
        observation.get("litellm_call_id"),
    )


async def _filter_existing_observations(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    call_ids = sorted(
        {
            str(observation.get("litellm_call_id"))
            for record in records
            for observation in record.get("rate_limit_observations", [])
            if observation.get("litellm_call_id")
        }
    )
    if not call_ids:
        return records
    target_dsn = _build_aawm_dsn()
    if not target_dsn:
        raise RuntimeError("AAWM/tristore database configuration is missing")
    conn = await asyncpg.connect(target_dsn)
    try:
        await _ensure_session_history_schema(conn)
        rows = await conn.fetch(
            """
            SELECT source, quota_key, observed_at,
                   expected_reset_at AS provider_resets_at,
                   remaining_pct, litellm_call_id
            FROM public.rate_limit_observations
            WHERE litellm_call_id = ANY($1::text[])
            """,
            call_ids,
        )
    finally:
        await conn.close()
    existing = set()
    for row in rows:
        row_dict = dict(row)
        existing.add(
            (
                row_dict.get("source"),
                row_dict.get("quota_key"),
                str(row_dict.get("observed_at")),
                str(row_dict.get("provider_resets_at")),
                row_dict.get("remaining_pct"),
                row_dict.get("litellm_call_id"),
            )
        )
    filtered: List[Dict[str, Any]] = []
    for record in records:
        observations = [
            observation
            for observation in record.get("rate_limit_observations", [])
            if _observation_signature(observation) not in existing
        ]
        if observations:
            next_record = dict(record)
            next_record["rate_limit_observations"] = observations
            filtered.append(next_record)
    return filtered


def _configure_target_database_env(args: argparse.Namespace) -> None:
    for arg_name, env_name in _TARGET_DB_ARG_ENV_MAP.items():
        value = _clean_secret(getattr(args, arg_name, None))
        if value is not None:
            os.environ[env_name] = value


async def _verify_target_database(required_database: Optional[str]) -> None:
    target_dsn = _build_aawm_dsn()
    if not target_dsn:
        raise RuntimeError("AAWM/tristore database configuration is missing")
    conn = await asyncpg.connect(target_dsn)
    try:
        current_database = await conn.fetchval("select current_database()")
        if required_database and current_database != required_database:
            raise RuntimeError(
                "Refusing to apply rate-limit backfill to "
                f"{current_database!r}; expected {required_database!r}"
            )
        await _ensure_session_history_schema(conn)
    finally:
        await conn.close()


def _dedupe_records(records: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    deduped: List[Dict[str, Any]] = []
    seen: Set[Tuple[Any, ...]] = set()
    for record in records:
        observations = []
        for observation in record.get("rate_limit_observations", []):
            signature = _observation_signature(observation)
            if signature in seen:
                continue
            seen.add(signature)
            observations.append(observation)
        if observations:
            next_record = dict(record)
            next_record["rate_limit_observations"] = observations
            deduped.append(next_record)
    return deduped


@dataclass
class BackfillStats:
    scanned_rows: int = 0
    scanned_events: int = 0
    candidate_records: int = 0
    extracted_observations: int = 0
    inserted_observations: int = 0
    skipped_existing_observations: int = 0
    source_errors: int = 0


async def _persist_records(
    records: List[Dict[str, Any]],
    *,
    apply: bool,
    skip_existing: bool,
    stats: BackfillStats,
) -> None:
    records = _dedupe_records(records)
    stats.candidate_records += len(records)
    stats.extracted_observations += sum(
        len(record.get("rate_limit_observations", [])) for record in records
    )
    if not apply or not records:
        return
    filtered = await _filter_existing_observations(records) if skip_existing else records
    stats.skipped_existing_observations += (
        sum(len(record.get("rate_limit_observations", [])) for record in records)
        - sum(len(record.get("rate_limit_observations", [])) for record in filtered)
    )
    if not filtered:
        return
    await _persist_session_history_records(filtered)
    stats.inserted_observations += sum(
        len(record.get("rate_limit_observations", [])) for record in filtered
    )


async def _run_clickhouse_backfill(args: argparse.Namespace, client: ClickHouseClient) -> BackfillStats:
    stats = BackfillStats()
    cursor_start_time: Optional[datetime] = None
    cursor_id: Optional[str] = None
    processed = 0
    while True:
        rows = await client.fetch_observation_batch(
            limit=max(1, args.batch_size),
            cursor_start_time=cursor_start_time,
            cursor_id=cursor_id,
            from_start_time=args.from_start_time,
            to_start_time=args.to_start_time,
            trace_id=args.trace_id,
            include_input=args.include_input,
        )
        if not rows:
            break
        records: List[Dict[str, Any]] = []
        for row in rows:
            stats.scanned_rows += 1
            processed += 1
            record = build_record_from_clickhouse_row(row, include_input=args.include_input)
            if record is not None:
                records.append(record)
            if args.limit is not None and processed >= args.limit:
                break
        await _persist_records(
            records,
            apply=args.apply,
            skip_existing=args.skip_existing,
            stats=stats,
        )
        if args.limit is not None and processed >= args.limit:
            break
        last_row = rows[-1]
        cursor_id = str(last_row.get("observation_id") or "")
        cursor_start_time = _parse_optional_datetime(
            str(last_row.get("observation_start_time")).replace(" ", "T")
        )
    return stats


async def _run_minio_backfill(
    args: argparse.Namespace,
    client: ClickHouseClient,
    blob_client: MinioEventBlobClient,
) -> BackfillStats:
    stats = BackfillStats()
    cursor_created_at: Optional[datetime] = None
    cursor_id: Optional[str] = None
    processed = 0
    while True:
        rows = await client.fetch_blob_log_batch(
            limit=max(1, args.batch_size),
            cursor_created_at=cursor_created_at,
            cursor_id=cursor_id,
            from_created_at=args.from_start_time,
            to_created_at=args.to_start_time,
            entity_type=args.minio_entity_type,
        )
        if not rows:
            break
        records: List[Dict[str, Any]] = []
        for row in rows:
            stats.scanned_rows += 1
            processed += 1
            try:
                blob_payload = await asyncio.to_thread(
                    blob_client.read_event_blob,
                    str(row["bucket_name"]),
                    str(row["bucket_path"]),
                )
            except Exception:
                stats.source_errors += 1
                continue
            for event in iter_langfuse_events(blob_payload):
                stats.scanned_events += 1
                record = build_record_from_langfuse_event(
                    event,
                    source_locator=str(row.get("bucket_path") or ""),
                    include_input=args.include_input,
                )
                if record is not None:
                    records.append(record)
            if args.limit is not None and processed >= args.limit:
                break
        await _persist_records(
            records,
            apply=args.apply,
            skip_existing=args.skip_existing,
            stats=stats,
        )
        if args.limit is not None and processed >= args.limit:
            break
        last_row = rows[-1]
        cursor_id = str(last_row.get("id") or "")
        cursor_created_at = _parse_optional_datetime(
            str(last_row.get("created_at")).replace(" ", "T")
        )
    return stats


def _resolve_clickhouse_config(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "url": _normalize_clickhouse_url(
            args.clickhouse_url or _get_first_secret(_CLICKHOUSE_URL_ENV_VARS)
        ),
        "user": _clean_secret(args.clickhouse_user)
        or _get_first_secret(_CLICKHOUSE_USER_ENV_VARS)
        or "clickhouse",
        "password": _clean_secret(args.clickhouse_password)
        or _get_first_secret(_CLICKHOUSE_PASSWORD_ENV_VARS)
        or "clickhouse",
        "timeout_seconds": args.clickhouse_timeout_seconds,
    }


def _resolve_minio_config(args: argparse.Namespace) -> Dict[str, str]:
    return {
        "endpoint_url": _clean_secret(args.minio_endpoint)
        or _get_first_secret(_MINIO_ENDPOINT_ENV_VARS)
        or "http://127.0.0.1:9010",
        "access_key_id": _clean_secret(args.minio_access_key)
        or _get_first_secret(_MINIO_ACCESS_KEY_ENV_VARS)
        or "langfuse",
        "secret_access_key": _clean_secret(args.minio_secret_key)
        or _get_first_secret(_MINIO_SECRET_KEY_ENV_VARS)
        or "langfuse-secret",
    }


async def _run(args: argparse.Namespace) -> Dict[str, Any]:
    _configure_target_database_env(args)
    if args.apply:
        await _verify_target_database(args.require_target_database)
    clickhouse_config = _resolve_clickhouse_config(args)
    client = ClickHouseClient(**clickhouse_config)
    if args.source_mode == "clickhouse":
        stats = await _run_clickhouse_backfill(args, client)
    elif args.source_mode == "minio":
        blob_client = MinioEventBlobClient(**_resolve_minio_config(args))
        stats = await _run_minio_backfill(args, client, blob_client)
    else:
        raise ValueError(f"unsupported source mode: {args.source_mode}")
    return {
        "source_mode": args.source_mode,
        "applied": bool(args.apply),
        "clickhouse_url": clickhouse_config["url"],
        "target_dsn_redacted": (
            (_build_aawm_dsn() or "unresolved").split("@", 1)[-1]
        ),
        "stats": {
            "scanned_rows": stats.scanned_rows,
            "scanned_events": stats.scanned_events,
            "candidate_records": stats.candidate_records,
            "extracted_observations": stats.extracted_observations,
            "inserted_observations": stats.inserted_observations,
            "skipped_existing_observations": stats.skipped_existing_observations,
            "source_errors": stats.source_errors,
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Backfill public.rate_limit_observations from Langfuse history."
    )
    parser.add_argument("--source-mode", choices=("clickhouse", "minio"), default="clickhouse")
    parser.add_argument("--apply", action="store_true", help="Write extracted observations.")
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--from-start-time", default=None)
    parser.add_argument("--to-start-time", default=None)
    parser.add_argument("--trace-id", default=None)
    parser.add_argument(
        "--include-input",
        action="store_true",
        help="Also inspect structured observation input payloads. Defaults off.",
    )
    parser.add_argument(
        "--skip-existing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip observations already present with the same signature.",
    )
    parser.add_argument("--clickhouse-url", default=None)
    parser.add_argument("--clickhouse-user", default=None)
    parser.add_argument("--clickhouse-password", default=None)
    parser.add_argument("--clickhouse-timeout-seconds", type=int, default=90)
    parser.add_argument("--aawm-db-host", default=None)
    parser.add_argument("--aawm-db-port", default=None)
    parser.add_argument("--aawm-db-name", default=None)
    parser.add_argument("--aawm-db-user", default=None)
    parser.add_argument("--aawm-db-password", default=None)
    parser.add_argument(
        "--require-target-database",
        default="aawm_tristore",
        help="Refuse --apply unless current_database() matches this value.",
    )
    parser.add_argument("--minio-endpoint", default=None)
    parser.add_argument("--minio-access-key", default=None)
    parser.add_argument("--minio-secret-key", default=None)
    parser.add_argument("--minio-entity-type", default="observation")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = asyncio.run(_run(args))
    sys.stdout.write(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
