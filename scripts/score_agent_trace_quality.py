#!/usr/bin/env python3
"""Deterministically score suspicious agent traces.

The scorer is intentionally conservative and dry-run by default. It reads
candidate rows from ``public.session_history``, resolves the corresponding
Langfuse generation payload from ClickHouse/MinIO, then emits compact evidence
and optional Langfuse scores.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import re
import sys
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import Request, urlopen

import psycopg

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
        cleaned = value.strip()
        if len(cleaned) >= 2 and cleaned[0] == cleaned[-1] and cleaned[0] in {'"', "'"}:
            cleaned = cleaned[1:-1]
        os.environ[key] = cleaned


_load_repo_dotenv()


def _clean(value: Any) -> Optional[str]:
    if value is None:
        return None
    if not isinstance(value, str):
        value = str(value)
    cleaned = value.strip()
    return cleaned or None


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


def _coerce_int(value: Any, default: int = 0) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _parse_optional_datetime(value: Optional[str]) -> Optional[datetime]:
    cleaned = _clean(value)
    if not cleaned:
        return None
    return datetime.fromisoformat(cleaned.replace("Z", "+00:00"))


def _format_clickhouse_datetime(value: datetime) -> str:
    if value.tzinfo is not None:
        value = value.astimezone(timezone.utc).replace(tzinfo=None)
    return value.isoformat(sep=" ")


def _quote_clickhouse_string(value: str) -> str:
    return "'" + value.replace("\\", "\\\\").replace("'", "\\'") + "'"


def _quote_clickhouse_string_array(values: Sequence[str]) -> str:
    return "[" + ",".join(_quote_clickhouse_string(value) for value in values) + "]"


@dataclass(frozen=True)
class SessionCandidate:
    row_id: int
    created_at: Optional[str]
    trace_id: Optional[str]
    session_id: Optional[str]
    litellm_call_id: Optional[str]
    source_observation_id: Optional[str]
    provider: Optional[str]
    model: Optional[str]
    agent_name: Optional[str]
    repository: Optional[str]
    tenant_id: Optional[str]
    input_tokens: int
    output_tokens: int
    tool_call_count: int
    invalid_tool_call_count: int
    metadata: Dict[str, Any]


@dataclass(frozen=True)
class ObservationPayload:
    observation_id: Optional[str]
    trace_id: Optional[str]
    body: Dict[str, Any]
    source: str
    source_locator: Optional[str] = None


@dataclass
class ToolResultEvidence:
    message_index: int
    is_final_message: bool
    image_base64_lengths: List[int] = field(default_factory=list)
    text_lengths: List[int] = field(default_factory=list)


@dataclass
class ToolUseEvidence:
    message_index: int
    sequence_index: int
    name: str
    command: Optional[str]
    mutating: bool
    destructive_checkout: bool


@dataclass
class TraceScoreEvidence:
    trace_id: Optional[str]
    session_id: Optional[str]
    observation_id: Optional[str]
    source: str
    source_locator: Optional[str]
    agent_name: Optional[str]
    repository: Optional[str]
    model: Optional[str]
    provider_error_present: bool
    input_tokens: int
    output_tokens: int
    tool_call_count: int
    empty_output: bool
    no_tool_calls: bool
    message_count: int
    final_message_role: Optional[str]
    final_user_tool_result: bool
    final_tool_result_image_base64_max_bytes: int
    max_tool_result_image_base64_bytes: int
    tool_result_image_count: int
    large_tool_result_payload_risk: bool
    empty_completion_failure: bool
    invalid_tool_call_error_count: int
    invalid_tool_call_error_markers: List[str]
    mutating_tool_uses_before_destructive_checkout: int
    destructive_checkout_command: Optional[str]
    destructive_checkout_after_work: bool
    trace_quality_score: float
    reasons: List[str]
    errors: List[str] = field(default_factory=list)

    def to_json(self) -> Dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "session_id": self.session_id,
            "observation_id": self.observation_id,
            "source": self.source,
            "source_locator": self.source_locator,
            "agent_name": self.agent_name,
            "repository": self.repository,
            "model": self.model,
            "provider_error_present": self.provider_error_present,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "tool_call_count": self.tool_call_count,
            "empty_output": self.empty_output,
            "no_tool_calls": self.no_tool_calls,
            "message_count": self.message_count,
            "final_message_role": self.final_message_role,
            "final_user_tool_result": self.final_user_tool_result,
            "final_tool_result_image_base64_max_bytes": (
                self.final_tool_result_image_base64_max_bytes
            ),
            "max_tool_result_image_base64_bytes": self.max_tool_result_image_base64_bytes,
            "tool_result_image_count": self.tool_result_image_count,
            "large_tool_result_payload_risk": self.large_tool_result_payload_risk,
            "empty_completion_failure": self.empty_completion_failure,
            "invalid_tool_call_error_count": self.invalid_tool_call_error_count,
            "invalid_tool_call_error_markers": self.invalid_tool_call_error_markers,
            "mutating_tool_uses_before_destructive_checkout": (
                self.mutating_tool_uses_before_destructive_checkout
            ),
            "destructive_checkout_command": self.destructive_checkout_command,
            "destructive_checkout_after_work": self.destructive_checkout_after_work,
            "trace_quality_score": self.trace_quality_score,
            "reasons": self.reasons,
            "errors": self.errors,
        }


@dataclass(frozen=True)
class LangfuseScore:
    name: str
    value: float
    data_type: str
    trace_id: Optional[str]
    observation_id: Optional[str]
    session_id: Optional[str]
    comment: str

    def payload(self) -> Dict[str, Any]:
        identity = self.trace_id or self.session_id or ""
        score_id = str(
            uuid.uuid5(
                uuid.NAMESPACE_URL,
                f"aawm-agent-score:{self.name}:{identity}:{self.observation_id or ''}",
            )
        )
        payload: Dict[str, Any] = {
            "id": score_id,
            "name": self.name,
            "value": self.value,
            "dataType": self.data_type,
            "comment": self.comment[:500],
        }
        if self.trace_id:
            payload["traceId"] = self.trace_id
        if self.observation_id:
            payload["observationId"] = self.observation_id
        if self.session_id and not self.trace_id:
            payload["sessionId"] = self.session_id
        return payload


class ClickHouseClient:
    def __init__(self, *, url: str, user: str, password: str, timeout_seconds: int) -> None:
        self.url = url.rstrip("/")
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
        return [json.loads(line) for line in payload.splitlines() if line.strip()]


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
            raise RuntimeError("boto3/botocore are required for MinIO payloads") from exc

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
            return json.loads(body.read().decode("utf-8", errors="replace"))


def _normalize_clickhouse_url(value: Optional[str]) -> str:
    cleaned = _clean(value) or "http://127.0.0.1:8123"
    if not cleaned.startswith(("http://", "https://")):
        cleaned = f"http://{cleaned}"
    return cleaned.rstrip("/").replace("clickhouse", "127.0.0.1")


def _postgres_dsn_from_args(args: argparse.Namespace) -> str:
    if args.pg_dsn:
        return args.pg_dsn
    host = args.pg_host or os.getenv("AAWM_DB_HOST") or "127.0.0.1"
    port = args.pg_port or os.getenv("AAWM_DB_PORT") or "5434"
    db_name = args.target_db_name
    user = args.pg_user or os.getenv("AAWM_DB_USER") or "aawm"
    password = args.pg_password or os.getenv("AAWM_DB_PASSWORD") or "aawm_dev"
    return (
        f"postgresql://{quote(user)}:{quote(password)}@"
        f"{host}:{port}/{quote(db_name)}"
    )


def _resolve_clickhouse_client(args: argparse.Namespace) -> ClickHouseClient:
    return ClickHouseClient(
        url=_normalize_clickhouse_url(
            args.clickhouse_url
            or os.getenv("CLICKHOUSE_URL")
            or os.getenv("LANGFUSE_CLICKHOUSE_URL")
        ),
        user=(
            args.clickhouse_user
            or os.getenv("CLICKHOUSE_USER")
            or os.getenv("LANGFUSE_CLICKHOUSE_USER")
            or "clickhouse"
        ),
        password=(
            args.clickhouse_password
            or os.getenv("CLICKHOUSE_PASSWORD")
            or os.getenv("LANGFUSE_CLICKHOUSE_PASSWORD")
            or "clickhouse"
        ),
        timeout_seconds=args.clickhouse_timeout_seconds,
    )


def _resolve_minio_client(args: argparse.Namespace) -> MinioEventBlobClient:
    return MinioEventBlobClient(
        endpoint_url=(
            args.minio_endpoint
            or os.getenv("LANGFUSE_S3_EVENT_UPLOAD_ENDPOINT")
            or os.getenv("MINIO_ENDPOINT")
            or "http://127.0.0.1:9010"
        ),
        access_key_id=(
            args.minio_access_key
            or os.getenv("LANGFUSE_S3_EVENT_UPLOAD_ACCESS_KEY_ID")
            or os.getenv("MINIO_ROOT_USER")
            or "langfuse"
        ),
        secret_access_key=(
            args.minio_secret_key
            or os.getenv("LANGFUSE_S3_EVENT_UPLOAD_SECRET_ACCESS_KEY")
            or os.getenv("MINIO_ROOT_PASSWORD")
            or "langfuse-secret"
        ),
    )


def _verify_target_database(conn: psycopg.Connection, required_database: Optional[str]) -> str:
    with conn.cursor() as cur:
        cur.execute("select current_database()")
        current_database = str(cur.fetchone()[0])
    if required_database and current_database != required_database:
        raise RuntimeError(
            f"Refusing to use database {current_database!r}; expected {required_database!r}"
        )
    return current_database


def _metadata_dict(value: Any) -> Dict[str, Any]:
    parsed = _parse_json_value(value)
    return parsed if isinstance(parsed, dict) else {}


def _fetch_session_candidates(args: argparse.Namespace) -> List[SessionCandidate]:
    predicates = ["true"]
    params: Dict[str, Any] = {}
    if args.trace_id:
        predicates.append("trace_id = ANY(%(trace_ids)s)")
        params["trace_ids"] = args.trace_id
    if args.session_id:
        predicates.append("session_id = ANY(%(session_ids)s)")
        params["session_ids"] = args.session_id
    if args.repository:
        predicates.append("repository ILIKE %(repository)s")
        params["repository"] = f"%{args.repository}%"
    if args.agent_name:
        predicates.append("agent_name ILIKE %(agent_name)s")
        params["agent_name"] = f"%{args.agent_name}%"
    if args.model:
        predicates.append("model ILIKE %(model)s")
        params["model"] = f"%{args.model}%"
    if args.provider:
        predicates.append("provider = %(provider)s")
        params["provider"] = args.provider
    from_created_at = _parse_optional_datetime(args.from_created_at)
    to_created_at = _parse_optional_datetime(args.to_created_at)
    if from_created_at:
        predicates.append("created_at >= %(from_created_at)s")
        params["from_created_at"] = from_created_at
    if to_created_at:
        predicates.append("created_at <= %(to_created_at)s")
        params["to_created_at"] = to_created_at
    if args.candidate_only:
        predicates.append("coalesce(output_tokens, 0) <= %(max_output_tokens)s")
        predicates.append("coalesce(tool_call_count, 0) <= %(max_tool_call_count)s")
        params["max_output_tokens"] = args.max_output_tokens
        params["max_tool_call_count"] = args.max_tool_call_count

    query = f"""
SELECT
  id,
  created_at,
  trace_id,
  session_id,
  litellm_call_id,
  coalesce(metadata->>'source_observation_id', litellm_call_id) AS source_observation_id,
  provider,
  model,
  agent_name,
  repository,
  tenant_id,
  coalesce(input_tokens, 0) AS input_tokens,
  coalesce(output_tokens, 0) AS output_tokens,
  coalesce(tool_call_count, 0) AS tool_call_count,
  coalesce(invalid_tool_call_count, 0) AS invalid_tool_call_count,
  coalesce(metadata, '{{}}'::jsonb) AS metadata
FROM public.session_history
WHERE {' AND '.join(predicates)}
ORDER BY created_at DESC, id DESC
LIMIT %(limit)s
"""
    params["limit"] = max(1, args.limit)
    with psycopg.connect(_postgres_dsn_from_args(args)) as conn:
        _verify_target_database(conn, args.require_target_database)
        with conn.cursor(row_factory=psycopg.rows.dict_row) as cur:
            cur.execute(query, params)
            rows = cur.fetchall()

    candidates: List[SessionCandidate] = []
    for row in rows:
        candidates.append(
            SessionCandidate(
                row_id=int(row["id"]),
                created_at=str(row["created_at"]) if row.get("created_at") else None,
                trace_id=_clean(row.get("trace_id")),
                session_id=_clean(row.get("session_id")),
                litellm_call_id=_clean(row.get("litellm_call_id")),
                source_observation_id=_clean(row.get("source_observation_id")),
                provider=_clean(row.get("provider")),
                model=_clean(row.get("model")),
                agent_name=_clean(row.get("agent_name")),
                repository=_clean(row.get("repository")),
                tenant_id=_clean(row.get("tenant_id")),
                input_tokens=_coerce_int(row.get("input_tokens")),
                output_tokens=_coerce_int(row.get("output_tokens")),
                tool_call_count=_coerce_int(row.get("tool_call_count")),
                invalid_tool_call_count=_coerce_int(row.get("invalid_tool_call_count")),
                metadata=_metadata_dict(row.get("metadata")),
            )
        )
    return candidates


def _fetch_provider_error_keys(
    args: argparse.Namespace,
    candidates: Sequence[SessionCandidate],
) -> Tuple[set[str], set[str]]:
    trace_ids = sorted({candidate.trace_id for candidate in candidates if candidate.trace_id})
    call_ids = sorted(
        {
            candidate.source_observation_id or candidate.litellm_call_id
            for candidate in candidates
            if candidate.source_observation_id or candidate.litellm_call_id
        }
    )
    if not trace_ids and not call_ids:
        return set(), set()
    predicates: List[str] = []
    params: Dict[str, Any] = {}
    if trace_ids:
        predicates.append("trace_id = ANY(%(trace_ids)s)")
        params["trace_ids"] = trace_ids
    if call_ids:
        predicates.append("litellm_call_id = ANY(%(call_ids)s)")
        params["call_ids"] = call_ids
    query = f"""
SELECT trace_id, litellm_call_id
FROM public.provider_error_observations
WHERE {' OR '.join(predicates)}
"""
    with psycopg.connect(_postgres_dsn_from_args(args)) as conn:
        _verify_target_database(conn, args.require_target_database)
        with conn.cursor(row_factory=psycopg.rows.dict_row) as cur:
            cur.execute(query, params)
            rows = cur.fetchall()
    return (
        {str(row["trace_id"]) for row in rows if row.get("trace_id")},
        {str(row["litellm_call_id"]) for row in rows if row.get("litellm_call_id")},
    )


def _fetch_clickhouse_observations(
    client: ClickHouseClient,
    candidates: Sequence[SessionCandidate],
) -> Dict[str, ObservationPayload]:
    observation_ids = sorted(
        {
            candidate.source_observation_id or candidate.litellm_call_id
            for candidate in candidates
            if candidate.source_observation_id or candidate.litellm_call_id
        }
    )
    trace_ids = sorted({candidate.trace_id for candidate in candidates if candidate.trace_id})
    predicates: List[str] = ["type = 'GENERATION'", "is_deleted = 0"]
    id_predicates: List[str] = []
    if observation_ids:
        id_predicates.append(f"id IN {_quote_clickhouse_string_array(observation_ids)}")
    if trace_ids:
        id_predicates.append(f"trace_id IN {_quote_clickhouse_string_array(trace_ids)}")
    if not id_predicates:
        return {}
    predicates.append("(" + " OR ".join(id_predicates) + ")")
    query = f"""
SELECT
  id,
  trace_id,
  start_time,
  end_time,
  name,
  metadata,
  input,
  output,
  provided_model_name,
  environment
FROM observations
WHERE {' AND '.join(predicates)}
ORDER BY start_time DESC
FORMAT JSONEachRow
"""
    payloads: Dict[str, ObservationPayload] = {}
    for row in client.request_rows(query):
        body = {
            "id": row.get("id"),
            "traceId": row.get("trace_id"),
            "startTime": row.get("start_time"),
            "endTime": row.get("end_time"),
            "name": row.get("name"),
            "metadata": row.get("metadata"),
            "input": row.get("input"),
            "output": row.get("output"),
            "model": row.get("provided_model_name"),
            "environment": row.get("environment"),
        }
        payload = ObservationPayload(
            observation_id=_clean(row.get("id")),
            trace_id=_clean(row.get("trace_id")),
            body=body,
            source="clickhouse",
            source_locator=_clean(row.get("id")),
        )
        if payload.observation_id:
            payloads[payload.observation_id] = payload
        if payload.trace_id and payload.trace_id not in payloads:
            payloads[payload.trace_id] = payload
    return payloads


def _fetch_blob_log_rows(
    client: ClickHouseClient,
    candidates: Sequence[SessionCandidate],
    *,
    extra_observation_ids: Sequence[str] = (),
) -> Dict[str, Dict[str, Any]]:
    observation_ids = sorted(
        {
            candidate.source_observation_id or candidate.litellm_call_id
            for candidate in candidates
            if candidate.source_observation_id or candidate.litellm_call_id
        }.union({_clean(value) for value in extra_observation_ids if _clean(value)})
    )
    if not observation_ids:
        return {}
    query = f"""
SELECT
  entity_id,
  bucket_name,
  bucket_path,
  created_at
FROM blob_storage_file_log
WHERE is_deleted = 0
  AND entity_type = 'observation'
  AND entity_id IN {_quote_clickhouse_string_array(observation_ids)}
ORDER BY created_at DESC
FORMAT JSONEachRow
"""
    rows_by_entity: Dict[str, Dict[str, Any]] = {}
    for row in client.request_rows(query):
        entity_id = _clean(row.get("entity_id"))
        if entity_id and entity_id not in rows_by_entity:
            rows_by_entity[entity_id] = row
    return rows_by_entity


def _iter_langfuse_events(blob_payload: Any) -> Iterator[Dict[str, Any]]:
    parsed = _parse_json_value(blob_payload)
    if isinstance(parsed, dict):
        parsed = [parsed]
    if not isinstance(parsed, list):
        return
    for event in parsed:
        if isinstance(event, dict):
            yield event


def _body_from_langfuse_blob_payload(blob_payload: Any) -> Optional[Dict[str, Any]]:
    for event in _iter_langfuse_events(blob_payload):
        body = event.get("body")
        if isinstance(body, dict):
            return dict(body)
        if {"input", "output", "traceId"}.intersection(event):
            return dict(event)
    return None


def _resolve_observation_payloads(
    args: argparse.Namespace,
    candidates: Sequence[SessionCandidate],
) -> Dict[str, ObservationPayload]:
    if not candidates:
        return {}
    clickhouse_client = _resolve_clickhouse_client(args)
    clickhouse_payloads = _fetch_clickhouse_observations(clickhouse_client, candidates)
    if args.source_mode == "clickhouse":
        return clickhouse_payloads

    payloads: Dict[str, ObservationPayload] = {}
    minio_error: Optional[str] = None
    try:
        clickhouse_observation_ids = sorted(
            {
                payload.observation_id
                for payload in clickhouse_payloads.values()
                if payload.observation_id
            }
        )
        blob_rows = _fetch_blob_log_rows(
            clickhouse_client,
            candidates,
            extra_observation_ids=clickhouse_observation_ids,
        )
        minio_client = _resolve_minio_client(args)
        for candidate in candidates:
            clickhouse_hint = _payload_for_candidate(clickhouse_payloads, candidate)
            candidate_keys = [
                candidate.source_observation_id,
                candidate.litellm_call_id,
                clickhouse_hint.observation_id if clickhouse_hint is not None else None,
            ]
            blob_row = next(
                (blob_rows[key] for key in candidate_keys if key and key in blob_rows),
                None,
            )
            if not blob_row:
                continue
            observation_id = _clean(blob_row.get("entity_id")) or next(
                (key for key in candidate_keys if key),
                None,
            )
            blob_payload = minio_client.read_event_blob(
                str(blob_row["bucket_name"]),
                str(blob_row["bucket_path"]),
            )
            body = _body_from_langfuse_blob_payload(blob_payload)
            if body is None:
                continue
            payload = ObservationPayload(
                observation_id=observation_id,
                trace_id=candidate.trace_id,
                body=body,
                source="minio",
                source_locator=str(blob_row.get("bucket_path") or ""),
            )
            for key in (*candidate_keys, candidate.trace_id):
                if key:
                    payloads[key] = payload
    except Exception as exc:
        minio_error = str(exc)
        if args.source_mode == "minio":
            raise

    if args.source_mode == "minio":
        return payloads
    for key, payload in clickhouse_payloads.items():
        if key not in payloads:
            next_body = dict(payload.body)
            if minio_error:
                metadata = _metadata_dict(next_body.get("metadata"))
                metadata["agent_score_minio_error"] = minio_error
                next_body["metadata"] = metadata
            payloads[key] = ObservationPayload(
                observation_id=payload.observation_id,
                trace_id=payload.trace_id,
                body=next_body,
                source="clickhouse_fallback" if minio_error else payload.source,
                source_locator=payload.source_locator,
            )
    return payloads


def _extract_anthropic_request_from_langfuse_input(value: Any) -> Dict[str, Any]:
    parsed = _parse_json_value(value)
    if not isinstance(parsed, dict):
        return {}
    messages = parsed.get("messages")
    if isinstance(messages, list):
        for message in messages:
            if not isinstance(message, dict):
                continue
            content = message.get("content")
            nested = _parse_json_value(content)
            if isinstance(nested, dict) and isinstance(nested.get("messages"), list):
                return nested
        return parsed
    body = parsed.get("body")
    if isinstance(body, dict):
        return _extract_anthropic_request_from_langfuse_input(body)
    return {}


def _content_blocks(message: Dict[str, Any]) -> List[Any]:
    blocks: List[Any] = []
    content = message.get("content")
    parsed = _parse_json_value(content)
    if isinstance(parsed, list):
        blocks.extend(parsed)
    elif isinstance(parsed, dict):
        blocks.append(parsed)
    elif isinstance(content, list):
        blocks.extend(content)
    elif isinstance(content, dict):
        blocks.append(content)
    elif message.get("role") == "tool":
        blocks.append({"type": "tool_result", "content": content})

    tool_calls = message.get("tool_calls")
    if isinstance(tool_calls, list):
        for tool_call in tool_calls:
            if not isinstance(tool_call, dict):
                continue
            function = tool_call.get("function")
            if isinstance(function, dict):
                blocks.append(
                    {
                        "type": "tool_use",
                        "name": function.get("name"),
                        "input": function.get("arguments"),
                    }
                )
            else:
                blocks.append(
                    {
                        "type": "tool_use",
                        "name": tool_call.get("name"),
                        "input": tool_call.get("arguments") or tool_call.get("input"),
                    }
                )
    return blocks


def _iter_nested_blocks(value: Any) -> Iterator[Dict[str, Any]]:
    parsed = _parse_json_value(value)
    if isinstance(parsed, dict):
        yield parsed
        for nested in parsed.values():
            yield from _iter_nested_blocks(nested)
    elif isinstance(parsed, list):
        for item in parsed:
            yield from _iter_nested_blocks(item)


def _collect_tool_results(messages: Sequence[Any]) -> List[ToolResultEvidence]:
    results: List[ToolResultEvidence] = []
    final_index = len(messages) - 1
    for message_index, message in enumerate(messages):
        if not isinstance(message, dict):
            continue
        for block in _content_blocks(message):
            if not isinstance(block, dict) or block.get("type") != "tool_result":
                continue
            evidence = ToolResultEvidence(
                message_index=message_index,
                is_final_message=message_index == final_index,
            )
            content = block.get("content")
            for nested in _iter_nested_blocks(content):
                source = nested.get("source")
                if isinstance(source, dict) and source.get("type") == "base64":
                    data = source.get("data")
                    if isinstance(data, str):
                        evidence.image_base64_lengths.append(len(data))
                if nested.get("type") == "image":
                    source = nested.get("source")
                    if isinstance(source, dict) and isinstance(source.get("data"), str):
                        evidence.image_base64_lengths.append(len(source["data"]))
                if nested.get("type") == "text" and isinstance(nested.get("text"), str):
                    evidence.text_lengths.append(len(nested["text"]))
            if isinstance(content, str):
                evidence.text_lengths.append(len(content))
            results.append(evidence)
    return results


_DESTRUCTIVE_GIT_PATTERNS = (
    re.compile(r"\bgit\s+reset\s+--hard\b"),
    re.compile(r"\bgit\s+checkout\b(?=[^\n;]*\s(?:--|-f|HEAD\b))"),
    re.compile(r"\bgit\s+checkout\s+(?:\.|:[/\\]|--\s*(?:\.|/|[A-Za-z0-9_.\-/]))"),
    re.compile(r"\bgit\s+restore\b(?=[^\n;]*(?:\.|--source|--worktree|--staged|:/))"),
)
_SAFE_CHECKOUT_PATTERNS = (
    re.compile(r"\bgit\s+checkout\s+-b\b"),
    re.compile(r"\bgit\s+checkout\s+--detach\b"),
)
_MUTATING_BASH_PATTERNS = (
    re.compile(r"\bapply_patch\b"),
    re.compile(r"\bsed\s+-i\b"),
    re.compile(r"\btee\s+[-A-Za-z0-9_/.\"]+"),
    re.compile(r">\s*[-A-Za-z0-9_/.\"]+"),
    re.compile(r"\bgit\s+apply\b"),
)
_MUTATING_TOOL_NAMES = {
    "edit",
    "multiedit",
    "notebookedit",
    "write",
}
_INVALID_TOOL_CALL_ERROR_MARKERS = (
    ("InputValidationError", re.compile(r"\bInputValidationError\b", re.IGNORECASE)),
    ("tool_use_error", re.compile(r"<\s*tool_use_error\s*>", re.IGNORECASE)),
    ("unexpected_parameter", re.compile(r"unexpected\s+parameter", re.IGNORECASE)),
    ("invalid_tool", re.compile(r"\binvalid\s+tool\b", re.IGNORECASE)),
    (
        "unrecognized_key_or_parameter",
        re.compile(r"unrecognized\s+(?:key|parameter)", re.IGNORECASE),
    ),
)


def _command_from_tool_input(tool_input: Any) -> Optional[str]:
    parsed = _parse_json_value(tool_input)
    if isinstance(parsed, dict):
        for key in ("cmd", "command", "script"):
            value = parsed.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    if isinstance(tool_input, str) and tool_input.strip():
        return tool_input.strip()
    return None


def _iter_text_values(value: Any) -> Iterator[str]:
    parsed = _parse_json_value(value)
    if isinstance(parsed, str):
        yield parsed
        return
    if isinstance(parsed, list):
        for item in parsed:
            yield from _iter_text_values(item)
        return
    if isinstance(parsed, dict):
        for key, nested in parsed.items():
            if key == "type":
                continue
            yield from _iter_text_values(nested)
        return


def _collect_invalid_tool_call_errors(
    messages: Sequence[Any],
) -> Tuple[List[str], int]:
    seen_markers: List[str] = []
    count = 0
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = _clean(message.get("role"))
        if role == "tool":
            blocks: Sequence[Any] = ({"type": "tool_result", "content": message.get("content")},)
        elif role == "user":
            blocks = [
                block
                for block in _content_blocks(message)
                if isinstance(block, dict) and block.get("type") == "tool_result"
            ]
        else:
            continue

        for block in blocks:
            block_has_error = False
            if not isinstance(block, dict):
                continue
            for text in _iter_text_values(block.get("content")):
                for marker_name, marker_re in _INVALID_TOOL_CALL_ERROR_MARKERS:
                    if marker_re.search(text):
                        if marker_name not in seen_markers:
                            seen_markers.append(marker_name)
                        block_has_error = True
                if block_has_error:
                    break
            if block_has_error:
                count += 1
    return seen_markers, count


def _is_destructive_checkout_command(command: Optional[str]) -> bool:
    if not command:
        return False
    if any(pattern.search(command) for pattern in _SAFE_CHECKOUT_PATTERNS):
        return False
    return any(pattern.search(command) for pattern in _DESTRUCTIVE_GIT_PATTERNS)


def _is_mutating_tool_use(tool_name: str, command: Optional[str]) -> bool:
    normalized = tool_name.strip().lower()
    if normalized in _MUTATING_TOOL_NAMES:
        return True
    if normalized == "bash" and command:
        return any(pattern.search(command) for pattern in _MUTATING_BASH_PATTERNS)
    return False


def _collect_tool_uses(messages: Sequence[Any]) -> List[ToolUseEvidence]:
    uses: List[ToolUseEvidence] = []
    sequence_index = 0
    for message_index, message in enumerate(messages):
        if not isinstance(message, dict):
            continue
        for block in _content_blocks(message):
            if not isinstance(block, dict) or block.get("type") != "tool_use":
                continue
            name = _clean(block.get("name")) or ""
            command = _command_from_tool_input(block.get("input"))
            uses.append(
                ToolUseEvidence(
                    message_index=message_index,
                    sequence_index=sequence_index,
                    name=name,
                    command=command,
                    mutating=_is_mutating_tool_use(name, command),
                    destructive_checkout=_is_destructive_checkout_command(command),
                )
            )
            sequence_index += 1
    return uses


def _detect_destructive_checkout_after_work(
    tool_uses: Sequence[ToolUseEvidence],
) -> Tuple[bool, int, Optional[str]]:
    prior_mutating = 0
    for tool_use in tool_uses:
        if tool_use.destructive_checkout and prior_mutating > 0:
            return True, prior_mutating, tool_use.command
        if tool_use.mutating:
            prior_mutating += 1
    return False, prior_mutating, None


def _extract_assistant_output(output: Any) -> Tuple[str, int]:
    parsed = _parse_json_value(output)
    if not isinstance(parsed, dict):
        return (str(parsed or ""), 0)
    message = parsed
    choices = parsed.get("choices")
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        maybe_message = choices[0].get("message")
        if isinstance(maybe_message, dict):
            message = maybe_message
    content = message.get("content")
    text_parts: List[str] = []
    tool_call_count = 0
    if isinstance(content, str):
        text_parts.append(content)
    elif isinstance(content, list):
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") in {"text", "output_text"} and isinstance(
                block.get("text"), str
            ):
                text_parts.append(block["text"])
            if block.get("type") in {"tool_use", "tool_call"}:
                tool_call_count += 1
    for key in ("tool_calls", "toolCalls"):
        tool_calls = message.get(key)
        if isinstance(tool_calls, list):
            tool_call_count += len(tool_calls)
    return "".join(text_parts), tool_call_count


def _payload_for_candidate(
    payloads: Dict[str, ObservationPayload],
    candidate: SessionCandidate,
) -> Optional[ObservationPayload]:
    for key in (
        candidate.source_observation_id,
        candidate.litellm_call_id,
        candidate.trace_id,
    ):
        if key and key in payloads:
            return payloads[key]
    return None


def score_candidate(
    candidate: SessionCandidate,
    payload: Optional[ObservationPayload],
    *,
    provider_error_present: bool,
    max_output_tokens: int,
    large_base64_threshold: int,
) -> TraceScoreEvidence:
    errors: List[str] = []
    body = payload.body if payload is not None else {}
    if payload is None:
        errors.append("missing_observation_payload")
    request_body = _extract_anthropic_request_from_langfuse_input(body.get("input"))
    messages = request_body.get("messages") if isinstance(request_body, dict) else None
    if not isinstance(messages, list):
        messages = []
        errors.append("missing_request_messages")
    tool_results = _collect_tool_results(messages)
    tool_uses = _collect_tool_uses(messages)
    destructive_checkout, prior_mutating, destructive_command = (
        _detect_destructive_checkout_after_work(tool_uses)
    )

    final_message = messages[-1] if messages and isinstance(messages[-1], dict) else {}
    final_role = _clean(final_message.get("role")) if isinstance(final_message, dict) else None
    final_tool_results = [result for result in tool_results if result.is_final_message]
    final_user_tool_result = final_role == "user" and bool(final_tool_results)
    final_image_lengths = [
        length
        for result in final_tool_results
        for length in result.image_base64_lengths
    ]
    all_image_lengths = [
        length for result in tool_results for length in result.image_base64_lengths
    ]
    final_image_max = max(final_image_lengths, default=0)
    all_image_max = max(all_image_lengths, default=0)
    large_payload_risk = all_image_max >= large_base64_threshold

    assistant_text, output_tool_calls = _extract_assistant_output(body.get("output"))
    no_tool_calls = candidate.tool_call_count == 0 and output_tool_calls == 0
    empty_output = (
        candidate.output_tokens <= max_output_tokens
        and not assistant_text.strip()
        and no_tool_calls
    )
    invalid_tool_call_error_markers, payload_invalid_tool_call_error_count = (
        _collect_invalid_tool_call_errors(messages)
    )
    invalid_tool_call_error_count = max(
        payload_invalid_tool_call_error_count,
        candidate.invalid_tool_call_count,
    )
    if (
        candidate.invalid_tool_call_count > payload_invalid_tool_call_error_count
        and "session_history_invalid_tool_call_count" not in invalid_tool_call_error_markers
    ):
        invalid_tool_call_error_markers.append("session_history_invalid_tool_call_count")
    empty_completion_failure = (
        empty_output
        and final_user_tool_result
        and final_image_max >= large_base64_threshold
        and not provider_error_present
    )

    reasons: List[str] = []
    if empty_completion_failure:
        reasons.append("empty_completion_after_large_final_tool_result")
    if destructive_checkout:
        reasons.append("destructive_checkout_after_mutating_tool_use")
    if provider_error_present and empty_output:
        reasons.append("empty_output_has_provider_error")
    if large_payload_risk and not empty_completion_failure:
        reasons.append("large_tool_result_payload_seen")
    if invalid_tool_call_error_count:
        reasons.append("invalid_tool_call_error_seen")
    failed = empty_completion_failure or destructive_checkout
    return TraceScoreEvidence(
        trace_id=candidate.trace_id,
        session_id=candidate.session_id,
        observation_id=(
            payload.observation_id if payload is not None else candidate.source_observation_id
        ),
        source=payload.source if payload is not None else "missing",
        source_locator=payload.source_locator if payload is not None else None,
        agent_name=candidate.agent_name,
        repository=candidate.repository,
        model=candidate.model,
        provider_error_present=provider_error_present,
        input_tokens=candidate.input_tokens,
        output_tokens=candidate.output_tokens,
        tool_call_count=candidate.tool_call_count,
        empty_output=empty_output,
        no_tool_calls=no_tool_calls,
        message_count=len(messages),
        final_message_role=final_role,
        final_user_tool_result=final_user_tool_result,
        final_tool_result_image_base64_max_bytes=final_image_max,
        max_tool_result_image_base64_bytes=all_image_max,
        tool_result_image_count=len(all_image_lengths),
        large_tool_result_payload_risk=large_payload_risk,
        empty_completion_failure=empty_completion_failure,
        invalid_tool_call_error_count=invalid_tool_call_error_count,
        invalid_tool_call_error_markers=invalid_tool_call_error_markers,
        mutating_tool_uses_before_destructive_checkout=prior_mutating,
        destructive_checkout_command=(
            destructive_command[:240] if destructive_command is not None else None
        ),
        destructive_checkout_after_work=destructive_checkout,
        trace_quality_score=0.0 if failed else 1.0,
        reasons=reasons,
        errors=errors,
    )


def _score_comment(evidence: TraceScoreEvidence) -> str:
    parts = [
        ",".join(evidence.reasons) or "no_deterministic_failure",
        f"output_tokens={evidence.output_tokens}",
        f"tool_calls={evidence.tool_call_count}",
        f"max_image_b64={evidence.max_tool_result_image_base64_bytes}",
        f"invalid_tool_call_errors={evidence.invalid_tool_call_error_count}",
        f"source={evidence.source}",
    ]
    if evidence.invalid_tool_call_error_markers:
        parts.append(
            "invalid_tool_call_error_markers="
            + ",".join(evidence.invalid_tool_call_error_markers)
        )
    if evidence.destructive_checkout_command:
        digest = hashlib.sha256(evidence.destructive_checkout_command.encode()).hexdigest()[:12]
        parts.append(f"destructive_cmd_sha256={digest}")
    return "; ".join(parts)


def build_langfuse_scores(evidence: TraceScoreEvidence) -> List[LangfuseScore]:
    trace_id = evidence.trace_id
    observation_id = evidence.observation_id if trace_id else None
    session_id = evidence.session_id
    comment = _score_comment(evidence)
    return [
        LangfuseScore(
            name="aawm.agent.trace_quality",
            value=evidence.trace_quality_score,
            data_type="NUMERIC",
            trace_id=trace_id,
            observation_id=observation_id,
            session_id=session_id,
            comment=comment,
        ),
        LangfuseScore(
            name="aawm.agent.empty_completion_failure",
            value=1.0 if evidence.empty_completion_failure else 0.0,
            data_type="BOOLEAN",
            trace_id=trace_id,
            observation_id=observation_id,
            session_id=session_id,
            comment=comment,
        ),
        LangfuseScore(
            name="aawm.agent.large_tool_result_payload_risk",
            value=1.0 if evidence.large_tool_result_payload_risk else 0.0,
            data_type="BOOLEAN",
            trace_id=trace_id,
            observation_id=observation_id,
            session_id=session_id,
            comment=comment,
        ),
        LangfuseScore(
            name="aawm.agent.destructive_checkout_after_work",
            value=1.0 if evidence.destructive_checkout_after_work else 0.0,
            data_type="BOOLEAN",
            trace_id=trace_id,
            observation_id=observation_id,
            session_id=session_id,
            comment=comment,
        ),
        LangfuseScore(
            name="aawm.agent.invalid_tool_call_error",
            value=1.0 if evidence.invalid_tool_call_error_count > 0 else 0.0,
            data_type="BOOLEAN",
            trace_id=trace_id,
            observation_id=observation_id,
            session_id=session_id,
            comment=comment,
        ),
    ]


class LangfuseScoreClient:
    def __init__(
        self,
        *,
        host: str,
        public_key: str,
        secret_key: str,
        timeout_seconds: int,
    ) -> None:
        self.host = host.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self._auth_header = "Basic " + base64.b64encode(
            f"{public_key}:{secret_key}".encode("utf-8")
        ).decode("ascii")

    def create_score(self, score: LangfuseScore) -> Dict[str, Any]:
        request = Request(
            url=f"{self.host}/api/public/scores",
            headers={
                "Authorization": self._auth_header,
                "Content-Type": "application/json",
            },
            data=json.dumps(score.payload()).encode("utf-8"),
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                payload = response.read().decode("utf-8")
        except HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"Langfuse score write failed with HTTP {exc.code}: {error_body}"
            ) from exc
        return json.loads(payload) if payload.strip() else {}


def _resolve_langfuse_score_client(args: argparse.Namespace) -> LangfuseScoreClient:
    public_key = args.langfuse_public_key or os.getenv("LANGFUSE_PUBLIC_KEY")
    secret_key = args.langfuse_secret_key or os.getenv("LANGFUSE_SECRET_KEY")
    if not public_key or not secret_key:
        raise RuntimeError(
            "LANGFUSE_PUBLIC_KEY/LANGFUSE_SECRET_KEY or explicit Langfuse key args are required"
        )
    return LangfuseScoreClient(
        host=args.langfuse_host or os.getenv("LANGFUSE_HOST") or "http://127.0.0.1:3000",
        public_key=public_key,
        secret_key=secret_key,
        timeout_seconds=args.langfuse_timeout_seconds,
    )


def _run(args: argparse.Namespace) -> Dict[str, Any]:
    candidates = _fetch_session_candidates(args)
    provider_error_trace_ids, provider_error_call_ids = _fetch_provider_error_keys(
        args,
        candidates,
    )
    payloads = _resolve_observation_payloads(args, candidates)
    evidences: List[TraceScoreEvidence] = []
    for candidate in candidates:
        payload = _payload_for_candidate(payloads, candidate)
        provider_error_present = bool(
            (candidate.trace_id and candidate.trace_id in provider_error_trace_ids)
            or (
                candidate.source_observation_id
                and candidate.source_observation_id in provider_error_call_ids
            )
            or (candidate.litellm_call_id and candidate.litellm_call_id in provider_error_call_ids)
        )
        evidence = score_candidate(
            candidate,
            payload,
            provider_error_present=provider_error_present,
            max_output_tokens=args.max_output_tokens,
            large_base64_threshold=args.large_base64_threshold,
        )
        if args.include_passing or evidence.trace_quality_score < 1.0 or evidence.reasons:
            evidences.append(evidence)

    score_results: List[Dict[str, Any]] = []
    if args.apply and evidences:
        score_client = _resolve_langfuse_score_client(args)
        for evidence in evidences:
            for score in build_langfuse_scores(evidence):
                score_results.append(score_client.create_score(score))

    return {
        "applied": bool(args.apply),
        "candidate_count": len(candidates),
        "evidence_count": len(evidences),
        "score_write_count": len(score_results),
        "scores": [score.to_json() for score in evidences],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Deterministically score suspicious agent trace quality patterns."
    )
    parser.add_argument("--apply", action="store_true", help="Write scores to Langfuse.")
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--trace-id", action="append", default=[])
    parser.add_argument("--session-id", action="append", default=[])
    parser.add_argument("--repository", default=None)
    parser.add_argument("--agent-name", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--provider", default=None)
    parser.add_argument("--from-created-at", default=None)
    parser.add_argument("--to-created-at", default=None)
    parser.add_argument(
        "--candidate-only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Only scan low-output/no-tool-call session_history rows.",
    )
    parser.add_argument(
        "--include-passing",
        action="store_true",
        help="Include passing rows in output.",
    )
    parser.add_argument("--max-output-tokens", type=int, default=5)
    parser.add_argument("--max-tool-call-count", type=int, default=0)
    parser.add_argument("--large-base64-threshold", type=int, default=100_000)
    parser.add_argument(
        "--source-mode",
        choices=("auto", "minio", "clickhouse"),
        default="auto",
        help="Payload source. auto prefers MinIO and falls back to ClickHouse.",
    )
    parser.add_argument("--pg-dsn", default=None)
    parser.add_argument("--pg-host", default=None)
    parser.add_argument("--pg-port", default=None)
    parser.add_argument("--pg-user", default=None)
    parser.add_argument("--pg-password", default=None)
    parser.add_argument("--target-db-name", default="aawm_tristore")
    parser.add_argument("--require-target-database", default="aawm_tristore")
    parser.add_argument("--clickhouse-url", default=None)
    parser.add_argument("--clickhouse-user", default=None)
    parser.add_argument("--clickhouse-password", default=None)
    parser.add_argument("--clickhouse-timeout-seconds", type=int, default=90)
    parser.add_argument("--minio-endpoint", default=None)
    parser.add_argument("--minio-access-key", default=None)
    parser.add_argument("--minio-secret-key", default=None)
    parser.add_argument("--langfuse-host", default=None)
    parser.add_argument("--langfuse-public-key", default=None)
    parser.add_argument("--langfuse-secret-key", default=None)
    parser.add_argument("--langfuse-timeout-seconds", type=int, default=30)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = _run(args)
    sys.stdout.write(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
