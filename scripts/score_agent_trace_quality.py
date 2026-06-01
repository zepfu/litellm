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
import shlex
import subprocess
import sys
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
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

from litellm.integrations.aawm_agent_quality_rules import (  # noqa: E402
    AgentQualityCommand,
    score_agent_quality_context,
)


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


def _codex_transcript_session_history_provider(value: Any) -> Optional[str]:
    provider = _clean(value)
    if provider is None:
        return None
    if provider.lower() in {"litellm", "unknown", "none", "null"}:
        return None
    return provider


def _codex_transcript_session_history_model(value: Any) -> Optional[str]:
    model = _clean(value)
    if model is None:
        return "codex-transcript"
    if model.lower() == "aawm-codex-agent-auto":
        return "codex-transcript"
    return model


def _codex_transcript_session_history_model_group(value: Any) -> Optional[str]:
    model = _clean(value)
    if model is None:
        return None
    if model.lower() in {"aawm-codex-agent-auto", "codex-transcript"}:
        return None
    return model


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


def _coerce_float_or_none(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


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
    llm_upstream_elapsed_ms: Optional[float]
    total_server_elapsed_ms: Optional[float]
    ttft_ms: Optional[float]
    metadata: Dict[str, Any]


@dataclass(frozen=True)
class ObservationPayload:
    observation_id: Optional[str]
    trace_id: Optional[str]
    body: Dict[str, Any]
    source: str
    source_locator: Optional[str] = None


@dataclass(frozen=True)
class TranscriptCandidateBundle:
    candidate: SessionCandidate
    payload: ObservationPayload


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
    command_timestamp: Optional[str]
    affected_paths: List[str]
    forced_git_tracking: bool
    git_tracking_paths: List[str]
    mutating: bool
    destructive_checkout: bool


@dataclass
class TraceScoreEvidence:
    row_id: int
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
    score_payload_resolved: bool
    response_meaningfulness_score: Optional[float]
    read_only_policy_compliance_score: Optional[float]
    read_only_policy_violation_count: int
    read_only_policy_violation_reasons: List[str]
    instruction_adherence_score: Optional[float]
    answer_completeness_score: Optional[float]
    evidence_fidelity_score: Optional[float]
    tool_result_fidelity_score: Optional[float]
    error_attribution_quality_score: Optional[float]
    repetition_loop_risk_score: Optional[float]
    context_retention_score: Optional[float]
    tool_use_validity_score: Optional[float]
    tool_error_recovery_score: Optional[float]
    stall_risk_score: Optional[float]
    output_contract_compliance_score: Optional[float]
    task_progress_score: Optional[float]
    scope_control_score: Optional[float]
    destructive_action_policy_score: Optional[float]
    ignored_path_tracking_policy_score: Optional[float]
    ignored_path_tracking_violation_count: int
    baseline_deflection_attempted_score: Optional[float]
    baseline_deflection_incident_score: Optional[float]
    baseline_deflection_attempt_count: Optional[int]
    baseline_deflection_tool_call_count: Optional[int]
    baseline_deflection_input_tokens: Optional[int]
    baseline_deflection_elapsed_ms: Optional[float]
    quality_gate_trigger_count: Optional[int]
    quality_gate_fix_attempt_count: Optional[int]
    quality_gate_rerun_count: Optional[int]
    sleep_wellness_interruption_attempted_score: Optional[float]
    sleep_wellness_interruption_incident_score: Optional[float]
    sleep_wellness_interruption_count: Optional[int]
    sleep_wellness_interruption_output_tokens: Optional[int]
    sleep_wellness_interruption_input_tokens: Optional[int]
    sleep_wellness_interruption_elapsed_ms: Optional[float]
    sleep_wellness_interruption_after_user_pushback_count: Optional[int]
    sleep_wellness_interruption_repeated_count: Optional[int]
    terminal_completion_score: Optional[float]
    discovery_inventory_coverage_score: Optional[float]
    discovery_inventory_missing_count: Optional[int]
    read_only_instruction_evidence: List[Dict[str, Any]]
    read_only_policy_violation_evidence: List[Dict[str, Any]]
    ignored_path_tracking_evidence: List[Dict[str, Any]]
    agent_score_reasons: Dict[str, Any]
    reasons: List[str]
    errors: List[str] = field(default_factory=list)

    def to_json(self) -> Dict[str, Any]:
        return {
            "row_id": self.row_id,
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
            "score_payload_resolved": self.score_payload_resolved,
            "response_meaningfulness_score": self.response_meaningfulness_score,
            "read_only_policy_compliance_score": (
                self.read_only_policy_compliance_score
            ),
            "read_only_policy_violation_count": self.read_only_policy_violation_count,
            "read_only_policy_violation_reasons": (
                self.read_only_policy_violation_reasons
            ),
            "read_only_instruction_evidence": self.read_only_instruction_evidence,
            "read_only_policy_violation_evidence": (
                self.read_only_policy_violation_evidence
            ),
            "instruction_adherence_score": self.instruction_adherence_score,
            "answer_completeness_score": self.answer_completeness_score,
            "evidence_fidelity_score": self.evidence_fidelity_score,
            "tool_result_fidelity_score": self.tool_result_fidelity_score,
            "error_attribution_quality_score": self.error_attribution_quality_score,
            "repetition_loop_risk_score": self.repetition_loop_risk_score,
            "context_retention_score": self.context_retention_score,
            "tool_use_validity_score": self.tool_use_validity_score,
            "tool_error_recovery_score": self.tool_error_recovery_score,
            "stall_risk_score": self.stall_risk_score,
            "output_contract_compliance_score": (
                self.output_contract_compliance_score
            ),
            "task_progress_score": self.task_progress_score,
            "scope_control_score": self.scope_control_score,
            "destructive_action_policy_score": self.destructive_action_policy_score,
            "ignored_path_tracking_policy_score": (
                self.ignored_path_tracking_policy_score
            ),
            "ignored_path_tracking_violation_count": (
                self.ignored_path_tracking_violation_count
            ),
            "baseline_deflection_attempted_score": (
                self.baseline_deflection_attempted_score
            ),
            "baseline_deflection_incident_score": (
                self.baseline_deflection_incident_score
            ),
            "baseline_deflection_attempt_count": self.baseline_deflection_attempt_count,
            "baseline_deflection_tool_call_count": (
                self.baseline_deflection_tool_call_count
            ),
            "baseline_deflection_input_tokens": self.baseline_deflection_input_tokens,
            "baseline_deflection_elapsed_ms": self.baseline_deflection_elapsed_ms,
            "quality_gate_trigger_count": self.quality_gate_trigger_count,
            "quality_gate_fix_attempt_count": self.quality_gate_fix_attempt_count,
            "quality_gate_rerun_count": self.quality_gate_rerun_count,
            "sleep_wellness_interruption_attempted_score": (
                self.sleep_wellness_interruption_attempted_score
            ),
            "sleep_wellness_interruption_incident_score": (
                self.sleep_wellness_interruption_incident_score
            ),
            "sleep_wellness_interruption_count": (
                self.sleep_wellness_interruption_count
            ),
            "sleep_wellness_interruption_output_tokens": (
                self.sleep_wellness_interruption_output_tokens
            ),
            "sleep_wellness_interruption_input_tokens": (
                self.sleep_wellness_interruption_input_tokens
            ),
            "sleep_wellness_interruption_elapsed_ms": (
                self.sleep_wellness_interruption_elapsed_ms
            ),
            "sleep_wellness_interruption_after_user_pushback_count": (
                self.sleep_wellness_interruption_after_user_pushback_count
            ),
            "sleep_wellness_interruption_repeated_count": (
                self.sleep_wellness_interruption_repeated_count
            ),
            "terminal_completion_score": self.terminal_completion_score,
            "discovery_inventory_coverage_score": (
                self.discovery_inventory_coverage_score
            ),
            "discovery_inventory_missing_count": (
                self.discovery_inventory_missing_count
            ),
            "ignored_path_tracking_evidence": self.ignored_path_tracking_evidence,
            "agent_score_reasons": self.agent_score_reasons,
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
        max_pool_connections: int = 10,
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
            config=Config(
                signature_version="s3v4",
                max_pool_connections=max(10, max_pool_connections),
            ),
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
        max_pool_connections=max(
            10, int(getattr(args, "minio_read_workers", 1) or 1)
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
    if getattr(args, "missing_agent_quality_backfill_only", False):
        predicates.append(
            "("
            "baseline_deflection_attempted_score IS NULL "
            "OR sleep_wellness_interruption_attempted_score IS NULL"
            ")"
        )

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
  llm_upstream_elapsed_ms,
  total_server_elapsed_ms,
  ttft_ms,
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
                llm_upstream_elapsed_ms=_coerce_float_or_none(
                    row.get("llm_upstream_elapsed_ms")
                ),
                total_server_elapsed_ms=_coerce_float_or_none(
                    row.get("total_server_elapsed_ms")
                ),
                ttft_ms=_coerce_float_or_none(row.get("ttft_ms")),
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


def _chunked_strings(values: Sequence[str], size: int = 500) -> Iterator[List[str]]:
    for index in range(0, len(values), size):
        yield list(values[index : index + size])


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
    if not observation_ids and not trace_ids:
        return {}

    payloads: Dict[str, ObservationPayload] = {}

    def fetch_rows(id_predicate: str) -> None:
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
WHERE type = 'GENERATION'
  AND is_deleted = 0
  AND {id_predicate}
ORDER BY start_time DESC
FORMAT JSONEachRow
"""
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

    for chunk in _chunked_strings(observation_ids):
        fetch_rows(f"id IN {_quote_clickhouse_string_array(chunk)}")
    for chunk in _chunked_strings(trace_ids):
        fetch_rows(f"trace_id IN {_quote_clickhouse_string_array(chunk)}")
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
    rows_by_entity: Dict[str, Dict[str, Any]] = {}
    for chunk in _chunked_strings(observation_ids):
        query = f"""
SELECT
  entity_id,
  bucket_name,
  bucket_path,
  created_at
FROM blob_storage_file_log
WHERE is_deleted = 0
  AND entity_type = 'observation'
  AND entity_id IN {_quote_clickhouse_string_array(chunk)}
ORDER BY created_at DESC
FORMAT JSONEachRow
"""
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
    if args.source_mode == "clickhouse":
        return _fetch_clickhouse_observations(clickhouse_client, candidates)

    payloads: Dict[str, ObservationPayload] = {}
    minio_error: Optional[str] = None

    def load_minio_payloads(
        candidate_batch: Sequence[SessionCandidate],
        clickhouse_payload_hints: Dict[str, ObservationPayload],
    ) -> None:
        clickhouse_observation_ids = sorted(
            {
                payload.observation_id
                for payload in clickhouse_payload_hints.values()
                if payload.observation_id
            }
        )
        blob_rows = _fetch_blob_log_rows(
            clickhouse_client,
            candidate_batch,
            extra_observation_ids=clickhouse_observation_ids,
        )
        minio_client = _resolve_minio_client(args)

        def read_payload(
            task: Tuple[SessionCandidate, List[Optional[str]], Dict[str, Any]],
        ) -> Optional[Tuple[SessionCandidate, List[Optional[str]], ObservationPayload]]:
            candidate, candidate_keys, blob_row = task
            observation_id = _clean(blob_row.get("entity_id")) or next(
                (key for key in candidate_keys if key),
                None,
            )
            try:
                blob_payload = minio_client.read_event_blob(
                    str(blob_row["bucket_name"]),
                    str(blob_row["bucket_path"]),
                )
            except Exception as exc:
                error_code = (
                    getattr(exc, "response", {})
                    .get("Error", {})
                    .get("Code")
                )
                if error_code in {"NoSuchKey", "404", "NotFound"}:
                    return None
                raise
            body = _body_from_langfuse_blob_payload(blob_payload)
            if body is None:
                return None
            return (
                candidate,
                candidate_keys,
                ObservationPayload(
                    observation_id=observation_id,
                    trace_id=candidate.trace_id,
                    body=body,
                    source="minio",
                    source_locator=str(blob_row.get("bucket_path") or ""),
                ),
            )

        def store_payload(
            result: Optional[Tuple[SessionCandidate, List[Optional[str]], ObservationPayload]],
        ) -> None:
            if result is None:
                return
            candidate, candidate_keys, payload = result
            for key in (*candidate_keys, candidate.trace_id):
                if key:
                    payloads[key] = payload

        read_tasks: List[Tuple[SessionCandidate, List[Optional[str]], Dict[str, Any]]] = []
        for candidate in candidate_batch:
            if _payload_for_candidate(payloads, candidate) is not None:
                continue
            clickhouse_hint = _payload_for_candidate(
                clickhouse_payload_hints, candidate
            )
            candidate_keys = [
                candidate.source_observation_id,
                candidate.litellm_call_id,
                clickhouse_hint.observation_id if clickhouse_hint is not None else None,
            ]
            blob_row = next(
                (blob_rows[key] for key in candidate_keys if key and key in blob_rows),
                None,
            )
            if blob_row:
                read_tasks.append((candidate, candidate_keys, blob_row))

        worker_count = max(1, int(getattr(args, "minio_read_workers", 1) or 1))
        if worker_count == 1 or len(read_tasks) <= 1:
            for task in read_tasks:
                store_payload(read_payload(task))
            return

        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = [executor.submit(read_payload, task) for task in read_tasks]
            for future in as_completed(futures):
                store_payload(future.result())

    try:
        load_minio_payloads(candidates, {})
    except Exception as exc:
        minio_error = str(exc)
        if args.source_mode == "minio":
            raise

    if args.source_mode == "minio":
        return payloads

    missing_candidates = [
        candidate
        for candidate in candidates
        if _payload_for_candidate(payloads, candidate) is None
    ]
    clickhouse_payloads = _fetch_clickhouse_observations(
        clickhouse_client, missing_candidates
    )
    try:
        load_minio_payloads(missing_candidates, clickhouse_payloads)
    except Exception as exc:
        minio_error = str(exc)

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


def _iter_jsonl_events(path: Path) -> Iterator[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            parsed = json.loads(line)
            if not isinstance(parsed, dict):
                raise ValueError(f"{path}:{line_number} did not contain a JSON object")
            yield parsed


def _extract_spawn_message_for_agent(
    parent_transcript: Path,
    agent_id: str,
) -> Optional[str]:
    spawn_calls: Dict[str, Dict[str, Any]] = {}
    for event in _iter_jsonl_events(parent_transcript):
        if event.get("type") != "response_item":
            continue
        payload = event.get("payload")
        if not isinstance(payload, dict):
            continue
        payload_type = payload.get("type")
        call_id = _clean(payload.get("call_id"))
        if (
            payload_type == "function_call"
            and call_id
            and payload.get("name") == "spawn_agent"
        ):
            arguments = _parse_json_value(payload.get("arguments"))
            if isinstance(arguments, dict):
                spawn_calls[call_id] = arguments
            continue
        if payload_type != "function_call_output" or not call_id:
            continue
        output = _parse_json_value(payload.get("output"))
        if not isinstance(output, dict) or output.get("agent_id") != agent_id:
            continue
        arguments = spawn_calls.get(call_id)
        if not isinstance(arguments, dict):
            continue
        message = arguments.get("message")
        return message if isinstance(message, str) and message.strip() else None
    return None


def _repository_from_cwd(cwd: Optional[str]) -> Optional[str]:
    cleaned = _clean(cwd)
    if not cleaned:
        return None
    marker = "/home/zepfu/projects/"
    if marker in cleaned:
        remainder = cleaned.split(marker, 1)[1].strip("/")
        return remainder.split("/", 1)[0] if remainder else None
    return Path(cleaned).name or None


def _codex_output_text_from_message_payload(payload: Dict[str, Any]) -> Optional[str]:
    content = payload.get("content")
    if isinstance(content, list):
        text_parts: List[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "output_text":
                text = item.get("text")
                if isinstance(text, str):
                    text_parts.append(text)
        return "".join(text_parts) if text_parts else None
    if isinstance(content, str):
        return content
    return None


_CODEX_TRANSCRIPT_UNSUPPORTED_TOOL_NAMES = {
    "list_dir",
    "list_files",
    "run_command",
}


def _build_codex_transcript_bundle(
    transcript: Path,
    *,
    parent_transcript: Optional[Path] = None,
) -> TranscriptCandidateBundle:
    events = list(_iter_jsonl_events(transcript))
    session_meta = next(
        (
            event.get("payload")
            for event in events
            if event.get("type") == "session_meta" and isinstance(event.get("payload"), dict)
        ),
        None,
    )
    if not isinstance(session_meta, dict):
        raise ValueError(f"{transcript} is missing session_meta")
    turn_context = next(
        (
            event.get("payload")
            for event in events
            if event.get("type") == "turn_context" and isinstance(event.get("payload"), dict)
        ),
        {},
    )
    if not isinstance(turn_context, dict):
        turn_context = {}

    session_id = _clean(session_meta.get("id"))
    if not session_id:
        raise ValueError(f"{transcript} session_meta is missing payload.id")
    cwd = _clean(session_meta.get("cwd")) or _clean(turn_context.get("cwd"))
    repository = _repository_from_cwd(cwd)
    created_at = _clean(session_meta.get("timestamp")) or _clean(events[0].get("timestamp"))
    end_time = created_at
    llm_elapsed_ms: Optional[float] = None
    task_complete = next(
        (
            event.get("payload")
            for event in reversed(events)
            if event.get("type") == "event_msg"
            and isinstance(event.get("payload"), dict)
            and event["payload"].get("type") == "task_complete"
        ),
        None,
    )
    if isinstance(task_complete, dict):
        completed_at = _coerce_float_or_none(task_complete.get("completed_at"))
        if completed_at is not None:
            end_time = datetime.fromtimestamp(completed_at, tz=timezone.utc).isoformat()
        llm_elapsed_ms = _coerce_float_or_none(task_complete.get("duration_ms"))

    parent_spawn_message: Optional[str] = None
    parent_thread_id: Optional[str] = None
    source = session_meta.get("source")
    if isinstance(source, dict):
        subagent = source.get("subagent")
        if isinstance(subagent, dict):
            thread_spawn = subagent.get("thread_spawn")
            if isinstance(thread_spawn, dict):
                parent_thread_id = _clean(thread_spawn.get("parent_thread_id"))
    if parent_transcript is not None:
        parent_spawn_message = _extract_spawn_message_for_agent(parent_transcript, session_id)

    messages: List[Dict[str, Any]] = []
    if parent_spawn_message:
        messages.append({"role": "user", "content": parent_spawn_message})

    final_assistant_text: Optional[str] = None
    assistant_progress_message_count = 0
    function_call_count = 0
    unsupported_tool_names: List[str] = []
    for event in events:
        if event.get("type") != "response_item":
            continue
        payload = event.get("payload")
        if not isinstance(payload, dict):
            continue
        payload_type = payload.get("type")
        if payload_type == "function_call":
            function_call_count += 1
            tool_name = _clean(payload.get("name")) or "unknown"
            if tool_name in _CODEX_TRANSCRIPT_UNSUPPORTED_TOOL_NAMES:
                unsupported_tool_names.append(tool_name)
            arguments = _parse_json_value(payload.get("arguments"))
            messages.append(
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "name": tool_name,
                            "input": arguments if isinstance(arguments, dict) else {},
                            "timestamp": _clean(event.get("timestamp")),
                        }
                    ],
                }
            )
        elif payload_type == "function_call_output":
            output = payload.get("output")
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "content": output if isinstance(output, str) else json.dumps(output),
                        }
                    ],
                }
            )
        elif payload_type == "message":
            text = _codex_output_text_from_message_payload(payload)
            if text:
                assistant_progress_message_count += 1
                final_assistant_text = text
                messages.append({"role": "assistant", "content": text})

    terminal_state: Optional[str] = None
    if isinstance(task_complete, dict):
        message = task_complete.get("last_agent_message")
        if isinstance(message, str) and message.strip():
            terminal_state = "completed"
            final_assistant_text = message
            messages.append({"role": "assistant", "content": message})
        elif isinstance(message, str):
            terminal_state = "empty_final_message"
            final_assistant_text = None
        else:
            terminal_state = "null_final_message"
            final_assistant_text = None
    elif final_assistant_text:
        terminal_state = "message_without_task_complete"
    else:
        terminal_state = "missing_task_complete"

    total_usage: Dict[str, Any] = {}
    for event in events:
        if event.get("type") != "event_msg":
            continue
        payload = event.get("payload")
        if not isinstance(payload, dict) or payload.get("type") != "token_count":
            continue
        info = payload.get("info")
        if isinstance(info, dict) and isinstance(info.get("total_token_usage"), dict):
            total_usage = dict(info["total_token_usage"])
    input_tokens = _coerce_int(total_usage.get("input_tokens"))
    output_tokens = _coerce_int(total_usage.get("output_tokens"))
    total_tokens = _coerce_int(total_usage.get("total_tokens")) or input_tokens + output_tokens

    transcript_provider_alias = _clean(session_meta.get("model_provider"))
    transcript_model_alias = _clean(turn_context.get("model")) or _clean(
        session_meta.get("model")
    )
    session_history_provider = _codex_transcript_session_history_provider(
        transcript_provider_alias
    )
    session_history_model = _codex_transcript_session_history_model(
        transcript_model_alias
    )
    langfuse_input = {
        "messages": [
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "model": transcript_model_alias,
                        "messages": messages,
                    }
                ),
            }
        ]
    }
    output_body = {
        "role": "assistant",
        "content": final_assistant_text or "",
        "tool_calls": None,
    }
    metadata = {
        "source": "codex_transcript",
        "codex_transcript_path": str(transcript),
        "codex_parent_transcript_path": str(parent_transcript) if parent_transcript else None,
        "codex_parent_thread_id": parent_thread_id,
        "codex_turn_id": _clean(turn_context.get("turn_id")),
        "cwd": cwd,
        "repository": repository,
        "thread_source": _clean(session_meta.get("thread_source")),
        "originator": _clean(session_meta.get("originator")),
        "cli_version": _clean(session_meta.get("cli_version")),
        "agent_role": _clean(session_meta.get("agent_role")),
        "agent_nickname": _clean(session_meta.get("agent_nickname")),
        "codex_transcript_model_provider_alias": transcript_provider_alias,
        "codex_transcript_model_alias": transcript_model_alias,
        "codex_transcript_terminal_state": terminal_state,
        "codex_transcript_task_complete_seen": isinstance(task_complete, dict),
        "codex_transcript_final_message_present": bool(final_assistant_text),
        "codex_transcript_non_empty_assistant_progress_count": (
            assistant_progress_message_count
        ),
        "codex_transcript_function_call_count": function_call_count,
        "codex_transcript_unsupported_tool_call_count": len(unsupported_tool_names),
        "codex_transcript_unsupported_tool_names": sorted(set(unsupported_tool_names)),
    }
    metadata = {key: value for key, value in metadata.items() if value is not None}
    observation_id = f"codex-transcript:{session_id}"
    candidate = SessionCandidate(
        row_id=-1,
        created_at=created_at,
        trace_id=session_id,
        session_id=session_id,
        litellm_call_id=observation_id,
        source_observation_id=observation_id,
        provider=session_history_provider,
        model=session_history_model,
        agent_name=_clean(session_meta.get("agent_nickname")),
        repository=repository,
        tenant_id=repository,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        tool_call_count=function_call_count,
        invalid_tool_call_count=len(unsupported_tool_names),
        llm_upstream_elapsed_ms=llm_elapsed_ms,
        total_server_elapsed_ms=llm_elapsed_ms,
        ttft_ms=None,
        metadata=metadata,
    )
    payload = ObservationPayload(
        observation_id=observation_id,
        trace_id=session_id,
        body={
            "id": observation_id,
            "traceId": session_id,
            "startTime": created_at,
            "endTime": end_time,
            "name": "codex_transcript",
            "metadata": metadata,
            "input": json.dumps(langfuse_input),
            "output": json.dumps(output_body),
            "model": candidate.model,
        },
        source="codex_transcript",
        source_locator=str(transcript),
    )
    # Preserve total tokens for the transcript upsert without changing the stable
    # candidate shape used by normal session_history rows.
    candidate.metadata["codex_transcript_total_tokens"] = total_tokens
    return TranscriptCandidateBundle(candidate=candidate, payload=payload)


def _resolve_codex_transcript_bundles(
    args: argparse.Namespace,
) -> List[TranscriptCandidateBundle]:
    transcripts = [Path(value).expanduser() for value in (args.codex_transcript or [])]
    parent_transcripts = [
        Path(value).expanduser() for value in (args.codex_parent_transcript or [])
    ]
    bundles: List[TranscriptCandidateBundle] = []
    for index, transcript in enumerate(transcripts):
        parent_transcript = (
            parent_transcripts[index]
            if index < len(parent_transcripts)
            else parent_transcripts[0]
            if len(parent_transcripts) == 1
            else None
        )
        bundles.append(
            _build_codex_transcript_bundle(
                transcript,
                parent_transcript=parent_transcript,
            )
        )
    return bundles


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
    re.compile(r"\bopen\s*\([^)]*,\s*['\"][wa+x]"),
    re.compile(r"\.write_(?:text|bytes)\s*\("),
    re.compile(r"\b(?:rm|mv|cp)\s+(?:-[A-Za-z]+\s+)?[-A-Za-z0-9_/.\"]+"),
    re.compile(r"\bgit\s+reset\b"),
    re.compile(r"\bgit\s+(?:-[A-Za-z]\s+\S+\s+)*add\b"),
    re.compile(r"\bgit\s+(?:-[A-Za-z]\s+\S+\s+)*update-index\b"),
    re.compile(r"\bgit\s+apply\b"),
)
_SHELL_COMMAND_TOOL_NAMES = {
    "bash",
    "exec_command",
    "shell",
}
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
_READ_ONLY_INSTRUCTION_PATTERNS = (
    (
        "read_only",
        re.compile(
            r"\bread[- ]only\s+(?:investigation|audit|review|pass|scan|task|mode|only)\b",
            re.IGNORECASE,
        ),
    ),
    ("do_not_edit", re.compile(r"\bdo\s+not\s+edit\s+files?\b", re.IGNORECASE)),
    ("dont_edit", re.compile(r"\bdon'?t\s+edit\s+files?\b", re.IGNORECASE)),
    ("no_code_changes", re.compile(r"\bno\s+code\s+changes\b", re.IGNORECASE)),
    ("do_not_make_changes", re.compile(r"\bdo\s+not\s+make\s+changes\b", re.IGNORECASE)),
    ("dont_make_changes", re.compile(r"\bdon'?t\s+make\s+changes\b", re.IGNORECASE)),
)
_NO_LIVE_COMMAND_INSTRUCTION_PATTERNS = (
    (
        "no_live_db",
        re.compile(r"\bdo\s+not\s+run\s+live\s+(?:db|database)\b", re.IGNORECASE),
    ),
    (
        "no_live_container",
        re.compile(r"\bdo\s+not\s+run\s+live\s+.*\bcontainers?\b", re.IGNORECASE),
    ),
)
_LIVE_DB_CONTAINER_COMMAND_PATTERNS = (
    ("live_db_command", re.compile(r"\b(?:psql|mysql|sqlite3)\b")),
    ("live_container_command", re.compile(r"\bdocker\s+(?:exec|compose|ps|logs|restart|run|cp)\b")),
    ("live_kubernetes_command", re.compile(r"\bkubectl\b")),
)
_FORCED_GIT_TRACKING_COMMAND_PATTERNS = (
    re.compile(
        r"\bgit\s+(?:-[A-Za-z]\s+\S+\s+)*add\b(?=[^\n;&|]*(?:\s|=)(?:-f|--force|--no-ignore)\b)",
        re.IGNORECASE,
    ),
    re.compile(
        r"['\"]git['\"]\s*,\s*(?:['\"]-C['\"]\s*,\s*['\"][^'\"]+['\"]\s*,\s*)?['\"]add['\"]"
        r"(?=[\s\S]{0,400}(?:['\"]-f['\"]|['\"]--force['\"]|['\"]--no-ignore['\"]))",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bgit\s+(?:-[A-Za-z]\s+\S+\s+)*update-index\b(?=[^\n;&|]*(?:--add|--cacheinfo|--index-info)\b)",
        re.IGNORECASE,
    ),
    re.compile(
        r"['\"]git['\"]\s*,\s*(?:['\"]-C['\"]\s*,\s*['\"][^'\"]+['\"]\s*,\s*)?['\"]update-index['\"]"
        r"(?=[\s\S]{0,400}(?:['\"]--add['\"]|['\"]--cacheinfo['\"]|['\"]--index-info['\"]))",
        re.IGNORECASE,
    ),
)
_IGNORED_TRACKING_AUTHORIZATION_PATTERNS = (
    re.compile(
        r"\b(?:allow|allowed|authorize|authorized|intentional(?:ly)?|go ahead|please|need to|must|should)\b"
        r".{0,80}\b(?:force-?add|git\s+add\s+-f|track|version|commit|add)\b"
        r".{0,80}\b(?:ignored|gitignored|\.gitignored|seed|workflow)\b",
        re.IGNORECASE | re.DOTALL,
    ),
    re.compile(
        r"\b(?:force-?add|git\s+add\s+-f)\b.{0,80}\b(?:ignored|gitignored|seed|workflow)\b",
        re.IGNORECASE | re.DOTALL,
    ),
)
_IGNORED_TRACKING_PROHIBITION_PATTERNS = (
    re.compile(
        r"\b(?:do\s+not|don't|dont|never)\b.{0,80}\b(?:force-?add|git\s+add\s+-f|track|version|commit|add)\b"
        r".{0,80}\b(?:ignored|gitignored|\.gitignored)\b",
        re.IGNORECASE | re.DOTALL,
    ),
)
_COMMON_IGNORED_AGENT_STATE_PREFIXES = (
    ".analysis/",
    ".claude/",
    ".codex/",
    ".gemini/",
    ".grok/",
    ".llmpeerreview",
)
_TRIVIAL_NOOP_TEXTS = {
    ".",
    "...",
    "ok",
    "okay",
    "done",
    "ack",
    "acknowledged",
}
_SHELL_PATH_RE = re.compile(
    r"(?P<path>(?:/home/zepfu/projects/|/workspace/|\.{1,2}/|[A-Za-z0-9_.-]+/)"
    r"[A-Za-z0-9_./-]+)"
)
_PYTHON_OPEN_PATH_RE = re.compile(r"\bopen\s*\(\s*['\"](?P<path>[^'\"]+)['\"]")
_SHELL_REDIRECT_PATH_RE = re.compile(
    r"(?:^|\s)>{1,2}\s*(?P<path>[-A-Za-z0-9_./]+)"
)
_SHELL_FILE_MUTATION_PATH_RE = re.compile(
    r"\b(?:rm|mv|cp)\s+(?:-[A-Za-z]+\s+)?(?P<path>[-A-Za-z0-9_./]+)"
)


def _snippet(value: Optional[str], limit: int = 240) -> Optional[str]:
    cleaned = _clean(value)
    if cleaned is None:
        return None
    return cleaned if len(cleaned) <= limit else cleaned[: limit - 3] + "..."


def _timestamp_from_mapping(value: Any) -> Optional[str]:
    if not isinstance(value, dict):
        return None
    for key in (
        "timestamp",
        "created_at",
        "createdAt",
        "start_time",
        "startTime",
        "time",
    ):
        cleaned = _clean(value.get(key))
        if cleaned:
            return cleaned
    return None


def _extract_affected_paths(tool_input: Any, command: Optional[str]) -> List[str]:
    paths: List[str] = []
    parsed = _parse_json_value(tool_input)
    if isinstance(parsed, dict):
        for key in ("file_path", "path", "notebook_path", "filename"):
            value = parsed.get(key)
            if isinstance(value, str) and value.strip():
                paths.append(value.strip())
        for key in ("file_paths", "paths", "files"):
            value = parsed.get(key)
            if isinstance(value, list):
                paths.extend(item.strip() for item in value if isinstance(item, str))

    if command:
        for match in _PYTHON_OPEN_PATH_RE.finditer(command):
            paths.append(match.group("path"))
        for match in _SHELL_REDIRECT_PATH_RE.finditer(command):
            paths.append(match.group("path"))
        for match in _SHELL_FILE_MUTATION_PATH_RE.finditer(command):
            paths.append(match.group("path"))
        for match in _SHELL_PATH_RE.finditer(command):
            candidate = match.group("path").strip("\"'")
            if candidate and not candidate.startswith(("http://", "https://")):
                paths.append(candidate)

    deduped: List[str] = []
    for path in paths:
        if path not in deduped:
            deduped.append(path)
    return deduped[:20]


def _split_shell_words(fragment: str) -> List[str]:
    try:
        return shlex.split(fragment)
    except ValueError:
        return fragment.split()


def _iter_git_invocations(command: str) -> Iterator[Tuple[str, List[str]]]:
    for fragment in re.split(r"(?:&&|\|\||[;\n])", command):
        words = _split_shell_words(fragment)
        for index, word in enumerate(words):
            if word != "git":
                continue
            cursor = index + 1
            while cursor < len(words) and words[cursor].startswith("-"):
                option = words[cursor]
                cursor += 1
                if option in {"-C", "--git-dir", "--work-tree"} and cursor < len(words):
                    cursor += 1
            if cursor >= len(words):
                continue
            yield words[cursor], words[cursor + 1 :]


def _extract_git_add_pathspecs(args: Sequence[str]) -> Tuple[bool, List[str]]:
    forced = False
    pathspecs: List[str] = []
    pathspec_mode = False
    skip_next = False
    for arg in args:
        if skip_next:
            skip_next = False
            continue
        if pathspec_mode:
            pathspecs.append(arg)
            continue
        if arg == "--":
            pathspec_mode = True
            continue
        if arg in {"-f", "--force", "--no-ignore"}:
            forced = True
            continue
        if arg.startswith("-") and not arg.startswith("--") and "f" in arg:
            forced = True
            continue
        if arg.startswith("--pathspec-from-file"):
            if arg == "--pathspec-from-file":
                skip_next = True
            continue
        if arg.startswith("-"):
            continue
        pathspecs.append(arg)
    return forced, pathspecs


def _extract_update_index_pathspecs(args: Sequence[str]) -> Tuple[bool, List[str]]:
    tracking = any(arg in {"--add", "--cacheinfo", "--index-info"} for arg in args)
    if not tracking:
        return False, []
    pathspecs: List[str] = []
    for index, arg in enumerate(args):
        if arg == "--cacheinfo" and index + 3 < len(args):
            pathspecs.append(args[index + 3])
            continue
        if arg.startswith("-"):
            continue
        if re.fullmatch(r"[0-7]{3,6}", arg) or re.fullmatch(r"[0-9a-fA-F]{40,64}", arg):
            continue
        pathspecs.append(arg)
    return True, pathspecs


def _forced_git_tracking_paths(
    command: Optional[str],
    affected_paths: Sequence[str],
) -> Tuple[bool, List[str]]:
    if not command:
        return False, []
    forced = any(pattern.search(command) for pattern in _FORCED_GIT_TRACKING_COMMAND_PATTERNS)
    pathspecs: List[str] = []
    for subcommand, args in _iter_git_invocations(command):
        if subcommand == "add":
            add_forced, add_paths = _extract_git_add_pathspecs(args)
            forced = forced or add_forced
            if add_forced:
                pathspecs.extend(add_paths)
        elif subcommand == "update-index":
            update_forced, update_paths = _extract_update_index_pathspecs(args)
            forced = forced or update_forced
            if update_forced:
                pathspecs.extend(update_paths)
    if forced and not pathspecs:
        pathspecs.extend(affected_paths)
    deduped: List[str] = []
    for path in pathspecs:
        cleaned = path.strip().strip("\"'")
        if cleaned and cleaned not in deduped:
            deduped.append(cleaned)
    return forced, deduped[:20]


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


def _message_instruction_texts(messages: Sequence[Any]) -> Iterator[str]:
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = _clean(message.get("role"))
        if role not in {"system", "user"}:
            continue
        for block in _content_blocks(message):
            if isinstance(block, str):
                yield block
            elif isinstance(block, dict):
                if block.get("type") == "tool_result":
                    continue
                for text in _iter_text_values(block):
                    yield text
        content = message.get("content")
        if isinstance(content, str):
            yield content


def _iter_instruction_texts_with_context(
    messages: Sequence[Any],
) -> Iterator[Tuple[int, Optional[str], str]]:
    for message_index, message in enumerate(messages):
        if not isinstance(message, dict):
            continue
        role = _clean(message.get("role"))
        if role not in {"system", "user"}:
            continue
        for block in _content_blocks(message):
            if isinstance(block, str):
                yield message_index, role, block
            elif isinstance(block, dict):
                if block.get("type") == "tool_result":
                    continue
                for text in _iter_text_values(block):
                    yield message_index, role, text
        content = message.get("content")
        if isinstance(content, str):
            yield message_index, role, content


def _detect_read_only_instruction_evidence(
    messages: Sequence[Any],
) -> List[Dict[str, Any]]:
    evidence: List[Dict[str, Any]] = []
    seen: set[Tuple[str, int, str]] = set()
    patterns = (*_READ_ONLY_INSTRUCTION_PATTERNS, *_NO_LIVE_COMMAND_INSTRUCTION_PATTERNS)
    for message_index, role, text in _iter_instruction_texts_with_context(messages):
        for marker, pattern in patterns:
            if not pattern.search(text):
                continue
            key = (marker, message_index, text[:120])
            if key in seen:
                continue
            seen.add(key)
            evidence.append(
                {
                    "marker": marker,
                    "message_index": message_index,
                    "role": role,
                    "instruction_snippet": _snippet(text),
                }
            )
    return evidence


def _markers_from_instruction_evidence(
    instruction_evidence: Sequence[Dict[str, Any]],
) -> List[str]:
    markers: List[str] = []
    for item in instruction_evidence:
        marker = _clean(item.get("marker")) if isinstance(item, dict) else None
        if marker and marker not in markers:
            markers.append(marker)
    return markers


def _detect_read_only_policy_violations(
    tool_uses: Sequence[ToolUseEvidence],
    instruction_markers: Sequence[str],
) -> Tuple[int, List[str], List[Dict[str, Any]]]:
    violations: List[str] = []
    violation_evidence: List[Dict[str, Any]] = []
    if not instruction_markers:
        return 0, violations, violation_evidence

    read_only_requested = any(
        marker
        in {
            "read_only",
            "do_not_edit",
            "dont_edit",
            "no_code_changes",
            "do_not_make_changes",
            "dont_make_changes",
        }
        for marker in instruction_markers
    )
    live_command_forbidden = any(
        marker in {"no_live_db", "no_live_container"} for marker in instruction_markers
    )
    for tool_use in tool_uses:
        if read_only_requested and tool_use.mutating:
            reason = f"mutating_tool:{tool_use.name or 'unknown'}"
            violations.append(reason)
            violation_evidence.append(
                {
                    "reason": reason,
                    "tool_name": tool_use.name or "unknown",
                    "message_index": tool_use.message_index,
                    "sequence_index": tool_use.sequence_index,
                    "command_snippet": _snippet(tool_use.command),
                    "command_timestamp": tool_use.command_timestamp,
                    "affected_paths": tool_use.affected_paths,
                }
            )
        if live_command_forbidden and tool_use.command:
            for marker, pattern in _LIVE_DB_CONTAINER_COMMAND_PATTERNS:
                if pattern.search(tool_use.command):
                    violations.append(marker)
                    violation_evidence.append(
                        {
                            "reason": marker,
                            "tool_name": tool_use.name or "unknown",
                            "message_index": tool_use.message_index,
                            "sequence_index": tool_use.sequence_index,
                            "command_snippet": _snippet(tool_use.command),
                            "command_timestamp": tool_use.command_timestamp,
                            "affected_paths": tool_use.affected_paths,
                        }
                    )
                    break
    return len(violations), violations, violation_evidence


def _ignored_path_tracking_authorized(messages: Sequence[Any]) -> bool:
    text = "\n".join(_message_instruction_texts(messages))
    if not text:
        return False
    if any(pattern.search(text) for pattern in _IGNORED_TRACKING_PROHIBITION_PATTERNS):
        return False
    return any(pattern.search(text) for pattern in _IGNORED_TRACKING_AUTHORIZATION_PATTERNS)


def _candidate_repo_root(candidate: SessionCandidate) -> Optional[Path]:
    candidates: List[Path] = []
    cwd = _clean(candidate.metadata.get("cwd")) if isinstance(candidate.metadata, dict) else None
    if cwd:
        candidates.append(Path(cwd).expanduser())
    repository = _clean(candidate.repository) or _clean(candidate.tenant_id)
    if repository:
        repo_name = repository.rsplit("/", 1)[-1]
        candidates.append(Path("/home/zepfu/projects") / repo_name)
        if repo_name == REPO_ROOT.name:
            candidates.append(REPO_ROOT)
    for candidate_path in candidates:
        path = candidate_path
        if path.is_file():
            path = path.parent
        for current in (path, *path.parents):
            if (current / ".git").exists():
                return current
            if current == current.parent:
                break
    return None


def _repo_relative_path(repo_root: Path, path: str) -> Optional[str]:
    cleaned = path.strip().strip("\"'")
    if not cleaned:
        return None
    if cleaned in {".", "./"}:
        return "."
    path_obj = Path(cleaned).expanduser()
    if path_obj.is_absolute():
        try:
            return path_obj.resolve().relative_to(repo_root.resolve()).as_posix()
        except ValueError:
            return None
    return cleaned[2:] if cleaned.startswith("./") else cleaned


def _path_has_common_ignored_prefix(path: str) -> bool:
    normalized = path.strip().strip("\"'").replace("\\", "/")
    if "/home/zepfu/projects/" in normalized:
        parts = normalized.split("/home/zepfu/projects/", 1)[1].split("/", 1)
        normalized = parts[1] if len(parts) > 1 else ""
    while normalized.startswith("./"):
        normalized = normalized[2:]
    return any(
        normalized == prefix.rstrip("/") or normalized.startswith(prefix)
        for prefix in _COMMON_IGNORED_AGENT_STATE_PREFIXES
    )


def _run_git(
    repo_root: Path,
    args: Sequence[str],
) -> Optional[subprocess.CompletedProcess[str]]:
    try:
        return subprocess.run(
            ["git", "-C", str(repo_root), *args],
            capture_output=True,
            check=False,
            text=True,
            timeout=3,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None


def _git_check_ignore_match(repo_root: Path, rel_path: str) -> Optional[Dict[str, Any]]:
    variants = [rel_path]
    if rel_path not in {".", "./"} and not rel_path.endswith("/"):
        variants.append(f"{rel_path}/")
    for variant in variants:
        result = _run_git(repo_root, ["check-ignore", "--no-index", "-v", "--", variant])
        if result is None or result.returncode != 0:
            continue
        output = result.stdout.strip().splitlines()
        if not output:
            continue
        raw = output[0]
        source = None
        pattern = None
        if "\t" in raw:
            rule, _matched_path = raw.split("\t", 1)
            parts = rule.split(":", 2)
            if len(parts) == 3:
                source = f"{parts[0]}:{parts[1]}"
                pattern = parts[2]
        return {
            "ignored": True,
            "ignored_path": variant,
            "ignore_rule_source": source,
            "ignore_rule_pattern": pattern,
            "ignore_rule_raw": raw[:240],
        }
    return None


def _git_ignored_tracked_paths(repo_root: Path, rel_path: str) -> List[str]:
    result = _run_git(
        repo_root,
        ["ls-files", "--cached", "--ignored", "--exclude-standard", "-z", "--", rel_path],
    )
    if result is None or result.returncode != 0 or not result.stdout:
        return []
    return [path for path in result.stdout.split("\0") if path][:10]


def _detect_ignored_path_tracking_violations(
    candidate: SessionCandidate,
    messages: Sequence[Any],
    tool_uses: Sequence[ToolUseEvidence],
) -> Tuple[int, List[Dict[str, Any]]]:
    if _ignored_path_tracking_authorized(messages):
        return 0, []
    repo_root = _candidate_repo_root(candidate)
    evidence: List[Dict[str, Any]] = []
    seen: set[Tuple[int, str]] = set()
    for tool_use in tool_uses:
        if not tool_use.forced_git_tracking:
            continue
        candidate_paths = tool_use.git_tracking_paths or tool_use.affected_paths or ["."]
        for raw_path in candidate_paths:
            rel_path = (
                _repo_relative_path(repo_root, raw_path) if repo_root is not None else None
            )
            tracked_ignored_paths = (
                _git_ignored_tracked_paths(repo_root, rel_path)
                if repo_root is not None and rel_path is not None
                else []
            )
            if tracked_ignored_paths:
                for tracked_path in tracked_ignored_paths:
                    key = (tool_use.sequence_index, tracked_path)
                    if key in seen:
                        continue
                    seen.add(key)
                    evidence.append(
                        {
                            "reason": "forced_tracking_ignored_path",
                            "tool_name": tool_use.name or "unknown",
                            "message_index": tool_use.message_index,
                            "sequence_index": tool_use.sequence_index,
                            "command_snippet": _snippet(tool_use.command),
                            "command_timestamp": tool_use.command_timestamp,
                            "path": tracked_path,
                            "requested_path": raw_path,
                            "ignored_check": "tracked_ignored",
                        }
                    )
                continue

            ignore_match = (
                _git_check_ignore_match(repo_root, rel_path)
                if repo_root is not None and rel_path is not None
                else None
            )
            common_ignored_prefix = _path_has_common_ignored_prefix(raw_path)
            if ignore_match is None and not common_ignored_prefix:
                continue
            evidence_path = (
                str(ignore_match.get("ignored_path")) if ignore_match else raw_path
            )
            key = (tool_use.sequence_index, evidence_path)
            if key in seen:
                continue
            seen.add(key)
            item = {
                "reason": "forced_tracking_ignored_path",
                "tool_name": tool_use.name or "unknown",
                "message_index": tool_use.message_index,
                "sequence_index": tool_use.sequence_index,
                "command_snippet": _snippet(tool_use.command),
                "command_timestamp": tool_use.command_timestamp,
                "path": evidence_path,
                "requested_path": raw_path,
                "ignored_check": "confirmed_ignored" if ignore_match else "common_agent_state_path",
            }
            if ignore_match:
                item.update(ignore_match)
            evidence.append(item)
    return len(evidence), evidence[:10]


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
    if normalized in _SHELL_COMMAND_TOOL_NAMES and command:
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
            tool_input = block.get("input")
            command = _command_from_tool_input(tool_input)
            affected_paths = _extract_affected_paths(tool_input, command)
            forced_git_tracking, git_tracking_paths = _forced_git_tracking_paths(
                command,
                affected_paths,
            )
            uses.append(
                ToolUseEvidence(
                    message_index=message_index,
                    sequence_index=sequence_index,
                    name=name,
                    command=command,
                    command_timestamp=(
                        _timestamp_from_mapping(block) or _timestamp_from_mapping(message)
                    ),
                    affected_paths=affected_paths,
                    forced_git_tracking=forced_git_tracking,
                    git_tracking_paths=git_tracking_paths,
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


def _is_trivial_noop_text(text: str, output_tokens: int, max_output_tokens: int) -> bool:
    normalized = re.sub(r"\s+", " ", text.strip().lower())
    if output_tokens > max_output_tokens:
        return False
    return normalized in _TRIVIAL_NOOP_TEXTS


def _malformed_success_output_shape(output: Any) -> bool:
    parsed = _parse_json_value(output)
    if not isinstance(parsed, dict):
        return False
    if not parsed:
        return True
    choices = parsed.get("choices")
    if isinstance(choices, list):
        if not choices:
            return True
        first_choice = choices[0] if isinstance(choices[0], dict) else None
        if first_choice is None:
            return True
        message = first_choice.get("message")
        if message is None and not first_choice.get("text"):
            return True
    return False


def _scope_control_score_and_reasons(
    candidate: SessionCandidate,
    tool_uses: Sequence[ToolUseEvidence],
    *,
    payload_resolved: bool,
) -> Tuple[Optional[float], List[str]]:
    if not payload_resolved:
        return None, []

    repository = _clean(candidate.repository) or _clean(candidate.tenant_id)
    repo_names = {repository} if repository else set()
    if repository and "/" in repository:
        repo_names.add(repository.rsplit("/", 1)[-1])

    reasons: List[str] = []
    for tool_use in tool_uses:
        for path in tool_use.affected_paths:
            normalized = path.strip()
            if normalized.startswith("../") or "/../" in normalized:
                reasons.append(f"path_scope_escape:{normalized[:120]}")
                continue
            if normalized.startswith(("/home/zepfu/projects/", "/workspace/")) and repo_names:
                if not any(f"/{repo_name}/" in f"{normalized}/" for repo_name in repo_names):
                    reasons.append(f"outside_repository_path:{normalized[:120]}")

    return (0.0, reasons[:10]) if reasons else (1.0, [])


def _text_has_any(text: str, phrases: Sequence[str]) -> bool:
    normalized = text.lower()
    return any(phrase in normalized for phrase in phrases)


def _repetition_loop_risk_score_and_reasons(
    tool_uses: Sequence[ToolUseEvidence],
    *,
    payload_resolved: bool,
) -> Tuple[Optional[float], List[str]]:
    if not payload_resolved:
        return None, []
    if not tool_uses:
        return None, []

    command_counts: Dict[Tuple[str, str], int] = {}
    for tool_use in tool_uses:
        command = (tool_use.command or "").strip()
        if not command:
            continue
        key = (tool_use.name.lower(), command)
        command_counts[key] = command_counts.get(key, 0) + 1

    repeated = [
        f"repeated_tool_command:{name}:{_snippet(command, 80) or 'unknown'}"
        for (name, command), count in command_counts.items()
        if count >= 2
    ]
    return (1.0, repeated[:10]) if repeated else (0.0, [])


def _error_attribution_quality_score(
    assistant_text: str,
    *,
    payload_resolved: bool,
    provider_error_present: bool,
    invalid_tool_call_error_count: int,
) -> Optional[float]:
    if not payload_resolved:
        return None
    if not (provider_error_present or invalid_tool_call_error_count > 0):
        return None
    if _text_has_any(
        assistant_text,
        (
            "error",
            "failure",
            "failed",
            "recover",
            "recovered",
            "validation",
            "provider",
            "tool",
            "because",
            "root cause",
            "blocked",
        ),
    ):
        return 1.0
    return 0.0


def _agent_quality_context_from_messages(
    messages: Sequence[Any],
    *,
    assistant_output_text: str,
    tool_uses: Sequence[ToolUseEvidence],
    input_tokens: int,
    output_tokens: int,
    elapsed_ms: Optional[float],
    task_progress: bool,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    user_texts: List[str] = []
    assistant_texts: List[str] = []
    tool_result_texts: List[str] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = _clean(message.get("role"))
        target = (
            assistant_texts
            if role == "assistant"
            else tool_result_texts
            if role == "tool"
            else user_texts
            if role in {"system", "user"}
            else None
        )
        if target is None:
            continue
        for block in _content_blocks(message):
            if isinstance(block, str):
                target.append(block)
            elif isinstance(block, dict):
                if block.get("type") == "tool_use":
                    continue
                if block.get("type") == "tool_result":
                    for text in _iter_text_values(block.get("content")):
                        tool_result_texts.append(text)
                    continue
                for text in _iter_text_values(block):
                    target.append(text)
        content = message.get("content")
        if isinstance(content, str):
            target.append(content)
    if assistant_output_text:
        assistant_texts.append(assistant_output_text)
    commands = [
        AgentQualityCommand(
            name=tool_use.name,
            command=tool_use.command or "",
            timestamp=tool_use.command_timestamp,
            affected_paths=tuple(tool_use.git_tracking_paths or tool_use.affected_paths),
        )
        for tool_use in tool_uses
        if tool_use.command
    ]
    result = score_agent_quality_context(
        user_texts=user_texts,
        assistant_texts=assistant_texts,
        tool_result_texts=tool_result_texts,
        commands=commands,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        elapsed_ms=elapsed_ms,
        task_progress=task_progress,
    )
    return result.fields, result.reasons


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
    payload_resolved = payload is not None and "missing_request_messages" not in errors

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
    terminal_state = _clean(candidate.metadata.get("codex_transcript_terminal_state"))
    terminal_completion_failure = terminal_state in {
        "empty_final_message",
        "missing_task_complete",
        "null_final_message",
    }
    terminal_completion_score = (
        None if terminal_state is None else (0.0 if terminal_completion_failure else 1.0)
    )
    transcript_progress_count = _coerce_int(
        candidate.metadata.get("codex_transcript_non_empty_assistant_progress_count")
    )
    no_tool_calls = candidate.tool_call_count == 0 and output_tool_calls == 0
    trivial_noop_text = _is_trivial_noop_text(
        assistant_text,
        candidate.output_tokens,
        max_output_tokens,
    )
    malformed_output_shape = (
        payload_resolved
        and not provider_error_present
        and _malformed_success_output_shape(body.get("output"))
    )
    has_meaningful_output = (
        (bool(assistant_text.strip()) and not trivial_noop_text)
        or output_tool_calls > 0
    )
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
    unsupported_tool_names = candidate.metadata.get(
        "codex_transcript_unsupported_tool_names"
    )
    if isinstance(unsupported_tool_names, list):
        for name in unsupported_tool_names:
            if not isinstance(name, str) or not name:
                continue
            marker = f"unsupported_tool_call:{name}"
            if marker not in invalid_tool_call_error_markers:
                invalid_tool_call_error_markers.append(marker)
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
    if terminal_completion_failure:
        reasons.append(f"terminal_completion_{terminal_state}")
    read_only_instruction_evidence = _detect_read_only_instruction_evidence(messages)
    instruction_markers = _markers_from_instruction_evidence(
        read_only_instruction_evidence
    )
    (
        read_only_violation_count,
        read_only_violation_reasons,
        read_only_violation_evidence,
    ) = (
        _detect_read_only_policy_violations(tool_uses, instruction_markers)
    )
    if read_only_violation_count:
        reasons.append("read_only_policy_violation")
    (
        ignored_path_tracking_violation_count,
        ignored_path_tracking_evidence,
    ) = _detect_ignored_path_tracking_violations(candidate, messages, tool_uses)
    if ignored_path_tracking_violation_count:
        reasons.append("ignored_path_tracking_policy_violation")

    no_op_response_failure = (
        (
            terminal_completion_failure
            or empty_output
            or (trivial_noop_text and no_tool_calls)
        )
        and not provider_error_present
    )
    response_meaningfulness_score = (
        None
        if not payload_resolved or provider_error_present
        else (0.0 if no_op_response_failure else 1.0)
    )
    read_only_policy_compliance_score = (
        None
        if not instruction_markers
        else (0.0 if read_only_violation_count else 1.0)
    )
    instruction_adherence_score = read_only_policy_compliance_score
    answer_completeness_score = (
        None
        if not payload_resolved or provider_error_present
        else (0.0 if no_op_response_failure else 1.0)
    )
    evidence_fidelity_score = (
        None
        if not payload_resolved or provider_error_present or not tool_results
        else (0.0 if no_op_response_failure else 1.0)
    )
    tool_result_fidelity_score = (
        None
        if not payload_resolved or provider_error_present or not tool_results
        else (0.0 if empty_completion_failure or no_op_response_failure else 1.0)
    )
    error_attribution_quality_score = _error_attribution_quality_score(
        assistant_text,
        payload_resolved=payload_resolved,
        provider_error_present=provider_error_present,
        invalid_tool_call_error_count=invalid_tool_call_error_count,
    )
    repetition_loop_risk_score, repetition_loop_risk_reasons = (
        _repetition_loop_risk_score_and_reasons(
            tool_uses,
            payload_resolved=payload_resolved,
        )
    )
    context_retention_score = (
        None
        if not payload_resolved or not instruction_markers
        else (0.0 if read_only_violation_count or destructive_checkout else 1.0)
    )
    tool_use_validity_score = (
        None if not payload_resolved else (0.0 if invalid_tool_call_error_count else 1.0)
    )
    tool_error_recovery_score = (
        None
        if not payload_resolved or provider_error_present
        else 1.0
        if invalid_tool_call_error_count == 0
        else (1.0 if has_meaningful_output or output_tool_calls > 0 else 0.0)
    )
    elapsed_ms = max(
        [
            value
            for value in (
                candidate.llm_upstream_elapsed_ms,
                candidate.total_server_elapsed_ms,
            )
            if value is not None
        ],
        default=None,
    )
    stall_risk_score = (
        None
        if provider_error_present or elapsed_ms is None
        else (
            1.0
            if elapsed_ms >= 30 * 60 * 1000
            and candidate.output_tokens <= max_output_tokens
            and no_tool_calls
            else 0.0
        )
    )
    output_contract_failure = (
        terminal_completion_failure or no_op_response_failure or malformed_output_shape
    )
    output_contract_compliance_score = (
        None
        if not payload_resolved or provider_error_present
        else (0.0 if output_contract_failure else 1.0)
    )
    task_progress_score = (
        None
        if not payload_resolved or provider_error_present
        else (
            1.0
            if has_meaningful_output
            or transcript_progress_count > 0
            or candidate.tool_call_count > 0
            else 0.0
        )
    )
    agent_quality_fields, agent_quality_reasons = _agent_quality_context_from_messages(
        messages,
        assistant_output_text=assistant_text,
        tool_uses=tool_uses,
        input_tokens=candidate.input_tokens,
        output_tokens=candidate.output_tokens,
        elapsed_ms=elapsed_ms,
        task_progress=bool(task_progress_score),
    )
    command_only_ignored_count = _coerce_int(
        agent_quality_fields.get("ignored_path_tracking_violation_count")
    )
    if not ignored_path_tracking_violation_count and command_only_ignored_count:
        ignored_path_tracking_violation_count = command_only_ignored_count
        ignored_path_tracking_evidence = list(
            agent_quality_fields.get("ignored_path_tracking_evidence") or []
        )
        reasons.append("ignored_path_tracking_policy_violation")
    baseline_deflection_attempted_score = _coerce_float_or_none(
        agent_quality_fields.get("baseline_deflection_attempted_score")
    )
    baseline_deflection_incident_score = _coerce_float_or_none(
        agent_quality_fields.get("baseline_deflection_incident_score")
    )
    baseline_deflection_attempt_count = _coerce_int(
        agent_quality_fields.get("baseline_deflection_attempt_count")
    )
    baseline_deflection_tool_call_count = _coerce_int(
        agent_quality_fields.get("baseline_deflection_tool_call_count")
    )
    baseline_deflection_input_tokens = _coerce_int(
        agent_quality_fields.get("baseline_deflection_input_tokens")
    )
    baseline_deflection_elapsed_ms = _coerce_float_or_none(
        agent_quality_fields.get("baseline_deflection_elapsed_ms")
    )
    quality_gate_trigger_count = _coerce_int(
        agent_quality_fields.get("quality_gate_trigger_count")
    )
    quality_gate_fix_attempt_count = _coerce_int(
        agent_quality_fields.get("quality_gate_fix_attempt_count")
    )
    quality_gate_rerun_count = _coerce_int(
        agent_quality_fields.get("quality_gate_rerun_count")
    )
    sleep_wellness_interruption_attempted_score = _coerce_float_or_none(
        agent_quality_fields.get("sleep_wellness_interruption_attempted_score")
    )
    sleep_wellness_interruption_incident_score = _coerce_float_or_none(
        agent_quality_fields.get("sleep_wellness_interruption_incident_score")
    )
    sleep_wellness_interruption_count = _coerce_int(
        agent_quality_fields.get("sleep_wellness_interruption_count")
    )
    sleep_wellness_interruption_output_tokens = _coerce_int(
        agent_quality_fields.get("sleep_wellness_interruption_output_tokens")
    )
    sleep_wellness_interruption_input_tokens = _coerce_int(
        agent_quality_fields.get("sleep_wellness_interruption_input_tokens")
    )
    sleep_wellness_interruption_elapsed_ms = _coerce_float_or_none(
        agent_quality_fields.get("sleep_wellness_interruption_elapsed_ms")
    )
    sleep_wellness_interruption_after_user_pushback_count = _coerce_int(
        agent_quality_fields.get(
            "sleep_wellness_interruption_after_user_pushback_count"
        )
    )
    sleep_wellness_interruption_repeated_count = _coerce_int(
        agent_quality_fields.get("sleep_wellness_interruption_repeated_count")
    )
    discovery_inventory_coverage_score = _coerce_float_or_none(
        agent_quality_fields.get("discovery_inventory_coverage_score")
    )
    discovery_inventory_missing_count = (
        None
        if discovery_inventory_coverage_score is None
        else _coerce_int(agent_quality_fields.get("discovery_inventory_missing_count"))
    )
    if baseline_deflection_attempted_score == 1.0:
        reasons.append("baseline_deflection_attempted")
    if baseline_deflection_incident_score == 1.0:
        reasons.append("baseline_deflection_incident")
    if sleep_wellness_interruption_attempted_score == 1.0:
        reasons.append("sleep_wellness_interruption_attempted")
    if sleep_wellness_interruption_incident_score == 1.0:
        reasons.append("sleep_wellness_interruption_incident")
    if discovery_inventory_coverage_score == 0.0:
        reasons.append("discovery_inventory_coverage_failure")
    scope_control_score, scope_control_reasons = _scope_control_score_and_reasons(
        candidate,
        tool_uses,
        payload_resolved=payload_resolved,
    )
    read_only_requested = any(
        marker
        in {
            "read_only",
            "do_not_edit",
            "dont_edit",
            "no_code_changes",
            "do_not_make_changes",
            "dont_make_changes",
        }
        for marker in instruction_markers
    )
    destructive_action_policy_score = (
        None
        if not payload_resolved or not read_only_requested
        else (0.0 if read_only_violation_count or destructive_checkout else 1.0)
    )
    ignored_path_tracking_policy_score = (
        None
        if not payload_resolved
        else (0.0 if ignored_path_tracking_violation_count else 1.0)
    )
    agent_score_reasons = {
        "response_meaningfulness": (
            ["no_meaningful_output"] if response_meaningfulness_score == 0.0 else []
        ),
        "read_only_policy_compliance": read_only_violation_reasons,
        "instruction_adherence": read_only_violation_reasons,
        "answer_completeness": (
            ["incomplete_answer"] if answer_completeness_score == 0.0 else []
        ),
        "evidence_fidelity": (
            ["no_evidence_grounding"] if evidence_fidelity_score == 0.0 else []
        ),
        "tool_result_fidelity": (
            ["tool_result_not_reflected"]
            if tool_result_fidelity_score == 0.0
            else []
        ),
        "error_attribution_quality": (
            ["misattributed_error"] if error_attribution_quality_score == 0.0 else []
        ),
        "repetition_loop_risk": repetition_loop_risk_reasons,
        "context_retention": (
            ["lost_active_constraints"] if context_retention_score == 0.0 else []
        ),
        "read_only_instruction_evidence": read_only_instruction_evidence,
        "read_only_policy_violation_evidence": read_only_violation_evidence,
        "tool_use_validity": invalid_tool_call_error_markers,
        "tool_error_recovery": (
            ["invalid_tool_error_without_recovery"]
            if tool_error_recovery_score == 0.0
            else []
        ),
        "stall_risk": ["long_elapsed_low_output"] if stall_risk_score == 1.0 else [],
        "output_contract_compliance": (
            (
                [f"terminal_completion_{terminal_state}"]
                if terminal_completion_failure
                else ["malformed_response_shape"]
                if malformed_output_shape
                else ["empty_success_response"]
            )
            if output_contract_compliance_score == 0.0
            else []
        ),
        "task_progress": ["no_task_progress_signal"] if task_progress_score == 0.0 else [],
        "terminal_completion": (
            [str(terminal_state)] if terminal_completion_score == 0.0 else []
        ),
        "scope_control": scope_control_reasons,
        "destructive_action_policy": (
            ["destructive_action_against_prompt_policy"]
            if destructive_action_policy_score == 0.0
            else []
        ),
        "ignored_path_tracking_policy": (
            ["forced_tracking_ignored_path"]
            if ignored_path_tracking_policy_score == 0.0
            else []
        ),
        "ignored_path_tracking_evidence": ignored_path_tracking_evidence,
        "baseline_deflection": agent_quality_reasons.get("baseline_deflection", []),
        "baseline_deflection_evidence": agent_quality_reasons.get(
            "baseline_deflection_evidence",
            [],
        ),
        "sleep_wellness_interruption": agent_quality_reasons.get(
            "sleep_wellness_interruption",
            [],
        ),
        "sleep_wellness_interruption_evidence": agent_quality_reasons.get(
            "sleep_wellness_interruption_evidence",
            [],
        ),
        "discovery_inventory_coverage": agent_quality_reasons.get(
            "discovery_inventory_coverage",
            [],
        ),
        "discovery_inventory_evidence": agent_quality_reasons.get(
            "discovery_inventory_evidence",
            [],
        ),
        "agent_quality_rule_catalog_version": agent_quality_reasons.get(
            "agent_quality_rule_catalog_version"
        ),
    }
    failed = (
        empty_completion_failure
        or destructive_checkout
        or ignored_path_tracking_violation_count > 0
        or baseline_deflection_incident_score == 1.0
        or sleep_wellness_interruption_incident_score == 1.0
        or terminal_completion_score == 0.0
        or discovery_inventory_coverage_score == 0.0
    )
    return TraceScoreEvidence(
        row_id=candidate.row_id,
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
        score_payload_resolved=payload_resolved,
        response_meaningfulness_score=response_meaningfulness_score,
        read_only_policy_compliance_score=read_only_policy_compliance_score,
        read_only_policy_violation_count=read_only_violation_count,
        read_only_policy_violation_reasons=read_only_violation_reasons,
        read_only_instruction_evidence=read_only_instruction_evidence,
        read_only_policy_violation_evidence=read_only_violation_evidence,
        instruction_adherence_score=instruction_adherence_score,
        answer_completeness_score=answer_completeness_score,
        evidence_fidelity_score=evidence_fidelity_score,
        tool_result_fidelity_score=tool_result_fidelity_score,
        error_attribution_quality_score=error_attribution_quality_score,
        repetition_loop_risk_score=repetition_loop_risk_score,
        context_retention_score=context_retention_score,
        tool_use_validity_score=tool_use_validity_score,
        tool_error_recovery_score=tool_error_recovery_score,
        stall_risk_score=stall_risk_score,
        output_contract_compliance_score=output_contract_compliance_score,
        task_progress_score=task_progress_score,
        scope_control_score=scope_control_score,
        destructive_action_policy_score=destructive_action_policy_score,
        ignored_path_tracking_policy_score=ignored_path_tracking_policy_score,
        ignored_path_tracking_violation_count=ignored_path_tracking_violation_count,
        baseline_deflection_attempted_score=baseline_deflection_attempted_score,
        baseline_deflection_incident_score=baseline_deflection_incident_score,
        baseline_deflection_attempt_count=baseline_deflection_attempt_count,
        baseline_deflection_tool_call_count=baseline_deflection_tool_call_count,
        baseline_deflection_input_tokens=baseline_deflection_input_tokens,
        baseline_deflection_elapsed_ms=baseline_deflection_elapsed_ms,
        quality_gate_trigger_count=quality_gate_trigger_count,
        quality_gate_fix_attempt_count=quality_gate_fix_attempt_count,
        quality_gate_rerun_count=quality_gate_rerun_count,
        sleep_wellness_interruption_attempted_score=(
            sleep_wellness_interruption_attempted_score
        ),
        sleep_wellness_interruption_incident_score=(
            sleep_wellness_interruption_incident_score
        ),
        sleep_wellness_interruption_count=sleep_wellness_interruption_count,
        sleep_wellness_interruption_output_tokens=(
            sleep_wellness_interruption_output_tokens
        ),
        sleep_wellness_interruption_input_tokens=(
            sleep_wellness_interruption_input_tokens
        ),
        sleep_wellness_interruption_elapsed_ms=(
            sleep_wellness_interruption_elapsed_ms
        ),
        sleep_wellness_interruption_after_user_pushback_count=(
            sleep_wellness_interruption_after_user_pushback_count
        ),
        sleep_wellness_interruption_repeated_count=(
            sleep_wellness_interruption_repeated_count
        ),
        terminal_completion_score=terminal_completion_score,
        discovery_inventory_coverage_score=discovery_inventory_coverage_score,
        discovery_inventory_missing_count=discovery_inventory_missing_count,
        ignored_path_tracking_evidence=ignored_path_tracking_evidence,
        agent_score_reasons=agent_score_reasons,
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
    scores: List[LangfuseScore] = []
    if evidence.score_payload_resolved:
        scores.extend(
            [
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
        )
    elif evidence.invalid_tool_call_error_count > 0:
        scores.append(
            LangfuseScore(
                name="aawm.agent.invalid_tool_call_error",
                value=1.0,
                data_type="BOOLEAN",
                trace_id=trace_id,
                observation_id=observation_id,
                session_id=session_id,
                comment=comment,
            )
        )
    optional_scores = (
        (
            "aawm.agent.read_only_policy_compliance",
            evidence.read_only_policy_compliance_score,
            "NUMERIC",
        ),
        (
            "aawm.agent.response_meaningfulness",
            evidence.response_meaningfulness_score,
            "NUMERIC",
        ),
        (
            "aawm.agent.instruction_adherence",
            evidence.instruction_adherence_score,
            "NUMERIC",
        ),
        (
            "aawm.agent.answer_completeness",
            evidence.answer_completeness_score,
            "NUMERIC",
        ),
        (
            "aawm.agent.evidence_fidelity",
            evidence.evidence_fidelity_score,
            "NUMERIC",
        ),
        (
            "aawm.agent.tool_result_fidelity",
            evidence.tool_result_fidelity_score,
            "NUMERIC",
        ),
        (
            "aawm.agent.error_attribution_quality",
            evidence.error_attribution_quality_score,
            "NUMERIC",
        ),
        (
            "aawm.agent.repetition_loop_risk",
            evidence.repetition_loop_risk_score,
            "NUMERIC",
        ),
        (
            "aawm.agent.context_retention",
            evidence.context_retention_score,
            "NUMERIC",
        ),
        (
            "aawm.agent.tool_use_validity",
            evidence.tool_use_validity_score,
            "NUMERIC",
        ),
        (
            "aawm.agent.tool_error_recovery",
            evidence.tool_error_recovery_score,
            "NUMERIC",
        ),
        ("aawm.agent.stall_risk", evidence.stall_risk_score, "BOOLEAN"),
        (
            "aawm.agent.output_contract_compliance",
            evidence.output_contract_compliance_score,
            "NUMERIC",
        ),
        ("aawm.agent.task_progress", evidence.task_progress_score, "NUMERIC"),
        ("aawm.agent.scope_control", evidence.scope_control_score, "NUMERIC"),
        (
            "aawm.agent.destructive_action_policy",
            evidence.destructive_action_policy_score,
            "NUMERIC",
        ),
        (
            "aawm.agent.ignored_path_tracking_policy",
            evidence.ignored_path_tracking_policy_score,
            "NUMERIC",
        ),
        (
            "aawm.agent.baseline_deflection_attempted",
            evidence.baseline_deflection_attempted_score,
            "BOOLEAN",
        ),
        (
            "aawm.agent.baseline_deflection_incident",
            evidence.baseline_deflection_incident_score,
            "BOOLEAN",
        ),
        (
            "aawm.agent.sleep_wellness_interruption_attempted",
            evidence.sleep_wellness_interruption_attempted_score,
            "BOOLEAN",
        ),
        (
            "aawm.agent.sleep_wellness_interruption_incident",
            evidence.sleep_wellness_interruption_incident_score,
            "BOOLEAN",
        ),
        (
            "aawm.agent.terminal_completion",
            evidence.terminal_completion_score,
            "NUMERIC",
        ),
        (
            "aawm.agent.discovery_inventory_coverage",
            evidence.discovery_inventory_coverage_score,
            "NUMERIC",
        ),
    )
    for name, value, data_type in optional_scores:
        if value is None:
            continue
        scores.append(
            LangfuseScore(
                name=name,
                value=value,
                data_type=data_type,
                trace_id=trace_id,
                observation_id=observation_id,
                session_id=session_id,
                comment=comment,
            )
        )
    return scores


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


_SESSION_HISTORY_SCORE_ALTER_STATEMENTS = (
    "ALTER TABLE public.session_history ADD COLUMN IF NOT EXISTS trace_quality_score DOUBLE PRECISION",
    "ALTER TABLE public.session_history ADD COLUMN IF NOT EXISTS empty_completion_failure BOOLEAN",
    "ALTER TABLE public.session_history ADD COLUMN IF NOT EXISTS large_tool_result_payload_risk BOOLEAN",
    "ALTER TABLE public.session_history ADD COLUMN IF NOT EXISTS destructive_checkout_after_work BOOLEAN",
    "ALTER TABLE public.session_history ADD COLUMN IF NOT EXISTS invalid_tool_call_error BOOLEAN",
    "ALTER TABLE public.session_history ADD COLUMN IF NOT EXISTS read_only_policy_compliance_score DOUBLE PRECISION",
    "ALTER TABLE public.session_history ADD COLUMN IF NOT EXISTS read_only_policy_violation_count INTEGER",
    "ALTER TABLE public.session_history ADD COLUMN IF NOT EXISTS response_meaningfulness_score DOUBLE PRECISION",
    "ALTER TABLE public.session_history ADD COLUMN IF NOT EXISTS instruction_adherence_score DOUBLE PRECISION",
    "ALTER TABLE public.session_history ADD COLUMN IF NOT EXISTS answer_completeness_score DOUBLE PRECISION",
    "ALTER TABLE public.session_history ADD COLUMN IF NOT EXISTS evidence_fidelity_score DOUBLE PRECISION",
    "ALTER TABLE public.session_history ADD COLUMN IF NOT EXISTS tool_result_fidelity_score DOUBLE PRECISION",
    "ALTER TABLE public.session_history ADD COLUMN IF NOT EXISTS error_attribution_quality_score DOUBLE PRECISION",
    "ALTER TABLE public.session_history ADD COLUMN IF NOT EXISTS repetition_loop_risk_score DOUBLE PRECISION",
    "ALTER TABLE public.session_history ADD COLUMN IF NOT EXISTS context_retention_score DOUBLE PRECISION",
    "ALTER TABLE public.session_history ADD COLUMN IF NOT EXISTS tool_use_validity_score DOUBLE PRECISION",
    "ALTER TABLE public.session_history ADD COLUMN IF NOT EXISTS tool_error_recovery_score DOUBLE PRECISION",
    "ALTER TABLE public.session_history ADD COLUMN IF NOT EXISTS stall_risk_score DOUBLE PRECISION",
    "ALTER TABLE public.session_history ADD COLUMN IF NOT EXISTS output_contract_compliance_score DOUBLE PRECISION",
    "ALTER TABLE public.session_history ADD COLUMN IF NOT EXISTS task_progress_score DOUBLE PRECISION",
    "ALTER TABLE public.session_history ADD COLUMN IF NOT EXISTS scope_control_score DOUBLE PRECISION",
    "ALTER TABLE public.session_history ADD COLUMN IF NOT EXISTS destructive_action_policy_score DOUBLE PRECISION",
    "ALTER TABLE public.session_history ADD COLUMN IF NOT EXISTS ignored_path_tracking_policy_score DOUBLE PRECISION",
    "ALTER TABLE public.session_history ADD COLUMN IF NOT EXISTS ignored_path_tracking_violation_count INTEGER",
    "ALTER TABLE public.session_history ADD COLUMN IF NOT EXISTS baseline_deflection_attempted_score DOUBLE PRECISION",
    "ALTER TABLE public.session_history ADD COLUMN IF NOT EXISTS baseline_deflection_incident_score DOUBLE PRECISION",
    "ALTER TABLE public.session_history ADD COLUMN IF NOT EXISTS baseline_deflection_attempt_count INTEGER",
    "ALTER TABLE public.session_history ADD COLUMN IF NOT EXISTS baseline_deflection_tool_call_count INTEGER",
    "ALTER TABLE public.session_history ADD COLUMN IF NOT EXISTS baseline_deflection_input_tokens INTEGER",
    "ALTER TABLE public.session_history ADD COLUMN IF NOT EXISTS baseline_deflection_elapsed_ms DOUBLE PRECISION",
    "ALTER TABLE public.session_history ADD COLUMN IF NOT EXISTS quality_gate_trigger_count INTEGER",
    "ALTER TABLE public.session_history ADD COLUMN IF NOT EXISTS quality_gate_fix_attempt_count INTEGER",
    "ALTER TABLE public.session_history ADD COLUMN IF NOT EXISTS quality_gate_rerun_count INTEGER",
    "ALTER TABLE public.session_history ADD COLUMN IF NOT EXISTS sleep_wellness_interruption_attempted_score DOUBLE PRECISION",
    "ALTER TABLE public.session_history ADD COLUMN IF NOT EXISTS sleep_wellness_interruption_incident_score DOUBLE PRECISION",
    "ALTER TABLE public.session_history ADD COLUMN IF NOT EXISTS sleep_wellness_interruption_count INTEGER",
    "ALTER TABLE public.session_history ADD COLUMN IF NOT EXISTS sleep_wellness_interruption_output_tokens INTEGER",
    "ALTER TABLE public.session_history ADD COLUMN IF NOT EXISTS sleep_wellness_interruption_input_tokens INTEGER",
    "ALTER TABLE public.session_history ADD COLUMN IF NOT EXISTS sleep_wellness_interruption_elapsed_ms DOUBLE PRECISION",
    "ALTER TABLE public.session_history ADD COLUMN IF NOT EXISTS sleep_wellness_interruption_after_user_pushback_count INTEGER",
    "ALTER TABLE public.session_history ADD COLUMN IF NOT EXISTS sleep_wellness_interruption_repeated_count INTEGER",
    "ALTER TABLE public.session_history ADD COLUMN IF NOT EXISTS terminal_completion_score DOUBLE PRECISION",
    "ALTER TABLE public.session_history ADD COLUMN IF NOT EXISTS discovery_inventory_coverage_score DOUBLE PRECISION",
    "ALTER TABLE public.session_history ADD COLUMN IF NOT EXISTS discovery_inventory_missing_count INTEGER",
    "ALTER TABLE public.session_history ADD COLUMN IF NOT EXISTS agent_score_reasons "
    "JSONB NOT NULL DEFAULT '{}'::jsonb",
)


def _session_history_score_values(
    evidence: TraceScoreEvidence,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    payload_resolved = evidence.score_payload_resolved
    trace_quality_score = evidence.trace_quality_score if payload_resolved else None
    empty_completion_failure = evidence.empty_completion_failure if payload_resolved else None
    large_tool_result_payload_risk = (
        evidence.large_tool_result_payload_risk if payload_resolved else None
    )
    destructive_checkout_after_work = (
        evidence.destructive_checkout_after_work if payload_resolved else None
    )
    invalid_tool_call_error = (
        True
        if evidence.invalid_tool_call_error_count > 0
        else (False if payload_resolved else None)
    )
    read_only_policy_violation_count = (
        evidence.read_only_policy_violation_count
        if evidence.read_only_policy_compliance_score is not None
        else None
    )
    ignored_path_tracking_violation_count = (
        evidence.ignored_path_tracking_violation_count
        if evidence.ignored_path_tracking_policy_score is not None
        else None
    )
    agent_score_reasons = {
        key: value for key, value in evidence.agent_score_reasons.items() if value
    }
    raw_score_metadata = {
        "usage_trace_quality_score": trace_quality_score,
        "usage_empty_completion_failure": empty_completion_failure,
        "usage_large_tool_result_payload_risk": large_tool_result_payload_risk,
        "usage_destructive_checkout_after_work": destructive_checkout_after_work,
        "usage_invalid_tool_call_error": invalid_tool_call_error,
        "usage_read_only_policy_compliance_score": (
            evidence.read_only_policy_compliance_score
        ),
        "usage_read_only_policy_violation_count": read_only_policy_violation_count,
        "usage_response_meaningfulness_score": evidence.response_meaningfulness_score,
        "usage_instruction_adherence_score": evidence.instruction_adherence_score,
        "usage_answer_completeness_score": evidence.answer_completeness_score,
        "usage_evidence_fidelity_score": evidence.evidence_fidelity_score,
        "usage_tool_result_fidelity_score": evidence.tool_result_fidelity_score,
        "usage_error_attribution_quality_score": evidence.error_attribution_quality_score,
        "usage_repetition_loop_risk_score": evidence.repetition_loop_risk_score,
        "usage_context_retention_score": evidence.context_retention_score,
        "usage_tool_use_validity_score": evidence.tool_use_validity_score,
        "usage_tool_error_recovery_score": evidence.tool_error_recovery_score,
        "usage_stall_risk_score": evidence.stall_risk_score,
        "usage_output_contract_compliance_score": (
            evidence.output_contract_compliance_score
        ),
        "usage_task_progress_score": evidence.task_progress_score,
        "usage_scope_control_score": evidence.scope_control_score,
        "usage_destructive_action_policy_score": (
            evidence.destructive_action_policy_score
        ),
        "usage_ignored_path_tracking_policy_score": (
            evidence.ignored_path_tracking_policy_score
        ),
        "usage_ignored_path_tracking_violation_count": (
            ignored_path_tracking_violation_count
        ),
        "usage_baseline_deflection_attempted_score": (
            evidence.baseline_deflection_attempted_score
        ),
        "usage_baseline_deflection_incident_score": (
            evidence.baseline_deflection_incident_score
        ),
        "usage_baseline_deflection_attempt_count": (
            evidence.baseline_deflection_attempt_count
        ),
        "usage_baseline_deflection_tool_call_count": (
            evidence.baseline_deflection_tool_call_count
        ),
        "usage_baseline_deflection_input_tokens": (
            evidence.baseline_deflection_input_tokens
        ),
        "usage_baseline_deflection_elapsed_ms": (
            evidence.baseline_deflection_elapsed_ms
        ),
        "usage_quality_gate_trigger_count": evidence.quality_gate_trigger_count,
        "usage_quality_gate_fix_attempt_count": (
            evidence.quality_gate_fix_attempt_count
        ),
        "usage_quality_gate_rerun_count": evidence.quality_gate_rerun_count,
        "usage_sleep_wellness_interruption_attempted_score": (
            evidence.sleep_wellness_interruption_attempted_score
        ),
        "usage_sleep_wellness_interruption_incident_score": (
            evidence.sleep_wellness_interruption_incident_score
        ),
        "usage_sleep_wellness_interruption_count": (
            evidence.sleep_wellness_interruption_count
        ),
        "usage_sleep_wellness_interruption_output_tokens": (
            evidence.sleep_wellness_interruption_output_tokens
        ),
        "usage_sleep_wellness_interruption_input_tokens": (
            evidence.sleep_wellness_interruption_input_tokens
        ),
        "usage_sleep_wellness_interruption_elapsed_ms": (
            evidence.sleep_wellness_interruption_elapsed_ms
        ),
        "usage_sleep_wellness_interruption_after_user_pushback_count": (
            evidence.sleep_wellness_interruption_after_user_pushback_count
        ),
        "usage_sleep_wellness_interruption_repeated_count": (
            evidence.sleep_wellness_interruption_repeated_count
        ),
        "usage_terminal_completion_score": evidence.terminal_completion_score,
        "usage_discovery_inventory_coverage_score": (
            evidence.discovery_inventory_coverage_score
        ),
        "usage_discovery_inventory_missing_count": (
            evidence.discovery_inventory_missing_count
        ),
        "usage_agent_score_reasons": agent_score_reasons,
        "usage_agent_score_source": "score_agent_trace_quality.py",
    }
    score_metadata = {
        key: value
        for key, value in raw_score_metadata.items()
        if value is not None and value != {}
    }
    params = {
        "trace_quality_score": trace_quality_score,
        "empty_completion_failure": empty_completion_failure,
        "large_tool_result_payload_risk": large_tool_result_payload_risk,
        "destructive_checkout_after_work": destructive_checkout_after_work,
        "invalid_tool_call_error": invalid_tool_call_error,
        "read_only_policy_compliance_score": evidence.read_only_policy_compliance_score,
        "read_only_policy_violation_count": read_only_policy_violation_count,
        "response_meaningfulness_score": evidence.response_meaningfulness_score,
        "instruction_adherence_score": evidence.instruction_adherence_score,
        "answer_completeness_score": evidence.answer_completeness_score,
        "evidence_fidelity_score": evidence.evidence_fidelity_score,
        "tool_result_fidelity_score": evidence.tool_result_fidelity_score,
        "error_attribution_quality_score": evidence.error_attribution_quality_score,
        "repetition_loop_risk_score": evidence.repetition_loop_risk_score,
        "context_retention_score": evidence.context_retention_score,
        "tool_use_validity_score": evidence.tool_use_validity_score,
        "tool_error_recovery_score": evidence.tool_error_recovery_score,
        "stall_risk_score": evidence.stall_risk_score,
        "output_contract_compliance_score": evidence.output_contract_compliance_score,
        "task_progress_score": evidence.task_progress_score,
        "scope_control_score": evidence.scope_control_score,
        "destructive_action_policy_score": evidence.destructive_action_policy_score,
        "ignored_path_tracking_policy_score": (
            evidence.ignored_path_tracking_policy_score
        ),
        "ignored_path_tracking_violation_count": ignored_path_tracking_violation_count,
        "baseline_deflection_attempted_score": (
            evidence.baseline_deflection_attempted_score
        ),
        "baseline_deflection_incident_score": (
            evidence.baseline_deflection_incident_score
        ),
        "baseline_deflection_attempt_count": evidence.baseline_deflection_attempt_count,
        "baseline_deflection_tool_call_count": (
            evidence.baseline_deflection_tool_call_count
        ),
        "baseline_deflection_input_tokens": evidence.baseline_deflection_input_tokens,
        "baseline_deflection_elapsed_ms": evidence.baseline_deflection_elapsed_ms,
        "quality_gate_trigger_count": evidence.quality_gate_trigger_count,
        "quality_gate_fix_attempt_count": evidence.quality_gate_fix_attempt_count,
        "quality_gate_rerun_count": evidence.quality_gate_rerun_count,
        "sleep_wellness_interruption_attempted_score": (
            evidence.sleep_wellness_interruption_attempted_score
        ),
        "sleep_wellness_interruption_incident_score": (
            evidence.sleep_wellness_interruption_incident_score
        ),
        "sleep_wellness_interruption_count": (
            evidence.sleep_wellness_interruption_count
        ),
        "sleep_wellness_interruption_output_tokens": (
            evidence.sleep_wellness_interruption_output_tokens
        ),
        "sleep_wellness_interruption_input_tokens": (
            evidence.sleep_wellness_interruption_input_tokens
        ),
        "sleep_wellness_interruption_elapsed_ms": (
            evidence.sleep_wellness_interruption_elapsed_ms
        ),
        "sleep_wellness_interruption_after_user_pushback_count": (
            evidence.sleep_wellness_interruption_after_user_pushback_count
        ),
        "sleep_wellness_interruption_repeated_count": (
            evidence.sleep_wellness_interruption_repeated_count
        ),
        "terminal_completion_score": evidence.terminal_completion_score,
        "discovery_inventory_coverage_score": (
            evidence.discovery_inventory_coverage_score
        ),
        "discovery_inventory_missing_count": evidence.discovery_inventory_missing_count,
        "agent_score_reasons": json.dumps(agent_score_reasons),
        "score_metadata": json.dumps(score_metadata),
    }
    return params, score_metadata


def _update_session_history_scores(
    args: argparse.Namespace,
    evidences: Sequence[TraceScoreEvidence],
) -> int:
    if not evidences:
        return 0
    update_sql = """
UPDATE public.session_history
SET
  trace_quality_score = %(trace_quality_score)s,
  empty_completion_failure = %(empty_completion_failure)s,
  large_tool_result_payload_risk = %(large_tool_result_payload_risk)s,
  destructive_checkout_after_work = %(destructive_checkout_after_work)s,
  invalid_tool_call_error = %(invalid_tool_call_error)s,
  read_only_policy_compliance_score = %(read_only_policy_compliance_score)s,
  read_only_policy_violation_count = %(read_only_policy_violation_count)s,
  response_meaningfulness_score = %(response_meaningfulness_score)s,
  instruction_adherence_score = %(instruction_adherence_score)s,
  answer_completeness_score = %(answer_completeness_score)s,
  evidence_fidelity_score = %(evidence_fidelity_score)s,
  tool_result_fidelity_score = %(tool_result_fidelity_score)s,
  error_attribution_quality_score = %(error_attribution_quality_score)s,
  repetition_loop_risk_score = %(repetition_loop_risk_score)s,
  context_retention_score = %(context_retention_score)s,
  tool_use_validity_score = %(tool_use_validity_score)s,
  tool_error_recovery_score = %(tool_error_recovery_score)s,
  stall_risk_score = %(stall_risk_score)s,
  output_contract_compliance_score = %(output_contract_compliance_score)s,
  task_progress_score = %(task_progress_score)s,
  scope_control_score = %(scope_control_score)s,
  destructive_action_policy_score = %(destructive_action_policy_score)s,
  ignored_path_tracking_policy_score = %(ignored_path_tracking_policy_score)s,
  ignored_path_tracking_violation_count = %(ignored_path_tracking_violation_count)s,
  baseline_deflection_attempted_score = %(baseline_deflection_attempted_score)s,
  baseline_deflection_incident_score = %(baseline_deflection_incident_score)s,
  baseline_deflection_attempt_count = %(baseline_deflection_attempt_count)s,
  baseline_deflection_tool_call_count = %(baseline_deflection_tool_call_count)s,
  baseline_deflection_input_tokens = %(baseline_deflection_input_tokens)s,
  baseline_deflection_elapsed_ms = %(baseline_deflection_elapsed_ms)s,
  quality_gate_trigger_count = %(quality_gate_trigger_count)s,
  quality_gate_fix_attempt_count = %(quality_gate_fix_attempt_count)s,
  quality_gate_rerun_count = %(quality_gate_rerun_count)s,
  sleep_wellness_interruption_attempted_score = %(sleep_wellness_interruption_attempted_score)s,
  sleep_wellness_interruption_incident_score = %(sleep_wellness_interruption_incident_score)s,
  sleep_wellness_interruption_count = %(sleep_wellness_interruption_count)s,
  sleep_wellness_interruption_output_tokens = %(sleep_wellness_interruption_output_tokens)s,
  sleep_wellness_interruption_input_tokens = %(sleep_wellness_interruption_input_tokens)s,
  sleep_wellness_interruption_elapsed_ms = %(sleep_wellness_interruption_elapsed_ms)s,
  sleep_wellness_interruption_after_user_pushback_count = %(sleep_wellness_interruption_after_user_pushback_count)s,
  sleep_wellness_interruption_repeated_count = %(sleep_wellness_interruption_repeated_count)s,
  terminal_completion_score = %(terminal_completion_score)s,
  discovery_inventory_coverage_score = %(discovery_inventory_coverage_score)s,
  discovery_inventory_missing_count = %(discovery_inventory_missing_count)s,
  agent_score_reasons = %(agent_score_reasons)s::jsonb,
  metadata = COALESCE(metadata, '{}'::jsonb) || %(score_metadata)s::jsonb
WHERE id = %(row_id)s
"""
    updated = 0
    with psycopg.connect(_postgres_dsn_from_args(args)) as conn:
        _verify_target_database(conn, args.require_target_database)
        if args.ensure_session_history_score_schema:
            with conn.cursor() as cur:
                for statement in _SESSION_HISTORY_SCORE_ALTER_STATEMENTS:
                    cur.execute(statement)
        with conn.cursor() as cur:
            param_rows: List[Dict[str, Any]] = []
            for evidence in evidences:
                params, _score_metadata = _session_history_score_values(evidence)
                params["row_id"] = evidence.row_id
                param_rows.append(params)
            if hasattr(cur, "executemany"):
                cur.executemany(update_sql, param_rows)
                updated += cur.rowcount if cur.rowcount >= 0 else len(param_rows)
            else:
                for params in param_rows:
                    cur.execute(update_sql, params)
                    updated += cur.rowcount
        conn.commit()
    return updated


def _upsert_codex_transcript_session_history(
    args: argparse.Namespace,
    pairs: Sequence[Tuple[SessionCandidate, TraceScoreEvidence]],
) -> int:
    if not pairs:
        return 0
    upsert_sql = """
INSERT INTO public.session_history (
  created_at,
  litellm_call_id,
  session_id,
  trace_id,
  provider,
  model,
  model_group,
  agent_name,
  tenant_id,
  call_type,
  start_time,
  end_time,
  input_tokens,
  output_tokens,
  total_tokens,
  tool_call_count,
  invalid_tool_call_count,
  litellm_environment,
  client_name,
  client_version,
  metadata,
  repository,
  llm_upstream_elapsed_ms,
  total_server_elapsed_ms,
  trace_quality_score,
  empty_completion_failure,
  large_tool_result_payload_risk,
  destructive_checkout_after_work,
  invalid_tool_call_error,
  read_only_policy_compliance_score,
  read_only_policy_violation_count,
  response_meaningfulness_score,
  instruction_adherence_score,
  answer_completeness_score,
  evidence_fidelity_score,
  tool_result_fidelity_score,
  error_attribution_quality_score,
  repetition_loop_risk_score,
  context_retention_score,
  tool_use_validity_score,
  tool_error_recovery_score,
  stall_risk_score,
  output_contract_compliance_score,
  task_progress_score,
  scope_control_score,
  destructive_action_policy_score,
  ignored_path_tracking_policy_score,
  ignored_path_tracking_violation_count,
  baseline_deflection_attempted_score,
  baseline_deflection_incident_score,
  baseline_deflection_attempt_count,
  baseline_deflection_tool_call_count,
  baseline_deflection_input_tokens,
  baseline_deflection_elapsed_ms,
  quality_gate_trigger_count,
  quality_gate_fix_attempt_count,
  quality_gate_rerun_count,
  sleep_wellness_interruption_attempted_score,
  sleep_wellness_interruption_incident_score,
  sleep_wellness_interruption_count,
  sleep_wellness_interruption_output_tokens,
  sleep_wellness_interruption_input_tokens,
  sleep_wellness_interruption_elapsed_ms,
  sleep_wellness_interruption_after_user_pushback_count,
  sleep_wellness_interruption_repeated_count,
  terminal_completion_score,
  discovery_inventory_coverage_score,
  discovery_inventory_missing_count,
  agent_score_reasons
) VALUES (
  %(created_at)s,
  %(litellm_call_id)s,
  %(session_id)s,
  %(trace_id)s,
  %(provider)s,
  %(model)s,
  %(model_group)s,
  %(agent_name)s,
  %(tenant_id)s,
  %(call_type)s,
  %(start_time)s,
  %(end_time)s,
  %(input_tokens)s,
  %(output_tokens)s,
  %(total_tokens)s,
  %(tool_call_count)s,
  %(invalid_tool_call_count)s,
  %(litellm_environment)s,
  %(client_name)s,
  %(client_version)s,
  %(metadata)s::jsonb,
  %(repository)s,
  %(llm_upstream_elapsed_ms)s,
  %(total_server_elapsed_ms)s,
  %(trace_quality_score)s,
  %(empty_completion_failure)s,
  %(large_tool_result_payload_risk)s,
  %(destructive_checkout_after_work)s,
  %(invalid_tool_call_error)s,
  %(read_only_policy_compliance_score)s,
  %(read_only_policy_violation_count)s,
  %(response_meaningfulness_score)s,
  %(instruction_adherence_score)s,
  %(answer_completeness_score)s,
  %(evidence_fidelity_score)s,
  %(tool_result_fidelity_score)s,
  %(error_attribution_quality_score)s,
  %(repetition_loop_risk_score)s,
  %(context_retention_score)s,
  %(tool_use_validity_score)s,
  %(tool_error_recovery_score)s,
  %(stall_risk_score)s,
  %(output_contract_compliance_score)s,
  %(task_progress_score)s,
  %(scope_control_score)s,
  %(destructive_action_policy_score)s,
  %(ignored_path_tracking_policy_score)s,
  %(ignored_path_tracking_violation_count)s,
  %(baseline_deflection_attempted_score)s,
  %(baseline_deflection_incident_score)s,
  %(baseline_deflection_attempt_count)s,
  %(baseline_deflection_tool_call_count)s,
  %(baseline_deflection_input_tokens)s,
  %(baseline_deflection_elapsed_ms)s,
  %(quality_gate_trigger_count)s,
  %(quality_gate_fix_attempt_count)s,
  %(quality_gate_rerun_count)s,
  %(sleep_wellness_interruption_attempted_score)s,
  %(sleep_wellness_interruption_incident_score)s,
  %(sleep_wellness_interruption_count)s,
  %(sleep_wellness_interruption_output_tokens)s,
  %(sleep_wellness_interruption_input_tokens)s,
  %(sleep_wellness_interruption_elapsed_ms)s,
  %(sleep_wellness_interruption_after_user_pushback_count)s,
  %(sleep_wellness_interruption_repeated_count)s,
  %(terminal_completion_score)s,
  %(discovery_inventory_coverage_score)s,
  %(discovery_inventory_missing_count)s,
  %(agent_score_reasons)s::jsonb
)
ON CONFLICT (litellm_call_id) DO UPDATE SET
  created_at = LEAST(session_history.created_at, EXCLUDED.created_at),
  session_id = EXCLUDED.session_id,
  trace_id = COALESCE(EXCLUDED.trace_id, session_history.trace_id),
  provider = CASE
    WHEN session_history.call_type = 'codex_transcript'
      AND session_history.provider = 'litellm'
      AND EXCLUDED.provider IS NULL
      THEN NULL
    ELSE COALESCE(EXCLUDED.provider, session_history.provider)
  END,
  model = CASE
    WHEN session_history.call_type = 'codex_transcript'
      AND session_history.model = 'aawm-codex-agent-auto'
      AND EXCLUDED.model IS NULL
      THEN NULL
    ELSE COALESCE(EXCLUDED.model, session_history.model)
  END,
  model_group = CASE
    WHEN session_history.call_type = 'codex_transcript'
      AND session_history.model_group = 'aawm-codex-agent-auto'
      AND EXCLUDED.model_group IS NULL
      THEN NULL
    ELSE COALESCE(EXCLUDED.model_group, session_history.model_group)
  END,
  agent_name = COALESCE(EXCLUDED.agent_name, session_history.agent_name),
  tenant_id = COALESCE(EXCLUDED.tenant_id, session_history.tenant_id),
  call_type = COALESCE(EXCLUDED.call_type, session_history.call_type),
  start_time = COALESCE(session_history.start_time, EXCLUDED.start_time),
  end_time = COALESCE(EXCLUDED.end_time, session_history.end_time),
  input_tokens = GREATEST(session_history.input_tokens, EXCLUDED.input_tokens),
  output_tokens = GREATEST(session_history.output_tokens, EXCLUDED.output_tokens),
  total_tokens = GREATEST(session_history.total_tokens, EXCLUDED.total_tokens),
  tool_call_count = GREATEST(session_history.tool_call_count, EXCLUDED.tool_call_count),
  invalid_tool_call_count = GREATEST(
    session_history.invalid_tool_call_count,
    EXCLUDED.invalid_tool_call_count
  ),
  litellm_environment = COALESCE(
    EXCLUDED.litellm_environment,
    session_history.litellm_environment
  ),
  client_name = COALESCE(EXCLUDED.client_name, session_history.client_name),
  client_version = COALESCE(EXCLUDED.client_version, session_history.client_version),
  metadata = COALESCE(session_history.metadata, '{}'::jsonb) || EXCLUDED.metadata,
  repository = COALESCE(EXCLUDED.repository, session_history.repository),
  llm_upstream_elapsed_ms = COALESCE(
    EXCLUDED.llm_upstream_elapsed_ms,
    session_history.llm_upstream_elapsed_ms
  ),
  total_server_elapsed_ms = COALESCE(
    EXCLUDED.total_server_elapsed_ms,
    session_history.total_server_elapsed_ms
  ),
  trace_quality_score = COALESCE(
    EXCLUDED.trace_quality_score,
    session_history.trace_quality_score
  ),
  empty_completion_failure = COALESCE(
    EXCLUDED.empty_completion_failure,
    session_history.empty_completion_failure
  ),
  large_tool_result_payload_risk = COALESCE(
    EXCLUDED.large_tool_result_payload_risk,
    session_history.large_tool_result_payload_risk
  ),
  destructive_checkout_after_work = COALESCE(
    EXCLUDED.destructive_checkout_after_work,
    session_history.destructive_checkout_after_work
  ),
  invalid_tool_call_error = COALESCE(
    EXCLUDED.invalid_tool_call_error,
    session_history.invalid_tool_call_error
  ),
  read_only_policy_compliance_score = COALESCE(
    EXCLUDED.read_only_policy_compliance_score,
    session_history.read_only_policy_compliance_score
  ),
  read_only_policy_violation_count = COALESCE(
    EXCLUDED.read_only_policy_violation_count,
    session_history.read_only_policy_violation_count
  ),
  response_meaningfulness_score = COALESCE(
    EXCLUDED.response_meaningfulness_score,
    session_history.response_meaningfulness_score
  ),
  instruction_adherence_score = COALESCE(
    EXCLUDED.instruction_adherence_score,
    session_history.instruction_adherence_score
  ),
  answer_completeness_score = COALESCE(
    EXCLUDED.answer_completeness_score,
    session_history.answer_completeness_score
  ),
  evidence_fidelity_score = COALESCE(
    EXCLUDED.evidence_fidelity_score,
    session_history.evidence_fidelity_score
  ),
  tool_result_fidelity_score = COALESCE(
    EXCLUDED.tool_result_fidelity_score,
    session_history.tool_result_fidelity_score
  ),
  error_attribution_quality_score = COALESCE(
    EXCLUDED.error_attribution_quality_score,
    session_history.error_attribution_quality_score
  ),
  repetition_loop_risk_score = COALESCE(
    EXCLUDED.repetition_loop_risk_score,
    session_history.repetition_loop_risk_score
  ),
  context_retention_score = COALESCE(
    EXCLUDED.context_retention_score,
    session_history.context_retention_score
  ),
  tool_use_validity_score = COALESCE(
    EXCLUDED.tool_use_validity_score,
    session_history.tool_use_validity_score
  ),
  tool_error_recovery_score = COALESCE(
    EXCLUDED.tool_error_recovery_score,
    session_history.tool_error_recovery_score
  ),
  stall_risk_score = COALESCE(
    EXCLUDED.stall_risk_score,
    session_history.stall_risk_score
  ),
  output_contract_compliance_score = COALESCE(
    EXCLUDED.output_contract_compliance_score,
    session_history.output_contract_compliance_score
  ),
  task_progress_score = COALESCE(
    EXCLUDED.task_progress_score,
    session_history.task_progress_score
  ),
  scope_control_score = COALESCE(
    EXCLUDED.scope_control_score,
    session_history.scope_control_score
  ),
  destructive_action_policy_score = COALESCE(
    EXCLUDED.destructive_action_policy_score,
    session_history.destructive_action_policy_score
  ),
  ignored_path_tracking_policy_score = COALESCE(
    EXCLUDED.ignored_path_tracking_policy_score,
    session_history.ignored_path_tracking_policy_score
  ),
  ignored_path_tracking_violation_count = COALESCE(
    EXCLUDED.ignored_path_tracking_violation_count,
    session_history.ignored_path_tracking_violation_count
  ),
  baseline_deflection_attempted_score = COALESCE(
    EXCLUDED.baseline_deflection_attempted_score,
    session_history.baseline_deflection_attempted_score
  ),
  baseline_deflection_incident_score = COALESCE(
    EXCLUDED.baseline_deflection_incident_score,
    session_history.baseline_deflection_incident_score
  ),
  baseline_deflection_attempt_count = COALESCE(
    EXCLUDED.baseline_deflection_attempt_count,
    session_history.baseline_deflection_attempt_count
  ),
  baseline_deflection_tool_call_count = COALESCE(
    EXCLUDED.baseline_deflection_tool_call_count,
    session_history.baseline_deflection_tool_call_count
  ),
  baseline_deflection_input_tokens = COALESCE(
    EXCLUDED.baseline_deflection_input_tokens,
    session_history.baseline_deflection_input_tokens
  ),
  baseline_deflection_elapsed_ms = COALESCE(
    EXCLUDED.baseline_deflection_elapsed_ms,
    session_history.baseline_deflection_elapsed_ms
  ),
  quality_gate_trigger_count = COALESCE(
    EXCLUDED.quality_gate_trigger_count,
    session_history.quality_gate_trigger_count
  ),
  quality_gate_fix_attempt_count = COALESCE(
    EXCLUDED.quality_gate_fix_attempt_count,
    session_history.quality_gate_fix_attempt_count
  ),
  quality_gate_rerun_count = COALESCE(
    EXCLUDED.quality_gate_rerun_count,
    session_history.quality_gate_rerun_count
  ),
  sleep_wellness_interruption_attempted_score = COALESCE(
    EXCLUDED.sleep_wellness_interruption_attempted_score,
    session_history.sleep_wellness_interruption_attempted_score
  ),
  sleep_wellness_interruption_incident_score = COALESCE(
    EXCLUDED.sleep_wellness_interruption_incident_score,
    session_history.sleep_wellness_interruption_incident_score
  ),
  sleep_wellness_interruption_count = COALESCE(
    EXCLUDED.sleep_wellness_interruption_count,
    session_history.sleep_wellness_interruption_count
  ),
  sleep_wellness_interruption_output_tokens = COALESCE(
    EXCLUDED.sleep_wellness_interruption_output_tokens,
    session_history.sleep_wellness_interruption_output_tokens
  ),
  sleep_wellness_interruption_input_tokens = COALESCE(
    EXCLUDED.sleep_wellness_interruption_input_tokens,
    session_history.sleep_wellness_interruption_input_tokens
  ),
  sleep_wellness_interruption_elapsed_ms = COALESCE(
    EXCLUDED.sleep_wellness_interruption_elapsed_ms,
    session_history.sleep_wellness_interruption_elapsed_ms
  ),
  sleep_wellness_interruption_after_user_pushback_count = COALESCE(
    EXCLUDED.sleep_wellness_interruption_after_user_pushback_count,
    session_history.sleep_wellness_interruption_after_user_pushback_count
  ),
  sleep_wellness_interruption_repeated_count = COALESCE(
    EXCLUDED.sleep_wellness_interruption_repeated_count,
    session_history.sleep_wellness_interruption_repeated_count
  ),
  terminal_completion_score = COALESCE(
    EXCLUDED.terminal_completion_score,
    session_history.terminal_completion_score
  ),
  discovery_inventory_coverage_score = COALESCE(
    EXCLUDED.discovery_inventory_coverage_score,
    session_history.discovery_inventory_coverage_score
  ),
  discovery_inventory_missing_count = COALESCE(
    EXCLUDED.discovery_inventory_missing_count,
    session_history.discovery_inventory_missing_count
  ),
  agent_score_reasons = COALESCE(
    NULLIF(EXCLUDED.agent_score_reasons, '{}'::jsonb),
    session_history.agent_score_reasons
  )
RETURNING id
"""
    updated = 0
    with psycopg.connect(_postgres_dsn_from_args(args)) as conn:
        _verify_target_database(conn, args.require_target_database)
        if args.ensure_session_history_score_schema:
            with conn.cursor() as cur:
                for statement in _SESSION_HISTORY_SCORE_ALTER_STATEMENTS:
                    cur.execute(statement)
        with conn.cursor() as cur:
            for candidate, evidence in pairs:
                score_params, score_metadata = _session_history_score_values(evidence)
                metadata = dict(candidate.metadata)
                metadata.update(score_metadata)
                metadata.update(
                    {
                        "synthetic": True,
                        "session_history_usage_record": False,
                        "session_history_reporting_excluded": True,
                        "session_history_reporting_exclusion_reason": (
                            "synthetic_codex_transcript"
                        ),
                        "source_status": "transcript_derived",
                        "session_history_repair_source": (
                            "d1_159_codex_transcript_policy_scoring"
                        ),
                        "langfuse_score_emit_status": (
                            "requested"
                            if getattr(args, "apply", False)
                            else "not_emitted_synthetic_transcript"
                        ),
                    }
                )
                params = {
                    **score_params,
                    "created_at": candidate.created_at,
                    "litellm_call_id": candidate.litellm_call_id,
                    "session_id": candidate.session_id,
                    "trace_id": candidate.trace_id,
                    "provider": _codex_transcript_session_history_provider(
                        candidate.provider
                    ),
                    "model": _codex_transcript_session_history_model(candidate.model),
                    "model_group": _codex_transcript_session_history_model_group(
                        candidate.model
                    ),
                    "agent_name": candidate.agent_name,
                    "tenant_id": candidate.tenant_id,
                    "call_type": "codex_transcript",
                    "start_time": candidate.created_at,
                    "end_time": candidate.created_at,
                    "input_tokens": candidate.input_tokens,
                    "output_tokens": candidate.output_tokens,
                    "total_tokens": _coerce_int(
                        candidate.metadata.get("codex_transcript_total_tokens")
                    )
                    or candidate.input_tokens
                    + candidate.output_tokens,
                    "tool_call_count": candidate.tool_call_count,
                    "invalid_tool_call_count": candidate.invalid_tool_call_count,
                    "litellm_environment": "transcript",
                    "client_name": _clean(candidate.metadata.get("originator")),
                    "client_version": _clean(candidate.metadata.get("cli_version")),
                    "metadata": json.dumps(metadata),
                    "repository": candidate.repository,
                    "llm_upstream_elapsed_ms": candidate.llm_upstream_elapsed_ms,
                    "total_server_elapsed_ms": candidate.total_server_elapsed_ms,
                }
                cur.execute(upsert_sql, params)
                if cur.fetchone():
                    updated += 1
        conn.commit()
    return updated


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


def _should_fetch_session_history_candidates(args: argparse.Namespace) -> bool:
    if not args.codex_transcript:
        return True
    return any(
        (
            args.trace_id,
            args.session_id,
            args.repository,
            args.agent_name,
            args.model,
            args.provider,
            args.from_created_at,
            args.to_created_at,
            args.candidate_only,
        )
    )


def _run(args: argparse.Namespace) -> Dict[str, Any]:
    candidates = (
        _fetch_session_candidates(args)
        if _should_fetch_session_history_candidates(args)
        else []
    )
    transcript_bundles = _resolve_codex_transcript_bundles(args)
    transcript_candidate_by_call_id = {
        bundle.candidate.litellm_call_id: bundle.candidate
        for bundle in transcript_bundles
        if bundle.candidate.litellm_call_id
    }
    candidates.extend(bundle.candidate for bundle in transcript_bundles)
    db_candidates = [candidate for candidate in candidates if candidate.row_id >= 0]
    provider_error_trace_ids, provider_error_call_ids = _fetch_provider_error_keys(
        args,
        db_candidates,
    )
    payloads = _resolve_observation_payloads(args, db_candidates)
    for bundle in transcript_bundles:
        candidate = bundle.candidate
        for key in (
            candidate.source_observation_id,
            candidate.litellm_call_id,
            candidate.trace_id,
        ):
            if key:
                payloads[key] = bundle.payload
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

    session_history_update_count = 0
    if args.update_session_history_scores and evidences:
        db_evidences = [evidence for evidence in evidences if evidence.row_id >= 0]
        session_history_update_count = _update_session_history_scores(args, db_evidences)
        if args.upsert_codex_transcript_session_history:
            transcript_pairs = [
                (candidate, evidence)
                for evidence in evidences
                for candidate in [
                    transcript_candidate_by_call_id.get(evidence.observation_id)
                ]
                if candidate is not None
            ]
            session_history_update_count += _upsert_codex_transcript_session_history(
                args,
                transcript_pairs,
            )

    return {
        "applied": bool(args.apply),
        "session_history_scores_updated": bool(args.update_session_history_scores),
        "codex_transcript_session_history_upserted": bool(
            args.upsert_codex_transcript_session_history
        ),
        "candidate_count": len(candidates),
        "evidence_count": len(evidences),
        "score_write_count": len(score_results),
        "session_history_update_count": session_history_update_count,
        "scores": []
        if getattr(args, "summary_only", False)
        else [score.to_json() for score in evidences],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Deterministically score suspicious agent trace quality patterns."
    )
    parser.add_argument("--apply", action="store_true", help="Write scores to Langfuse.")
    parser.add_argument(
        "--update-session-history-scores",
        action="store_true",
        help="Write deterministic score columns and score metadata to session_history.",
    )
    parser.add_argument(
        "--ensure-session-history-score-schema",
        action="store_true",
        help="Apply additive session_history score columns before updating rows.",
    )
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--trace-id", action="append", default=[])
    parser.add_argument("--session-id", action="append", default=[])
    parser.add_argument(
        "--codex-transcript",
        action="append",
        default=[],
        help="Score a local Codex JSONL transcript as a transcript-derived candidate.",
    )
    parser.add_argument(
        "--codex-parent-transcript",
        action="append",
        default=[],
        help="Parent Codex JSONL transcript containing the spawn_agent prompt.",
    )
    parser.add_argument(
        "--upsert-codex-transcript-session-history",
        action="store_true",
        help=(
            "When updating session_history scores, insert/update transcript-derived "
            "Codex rows that have no native session_history row."
        ),
    )
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
    parser.add_argument(
        "--missing-agent-quality-backfill-only",
        action="store_true",
        help=(
            "Only scan rows missing the incident-oriented agent-quality scores. "
            "This is intended for resumable session_history backfills; rows with "
            "unavailable payloads may still leave ignored-path policy unevaluated."
        ),
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Print aggregate counts without per-row score payloads.",
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
    parser.add_argument(
        "--minio-read-workers",
        type=int,
        default=1,
        help="Number of parallel MinIO blob reads for offline backfills.",
    )
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
