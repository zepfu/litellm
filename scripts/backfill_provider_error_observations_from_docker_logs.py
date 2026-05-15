#!/usr/bin/env python3
# ruff: noqa: T201
"""
Backfill public.provider_error_observations from retained Docker process logs.

This is intentionally a preservation tool for restart windows where the running
callback cannot yet persist normalized provider failures. It reads Docker's
current json-file log stream through `docker logs --timestamps`, normalizes the
same error shapes the runtime callback records, and dedupes repeated runs via a
metadata log signature.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Set, Tuple

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
    _AAWM_PROVIDER_ERROR_OBSERVATION_INSERT_SQL,
    _build_aawm_dsn,
    _build_provider_error_observation_db_payload,
    _classify_provider_error,
    _ensure_session_history_schema,
)

ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
TIMESTAMP_RE = re.compile(r"^(?P<ts>\S+)\s+(?P<message>.*)$")
EXCEPTION_RE = re.compile(r"Exception occured - (?P<status>\d{3}):\s*(?P<detail>.*)")
ACCESS_RE = re.compile(
    r'INFO:\s+(?P<client>\S+)\s+-\s+"(?P<method>[A-Z]+)\s+'
    r'(?P<path>\S+)\s+HTTP/[0-9.]+"\s+(?P<status>\d{3})\s+(?P<phrase>.*)$'
)
MODEL_RE = re.compile(r"\bmodel\s*[:=]\s*['\"]?(?P<model>[A-Za-z0-9_.:/\-\[\]]+)")
REQUEST_ID_RE = re.compile(r"\brequest[_-]?id\s*[:=]\s*['\"](?P<request_id>[^'\"]+)")
TRACE_ID_RE = re.compile(r"\b(?:trace_id|langfuse_trace_id)\s*[:=]\s*['\"](?P<trace_id>[^'\"]+)")
CALL_ID_RE = re.compile(r"\b(?:litellm_call_id|call_id)\s*[:=]\s*['\"](?P<call_id>[^'\"]+)")

HTTP_ERROR_CODE_BY_STATUS = {
    400: "bad_request",
    401: "unauthorized",
    403: "forbidden",
    404: "not_found",
    405: "method_not_allowed",
    408: "request_timeout",
    409: "conflict",
    422: "validation_error",
    429: "rate_limited",
    500: "internal_server_error",
    502: "bad_gateway",
    503: "service_unavailable",
    504: "gateway_timeout",
    529: "overloaded",
}

PROXY_INTERNAL_MARKERS = (
    "Exception in ASGI application",
    "asyncpg.",
    "database system is in recovery mode",
    "ConnectionDoesNotExistError",
    "CannotConnectNowError",
    "ConnectionResetError",
    "connection was closed in the middle of operation",
)


@dataclass
class LogEntry:
    line_index: int
    timestamp_text: str
    observed_at: datetime
    content: str
    clean_content: str


@dataclass
class AccessLog:
    entry: LogEntry
    method: str
    path: str
    status_code: int
    phrase: str


@dataclass
class ExceptionLog:
    entry: LogEntry
    status_code: int
    detail: str


@dataclass
class ParsedLogs:
    entries: List[LogEntry]
    access_logs: List[AccessLog]
    exception_logs: List[ExceptionLog]


def _clean_ansi(value: str) -> str:
    return ANSI_RE.sub("", value)


def _parse_timestamp(value: str) -> datetime:
    normalized = value.strip()
    if normalized.endswith("Z"):
        normalized = normalized[:-1] + "+00:00"
    if "." in normalized:
        head, tail = normalized.split(".", 1)
        offset = ""
        for marker in ("+", "-"):
            if marker in tail:
                fraction, offset = tail.split(marker, 1)
                offset = marker + offset
                break
        else:
            fraction = tail
        if len(fraction) > 6:
            fraction = fraction[:6]
        normalized = f"{head}.{fraction}{offset}"
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _iter_docker_log_lines(
    *,
    container: str,
    since: Optional[str],
    until: Optional[str],
    raw_log_file: Optional[Path],
) -> Iterator[str]:
    if raw_log_file is not None:
        with raw_log_file.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                yield line.rstrip("\n")
        return

    command = ["docker", "logs", "--timestamps"]
    if since:
        command.extend(["--since", since])
    if until:
        command.extend(["--until", until])
    command.append(container)
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    assert process.stdout is not None
    for line in process.stdout:
        yield line.rstrip("\n")
    process.wait()
    if process.returncode != 0:
        raise RuntimeError(f"docker logs failed with exit {process.returncode}")


def _parse_logs(lines: Iterable[str]) -> ParsedLogs:
    entries: List[LogEntry] = []
    access_logs: List[AccessLog] = []
    exception_logs: List[ExceptionLog] = []

    for line_index, raw_line in enumerate(lines, start=1):
        match = TIMESTAMP_RE.match(raw_line)
        if match is None:
            continue
        try:
            observed_at = _parse_timestamp(match.group("ts"))
        except ValueError:
            continue
        clean_content = _clean_ansi(match.group("message"))
        entry = LogEntry(
            line_index=line_index,
            timestamp_text=match.group("ts"),
            observed_at=observed_at,
            content=match.group("message"),
            clean_content=clean_content,
        )
        entries.append(entry)

        exception_match = EXCEPTION_RE.search(clean_content)
        if exception_match is not None:
            exception_logs.append(
                ExceptionLog(
                    entry=entry,
                    status_code=int(exception_match.group("status")),
                    detail=exception_match.group("detail").strip(),
                )
            )
            continue

        access_match = ACCESS_RE.search(clean_content)
        if access_match is not None:
            status_code = int(access_match.group("status"))
            if status_code >= 400:
                access_logs.append(
                    AccessLog(
                        entry=entry,
                        method=access_match.group("method"),
                        path=access_match.group("path"),
                        status_code=status_code,
                        phrase=access_match.group("phrase").strip(),
                    )
                )

    return ParsedLogs(entries=entries, access_logs=access_logs, exception_logs=exception_logs)


def _extract_first_json_object(value: str) -> Optional[Dict[str, Any]]:
    start = value.find("{")
    if start < 0:
        return None

    depth = 0
    in_string = False
    escape = False
    for offset, char in enumerate(value[start:], start=start):
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                candidate = value[start : offset + 1]
                try:
                    parsed = json.loads(candidate)
                except json.JSONDecodeError:
                    return None
                return parsed if isinstance(parsed, dict) else None
    return None


def _first_non_empty(*values: Any) -> Optional[str]:
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return None


def _extract_json_path(payload: Dict[str, Any], path: Sequence[str]) -> Optional[Any]:
    current: Any = payload
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def _extract_error_details(detail: str) -> Tuple[Optional[str], Optional[str], str, Optional[str]]:
    payload = _extract_first_json_object(detail)
    request_id = None
    if payload is not None:
        error_obj = payload.get("error")
        if not isinstance(error_obj, dict):
            error_obj = payload
        request_id = _first_non_empty(
            payload.get("request_id"),
            payload.get("requestId"),
            _extract_json_path(payload, ("error", "request_id")),
        )
        error_type = _first_non_empty(
            error_obj.get("type") if isinstance(error_obj, dict) else None,
            payload.get("type"),
        )
        error_code = _first_non_empty(
            error_obj.get("code") if isinstance(error_obj, dict) else None,
            payload.get("code"),
        )
        error_text = _first_non_empty(
            error_obj.get("message") if isinstance(error_obj, dict) else None,
            payload.get("message"),
            detail,
        )
        return error_type, error_code, error_text or detail, request_id

    request_match = REQUEST_ID_RE.search(detail)
    if request_match is not None:
        request_id = request_match.group("request_id")
    return None, None, detail, request_id


def _extract_model(detail: str, payload: Optional[Dict[str, Any]]) -> Optional[str]:
    if payload is not None:
        model = _first_non_empty(
            payload.get("model"),
            _extract_json_path(payload, ("error", "model")),
        )
        if model:
            return model
    match = MODEL_RE.search(detail)
    if match is None:
        return None
    return match.group("model").rstrip(".,;")


def _extract_trace_id(detail: str) -> Optional[str]:
    match = TRACE_ID_RE.search(detail)
    return match.group("trace_id") if match is not None else None


def _extract_call_id(detail: str) -> Optional[str]:
    match = CALL_ID_RE.search(detail)
    return match.group("call_id") if match is not None else None


def _infer_provider_and_route_family(
    *,
    path: Optional[str],
    detail: str,
    status_code: int,
    context_lines: Sequence[LogEntry],
) -> Tuple[str, Optional[str], str]:
    normalized = f"{path or ''} {detail}".lower()
    context_text = "\n".join(entry.clean_content for entry in context_lines)
    if status_code >= 500 and _looks_proxy_internal(f"{detail}\n{context_text}"):
        return "proxy_internal", "proxy_internal", "proxy_internal"
    if "cloudcode-pa.googleapis.com" in normalized or "google code assist" in normalized:
        return "gemini", "google_code_assist_generate_content", "provider_exception"
    if path and path.startswith("/anthropic"):
        return "anthropic", "anthropic_messages", "provider_exception"
    if "anthropic" in normalized or "claude-" in normalized:
        return "anthropic", "anthropic_messages", "provider_exception"
    if path and path.startswith("/openai_passthrough/responses"):
        return "openai", "openai_responses", "provider_exception"
    if path and (path.startswith("/v1/chat/completions") or path.startswith("/chat/completions")):
        return "openai", "chat_completions", "provider_exception"
    if path and path.startswith("/v1/embeddings"):
        return "unknown", "embeddings", "provider_exception"
    return "unknown", None, "provider_exception"


def _looks_proxy_internal(value: str) -> bool:
    return any(marker.lower() in value.lower() for marker in PROXY_INTERNAL_MARKERS)


def _status_error_code(status_code: int) -> str:
    return HTTP_ERROR_CODE_BY_STATUS.get(status_code, f"http_{status_code}")


def _find_nearby_context(
    entries: Sequence[LogEntry],
    *,
    observed_at: datetime,
    line_index: int,
    seconds: float,
    limit: int = 8,
) -> List[LogEntry]:
    candidates: List[Tuple[float, LogEntry]] = []
    for entry in entries:
        if abs(entry.line_index - line_index) > 2000:
            continue
        delta = abs((entry.observed_at - observed_at).total_seconds())
        if delta > seconds or entry.line_index == line_index:
            continue
        content = entry.clean_content
        if any(
            marker in content
            for marker in (
                "ERROR:",
                "Traceback",
                "Exception",
                "Connection",
                "asyncpg",
                "database system",
                "Internal Server Error",
            )
        ):
            candidates.append((delta, entry))
    candidates.sort(key=lambda item: (item[0], item[1].line_index))
    return [entry for _, entry in candidates[:limit]]


def _log_signature(
    *,
    container: str,
    timestamp_text: str,
    line_index: int,
    content: str,
) -> str:
    digest = hashlib.sha256()
    digest.update(container.encode("utf-8"))
    digest.update(b"\0")
    digest.update(timestamp_text.encode("utf-8"))
    digest.update(b"\0")
    digest.update(str(line_index).encode("ascii"))
    digest.update(b"\0")
    digest.update(content.encode("utf-8", errors="replace"))
    return digest.hexdigest()


def _nearest_access_log(
    exception_log: ExceptionLog,
    access_logs: Sequence[AccessLog],
    used_access_indexes: Set[int],
    *,
    max_delta_seconds: float,
) -> Optional[AccessLog]:
    candidates: List[Tuple[float, AccessLog]] = []
    for access_log in access_logs:
        if access_log.entry.line_index in used_access_indexes:
            continue
        if access_log.status_code != exception_log.status_code:
            continue
        delta = abs((access_log.entry.observed_at - exception_log.entry.observed_at).total_seconds())
        if delta <= max_delta_seconds:
            candidates.append((delta, access_log))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (item[0], item[1].entry.line_index))
    return candidates[0][1]


def _build_observations(
    *,
    parsed_logs: ParsedLogs,
    container: str,
    environment: str,
    max_correlation_seconds: float,
) -> List[Dict[str, Any]]:
    observations: List[Dict[str, Any]] = []
    used_access_indexes: Set[int] = set()

    for exception_log in parsed_logs.exception_logs:
        access_log = _nearest_access_log(
            exception_log,
            parsed_logs.access_logs,
            used_access_indexes,
            max_delta_seconds=max_correlation_seconds,
        )
        if access_log is not None:
            used_access_indexes.add(access_log.entry.line_index)

        payload = _extract_first_json_object(exception_log.detail)
        error_type, error_code, error_text, request_id = _extract_error_details(exception_log.detail)
        path = access_log.path if access_log is not None else None
        context_lines = _find_nearby_context(
            parsed_logs.entries,
            observed_at=exception_log.entry.observed_at,
            line_index=exception_log.entry.line_index,
            seconds=3.0,
        )
        provider, route_family, error_origin = _infer_provider_and_route_family(
            path=path,
            detail=exception_log.detail,
            status_code=exception_log.status_code,
            context_lines=context_lines,
        )
        observations.append(
            {
                "observed_at": exception_log.entry.observed_at,
                "environment": environment,
                "provider": provider,
                "model": _extract_model(exception_log.detail, payload),
                "model_group": None,
                "route_family": route_family,
                "status_code": exception_log.status_code,
                "error_type": error_type,
                "error_code": error_code,
                "error_class": _classify_provider_error(
                    status_code=exception_log.status_code,
                    error_code=error_code,
                    error_type=error_type,
                    error_text=error_text,
                ),
                "retry_after_seconds": None,
                "expected_reset_at": None,
                "session_id": None,
                "trace_id": _extract_trace_id(exception_log.detail),
                "litellm_call_id": _extract_call_id(exception_log.detail),
                "metadata": _build_metadata(
                    container=container,
                    environment=environment,
                    source_entry=exception_log.entry,
                    source_kind="exception_log",
                    raw_excerpt=exception_log.detail,
                    normalized_error_text=error_text,
                    request_id=request_id,
                    access_log=access_log,
                    context_lines=context_lines,
                    error_origin=error_origin,
                ),
            }
        )

    for access_log in parsed_logs.access_logs:
        if access_log.entry.line_index in used_access_indexes:
            continue
        context_lines = _find_nearby_context(
            parsed_logs.entries,
            observed_at=access_log.entry.observed_at,
            line_index=access_log.entry.line_index,
            seconds=5.0,
        )
        detail = f"{access_log.method} {access_log.path} returned {access_log.status_code} {access_log.phrase}"
        if context_lines:
            detail = f"{detail}; nearby_error={context_lines[0].clean_content}"
        provider, route_family, error_origin = _infer_provider_and_route_family(
            path=access_log.path,
            detail=detail,
            status_code=access_log.status_code,
            context_lines=context_lines,
        )
        error_code = _status_error_code(access_log.status_code)
        observations.append(
            {
                "observed_at": access_log.entry.observed_at,
                "environment": environment,
                "provider": provider,
                "model": _extract_model(detail, None),
                "model_group": None,
                "route_family": route_family,
                "status_code": access_log.status_code,
                "error_type": "http_status",
                "error_code": error_code,
                "error_class": _classify_provider_error(
                    status_code=access_log.status_code,
                    error_code=error_code,
                    error_type="http_status",
                    error_text=detail,
                ),
                "retry_after_seconds": None,
                "expected_reset_at": None,
                "session_id": None,
                "trace_id": None,
                "litellm_call_id": None,
                "metadata": _build_metadata(
                    container=container,
                    environment=environment,
                    source_entry=access_log.entry,
                    source_kind="access_log",
                    raw_excerpt=access_log.entry.clean_content,
                    normalized_error_text=detail,
                    request_id=None,
                    access_log=access_log,
                    context_lines=context_lines,
                    error_origin=error_origin,
                ),
            }
        )

    observations.sort(key=lambda row: (row["observed_at"], row["metadata"]["docker_line_index"]))
    return observations


def _build_metadata(
    *,
    container: str,
    environment: str,
    source_entry: LogEntry,
    source_kind: str,
    raw_excerpt: str,
    normalized_error_text: str,
    request_id: Optional[str],
    access_log: Optional[AccessLog],
    context_lines: Sequence[LogEntry],
    error_origin: str,
) -> Dict[str, Any]:
    metadata: Dict[str, Any] = {
        "source": "docker_log_backfill",
        "parser": "backfill_provider_error_observations_from_docker_logs",
        "parser_version": 1,
        "container": container,
        "environment": environment,
        "source_kind": source_kind,
        "error_origin": error_origin,
        "docker_timestamp": source_entry.timestamp_text,
        "docker_line_index": source_entry.line_index,
        "log_signature": _log_signature(
            container=container,
            timestamp_text=source_entry.timestamp_text,
            line_index=source_entry.line_index,
            content=source_entry.clean_content,
        ),
        "raw_log_excerpt": raw_excerpt[:1000],
        "normalized_error_text": normalized_error_text[:1000],
    }
    if request_id:
        metadata["request_id"] = request_id
    if access_log is not None:
        metadata.update(
            {
                "access_method": access_log.method,
                "access_path": access_log.path,
                "access_status_phrase": access_log.phrase,
                "access_docker_timestamp": access_log.entry.timestamp_text,
                "access_line_index": access_log.entry.line_index,
                "access_delta_seconds": round(
                    (access_log.entry.observed_at - source_entry.observed_at).total_seconds(),
                    6,
                ),
            }
        )
    if context_lines:
        metadata["nearby_error_lines"] = [
            {
                "docker_timestamp": entry.timestamp_text,
                "docker_line_index": entry.line_index,
                "text": entry.clean_content[:500],
            }
            for entry in context_lines[:5]
        ]
    return metadata


def _summarize(rows: Sequence[Dict[str, Any]]) -> List[Tuple[Any, ...]]:
    summary: Dict[Tuple[Any, ...], int] = {}
    for row in rows:
        key = (
            row.get("status_code"),
            row.get("error_class"),
            row.get("provider"),
            row.get("route_family"),
        )
        summary[key] = summary.get(key, 0) + 1
    return [(*key, count) for key, count in sorted(summary.items(), key=lambda item: item[0])]


async def _fetch_existing_signatures(conn: Any, *, container: str) -> Set[str]:
    rows = await conn.fetch(
        """
        SELECT metadata->>'log_signature' AS log_signature
        FROM public.provider_error_observations
        WHERE metadata->>'source' = 'docker_log_backfill'
          AND metadata->>'container' = $1
          AND metadata ? 'log_signature'
        """,
        container,
    )
    return {
        row["log_signature"]
        for row in rows
        if isinstance(row.get("log_signature"), str) and row["log_signature"]
    }


async def _run(args: argparse.Namespace) -> int:
    raw_log_file = Path(args.raw_log_file) if args.raw_log_file else None
    parsed_logs = _parse_logs(
        _iter_docker_log_lines(
            container=args.container,
            since=args.since,
            until=args.until,
            raw_log_file=raw_log_file,
        )
    )
    observations = _build_observations(
        parsed_logs=parsed_logs,
        container=args.container,
        environment=args.environment,
        max_correlation_seconds=args.max_correlation_seconds,
    )
    if args.limit is not None:
        observations = observations[: args.limit]

    dsn = args.dsn or _build_aawm_dsn()
    conn = await asyncpg.connect(dsn)
    try:
        if args.apply:
            await _ensure_session_history_schema(conn)
        existing_signatures = await _fetch_existing_signatures(conn, container=args.container)
        new_observations = [
            row
            for row in observations
            if row.get("metadata", {}).get("log_signature") not in existing_signatures
        ]
        print(f"container={args.container} environment={args.environment}")
        print(
            "parsed "
            f"lines={len(parsed_logs.entries)} "
            f"exception_logs={len(parsed_logs.exception_logs)} "
            f"access_4xx_5xx_logs={len(parsed_logs.access_logs)} "
            f"candidate_observations={len(observations)} "
            f"existing_backfill_signatures={len(existing_signatures)} "
            f"new_observations={len(new_observations)}"
        )
        print("summary status_code,error_class,provider,route_family,count")
        for status_code, error_class, provider, route_family, count in _summarize(new_observations):
            print(f"{status_code},{error_class},{provider},{route_family},{count}")
        if args.show_samples:
            print("samples")
            for row in new_observations[: args.show_samples]:
                metadata = row["metadata"]
                print(
                    json.dumps(
                        {
                            "observed_at": row["observed_at"].isoformat(),
                            "provider": row["provider"],
                            "model": row.get("model"),
                            "route_family": row.get("route_family"),
                            "status_code": row.get("status_code"),
                            "error_type": row.get("error_type"),
                            "error_code": row.get("error_code"),
                            "error_class": row["error_class"],
                            "source_kind": metadata.get("source_kind"),
                            "access_path": metadata.get("access_path"),
                            "raw_log_excerpt": metadata.get("raw_log_excerpt"),
                        },
                        sort_keys=True,
                    )
                )
        if args.apply and new_observations:
            await conn.executemany(
                _AAWM_PROVIDER_ERROR_OBSERVATION_INSERT_SQL,
                [
                    _build_provider_error_observation_db_payload(row)
                    for row in new_observations
                ],
            )
            print(f"inserted={len(new_observations)}")
        elif args.apply:
            print("inserted=0")
        else:
            print("dry_run=true inserted=0")
    finally:
        await conn.close()
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--container", required=True)
    parser.add_argument("--environment", required=True)
    parser.add_argument("--since")
    parser.add_argument("--until")
    parser.add_argument("--raw-log-file")
    parser.add_argument("--dsn")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--show-samples", type=int, default=3)
    parser.add_argument("--max-correlation-seconds", type=float, default=20.0)
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())
