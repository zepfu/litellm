#!/usr/bin/env python3
"""
Backfill session_history runtime/client identity fields from Langfuse ClickHouse.

This utility is intentionally data-only. It reads historical Langfuse
observations, derives the same runtime identity fields that
`AawmAgentIdentity` derives for live traffic, then updates missing
`public.session_history` columns in the AAWM/tristore database.

Default behavior is a dry run. Use `--apply` to write.

Typical local usage:

    ./.venv/bin/python scripts/backfill_session_history_runtime_identity.py

    ./.venv/bin/python scripts/backfill_session_history_runtime_identity.py \\
      --target-db-name aawm_tristore \\
      --apply

ClickHouse fetch is keyset-paginated (RR-072). Large histories should use
`--clickhouse-page-size` (default 1000) and can resume after a partial run with
`--clickhouse-resume-after-id <last_observation_id>`. Optional
`--clickhouse-max-pages` (positive integer) caps one invocation. PostgreSQL
temp-table loads use `--insert-batch-size` (default 1000). Auth is resolved
once in ``main()`` and threaded through preflight/fetch (not re-derived per
request layer). Each ClickHouse page is derived and inserted before the next
page is requested, so Python identity retention stays O(page size) rather than
O(total rows).

    ./.venv/bin/python scripts/backfill_session_history_runtime_identity.py \\
      --clickhouse-page-size 500 \\
      --clickhouse-resume-after-id obs-abc \\
      --insert-batch-size 500

Port-derived environment correction is opt-in:

    ./.venv/bin/python scripts/backfill_session_history_runtime_identity.py \\
      --apply \\
      --derive-environment-from-port \\
      --correct-default-environment-from-port

The target database guard exists because inherited environments have previously
pointed at `xx_aawm_dev`. The script aborts unless `current_database()` matches
`--target-db-name`.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence
from urllib.error import HTTPError
from urllib.parse import quote, urlsplit, urlunsplit

_CLICKHOUSE_URL_ENV_VARS = ("CLICKHOUSE_URL", "LANGFUSE_CLICKHOUSE_URL")
_CLICKHOUSE_USER_ENV_VARS = ("CLICKHOUSE_USER", "LANGFUSE_CLICKHOUSE_USER")
_CLICKHOUSE_PASSWORD_ENV_VARS = ("CLICKHOUSE_PASSWORD", "LANGFUSE_CLICKHOUSE_PASSWORD")

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
    _build_session_runtime_identity,
)


RUNTIME_IDENTITY_FIELDS = (
    "litellm_environment",
    "litellm_version",
    "litellm_fork_version",
    "litellm_wheel_versions",
    "client_name",
    "client_version",
    "client_user_agent",
)


@dataclass(frozen=True)
class ObservationIdentity:
    observation_id: str
    trace_id: Optional[str]
    litellm_environment: Optional[str]
    litellm_version: Optional[str]
    litellm_fork_version: Optional[str]
    litellm_wheel_versions: Optional[str]
    client_name: Optional[str]
    client_version: Optional[str]
    client_user_agent: Optional[str]
    port_environment: Optional[str]
    port_host: Optional[str]


@dataclass(frozen=True)
class TraceIdentity:
    trace_id: str
    litellm_environment: Optional[str]
    litellm_version: Optional[str]
    litellm_fork_version: Optional[str]
    litellm_wheel_versions: Optional[str]
    client_name: Optional[str]
    client_version: Optional[str]
    client_user_agent: Optional[str]
    port_environment: Optional[str]


def _clean(value: Any) -> Optional[str]:
    if value is None:
        return None
    if not isinstance(value, str):
        value = str(value)
    cleaned = value.strip()
    return cleaned or None


def _parse_clickhouse_value(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    if not stripped:
        return value
    try:
        return json.loads(stripped)
    except (TypeError, ValueError, json.JSONDecodeError):
        return value


def _coerce_json_object(value: Any) -> Optional[str]:
    if not isinstance(value, dict):
        return None
    cleaned = {
        str(key).strip(): str(item).strip()
        for key, item in value.items()
        if _clean(key) and _clean(item)
    }
    if not cleaned:
        return None
    return json.dumps(cleaned, sort_keys=True, separators=(",", ":"))


def _normalize_clickhouse_url(value: Optional[str]) -> str:
    cleaned = _clean(value) or "http://127.0.0.1:8123"
    normalized = cleaned.rstrip("/")
    if not normalized.startswith(("http://", "https://")):
        normalized = f"http://{normalized}"
    parsed = urlsplit(normalized)
    hostname = (parsed.hostname or "127.0.0.1").replace("clickhouse", "127.0.0.1")
    port = f":{parsed.port}" if parsed.port else ""
    netloc = f"{hostname}{port}"
    return urlunsplit(
        (parsed.scheme or "http", netloc, parsed.path, parsed.query, parsed.fragment)
    )


def _redact_clickhouse_url_userinfo(url: Optional[str]) -> Optional[str]:
    if url is None:
        return None
    parsed = urlsplit(url)
    if not parsed.scheme and not parsed.netloc:
        return url
    hostname = parsed.hostname or ""
    port = f":{parsed.port}" if parsed.port else ""
    userinfo = ""
    if parsed.username is not None:
        userinfo = parsed.username
        if parsed.password is not None:
            userinfo = f"{userinfo}:***"
        userinfo = f"{userinfo}@"
    redacted_netloc = f"{userinfo}{hostname}{port}"
    return urlunsplit(
        (parsed.scheme, redacted_netloc, parsed.path, parsed.query, parsed.fragment)
    )


def _build_clickhouse_auth_diagnostics(auth: Dict[str, str]) -> Dict[str, Any]:
    url_input = auth.get("url_input")
    return {
        "clickhouse_url_normalized": auth["url"],
        "clickhouse_url_raw": _redact_clickhouse_url_userinfo(url_input),
        "clickhouse_url_source_input": _redact_clickhouse_url_userinfo(url_input),
        "clickhouse_user": auth["user"],
        "url_source": auth["url_source"],
        "user_source": auth["user_source"],
        "password_source": auth["password_source"],
        "using_builtin_local_url_default": auth["url_source"]
        == "default:local_http_8123",
        "using_builtin_clickhouse_credentials": (
            auth["user_source"] == "default:clickhouse_builtin"
            and auth["password_source"] == "default:clickhouse_builtin"
        ),
    }


def _postgres_dsn_from_args(args: argparse.Namespace) -> str:
    if args.pg_dsn:
        return args.pg_dsn

    has_component_override = any(
        (args.pg_host, args.pg_port, args.pg_user, args.pg_password)
    )
    direct_dsn = os.getenv("AAWM_DIRECT_DATABASE_URL")
    if not has_component_override and direct_dsn and direct_dsn.strip():
        return direct_dsn.strip()

    host = args.pg_host or os.getenv("AAWM_DB_HOST") or "127.0.0.1"
    port = args.pg_port or os.getenv("AAWM_DB_PORT") or "5434"
    db_name = args.target_db_name
    user = args.pg_user or os.getenv("AAWM_DB_USER") or "aawm"
    password = args.pg_password or os.getenv("AAWM_DB_PASSWORD") or "aawm_dev"
    return (
        f"postgresql://{quote(user)}:{quote(password)}@"
        f"{host}:{port}/{quote(db_name)}"
    )


def _first_env_var(names: Sequence[str]) -> Optional[str]:
    for name in names:
        if _clean(os.getenv(name)):
            return name
    return None


def _resolve_clickhouse_auth_sources(args: argparse.Namespace) -> Dict[str, str]:
    url_arg = _clean(args.clickhouse_url)
    user_arg = _clean(args.clickhouse_user)
    password_arg = _clean(args.clickhouse_password)

    if url_arg:
        url_source = "cli:clickhouse-url"
        url_input: Optional[str] = url_arg
    else:
        url_env = _first_env_var(_CLICKHOUSE_URL_ENV_VARS)
        url_source = f"env:{url_env}" if url_env else "default:local_http_8123"
        url_input = _clean(os.getenv(url_env)) if url_env else None

    if user_arg:
        user_source = "cli:clickhouse-user"
        user = user_arg
    else:
        user_env = _first_env_var(_CLICKHOUSE_USER_ENV_VARS)
        if user_env:
            user_source = f"env:{user_env}"
            user = _clean(os.getenv(user_env)) or "clickhouse"
        else:
            user_source = "default:clickhouse_builtin"
            user = "clickhouse"

    if password_arg:
        password_source = "cli:clickhouse-password"
        password = password_arg
    else:
        password_env = _first_env_var(_CLICKHOUSE_PASSWORD_ENV_VARS)
        if password_env:
            password_source = f"env:{password_env}"
            password = _clean(os.getenv(password_env)) or "clickhouse"
        else:
            password_source = "default:clickhouse_builtin"
            password = "clickhouse"

    return {
        "url": _normalize_clickhouse_url(url_input),
        "url_input": url_input,
        "user": user,
        "password": password,
        "url_source": url_source,
        "user_source": user_source,
        "password_source": password_source,
    }


def _clickhouse_auth_diagnostics(auth: Dict[str, str]) -> Dict[str, Any]:
    return _build_clickhouse_auth_diagnostics(auth)


def _preflight_clickhouse_connection(
    auth: Dict[str, str], *, timeout_seconds: int
) -> None:
    header = _clickhouse_auth_header(auth)
    parsed = urlsplit(auth["url"])
    query_url = urlunsplit(
        (
            parsed.scheme,
            parsed.netloc,
            parsed.path or "/",
            "default_format=JSONEachRow",
            parsed.fragment,
        )
    )
    request = Request(
        query_url,
        headers={
            "Authorization": header,
            "Content-Type": "text/plain; charset=utf-8",
        },
        data=b"SELECT 1 AS ok FORMAT JSONEachRow",
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            response.read()
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"ClickHouse preflight failed with HTTP {exc.code}: {body}"
        ) from exc


def _clickhouse_auth_header(auth: Dict[str, str]) -> str:
    """Build Basic auth from a pre-resolved auth dict (RR-072: resolve once)."""
    return "Basic " + base64.b64encode(
        f"{auth['user']}:{auth['password']}".encode()
    ).decode("ascii")


def _request_clickhouse_rows(
    auth: Dict[str, str],
    query: str,
    *,
    timeout_seconds: int,
) -> list[Dict[str, Any]]:
    """POST a ClickHouse query and parse JSONEachRow lines.

    Auth must be resolved once by the caller and threaded in (RR-072).
    """
    base_url = auth["url"]
    parsed = urlsplit(base_url)
    query_url = urlunsplit(
        (
            parsed.scheme,
            parsed.netloc,
            parsed.path or "/",
            "default_format=JSONEachRow",
            parsed.fragment,
        )
    )
    request = Request(
        query_url,
        headers={
            "Authorization": _clickhouse_auth_header(auth),
            "Content-Type": "text/plain; charset=utf-8",
        },
        data=query.encode("utf-8"),
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            payload = response.read().decode("utf-8")
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"ClickHouse query failed with HTTP {exc.code}: {body}"
        ) from exc
    return [json.loads(line) for line in payload.splitlines() if line.strip()]


def _port_environment_from_requester_metadata(
    value: Any,
) -> tuple[Optional[str], Optional[str]]:
    parsed = _parse_clickhouse_value(value)
    if not isinstance(parsed, dict):
        return None, None
    headers = parsed.get("headers")
    if not isinstance(headers, dict):
        return None, None
    host = _clean(headers.get("host") or headers.get("Host"))
    if host is None:
        return None, None
    if host.endswith(":4000"):
        return "prod", host
    if host.endswith(":4001"):
        return "dev", host
    return None, host


def _sql_string_literal(value: str) -> str:
    """Escape a string for safe embedding as a ClickHouse single-quoted literal."""
    escaped = value.replace(chr(92), chr(92) * 2).replace(chr(39), chr(92) + chr(39))
    return chr(39) + escaped + chr(39)


def _build_observation_identity_page_query(
    *,
    limit: int,
    cursor_id: Optional[str] = None,
) -> str:
    """Build one paged GENERATION observation query (ORDER BY id + LIMIT).

    Keyset pagination on ``o.id`` keeps pages stable and resumable without
    OFFSET scan cost (RR-072).
    """
    if limit < 1:
        raise ValueError("clickhouse page limit must be >= 1")

    predicates = [
        "o.type = 'GENERATION'",
        "o.is_deleted = 0",
        "("
        "length(t.environment) > 0 OR "
        "length(o.metadata['litellm_environment']) > 0 OR "
        "length(o.metadata['trace_environment']) > 0 OR "
        "length(o.metadata['source_trace_environment']) > 0 OR "
        "length(o.metadata['litellm_version']) > 0 OR "
        "length(o.metadata['litellm_fork_version']) > 0 OR "
        "length(o.metadata['litellm_wheel_versions']) > 0 OR "
        "length(o.metadata['client_name']) > 0 OR "
        "length(o.metadata['client_version']) > 0 OR "
        "length(o.metadata['client_user_agent']) > 0 OR "
        "length(o.metadata['user_agent']) > 0 OR "
        "length(o.metadata['http_user_agent']) > 0 OR "
        "length(o.metadata['cc_version']) > 0 OR "
        "length(o.metadata['cc_entrypoint']) > 0 OR "
        "length(o.metadata['anthropic_billing_header_fields']) > 0 OR "
        "length(o.metadata['requester_metadata']) > 0"
        ")",
    ]
    if cursor_id:
        predicates.append(f"o.id > {_sql_string_literal(cursor_id)}")

    where_sql = " AND ".join(predicates)
    return f"""
SELECT
  o.id AS observation_id,
  o.trace_id AS trace_id,
  o.metadata AS observation_metadata,
  o.metadata['requester_metadata'] AS requester_metadata,
  t.environment AS trace_environment
FROM observations AS o
LEFT JOIN traces AS t ON t.id = o.trace_id AND t.is_deleted = 0
WHERE {where_sql}
ORDER BY o.id ASC
LIMIT {int(limit)}
FORMAT JSONEachRow
"""


def _observation_identity_from_row(
    row: Dict[str, Any],
    *,
    derive_environment_from_port: bool,
    correct_default_environment_from_port: bool,
) -> Optional[ObservationIdentity]:
    observation_id = _clean(row.get("observation_id"))
    if observation_id is None:
        return None

    metadata = {
        str(key): _parse_clickhouse_value(value)
        for key, value in (row.get("observation_metadata") or {}).items()
    }
    trace_environment = _clean(row.get("trace_environment"))
    if trace_environment and not metadata.get("source_trace_environment"):
        metadata["source_trace_environment"] = trace_environment

    identity = _build_session_runtime_identity(
        metadata=metadata,
        trace_environment=trace_environment,
        allow_runtime=False,
    )
    port_environment, port_host = _port_environment_from_requester_metadata(
        row.get("requester_metadata")
    )

    litellm_environment = _clean(identity.get("litellm_environment"))
    if derive_environment_from_port and port_environment:
        if litellm_environment is None or (
            correct_default_environment_from_port and litellm_environment == "default"
        ):
            litellm_environment = port_environment

    item = ObservationIdentity(
        observation_id=observation_id,
        trace_id=_clean(row.get("trace_id")),
        litellm_environment=litellm_environment,
        litellm_version=_clean(identity.get("litellm_version")),
        litellm_fork_version=_clean(identity.get("litellm_fork_version")),
        litellm_wheel_versions=_coerce_json_object(
            identity.get("litellm_wheel_versions")
        ),
        client_name=_clean(identity.get("client_name")),
        client_version=_clean(identity.get("client_version")),
        client_user_agent=_clean(identity.get("client_user_agent")),
        port_environment=port_environment,
        port_host=port_host,
    )
    if any(getattr(item, field) for field in RUNTIME_IDENTITY_FIELDS):
        return item
    return None


def _iter_observation_identity_pages(
    args: argparse.Namespace,
    auth: Dict[str, str],
) -> Iterable[list[ObservationIdentity]]:
    """Yield one page of derived observation identities at a time.

    Uses keyset pagination (``o.id > cursor`` + LIMIT) so large histories do not
    require one unbounded HTTP response (RR-072). Callers should consume and
    persist each page before requesting the next so peak Python retention stays
    O(page size). Cross-page de-duplication relies on the stable ``o.id`` keyset
    cursor; only page-local de-duplication is retained for malformed responses.
    """
    page_size = int(getattr(args, "clickhouse_page_size", 1000) or 1000)
    if page_size < 1:
        raise ValueError("clickhouse_page_size must be >= 1")
    max_pages = getattr(args, "clickhouse_max_pages", None)
    if max_pages is not None:
        max_pages = int(max_pages)
        if max_pages < 1:
            raise ValueError("clickhouse_max_pages must be >= 1")

    cursor_id: Optional[str] = _clean(getattr(args, "clickhouse_resume_after_id", None))
    pages_fetched = 0

    try:
        while True:
            if max_pages is not None and pages_fetched >= max_pages:
                break
            query = _build_observation_identity_page_query(
                limit=page_size,
                cursor_id=cursor_id,
            )
            rows = _request_clickhouse_rows(
                auth,
                query,
                timeout_seconds=args.clickhouse_timeout_seconds,
            )
            pages_fetched += 1
            if not rows:
                break

            page_last_id: Optional[str] = None
            page_seen: set[str] = set()
            page_identities: list[ObservationIdentity] = []
            for row in rows:
                observation_id = _clean(row.get("observation_id"))
                if observation_id is not None:
                    page_last_id = observation_id
                if observation_id is None or observation_id in page_seen:
                    continue
                page_seen.add(observation_id)
                item = _observation_identity_from_row(
                    row,
                    derive_environment_from_port=bool(
                        args.derive_environment_from_port
                    ),
                    correct_default_environment_from_port=bool(
                        args.correct_default_environment_from_port
                    ),
                )
                if item is not None:
                    page_identities.append(item)

            yield page_identities

            # Always record the last raw row id for resume/diagnostics, even when
            # the final page is short (RR-072).
            if page_last_id is None:
                break
            cursor_id = page_last_id
            if len(rows) < page_size:
                break
    finally:
        # Stash pagination stats even if the consumer stops early.
        args._clickhouse_pages_fetched = pages_fetched  # type: ignore[attr-defined]
        args._clickhouse_last_cursor_id = cursor_id  # type: ignore[attr-defined]


def _fetch_observation_identities(
    args: argparse.Namespace,
    auth: Dict[str, str],
) -> list[ObservationIdentity]:
    """Materialize all observation-identity pages (test/helper path only).

    Production ``main()`` streams pages via ``_iter_observation_identity_pages``
    so peak identity retention stays O(page size).
    """
    identities: list[ObservationIdentity] = []
    for page in _iter_observation_identity_pages(args, auth):
        identities.extend(page)
    return identities


def _derive_trace_identities(
    observations: Iterable[ObservationIdentity],
) -> list[TraceIdentity]:
    values_by_trace: dict[str, dict[str, set[str]]] = defaultdict(
        lambda: defaultdict(set)
    )
    for observation in observations:
        if not observation.trace_id:
            continue
        for field in (*RUNTIME_IDENTITY_FIELDS, "port_environment"):
            value = getattr(observation, field)
            if value:
                values_by_trace[observation.trace_id][field].add(value)

    traces: list[TraceIdentity] = []
    for trace_id, values_by_field in values_by_trace.items():
        kwargs: dict[str, Optional[str]] = {"trace_id": trace_id}
        for field in (*RUNTIME_IDENTITY_FIELDS, "port_environment"):
            values = values_by_field.get(field) or set()
            kwargs[field] = next(iter(values)) if len(values) == 1 else None
        if any(kwargs.get(field) for field in RUNTIME_IDENTITY_FIELDS):
            traces.append(TraceIdentity(**kwargs))  # type: ignore[arg-type]
    return traces


def _iter_batches(items: Sequence[Any], batch_size: int) -> Iterable[Sequence[Any]]:
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    for index in range(0, len(items), batch_size):
        yield items[index : index + batch_size]


def _create_temp_identity_tables(
    cur: psycopg.Cursor,
) -> None:
    cur.execute(
        """
CREATE TEMPORARY TABLE tmp_session_history_runtime_identity_obs (
    observation_id text primary key,
    trace_id text,
    litellm_environment text,
    litellm_version text,
    litellm_fork_version text,
    litellm_wheel_versions text,
    client_name text,
    client_version text,
    client_user_agent text,
    port_environment text,
    port_host text
) ON COMMIT DROP
"""
    )
    cur.execute(
        """
CREATE TEMPORARY TABLE tmp_session_history_runtime_identity_trace (
    trace_id text primary key,
    litellm_environment text,
    litellm_version text,
    litellm_fork_version text,
    litellm_wheel_versions text,
    client_name text,
    client_version text,
    client_user_agent text,
    port_environment text
) ON COMMIT DROP
"""
    )


def _insert_observation_identity_rows(
    cur: psycopg.Cursor,
    observations: list[ObservationIdentity],
    *,
    insert_batch_size: int = 1000,
) -> None:
    if not observations:
        return
    obs_sql = """
INSERT INTO tmp_session_history_runtime_identity_obs VALUES
(%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
"""
    for batch in _iter_batches(observations, insert_batch_size):
        cur.executemany(
            obs_sql,
            [
                (
                    item.observation_id,
                    item.trace_id,
                    item.litellm_environment,
                    item.litellm_version,
                    item.litellm_fork_version,
                    item.litellm_wheel_versions,
                    item.client_name,
                    item.client_version,
                    item.client_user_agent,
                    item.port_environment,
                    item.port_host,
                )
                for item in batch
            ],
        )


def _insert_trace_identity_rows(
    cur: psycopg.Cursor,
    traces: list[TraceIdentity],
    *,
    insert_batch_size: int = 1000,
) -> None:
    if not traces:
        return
    trace_sql = """
INSERT INTO tmp_session_history_runtime_identity_trace VALUES
(%s, %s, %s, %s, %s, %s, %s, %s, %s)
"""
    for batch in _iter_batches(traces, insert_batch_size):
        cur.executemany(
            trace_sql,
            [
                (
                    item.trace_id,
                    item.litellm_environment,
                    item.litellm_version,
                    item.litellm_fork_version,
                    item.litellm_wheel_versions,
                    item.client_name,
                    item.client_version,
                    item.client_user_agent,
                    item.port_environment,
                )
                for item in batch
            ],
        )


def _populate_trace_identities_from_obs_temp(cur: psycopg.Cursor) -> int:
    """Derive fill-safe per-trace identities from the temp observation table.

    Mirrors ``_derive_trace_identities``: keep a field only when all non-null
    observation values for that trace agree. Avoids retaining O(total rows) of
    observation identities in Python after pages have been loaded.
    """
    cur.execute(
        """
INSERT INTO tmp_session_history_runtime_identity_trace (
    trace_id,
    litellm_environment,
    litellm_version,
    litellm_fork_version,
    litellm_wheel_versions,
    client_name,
    client_version,
    client_user_agent,
    port_environment
)
SELECT
    o.trace_id,
    CASE
        WHEN count(DISTINCT o.litellm_environment)
            FILTER (WHERE o.litellm_environment IS NOT NULL) = 1
        THEN min(o.litellm_environment)
            FILTER (WHERE o.litellm_environment IS NOT NULL)
        ELSE NULL
    END AS litellm_environment,
    CASE
        WHEN count(DISTINCT o.litellm_version)
            FILTER (WHERE o.litellm_version IS NOT NULL) = 1
        THEN min(o.litellm_version) FILTER (WHERE o.litellm_version IS NOT NULL)
        ELSE NULL
    END AS litellm_version,
    CASE
        WHEN count(DISTINCT o.litellm_fork_version)
            FILTER (WHERE o.litellm_fork_version IS NOT NULL) = 1
        THEN min(o.litellm_fork_version)
            FILTER (WHERE o.litellm_fork_version IS NOT NULL)
        ELSE NULL
    END AS litellm_fork_version,
    CASE
        WHEN count(DISTINCT o.litellm_wheel_versions)
            FILTER (WHERE o.litellm_wheel_versions IS NOT NULL) = 1
        THEN min(o.litellm_wheel_versions)
            FILTER (WHERE o.litellm_wheel_versions IS NOT NULL)
        ELSE NULL
    END AS litellm_wheel_versions,
    CASE
        WHEN count(DISTINCT o.client_name)
            FILTER (WHERE o.client_name IS NOT NULL) = 1
        THEN min(o.client_name) FILTER (WHERE o.client_name IS NOT NULL)
        ELSE NULL
    END AS client_name,
    CASE
        WHEN count(DISTINCT o.client_version)
            FILTER (WHERE o.client_version IS NOT NULL) = 1
        THEN min(o.client_version) FILTER (WHERE o.client_version IS NOT NULL)
        ELSE NULL
    END AS client_version,
    CASE
        WHEN count(DISTINCT o.client_user_agent)
            FILTER (WHERE o.client_user_agent IS NOT NULL) = 1
        THEN min(o.client_user_agent)
            FILTER (WHERE o.client_user_agent IS NOT NULL)
        ELSE NULL
    END AS client_user_agent,
    CASE
        WHEN count(DISTINCT o.port_environment)
            FILTER (WHERE o.port_environment IS NOT NULL) = 1
        THEN min(o.port_environment)
            FILTER (WHERE o.port_environment IS NOT NULL)
        ELSE NULL
    END AS port_environment
FROM tmp_session_history_runtime_identity_obs AS o
WHERE o.trace_id IS NOT NULL
  AND o.trace_id <> ''
GROUP BY o.trace_id
HAVING
    (
        CASE
            WHEN count(DISTINCT o.litellm_environment)
                FILTER (WHERE o.litellm_environment IS NOT NULL) = 1
            THEN min(o.litellm_environment)
                FILTER (WHERE o.litellm_environment IS NOT NULL)
            ELSE NULL
        END
    ) IS NOT NULL
    OR (
        CASE
            WHEN count(DISTINCT o.litellm_version)
                FILTER (WHERE o.litellm_version IS NOT NULL) = 1
            THEN min(o.litellm_version)
                FILTER (WHERE o.litellm_version IS NOT NULL)
            ELSE NULL
        END
    ) IS NOT NULL
    OR (
        CASE
            WHEN count(DISTINCT o.litellm_fork_version)
                FILTER (WHERE o.litellm_fork_version IS NOT NULL) = 1
            THEN min(o.litellm_fork_version)
                FILTER (WHERE o.litellm_fork_version IS NOT NULL)
            ELSE NULL
        END
    ) IS NOT NULL
    OR (
        CASE
            WHEN count(DISTINCT o.litellm_wheel_versions)
                FILTER (WHERE o.litellm_wheel_versions IS NOT NULL) = 1
            THEN min(o.litellm_wheel_versions)
                FILTER (WHERE o.litellm_wheel_versions IS NOT NULL)
            ELSE NULL
        END
    ) IS NOT NULL
    OR (
        CASE
            WHEN count(DISTINCT o.client_name)
                FILTER (WHERE o.client_name IS NOT NULL) = 1
            THEN min(o.client_name) FILTER (WHERE o.client_name IS NOT NULL)
            ELSE NULL
        END
    ) IS NOT NULL
    OR (
        CASE
            WHEN count(DISTINCT o.client_version)
                FILTER (WHERE o.client_version IS NOT NULL) = 1
            THEN min(o.client_version)
                FILTER (WHERE o.client_version IS NOT NULL)
            ELSE NULL
        END
    ) IS NOT NULL
    OR (
        CASE
            WHEN count(DISTINCT o.client_user_agent)
                FILTER (WHERE o.client_user_agent IS NOT NULL) = 1
            THEN min(o.client_user_agent)
                FILTER (WHERE o.client_user_agent IS NOT NULL)
            ELSE NULL
        END
    ) IS NOT NULL
"""
    )
    cur.execute("SELECT count(*) FROM tmp_session_history_runtime_identity_trace")
    return int(cur.fetchone()[0])


def _insert_temp_rows(
    cur: psycopg.Cursor,
    observations: list[ObservationIdentity],
    traces: list[TraceIdentity],
    *,
    insert_batch_size: int = 1000,
) -> None:
    """Create temp tables and load pre-materialized identity lists (test helper)."""
    _create_temp_identity_tables(cur)
    _insert_observation_identity_rows(
        cur, observations, insert_batch_size=insert_batch_size
    )
    _insert_trace_identity_rows(cur, traces, insert_batch_size=insert_batch_size)


def _matched_cte(correct_default_environment_from_port: bool) -> str:
    default_env_clause = (
        " OR (coalesce(litellm_environment, '') = 'default' "
        "AND n_port_environment IS NOT NULL)"
        if correct_default_environment_from_port
        else ""
    )
    return f"""
WITH matched AS (
  SELECT
    sh.id,
    coalesce(src.litellm_environment, callsrc.litellm_environment, tracesrc.litellm_environment) AS n_litellm_environment,
    coalesce(src.litellm_version, callsrc.litellm_version, tracesrc.litellm_version) AS n_litellm_version,
    coalesce(src.litellm_fork_version, callsrc.litellm_fork_version, tracesrc.litellm_fork_version) AS n_litellm_fork_version,
    coalesce(src.litellm_wheel_versions, callsrc.litellm_wheel_versions, tracesrc.litellm_wheel_versions) AS n_litellm_wheel_versions,
    coalesce(src.client_name, callsrc.client_name, tracesrc.client_name) AS n_client_name,
    coalesce(src.client_version, callsrc.client_version, tracesrc.client_version) AS n_client_version,
    coalesce(src.client_user_agent, callsrc.client_user_agent, tracesrc.client_user_agent) AS n_client_user_agent,
    coalesce(src.port_environment, callsrc.port_environment, tracesrc.port_environment) AS n_port_environment,
    sh.litellm_environment,
    sh.litellm_version,
    sh.litellm_fork_version,
    sh.litellm_wheel_versions,
    sh.client_name,
    sh.client_version,
    sh.client_user_agent
  FROM public.session_history sh
  LEFT JOIN tmp_session_history_runtime_identity_obs src
    ON src.observation_id = sh.metadata->>'source_observation_id'
  LEFT JOIN tmp_session_history_runtime_identity_obs callsrc
    ON callsrc.observation_id = sh.litellm_call_id
  LEFT JOIN tmp_session_history_runtime_identity_trace tracesrc
    ON tracesrc.trace_id = sh.trace_id
  WHERE src.observation_id IS NOT NULL
     OR callsrc.observation_id IS NOT NULL
     OR tracesrc.trace_id IS NOT NULL
), changes AS (
  SELECT
    *,
    (coalesce(litellm_environment, '') = '' AND n_litellm_environment IS NOT NULL{default_env_clause}) AS change_litellm_environment,
    (coalesce(litellm_version, '') = '' AND n_litellm_version IS NOT NULL) AS change_litellm_version,
    (coalesce(litellm_fork_version, '') = '' AND n_litellm_fork_version IS NOT NULL) AS change_litellm_fork_version,
    ((litellm_wheel_versions IS NULL OR litellm_wheel_versions = '{{}}'::jsonb) AND n_litellm_wheel_versions IS NOT NULL) AS change_litellm_wheel_versions,
    (coalesce(client_name, '') = '' AND n_client_name IS NOT NULL) AS change_client_name,
    (coalesce(client_version, '') = '' AND n_client_version IS NOT NULL) AS change_client_version,
    (coalesce(client_user_agent, '') = '' AND n_client_user_agent IS NOT NULL) AS change_client_user_agent
  FROM matched
)
"""


def _dry_run(
    cur: psycopg.Cursor, *, correct_default_environment_from_port: bool
) -> Dict[str, Any]:
    cur.execute(
        _matched_cte(correct_default_environment_from_port)
        + """
SELECT
  count(*) AS matched_rows,
  count(*) FILTER (WHERE change_litellm_environment) AS litellm_environment,
  count(*) FILTER (WHERE change_litellm_version) AS litellm_version,
  count(*) FILTER (WHERE change_litellm_fork_version) AS litellm_fork_version,
  count(*) FILTER (WHERE change_litellm_wheel_versions) AS litellm_wheel_versions,
  count(*) FILTER (WHERE change_client_name) AS client_name,
  count(*) FILTER (WHERE change_client_version) AS client_version,
  count(*) FILTER (WHERE change_client_user_agent) AS client_user_agent
FROM changes
"""
    )
    columns = [description.name for description in cur.description]
    return dict(zip(columns, cur.fetchone()))


def _apply(cur: psycopg.Cursor, *, correct_default_environment_from_port: bool) -> int:
    cur.execute(
        _matched_cte(correct_default_environment_from_port)
        + """
, updated AS (
  UPDATE public.session_history sh
  SET
    litellm_environment = CASE WHEN changes.change_litellm_environment THEN changes.n_litellm_environment ELSE sh.litellm_environment END,
    litellm_version = CASE WHEN changes.change_litellm_version THEN changes.n_litellm_version ELSE sh.litellm_version END,
    litellm_fork_version = CASE WHEN changes.change_litellm_fork_version THEN changes.n_litellm_fork_version ELSE sh.litellm_fork_version END,
    litellm_wheel_versions = CASE WHEN changes.change_litellm_wheel_versions THEN changes.n_litellm_wheel_versions::jsonb ELSE sh.litellm_wheel_versions END,
    client_name = CASE WHEN changes.change_client_name THEN changes.n_client_name ELSE sh.client_name END,
    client_version = CASE WHEN changes.change_client_version THEN changes.n_client_version ELSE sh.client_version END,
    client_user_agent = CASE WHEN changes.change_client_user_agent THEN changes.n_client_user_agent ELSE sh.client_user_agent END,
    metadata = coalesce(sh.metadata, '{}'::jsonb) || jsonb_strip_nulls(jsonb_build_object(
      'litellm_environment', CASE WHEN changes.change_litellm_environment THEN to_jsonb(changes.n_litellm_environment) ELSE NULL END,
      'litellm_version', CASE WHEN changes.change_litellm_version THEN to_jsonb(changes.n_litellm_version) ELSE NULL END,
      'litellm_fork_version', CASE WHEN changes.change_litellm_fork_version THEN to_jsonb(changes.n_litellm_fork_version) ELSE NULL END,
      'litellm_wheel_versions', CASE WHEN changes.change_litellm_wheel_versions THEN changes.n_litellm_wheel_versions::jsonb ELSE NULL END,
      'client_name', CASE WHEN changes.change_client_name THEN to_jsonb(changes.n_client_name) ELSE NULL END,
      'client_version', CASE WHEN changes.change_client_version THEN to_jsonb(changes.n_client_version) ELSE NULL END,
      'client_user_agent', CASE WHEN changes.change_client_user_agent THEN to_jsonb(changes.n_client_user_agent) ELSE NULL END
    ))
  FROM changes
  WHERE sh.id = changes.id
    AND (
      changes.change_litellm_environment OR
      changes.change_litellm_version OR
      changes.change_litellm_fork_version OR
      changes.change_litellm_wheel_versions OR
      changes.change_client_name OR
      changes.change_client_version OR
      changes.change_client_user_agent
    )
  RETURNING sh.id
)
SELECT count(*) FROM updated
"""
    )
    return int(cur.fetchone()[0])


def _final_counts(cur: psycopg.Cursor) -> Dict[str, Any]:
    cur.execute(
        """
SELECT
  count(*) AS rows,
  count(*) FILTER (WHERE litellm_environment IS NOT NULL AND litellm_environment <> '') AS litellm_environment,
  count(*) FILTER (WHERE litellm_version IS NOT NULL AND litellm_version <> '') AS litellm_version,
  count(*) FILTER (WHERE litellm_fork_version IS NOT NULL AND litellm_fork_version <> '') AS litellm_fork_version,
  count(*) FILTER (WHERE litellm_wheel_versions IS NOT NULL AND litellm_wheel_versions <> '{}'::jsonb) AS litellm_wheel_versions,
  count(*) FILTER (WHERE client_name IS NOT NULL AND client_name <> '') AS client_name,
  count(*) FILTER (WHERE client_version IS NOT NULL AND client_version <> '') AS client_version,
  count(*) FILTER (WHERE client_user_agent IS NOT NULL AND client_user_agent <> '') AS client_user_agent
FROM public.session_history
"""
    )
    columns = [description.name for description in cur.description]
    return dict(zip(columns, cur.fetchone()))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Backfill public.session_history litellm*/client* columns from "
            "Langfuse ClickHouse observations using AawmAgentIdentity rules."
        )
    )
    parser.add_argument(
        "--apply", action="store_true", help="Write updates. Default is dry-run."
    )
    parser.add_argument(
        "--target-db-name",
        default="aawm_tristore",
        help="Database name that must be returned by current_database().",
    )
    parser.add_argument("--pg-dsn", default=None, help="PostgreSQL DSN override.")
    parser.add_argument("--pg-host", default=None, help="PostgreSQL host override.")
    parser.add_argument("--pg-port", default=None, help="PostgreSQL port override.")
    parser.add_argument("--pg-user", default=None, help="PostgreSQL user override.")
    parser.add_argument(
        "--pg-password", default=None, help="PostgreSQL password override."
    )
    parser.add_argument("--clickhouse-url", default=None, help="ClickHouse HTTP URL.")
    parser.add_argument("--clickhouse-user", default=None, help="ClickHouse user.")
    parser.add_argument(
        "--clickhouse-password", default=None, help="ClickHouse password."
    )
    parser.add_argument("--clickhouse-timeout-seconds", type=int, default=90)
    parser.add_argument(
        "--clickhouse-page-size",
        type=int,
        default=1000,
        help=(
            "ClickHouse page size (LIMIT) for observation identity fetch. "
            "Pages use keyset pagination on observation id (RR-072)."
        ),
    )
    parser.add_argument(
        "--clickhouse-max-pages",
        type=int,
        default=None,
        help=(
            "Optional safety cap on ClickHouse pages fetched in one run "
            "(positive integer). Omit to read until the table is exhausted."
        ),
    )
    parser.add_argument(
        "--clickhouse-resume-after-id",
        default=None,
        help=(
            "Resume ClickHouse keyset pagination after this observation id "
            "(exclusive). Useful after a partial/timeout run."
        ),
    )
    parser.add_argument(
        "--insert-batch-size",
        type=int,
        default=1000,
        help="PostgreSQL executemany batch size when loading temp identity tables.",
    )
    parser.add_argument(
        "--derive-environment-from-port",
        action="store_true",
        help="Map requester_metadata.headers.host :4000 to prod and :4001 to dev.",
    )
    parser.add_argument(
        "--correct-default-environment-from-port",
        action="store_true",
        help=(
            "When port derivation is enabled, allow port prod/dev to replace "
            "an existing litellm_environment='default'."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if (
        args.correct_default_environment_from_port
        and not args.derive_environment_from_port
    ):
        raise SystemExit(
            "--correct-default-environment-from-port requires --derive-environment-from-port"
        )

    # Resolve ClickHouse auth once and thread it through all CH call sites (RR-072).
    clickhouse_auth = _resolve_clickhouse_auth_sources(args)
    _preflight_clickhouse_connection(
        clickhouse_auth,
        timeout_seconds=args.clickhouse_timeout_seconds,
    )

    if int(args.clickhouse_page_size) < 1:
        raise SystemExit("--clickhouse-page-size must be >= 1")
    if int(args.insert_batch_size) < 1:
        raise SystemExit("--insert-batch-size must be >= 1")
    if args.clickhouse_max_pages is not None and int(args.clickhouse_max_pages) < 1:
        raise SystemExit("--clickhouse-max-pages must be >= 1")

    dsn = _postgres_dsn_from_args(args)

    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute("select current_database()")
            database = cur.fetchone()[0]
            if database != args.target_db_name:
                raise SystemExit(
                    f"Refusing to run against {database!r}; expected {args.target_db_name!r}."
                )

            _create_temp_identity_tables(cur)
            observation_identity_rows = 0
            for page in _iter_observation_identity_pages(args, clickhouse_auth):
                _insert_observation_identity_rows(
                    cur,
                    page,
                    insert_batch_size=int(args.insert_batch_size),
                )
                observation_identity_rows += len(page)
            trace_identity_rows = _populate_trace_identities_from_obs_temp(cur)
            dry_run = _dry_run(
                cur,
                correct_default_environment_from_port=args.correct_default_environment_from_port,
            )
            result: Dict[str, Any] = {
                "database": database,
                "mode": "apply" if args.apply else "dry-run",
                "clickhouse_auth": _clickhouse_auth_diagnostics(clickhouse_auth),
                "observation_identity_rows": observation_identity_rows,
                "trace_identity_rows": trace_identity_rows,
                "clickhouse_page_size": int(args.clickhouse_page_size),
                "clickhouse_pages_fetched": int(
                    getattr(args, "_clickhouse_pages_fetched", 0) or 0
                ),
                "clickhouse_last_cursor_id": getattr(
                    args, "_clickhouse_last_cursor_id", None
                ),
                "insert_batch_size": int(args.insert_batch_size),
                "would_fill": dry_run,
            }
            if args.apply:
                result["updated_rows"] = _apply(
                    cur,
                    correct_default_environment_from_port=args.correct_default_environment_from_port,
                )
                result["final_counts"] = _final_counts(cur)
            else:
                conn.rollback()

    print(json.dumps(result, indent=2, sort_keys=True))  # noqa: T201
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
