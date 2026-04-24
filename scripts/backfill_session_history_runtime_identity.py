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
from typing import Any, Dict, Iterable, Optional
from urllib.error import HTTPError
from urllib.parse import quote, urlsplit, urlunsplit
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


def _clickhouse_auth_header(args: argparse.Namespace) -> str:
    user = args.clickhouse_user or os.getenv("CLICKHOUSE_USER") or os.getenv(
        "LANGFUSE_CLICKHOUSE_USER"
    ) or "clickhouse"
    password = args.clickhouse_password or os.getenv("CLICKHOUSE_PASSWORD") or os.getenv(
        "LANGFUSE_CLICKHOUSE_PASSWORD"
    ) or "clickhouse"
    return "Basic " + base64.b64encode(f"{user}:{password}".encode()).decode("ascii")


def _request_clickhouse_rows(args: argparse.Namespace, query: str) -> list[Dict[str, Any]]:
    base_url = _normalize_clickhouse_url(
        args.clickhouse_url
        or os.getenv("CLICKHOUSE_URL")
        or os.getenv("LANGFUSE_CLICKHOUSE_URL")
    )
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
            "Authorization": _clickhouse_auth_header(args),
            "Content-Type": "text/plain; charset=utf-8",
        },
        data=query.encode("utf-8"),
        method="POST",
    )
    try:
        with urlopen(request, timeout=args.clickhouse_timeout_seconds) as response:
            payload = response.read().decode("utf-8")
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"ClickHouse query failed with HTTP {exc.code}: {body}") from exc
    return [json.loads(line) for line in payload.splitlines() if line.strip()]


def _port_environment_from_requester_metadata(value: Any) -> tuple[Optional[str], Optional[str]]:
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


def _fetch_observation_identities(args: argparse.Namespace) -> list[ObservationIdentity]:
    query = """
SELECT
  o.id AS observation_id,
  o.trace_id AS trace_id,
  o.metadata AS observation_metadata,
  o.metadata['requester_metadata'] AS requester_metadata,
  t.environment AS trace_environment
FROM observations AS o
LEFT JOIN traces AS t ON t.id = o.trace_id AND t.is_deleted = 0
WHERE o.type = 'GENERATION'
  AND o.is_deleted = 0
  AND (
    length(t.environment) > 0 OR
    length(o.metadata['litellm_environment']) > 0 OR
    length(o.metadata['trace_environment']) > 0 OR
    length(o.metadata['source_trace_environment']) > 0 OR
    length(o.metadata['litellm_version']) > 0 OR
    length(o.metadata['litellm_fork_version']) > 0 OR
    length(o.metadata['litellm_wheel_versions']) > 0 OR
    length(o.metadata['client_name']) > 0 OR
    length(o.metadata['client_version']) > 0 OR
    length(o.metadata['client_user_agent']) > 0 OR
    length(o.metadata['user_agent']) > 0 OR
    length(o.metadata['http_user_agent']) > 0 OR
    length(o.metadata['cc_version']) > 0 OR
    length(o.metadata['cc_entrypoint']) > 0 OR
    length(o.metadata['anthropic_billing_header_fields']) > 0 OR
    length(o.metadata['requester_metadata']) > 0
  )
FORMAT JSONEachRow
"""
    identities: list[ObservationIdentity] = []
    seen: set[str] = set()
    for row in _request_clickhouse_rows(args, query):
        observation_id = _clean(row.get("observation_id"))
        if observation_id is None or observation_id in seen:
            continue
        seen.add(observation_id)

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
        if args.derive_environment_from_port and port_environment:
            if litellm_environment is None or (
                args.correct_default_environment_from_port
                and litellm_environment == "default"
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
            identities.append(item)
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


def _insert_temp_rows(
    cur: psycopg.Cursor,
    observations: list[ObservationIdentity],
    traces: list[TraceIdentity],
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
    cur.executemany(
        """
INSERT INTO tmp_session_history_runtime_identity_obs VALUES
(%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
""",
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
            for item in observations
        ],
    )
    cur.executemany(
        """
INSERT INTO tmp_session_history_runtime_identity_trace VALUES
(%s, %s, %s, %s, %s, %s, %s, %s, %s)
""",
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
            for item in traces
        ],
    )


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


def _dry_run(cur: psycopg.Cursor, *, correct_default_environment_from_port: bool) -> Dict[str, Any]:
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
    parser.add_argument("--apply", action="store_true", help="Write updates. Default is dry-run.")
    parser.add_argument(
        "--target-db-name",
        default="aawm_tristore",
        help="Database name that must be returned by current_database().",
    )
    parser.add_argument("--pg-dsn", default=None, help="PostgreSQL DSN override.")
    parser.add_argument("--pg-host", default=None, help="PostgreSQL host override.")
    parser.add_argument("--pg-port", default=None, help="PostgreSQL port override.")
    parser.add_argument("--pg-user", default=None, help="PostgreSQL user override.")
    parser.add_argument("--pg-password", default=None, help="PostgreSQL password override.")
    parser.add_argument("--clickhouse-url", default=None, help="ClickHouse HTTP URL.")
    parser.add_argument("--clickhouse-user", default=None, help="ClickHouse user.")
    parser.add_argument("--clickhouse-password", default=None, help="ClickHouse password.")
    parser.add_argument("--clickhouse-timeout-seconds", type=int, default=90)
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
    if args.correct_default_environment_from_port and not args.derive_environment_from_port:
        raise SystemExit(
            "--correct-default-environment-from-port requires --derive-environment-from-port"
        )

    observations = _fetch_observation_identities(args)
    traces = _derive_trace_identities(observations)
    dsn = _postgres_dsn_from_args(args)

    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute("select current_database()")
            database = cur.fetchone()[0]
            if database != args.target_db_name:
                raise SystemExit(
                    f"Refusing to run against {database!r}; expected {args.target_db_name!r}."
                )

            _insert_temp_rows(cur, observations, traces)
            dry_run = _dry_run(
                cur,
                correct_default_environment_from_port=args.correct_default_environment_from_port,
            )
            result: Dict[str, Any] = {
                "database": database,
                "mode": "apply" if args.apply else "dry-run",
                "observation_identity_rows": len(observations),
                "trace_identity_rows": len(traces),
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

    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
