#!/usr/bin/env python3
"""Run provider status observations on a fixed container-friendly cadence."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import re
import signal
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Sequence
from urllib.parse import urlsplit

try:
    from scripts import record_provider_status_observations as probes
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    probes = importlib.import_module("record_provider_status_observations")

try:
    from scripts import grok_oidc_refresh
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    grok_oidc_refresh = importlib.import_module("grok_oidc_refresh")


TRUE_VALUES = {"1", "true", "yes", "on"}
FALSE_VALUES = {"0", "false", "no", "off"}
PROVIDER_FAILURE_SUMMARY_LIMIT = 8
PROVIDER_FAILURE_FIELD_LIMIT = 160
PROVIDER_FAILURE_MESSAGE_LIMIT = 240
DEFAULT_GROK_OIDC_AUTH_FILE = "/home/zepfu/.grok/auth.json"
DEFAULT_GROK_OIDC_LOCK_FILE = "/home/zepfu/.grok/auth.json.lock"
DEFAULT_GROK_OIDC_REFRESH_INTERVAL_SECONDS = 3600.0
DEFAULT_GROK_OIDC_HTTP_TIMEOUT_SECONDS = 30.0
PROVIDER_FAILURE_SECRET_RE = re.compile(
    "|".join(
        (
            r"Bearer\s+[A-Za-z0-9\-._~+/]{10,}=*",
            r"Basic\s+[A-Za-z0-9+/]{10,}={0,2}",
            r"sk-[A-Za-z0-9\-_]{20,}",
            r"(?:api[_-]?key|x-api-key|api-key|token|password|passwd|secret)"
            r"['\"]?\s*[:=]\s*['\"]?[^\s,'\"})\]{}>]+",
            r"(?<=://)[^\s'\"]*:[^\s'\"@]+(?=@)",
            r"\b(?:\d{1,3}\.){3}\d{1,3}\b",
            r"\b(?:[0-9A-Fa-f]{1,4}:){2,}[0-9A-Fa-f:.]{1,}\b",
        )
    ),
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ProviderStatusLoopConfig:
    apply: bool
    dsn: Optional[str]
    environment: str
    interval_seconds: float
    timeout: float
    ping_count: int
    ping_timeout: int
    skip_icmp: bool
    once: bool
    setup_schema: bool
    db_lock_timeout_ms: int
    db_statement_timeout_ms: int
    schema_dsn: Optional[str] = None
    require_pgbouncer: bool = False
    grok_oidc_refresh_enabled: bool = False
    grok_oidc_auth_file: str = DEFAULT_GROK_OIDC_AUTH_FILE
    grok_oidc_lock_file: str = DEFAULT_GROK_OIDC_LOCK_FILE
    grok_oidc_refresh_interval_seconds: float = (
        DEFAULT_GROK_OIDC_REFRESH_INTERVAL_SECONDS
    )
    grok_oidc_refresh_buffer_seconds: int = (
        grok_oidc_refresh.DEFAULT_GROK_OIDC_REFRESH_BUFFER_SECONDS
    )
    grok_oidc_force_refresh: bool = False
    grok_oidc_http_timeout_seconds: float = DEFAULT_GROK_OIDC_HTTP_TIMEOUT_SECONDS


@dataclass
class SidecarTaskState:
    grok_oidc_last_attempt_monotonic: Optional[float] = None


def _env_bool(name: str, default: bool) -> bool:
    raw_value = os.getenv(name)
    if raw_value is None or raw_value == "":
        return default

    value = raw_value.strip().lower()
    if value in TRUE_VALUES:
        return True
    if value in FALSE_VALUES:
        return False
    raise ValueError(f"{name} must be one of {sorted(TRUE_VALUES | FALSE_VALUES)}")


def _env_float(name: str, default: float) -> float:
    raw_value = os.getenv(name)
    if raw_value is None or raw_value == "":
        return default
    return float(raw_value)


def _env_int(name: str, default: int) -> int:
    raw_value = os.getenv(name)
    if raw_value is None or raw_value == "":
        return default
    return int(raw_value)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--apply",
        dest="apply",
        action="store_true",
        default=_env_bool("AAWM_PROVIDER_STATUS_APPLY", True),
        help="Insert observations into Postgres. Defaults to AAWM_PROVIDER_STATUS_APPLY or true.",
    )
    parser.add_argument(
        "--dry-run",
        dest="apply",
        action="store_false",
        help="Collect and log a summary without inserting observations.",
    )
    parser.add_argument("--dsn", default=os.getenv("AAWM_PROVIDER_STATUS_DSN"))
    parser.add_argument(
        "--schema-dsn",
        default=(
            os.getenv("AAWM_PROVIDER_STATUS_SCHEMA_DSN")
            or os.getenv("AAWM_DIRECT_DATABASE_URL")
        ),
        help=(
            "Direct Postgres DSN for explicit provider-status schema setup. "
            "Defaults to AAWM_PROVIDER_STATUS_SCHEMA_DSN or AAWM_DIRECT_DATABASE_URL."
        ),
    )
    parser.add_argument(
        "--environment",
        default=os.getenv("AAWM_LITELLM_ENVIRONMENT", "dev"),
    )
    parser.add_argument(
        "--interval-seconds",
        type=float,
        default=_env_float("AAWM_PROVIDER_STATUS_INTERVAL_SECONDS", 300.0),
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=_env_float("AAWM_PROVIDER_STATUS_TIMEOUT", 2.0),
    )
    parser.add_argument(
        "--ping-count",
        type=int,
        default=_env_int("AAWM_PROVIDER_STATUS_PING_COUNT", 1),
    )
    parser.add_argument(
        "--ping-timeout",
        type=int,
        default=_env_int("AAWM_PROVIDER_STATUS_PING_TIMEOUT", 2),
    )
    parser.add_argument(
        "--skip-icmp",
        action="store_true",
        default=_env_bool("AAWM_PROVIDER_STATUS_SKIP_ICMP", False),
    )
    parser.add_argument(
        "--once",
        action="store_true",
        default=_env_bool("AAWM_PROVIDER_STATUS_ONCE", False),
        help="Run exactly one cycle, then exit.",
    )
    schema_group = parser.add_mutually_exclusive_group()
    schema_group.add_argument(
        "--setup-schema",
        dest="setup_schema",
        action="store_true",
        default=_env_bool("AAWM_PROVIDER_STATUS_SETUP_SCHEMA_ON_START", False),
        help=(
            "Run provider-status schema setup once before the loop starts. "
            "Defaults to AAWM_PROVIDER_STATUS_SETUP_SCHEMA_ON_START or false."
        ),
    )
    schema_group.add_argument(
        "--no-setup-schema",
        dest="setup_schema",
        action="store_false",
        help="Skip startup schema setup; steady-state cycles never run DDL.",
    )
    parser.add_argument(
        "--db-lock-timeout-ms",
        type=int,
        default=_env_int(
            "AAWM_PROVIDER_STATUS_DB_LOCK_TIMEOUT_MS",
            probes.DEFAULT_DB_LOCK_TIMEOUT_MS,
        ),
    )
    parser.add_argument(
        "--db-statement-timeout-ms",
        type=int,
        default=_env_int(
            "AAWM_PROVIDER_STATUS_DB_STATEMENT_TIMEOUT_MS",
            probes.DEFAULT_DB_STATEMENT_TIMEOUT_MS,
        ),
    )
    parser.add_argument(
        "--require-pgbouncer",
        action="store_true",
        default=_env_bool("AAWM_PROVIDER_STATUS_REQUIRE_PGBOUNCER", False),
        help=(
            "Fail startup if steady-state writes do not resolve to the "
            "PgBouncer transaction pool."
        ),
    )
    grok_group = parser.add_mutually_exclusive_group()
    grok_group.add_argument(
        "--grok-oidc-refresh-enabled",
        dest="grok_oidc_refresh_enabled",
        action="store_true",
        default=_env_bool("AAWM_GROK_OIDC_REFRESH_ENABLED", False),
        help=(
            "Run the Grok OIDC auth-file refresh task from this sidecar loop. "
            "Defaults to AAWM_GROK_OIDC_REFRESH_ENABLED or false."
        ),
    )
    grok_group.add_argument(
        "--no-grok-oidc-refresh",
        dest="grok_oidc_refresh_enabled",
        action="store_false",
        help="Disable the Grok OIDC auth-file refresh task.",
    )
    parser.add_argument(
        "--grok-oidc-auth-file",
        default=os.getenv("AAWM_GROK_OIDC_AUTH_FILE", DEFAULT_GROK_OIDC_AUTH_FILE),
        help=(
            "Grok CLI auth JSON file maintained by this sidecar. Defaults to "
            "AAWM_GROK_OIDC_AUTH_FILE or /home/zepfu/.grok/auth.json."
        ),
    )
    parser.add_argument(
        "--grok-oidc-lock-file",
        default=os.getenv("AAWM_GROK_OIDC_LOCK_FILE", DEFAULT_GROK_OIDC_LOCK_FILE),
        help=(
            "Lock file for sidecar Grok OIDC refresh writes. Defaults to "
            "AAWM_GROK_OIDC_LOCK_FILE or /home/zepfu/.grok/auth.json.lock."
        ),
    )
    parser.add_argument(
        "--grok-oidc-refresh-interval-seconds",
        type=float,
        default=_env_float(
            "AAWM_GROK_OIDC_REFRESH_INTERVAL_SECONDS",
            DEFAULT_GROK_OIDC_REFRESH_INTERVAL_SECONDS,
        ),
        help=(
            "Minimum seconds between Grok OIDC refresh attempts. Defaults to "
            "AAWM_GROK_OIDC_REFRESH_INTERVAL_SECONDS or 3600."
        ),
    )
    parser.add_argument(
        "--grok-oidc-refresh-buffer-seconds",
        type=int,
        default=_env_int(
            "AAWM_GROK_OIDC_REFRESH_BUFFER_SECONDS",
            grok_oidc_refresh.DEFAULT_GROK_OIDC_REFRESH_BUFFER_SECONDS,
        ),
        help=(
            "Refresh buffer for non-forced Grok OIDC refreshes. Defaults to "
            "AAWM_GROK_OIDC_REFRESH_BUFFER_SECONDS or 300."
        ),
    )
    parser.add_argument(
        "--grok-oidc-force-refresh",
        action="store_true",
        default=_env_bool("AAWM_GROK_OIDC_FORCE_REFRESH", False),
        help=(
            "Refresh the Grok OIDC credential on every scheduled attempt even "
            "when the current access token is still valid."
        ),
    )
    parser.add_argument(
        "--grok-oidc-http-timeout-seconds",
        type=float,
        default=_env_float(
            "AAWM_GROK_OIDC_HTTP_TIMEOUT_SECONDS",
            DEFAULT_GROK_OIDC_HTTP_TIMEOUT_SECONDS,
        ),
        help=(
            "HTTP timeout for Grok OIDC token endpoint calls. Defaults to "
            "AAWM_GROK_OIDC_HTTP_TIMEOUT_SECONDS or 30."
        ),
    )
    return parser


def parse_config(argv: Optional[Sequence[str]] = None) -> ProviderStatusLoopConfig:
    args = _build_parser().parse_args(argv)
    if args.interval_seconds <= 0:
        raise SystemExit("--interval-seconds must be greater than 0")
    if args.timeout <= 0:
        raise SystemExit("--timeout must be greater than 0")
    if args.ping_count <= 0:
        raise SystemExit("--ping-count must be greater than 0")
    if args.ping_timeout <= 0:
        raise SystemExit("--ping-timeout must be greater than 0")
    if args.db_lock_timeout_ms <= 0:
        raise SystemExit("--db-lock-timeout-ms must be greater than 0")
    if args.db_statement_timeout_ms <= 0:
        raise SystemExit("--db-statement-timeout-ms must be greater than 0")
    if args.grok_oidc_refresh_interval_seconds <= 0:
        raise SystemExit("--grok-oidc-refresh-interval-seconds must be greater than 0")
    if args.grok_oidc_refresh_buffer_seconds < 0:
        raise SystemExit("--grok-oidc-refresh-buffer-seconds must be non-negative")
    if args.grok_oidc_http_timeout_seconds <= 0:
        raise SystemExit("--grok-oidc-http-timeout-seconds must be greater than 0")

    return ProviderStatusLoopConfig(
        apply=args.apply,
        dsn=args.dsn,
        environment=args.environment,
        interval_seconds=args.interval_seconds,
        timeout=args.timeout,
        ping_count=args.ping_count,
        ping_timeout=args.ping_timeout,
        skip_icmp=args.skip_icmp,
        once=args.once,
        setup_schema=args.setup_schema,
        db_lock_timeout_ms=args.db_lock_timeout_ms,
        db_statement_timeout_ms=args.db_statement_timeout_ms,
        schema_dsn=args.schema_dsn,
        require_pgbouncer=args.require_pgbouncer,
        grok_oidc_refresh_enabled=args.grok_oidc_refresh_enabled,
        grok_oidc_auth_file=args.grok_oidc_auth_file,
        grok_oidc_lock_file=args.grok_oidc_lock_file,
        grok_oidc_refresh_interval_seconds=args.grok_oidc_refresh_interval_seconds,
        grok_oidc_refresh_buffer_seconds=args.grok_oidc_refresh_buffer_seconds,
        grok_oidc_force_refresh=args.grok_oidc_force_refresh,
        grok_oidc_http_timeout_seconds=args.grok_oidc_http_timeout_seconds,
    )


def _dsn_args(config: ProviderStatusLoopConfig) -> argparse.Namespace:
    return argparse.Namespace(
        dsn=config.dsn,
        pg_host=None,
        pg_port=None,
        pg_database=None,
        pg_user=None,
        pg_password=None,
        pg_sslmode=None,
    )


def _resolve_dsn(config: ProviderStatusLoopConfig) -> str:
    dsn = probes._build_dsn(_dsn_args(config))
    if not dsn:
        raise RuntimeError("No database DSN found. Set AAWM_DB_* or AAWM_PROVIDER_STATUS_DSN.")
    return dsn


def _resolve_schema_dsn(config: ProviderStatusLoopConfig) -> str:
    if config.schema_dsn:
        return probes._append_dsn_query_params(
            config.schema_dsn,
            {"application_name": probes._provider_status_db_application_name()},
        )
    if config.dsn:
        return probes._append_dsn_query_params(
            config.dsn,
            {"application_name": probes._provider_status_db_application_name()},
        )
    raise RuntimeError(
        "Provider-status schema setup requires AAWM_PROVIDER_STATUS_SCHEMA_DSN, "
        "AAWM_DIRECT_DATABASE_URL, or explicit --dsn. Steady-state PgBouncer "
        "DSNs are not used for schema setup by default."
    )


def _dsn_targets_pgbouncer(dsn: str) -> bool:
    try:
        parsed = urlsplit(dsn)
    except ValueError:
        return False
    return parsed.hostname == "pgbouncer" and parsed.port == 6432


def validate_runtime_guardrails(config: ProviderStatusLoopConfig) -> None:
    if config.setup_schema:
        _resolve_schema_dsn(config)
    if not config.apply or not config.require_pgbouncer:
        return
    dsn = _resolve_dsn(config)
    if not _dsn_targets_pgbouncer(dsn):
        raise RuntimeError(
            "Provider-status steady-state writes require "
            "pgbouncer:6432 when AAWM_PROVIDER_STATUS_REQUIRE_PGBOUNCER=1"
        )


def setup_schema_once(config: ProviderStatusLoopConfig) -> Dict[str, Any]:
    started = time.perf_counter()
    dsn = _resolve_schema_dsn(config)
    try:
        probes.setup_schema(
            dsn,
            lock_timeout_ms=config.db_lock_timeout_ms,
            statement_timeout_ms=config.db_statement_timeout_ms,
        )
    except probes.ProviderStatusDatabaseWriteSkipped as exc:
        return {
            "event": "provider_status_observations_schema_skipped",
            "observed_at": _utc_timestamp(),
            "environment": config.environment,
            "error_class": exc.error_class,
            "error_message": str(exc),
            "duration_ms": round((time.perf_counter() - started) * 1000, 3),
        }
    return {
        "event": "provider_status_observations_schema_ready",
        "observed_at": _utc_timestamp(),
        "environment": config.environment,
        "duration_ms": round((time.perf_counter() - started) * 1000, 3),
    }


def _bounded_summary_field(value: Any, *, limit: int = PROVIDER_FAILURE_FIELD_LIMIT) -> Optional[str]:
    if value is None:
        return None
    text = re.sub(r"\s+", " ", str(value)).strip()
    if not text:
        return None
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def _redacted_failure_message(value: Any) -> Optional[str]:
    text = _bounded_summary_field(value, limit=4096)
    if text is None:
        return None
    text = PROVIDER_FAILURE_SECRET_RE.sub("REDACTED", text)
    return _bounded_summary_field(text, limit=PROVIDER_FAILURE_MESSAGE_LIMIT)


def _redacted_summary_field(value: Any, *, limit: int = PROVIDER_FAILURE_FIELD_LIMIT) -> Optional[str]:
    text = _bounded_summary_field(value, limit=limit)
    if text is None:
        return None
    return PROVIDER_FAILURE_SECRET_RE.sub("REDACTED", text)


def _provider_failure_summaries(rows: Sequence[Dict[str, Any]]) -> tuple[list[Dict[str, Any]], int]:
    failed_rows = [row for row in rows if not row.get("success")]
    summaries: list[Dict[str, Any]] = []
    for row in failed_rows[:PROVIDER_FAILURE_SUMMARY_LIMIT]:
        summaries.append(
            {
                "provider": _redacted_summary_field(row.get("provider")),
                "endpoint_key": _redacted_summary_field(row.get("endpoint_key")),
                "probe_type": _redacted_summary_field(row.get("probe_type")),
                "error_class": _redacted_summary_field(row.get("error_class")),
                "error_message": _redacted_failure_message(row.get("error_message")),
            }
        )
    return summaries, max(0, len(failed_rows) - len(summaries))


def run_cycle(config: ProviderStatusLoopConfig) -> Dict[str, Any]:
    started = time.perf_counter()
    rows = probes.collect_observations(
        probes.DEFAULT_ENDPOINTS,
        environment=config.environment,
        timeout=config.timeout,
        ping_count=config.ping_count,
        ping_timeout=config.ping_timeout,
        skip_icmp=config.skip_icmp,
    )
    inserted = False
    skipped = False
    skip_reason: Optional[str] = None
    skip_error_class: Optional[str] = None
    if config.apply:
        dsn = _resolve_dsn(config)
        try:
            probes.insert_observations(
                dsn,
                rows,
                lock_timeout_ms=config.db_lock_timeout_ms,
                statement_timeout_ms=config.db_statement_timeout_ms,
            )
        except probes.ProviderStatusDatabaseWriteSkipped as exc:
            skipped = True
            skip_reason = str(exc)
            skip_error_class = exc.error_class
        else:
            inserted = True

    elapsed_ms = round((time.perf_counter() - started) * 1000, 3)
    successes = sum(1 for row in rows if row.get("success"))
    failures = len(rows) - successes
    summary = {
        "event": "provider_status_observations_cycle",
        "observed_at": _utc_timestamp(),
        "apply": config.apply,
        "inserted": inserted,
        "skipped": skipped,
        "skip_error_class": skip_error_class,
        "skip_reason": skip_reason,
        "environment": config.environment,
        "row_count": len(rows),
        "success_count": successes,
        "failure_count": failures,
        "duration_ms": elapsed_ms,
    }
    if failures:
        failure_summaries, omitted_count = _provider_failure_summaries(rows)
        summary["failure_summaries"] = failure_summaries
        summary["failure_summaries_omitted_count"] = omitted_count
    return summary


def run_due_sidecar_tasks(
    config: ProviderStatusLoopConfig,
    state: SidecarTaskState,
    *,
    now_monotonic: Optional[float] = None,
) -> list[Dict[str, Any]]:
    if not config.grok_oidc_refresh_enabled:
        return []

    now = time.monotonic() if now_monotonic is None else now_monotonic
    last_attempt = state.grok_oidc_last_attempt_monotonic
    if (
        last_attempt is not None
        and now - last_attempt < config.grok_oidc_refresh_interval_seconds
    ):
        return []

    state.grok_oidc_last_attempt_monotonic = now
    try:
        summary = grok_oidc_refresh.refresh_grok_oidc_auth_file(
            config.grok_oidc_auth_file,
            buffer_seconds=config.grok_oidc_refresh_buffer_seconds,
            force=config.grok_oidc_force_refresh,
            lock_file=config.grok_oidc_lock_file,
            http_timeout_seconds=config.grok_oidc_http_timeout_seconds,
        )
    except Exception as exc:
        summary = {
            "attempted": True,
            "refreshed": False,
            "skipped": False,
            "auth_file": config.grok_oidc_auth_file,
            "scope": None,
            "error_class": exc.__class__.__name__,
            "error_message": _redacted_failure_message(str(exc)),
        }

    return [
        {
            "event": "grok_oidc_refresh",
            "observed_at": _utc_timestamp(),
            "environment": config.environment,
            **summary,
        }
    ]


def _emit(payload: Dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(payload, sort_keys=True))
    sys.stdout.write("\n")
    sys.stdout.flush()


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def main(argv: Optional[Sequence[str]] = None) -> int:
    config = parse_config(argv)
    try:
        validate_runtime_guardrails(config)
    except Exception as exc:
        _emit(
            {
                "event": "provider_status_observations_guardrail_error",
                "observed_at": _utc_timestamp(),
                "environment": config.environment,
                "error_class": exc.__class__.__name__,
                "error_message": str(exc),
            }
        )
        return 1
    stopping = False

    def _stop(_signum: int, _frame: object) -> None:
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    if config.apply and config.setup_schema:
        try:
            _emit(setup_schema_once(config))
        except Exception as exc:
            _emit(
                {
                    "event": "provider_status_observations_schema_error",
                    "observed_at": _utc_timestamp(),
                    "environment": config.environment,
                    "error_class": exc.__class__.__name__,
                    "error_message": str(exc),
                    "traceback": traceback.format_exc(limit=5),
                }
            )
            if config.once:
                return 1

    sidecar_state = SidecarTaskState()
    while not stopping:
        cycle_started = time.monotonic()
        try:
            _emit(run_cycle(config))
        except Exception as exc:
            _emit(
                {
                    "event": "provider_status_observations_error",
                    "observed_at": _utc_timestamp(),
                    "environment": config.environment,
                    "error_class": exc.__class__.__name__,
                    "error_message": str(exc),
                    "traceback": traceback.format_exc(limit=5),
                }
            )
            if config.once:
                return 1
        try:
            for event in run_due_sidecar_tasks(config, sidecar_state):
                _emit(event)
        except Exception as exc:
            _emit(
                {
                    "event": "provider_status_sidecar_task_error",
                    "observed_at": _utc_timestamp(),
                    "environment": config.environment,
                    "task": "grok_oidc_refresh",
                    "error_class": exc.__class__.__name__,
                    "error_message": _redacted_failure_message(str(exc)),
                    "traceback": traceback.format_exc(limit=5),
                }
            )
            if config.once:
                return 1

        if config.once:
            return 0

        remaining_seconds = config.interval_seconds - (time.monotonic() - cycle_started)
        while remaining_seconds > 0 and not stopping:
            time.sleep(min(remaining_seconds, 1.0))
            remaining_seconds = config.interval_seconds - (time.monotonic() - cycle_started)

    _emit(
        {
            "event": "provider_status_observations_stopped",
            "observed_at": _utc_timestamp(),
            "environment": config.environment,
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
