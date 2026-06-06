#!/usr/bin/env python3
"""Run provider status observations on a fixed container-friendly cadence."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import signal
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

try:
    from scripts import record_provider_status_observations as probes
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    probes = importlib.import_module("record_provider_status_observations")


TRUE_VALUES = {"1", "true", "yes", "on"}
FALSE_VALUES = {"0", "false", "no", "off"}


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
        default=_env_bool("AAWM_PROVIDER_STATUS_SETUP_SCHEMA_ON_START", True),
        help="Run provider-status schema setup once before the loop starts.",
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


def setup_schema_once(config: ProviderStatusLoopConfig) -> Dict[str, Any]:
    started = time.perf_counter()
    dsn = _resolve_dsn(config)
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
    return {
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


def _emit(payload: Dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(payload, sort_keys=True))
    sys.stdout.write("\n")
    sys.stdout.flush()


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def main(argv: Optional[Sequence[str]] = None) -> int:
    config = parse_config(argv)
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
