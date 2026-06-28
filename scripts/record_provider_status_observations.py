#!/usr/bin/env python3
"""Collect non-inference provider status observations.

The script intentionally avoids chat/completion/messages/generate endpoints. It
records network/front-door signals only: ICMP ping, DNS, TCP connect, and TLS
handshake. By default it prints JSON rows; pass --apply to insert into
public.provider_status_observations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import socket
import ssl
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional
from urllib.parse import parse_qsl, quote, urlencode, urlsplit, urlunsplit

import psycopg

DEFAULT_DB_LOCK_TIMEOUT_MS = 1000
DEFAULT_DB_STATEMENT_TIMEOUT_MS = 5000
DEFAULT_DB_APPLICATION_NAME = "aawm-provider-status-observations"
DEFAULT_CODEX_RESET_CREDIT_DETAIL_URL = (
    "https://chatgpt.com/backend-api/wham/rate-limit-reset-credits"
)
LEGACY_CODEX_WHAM_USAGE_URL = "https://chatgpt.com/backend-api/wham/usage"

CODEX_RESET_CREDIT_SEED_METADATA: tuple[Dict[str, Any], ...] = (
    {
        "granted_at": datetime(2026, 6, 12, 16, 17, tzinfo=timezone.utc),
        "expires_at": datetime(2026, 7, 12, 16, 17, tzinfo=timezone.utc),
        "reset_type": "codex_rate_limits",
        "operator_annotation": None,
        "source_url": "https://x.com/thsottiaux/status/2065468501750649006",
    },
    {
        "granted_at": datetime(2026, 6, 19, 0, 10, tzinfo=timezone.utc),
        "expires_at": datetime(2026, 7, 19, 0, 10, tzinfo=timezone.utc),
        "reset_type": "codex_rate_limits",
        "operator_annotation": None,
        "source_url": "https://x.com/thsottiaux/status/2067399435009622521",
    },
    {
        "granted_at": datetime(2026, 6, 24, 21, 53, tzinfo=timezone.utc),
        "expires_at": datetime(2026, 7, 24, 21, 53, tzinfo=timezone.utc),
        "reset_type": "codex_rate_limits",
        "operator_annotation": "Invite Promotion",
        "source_url": None,
    },
    {
        "granted_at": datetime(
            2026, 6, 24, 22, 41, 38, 714466, tzinfo=timezone.utc
        ),
        "expires_at": datetime(
            2026, 7, 24, 22, 41, 38, 714466, tzinfo=timezone.utc
        ),
        "reset_type": "codex_rate_limits",
        "operator_annotation": "Invite Promotion",
        "source_url": None,
    },
)



PING_STATS_RE = re.compile(
    r"(?P<sent>\d+)\s+packets transmitted,\s+"
    r"(?P<received>\d+)\s+(?:packets )?received,\s+"
    r"(?P<loss>[0-9.]+)% packet loss"
)
PING_RTT_RE = re.compile(
    r"rtt min/avg/max/(?:mdev|stddev) = "
    r"(?P<min>[0-9.]+)/(?P<avg>[0-9.]+)/(?P<max>[0-9.]+)/(?P<mdev>[0-9.]+) ms"
)
PING_IP_RE = re.compile(r"PING\s+[^\s]+\s+\((?P<ip>[^)]+)\)")


@dataclass(frozen=True)
class Endpoint:
    provider: str
    endpoint_key: str
    host: str
    port: int = 443


DEFAULT_ENDPOINTS: tuple[Endpoint, ...] = (
    Endpoint("anthropic", "api.anthropic.com:443", "api.anthropic.com"),
    Endpoint("openai", "api.openai.com:443", "api.openai.com"),
    Endpoint("openrouter", "openrouter.ai:443", "openrouter.ai"),
    Endpoint("xai", "cli-chat-proxy.grok.com:443", "cli-chat-proxy.grok.com"),
    Endpoint("xai", "api.x.ai:443", "api.x.ai"),
    Endpoint("nvidia_nim", "integrate.api.nvidia.com:443", "integrate.api.nvidia.com"),
    Endpoint("gemini", "generativelanguage.googleapis.com:443", "generativelanguage.googleapis.com"),
    Endpoint("gemini", "cloudcode-pa.googleapis.com:443", "cloudcode-pa.googleapis.com"),
    Endpoint("control", "control:google.com", "google.com"),
)


PROVIDER_STATUS_INSERT_SQL = """
INSERT INTO public.provider_status_observations (
    observed_at,
    environment,
    provider,
    endpoint_key,
    probe_type,
    success,
    status_code,
    address_family,
    resolved_ip,
    packet_loss_pct,
    icmp_rtt_min_ms,
    icmp_rtt_avg_ms,
    icmp_rtt_max_ms,
    icmp_rtt_mdev_ms,
    dns_ms,
    tcp_ms,
    tls_ms,
    ttfb_ms,
    total_ms,
    status_summary,
    error_class,
    error_message,
    metadata
) VALUES (
    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
    %s, %s, %s::jsonb
)
"""
PROVIDER_STATUS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS public.provider_status_observations (
    id BIGSERIAL PRIMARY KEY,
    observed_at TIMESTAMPTZ NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    environment TEXT NOT NULL,
    provider TEXT NOT NULL,
    endpoint_key TEXT NOT NULL,
    probe_type TEXT NOT NULL,
    success BOOLEAN NOT NULL,
    status_code INTEGER,
    address_family TEXT,
    resolved_ip TEXT,
    packet_loss_pct DOUBLE PRECISION,
    icmp_rtt_min_ms DOUBLE PRECISION,
    icmp_rtt_avg_ms DOUBLE PRECISION,
    icmp_rtt_max_ms DOUBLE PRECISION,
    icmp_rtt_mdev_ms DOUBLE PRECISION,
    dns_ms DOUBLE PRECISION,
    tcp_ms DOUBLE PRECISION,
    tls_ms DOUBLE PRECISION,
    ttfb_ms DOUBLE PRECISION,
    total_ms DOUBLE PRECISION,
    status_summary TEXT,
    error_class TEXT,
    error_message TEXT,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb
)
"""
PROVIDER_STATUS_ALTER_STATEMENTS = (
    "ALTER TABLE public.provider_status_observations ADD COLUMN IF NOT EXISTS address_family TEXT",
    "ALTER TABLE public.provider_status_observations ADD COLUMN IF NOT EXISTS resolved_ip TEXT",
    "ALTER TABLE public.provider_status_observations ADD COLUMN IF NOT EXISTS packet_loss_pct DOUBLE PRECISION",
    "ALTER TABLE public.provider_status_observations ADD COLUMN IF NOT EXISTS icmp_rtt_min_ms DOUBLE PRECISION",
    "ALTER TABLE public.provider_status_observations ADD COLUMN IF NOT EXISTS icmp_rtt_avg_ms DOUBLE PRECISION",
    "ALTER TABLE public.provider_status_observations ADD COLUMN IF NOT EXISTS icmp_rtt_max_ms DOUBLE PRECISION",
    "ALTER TABLE public.provider_status_observations ADD COLUMN IF NOT EXISTS icmp_rtt_mdev_ms DOUBLE PRECISION",
)
PROVIDER_STATUS_INDEX_STATEMENTS = (
    "CREATE INDEX IF NOT EXISTS provider_status_observations_provider_time_idx ON public.provider_status_observations (provider, observed_at DESC)",
    "CREATE INDEX IF NOT EXISTS provider_status_observations_endpoint_time_idx ON public.provider_status_observations (provider, endpoint_key, observed_at DESC)",
    "CREATE INDEX IF NOT EXISTS provider_status_observations_probe_time_idx ON public.provider_status_observations (probe_type, observed_at DESC)",
)

PROVIDER_AUTH_OBSERVATIONS_INSERT_SQL = """
INSERT INTO public.provider_auth_observations (
    observed_at,
    environment,
    provider,
    auth_family,
    credential_scope,
    auth_file_hash,
    status,
    attempted,
    refreshed,
    skipped,
    expires_at,
    last_success_at,
    source_task,
    error_class,
    error_message,
    metadata
) VALUES (
    %s, %s, %s, %s, %s, %s, %s, %s,
    %s, %s, %s, %s, %s, %s, %s, %s::jsonb
)
"""
PROVIDER_AUTH_OBSERVATIONS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS public.provider_auth_observations (
    id BIGSERIAL PRIMARY KEY,
    observed_at TIMESTAMPTZ NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    environment TEXT NOT NULL,
    provider TEXT NOT NULL,
    auth_family TEXT NOT NULL,
    credential_scope TEXT,
    auth_file_hash TEXT,
    status TEXT NOT NULL,
    attempted BOOLEAN NOT NULL DEFAULT FALSE,
    refreshed BOOLEAN NOT NULL DEFAULT FALSE,
    skipped BOOLEAN NOT NULL DEFAULT FALSE,
    expires_at TIMESTAMPTZ,
    last_success_at TIMESTAMPTZ,
    source_task TEXT NOT NULL,
    error_class TEXT,
    error_message TEXT,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb
)
"""
PROVIDER_AUTH_OBSERVATIONS_ALTER_STATEMENTS = (
    "ALTER TABLE public.provider_auth_observations ADD COLUMN IF NOT EXISTS credential_scope TEXT",
    "ALTER TABLE public.provider_auth_observations ADD COLUMN IF NOT EXISTS auth_file_hash TEXT",
    "ALTER TABLE public.provider_auth_observations ADD COLUMN IF NOT EXISTS attempted BOOLEAN NOT NULL DEFAULT FALSE",
    "ALTER TABLE public.provider_auth_observations ADD COLUMN IF NOT EXISTS refreshed BOOLEAN NOT NULL DEFAULT FALSE",
    "ALTER TABLE public.provider_auth_observations ADD COLUMN IF NOT EXISTS skipped BOOLEAN NOT NULL DEFAULT FALSE",
    "ALTER TABLE public.provider_auth_observations ADD COLUMN IF NOT EXISTS expires_at TIMESTAMPTZ",
    "ALTER TABLE public.provider_auth_observations ADD COLUMN IF NOT EXISTS last_success_at TIMESTAMPTZ",
    "ALTER TABLE public.provider_auth_observations ADD COLUMN IF NOT EXISTS source_task TEXT NOT NULL DEFAULT 'unknown'",
    "ALTER TABLE public.provider_auth_observations ADD COLUMN IF NOT EXISTS error_class TEXT",
    "ALTER TABLE public.provider_auth_observations ADD COLUMN IF NOT EXISTS error_message TEXT",
    "ALTER TABLE public.provider_auth_observations ADD COLUMN IF NOT EXISTS metadata JSONB NOT NULL DEFAULT '{}'::jsonb",
)
PROVIDER_AUTH_OBSERVATIONS_INDEX_STATEMENTS = (
    "CREATE INDEX IF NOT EXISTS provider_auth_observations_provider_time_idx ON public.provider_auth_observations (provider, observed_at DESC)",
    "CREATE INDEX IF NOT EXISTS provider_auth_observations_identity_time_idx ON public.provider_auth_observations (environment, provider, auth_family, credential_scope, auth_file_hash, observed_at DESC)",
    "CREATE INDEX IF NOT EXISTS provider_auth_observations_status_time_idx ON public.provider_auth_observations (status, observed_at DESC)",
    "CREATE INDEX IF NOT EXISTS provider_auth_observations_expires_idx ON public.provider_auth_observations (expires_at)",
)
PROVIDER_AUTH_CURRENT_VIEW_SQL = """
CREATE OR REPLACE VIEW public.provider_auth_current AS
SELECT DISTINCT ON (
    environment,
    provider,
    auth_family,
    COALESCE(credential_scope, ''),
    COALESCE(auth_file_hash, '')
)
    id,
    observed_at,
    created_at,
    environment,
    provider,
    auth_family,
    credential_scope,
    auth_file_hash,
    status,
    attempted,
    refreshed,
    skipped,
    expires_at,
    last_success_at,
    source_task,
    error_class,
    error_message,
    metadata
FROM public.provider_auth_observations
ORDER BY
    environment,
    provider,
    auth_family,
    COALESCE(credential_scope, ''),
    COALESCE(auth_file_hash, ''),
    observed_at DESC,
    id DESC
"""


PROVIDER_CREDIT_OBSERVATIONS_INSERT_SQL = """
WITH candidate AS (
    SELECT
        %s::timestamptz AS observed_at,
        %s::text AS environment,
        %s::text AS provider,
        %s::text AS account_hash,
        %s::text AS credit_family,
        %s::text AS credit_type,
        %s::text AS credit_identity,
        %s::integer AS available_count,
        %s::timestamptz AS granted_at,
        %s::timestamptz AS expires_at,
        %s::text AS status,
        %s::timestamptz AS redeem_started_at,
        %s::timestamptz AS redeemed_at,
        %s::text AS operator_annotation,
        %s::text AS source_url,
        %s::jsonb AS raw_provider_fields,
        %s::jsonb AS evidence,
        %s::text AS source
),
locked AS (
    SELECT pg_advisory_xact_lock(
        hashtext(
            CONCAT_WS(
                '|',
                candidate.environment,
                candidate.provider,
                COALESCE(candidate.account_hash, '<null>'),
                candidate.credit_family,
                COALESCE(candidate.credit_identity, '<null>'),
                COALESCE(candidate.source, '<null>')
            )
        )::bigint
    ) AS lock_acquired
    FROM candidate
)
INSERT INTO public.provider_credit_observations (
    observed_at,
    environment,
    provider,
    account_hash,
    credit_family,
    credit_type,
    credit_identity,
    available_count,
    granted_at,
    expires_at,
    status,
    redeem_started_at,
    redeemed_at,
    operator_annotation,
    source_url,
    raw_provider_fields,
    evidence,
    source
)
SELECT
    candidate.observed_at,
    candidate.environment,
    candidate.provider,
    candidate.account_hash,
    candidate.credit_family,
    candidate.credit_type,
    candidate.credit_identity,
    candidate.available_count,
    candidate.granted_at,
    candidate.expires_at,
    candidate.status,
    candidate.redeem_started_at,
    candidate.redeemed_at,
    candidate.operator_annotation,
    candidate.source_url,
    COALESCE(candidate.raw_provider_fields, '{}'::jsonb),
    COALESCE(candidate.evidence, '{}'::jsonb),
    candidate.source
FROM candidate
CROSS JOIN locked
WHERE NOT EXISTS (
    SELECT 1
    FROM (
        SELECT
            latest.available_count,
            latest.granted_at,
            latest.expires_at,
            latest.status,
            latest.redeem_started_at,
            latest.redeemed_at,
            latest.operator_annotation,
            latest.source_url
        FROM public.provider_credit_observations AS latest
        WHERE latest.environment = candidate.environment
          AND latest.provider = candidate.provider
          AND latest.credit_family = candidate.credit_family
          AND latest.account_hash IS NOT DISTINCT FROM candidate.account_hash
          AND latest.credit_identity IS NOT DISTINCT FROM candidate.credit_identity
          AND latest.source IS NOT DISTINCT FROM candidate.source
        ORDER BY latest.observed_at DESC, latest.id DESC
        LIMIT 1
    ) AS latest
    WHERE latest.available_count IS NOT DISTINCT FROM candidate.available_count
      AND latest.granted_at IS NOT DISTINCT FROM candidate.granted_at
      AND latest.expires_at IS NOT DISTINCT FROM candidate.expires_at
      AND latest.status IS NOT DISTINCT FROM candidate.status
      AND latest.redeem_started_at IS NOT DISTINCT FROM candidate.redeem_started_at
      AND latest.redeemed_at IS NOT DISTINCT FROM candidate.redeemed_at
      AND latest.operator_annotation IS NOT DISTINCT FROM candidate.operator_annotation
      AND latest.source_url IS NOT DISTINCT FROM candidate.source_url
)
"""
PROVIDER_CREDIT_OBSERVATIONS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS public.provider_credit_observations (
    id BIGSERIAL PRIMARY KEY,
    observed_at TIMESTAMPTZ NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    environment TEXT NOT NULL,
    provider TEXT NOT NULL,
    account_hash TEXT NOT NULL,
    credit_family TEXT NOT NULL,
    credit_type TEXT NOT NULL,
    credit_identity TEXT NOT NULL DEFAULT '',
    available_count INTEGER NOT NULL,
    granted_at TIMESTAMPTZ,
    expires_at TIMESTAMPTZ,
    status TEXT NOT NULL DEFAULT 'available',
    redeem_started_at TIMESTAMPTZ,
    redeemed_at TIMESTAMPTZ,
    operator_annotation TEXT,
    source_url TEXT,
    source TEXT NOT NULL,
    raw_provider_fields JSONB NOT NULL DEFAULT '{}'::jsonb,
    evidence JSONB NOT NULL DEFAULT '{}'::jsonb
)
"""
PROVIDER_CREDIT_OBSERVATIONS_ALTER_STATEMENTS = (
    "ALTER TABLE public.provider_credit_observations ADD COLUMN IF NOT EXISTS credit_identity TEXT NOT NULL DEFAULT ''",
    "ALTER TABLE public.provider_credit_observations ADD COLUMN IF NOT EXISTS granted_at TIMESTAMPTZ",
    "ALTER TABLE public.provider_credit_observations ADD COLUMN IF NOT EXISTS status TEXT NOT NULL DEFAULT 'available'",
    "ALTER TABLE public.provider_credit_observations ADD COLUMN IF NOT EXISTS redeem_started_at TIMESTAMPTZ",
    "ALTER TABLE public.provider_credit_observations ADD COLUMN IF NOT EXISTS redeemed_at TIMESTAMPTZ",
    "ALTER TABLE public.provider_credit_observations ADD COLUMN IF NOT EXISTS operator_annotation TEXT",
    "ALTER TABLE public.provider_credit_observations ADD COLUMN IF NOT EXISTS source_url TEXT",
)
PROVIDER_CREDIT_OBSERVATIONS_INDEX_STATEMENTS = (
    "CREATE INDEX IF NOT EXISTS provider_credit_observations_provider_time_idx ON public.provider_credit_observations (provider, observed_at DESC)",
    "CREATE INDEX IF NOT EXISTS provider_credit_observations_identity_time_idx ON public.provider_credit_observations (environment, provider, account_hash, credit_family, source, observed_at DESC)",
    "CREATE INDEX IF NOT EXISTS provider_credit_observations_credit_identity_time_idx ON public.provider_credit_observations (environment, provider, account_hash, credit_family, credit_identity, source, observed_at DESC)",
    "CREATE INDEX IF NOT EXISTS provider_credit_observations_expires_idx ON public.provider_credit_observations (expires_at)",
    "CREATE INDEX IF NOT EXISTS provider_credit_observations_status_time_idx ON public.provider_credit_observations (status, observed_at DESC)",
)
PROVIDER_CREDIT_CURRENT_VIEW_SQL = """
CREATE OR REPLACE VIEW public.provider_credit_current AS
SELECT DISTINCT ON (
    environment,
    provider,
    account_hash,
    credit_family,
    credit_identity,
    COALESCE(source, '')
)
    id,
    observed_at,
    created_at,
    environment,
    provider,
    account_hash,
    credit_family,
    credit_type,
    available_count,
    expires_at,
    source,
    raw_provider_fields,
    evidence,
    credit_identity,
    granted_at,
    status,
    redeem_started_at,
    redeemed_at,
    operator_annotation,
    source_url
FROM public.provider_credit_observations AS current_credit
WHERE current_credit.credit_identity <> ''
   OR NOT EXISTS (
       SELECT 1
       FROM public.provider_credit_observations AS detail_credit
       WHERE detail_credit.environment = current_credit.environment
         AND detail_credit.provider = current_credit.provider
         AND detail_credit.account_hash = current_credit.account_hash
         AND detail_credit.credit_family = current_credit.credit_family
         AND detail_credit.source IS NOT DISTINCT FROM current_credit.source
         AND detail_credit.credit_identity <> ''
   )
ORDER BY
    environment,
    provider,
    account_hash,
    credit_family,
    credit_identity,
    COALESCE(source, ''),
    observed_at DESC,
    id DESC
"""

PROVIDER_CREDIT_CURRENT_SELECT_SQL = """
SELECT
    id,
    observed_at,
    environment,
    provider,
    account_hash,
    credit_family,
    credit_type,
    credit_identity,
    available_count,
    granted_at,
    expires_at,
    status,
    redeem_started_at,
    redeemed_at,
    operator_annotation,
    source_url,
    source,
    raw_provider_fields,
    evidence
FROM public.provider_credit_current
WHERE environment = %s
  AND provider = %s
  AND account_hash = %s
  AND credit_family = %s
  AND source = %s
"""


class ProviderStatusDatabaseWriteSkipped(RuntimeError):
    def __init__(self, *, error_class: str, message: str) -> None:
        super().__init__(message)
        self.error_class = error_class


def _append_dsn_query_params(dsn: str, params: Dict[str, Optional[str]]) -> str:
    parsed = urlsplit(dsn)
    if not parsed.scheme:
        return dsn

    query_items = parse_qsl(parsed.query, keep_blank_values=True)
    existing_keys = {key for key, _value in query_items}
    for key, value in params.items():
        cleaned_value = value.strip() if isinstance(value, str) else None
        if cleaned_value and key not in existing_keys:
            query_items.append((key, cleaned_value))
            existing_keys.add(key)
    return urlunsplit(
        (
            parsed.scheme,
            parsed.netloc,
            parsed.path,
            urlencode(query_items),
            parsed.fragment,
        )
    )


def _provider_status_db_application_name() -> str:
    return (
        os.getenv("AAWM_PROVIDER_STATUS_DB_APPLICATION_NAME")
        or os.getenv("AAWM_DB_APPLICATION_NAME")
        or os.getenv("PGAPPNAME")
        or DEFAULT_DB_APPLICATION_NAME
    ).strip() or DEFAULT_DB_APPLICATION_NAME


def _build_dsn(args: argparse.Namespace) -> Optional[str]:
    if args.dsn:
        return _append_dsn_query_params(
            args.dsn,
            {"application_name": _provider_status_db_application_name()},
        )

    host = args.pg_host or os.getenv("AAWM_DB_HOST") or os.getenv("PGHOST")
    database = args.pg_database or os.getenv("AAWM_DB_NAME") or os.getenv("PGDATABASE")
    user = args.pg_user or os.getenv("AAWM_DB_USER") or os.getenv("PGUSER")
    password = args.pg_password or os.getenv("AAWM_DB_PASSWORD") or os.getenv("PGPASSWORD")
    configured_port = args.pg_port or os.getenv("AAWM_DB_PORT") or os.getenv("PGPORT")
    port = configured_port or "5432"
    sslmode = args.pg_sslmode or os.getenv("AAWM_DB_SSLMODE") or os.getenv("PGSSLMODE")
    has_component_config = any((host, database, user, password, configured_port, sslmode))
    if has_component_config:
        if not (host and database and user):
            return None

        dsn = (
            f"postgresql://{quote(user, safe='')}"
            f"{':' + quote(password, safe='') if password else ''}"
            f"@{host}:{port}/{quote(database, safe='')}"
        )
        if sslmode:
            dsn += f"?{urlencode({'sslmode': sslmode})}"
        return _append_dsn_query_params(
            dsn,
            {"application_name": _provider_status_db_application_name()},
        )

    for key in ("AAWM_DB_URL", "AAWM_DATABASE_URL", "AAWM_POSTGRES_URL"):
        value = os.getenv(key)
        if value:
            return _append_dsn_query_params(
                value,
                {"application_name": _provider_status_db_application_name()},
            )
    return None


def _empty_observation(
    *,
    endpoint: Endpoint,
    environment: str,
    observed_at: datetime,
    probe_type: str,
) -> Dict[str, Any]:
    return {
        "observed_at": observed_at,
        "environment": environment,
        "provider": endpoint.provider,
        "endpoint_key": endpoint.endpoint_key,
        "probe_type": probe_type,
        "success": False,
        "status_code": None,
        "address_family": None,
        "resolved_ip": None,
        "packet_loss_pct": None,
        "icmp_rtt_min_ms": None,
        "icmp_rtt_avg_ms": None,
        "icmp_rtt_max_ms": None,
        "icmp_rtt_mdev_ms": None,
        "dns_ms": None,
        "tcp_ms": None,
        "tls_ms": None,
        "ttfb_ms": None,
        "total_ms": None,
        "status_summary": None,
        "error_class": None,
        "error_message": None,
        "metadata": {},
    }


def _family_name(family: int) -> str:
    if family == socket.AF_INET:
        return "ipv4"
    if family == socket.AF_INET6:
        return "ipv6"
    return str(family)


def _resolve_host(
    endpoint: Endpoint,
    *,
    environment: str,
    observed_at: datetime,
    timeout: float,
) -> tuple[Dict[str, Any], Optional[str], Optional[int]]:
    observation = _empty_observation(
        endpoint=endpoint,
        environment=environment,
        observed_at=observed_at,
        probe_type="dns",
    )
    started = time.perf_counter()
    previous_timeout = socket.getdefaulttimeout()
    socket.setdefaulttimeout(timeout)
    try:
        results = socket.getaddrinfo(endpoint.host, endpoint.port, type=socket.SOCK_STREAM)
    except OSError as exc:
        observation["error_class"] = "dns_error"
        observation["error_message"] = str(exc)[:300]
        observation["total_ms"] = round((time.perf_counter() - started) * 1000, 3)
        observation["dns_ms"] = observation["total_ms"]
        return observation, None, None
    finally:
        socket.setdefaulttimeout(previous_timeout)

    elapsed_ms = round((time.perf_counter() - started) * 1000, 3)
    if not results:
        observation["error_class"] = "dns_empty"
        observation["dns_ms"] = elapsed_ms
        observation["total_ms"] = elapsed_ms
        return observation, None, None

    family, _socktype, _proto, _canonname, sockaddr = results[0]
    resolved_ip = sockaddr[0]
    observation.update(
        {
            "success": True,
            "address_family": _family_name(family),
            "resolved_ip": resolved_ip,
            "dns_ms": elapsed_ms,
            "total_ms": elapsed_ms,
            "metadata": {"address_count": len(results)},
        }
    )
    return observation, str(resolved_ip), int(family)


def _tcp_probe(
    endpoint: Endpoint,
    *,
    environment: str,
    observed_at: datetime,
    resolved_ip: Optional[str],
    family: Optional[int],
    timeout: float,
) -> Dict[str, Any]:
    observation = _empty_observation(
        endpoint=endpoint,
        environment=environment,
        observed_at=observed_at,
        probe_type="tcp_connect",
    )
    observation["resolved_ip"] = resolved_ip
    observation["address_family"] = _family_name(family) if family is not None else None
    if resolved_ip is None:
        observation["error_class"] = "dns_error"
        return observation

    started = time.perf_counter()
    try:
        with socket.create_connection((resolved_ip, endpoint.port), timeout=timeout):
            pass
    except OSError as exc:
        observation["error_class"] = "tcp_error"
        observation["error_message"] = str(exc)[:300]
    else:
        observation["success"] = True
    elapsed_ms = round((time.perf_counter() - started) * 1000, 3)
    observation["tcp_ms"] = elapsed_ms
    observation["total_ms"] = elapsed_ms
    return observation


def _tls_probe(
    endpoint: Endpoint,
    *,
    environment: str,
    observed_at: datetime,
    resolved_ip: Optional[str],
    family: Optional[int],
    timeout: float,
) -> Dict[str, Any]:
    observation = _empty_observation(
        endpoint=endpoint,
        environment=environment,
        observed_at=observed_at,
        probe_type="tls_handshake",
    )
    observation["resolved_ip"] = resolved_ip
    observation["address_family"] = _family_name(family) if family is not None else None
    if resolved_ip is None:
        observation["error_class"] = "dns_error"
        return observation

    context = ssl.create_default_context()
    started = time.perf_counter()
    try:
        with socket.create_connection((resolved_ip, endpoint.port), timeout=timeout) as raw_sock:
            connect_ms = round((time.perf_counter() - started) * 1000, 3)
            tls_started = time.perf_counter()
            with context.wrap_socket(raw_sock, server_hostname=endpoint.host) as tls_sock:
                cert = tls_sock.getpeercert() or {}
                tls_ms = round((time.perf_counter() - tls_started) * 1000, 3)
                observation.update(
                    {
                        "success": True,
                        "tcp_ms": connect_ms,
                        "tls_ms": tls_ms,
                        "metadata": {
                            "tls_version": tls_sock.version(),
                            "cert_subject": cert.get("subject"),
                        },
                    }
                )
    except OSError as exc:
        observation["error_class"] = "tls_error"
        observation["error_message"] = str(exc)[:300]
    elapsed_ms = round((time.perf_counter() - started) * 1000, 3)
    observation["total_ms"] = elapsed_ms
    return observation


def parse_ping_output(output: str) -> Dict[str, Any]:
    stats_match = PING_STATS_RE.search(output)
    rtt_match = PING_RTT_RE.search(output)
    ip_match = PING_IP_RE.search(output)
    parsed: Dict[str, Any] = {}
    if ip_match:
        parsed["resolved_ip"] = ip_match.group("ip")
    if stats_match:
        parsed["sent"] = int(stats_match.group("sent"))
        parsed["received"] = int(stats_match.group("received"))
        parsed["packet_loss_pct"] = float(stats_match.group("loss"))
    if rtt_match:
        parsed["icmp_rtt_min_ms"] = float(rtt_match.group("min"))
        parsed["icmp_rtt_avg_ms"] = float(rtt_match.group("avg"))
        parsed["icmp_rtt_max_ms"] = float(rtt_match.group("max"))
        parsed["icmp_rtt_mdev_ms"] = float(rtt_match.group("mdev"))
    return parsed


def _icmp_probe(
    endpoint: Endpoint,
    *,
    environment: str,
    observed_at: datetime,
    count: int,
    timeout: int,
) -> Dict[str, Any]:
    observation = _empty_observation(
        endpoint=endpoint,
        environment=environment,
        observed_at=observed_at,
        probe_type="icmp_ping",
    )
    started = time.perf_counter()
    command = ["ping", "-c", str(count), "-W", str(timeout), endpoint.host]
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            check=False,
            text=True,
            timeout=max(timeout * count + 2, timeout + 2),
        )
    except subprocess.TimeoutExpired as exc:
        output = "\n".join(
            part
            for part in (
                exc.stdout.decode() if isinstance(exc.stdout, bytes) else exc.stdout,
                exc.stderr.decode() if isinstance(exc.stderr, bytes) else exc.stderr,
            )
            if part
        )
        observation.update(
            {
                "total_ms": round((time.perf_counter() - started) * 1000, 3),
                "error_class": "icmp_timeout",
                "error_message": output[-300:] if output else str(exc),
            }
        )
        return observation

    output = "\n".join(part for part in (completed.stdout, completed.stderr) if part)
    parsed = parse_ping_output(output)
    observation.update(
        {
            "success": completed.returncode == 0 and parsed.get("packet_loss_pct") != 100.0,
            "resolved_ip": parsed.get("resolved_ip"),
            "packet_loss_pct": parsed.get("packet_loss_pct"),
            "icmp_rtt_min_ms": parsed.get("icmp_rtt_min_ms"),
            "icmp_rtt_avg_ms": parsed.get("icmp_rtt_avg_ms"),
            "icmp_rtt_max_ms": parsed.get("icmp_rtt_max_ms"),
            "icmp_rtt_mdev_ms": parsed.get("icmp_rtt_mdev_ms"),
            "total_ms": round((time.perf_counter() - started) * 1000, 3),
            "metadata": {
                "packets_sent": parsed.get("sent"),
                "packets_received": parsed.get("received"),
            },
        }
    )
    if not observation["success"]:
        observation["error_class"] = "icmp_unavailable"
        observation["error_message"] = output[-300:] if output else "ping failed"
    return observation


def collect_observations(
    endpoints: Iterable[Endpoint],
    *,
    environment: str,
    timeout: float,
    ping_count: int,
    ping_timeout: int,
    skip_icmp: bool,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for endpoint in endpoints:
        observed_at = datetime.now(timezone.utc)
        if not skip_icmp:
            rows.append(
                _icmp_probe(
                    endpoint,
                    environment=environment,
                    observed_at=observed_at,
                    count=ping_count,
                    timeout=ping_timeout,
                )
            )
        dns_row, resolved_ip, family = _resolve_host(
            endpoint,
            environment=environment,
            observed_at=observed_at,
            timeout=timeout,
        )
        rows.append(dns_row)
        rows.append(
            _tcp_probe(
                endpoint,
                environment=environment,
                observed_at=observed_at,
                resolved_ip=resolved_ip,
                family=family,
                timeout=timeout,
            )
        )
        rows.append(
            _tls_probe(
                endpoint,
                environment=environment,
                observed_at=observed_at,
                resolved_ip=resolved_ip,
                family=family,
                timeout=timeout,
            )
        )
    return rows


def _db_payload(row: Dict[str, Any]) -> tuple[Any, ...]:
    return (
        row["observed_at"],
        row["environment"],
        row["provider"],
        row["endpoint_key"],
        row["probe_type"],
        row["success"],
        row.get("status_code"),
        row.get("address_family"),
        row.get("resolved_ip"),
        row.get("packet_loss_pct"),
        row.get("icmp_rtt_min_ms"),
        row.get("icmp_rtt_avg_ms"),
        row.get("icmp_rtt_max_ms"),
        row.get("icmp_rtt_mdev_ms"),
        row.get("dns_ms"),
        row.get("tcp_ms"),
        row.get("tls_ms"),
        row.get("ttfb_ms"),
        row.get("total_ms"),
        row.get("status_summary"),
        row.get("error_class"),
        row.get("error_message"),
        json.dumps(row.get("metadata") or {}),
    )


def _set_database_timeouts(
    cur: Any,
    *,
    lock_timeout_ms: int,
    statement_timeout_ms: int,
) -> None:
    cur.execute(
        "SELECT set_config('application_name', %s, false)",
        (_provider_status_db_application_name(),),
    )
    cur.execute("SELECT set_config('lock_timeout', %s, true)", (f"{lock_timeout_ms}ms",))
    cur.execute(
        "SELECT set_config('statement_timeout', %s, true)",
        (f"{statement_timeout_ms}ms",),
    )


def setup_schema(
    dsn: str,
    *,
    lock_timeout_ms: int = DEFAULT_DB_LOCK_TIMEOUT_MS,
    statement_timeout_ms: int = DEFAULT_DB_STATEMENT_TIMEOUT_MS,
) -> None:
    with psycopg.connect(dsn) as conn:
        try:
            with conn.cursor() as cur:
                _set_database_timeouts(
                    cur,
                    lock_timeout_ms=lock_timeout_ms,
                    statement_timeout_ms=statement_timeout_ms,
                )
                cur.execute(PROVIDER_STATUS_TABLE_SQL)
                for statement in PROVIDER_STATUS_ALTER_STATEMENTS:
                    cur.execute(statement)
                for statement in PROVIDER_STATUS_INDEX_STATEMENTS:
                    cur.execute(statement)
                cur.execute(PROVIDER_AUTH_OBSERVATIONS_TABLE_SQL)
                for statement in PROVIDER_AUTH_OBSERVATIONS_ALTER_STATEMENTS:
                    cur.execute(statement)
                for statement in PROVIDER_AUTH_OBSERVATIONS_INDEX_STATEMENTS:
                    cur.execute(statement)
                cur.execute(PROVIDER_AUTH_CURRENT_VIEW_SQL)
                cur.execute(PROVIDER_CREDIT_OBSERVATIONS_TABLE_SQL)
                for statement in PROVIDER_CREDIT_OBSERVATIONS_ALTER_STATEMENTS:
                    cur.execute(statement)
                for statement in PROVIDER_CREDIT_OBSERVATIONS_INDEX_STATEMENTS:
                    cur.execute(statement)
                cur.execute(PROVIDER_CREDIT_CURRENT_VIEW_SQL)
        except (psycopg.errors.LockNotAvailable, psycopg.errors.QueryCanceled) as exc:
            conn.rollback()
            raise ProviderStatusDatabaseWriteSkipped(
                error_class=exc.__class__.__name__,
                message=str(exc),
            ) from exc
        conn.commit()


def insert_observations(
    dsn: str,
    rows: List[Dict[str, Any]],
    *,
    lock_timeout_ms: int = DEFAULT_DB_LOCK_TIMEOUT_MS,
    statement_timeout_ms: int = DEFAULT_DB_STATEMENT_TIMEOUT_MS,
) -> None:
    if not rows:
        return

    with psycopg.connect(dsn) as conn:
        try:
            with conn.cursor() as cur:
                _set_database_timeouts(
                    cur,
                    lock_timeout_ms=lock_timeout_ms,
                    statement_timeout_ms=statement_timeout_ms,
                )
                cur.executemany(
                    PROVIDER_STATUS_INSERT_SQL,
                    [_db_payload(row) for row in rows],
                )
        except (psycopg.errors.LockNotAvailable, psycopg.errors.QueryCanceled) as exc:
            conn.rollback()
            raise ProviderStatusDatabaseWriteSkipped(
                error_class=exc.__class__.__name__,
                message=str(exc),
            ) from exc
        conn.commit()


def _auth_observation_db_payload(row: Dict[str, Any]) -> tuple[Any, ...]:
    return (
        row.get("observed_at"),
        row.get("environment"),
        row.get("provider"),
        row.get("auth_family"),
        row.get("credential_scope"),
        row.get("auth_file_hash"),
        row.get("status"),
        bool(row.get("attempted")),
        bool(row.get("refreshed")),
        bool(row.get("skipped")),
        row.get("expires_at"),
        row.get("last_success_at"),
        row.get("source_task"),
        row.get("error_class"),
        row.get("error_message"),
        json.dumps(row.get("metadata") or {}, default=_json_default, sort_keys=True),
    )


def auth_file_identity_hash(auth_file: Any) -> Optional[str]:
    if auth_file is None:
        return None
    value = str(auth_file).strip()
    if not value:
        return None
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def insert_provider_auth_observations(
    dsn: str,
    rows: List[Dict[str, Any]],
    *,
    lock_timeout_ms: int = DEFAULT_DB_LOCK_TIMEOUT_MS,
    statement_timeout_ms: int = DEFAULT_DB_STATEMENT_TIMEOUT_MS,
) -> int:
    if not rows:
        return 0

    with psycopg.connect(dsn) as conn:
        try:
            with conn.cursor() as cur:
                _set_database_timeouts(
                    cur,
                    lock_timeout_ms=lock_timeout_ms,
                    statement_timeout_ms=statement_timeout_ms,
                )
                cur.executemany(
                    PROVIDER_AUTH_OBSERVATIONS_INSERT_SQL,
                    [_auth_observation_db_payload(row) for row in rows],
                )
                inserted_count = max(0, cur.rowcount)
        except (psycopg.errors.LockNotAvailable, psycopg.errors.QueryCanceled) as exc:
            conn.rollback()
            raise ProviderStatusDatabaseWriteSkipped(
                error_class=exc.__class__.__name__,
                message=str(exc),
            ) from exc
        conn.commit()
    return inserted_count


def _json_default(value: Any) -> str:
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


def account_identity_hash(value: Any) -> Optional[str]:
    if value is None:
        return None
    cleaned = str(value).strip()
    if not cleaned:
        return None
    return hashlib.sha256(cleaned.encode("utf-8")).hexdigest()[:12]




def _normalize_provider_credit_timestamp(value: Any) -> Optional[datetime]:
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, (int, float)):
        try:
            parsed = datetime.fromtimestamp(float(value), timezone.utc)
        except (OSError, OverflowError, ValueError):
            return None
    elif isinstance(value, str):
        normalized = value.strip()
        if not normalized:
            return None
        if normalized.endswith("Z"):
            normalized = normalized[:-1] + "+00:00"
        try:
            parsed = datetime.fromisoformat(normalized)
        except ValueError:
            return None
    else:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def derive_provider_credit_identity(
    *,
    account_hash: str,
    credit_family: str,
    granted_at: Optional[datetime],
    expires_at: Optional[datetime],
    reset_type: Optional[str],
    provider_credit_id: Optional[str] = None,
) -> str:
    if isinstance(provider_credit_id, str) and provider_credit_id.strip():
        return provider_credit_id.strip()
    identity_material = "|".join(
        [
            account_hash,
            credit_family,
            granted_at.isoformat() if granted_at is not None else "",
            expires_at.isoformat() if expires_at is not None else "",
            (reset_type or "").strip(),
        ]
    )
    return hashlib.sha256(identity_material.encode("utf-8")).hexdigest()[:16]


def _provider_credit_seed_metadata_match(
    *,
    granted_at: Optional[datetime],
    expires_at: Optional[datetime],
) -> Dict[str, Optional[str]]:
    if granted_at is None:
        return {"operator_annotation": None, "source_url": None}
    if not isinstance(granted_at, datetime):
        granted_at = _normalize_provider_credit_timestamp(granted_at)
    if granted_at is None:
        return {"operator_annotation": None, "source_url": None}
    if not isinstance(expires_at, datetime):
        expires_at = _normalize_provider_credit_timestamp(expires_at)
    for seed in CODEX_RESET_CREDIT_SEED_METADATA:
        seed_granted = seed.get("granted_at")
        if not isinstance(seed_granted, datetime):
            continue
        if abs((granted_at - seed_granted).total_seconds()) > 120:
            continue
        seed_expires = seed.get("expires_at")
        if isinstance(seed_expires, datetime) and expires_at is not None:
            if abs((expires_at - seed_expires).total_seconds()) > 120:
                continue
        return {
            "operator_annotation": seed.get("operator_annotation"),
            "source_url": seed.get("source_url"),
        }
    return {"operator_annotation": None, "source_url": None}


def apply_provider_credit_seed_metadata(row: Dict[str, Any]) -> Dict[str, Any]:
    granted_at = row.get("granted_at")
    expires_at = row.get("expires_at")
    seed = _provider_credit_seed_metadata_match(
        granted_at=granted_at,
        expires_at=expires_at,
    )
    updated = dict(row)
    if seed["operator_annotation"] and not updated.get("operator_annotation"):
        updated["operator_annotation"] = seed["operator_annotation"]
    if seed["source_url"] and not updated.get("source_url"):
        updated["source_url"] = seed["source_url"]
    return updated


def load_provider_credit_current_rows(
    dsn: str,
    *,
    environment: str,
    provider: str,
    account_hash: str,
    credit_family: str,
    source: str,
    lock_timeout_ms: int = DEFAULT_DB_LOCK_TIMEOUT_MS,
    statement_timeout_ms: int = DEFAULT_DB_STATEMENT_TIMEOUT_MS,
) -> List[Dict[str, Any]]:
    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            _set_database_timeouts(
                cur,
                lock_timeout_ms=lock_timeout_ms,
                statement_timeout_ms=statement_timeout_ms,
            )
            cur.execute(
                PROVIDER_CREDIT_CURRENT_SELECT_SQL,
                (environment, provider, account_hash, credit_family, source),
            )
            columns = [desc.name for desc in cur.description or ()]
            return [dict(zip(columns, row)) for row in cur.fetchall()]

def _provider_credit_db_payload(row: Dict[str, Any]) -> tuple[Any, ...]:
    return (
        row.get("observed_at"),
        row.get("environment"),
        row.get("provider"),
        row.get("account_hash"),
        row.get("credit_family"),
        row.get("credit_type"),
        row.get("credit_identity") or "",
        row.get("available_count"),
        row.get("granted_at"),
        row.get("expires_at"),
        row.get("status") or "available",
        row.get("redeem_started_at"),
        row.get("redeemed_at"),
        row.get("operator_annotation"),
        row.get("source_url"),
        json.dumps(row.get("raw_provider_fields") or {}, sort_keys=True, default=_json_default),
        json.dumps(row.get("evidence") or {}, sort_keys=True, default=_json_default),
        row.get("source"),
    )


def insert_provider_credit_observations(
    dsn: str,
    rows: List[Dict[str, Any]],
    *,
    lock_timeout_ms: int = DEFAULT_DB_LOCK_TIMEOUT_MS,
    statement_timeout_ms: int = DEFAULT_DB_STATEMENT_TIMEOUT_MS,
) -> int:
    if not rows:
        return 0

    with psycopg.connect(dsn) as conn:
        try:
            with conn.cursor() as cur:
                _set_database_timeouts(
                    cur,
                    lock_timeout_ms=lock_timeout_ms,
                    statement_timeout_ms=statement_timeout_ms,
                )
                inserted_count = 0
                for row in rows:
                    payload_row = apply_provider_credit_seed_metadata(row)
                    cur.execute(
                        PROVIDER_CREDIT_OBSERVATIONS_INSERT_SQL,
                        _provider_credit_db_payload(payload_row),
                    )
                    inserted_count += max(0, cur.rowcount)
        except (psycopg.errors.LockNotAvailable, psycopg.errors.QueryCanceled) as exc:
            conn.rollback()
            raise ProviderStatusDatabaseWriteSkipped(
                error_class=exc.__class__.__name__,
                message=str(exc),
            ) from exc
        conn.commit()
    return inserted_count


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--setup-schema",
        action="store_true",
        help="Run provider-status schema setup. Without --apply, setup and exit.",
    )
    parser.add_argument("--apply", action="store_true", help="Insert rows into Postgres.")
    parser.add_argument("--dsn", help="PostgreSQL DSN. Defaults to AAWM_DB_* env vars.")
    parser.add_argument("--environment", default=os.getenv("AAWM_LITELLM_ENVIRONMENT", "unknown"))
    parser.add_argument("--timeout", type=float, default=3.0, help="DNS/TCP/TLS timeout seconds.")
    parser.add_argument("--ping-count", type=int, default=3)
    parser.add_argument("--ping-timeout", type=int, default=2)
    parser.add_argument("--skip-icmp", action="store_true")
    parser.add_argument(
        "--db-lock-timeout-ms",
        type=int,
        default=int(
            os.getenv(
                "AAWM_PROVIDER_STATUS_DB_LOCK_TIMEOUT_MS",
                str(DEFAULT_DB_LOCK_TIMEOUT_MS),
            )
        ),
    )
    parser.add_argument(
        "--db-statement-timeout-ms",
        type=int,
        default=int(
            os.getenv(
                "AAWM_PROVIDER_STATUS_DB_STATEMENT_TIMEOUT_MS",
                str(DEFAULT_DB_STATEMENT_TIMEOUT_MS),
            )
        ),
    )
    parser.add_argument("--pg-host")
    parser.add_argument("--pg-port")
    parser.add_argument("--pg-database")
    parser.add_argument("--pg-user")
    parser.add_argument("--pg-password")
    parser.add_argument("--pg-sslmode")
    args = parser.parse_args()
    dsn: Optional[str] = None
    if args.db_lock_timeout_ms <= 0:
        raise SystemExit("--db-lock-timeout-ms must be greater than 0")
    if args.db_statement_timeout_ms <= 0:
        raise SystemExit("--db-statement-timeout-ms must be greater than 0")

    if args.setup_schema or args.apply:
        dsn = _build_dsn(args)
        if not dsn:
            raise SystemExit("No database DSN found. Set AAWM_DB_* or pass --dsn.")
        if args.setup_schema:
            setup_schema(
                dsn,
                lock_timeout_ms=args.db_lock_timeout_ms,
                statement_timeout_ms=args.db_statement_timeout_ms,
            )
            if not args.apply:
                return 0

    rows = collect_observations(
        DEFAULT_ENDPOINTS,
        environment=args.environment,
        timeout=args.timeout,
        ping_count=args.ping_count,
        ping_timeout=args.ping_timeout,
        skip_icmp=args.skip_icmp,
    )
    if args.apply:
        if dsn is None:
            raise SystemExit("No database DSN found. Set AAWM_DB_* or pass --dsn.")
        insert_observations(
            dsn,
            rows,
            lock_timeout_ms=args.db_lock_timeout_ms,
            statement_timeout_ms=args.db_statement_timeout_ms,
        )

    sys.stdout.write(json.dumps(rows, indent=2, sort_keys=True, default=_json_default))
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
