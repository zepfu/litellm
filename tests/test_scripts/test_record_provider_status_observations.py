import base64
import hashlib
import json
import os
import time
from argparse import Namespace
from datetime import datetime, timedelta, timezone
from io import BytesIO
from pathlib import Path
from subprocess import TimeoutExpired
from urllib import error as urllib_error

import pytest

from litellm.integrations import aawm_agent_identity
from scripts import antigravity_oauth_refresh
from scripts import codex_oauth_refresh
from scripts import record_provider_status_observations as probes
from scripts import run_provider_status_observations_loop as loop
from scripts import xai_oauth_refresh


def _build_test_jwt(payload: dict) -> str:
    def encode_part(value: dict) -> str:
        encoded = base64.urlsafe_b64encode(json.dumps(value).encode("utf-8"))
        return encoded.rstrip(b"=").decode("ascii")

    return f"{encode_part({'alg': 'none'})}.{encode_part(payload)}.sig"


def _provider_status_row() -> dict:
    return {
        "observed_at": datetime(2026, 5, 14, 12, 0, tzinfo=timezone.utc),
        "environment": "dev",
        "provider": "control",
        "endpoint_key": "api.openai.com:443",
        "probe_type": "dns",
        "success": True,
        "status_code": None,
        "address_family": "ipv4",
        "resolved_ip": "172.217.215.101",
        "packet_loss_pct": None,
        "icmp_rtt_min_ms": None,
        "icmp_rtt_avg_ms": None,
        "icmp_rtt_max_ms": None,
        "icmp_rtt_mdev_ms": None,
        "dns_ms": 12.3,
        "tcp_ms": None,
        "tls_ms": None,
        "ttfb_ms": None,
        "total_ms": 12.3,
        "status_summary": None,
        "error_class": None,
        "error_message": None,
        "metadata": {"address_count": 1},
    }


class _FakeProviderStatusCursor:
    def __init__(self) -> None:
        self.execute_calls = []
        self.executemany_calls = []
        self.rowcount = 1

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def execute(self, statement, params=None) -> None:
        self.execute_calls.append((statement, params))

    def executemany(self, statement, payloads) -> None:
        self.executemany_calls.append((statement, payloads))


class _FakeProviderStatusConnection:
    def __init__(self) -> None:
        self.cursor_instance = _FakeProviderStatusCursor()
        self.commit_count = 0
        self.rollback_count = 0

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def cursor(self):
        return self.cursor_instance

    def commit(self) -> None:
        self.commit_count += 1

    def rollback(self) -> None:
        self.rollback_count += 1


def test_build_dsn_prefers_component_config_over_ambient_url(monkeypatch) -> None:
    monkeypatch.setenv("AAWM_DATABASE_URL", "postgresql://aawm:aawm_dev@postgres18:5432/aawm_tristore")
    monkeypatch.setenv("AAWM_DB_HOST", "127.0.0.1")
    monkeypatch.setenv("AAWM_DB_PORT", "5434")
    monkeypatch.setenv("AAWM_DB_NAME", "aawm_tristore")
    monkeypatch.setenv("AAWM_DB_USER", "aawm")
    monkeypatch.setenv("AAWM_DB_PASSWORD", "aawm_dev")
    args = Namespace(
        dsn=None,
        pg_host=None,
        pg_port=None,
        pg_database=None,
        pg_user=None,
        pg_password=None,
        pg_sslmode=None,
    )

    assert (
        probes._build_dsn(args)
        == "postgresql://aawm:aawm_dev@127.0.0.1:5434/aawm_tristore"
        "?application_name=aawm-provider-status-observations"
    )


def test_build_dsn_should_preserve_existing_application_name(monkeypatch) -> None:
    for key in (
        "AAWM_DB_HOST",
        "AAWM_DB_PORT",
        "AAWM_DB_NAME",
        "AAWM_DB_USER",
        "AAWM_DB_PASSWORD",
        "AAWM_DB_SSLMODE",
        "PGHOST",
        "PGPORT",
        "PGDATABASE",
        "PGUSER",
        "PGPASSWORD",
        "PGSSLMODE",
    ):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv(
        "AAWM_DATABASE_URL",
        (
            "postgresql://aawm:aawm_dev@pgbouncer:6432/aawm_tristore"
            "?application_name=custom-provider-status"
        ),
    )
    monkeypatch.setenv("AAWM_PROVIDER_STATUS_DB_APPLICATION_NAME", "ignored")
    args = Namespace(
        dsn=None,
        pg_host=None,
        pg_port=None,
        pg_database=None,
        pg_user=None,
        pg_password=None,
        pg_sslmode=None,
    )

    assert probes._build_dsn(args) == (
        "postgresql://aawm:aawm_dev@pgbouncer:6432/aawm_tristore"
        "?application_name=custom-provider-status"
    )


def test_parse_ping_output_extracts_loss_and_rtt() -> None:
    output = """
PING api.openai.com (172.66.0.243) 56(84) bytes of data.
64 bytes from 172.66.0.243: icmp_seq=1 ttl=53 time=20.8 ms

--- api.openai.com ping statistics ---
3 packets transmitted, 3 received, 0% packet loss, time 2192ms
rtt min/avg/max/mdev = 20.758/31.035/49.640/13.179 ms
"""

    parsed = probes.parse_ping_output(output)

    assert parsed["resolved_ip"] == "172.66.0.243"
    assert parsed["sent"] == 3
    assert parsed["received"] == 3
    assert parsed["packet_loss_pct"] == 0.0
    assert parsed["icmp_rtt_avg_ms"] == 31.035


def test_icmp_probe_records_timeout_without_raising(monkeypatch) -> None:
    def fake_run(*_args, **_kwargs):
        raise TimeoutExpired(
            cmd=["ping", "-c", "1", "-W", "2", "api.anthropic.com"],
            timeout=4,
            output="PING api.anthropic.com (160.79.104.10)",
        )

    monkeypatch.setattr(probes.subprocess, "run", fake_run)

    row = probes._icmp_probe(
        probes.Endpoint("anthropic", "api.anthropic.com:443", "api.anthropic.com"),
        environment="dev",
        observed_at=datetime(2026, 5, 17, 14, 30, tzinfo=timezone.utc),
        count=1,
        timeout=2,
    )

    assert row["success"] is False
    assert row["error_class"] == "icmp_timeout"
    assert "api.anthropic.com" in row["error_message"]
    assert row["total_ms"] is not None


def test_db_payload_preserves_icmp_fields() -> None:
    observed_at = datetime(2026, 5, 14, 12, 0, tzinfo=timezone.utc)
    row = {
        "observed_at": observed_at,
        "environment": "dev",
        "provider": "openai",
        "endpoint_key": "api.openai.com:443",
        "probe_type": "icmp_ping",
        "success": True,
        "status_code": None,
        "address_family": "ipv4",
        "resolved_ip": "172.217.215.101",
        "packet_loss_pct": 0.0,
        "icmp_rtt_min_ms": 29.206,
        "icmp_rtt_avg_ms": 34.297,
        "icmp_rtt_max_ms": 41.778,
        "icmp_rtt_mdev_ms": 5.403,
        "dns_ms": None,
        "tcp_ms": None,
        "tls_ms": None,
        "ttfb_ms": None,
        "total_ms": 2085.0,
        "status_summary": None,
        "error_class": None,
        "error_message": None,
        "metadata": {"packets_sent": 3, "packets_received": 3},
    }

    payload = probes._db_payload(row)

    assert payload[0] == observed_at
    assert payload[2] == "openai"
    assert payload[4] == "icmp_ping"
    assert payload[9] == 0.0
    assert payload[11] == 34.297
    assert '"packets_received": 3' in payload[22]


def test_default_endpoints_include_xai_front_doors() -> None:
    endpoints = {
        (endpoint.provider, endpoint.endpoint_key, endpoint.host)
        for endpoint in probes.DEFAULT_ENDPOINTS
    }

    assert (
        "xai",
        "cli-chat-proxy.grok.com:443",
        "cli-chat-proxy.grok.com",
    ) in endpoints
    assert ("xai", "api.x.ai:443", "api.x.ai") in endpoints


def test_default_endpoints_exclude_google_gemini_monitoring() -> None:
    endpoints = {
        (endpoint.provider, endpoint.endpoint_key, endpoint.host)
        for endpoint in probes.DEFAULT_ENDPOINTS
    }

    assert not any(provider in {"gemini", "control"} for provider, _, _ in endpoints)
    assert "generativelanguage.googleapis.com" not in {host for _, _, host in endpoints}
    assert "cloudcode-pa.googleapis.com" not in {host for _, _, host in endpoints}
    assert "google.com" not in {host for _, _, host in endpoints}
    assert ("anthropic", "api.anthropic.com:443", "api.anthropic.com") in endpoints
    assert ("openai", "api.openai.com:443", "api.openai.com") in endpoints


def test_provider_status_schema_matches_callback_schema() -> None:
    assert (
        probes.PROVIDER_STATUS_TABLE_SQL
        == aawm_agent_identity._AAWM_PROVIDER_STATUS_OBSERVATIONS_TABLE_SQL
    )
    assert (
        probes.PROVIDER_STATUS_ALTER_STATEMENTS
        == aawm_agent_identity._AAWM_PROVIDER_STATUS_OBSERVATIONS_ALTER_STATEMENTS
    )
    assert (
        probes.PROVIDER_STATUS_INDEX_STATEMENTS
        == aawm_agent_identity._AAWM_PROVIDER_STATUS_OBSERVATIONS_INDEX_STATEMENTS
    )


def test_setup_schema_executes_provider_status_ddl_with_timeouts(monkeypatch) -> None:
    fake_conn = _FakeProviderStatusConnection()
    monkeypatch.setattr(probes.psycopg, "connect", lambda _dsn: fake_conn)

    probes.setup_schema(
        "postgresql://example/db",
        lock_timeout_ms=123,
        statement_timeout_ms=456,
    )

    execute_calls = fake_conn.cursor_instance.execute_calls
    assert execute_calls[0] == (
        "SELECT set_config('application_name', %s, false)",
        ("aawm-provider-status-observations",),
    )
    assert execute_calls[1] == (
        "SELECT set_config('lock_timeout', %s, true)",
        ("123ms",),
    )
    assert execute_calls[2] == (
        "SELECT set_config('statement_timeout', %s, true)",
        ("456ms",),
    )
    ddl_statements = [statement for statement, _params in execute_calls[3:]]
    assert probes.PROVIDER_STATUS_TABLE_SQL in ddl_statements
    for statement in probes.PROVIDER_STATUS_ALTER_STATEMENTS:
        assert statement in ddl_statements
    for statement in probes.PROVIDER_STATUS_INDEX_STATEMENTS:
        assert statement in ddl_statements
    assert probes.PROVIDER_AUTH_OBSERVATIONS_TABLE_SQL in ddl_statements
    for statement in probes.PROVIDER_AUTH_OBSERVATIONS_ALTER_STATEMENTS:
        assert statement in ddl_statements
    for statement in probes.PROVIDER_AUTH_OBSERVATIONS_INDEX_STATEMENTS:
        assert statement in ddl_statements
    assert probes.PROVIDER_AUTH_CURRENT_VIEW_SQL in ddl_statements
    assert probes.PROVIDER_CREDIT_OBSERVATIONS_TABLE_SQL in ddl_statements
    for statement in probes.PROVIDER_CREDIT_OBSERVATIONS_INDEX_STATEMENTS:
        assert statement in ddl_statements
    assert probes.PROVIDER_CREDIT_CURRENT_VIEW_SQL in ddl_statements
    assert fake_conn.cursor_instance.executemany_calls == []
    assert fake_conn.commit_count == 1
    assert fake_conn.rollback_count == 0


def test_insert_observations_does_not_execute_provider_status_ddl(monkeypatch) -> None:
    fake_conn = _FakeProviderStatusConnection()
    monkeypatch.setattr(probes.psycopg, "connect", lambda _dsn: fake_conn)

    probes.insert_observations(
        "postgresql://example/db",
        [_provider_status_row()],
        lock_timeout_ms=321,
        statement_timeout_ms=654,
    )

    execute_calls = fake_conn.cursor_instance.execute_calls
    assert execute_calls == [
        (
            "SELECT set_config('application_name', %s, false)",
            ("aawm-provider-status-observations",),
        ),
        ("SELECT set_config('lock_timeout', %s, true)", ("321ms",)),
        ("SELECT set_config('statement_timeout', %s, true)", ("654ms",)),
    ]
    assert fake_conn.cursor_instance.executemany_calls
    insert_sql, payloads = fake_conn.cursor_instance.executemany_calls[0]
    assert insert_sql == probes.PROVIDER_STATUS_INSERT_SQL
    assert payloads[0][2] == "control"
    assert fake_conn.commit_count == 1
    assert fake_conn.rollback_count == 0


def _grok_billing_poll_config(**overrides):
    from dataclasses import replace

    config = loop.ProviderStatusLoopConfig(
        apply=True,
        dsn=None,
        environment="dev",
        interval_seconds=300.0,
        timeout=2.0,
        ping_count=1,
        ping_timeout=2,
        skip_icmp=False,
        once=True,
        setup_schema=False,
        db_lock_timeout_ms=1000,
        db_statement_timeout_ms=5000,
        grok_oidc_refresh_enabled=False,
        grok_oidc_auth_file="/home/zepfu/.grok/auth.json",
        grok_billing_poll_enabled=True,
        grok_billing_poll_interval_seconds=3600.0,
        grok_billing_poll_http_timeout_seconds=30.0,
        grok_billing_url="https://cli-chat-proxy.grok.com/v1/billing?format=credits",
        grok_billing_client_version="0.2.55",
        grok_billing_client_identifier="grok-cli",
        grok_billing_xai_token_auth="xai-grok-cli",
        grok_billing_model="grok-build",
        grok_billing_http_method="GET",
        grok_billing_include_model_override=True,
        grok_billing_poll_max_attempts=3,
        grok_billing_poll_retry_backoff_seconds=0.5,
    )
    if overrides:
        config = replace(config, **overrides)
    return config


def _grok_billing_payload() -> dict:
    return {
        "config": {
            "creditUsagePercent": 14.539333,
            "productUsage": [
                {"name": "GrokBuild", "usagePercent": 12.507334},
                {"name": "Api", "usagePercent": 2.032},
            ],
            "billingPeriodStart": "2026-06-01T00:00:00+00:00",
            "billingPeriodEnd": "2026-07-01T00:00:00+00:00",
        }
    }


def _grok_billing_auth_context(**overrides) -> dict:
    context = {
        "access_token": "access-token-secret",
        "identity_headers": {
            "x-userid": "user_123",
            "x-grok-user-id": "user_123",
            "x-teamid": "team_123",
            "x-email": "user@example.com",
        },
    }
    context.update(overrides)
    return context


def test_resolve_grok_sidecar_auth_file_prefers_aawm_override(tmp_path, monkeypatch) -> None:
    aawm_auth = tmp_path / "aawm-auth.json"
    native_auth = tmp_path / "native-auth.json"
    aawm_auth.write_text("{}", encoding="utf-8")
    native_auth.write_text("{}", encoding="utf-8")

    monkeypatch.setenv("AAWM_GROK_OIDC_AUTH_FILE", str(aawm_auth))
    monkeypatch.setenv("LITELLM_XAI_GROK_AUTH_FILE", str(native_auth))

    resolved_path, source = loop._resolve_grok_sidecar_auth_file(
        loop.DEFAULT_GROK_OIDC_AUTH_FILE
    )

    assert resolved_path == str(aawm_auth)
    assert source == "AAWM_GROK_OIDC_AUTH_FILE"


def test_resolve_grok_sidecar_auth_file_falls_back_to_native_precedence(tmp_path, monkeypatch) -> None:
    litellm_auth = tmp_path / "litellm-auth.json"
    oauth_auth = tmp_path / "oauth-auth.json"
    litellm_auth.write_text("{}", encoding="utf-8")
    oauth_auth.write_text("{}", encoding="utf-8")

    for env_name in (
        "AAWM_GROK_OIDC_AUTH_FILE",
        "LITELLM_XAI_GROK_AUTH_FILE",
        "LITELLM_XAI_OAUTH_GROK_AUTH_FILE",
        "GROK_AUTH_FILE",
        "GROK_HOME",
    ):
        monkeypatch.delenv(env_name, raising=False)

    monkeypatch.setenv("LITELLM_XAI_GROK_AUTH_FILE", str(litellm_auth))
    monkeypatch.setenv("LITELLM_XAI_OAUTH_GROK_AUTH_FILE", str(oauth_auth))

    resolved_path, source = loop._resolve_grok_sidecar_auth_file(
        loop.DEFAULT_GROK_OIDC_AUTH_FILE
    )

    assert resolved_path == str(litellm_auth)
    assert source == "LITELLM_XAI_GROK_AUTH_FILE"


def test_resolve_grok_sidecar_auth_file_keeps_configured_missing_path(
    tmp_path, monkeypatch
) -> None:
    missing_auth = tmp_path / "missing-auth.json"
    default_like_auth = tmp_path / "default-auth.json"
    default_like_auth.write_text("{}", encoding="utf-8")

    for env_name in (
        "AAWM_GROK_OIDC_AUTH_FILE",
        "LITELLM_XAI_GROK_AUTH_FILE",
        "LITELLM_XAI_OAUTH_GROK_AUTH_FILE",
        "GROK_AUTH_FILE",
        "GROK_HOME",
    ):
        monkeypatch.delenv(env_name, raising=False)

    monkeypatch.setenv("LITELLM_XAI_GROK_AUTH_FILE", str(missing_auth))
    monkeypatch.setenv("GROK_AUTH_FILE", str(default_like_auth))

    resolved_path, source = loop._resolve_grok_sidecar_auth_file(None)

    assert resolved_path == str(missing_auth)
    assert source == "LITELLM_XAI_GROK_AUTH_FILE"


def test_resolve_grok_sidecar_auth_file_uses_grok_home_and_default(tmp_path, monkeypatch) -> None:
    grok_home = tmp_path / "grok-home"
    grok_home.mkdir()
    grok_auth = grok_home / "auth.json"
    grok_auth.write_text("{}", encoding="utf-8")

    for env_name in (
        "AAWM_GROK_OIDC_AUTH_FILE",
        "LITELLM_XAI_GROK_AUTH_FILE",
        "LITELLM_XAI_OAUTH_GROK_AUTH_FILE",
        "GROK_AUTH_FILE",
        "GROK_HOME",
    ):
        monkeypatch.delenv(env_name, raising=False)

    monkeypatch.setenv("GROK_HOME", str(grok_home))

    resolved_path, source = loop._resolve_grok_sidecar_auth_file(
        loop.DEFAULT_GROK_OIDC_AUTH_FILE
    )

    assert resolved_path == str(grok_auth)
    assert source == "GROK_HOME"



def test_resolve_grok_billing_client_version_prefers_grok_client_version(monkeypatch) -> None:
    for env_name in (
        "AAWM_GROK_BILLING_CLIENT_VERSION",
        "LITELLM_XAI_GROK_CLIENT_VERSION",
        "GROK_CLIENT_VERSION",
    ):
        monkeypatch.delenv(env_name, raising=False)

    monkeypatch.setenv("GROK_CLIENT_VERSION", "0.2.70")

    assert loop._resolve_grok_billing_client_version() == "0.2.70"


def test_loop_config_defaults_use_grok_client_version_fallback(monkeypatch) -> None:
    for env_name in (
        "AAWM_GROK_BILLING_CLIENT_VERSION",
        "LITELLM_XAI_GROK_CLIENT_VERSION",
        "GROK_CLIENT_VERSION",
    ):
        monkeypatch.delenv(env_name, raising=False)

    monkeypatch.setenv("GROK_CLIENT_VERSION", "0.2.70")

    config = loop.parse_config([])

    assert config.grok_billing_client_version == "0.2.70"


def test_loop_config_defaults_match_container_schedule(monkeypatch) -> None:
    for env_name in (
        "AAWM_LITELLM_ENVIRONMENT",
        "AAWM_PROVIDER_STATUS_APPLY",
        "AAWM_PROVIDER_STATUS_INTERVAL_SECONDS",
        "AAWM_PROVIDER_STATUS_TIMEOUT",
        "AAWM_PROVIDER_STATUS_PING_COUNT",
        "AAWM_PROVIDER_STATUS_PING_TIMEOUT",
        "AAWM_PROVIDER_STATUS_SKIP_ICMP",
        "AAWM_PROVIDER_STATUS_ONCE",
        "AAWM_PROVIDER_STATUS_SETUP_SCHEMA_ON_START",
        "AAWM_PROVIDER_STATUS_SCHEMA_DSN",
        "AAWM_DIRECT_DATABASE_URL",
        "AAWM_PROVIDER_STATUS_REQUIRE_PGBOUNCER",
        "AAWM_PROVIDER_STATUS_DB_LOCK_TIMEOUT_MS",
        "AAWM_PROVIDER_STATUS_DB_STATEMENT_TIMEOUT_MS",
        "AAWM_GROK_BILLING_POLL_ENABLED",
        "AAWM_GROK_BILLING_POLL_INTERVAL_SECONDS",
        "AAWM_GROK_BILLING_POLL_HTTP_TIMEOUT_SECONDS",
        "AAWM_GROK_BILLING_URL",
        "AAWM_GROK_BILLING_CLIENT_VERSION",
        "AAWM_GROK_BILLING_CLIENT_IDENTIFIER",
        "AAWM_GROK_BILLING_XAI_TOKEN_AUTH",
        "AAWM_GROK_BILLING_MODEL",
        "AAWM_GROK_BILLING_HTTP_METHOD",
        "AAWM_GROK_BILLING_INCLUDE_MODEL_OVERRIDE",
        "AAWM_GROK_BILLING_POLL_MAX_ATTEMPTS",
        "AAWM_GROK_BILLING_POLL_RETRY_BACKOFF_SECONDS",
        "AAWM_OBSERVABILITY_ANOMALY_SCAN_ENABLED",
        "AAWM_OBSERVABILITY_ANOMALY_SCAN_INTERVAL_SECONDS",
        "AAWM_OBSERVABILITY_ANOMALY_SCAN_LOOKBACK_HOURS",
        "AAWM_OBSERVABILITY_ANOMALY_SCAN_ERROR_LOG_DIR",
        "LITELLM_XAI_GROK_CLIENT_VERSION",
        "LITELLM_XAI_GROK_CLIENT_IDENTIFIER",
        "LITELLM_XAI_GROK_XAI_TOKEN_AUTH",
        "LITELLM_AAWM_ERROR_LOG_DIR",
    ):
        monkeypatch.delenv(env_name, raising=False)

    config = loop.parse_config([])

    assert config.apply is True
    assert config.environment == "dev"
    assert config.interval_seconds == 300.0
    assert config.timeout == 2.0
    assert config.ping_count == 1
    assert config.ping_timeout == 2
    assert config.skip_icmp is False
    assert config.once is False
    assert config.setup_schema is False
    assert config.db_lock_timeout_ms == 1000
    assert config.db_statement_timeout_ms == 5000
    assert config.schema_dsn is None
    assert config.require_pgbouncer is False
    assert config.grok_billing_poll_enabled is False
    assert config.grok_billing_poll_interval_seconds == 3600.0
    assert config.grok_billing_poll_http_timeout_seconds == 30.0
    assert (
        config.grok_billing_url
        == "https://cli-chat-proxy.grok.com/v1/billing?format=credits"
    )
    assert config.grok_billing_client_version == "0.2.55"
    assert config.grok_billing_client_identifier == "grok-cli"
    assert config.grok_billing_xai_token_auth == "xai-grok-cli"
    assert config.grok_billing_model == "grok-build"
    assert config.grok_billing_http_method == "GET"
    assert config.grok_billing_include_model_override is True
    assert config.grok_billing_poll_max_attempts == 3
    assert config.grok_billing_poll_retry_backoff_seconds == 0.5
    assert config.observability_anomaly_scan_enabled is False
    assert config.observability_anomaly_scan_interval_seconds == 3600.0
    assert config.observability_anomaly_scan_lookback_hours == 4.0
    assert config.observability_anomaly_scan_error_log_dir == "/app/.analysis"


def test_loop_config_uses_explicit_direct_schema_dsn(monkeypatch) -> None:
    monkeypatch.setenv("AAWM_PROVIDER_STATUS_SETUP_SCHEMA_ON_START", "1")
    monkeypatch.setenv(
        "AAWM_DIRECT_DATABASE_URL",
        "postgresql://aawm:aawm_dev@postgres18:5432/aawm_tristore",
    )

    config = loop.parse_config([])

    assert config.setup_schema is True
    assert (
        config.schema_dsn
        == "postgresql://aawm:aawm_dev@postgres18:5432/aawm_tristore"
    )


def test_provider_status_compose_hardens_sidecar_db_path() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    compose_text = (repo_root / "docker-compose.dev.yml").read_text()
    dockerfile_text = (
        repo_root / "docker/Dockerfile.provider_status_observations"
    ).read_text()

    assert "container_name: aawm-provider-status-observations" in compose_text
    assert "AAWM_DB_HOST=${LITELLM_AAWM_DB_HOST:-pgbouncer}" in compose_text
    assert "AAWM_DB_PORT=${LITELLM_AAWM_DB_PORT:-6432}" in compose_text
    assert (
        "AAWM_DATABASE_URL=${LITELLM_AAWM_DATABASE_URL:-postgresql://aawm:aawm_dev@pgbouncer:6432/aawm_tristore?application_name=aawm-provider-status-observations}"
        in compose_text
    )
    assert (
        "AAWM_PROVIDER_STATUS_SCHEMA_DSN=${AAWM_PROVIDER_STATUS_SCHEMA_DSN:-postgresql://aawm:aawm_dev@postgres18:5432/aawm_tristore?application_name=aawm-provider-status-observations-schema}"
        in compose_text
    )
    assert (
        "AAWM_PROVIDER_STATUS_SETUP_SCHEMA_ON_START=${AAWM_PROVIDER_STATUS_SETUP_SCHEMA_ON_START:-0}"
        in compose_text
    )
    assert (
        "AAWM_PROVIDER_STATUS_REQUIRE_PGBOUNCER=${AAWM_PROVIDER_STATUS_REQUIRE_PGBOUNCER:-1}"
        in compose_text
    )
    assert (
        "AAWM_PROVIDER_STATUS_DB_LOCK_TIMEOUT_MS=${AAWM_PROVIDER_STATUS_DB_LOCK_TIMEOUT_MS:-1000}"
        in compose_text
    )
    assert (
        "AAWM_PROVIDER_STATUS_DB_STATEMENT_TIMEOUT_MS=${AAWM_PROVIDER_STATUS_DB_STATEMENT_TIMEOUT_MS:-5000}"
        in compose_text
    )
    assert "/home/zepfu/.grok:/home/zepfu/.grok:ro" in compose_text
    assert "/home/zepfu/.grok:/home/zepfu/.grok" in compose_text
    provider_status_block = compose_text.split("provider-status-observations:", 1)[1].split("\nnetworks:", 1)[0] if "provider-status-observations:" in compose_text else compose_text
    assert "/home/zepfu/.gemini:/home/zepfu/.gemini" not in provider_status_block
    assert "/home/zepfu/.local/bin/agy" not in provider_status_block
    assert (
        "LITELLM_XAI_GROK_AUTH_FILE=${LITELLM_XAI_GROK_AUTH_FILE:-/home/zepfu/.grok/auth.json}"
        in compose_text
    )
    assert (
        "LITELLM_ANTIGRAVITY_AUTH_FILE=${LITELLM_ANTIGRAVITY_AUTH_FILE:-/home/zepfu/.gemini/antigravity-cli/antigravity-oauth-token}"
        in compose_text
    )
    assert "LITELLM_XAI_GROK_SEED_AUTH_FILE" not in compose_text
    assert "LITELLM_XAI_GROK_AUTH_LOCK_FILE" not in compose_text
    assert (
        "AAWM_GROK_OIDC_REFRESH_ENABLED=${AAWM_GROK_OIDC_REFRESH_ENABLED:-1}"
        in compose_text
    )
    assert (
        "AAWM_GROK_OIDC_AUTH_FILE=${AAWM_GROK_OIDC_AUTH_FILE:-/home/zepfu/.grok/auth.json}"
        in compose_text
    )
    assert (
        "AAWM_GROK_OIDC_AUTH_FILE_UID=${AAWM_GROK_OIDC_AUTH_FILE_UID:-1000}"
        in compose_text
    )
    assert (
        "AAWM_GROK_OIDC_AUTH_FILE_GID=${AAWM_GROK_OIDC_AUTH_FILE_GID:-1000}"
        in compose_text
    )
    assert (
        "AAWM_GROK_OIDC_AUTH_FILE_MODE=${AAWM_GROK_OIDC_AUTH_FILE_MODE:-0o600}"
        in compose_text
    )
    assert (
        "AAWM_GROK_OIDC_REFRESH_INTERVAL_SECONDS=${AAWM_GROK_OIDC_REFRESH_INTERVAL_SECONDS:-3600}"
        in compose_text
    )
    assert "AAWM_GROK_OIDC_FORCE_REFRESH=${AAWM_GROK_OIDC_FORCE_REFRESH:-1}" in compose_text
    assert "AAWM_ANTIGRAVITY_OAUTH_REFRESH_ENABLED" not in provider_status_block
    assert "AAWM_ANTIGRAVITY_AUTH_FILE" not in provider_status_block
    assert "AAWM_ANTIGRAVITY_CLI_PATH" not in provider_status_block
    assert "- /home/zepfu/.codex:/home/zepfu/.codex" in compose_text
    for expected_codex_setting in (
        "AAWM_CODEX_OAUTH_REFRESH_ENABLED=${AAWM_CODEX_OAUTH_REFRESH_ENABLED:-1}",
        "AAWM_CODEX_AUTH_FILE=${AAWM_CODEX_AUTH_FILE:-/home/zepfu/.codex/auth.json}",
        "AAWM_CODEX_AUTH_FILE_UID=${AAWM_CODEX_AUTH_FILE_UID:-1000}",
        "AAWM_CODEX_AUTH_FILE_GID=${AAWM_CODEX_AUTH_FILE_GID:-1000}",
        "AAWM_CODEX_AUTH_FILE_MODE=${AAWM_CODEX_AUTH_FILE_MODE:-0o600}",
        "AAWM_CODEX_OAUTH_FORCE_REFRESH=${AAWM_CODEX_OAUTH_FORCE_REFRESH:-1}",
    ):
        assert expected_codex_setting in compose_text
    assert (
        "AAWM_GROK_BILLING_POLL_ENABLED=${AAWM_GROK_BILLING_POLL_ENABLED:-1}"
        in compose_text
    )
    assert (
        "AAWM_GROK_BILLING_POLL_INTERVAL_SECONDS=${AAWM_GROK_BILLING_POLL_INTERVAL_SECONDS:-3600}"
        in compose_text
    )
    assert (
        "AAWM_GROK_BILLING_POLL_HTTP_TIMEOUT_SECONDS=${AAWM_GROK_BILLING_POLL_HTTP_TIMEOUT_SECONDS:-30}"
        in compose_text
    )
    assert (
        "AAWM_GROK_BILLING_POLL_MAX_ATTEMPTS=${AAWM_GROK_BILLING_POLL_MAX_ATTEMPTS:-3}"
        in compose_text
    )
    assert (
        "AAWM_GROK_BILLING_POLL_RETRY_BACKOFF_SECONDS=${AAWM_GROK_BILLING_POLL_RETRY_BACKOFF_SECONDS:-0.5}"
        in compose_text
    )
    assert (
        "AAWM_GROK_BILLING_CLIENT_VERSION=${AAWM_GROK_BILLING_CLIENT_VERSION:-0.2.55}"
        in compose_text
    )
    assert (
        "AAWM_GROK_BILLING_CLIENT_IDENTIFIER=${AAWM_GROK_BILLING_CLIENT_IDENTIFIER:-grok-cli}"
        in compose_text
    )
    assert (
        "AAWM_GROK_BILLING_XAI_TOKEN_AUTH=${AAWM_GROK_BILLING_XAI_TOKEN_AUTH:-xai-grok-cli}"
        in compose_text
    )
    assert (
        "AAWM_GROK_BILLING_HTTP_METHOD=${AAWM_GROK_BILLING_HTTP_METHOD:-GET}"
        in compose_text
    )
    assert (
        "AAWM_GROK_BILLING_INCLUDE_MODEL_OVERRIDE=${AAWM_GROK_BILLING_INCLUDE_MODEL_OVERRIDE:-1}"
        in compose_text
    )
    assert "antigravity_oauth_refresh.py" not in dockerfile_text
    assert (
        "COPY scripts/codex_oauth_refresh.py "
        "/app/scripts/codex_oauth_refresh.py"
    ) in dockerfile_text


def test_provider_status_compose_wires_observability_anomaly_scan() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    compose_text = (repo_root / "docker-compose.dev.yml").read_text()

    assert "- ./.analysis:/app/.analysis" in compose_text
    assert (
        "AAWM_OBSERVABILITY_ANOMALY_SCAN_ENABLED=${AAWM_OBSERVABILITY_ANOMALY_SCAN_ENABLED:-1}"
        in compose_text
    )
    assert (
        "AAWM_OBSERVABILITY_ANOMALY_SCAN_INTERVAL_SECONDS=${AAWM_OBSERVABILITY_ANOMALY_SCAN_INTERVAL_SECONDS:-3600}"
        in compose_text
    )
    assert (
        "AAWM_OBSERVABILITY_ANOMALY_SCAN_LOOKBACK_HOURS=${AAWM_OBSERVABILITY_ANOMALY_SCAN_LOOKBACK_HOURS:-4}"
        in compose_text
    )
    assert (
        "AAWM_OBSERVABILITY_ANOMALY_SCAN_ERROR_LOG_DIR=${AAWM_OBSERVABILITY_ANOMALY_SCAN_ERROR_LOG_DIR:-/app/.analysis}"
        in compose_text
    )


def test_provider_status_compose_wires_managed_xai_oauth_sidecar_refresh() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    compose_text = (repo_root / "docker-compose.dev.yml").read_text()
    dockerfile_text = (
        repo_root / "docker/Dockerfile.provider_status_observations"
    ).read_text()

    assert "- /home/zepfu/.litellm/xai:/home/zepfu/.litellm/xai:ro" in compose_text
    assert "- /home/zepfu/.litellm/xai:/home/zepfu/.litellm/xai" in compose_text
    assert (
        "AAWM_XAI_OAUTH_REFRESH_ENABLED=${AAWM_XAI_OAUTH_REFRESH_ENABLED:-1}"
        in compose_text
    )
    assert (
        "AAWM_XAI_OAUTH_AUTH_FILE=${AAWM_XAI_OAUTH_AUTH_FILE:-/home/zepfu/.litellm/xai/oauth-auth.json}"
        in compose_text
    )
    assert (
        "AAWM_XAI_OAUTH_FORCE_REFRESH=${AAWM_XAI_OAUTH_FORCE_REFRESH:-1}"
        in compose_text
    )
    assert (
        "COPY scripts/xai_oauth_refresh.py "
        "/app/scripts/xai_oauth_refresh.py"
    ) in dockerfile_text


def test_run_cycle_inserts_rows_and_returns_summary(monkeypatch) -> None:
    config = loop.ProviderStatusLoopConfig(
        apply=True,
        dsn=None,
        environment="dev",
        interval_seconds=300.0,
        timeout=2.0,
        ping_count=1,
        ping_timeout=2,
        skip_icmp=False,
        once=True,
        setup_schema=True,
        db_lock_timeout_ms=1000,
        db_statement_timeout_ms=5000,
    )
    rows = [
        {
            "provider": "control",
            "endpoint_key": "api.openai.com:443",
            "probe_type": "dns",
            "success": True,
        },
        {
            "provider": "anthropic",
            "endpoint_key": "api.anthropic.com:443",
            "probe_type": "tls_handshake",
            "success": False,
            "error_class": "tls_error",
            "error_message": "handshake failed with api_key=sk-testsecret1234567890",
        },
    ]
    inserted = {}

    def fake_collect_observations(endpoints, **kwargs):
        assert endpoints == probes.DEFAULT_ENDPOINTS
        assert kwargs == {
            "environment": "dev",
            "timeout": 2.0,
            "ping_count": 1,
            "ping_timeout": 2,
            "skip_icmp": False,
        }
        return rows

    def fake_insert_observations(
        dsn,
        payload_rows,
        *,
        lock_timeout_ms,
        statement_timeout_ms,
    ):
        inserted["dsn"] = dsn
        inserted["rows"] = payload_rows
        inserted["lock_timeout_ms"] = lock_timeout_ms
        inserted["statement_timeout_ms"] = statement_timeout_ms

    monkeypatch.setattr(loop.probes, "collect_observations", fake_collect_observations)
    monkeypatch.setattr(loop.probes, "_build_dsn", lambda _args: "postgresql://example/db")
    monkeypatch.setattr(loop.probes, "insert_observations", fake_insert_observations)
    monkeypatch.setattr(
        loop.probes,
        "setup_schema",
        lambda *_args, **_kwargs: pytest.fail("run_cycle must not run schema setup"),
    )

    summary = loop.run_cycle(config)

    assert inserted == {
        "dsn": "postgresql://example/db",
        "rows": rows,
        "lock_timeout_ms": 1000,
        "statement_timeout_ms": 5000,
    }
    assert summary["event"] == "provider_status_observations_cycle"
    assert summary["apply"] is True
    assert summary["inserted"] is True
    assert summary["skipped"] is False
    assert summary["environment"] == "dev"
    assert summary["row_count"] == 2
    assert summary["success_count"] == 1
    assert summary["failure_count"] == 1
    assert summary["failure_summaries"] == [
        {
            "provider": "anthropic",
            "endpoint_key": "api.anthropic.com:443",
            "probe_type": "tls_handshake",
            "error_class": "tls_error",
            "error_message": "handshake failed with REDACTED",
        }
    ]
    assert summary["failure_summaries_omitted_count"] == 0


def test_run_due_sidecar_tasks_skips_when_grok_oidc_refresh_disabled(monkeypatch) -> None:
    config = loop.ProviderStatusLoopConfig(
        apply=False,
        dsn=None,
        environment="dev",
        interval_seconds=300.0,
        timeout=2.0,
        ping_count=1,
        ping_timeout=2,
        skip_icmp=False,
        once=True,
        setup_schema=False,
        db_lock_timeout_ms=1000,
        db_statement_timeout_ms=5000,
        grok_oidc_refresh_enabled=False,
    )

    monkeypatch.setattr(
        loop.grok_oidc_refresh,
        "refresh_grok_oidc_auth_file",
        lambda *_args, **_kwargs: pytest.fail("Grok OIDC refresh should not run"),
    )
    monkeypatch.setattr(
        loop.grok_oidc_refresh,
        "repair_grok_oidc_auth_file_metadata",
        lambda *_args, **_kwargs: pytest.fail("Grok OIDC metadata repair should not run"),
    )

    assert loop.run_due_sidecar_tasks(
        config,
        loop.SidecarTaskState(),
        now_monotonic=100.0,
    ) == []


def test_run_due_sidecar_tasks_runs_grok_oidc_refresh_when_due(monkeypatch) -> None:
    config = loop.ProviderStatusLoopConfig(
        apply=False,
        dsn=None,
        environment="dev",
        interval_seconds=300.0,
        timeout=2.0,
        ping_count=1,
        ping_timeout=2,
        skip_icmp=False,
        once=True,
        setup_schema=False,
        db_lock_timeout_ms=1000,
        db_statement_timeout_ms=5000,
        grok_oidc_refresh_enabled=True,
        grok_oidc_auth_file="/home/zepfu/.grok/auth.json",
        grok_oidc_lock_file="/home/zepfu/.grok/auth.json.lock",
        grok_oidc_refresh_interval_seconds=3600.0,
        grok_oidc_refresh_buffer_seconds=300,
        grok_oidc_force_refresh=True,
        grok_oidc_http_timeout_seconds=30.0,
    )
    calls = []

    def fake_refresh(*args, **kwargs):
        calls.append((args, kwargs))
        return {
            "attempted": True,
            "refreshed": True,
            "skipped": False,
            "auth_file": "/home/zepfu/.grok/auth.json",
            "scope": "scope",
            "expires_at": "2026-06-16T22:00:00Z",
            "error_class": None,
            "error_message": None,
        }

    monkeypatch.setattr(
        loop.grok_oidc_refresh,
        "refresh_grok_oidc_auth_file",
        fake_refresh,
    )
    monkeypatch.setattr(
        loop.grok_oidc_refresh,
        "repair_grok_oidc_auth_file_metadata",
        lambda *_args, **_kwargs: {
            "attempted": True,
            "repaired": False,
            "auth_file": "/home/zepfu/.grok/auth.json",
            "error_class": None,
            "error_message": None,
        },
    )

    state = loop.SidecarTaskState()
    events = loop.run_due_sidecar_tasks(config, state, now_monotonic=100.0)
    second_events = loop.run_due_sidecar_tasks(config, state, now_monotonic=200.0)
    third_events = loop.run_due_sidecar_tasks(config, state, now_monotonic=3701.0)

    assert len(calls) == 2
    assert calls[0] == (
        ("/home/zepfu/.grok/auth.json",),
        {
            "buffer_seconds": 300,
            "force": True,
            "lock_file": "/home/zepfu/.grok/auth.json.lock",
            "http_timeout_seconds": 30.0,
        },
    )
    assert events[0]["event"] == "grok_oidc_refresh"
    assert events[0]["environment"] == "dev"
    assert events[0]["refreshed"] is True
    assert "access-token" not in str(events)
    assert second_events == []
    assert third_events[0]["event"] == "grok_oidc_refresh"


def test_run_due_sidecar_tasks_repairs_grok_oidc_metadata_each_cycle(
    monkeypatch,
) -> None:
    config = loop.ProviderStatusLoopConfig(
        apply=False,
        dsn=None,
        environment="dev",
        interval_seconds=300.0,
        timeout=2.0,
        ping_count=1,
        ping_timeout=2,
        skip_icmp=False,
        once=True,
        setup_schema=False,
        db_lock_timeout_ms=1000,
        db_statement_timeout_ms=5000,
        grok_oidc_refresh_enabled=True,
        grok_oidc_auth_file="/home/zepfu/.grok/auth.json",
        grok_oidc_lock_file="/home/zepfu/.grok/auth.json.lock",
        grok_oidc_refresh_interval_seconds=3600.0,
        grok_oidc_refresh_buffer_seconds=300,
        grok_oidc_force_refresh=True,
        grok_oidc_http_timeout_seconds=30.0,
    )
    repair_calls = []

    def fake_repair(*args, **kwargs):
        repair_calls.append((args, kwargs))
        return {
            "attempted": True,
            "repaired": True,
            "auth_file": "/home/zepfu/.grok/auth.json",
            "error_class": None,
            "error_message": None,
        }

    monkeypatch.setattr(
        loop.grok_oidc_refresh,
        "repair_grok_oidc_auth_file_metadata",
        fake_repair,
    )
    monkeypatch.setattr(
        loop.grok_oidc_refresh,
        "refresh_grok_oidc_auth_file",
        lambda *_args, **_kwargs: {
            "attempted": True,
            "refreshed": True,
            "skipped": False,
            "auth_file": "/home/zepfu/.grok/auth.json",
            "scope": "scope",
            "expires_at": "2026-06-16T22:00:00Z",
            "error_class": None,
            "error_message": None,
        },
    )

    state = loop.SidecarTaskState(grok_oidc_last_attempt_monotonic=100.0)
    events = loop.run_due_sidecar_tasks(config, state, now_monotonic=200.0)

    assert repair_calls == [
        (
            ("/home/zepfu/.grok/auth.json",),
            {"lock_file": "/home/zepfu/.grok/auth.json.lock"},
        )
    ]
    assert events == [
        {
            "event": "grok_oidc_metadata_repair",
            "observed_at": events[0]["observed_at"],
            "environment": "dev",
            "attempted": True,
            "repaired": True,
            "auth_file": "/home/zepfu/.grok/auth.json",
            "error_class": None,
            "error_message": None,
        }
    ]


def test_run_cycle_requires_dsn_when_apply_enabled(monkeypatch) -> None:
    config = loop.ProviderStatusLoopConfig(
        apply=True,
        dsn=None,
        environment="dev",
        interval_seconds=300.0,
        timeout=2.0,
        ping_count=1,
        ping_timeout=2,
        skip_icmp=False,
        once=True,
        setup_schema=True,
        db_lock_timeout_ms=1000,
        db_statement_timeout_ms=5000,
    )

    monkeypatch.setattr(loop.probes, "collect_observations", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(loop.probes, "_build_dsn", lambda _args: None)

    with pytest.raises(RuntimeError, match="No database DSN found"):
        loop.run_cycle(config)


def test_run_cycle_omits_failure_summaries_for_green_cycle(monkeypatch) -> None:
    config = loop.ProviderStatusLoopConfig(
        apply=False,
        dsn=None,
        environment="dev",
        interval_seconds=300.0,
        timeout=2.0,
        ping_count=1,
        ping_timeout=2,
        skip_icmp=False,
        once=True,
        setup_schema=False,
        db_lock_timeout_ms=1000,
        db_statement_timeout_ms=5000,
    )
    rows = [
        {
            "provider": "control",
            "endpoint_key": "api.openai.com:443",
            "probe_type": "dns",
            "success": True,
        }
    ]

    monkeypatch.setattr(loop.probes, "collect_observations", lambda *_args, **_kwargs: rows)

    summary = loop.run_cycle(config)

    assert summary["failure_count"] == 0
    assert "failure_summaries" not in summary
    assert "failure_summaries_omitted_count" not in summary


def test_provider_failure_summaries_are_bounded_and_redacted() -> None:
    secret = "sk-abcdefghijklmnopqrstuvwxyz123456"
    rows = [
        {
            "provider": f"provider-{index}",
            "endpoint_key": f"160.79.104.{index}:443",
            "probe_type": "dns",
            "success": False,
            "error_class": "dns_error",
            "error_message": (
                f"PING api.anthropic.com (160.79.104.{index}) failure {index} "
                f"token={secret} ipv6=2001:db8::{index} "
                + ("extra detail " * 40)
            ),
            "metadata": {"raw_payload": secret},
            "resolved_ip": "203.0.113.10",
        }
        for index in range(loop.PROVIDER_FAILURE_SUMMARY_LIMIT + 2)
    ]

    summaries, omitted_count = loop._provider_failure_summaries(rows)

    assert len(summaries) == loop.PROVIDER_FAILURE_SUMMARY_LIMIT
    assert omitted_count == 2
    assert all(
        set(summary) == {
            "provider",
            "endpoint_key",
            "probe_type",
            "error_class",
            "error_message",
        }
        for summary in summaries
    )
    assert summaries[0]["provider"] == "provider-0"
    assert summaries[0]["endpoint_key"] == "REDACTED:443"
    assert summaries[0]["probe_type"] == "dns"
    assert summaries[0]["error_class"] == "dns_error"
    assert "REDACTED" in summaries[0]["error_message"]
    assert secret not in str(summaries)
    assert "160.79.104" not in str(summaries)
    assert "2001:db8" not in str(summaries)
    assert "raw_payload" not in str(summaries)
    assert "resolved_ip" not in str(summaries)
    assert len(summaries[0]["error_message"]) <= loop.PROVIDER_FAILURE_MESSAGE_LIMIT


def test_run_cycle_skips_database_timeout_without_raising(monkeypatch) -> None:
    config = loop.ProviderStatusLoopConfig(
        apply=True,
        dsn=None,
        environment="dev",
        interval_seconds=300.0,
        timeout=2.0,
        ping_count=1,
        ping_timeout=2,
        skip_icmp=False,
        once=True,
        setup_schema=True,
        db_lock_timeout_ms=1000,
        db_statement_timeout_ms=5000,
    )
    rows = [{"provider": "control", "probe_type": "dns", "success": True}]

    def fake_insert_observations(*_args, **_kwargs):
        raise probes.ProviderStatusDatabaseWriteSkipped(
            error_class="LockNotAvailable",
            message="canceling statement due to lock timeout",
        )

    monkeypatch.setattr(loop.probes, "collect_observations", lambda *_args, **_kwargs: rows)
    monkeypatch.setattr(loop.probes, "_build_dsn", lambda _args: "postgresql://example/db")
    monkeypatch.setattr(loop.probes, "insert_observations", fake_insert_observations)

    summary = loop.run_cycle(config)

    assert summary["event"] == "provider_status_observations_cycle"
    assert summary["inserted"] is False
    assert summary["skipped"] is True
    assert summary["skip_error_class"] == "LockNotAvailable"
    assert "lock timeout" in summary["skip_reason"]


def test_setup_schema_once_returns_ready_summary(monkeypatch) -> None:
    config = loop.ProviderStatusLoopConfig(
        apply=True,
        dsn=None,
        environment="dev",
        interval_seconds=300.0,
        timeout=2.0,
        ping_count=1,
        ping_timeout=2,
        skip_icmp=False,
        once=True,
        setup_schema=True,
        db_lock_timeout_ms=111,
        db_statement_timeout_ms=222,
        schema_dsn="postgresql://aawm:aawm_dev@postgres18:5432/aawm_tristore",
    )
    called = {}

    def fake_setup_schema(
        dsn,
        *,
        lock_timeout_ms,
        statement_timeout_ms,
    ):
        called["dsn"] = dsn
        called["lock_timeout_ms"] = lock_timeout_ms
        called["statement_timeout_ms"] = statement_timeout_ms

    monkeypatch.setattr(
        loop.probes,
        "_build_dsn",
        lambda _args: pytest.fail("schema setup must not use steady-state DSN"),
    )
    monkeypatch.setattr(loop.probes, "setup_schema", fake_setup_schema)

    summary = loop.setup_schema_once(config)

    assert called == {
        "dsn": (
            "postgresql://aawm:aawm_dev@postgres18:5432/aawm_tristore"
            "?application_name=aawm-provider-status-observations"
        ),
        "lock_timeout_ms": 111,
        "statement_timeout_ms": 222,
    }
    assert summary["event"] == "provider_status_observations_schema_ready"
    assert summary["environment"] == "dev"


def test_setup_schema_once_reports_skipped_lock_timeout(monkeypatch) -> None:
    config = loop.ProviderStatusLoopConfig(
        apply=True,
        dsn=None,
        environment="dev",
        interval_seconds=300.0,
        timeout=2.0,
        ping_count=1,
        ping_timeout=2,
        skip_icmp=False,
        once=True,
        setup_schema=True,
        db_lock_timeout_ms=111,
        db_statement_timeout_ms=222,
        schema_dsn="postgresql://aawm:aawm_dev@postgres18:5432/aawm_tristore",
    )

    def fake_setup_schema(*_args, **_kwargs):
        raise probes.ProviderStatusDatabaseWriteSkipped(
            error_class="QueryCanceled",
            message="canceling statement due to statement timeout",
        )

    monkeypatch.setattr(loop.probes, "setup_schema", fake_setup_schema)

    summary = loop.setup_schema_once(config)

    assert summary["event"] == "provider_status_observations_schema_skipped"
    assert summary["environment"] == "dev"
    assert summary["error_class"] == "QueryCanceled"
    assert "statement timeout" in summary["error_message"]


def test_setup_schema_once_requires_direct_schema_dsn() -> None:
    config = loop.ProviderStatusLoopConfig(
        apply=True,
        dsn=None,
        environment="dev",
        interval_seconds=300.0,
        timeout=2.0,
        ping_count=1,
        ping_timeout=2,
        skip_icmp=False,
        once=True,
        setup_schema=True,
        db_lock_timeout_ms=111,
        db_statement_timeout_ms=222,
    )

    with pytest.raises(RuntimeError, match="schema setup requires"):
        loop.setup_schema_once(config)


def test_validate_runtime_guardrails_requires_pgbouncer_when_enabled() -> None:
    config = loop.ProviderStatusLoopConfig(
        apply=True,
        dsn="postgresql://aawm:aawm_dev@postgres18:5432/aawm_tristore",
        environment="dev",
        interval_seconds=300.0,
        timeout=2.0,
        ping_count=1,
        ping_timeout=2,
        skip_icmp=False,
        once=True,
        setup_schema=False,
        db_lock_timeout_ms=111,
        db_statement_timeout_ms=222,
        require_pgbouncer=True,
    )

    with pytest.raises(RuntimeError, match="pgbouncer:6432"):
        loop.validate_runtime_guardrails(config)


def test_validate_runtime_guardrails_accepts_pgbouncer_when_required() -> None:
    config = loop.ProviderStatusLoopConfig(
        apply=True,
        dsn="postgresql://aawm:aawm_dev@pgbouncer:6432/aawm_tristore",
        environment="dev",
        interval_seconds=300.0,
        timeout=2.0,
        ping_count=1,
        ping_timeout=2,
        skip_icmp=False,
        once=True,
        setup_schema=False,
        db_lock_timeout_ms=111,
        db_statement_timeout_ms=222,
        require_pgbouncer=True,
    )

    loop.validate_runtime_guardrails(config)


def test_loop_config_reads_grok_billing_poll_env_defaults(monkeypatch) -> None:
    monkeypatch.setenv("AAWM_GROK_BILLING_POLL_ENABLED", "1")
    monkeypatch.setenv("AAWM_GROK_BILLING_POLL_INTERVAL_SECONDS", "7200")
    monkeypatch.setenv("AAWM_GROK_BILLING_POLL_HTTP_TIMEOUT_SECONDS", "45")
    monkeypatch.setenv(
        "AAWM_GROK_BILLING_URL",
        "https://cli-chat-proxy.grok.com/v1/billing?format=credits&lane=dev",
    )
    monkeypatch.setenv("AAWM_GROK_BILLING_CLIENT_VERSION", "0.2.60")
    monkeypatch.setenv("AAWM_GROK_BILLING_CLIENT_IDENTIFIER", "grok-cli-dev")
    monkeypatch.setenv("AAWM_GROK_BILLING_XAI_TOKEN_AUTH", "xai-grok-cli-dev")
    monkeypatch.setenv("AAWM_GROK_BILLING_MODEL", "grok-composer-2.5-fast")
    monkeypatch.setenv("AAWM_GROK_BILLING_INCLUDE_MODEL_OVERRIDE", "1")
    monkeypatch.setenv("AAWM_GROK_BILLING_POLL_MAX_ATTEMPTS", "5")
    monkeypatch.setenv("AAWM_GROK_BILLING_POLL_RETRY_BACKOFF_SECONDS", "1.25")

    config = loop.parse_config([])

    assert config.grok_billing_poll_enabled is True
    assert config.grok_billing_poll_interval_seconds == 7200.0
    assert config.grok_billing_poll_http_timeout_seconds == 45.0
    assert (
        config.grok_billing_url
        == "https://cli-chat-proxy.grok.com/v1/billing?format=credits&lane=dev"
    )
    assert config.grok_billing_client_version == "0.2.60"
    assert config.grok_billing_client_identifier == "grok-cli-dev"
    assert config.grok_billing_xai_token_auth == "xai-grok-cli-dev"
    assert config.grok_billing_model == "grok-composer-2.5-fast"
    assert config.grok_billing_include_model_override is True
    assert config.grok_billing_poll_max_attempts == 5
    assert config.grok_billing_poll_retry_backoff_seconds == 1.25


def test_run_due_sidecar_tasks_skips_when_grok_billing_poll_disabled(monkeypatch) -> None:
    config = _grok_billing_poll_config(grok_billing_poll_enabled=False)

    monkeypatch.setattr(
        loop,
        "_fetch_grok_billing_payload",
        lambda *_args, **_kwargs: pytest.fail("Grok billing poll should not run"),
    )

    assert loop.run_due_sidecar_tasks(
        config,
        loop.SidecarTaskState(),
        now_monotonic=100.0,
    ) == []


def test_run_due_sidecar_tasks_throttles_grok_billing_poll(monkeypatch) -> None:
    config = _grok_billing_poll_config(apply=False)
    calls = {"fetch": 0, "persist": 0}

    monkeypatch.setattr(
        loop,
        "_fetch_grok_billing_payload",
        lambda *_args, **_kwargs: (
            calls.__setitem__("fetch", calls["fetch"] + 1)
            or {"status_code": 200, "payload": _grok_billing_payload()}
        ),
    )
    monkeypatch.setattr(
        loop,
        "_persist_grok_billing_observations",
        lambda *_args, **_kwargs: (
            calls.__setitem__("persist", calls["persist"] + 1) or 1
        ),
    )

    state = loop.SidecarTaskState()
    first_events = loop.run_due_sidecar_tasks(config, state, now_monotonic=100.0)
    second_events = loop.run_due_sidecar_tasks(config, state, now_monotonic=200.0)
    third_events = loop.run_due_sidecar_tasks(config, state, now_monotonic=3701.0)

    assert calls == {"fetch": 2, "persist": 0}
    assert first_events[0]["event"] == "grok_billing_poll"
    assert first_events[0]["status_code"] == 200
    assert first_events[0]["observation_count"] == 1
    assert first_events[0]["inserted_count"] == 0
    assert first_events[0]["persisted"] is False
    assert second_events == []
    assert third_events[0]["event"] == "grok_billing_poll"


def test_loop_config_reads_grok_billing_http_method_override(monkeypatch) -> None:
    monkeypatch.setenv("AAWM_GROK_BILLING_HTTP_METHOD", "post")

    config = loop.parse_config([])

    assert config.grok_billing_http_method == "POST"


def test_loop_config_reads_observability_anomaly_scan_env_defaults(monkeypatch) -> None:
    monkeypatch.setenv("AAWM_OBSERVABILITY_ANOMALY_SCAN_ENABLED", "1")
    monkeypatch.setenv("AAWM_OBSERVABILITY_ANOMALY_SCAN_INTERVAL_SECONDS", "1800")
    monkeypatch.setenv("AAWM_OBSERVABILITY_ANOMALY_SCAN_LOOKBACK_HOURS", "6")
    monkeypatch.setenv(
        "AAWM_OBSERVABILITY_ANOMALY_SCAN_ERROR_LOG_DIR",
        "/tmp/aawm-errors",
    )

    config = loop.parse_config([])

    assert config.observability_anomaly_scan_enabled is True
    assert config.observability_anomaly_scan_interval_seconds == 1800.0
    assert config.observability_anomaly_scan_lookback_hours == 6.0
    assert config.observability_anomaly_scan_error_log_dir == "/tmp/aawm-errors"


def test_grok_billing_request_contract_summary_includes_safe_diagnostics() -> None:
    config = _grok_billing_poll_config(
        grok_billing_url="https://cli-chat-proxy.grok.com/v1/billing?format=credits",
        grok_billing_http_method="GET",
    )

    summary = loop._grok_billing_request_contract_summary(
        config,
        identity_headers=_grok_billing_auth_context()["identity_headers"],
    )

    assert summary["http_client"] == "urllib"
    assert summary["request_method"] == "GET"
    assert summary["billing_host"] == "cli-chat-proxy.grok.com"
    assert summary["billing_path"] == "/v1/billing"
    assert summary["billing_query_keys"] == ["format"]
    assert summary["billing_query_present"] is True
    assert summary["include_model_override"] is True
    assert summary["model_override_configured"] is True
    assert summary["client_identifier"] == "grok-cli"
    assert summary["client_version"] == "0.2.55"
    assert summary["x_xai_token_auth_configured"] is True
    assert summary["resolved_auth_file"] == "/home/zepfu/.grok/auth.json"
    assert summary["auth_file_source"] == "default"
    assert summary["poll_max_attempts"] == 3
    assert "authorization" in summary["header_names"]
    assert "x-userid" in summary["header_names"]
    assert len(summary["request_contract_fingerprint"]) == 64
    assert "xai-grok-cli" not in json.dumps(summary)


def test_run_due_sidecar_tasks_grok_billing_event_includes_safe_diagnostics(
    monkeypatch,
) -> None:
    config = _grok_billing_poll_config()

    monkeypatch.setattr(
        loop,
        "_fetch_grok_billing_payload",
        lambda *_args, **_kwargs: {
            "status_code": 200,
            "payload": _grok_billing_payload(),
            "attempt_count": 1,
            "retry_count": 0,
        },
    )
    monkeypatch.setattr(
        loop,
        "_load_grok_billing_auth_context",
        lambda _path: _grok_billing_auth_context(),
    )
    monkeypatch.setattr(
        loop,
        "_persist_grok_billing_observations",
        lambda *_args, **_kwargs: (1, 1),
    )

    events = loop.run_due_sidecar_tasks(
        config,
        loop.SidecarTaskState(),
        now_monotonic=100.0,
    )

    event = events[0]
    assert event["http_client"] == "urllib"
    assert event["request_method"] == "GET"
    assert event["billing_host"] == "cli-chat-proxy.grok.com"
    assert event["billing_path"] == "/v1/billing"
    assert event["billing_query_keys"] == ["format"]
    assert event["billing_query_present"] is True
    assert event["model_override_configured"] is True
    assert event["resolved_auth_file"] == "/home/zepfu/.grok/auth.json"
    assert event["auth_file_source"] == "default"
    assert event["poll_max_attempts"] == 3
    assert event["request_contract_fingerprint"]
    assert "access-token-secret" not in json.dumps(events)
    assert "xai-grok-cli" not in json.dumps(events)
    assert "user_123" not in json.dumps(events)
    assert "team_123" not in json.dumps(events)
    assert "user@example.com" not in json.dumps(events)


def test_run_due_sidecar_tasks_persists_grok_billing_snapshot(monkeypatch) -> None:
    config = _grok_billing_poll_config()
    captured = {}

    monkeypatch.setattr(
        loop,
        "_fetch_grok_billing_payload",
        lambda *_args, **_kwargs: {
            "status_code": 200,
            "payload": _grok_billing_payload(),
        },
    )
    monkeypatch.setattr(
        loop,
        "_load_grok_billing_auth_context",
        lambda _path: _grok_billing_auth_context(),
    )

    def fake_persist(cfg, *, observed_at, response_body, identity_headers=None):
        captured["config"] = cfg
        captured["observed_at"] = observed_at
        captured["response_body"] = response_body
        captured["identity_headers"] = identity_headers
        return 1, 1

    monkeypatch.setattr(loop, "_persist_grok_billing_observations", fake_persist)

    events = loop.run_due_sidecar_tasks(
        config,
        loop.SidecarTaskState(),
        now_monotonic=100.0,
    )

    assert captured["config"] is config
    assert captured["response_body"] == _grok_billing_payload()
    assert captured["identity_headers"] == _grok_billing_auth_context()[
        "identity_headers"
    ]
    assert events[0]["event"] == "grok_billing_poll"
    assert events[0]["persisted"] is True
    assert events[0]["observation_count"] == 1
    assert events[0]["inserted_count"] == 1
    assert events[0]["status_code"] == 200
    assert "access-token" not in json.dumps(events)


def test_run_due_sidecar_tasks_redacts_grok_billing_poll_failure(monkeypatch) -> None:
    config = _grok_billing_poll_config()

    def fake_fetch(_config):
        raise ValueError(
            "Grok billing poll failed with Authorization=Bearer secret-token "
            "and client_secret=super-secret and x_xai_token_auth=secret-xai-auth"
        )

    monkeypatch.setattr(loop, "_fetch_grok_billing_payload", fake_fetch)

    events = loop.run_due_sidecar_tasks(
        config,
        loop.SidecarTaskState(),
        now_monotonic=100.0,
    )

    assert events[0]["event"] == "grok_billing_poll"
    assert events[0]["persisted"] is False
    assert events[0]["error_class"] == "ValueError"
    assert "REDACTED" in events[0]["error_message"]
    assert "secret-token" not in json.dumps(events)
    assert "super-secret" not in json.dumps(events)
    assert "secret-xai-auth" not in json.dumps(events)


def test_run_due_sidecar_tasks_grok_billing_event_does_not_emit_identity_fields(
    monkeypatch,
) -> None:
    config = _grok_billing_poll_config()

    def fake_fetch(_config):
        raise ValueError(
            "Grok billing poll failed with x-userid=user_123 "
            "x-grok-user-id=user_123 x-teamid=team_123 x-email=user@example.com"
        )

    monkeypatch.setattr(loop, "_fetch_grok_billing_payload", fake_fetch)

    events = loop.run_due_sidecar_tasks(
        config,
        loop.SidecarTaskState(),
        now_monotonic=100.0,
    )

    assert events[0]["event"] == "grok_billing_poll"
    assert events[0]["persisted"] is False
    assert events[0]["error_class"] == "ValueError"
    assert set(events[0].keys()) <= {
        "event",
        "observed_at",
        "environment",
        "attempted",
        "persisted",
        "skipped",
        "auth_file",
        "resolved_auth_file",
        "auth_file_source",
        "billing_url",
        "client_version",
        "model",
        "observation_count",
        "inserted_count",
        "status_code",
        "attempt_count",
        "retry_count",
        "poll_max_attempts",
        "http_client",
        "request_method",
        "billing_host",
        "billing_path",
        "billing_query_keys",
        "billing_query_present",
        "header_names",
        "include_model_override",
        "model_override_configured",
        "client_identifier",
        "client_version",
        "request_contract_fingerprint",
        "x_xai_token_auth_configured",
        "error_class",
        "error_message",
    }
    assert "identity_headers" not in json.dumps(events)
    assert '"user_id"' not in json.dumps(events)
    assert '"team_id"' not in json.dumps(events)
    assert '"email"' not in json.dumps(events)
    assert "user_123" not in json.dumps(events)
    assert "team_123" not in json.dumps(events)
    assert "user@example.com" not in json.dumps(events)
    assert "REDACTED" in events[0]["error_message"]


def test_grok_billing_sidecar_payload_maps_percentage_snapshot() -> None:
    config = _grok_billing_poll_config(grok_billing_model="grok-composer-2.5-fast")
    observed_at = datetime(2026, 6, 16, 20, 4, tzinfo=timezone.utc)
    identity_headers = _grok_billing_auth_context()["identity_headers"]

    payload = loop._build_grok_billing_rate_limit_payload(
        config,
        observed_at=observed_at,
        response_body=_grok_billing_payload(),
        identity_headers=identity_headers,
    )

    assert payload[0] == observed_at
    assert payload[1] == "grok-build"
    assert payload[2] == "0.2.55"
    assert payload[3] is None
    assert payload[4] == "xai"
    assert payload[5] == "grok-composer-2.5-fast"
    assert payload[6] == "xai_grok_build_monthly_credits:credits"
    assert payload[7] == "monthly"
    assert payload[8] == "credits"
    assert payload[9] == datetime(2026, 7, 1, tzinfo=timezone.utc)
    assert payload[10] == pytest.approx(85.460667)
    assert payload[11] is None
    assert payload[12] is None
    assert payload[13] is None
    assert payload[14] == datetime(2026, 6, 1, tzinfo=timezone.utc)
    assert payload[15] == datetime(2026, 7, 1, tzinfo=timezone.utc)
    raw_provider_fields = json.loads(payload[16])
    assert raw_provider_fields["creditUsagePercent"] == pytest.approx(14.539333)
    assert raw_provider_fields["productUsage"][0]["name"] == "GrokBuild"
    assert raw_provider_fields["quota_unit"] == "grok_billing_credit_usage_percent"
    evidence = json.loads(payload[17])
    assert evidence["signals"] == [
        "grok_billing_payload",
        "grok_billing_percentage_only",
        "grok_billing_sidecar_request_contract",
    ]
    assert len(evidence["request_contract_fingerprint"]) == 64
    assert evidence["request_contract_source"] == "grok_billing_sidecar_poll"
    assert evidence["request_contract_method"] == "GET"
    assert evidence["request_contract_target_host"] == "cli-chat-proxy.grok.com"
    assert evidence["request_contract_target_path"] == "/v1/billing"
    assert evidence["request_contract_http_client"] == "urllib"
    assert "x-userid" in evidence["request_contract_header_names"]
    assert "authorization" in evidence["request_contract_header_names"]
    assert evidence["request_contract_x_xai_token_auth_configured"] is True
    evidence_json = json.dumps(evidence)
    assert "user_123" not in evidence_json
    assert "team_123" not in evidence_json
    assert "user@example.com" not in evidence_json
    assert "access-token-secret" not in evidence_json
    assert "xai-grok-cli" not in evidence_json
    assert payload[18] == "grok_billing"
    assert payload[19] is None
    assert payload[20] is None
    assert payload[21] == "grok-billing-poll-20260616200400"


def test_grok_billing_sidecar_payload_maps_period_only_snapshot() -> None:
    config = _grok_billing_poll_config()
    observed_at = datetime(2026, 6, 30, 12, 0, tzinfo=timezone.utc)
    response_body = {
        "config": {
            "billingPeriodStart": "2026-06-01T00:00:00+00:00",
            "billingPeriodEnd": "2026-07-01T00:00:00+00:00",
        }
    }

    payload = loop._build_grok_billing_rate_limit_payload(
        config,
        observed_at=observed_at,
        response_body=response_body,
    )

    assert payload[6] == "xai_grok_build_monthly_credits:credits"
    assert payload[8] == "credits"
    assert payload[9] == datetime(2026, 7, 1, tzinfo=timezone.utc)
    assert payload[10] is None
    assert payload[11] is None
    assert payload[12] is None
    assert payload[13] is None
    raw_provider_fields = json.loads(payload[16])
    assert raw_provider_fields["quota_unit"] == "grok_billing_period_only"
    assert "creditUsagePercent" not in raw_provider_fields
    assert "productUsage" not in raw_provider_fields
    evidence = json.loads(payload[17])
    assert "grok_billing_period_only" in evidence["signals"]


def test_grok_billing_sidecar_payload_raises_without_usage_or_period() -> None:
    config = _grok_billing_poll_config()
    observed_at = datetime(2026, 6, 30, 12, 0, tzinfo=timezone.utc)

    with pytest.raises(ValueError, match="absolute or percentage quota fields"):
        loop._build_grok_billing_rate_limit_payload(
            config,
            observed_at=observed_at,
            response_body={"config": {}},
        )


def test_run_due_sidecar_tasks_persists_grok_billing_period_only_snapshot(
    monkeypatch,
) -> None:
    config = _grok_billing_poll_config()
    period_only_payload = {
        "config": {
            "billingPeriodStart": "2026-06-01T00:00:00+00:00",
            "billingPeriodEnd": "2026-07-15T00:00:00+00:00",
        }
    }
    captured = {}

    monkeypatch.setattr(
        loop,
        "_fetch_grok_billing_payload",
        lambda *_args, **_kwargs: {
            "status_code": 200,
            "payload": period_only_payload,
        },
    )
    monkeypatch.setattr(
        loop,
        "_load_grok_billing_auth_context",
        lambda _path: _grok_billing_auth_context(),
    )

    def fake_persist(cfg, *, observed_at, response_body, identity_headers=None):
        captured["response_body"] = response_body
        return 1, 1

    monkeypatch.setattr(loop, "_persist_grok_billing_observations", fake_persist)

    events = loop.run_due_sidecar_tasks(
        config,
        loop.SidecarTaskState(),
        now_monotonic=100.0,
    )

    assert captured["response_body"] == period_only_payload
    assert events[0]["event"] == "grok_billing_poll"
    assert events[0]["persisted"] is True
    assert events[0]["observation_count"] == 1


def test_persist_grok_billing_observations_uses_sidecar_db_path(monkeypatch) -> None:
    config = _grok_billing_poll_config(
        dsn="postgresql://aawm:aawm_dev@pgbouncer:6432/aawm_tristore",
        db_lock_timeout_ms=123,
        db_statement_timeout_ms=456,
    )
    observed_at = datetime(2026, 6, 16, 20, 4, tzinfo=timezone.utc)
    fake_conn = _FakeProviderStatusConnection()
    monkeypatch.setattr(loop.probes.psycopg, "connect", lambda _dsn: fake_conn)

    observation_count, inserted_count = loop._persist_grok_billing_observations(
        config,
        observed_at=observed_at,
        response_body=_grok_billing_payload(),
    )

    assert observation_count == 1
    assert inserted_count == 1
    assert fake_conn.cursor_instance.execute_calls[:3] == [
        (
            "SELECT set_config('application_name', %s, false)",
            ("aawm-provider-status-observations-grok-billing",),
        ),
        ("SELECT set_config('lock_timeout', %s, true)", ("123ms",)),
        ("SELECT set_config('statement_timeout', %s, true)", ("456ms",)),
    ]
    insert_sql, insert_payload = fake_conn.cursor_instance.execute_calls[3]
    assert insert_sql == loop.GROK_BILLING_RATE_LIMIT_INSERT_SQL
    assert "latest.evidence" in insert_sql
    assert "candidate.evidence" in insert_sql
    assert insert_payload[6] == "xai_grok_build_monthly_credits:credits"
    assert insert_payload[11] is None
    assert insert_payload[12] is None
    assert insert_payload[13] is None
    evidence = json.loads(insert_payload[17])
    assert evidence["request_contract_source"] == "grok_billing_sidecar_poll"
    assert len(evidence["request_contract_fingerprint"]) == 64


def test_fetch_grok_billing_payload_retries_cancelled_timeout_then_succeeds(
    monkeypatch,
) -> None:
    config = _grok_billing_poll_config(
        grok_billing_poll_max_attempts=3,
        grok_billing_poll_retry_backoff_seconds=0.5,
    )
    sleeps: list[float] = []
    attempts = {"count": 0}
    captured_requests = []

    def fake_urlopen(request, timeout):
        attempts["count"] += 1
        captured_requests.append(request)
        if attempts["count"] < 3:
            body = (
                '{"code":"The operation was cancelled",'
                '"error":"Timeout expired"}'
            )
            raise urllib_error.HTTPError(
                config.grok_billing_url,
                400,
                "Bad Request",
                hdrs=None,
                fp=BytesIO(body.encode("utf-8")),
            )
        return type(
            "Resp",
            (),
            {
                "status": 200,
                "getcode": lambda self: 200,
                "read": lambda self: json.dumps(_grok_billing_payload()).encode(),
                "__enter__": lambda self: self,
                "__exit__": lambda self, *args: None,
            },
        )()

    monkeypatch.setattr(
        loop,
        "_load_grok_billing_auth_context",
        lambda _p: _grok_billing_auth_context(),
    )
    monkeypatch.setattr(loop.urllib_request, "urlopen", fake_urlopen)
    monkeypatch.setattr(
        loop,
        "GROK_BILLING_POLL_SLEEP_FN",
        lambda seconds: sleeps.append(seconds),
    )

    fetched = loop._fetch_grok_billing_payload(config)

    assert attempts["count"] == 3
    assert [request.get_method() for request in captured_requests] == [
        "GET",
        "GET",
        "GET",
    ]
    assert sleeps == [0.5, 1.0]
    assert fetched["status_code"] == 200
    assert fetched["attempt_count"] == 3
    assert fetched["retry_count"] == 2
    assert fetched["payload"] == _grok_billing_payload()


def test_fetch_grok_billing_payload_reloads_auth_context_between_retries(
    monkeypatch,
) -> None:
    config = _grok_billing_poll_config(
        grok_billing_poll_max_attempts=2,
        grok_billing_poll_retry_backoff_seconds=0,
    )
    auth_contexts = [
        _grok_billing_auth_context(access_token="first-token-secret"),
        _grok_billing_auth_context(access_token="second-token-secret"),
    ]
    captured_authorizations: list[str] = []

    def fake_load_auth_context(_path):
        return auth_contexts.pop(0)

    def fake_urlopen(request, timeout):
        captured_authorizations.append(request.headers["Authorization"])
        if len(captured_authorizations) == 1:
            body = (
                '{"code":"The operation was cancelled",'
                '"error":"Timeout expired"}'
            )
            raise urllib_error.HTTPError(
                config.grok_billing_url,
                400,
                "Bad Request",
                hdrs=None,
                fp=BytesIO(body.encode("utf-8")),
            )
        return type(
            "Resp",
            (),
            {
                "status": 200,
                "getcode": lambda self: 200,
                "read": lambda self: json.dumps(_grok_billing_payload()).encode(),
                "__enter__": lambda self: self,
                "__exit__": lambda self, *args: None,
            },
        )()

    monkeypatch.setattr(loop, "_load_grok_billing_auth_context", fake_load_auth_context)
    monkeypatch.setattr(loop.urllib_request, "urlopen", fake_urlopen)

    fetched = loop._fetch_grok_billing_payload(config)

    assert fetched["status_code"] == 200
    assert captured_authorizations == [
        "Bearer first-token-secret",
        "Bearer second-token-secret",
    ]


def test_grok_billing_identity_headers_derive_from_oidc_credential() -> None:
    credential = {
        "access_token": "access-token-secret",
        "user_id": "user_123",
        "team_id": "team_123",
        "email": "user@example.com",
    }

    headers = loop._grok_billing_identity_headers(credential)

    assert headers == {
        "x-userid": "user_123",
        "x-grok-user-id": "user_123",
        "x-teamid": "team_123",
        "x-email": "user@example.com",
    }


def test_grok_billing_identity_headers_require_all_fields() -> None:
    credential = {
        "access_token": "access-token-secret",
        "user_id": "user_123",
        "email": "user@example.com",
    }

    with pytest.raises(ValueError, match="team_id"):
        loop._grok_billing_identity_headers(credential)


def test_load_grok_billing_auth_context_reads_identity_from_oidc_credential(
    monkeypatch,
) -> None:
    credential = {
        "access_token": "access-token-secret",
        "user_id": "user_123",
        "team_id": "team_123",
        "email": "user@example.com",
    }

    monkeypatch.setattr(
        loop.grok_oidc_refresh,
        "_read_credential_payload",
        lambda _path: {"scope": credential},
    )
    monkeypatch.setattr(loop.grok_oidc_refresh, "_resolve_scope", lambda _scope: "scope")
    monkeypatch.setattr(
        loop.grok_oidc_refresh,
        "_select_credential_record",
        lambda _payload, _scope: credential,
    )

    auth_context = loop._load_grok_billing_auth_context("/home/zepfu/.grok/auth.json")

    assert auth_context["access_token"] == "access-token-secret"
    assert auth_context["identity_headers"] == {
        "x-userid": "user_123",
        "x-grok-user-id": "user_123",
        "x-teamid": "team_123",
        "x-email": "user@example.com",
    }


def test_build_grok_billing_request_headers_matches_native_passthrough_by_default() -> None:
    config = _grok_billing_poll_config(
        grok_billing_client_version="0.2.55",
        grok_billing_client_identifier="grok-cli",
        grok_billing_xai_token_auth="xai-grok-cli",
        grok_billing_model="grok-composer-2.5-fast",
    )

    headers = loop._build_grok_billing_request_headers(
        config,
        access_token="access-token-secret",
        identity_headers=_grok_billing_auth_context()["identity_headers"],
    )

    assert headers["accept"] == "application/json"
    assert headers["authorization"] == "Bearer access-token-secret"
    assert headers["content-type"] == "application/json"
    assert headers["user-agent"] == "grok/0.2.55"
    assert headers["x-grok-client-identifier"] == "grok-cli"
    assert headers["x-grok-client-version"] == "0.2.55"
    assert headers["x-grok-model-override"] == "grok-composer-2.5-fast"
    assert headers["x-xai-token-auth"] == "xai-grok-cli"
    assert headers["x-userid"] == "user_123"
    assert headers["x-grok-user-id"] == "user_123"
    assert headers["x-teamid"] == "team_123"
    assert headers["x-email"] == "user@example.com"
    assert headers["x-grok-req-id"] == headers["x-request-id"]


def test_build_grok_billing_request_headers_omits_native_shape_when_disabled() -> None:
    config = _grok_billing_poll_config(
        grok_billing_client_version="0.2.55",
        grok_billing_client_identifier="grok-cli",
        grok_billing_xai_token_auth="xai-grok-cli",
        grok_billing_model="grok-composer-2.5-fast",
        grok_billing_include_model_override=False,
    )

    headers = loop._build_grok_billing_request_headers(
        config,
        access_token="access-token-secret",
        identity_headers=_grok_billing_auth_context()["identity_headers"],
    )

    assert headers["content-type"] == "application/json"
    assert "x-grok-model-override" not in headers


def test_fetch_grok_billing_payload_uses_configured_http_method(monkeypatch) -> None:
    config = _grok_billing_poll_config(grok_billing_http_method="POST")
    captured = {}

    monkeypatch.setattr(
        loop,
        "_load_grok_billing_auth_context",
        lambda _path: _grok_billing_auth_context(),
    )

    def fake_urlopen(request, timeout):
        captured["method"] = request.get_method()
        return type(
            "Resp",
            (),
            {
                "status": 200,
                "getcode": lambda self: 200,
                "read": lambda self: json.dumps(_grok_billing_payload()).encode(),
                "__enter__": lambda self: self,
                "__exit__": lambda self, *args: None,
            },
        )()

    monkeypatch.setattr(loop.urllib_request, "urlopen", fake_urlopen)

    fetched = loop._fetch_grok_billing_payload(config)

    assert fetched["status_code"] == 200
    assert captured["method"] == "POST"


def test_fetch_grok_billing_payload_includes_identity_headers(monkeypatch) -> None:
    config = _grok_billing_poll_config()
    captured = {}

    monkeypatch.setattr(
        loop,
        "_load_grok_billing_auth_context",
        lambda _path: _grok_billing_auth_context(),
    )

    def fake_urlopen(request, timeout):
        captured["headers"] = dict(request.header_items())
        return type(
            "Resp",
            (),
            {
                "status": 200,
                "getcode": lambda self: 200,
                "read": lambda self: json.dumps(_grok_billing_payload()).encode(),
                "__enter__": lambda self: self,
                "__exit__": lambda self, *args: None,
            },
        )()

    monkeypatch.setattr(loop.urllib_request, "urlopen", fake_urlopen)

    fetched = loop._fetch_grok_billing_payload(config)

    assert fetched["status_code"] == 200
    assert captured["headers"]["Authorization"] == "Bearer access-token-secret"
    assert captured["headers"]["X-userid"] == "user_123"
    assert captured["headers"]["X-grok-user-id"] == "user_123"
    assert captured["headers"]["X-teamid"] == "team_123"
    assert captured["headers"]["X-email"] == "user@example.com"


def test_fetch_grok_billing_payload_does_not_retry_auth_failure(monkeypatch) -> None:
    config = _grok_billing_poll_config(grok_billing_poll_max_attempts=3)
    attempts = {"count": 0}
    sleeps: list[float] = []

    monkeypatch.setattr(
        loop,
        "_load_grok_billing_auth_context",
        lambda _p: _grok_billing_auth_context(access_token="token"),
    )
    monkeypatch.setattr(
        loop,
        "GROK_BILLING_POLL_SLEEP_FN",
        lambda seconds: sleeps.append(seconds),
    )

    def fake_urlopen(_request, timeout):
        attempts["count"] += 1
        body = '{"error":"invalid_token"}'
        raise urllib_error.HTTPError(
            config.grok_billing_url,
            401,
            "Unauthorized",
            hdrs=None,
            fp=BytesIO(body.encode("utf-8")),
        )

    monkeypatch.setattr(loop.urllib_request, "urlopen", fake_urlopen)

    with pytest.raises(ValueError, match="HTTP 401"):
        loop._fetch_grok_billing_payload(config)

    assert attempts["count"] == 1
    assert sleeps == []


def test_fetch_grok_billing_payload_does_not_retry_rate_limit(monkeypatch) -> None:
    config = _grok_billing_poll_config(grok_billing_poll_max_attempts=3)
    attempts = {"count": 0}
    sleeps: list[float] = []

    monkeypatch.setattr(
        loop,
        "_load_grok_billing_auth_context",
        lambda _p: _grok_billing_auth_context(access_token="token"),
    )
    monkeypatch.setattr(
        loop,
        "GROK_BILLING_POLL_SLEEP_FN",
        lambda seconds: sleeps.append(seconds),
    )

    def fake_urlopen(_request, timeout):
        attempts["count"] += 1
        body = '{"error":"usage_limit_reached"}'
        raise urllib_error.HTTPError(
            config.grok_billing_url,
            429,
            "Too Many Requests",
            hdrs=None,
            fp=BytesIO(body.encode("utf-8")),
        )

    monkeypatch.setattr(loop.urllib_request, "urlopen", fake_urlopen)

    with pytest.raises(loop.GrokBillingPollError, match="HTTP 429") as exc_info:
        loop._fetch_grok_billing_payload(config)

    assert attempts["count"] == 1
    assert sleeps == []
    assert exc_info.value.status_code == 429
    assert exc_info.value.attempt_count == 1
    assert exc_info.value.retry_count == 0


def test_fetch_grok_billing_payload_does_not_retry_rate_limit_timeout_hint(
    monkeypatch,
) -> None:
    config = _grok_billing_poll_config(grok_billing_poll_max_attempts=3)
    attempts = {"count": 0}
    sleeps: list[float] = []

    monkeypatch.setattr(
        loop,
        "_load_grok_billing_auth_context",
        lambda _p: _grok_billing_auth_context(access_token="token"),
    )
    monkeypatch.setattr(
        loop,
        "GROK_BILLING_POLL_SLEEP_FN",
        lambda seconds: sleeps.append(seconds),
    )

    def fake_urlopen(_request, timeout):
        attempts["count"] += 1
        body = '{"code":"rate_limited","error":"Timeout expired"}'
        raise urllib_error.HTTPError(
            config.grok_billing_url,
            429,
            "Too Many Requests",
            hdrs=None,
            fp=BytesIO(body.encode("utf-8")),
        )

    monkeypatch.setattr(loop.urllib_request, "urlopen", fake_urlopen)

    with pytest.raises(loop.GrokBillingPollError, match="HTTP 429") as exc_info:
        loop._fetch_grok_billing_payload(config)

    assert attempts["count"] == 1
    assert sleeps == []
    assert exc_info.value.status_code == 429
    assert exc_info.value.attempt_count == 1
    assert exc_info.value.retry_count == 0


def test_run_due_sidecar_tasks_reports_grok_billing_final_failure_status(
    monkeypatch,
) -> None:
    config = _grok_billing_poll_config(grok_billing_poll_max_attempts=2)

    def fake_fetch(_config):
        raise loop.GrokBillingPollError(
            "Grok billing poll failed with HTTP 400 (Timeout expired) "
            "and Authorization=Bearer secret-token",
            status_code=400,
            attempt_count=2,
            retry_count=1,
        )

    monkeypatch.setattr(loop, "_fetch_grok_billing_payload", fake_fetch)

    events = loop.run_due_sidecar_tasks(
        config,
        loop.SidecarTaskState(),
        now_monotonic=100.0,
    )

    assert events[0]["event"] == "grok_billing_poll"
    assert events[0]["status_code"] == 400
    assert events[0]["attempt_count"] == 2
    assert events[0]["retry_count"] == 1
    assert events[0]["error_class"] == "GrokBillingPollError"
    assert "HTTP 400" in events[0]["error_message"]
    assert "REDACTED" in events[0]["error_message"]
    assert "secret-token" not in json.dumps(events)


def test_run_due_sidecar_tasks_reports_grok_billing_auth_failure_once(
    monkeypatch,
) -> None:
    config = _grok_billing_poll_config(grok_billing_poll_max_attempts=3)

    def fake_fetch(_config):
        raise loop.GrokBillingPollError(
            "Grok billing poll failed with HTTP 401 (invalid_token).",
            status_code=401,
            attempt_count=1,
            retry_count=0,
        )

    monkeypatch.setattr(loop, "_fetch_grok_billing_payload", fake_fetch)

    events = loop.run_due_sidecar_tasks(
        config,
        loop.SidecarTaskState(),
        now_monotonic=100.0,
    )

    assert events[0]["event"] == "grok_billing_poll"
    assert events[0]["status_code"] == 401
    assert events[0]["attempt_count"] == 1
    assert events[0]["retry_count"] == 0
    assert events[0]["error_class"] == "GrokBillingPollError"


def test_run_due_sidecar_tasks_skips_when_observability_anomaly_scan_disabled(
    monkeypatch,
    tmp_path,
) -> None:
    config = _grok_billing_poll_config(
        grok_billing_poll_enabled=False,
        observability_anomaly_scan_enabled=False,
        observability_anomaly_scan_error_log_dir=str(tmp_path),
    )

    monkeypatch.setattr(
        loop,
        "_collect_observability_anomalies",
        lambda *_args, **_kwargs: pytest.fail("Anomaly scan should not run"),
    )

    assert loop.run_due_sidecar_tasks(
        config,
        loop.SidecarTaskState(),
        now_monotonic=100.0,
    ) == []


def test_run_due_sidecar_tasks_throttles_observability_anomaly_scan(
    monkeypatch,
    tmp_path,
) -> None:
    config = _grok_billing_poll_config(
        grok_billing_poll_enabled=False,
        observability_anomaly_scan_enabled=True,
        observability_anomaly_scan_interval_seconds=3600.0,
        observability_anomaly_scan_lookback_hours=4.0,
        observability_anomaly_scan_error_log_dir=str(tmp_path),
    )
    calls = {"scan": 0}

    def fake_collect(_config):
        calls["scan"] += 1
        return []

    monkeypatch.setattr(loop, "_collect_observability_anomalies", fake_collect)

    state = loop.SidecarTaskState()
    first_events = loop.run_due_sidecar_tasks(config, state, now_monotonic=100.0)
    second_events = loop.run_due_sidecar_tasks(config, state, now_monotonic=200.0)
    third_events = loop.run_due_sidecar_tasks(config, state, now_monotonic=3701.0)

    assert calls == {"scan": 2}
    assert first_events[0]["event"] == "observability_anomaly_scan"
    assert first_events[0]["status"] == "healthy"
    assert first_events[0]["anomaly_count"] == 0
    assert first_events[0]["error_log_record_count"] == 0
    assert second_events == []
    assert third_events[0]["event"] == "observability_anomaly_scan"


def test_run_due_sidecar_tasks_writes_observability_anomaly_jsonl(
    monkeypatch,
    tmp_path,
) -> None:
    config = _grok_billing_poll_config(
        grok_billing_poll_enabled=False,
        observability_anomaly_scan_enabled=True,
        observability_anomaly_scan_error_log_dir=str(tmp_path),
    )
    anomalies = [
        {
            "anomaly_class": "missing_repository_for_agent_context",
            "expected": "repository should be derivable",
            "row_count": 2,
            "examples": [
                {
                    "row_id": 123,
                    "session_id": "session-123",
                    "model": "grok-composer-2.5-fast",
                    "client_name": "Grok",
                }
            ],
        }
    ]

    monkeypatch.setattr(
        loop,
        "_collect_observability_anomalies",
        lambda _config: anomalies,
    )

    events = loop.run_due_sidecar_tasks(
        config,
        loop.SidecarTaskState(),
        now_monotonic=100.0,
    )

    error_path = tmp_path / "dev-error.jsonl"
    lines = error_path.read_text().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert events[0]["event"] == "observability_anomaly_scan"
    assert events[0]["status"] == "anomalies_found"
    assert events[0]["anomaly_count"] == 1
    assert events[0]["error_log_record_count"] == 1
    assert events[0]["error_log_path"] == str(error_path)
    assert record["event"] == "aawm_observability_anomaly"
    assert record["environment"] == "dev"
    assert record["anomaly_class"] == "missing_repository_for_agent_context"
    assert record["row_count"] == 2
    assert record["examples"][0]["row_id"] == 123
    assert ".analysis/todo.md" in record["recommended_todo"]
    assert "Clean up" in record["cleanup_requirement"]


def test_observability_anomaly_jsonl_repairs_bind_mount_owner_and_mode(
    monkeypatch,
    tmp_path,
) -> None:
    config = _grok_billing_poll_config(
        grok_billing_poll_enabled=False,
        observability_anomaly_scan_enabled=True,
        observability_anomaly_scan_error_log_dir=str(tmp_path),
    )
    monkeypatch.setenv("LITELLM_AAWM_ERROR_LOG_FILE_UID", "1234")
    monkeypatch.setenv("LITELLM_AAWM_ERROR_LOG_FILE_GID", "5678")
    monkeypatch.setenv("LITELLM_AAWM_ERROR_LOG_FILE_MODE", "0601")
    chown_calls = []
    chmod_calls = []

    def fake_chown(path, uid, gid):
        chown_calls.append((path, uid, gid))

    def fake_chmod(path, mode):
        chmod_calls.append((path, mode))

    monkeypatch.setattr(loop.os, "chown", fake_chown)
    monkeypatch.setattr(loop.os, "chmod", fake_chmod)

    written, path = loop._write_observability_anomaly_error_records(
        config,
        observed_at=datetime(2026, 6, 24, tzinfo=timezone.utc),
        anomalies=[
            {
                "anomaly_class": "missing_repository_for_agent_context",
                "expected": "repository should be derivable",
                "row_count": 1,
                "examples": [{"row_id": 123}],
            }
        ],
    )

    assert written == 1
    assert path == tmp_path / "dev-error.jsonl"
    assert chown_calls == [(path, 1234, 5678)]
    assert chmod_calls == [(path, 0o601)]
    record = json.loads(path.read_text().strip())
    assert record["event"] == "aawm_observability_anomaly"


def test_run_due_sidecar_tasks_reports_observability_anomaly_scan_failure(
    monkeypatch,
    tmp_path,
) -> None:
    config = _grok_billing_poll_config(
        grok_billing_poll_enabled=False,
        observability_anomaly_scan_enabled=True,
        observability_anomaly_scan_error_log_dir=str(tmp_path),
    )

    def fake_collect(_config):
        raise RuntimeError("database unavailable with token=secret-value")

    monkeypatch.setattr(loop, "_collect_observability_anomalies", fake_collect)

    events = loop.run_due_sidecar_tasks(
        config,
        loop.SidecarTaskState(),
        now_monotonic=100.0,
    )

    assert events[0]["event"] == "observability_anomaly_scan"
    assert events[0]["status"] == "scan_failed"
    assert events[0]["error_class"] == "RuntimeError"
    assert "REDACTED" in events[0]["error_message"]
    assert "secret-value" not in events[0]["error_message"]
    assert not (tmp_path / "dev-error.jsonl").exists()


def test_collect_observability_anomalies_runs_read_only_queries(monkeypatch) -> None:
    class FakeCursor:
        def __init__(self) -> None:
            self.execute_calls = []
            self.description = []
            self.rows = []

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            return None

        def execute(self, statement, params=None) -> None:
            self.execute_calls.append((statement, params))
            if statement == loop.OBSERVABILITY_SESSION_HISTORY_ANOMALY_SQL:
                self.description = [
                    ("anomaly_class",),
                    ("expected",),
                    ("row_count",),
                    ("examples",),
                ]
                self.rows = [
                    (
                        "missing_provider",
                        "provider should be populated",
                        2,
                        [{"row_id": 123}],
                    )
                ]
            elif statement == loop.OBSERVABILITY_RATE_LIMIT_ANOMALY_SQL:
                self.description = [
                    ("anomaly_class",),
                    ("expected",),
                    ("row_count",),
                    ("examples",),
                ]
                self.rows = [
                    (
                        "stale_rate_limit_reset_with_recent_traffic",
                        "reset should be future",
                        0,
                        [],
                    )
                ]
            else:
                self.description = []
                self.rows = []

        def fetchall(self):
            return self.rows

    class FakeConnection:
        def __init__(self) -> None:
            self.cursor_instance = FakeCursor()
            self.rollback_count = 0

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            return None

        def cursor(self):
            return self.cursor_instance

        def rollback(self) -> None:
            self.rollback_count += 1

    fake_conn = FakeConnection()
    config = _grok_billing_poll_config(
        grok_billing_poll_enabled=False,
        observability_anomaly_scan_enabled=True,
        observability_anomaly_scan_lookback_hours=6.0,
    )

    monkeypatch.setattr(loop, "_resolve_dsn", lambda _config: "postgresql://example/db")
    monkeypatch.setattr(loop.probes.psycopg, "connect", lambda _dsn: fake_conn)

    anomalies = loop._collect_observability_anomalies(config)

    assert anomalies == [
        {
            "anomaly_class": "missing_provider",
            "expected": "provider should be populated",
            "row_count": 2,
            "examples": [{"row_id": 123}],
        }
    ]
    assert (
        "LEFT(COALESCE(inbound_model_alias, ''), 5) = 'aawm-'"
        in loop.OBSERVABILITY_SESSION_HISTORY_ANOMALY_SQL
    )
    assert "LIKE 'aawm-%'" not in loop.OBSERVABILITY_SESSION_HISTORY_ANOMALY_SQL
    assert (
        "LOWER(COALESCE(provider, '')) = 'xai'"
        in loop.OBSERVABILITY_SESSION_HISTORY_ANOMALY_SQL
    )
    assert (
        "LOWER(COALESCE(client_name, '')) = 'grok-build'"
        in loop.OBSERVABILITY_SESSION_HISTORY_ANOMALY_SQL
    )
    assert (
        "LEFT(COALESCE(inbound_model_alias, ''), 5) <> 'aawm-'"
        in loop.OBSERVABILITY_SESSION_HISTORY_ANOMALY_SQL
    )
    assert (
        "LOWER(COALESCE(metadata->>'client_user_agent', '')) LIKE 'grok-pager/%%'"
        in loop.OBSERVABILITY_SESSION_HISTORY_ANOMALY_SQL
    )
    assert "'% grok-shell/%'" not in loop.OBSERVABILITY_SESSION_HISTORY_ANOMALY_SQL
    assert fake_conn.rollback_count == 0
    assert fake_conn.cursor_instance.execute_calls[:4] == [
        (
            "SELECT set_config('application_name', %s, false)",
            ("aawm-provider-status-observations-anomaly-scan",),
        ),
        ("SELECT set_config('lock_timeout', %s, true)", ("1000ms",)),
        ("SELECT set_config('statement_timeout', %s, true)", ("5000ms",)),
        ("SELECT set_config('jit', 'off', true)", None),
    ]
    assert fake_conn.cursor_instance.execute_calls[4:] == [
        (
            loop.OBSERVABILITY_SESSION_HISTORY_ANOMALY_SQL,
            (
                6.0,
                6.0,
                loop.OBSERVABILITY_NULL_REPOSITORY_CLUSTER_MIN_ROWS,
                loop.OBSERVABILITY_ANOMALY_SAMPLE_LIMIT,
            ),
        ),
        (
            loop.OBSERVABILITY_RATE_LIMIT_ANOMALY_SQL,
            (6.0, 6.0, loop.OBSERVABILITY_ANOMALY_SAMPLE_LIMIT),
        ),
    ]


def test_observability_session_history_anomaly_sql_classifies_null_repository_clusters() -> None:
    sql = loop.OBSERVABILITY_SESSION_HISTORY_ANOMALY_SQL

    assert "null_repository_clusters AS" in sql
    assert "'large_null_repository_cluster' AS anomaly_class" in sql
    assert "rendered_repository', 'unknown'" in sql
    assert "HAVING COUNT(*) >= %s::int" in sql
    assert "COALESCE(metadata->>'tenant_id_source', '') = 'repository_untrusted'" in sql
    assert "metadata->>'repository_tenant_fallback_skipped'" in sql


def test_observability_rate_limit_anomaly_sql_filters_recent_unscoped_observations() -> None:
    sql = loop.OBSERVABILITY_RATE_LIMIT_ANOMALY_SQL
    assert "observed_at >=" in sql
    assert "account_hash IS NULL" in sql
    assert sql.index("observed_at >=") < sql.index("recent_traffic AS")


def test_refresh_antigravity_oauth_token_file_writes_direct_response(
    tmp_path, monkeypatch
) -> None:
    token_path = tmp_path / "antigravity-oauth-token"
    token_path.write_text(
        json.dumps(
            {
                "auth_method": "consumer",
                "token": {
                    "access_token": "ya29.expired-antigravity",
                    "expiry": "2026-01-01T00:00:00Z",
                    "refresh_token": "refresh-token-123",
                    "token_type": "Bearer",
                },
            }
        ),
        encoding="utf-8",
    )
    token_path.chmod(0o600)

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            return None

        def read(self) -> bytes:
            return json.dumps(
                {
                    "access_token": "ya29.refreshed-antigravity",
                    "expires_in": 3600,
                }
            ).encode("utf-8")

    def fake_urlopen(request, timeout):
        assert timeout == 12.5
        assert request.full_url == "https://oauth2.example/token"
        assert request.data is not None
        form = request.data.decode("utf-8")
        assert "client_id=client-id" in form
        assert "client_secret=client-secret" in form
        assert "refresh_token=refresh-token-123" in form
        return FakeResponse()

    monkeypatch.setattr(
        antigravity_oauth_refresh.urllib_request,
        "urlopen",
        fake_urlopen,
    )

    summary = antigravity_oauth_refresh.refresh_antigravity_oauth_token_file(
        token_path,
        force=True,
        lock_file=tmp_path / "antigravity.lock",
        token_endpoint="https://oauth2.example/token",
        client_id="client-id",
        client_secret="client-secret",
        http_timeout_seconds=12.5,
    )

    persisted = json.loads(token_path.read_text(encoding="utf-8"))
    assert summary["attempted"] is True
    assert summary["refreshed"] is True
    assert summary["refresh_method"] == "direct"
    assert persisted["token"]["access_token"] == "ya29.refreshed-antigravity"
    assert persisted["token"]["refresh_token"] == "refresh-token-123"
    assert token_path.stat().st_mode & 0o777 == 0o600


def test_refresh_antigravity_oauth_token_file_uses_agy_cli_fallback(
    tmp_path, monkeypatch
) -> None:
    home_path = tmp_path / "home"
    token_path = (
        home_path
        / ".gemini"
        / "antigravity-cli"
        / "antigravity-oauth-token"
    )
    token_path.parent.mkdir(parents=True)
    token_path.write_text(
        json.dumps(
            {
                "auth_method": "consumer",
                "token": {
                    "access_token": "ya29.expired-antigravity",
                    "expiry": "2026-01-01T00:00:00Z",
                    "refresh_token": "refresh-token-123",
                    "token_type": "Bearer",
                },
            }
        ),
        encoding="utf-8",
    )
    cli_path = tmp_path / "agy"
    cli_path.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    cli_path.chmod(0o755)

    def fake_post_refresh_request(**_kwargs):
        raise antigravity_oauth_refresh._RefreshHttpError(
            "status=401, error=invalid_client",
            oauth_error="invalid_client",
        )

    class FakeCompletedProcess:
        returncode = 0

    def fake_run(args, stdout, stderr, env, timeout, check):
        assert args[-1] == "models"
        assert env["HOME"] == str(home_path)
        assert timeout == 9.0
        assert check is False
        refreshed = json.loads(token_path.read_text(encoding="utf-8"))
        refreshed["token"]["access_token"] = "ya29.refreshed-via-agy"
        refreshed["token"]["expiry"] = (
            datetime.now(timezone.utc) + timedelta(hours=1)
        ).isoformat()
        token_path.write_text(json.dumps(refreshed), encoding="utf-8")
        return FakeCompletedProcess()

    monkeypatch.setattr(
        antigravity_oauth_refresh,
        "_post_refresh_request",
        fake_post_refresh_request,
    )
    monkeypatch.setattr(antigravity_oauth_refresh.subprocess, "run", fake_run)

    summary = antigravity_oauth_refresh.refresh_antigravity_oauth_token_file(
        token_path,
        force=True,
        lock_file=tmp_path / "antigravity.lock",
        cli_path=cli_path,
        client_id="client-id",
        client_secret="client-secret",
        cli_timeout_seconds=9.0,
    )

    persisted = json.loads(token_path.read_text(encoding="utf-8"))
    assert summary["attempted"] is True
    assert summary["refreshed"] is True
    assert summary["refresh_method"] == "agy_cli"
    assert persisted["token"]["access_token"] == "ya29.refreshed-via-agy"


def test_refresh_antigravity_oauth_token_file_uses_seed_auth_for_agy_cli_fallback(
    tmp_path, monkeypatch
) -> None:
    home_path = tmp_path / "home"
    seed_path = (
        home_path
        / ".gemini"
        / "antigravity-cli"
        / "antigravity-oauth-token"
    )
    managed_path = tmp_path / "managed" / "antigravity-oauth-token"
    seed_path.parent.mkdir(parents=True)
    managed_path.parent.mkdir(parents=True)
    token_payload = {
        "auth_method": "consumer",
        "token": {
            "access_token": "ya29.expired-antigravity",
            "expiry": "2026-01-01T00:00:00Z",
            "refresh_token": "refresh-token-123",
            "token_type": "Bearer",
        },
    }
    seed_path.write_text(json.dumps(token_payload), encoding="utf-8")
    managed_path.write_text(json.dumps(token_payload), encoding="utf-8")
    seed_before = seed_path.read_text(encoding="utf-8")
    cli_path = tmp_path / "agy"
    cli_path.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    cli_path.chmod(0o755)

    def fake_post_refresh_request(**_kwargs):
        raise antigravity_oauth_refresh._RefreshHttpError(
            "status=401, error=invalid_client",
            oauth_error="invalid_client",
        )

    class FakeCompletedProcess:
        returncode = 0

    def fake_run(args, stdout, stderr, env, timeout, check):
        assert args[-1] == "models"
        assert env["HOME"] != str(home_path)
        assert timeout == 9.0
        assert check is False
        staged_auth_path = (
            Path(env["HOME"])
            / ".gemini"
            / "antigravity-cli"
            / "antigravity-oauth-token"
        )
        refreshed = json.loads(staged_auth_path.read_text(encoding="utf-8"))
        refreshed["token"]["access_token"] = "ya29.refreshed-via-agy"
        refreshed["token"]["expiry"] = (
            datetime.now(timezone.utc) + timedelta(hours=1)
        ).isoformat()
        staged_auth_path.write_text(json.dumps(refreshed), encoding="utf-8")
        return FakeCompletedProcess()

    monkeypatch.setattr(
        antigravity_oauth_refresh,
        "_post_refresh_request",
        fake_post_refresh_request,
    )
    monkeypatch.setattr(antigravity_oauth_refresh.subprocess, "run", fake_run)

    summary = antigravity_oauth_refresh.refresh_antigravity_oauth_token_file(
        managed_path,
        seed_auth_file=seed_path,
        force=True,
        lock_file=tmp_path / "antigravity.lock",
        cli_path=cli_path,
        client_id="client-id",
        client_secret="client-secret",
        cli_timeout_seconds=9.0,
    )

    managed_persisted = json.loads(managed_path.read_text(encoding="utf-8"))
    assert summary["attempted"] is True
    assert summary["refreshed"] is True
    assert summary["refresh_method"] == "agy_cli"
    assert summary["auth_file"] == str(managed_path)
    assert managed_persisted["token"]["access_token"] == "ya29.refreshed-via-agy"
    assert seed_path.read_text(encoding="utf-8") == seed_before


def test_refresh_antigravity_oauth_token_file_bootstraps_missing_managed_auth_from_seed(
    tmp_path,
) -> None:
    seed_path = (
        tmp_path
        / "home"
        / ".gemini"
        / "antigravity-cli"
        / "antigravity-oauth-token"
    )
    managed_path = tmp_path / "managed" / "antigravity-oauth-token"
    seed_path.parent.mkdir(parents=True)
    seed_payload = {
        "auth_method": "consumer",
        "token": {
            "access_token": "ya29.seed-antigravity",
            "expiry": (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat(),
            "refresh_token": "refresh-token-123",
            "token_type": "Bearer",
        },
    }
    seed_path.write_text(json.dumps(seed_payload), encoding="utf-8")

    summary = antigravity_oauth_refresh.refresh_antigravity_oauth_token_file(
        managed_path,
        seed_auth_file=seed_path,
        force=False,
        lock_file=tmp_path / "antigravity.lock",
    )

    managed_persisted = json.loads(managed_path.read_text(encoding="utf-8"))
    assert summary["attempted"] is True
    assert summary["refreshed"] is True
    assert summary["refresh_method"] == "seed_copy"
    assert summary["auth_file"] == str(managed_path)
    assert managed_persisted["token"]["access_token"] == "ya29.seed-antigravity"
    assert managed_path.stat().st_mode & 0o777 == 0o600


def test_refresh_codex_oauth_auth_file_writes_direct_response(
    tmp_path, monkeypatch
) -> None:
    auth_path = tmp_path / "auth.json"
    auth_path.write_text(
        json.dumps(
            {
                "auth_mode": "chatgpt",
                "last_refresh": "2026-01-01T00:00:00+00:00",
                "tokens": {
                    "access_token": _build_test_jwt({"exp": int(time.time()) - 60}),
                    "refresh_token": "codex-refresh-token",
                    "account_id": "acct_old",
                },
            }
        ),
        encoding="utf-8",
    )
    auth_path.chmod(0o600)
    refreshed_access_token = _build_test_jwt(
        {
            "exp": int(time.time()) + 3600,
            "https://api.openai.com/auth": {
                "chatgpt_account_id": "acct_refreshed"
            },
        }
    )

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            return None

        def read(self) -> bytes:
            return json.dumps(
                {
                    "access_token": refreshed_access_token,
                    "refresh_token": "codex-refresh-token-new",
                }
            ).encode("utf-8")

    def fake_urlopen(request, timeout):
        assert timeout == 12.5
        assert request.full_url == "https://auth.example/token"
        assert request.data is not None
        payload = json.loads(request.data.decode("utf-8"))
        assert payload == {
            "client_id": "codex-client-id",
            "grant_type": "refresh_token",
            "refresh_token": "codex-refresh-token",
            "scope": "openid profile email",
        }
        return FakeResponse()

    monkeypatch.setattr(
        codex_oauth_refresh.urllib_request,
        "urlopen",
        fake_urlopen,
    )
    lock_path = tmp_path / "codex.lock"
    monkeypatch.setenv("AAWM_CODEX_AUTH_FILE_UID", str(os.getuid()))
    monkeypatch.setenv("AAWM_CODEX_AUTH_FILE_GID", str(os.getgid()))
    monkeypatch.setenv("AAWM_CODEX_AUTH_FILE_MODE", "0o600")

    summary = codex_oauth_refresh.refresh_codex_oauth_auth_file(
        auth_path,
        force=True,
        lock_file=lock_path,
        token_endpoint="https://auth.example/token",
        client_id="codex-client-id",
        http_timeout_seconds=12.5,
    )

    persisted = json.loads(auth_path.read_text(encoding="utf-8"))
    assert summary["attempted"] is True
    assert summary["refreshed"] is True
    assert summary["account_id"] == "acct_refreshed"
    assert persisted["auth_mode"] == "chatgpt"
    assert persisted["tokens"]["access_token"] == refreshed_access_token
    assert persisted["tokens"]["refresh_token"] == "codex-refresh-token-new"
    assert persisted["tokens"]["account_id"] == "acct_refreshed"
    assert isinstance(persisted["tokens"]["expires_at"], int)
    assert persisted["last_refresh"] != "2026-01-01T00:00:00+00:00"
    assert "last_refresh" not in persisted["tokens"]
    assert auth_path.stat().st_mode & 0o777 == 0o600
    assert auth_path.stat().st_uid == os.getuid()
    assert auth_path.stat().st_gid == os.getgid()
    assert lock_path.stat().st_mode & 0o777 == 0o600
    assert lock_path.stat().st_uid == os.getuid()
    assert lock_path.stat().st_gid == os.getgid()


def test_refresh_codex_oauth_auth_file_repairs_metadata_when_skipped(
    tmp_path,
    monkeypatch,
) -> None:
    auth_path = tmp_path / "auth.json"
    lock_path = tmp_path / "auth.json.lock"
    auth_path.write_text(
        json.dumps(
            {
                "tokens": {
                    "access_token": _build_test_jwt(
                        {"exp": int(time.time()) + 3600}
                    ),
                    "refresh_token": "codex-refresh-token",
                },
            }
        ),
        encoding="utf-8",
    )
    auth_path.chmod(0o644)
    lock_path.write_text("", encoding="utf-8")
    lock_path.chmod(0o644)
    monkeypatch.setenv("AAWM_CODEX_AUTH_FILE_UID", str(os.getuid()))
    monkeypatch.setenv("AAWM_CODEX_AUTH_FILE_GID", str(os.getgid()))
    monkeypatch.setenv("AAWM_CODEX_AUTH_FILE_MODE", "0o600")

    summary = codex_oauth_refresh.refresh_codex_oauth_auth_file(
        auth_path,
        lock_file=lock_path,
    )

    assert summary["attempted"] is False
    assert summary["refreshed"] is False
    assert summary["skipped"] is True
    assert auth_path.stat().st_mode & 0o777 == 0o600
    assert auth_path.stat().st_uid == os.getuid()
    assert auth_path.stat().st_gid == os.getgid()
    assert lock_path.stat().st_mode & 0o777 == 0o600
    assert lock_path.stat().st_uid == os.getuid()
    assert lock_path.stat().st_gid == os.getgid()


def test_refresh_codex_oauth_auth_file_reports_missing_refresh_token(
    tmp_path,
) -> None:
    auth_path = tmp_path / "auth.json"
    auth_path.write_text(
        json.dumps(
            {
                "tokens": {
                    "access_token": _build_test_jwt({"exp": int(time.time()) - 60}),
                },
            }
        ),
        encoding="utf-8",
    )

    summary = codex_oauth_refresh.refresh_codex_oauth_auth_file(
        auth_path,
        force=True,
        lock_file=tmp_path / "codex.lock",
    )

    assert summary["attempted"] is True
    assert summary["refreshed"] is False
    assert summary["skipped"] is False
    assert summary["error_class"] == "ValueError"
    assert "refresh_token" not in summary["error_message"]


def test_refresh_xai_oauth_auth_file_writes_direct_response(
    tmp_path, monkeypatch
) -> None:
    auth_path = tmp_path / "oauth-auth.json"
    auth_path.write_text(
        json.dumps(
            {
                "access_token": "expired-xai-token",
                "key": "expired-xai-token",
                "refresh_token": "xai-refresh-token",
                "expires_at": (
                    datetime.now(timezone.utc) - timedelta(minutes=5)
                ).isoformat(),
                "oidc_client_id": "xai-oauth-client-id",
                "token_endpoint": "https://auth.example/token",
            }
        ),
        encoding="utf-8",
    )
    auth_path.chmod(0o600)

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            return None

        def read(self) -> bytes:
            return json.dumps(
                {
                    "access_token": "refreshed-xai-token",
                    "refresh_token": "refreshed-xai-refresh-token",
                    "expires_in": 3600,
                    "token_type": "Bearer",
                }
            ).encode("utf-8")

    def fake_urlopen(request, timeout):
        assert timeout == 12.5
        assert request.full_url == "https://auth.example/token"
        assert request.data is not None
        form = request.data.decode("utf-8")
        assert "grant_type=refresh_token" in form
        assert "refresh_token=xai-refresh-token" in form
        assert "client_id=xai-oauth-client-id" in form
        return FakeResponse()

    monkeypatch.setattr(
        xai_oauth_refresh.urllib_request,
        "urlopen",
        fake_urlopen,
    )

    summary = xai_oauth_refresh.refresh_xai_oauth_auth_file(
        auth_path,
        force=True,
        lock_file=tmp_path / "xai.lock",
        http_timeout_seconds=12.5,
    )

    persisted = json.loads(auth_path.read_text(encoding="utf-8"))
    assert summary["attempted"] is True
    assert summary["refreshed"] is True
    assert persisted["access_token"] == "refreshed-xai-token"
    assert persisted["key"] == "refreshed-xai-token"
    assert persisted["refresh_token"] == "refreshed-xai-refresh-token"
    assert persisted["token_type"] == "Bearer"
    assert isinstance(persisted["expires_at"], str)
    assert auth_path.stat().st_mode & 0o777 == 0o600


def test_refresh_xai_oauth_auth_file_reports_missing_refresh_token(tmp_path) -> None:
    auth_path = tmp_path / "oauth-auth.json"
    auth_path.write_text(
        json.dumps(
            {
                "access_token": "expired-xai-token",
                "expires_at": (
                    datetime.now(timezone.utc) - timedelta(minutes=5)
                ).isoformat(),
                "oidc_client_id": "xai-oauth-client-id",
            }
        ),
        encoding="utf-8",
    )

    summary = xai_oauth_refresh.refresh_xai_oauth_auth_file(
        auth_path,
        force=True,
        lock_file=tmp_path / "xai.lock",
    )

    assert summary["attempted"] is True
    assert summary["refreshed"] is False
    assert summary["skipped"] is False
    assert summary["error_class"] == "ValueError"
    assert "refresh_token" not in summary["error_message"]


def _grok_oidc_auth_persist_config(**overrides):
    from dataclasses import replace

    config = loop.ProviderStatusLoopConfig(
        apply=True,
        dsn="postgresql://aawm:aawm_dev@pgbouncer:6432/aawm_tristore",
        environment="dev",
        interval_seconds=300.0,
        timeout=2.0,
        ping_count=1,
        ping_timeout=2,
        skip_icmp=False,
        once=True,
        setup_schema=False,
        db_lock_timeout_ms=123,
        db_statement_timeout_ms=456,
        grok_oidc_refresh_enabled=True,
        grok_oidc_auth_file="/home/zepfu/.grok/auth.json",
        grok_oidc_auth_file_source="default",
        grok_oidc_lock_file="/home/zepfu/.grok/auth.json.lock",
        grok_oidc_refresh_interval_seconds=3600.0,
        grok_oidc_refresh_buffer_seconds=300,
        grok_oidc_force_refresh=False,
        grok_oidc_http_timeout_seconds=30.0,
        grok_billing_poll_enabled=False,
    )
    if overrides:
        config = replace(config, **overrides)
    return config


def _grok_oidc_refresh_sidecar_event(**overrides) -> dict:
    event = {
        "event": "grok_oidc_refresh",
        "observed_at": "2026-06-19T12:00:00Z",
        "environment": "dev",
        "attempted": True,
        "refreshed": True,
        "skipped": False,
        "auth_file": "/home/zepfu/.grok/auth.json",
        "scope": "https://auth.x.ai::b1a00492-073a-47ea-816f-4c329264a828",
        "expires_at": "2026-06-19T13:00:00Z",
        "error_class": None,
        "error_message": None,
    }
    event.update(overrides)
    return event


def test_build_grok_oidc_auth_observation_maps_successful_refresh() -> None:
    config = _grok_oidc_auth_persist_config()
    event = _grok_oidc_refresh_sidecar_event()

    observation = loop._build_grok_oidc_auth_observation(
        config,
        event,
    )

    assert observation["observed_at"] == datetime(2026, 6, 19, 12, 0, tzinfo=timezone.utc)
    assert observation["environment"] == "dev"
    assert observation["provider"] == "xai"
    assert observation["auth_family"] == "grok_oidc"
    assert observation["credential_scope"] == event["scope"]
    assert observation["auth_file_hash"] == hashlib.sha256(
        event["auth_file"].encode("utf-8")
    ).hexdigest()
    assert observation["status"] == "refreshed"
    assert observation["attempted"] is True
    assert observation["refreshed"] is True
    assert observation["skipped"] is False
    assert observation["expires_at"] == datetime(2026, 6, 19, 13, 0, tzinfo=timezone.utc)
    assert observation["last_success_at"] == observation["observed_at"]
    assert observation["source_task"] == "grok_oidc_refresh"
    assert observation["error_class"] is None
    assert observation["error_message"] is None
    metadata = observation["metadata"]
    assert metadata["auth_file_source"] == "default"
    observation_json = json.dumps(observation, default=str)
    assert "refresh_token" not in observation_json
    assert "access_token" not in observation_json
    assert "/home/zepfu/.grok/auth.json" not in observation_json


def test_build_grok_oidc_auth_observation_sanitizes_refresh_failure() -> None:
    config = _grok_oidc_auth_persist_config()
    event = _grok_oidc_refresh_sidecar_event(
        refreshed=False,
        skipped=False,
        expires_at=None,
        error_class="HTTPError",
        error_message=(
            "token refresh failed with refresh_token=super-secret "
            "and Authorization=Bearer leaked-token"
        ),
    )

    observation = loop._build_grok_oidc_auth_observation(
        config,
        event,
    )

    assert observation["status"] == "failed"
    assert observation["attempted"] is True
    assert observation["refreshed"] is False
    assert observation["skipped"] is False
    assert observation["last_success_at"] is None
    assert observation["error_class"] == "HTTPError"
    assert "REDACTED" in observation["error_message"]
    assert "super-secret" not in json.dumps(observation, default=str)
    assert "leaked-token" not in json.dumps(observation, default=str)


def test_build_grok_oidc_auth_observation_preserves_skipped_refresh_expiry() -> None:
    config = _grok_oidc_auth_persist_config()
    event = _grok_oidc_refresh_sidecar_event(
        attempted=False,
        refreshed=False,
        skipped=True,
        expires_at="2026-06-19T18:00:00Z",
    )

    observation = loop._build_grok_oidc_auth_observation(
        config,
        event,
    )

    assert observation["status"] == "skipped"
    assert observation["attempted"] is False
    assert observation["refreshed"] is False
    assert observation["skipped"] is True
    assert observation["expires_at"] == datetime(2026, 6, 19, 18, 0, tzinfo=timezone.utc)
    assert observation["last_success_at"] == observation["observed_at"]


def test_insert_provider_auth_observations_uses_provider_status_db_path(monkeypatch) -> None:
    config = _grok_oidc_auth_persist_config()
    fake_conn = _FakeProviderStatusConnection()
    monkeypatch.setattr(probes.psycopg, "connect", lambda _dsn: fake_conn)
    observation = loop._build_grok_oidc_auth_observation(
        config,
        _grok_oidc_refresh_sidecar_event(),
    )

    inserted = probes.insert_provider_auth_observations(
        "postgresql://example/db",
        [observation],
        lock_timeout_ms=123,
        statement_timeout_ms=456,
    )

    assert inserted == 1
    assert fake_conn.cursor_instance.execute_calls[:3] == [
        (
            "SELECT set_config('application_name', %s, false)",
            ("aawm-provider-status-observations",),
        ),
        ("SELECT set_config('lock_timeout', %s, true)", ("123ms",)),
        ("SELECT set_config('statement_timeout', %s, true)", ("456ms",)),
    ]
    insert_sql, payloads = fake_conn.cursor_instance.executemany_calls[0]
    assert insert_sql == probes.PROVIDER_AUTH_OBSERVATIONS_INSERT_SQL
    assert payloads[0][1] == "dev"
    assert payloads[0][2] == "xai"
    assert payloads[0][12] == "grok_oidc_refresh"
    assert fake_conn.commit_count == 1
    assert fake_conn.rollback_count == 0


def test_persist_grok_oidc_auth_observation_marks_db_write_failure(monkeypatch) -> None:
    config = _grok_oidc_auth_persist_config()

    def fake_connect(_dsn):
        raise probes.psycopg.OperationalError("connection refused")

    monkeypatch.setattr(loop.probes.psycopg, "connect", fake_connect)

    persisted, inserted, error_class, error_message = loop._persist_grok_oidc_auth_observation(
        config,
        _grok_oidc_refresh_sidecar_event(),
    )

    assert persisted is False
    assert inserted == 0
    assert error_class == "OperationalError"
    assert "connection refused" in error_message


def _codex_oauth_auth_persist_config(**overrides):
    from dataclasses import replace

    config = _grok_oidc_auth_persist_config(
        grok_oidc_refresh_enabled=False,
        codex_oauth_refresh_enabled=True,
        codex_auth_file="/home/zepfu/.codex/auth.json",
        codex_auth_file_source="default",
        codex_lock_file="/home/zepfu/.codex/auth.json.lock",
        codex_refresh_interval_seconds=3600.0,
        codex_refresh_buffer_seconds=300,
        codex_force_refresh=False,
        codex_http_timeout_seconds=30.0,
    )
    if overrides:
        config = replace(config, **overrides)
    return config


def _codex_oauth_refresh_sidecar_event(**overrides) -> dict:
    event = {
        "event": "codex_oauth_refresh",
        "observed_at": "2026-06-19T12:00:00Z",
        "environment": "dev",
        "attempted": True,
        "refreshed": True,
        "skipped": False,
        "auth_file": "/home/zepfu/.codex/auth.json",
        "account_id": "acct_refreshed",
        "expires_at": "2026-06-19T13:00:00Z",
        "error_class": None,
        "error_message": None,
    }
    event.update(overrides)
    return event


def test_build_codex_oauth_auth_observation_maps_successful_refresh() -> None:
    config = _codex_oauth_auth_persist_config()
    event = _codex_oauth_refresh_sidecar_event()

    observation = loop._build_codex_auth_observation(config, event)

    assert observation["observed_at"] == datetime(2026, 6, 19, 12, 0, tzinfo=timezone.utc)
    assert observation["environment"] == "dev"
    assert observation["provider"] == "openai"
    assert observation["auth_family"] == "codex_oauth"
    assert observation["credential_scope"] == "acct_refreshed"
    assert observation["auth_file_hash"] == hashlib.sha256(
        event["auth_file"].encode("utf-8")
    ).hexdigest()
    assert observation["status"] == "refreshed"
    assert observation["attempted"] is True
    assert observation["refreshed"] is True
    assert observation["skipped"] is False
    assert observation["expires_at"] == datetime(2026, 6, 19, 13, 0, tzinfo=timezone.utc)
    assert observation["last_success_at"] == observation["observed_at"]
    assert observation["source_task"] == "codex_oauth_refresh"
    assert observation["metadata"]["auth_file_source"] == "default"
    observation_json = json.dumps(observation, default=str)
    assert "refresh_token" not in observation_json
    assert "access_token" not in observation_json
    assert "/home/zepfu/.codex/auth.json" not in observation_json


def test_build_codex_oauth_auth_observation_sanitizes_refresh_failure() -> None:
    config = _codex_oauth_auth_persist_config()
    event = _codex_oauth_refresh_sidecar_event(
        refreshed=False,
        skipped=False,
        expires_at=None,
        error_class="ValueError",
        error_message=(
            "token refresh failed with refresh_token=super-secret "
            "and Authorization=Bearer leaked-token"
        ),
    )

    observation = loop._build_codex_auth_observation(config, event)

    assert observation["status"] == "failed"
    assert observation["last_success_at"] is None
    assert observation["error_class"] == "ValueError"
    assert "REDACTED" in observation["error_message"]
    assert "super-secret" not in json.dumps(observation, default=str)
    assert "leaked-token" not in json.dumps(observation, default=str)


def test_run_due_sidecar_tasks_persists_codex_auth_observation_when_apply_enabled(
    monkeypatch,
) -> None:
    config = _codex_oauth_auth_persist_config(
        codex_force_refresh=True,
        codex_refresh_interval_seconds=3600.0,
    )
    captured = {}

    monkeypatch.setattr(
        loop.codex_oauth_refresh,
        "refresh_codex_oauth_auth_file",
        lambda *_args, **_kwargs: {
            "attempted": True,
            "refreshed": True,
            "skipped": False,
            "auth_file": config.codex_auth_file,
            "account_id": "acct_refreshed",
            "expires_at": "2026-06-19T13:00:00Z",
            "error_class": None,
            "error_message": None,
        },
    )

    def fake_persist(persist_config, event):
        captured["config"] = persist_config
        captured["event"] = dict(event)
        return True, 1, None, None

    monkeypatch.setattr(loop, "_persist_codex_auth_observation", fake_persist)

    events = loop.run_due_sidecar_tasks(
        config,
        loop.SidecarTaskState(),
        now_monotonic=100.0,
    )

    refresh_events = [event for event in events if event.get("event") == "codex_oauth_refresh"]
    assert len(refresh_events) == 1
    assert refresh_events[0]["auth_observation_status"] == "refreshed"
    assert refresh_events[0]["auth_observation_persisted"] is True
    assert refresh_events[0]["auth_observation_inserted_count"] == 1
    assert captured["config"] is config
    assert captured["event"]["account_id"] == "acct_refreshed"


def _xai_oauth_auth_persist_config(**overrides):
    from dataclasses import replace

    config = _grok_oidc_auth_persist_config(
        grok_oidc_refresh_enabled=False,
        xai_oauth_refresh_enabled=True,
        xai_oauth_auth_file="/home/zepfu/.litellm/xai/oauth-auth.json",
        xai_oauth_auth_file_source="default",
        xai_oauth_lock_file="/home/zepfu/.litellm/xai/oauth-auth.json.lock",
        xai_oauth_scope=xai_oauth_refresh.DEFAULT_XAI_OAUTH_SCOPE,
        xai_oauth_refresh_interval_seconds=3600.0,
        xai_oauth_refresh_buffer_seconds=300,
        xai_oauth_force_refresh=False,
        xai_oauth_http_timeout_seconds=30.0,
    )
    if overrides:
        config = replace(config, **overrides)
    return config


def _xai_oauth_refresh_sidecar_event(**overrides) -> dict:
    event = {
        "event": "xai_oauth_refresh",
        "observed_at": "2026-06-19T12:00:00Z",
        "environment": "dev",
        "attempted": True,
        "refreshed": True,
        "skipped": False,
        "auth_file": "/home/zepfu/.litellm/xai/oauth-auth.json",
        "scope": xai_oauth_refresh.DEFAULT_XAI_OAUTH_SCOPE,
        "expires_at": "2026-06-19T13:00:00Z",
        "error_class": None,
        "error_message": None,
    }
    event.update(overrides)
    return event


def test_build_xai_oauth_auth_observation_maps_successful_refresh() -> None:
    config = _xai_oauth_auth_persist_config()
    event = _xai_oauth_refresh_sidecar_event()

    observation = loop._build_xai_oauth_auth_observation(config, event)

    assert observation["observed_at"] == datetime(2026, 6, 19, 12, 0, tzinfo=timezone.utc)
    assert observation["environment"] == "dev"
    assert observation["provider"] == "xai"
    assert observation["auth_family"] == "xai_oauth"
    assert observation["credential_scope"] == event["scope"]
    assert observation["auth_file_hash"] == hashlib.sha256(
        event["auth_file"].encode("utf-8")
    ).hexdigest()
    assert observation["status"] == "refreshed"
    assert observation["attempted"] is True
    assert observation["refreshed"] is True
    assert observation["skipped"] is False
    assert observation["expires_at"] == datetime(2026, 6, 19, 13, 0, tzinfo=timezone.utc)
    assert observation["last_success_at"] == observation["observed_at"]
    assert observation["source_task"] == "xai_oauth_refresh"
    assert observation["metadata"]["auth_file_source"] == "default"
    observation_json = json.dumps(observation, default=str)
    assert "refresh_token" not in observation_json
    assert "access_token" not in observation_json
    assert "/home/zepfu/.litellm/xai/oauth-auth.json" not in observation_json


def test_build_xai_oauth_auth_observation_sanitizes_refresh_failure() -> None:
    config = _xai_oauth_auth_persist_config()
    event = _xai_oauth_refresh_sidecar_event(
        refreshed=False,
        skipped=False,
        expires_at=None,
        error_class="ValueError",
        error_message=(
            "token refresh failed with refresh_token=super-secret "
            "and Authorization=Bearer leaked-token"
        ),
    )

    observation = loop._build_xai_oauth_auth_observation(config, event)

    assert observation["status"] == "failed"
    assert observation["last_success_at"] is None
    assert observation["error_class"] == "ValueError"
    assert "REDACTED" in observation["error_message"]
    assert "super-secret" not in json.dumps(observation, default=str)
    assert "leaked-token" not in json.dumps(observation, default=str)


def test_run_due_sidecar_tasks_persists_xai_oauth_auth_observation_when_apply_enabled(
    monkeypatch,
) -> None:
    config = _xai_oauth_auth_persist_config(
        xai_oauth_force_refresh=True,
        xai_oauth_refresh_interval_seconds=3600.0,
    )
    captured = {}

    monkeypatch.setattr(
        loop.xai_oauth_refresh,
        "refresh_xai_oauth_auth_file",
        lambda *_args, **_kwargs: {
            "attempted": True,
            "refreshed": True,
            "skipped": False,
            "auth_file": config.xai_oauth_auth_file,
            "scope": config.xai_oauth_scope,
            "expires_at": "2026-06-19T13:00:00Z",
            "error_class": None,
            "error_message": None,
        },
    )

    def fake_persist(persist_config, event):
        captured["config"] = persist_config
        captured["event"] = dict(event)
        return True, 1, None, None

    monkeypatch.setattr(loop, "_persist_xai_oauth_auth_observation", fake_persist)

    events = loop.run_due_sidecar_tasks(
        config,
        loop.SidecarTaskState(),
        now_monotonic=100.0,
    )

    refresh_events = [event for event in events if event.get("event") == "xai_oauth_refresh"]
    assert len(refresh_events) == 1
    assert refresh_events[0]["auth_observation_status"] == "refreshed"
    assert refresh_events[0]["auth_observation_persisted"] is True
    assert refresh_events[0]["auth_observation_inserted_count"] == 1
    assert captured["config"] is config
    assert captured["event"]["scope"] == config.xai_oauth_scope



def test_run_due_sidecar_tasks_persists_grok_oidc_auth_observation_when_apply_enabled(
    monkeypatch,
) -> None:
    config = _grok_oidc_auth_persist_config(
        grok_oidc_force_refresh=True,
        grok_oidc_refresh_interval_seconds=3600.0,
    )
    captured = {}

    monkeypatch.setattr(
        loop.grok_oidc_refresh,
        "repair_grok_oidc_auth_file_metadata",
        lambda *_args, **_kwargs: {
            "attempted": True,
            "repaired": False,
            "auth_file": config.grok_oidc_auth_file,
            "error_class": None,
            "error_message": None,
        },
    )
    monkeypatch.setattr(
        loop.grok_oidc_refresh,
        "refresh_grok_oidc_auth_file",
        lambda *_args, **_kwargs: {
            "attempted": True,
            "refreshed": True,
            "skipped": False,
            "auth_file": config.grok_oidc_auth_file,
            "scope": "https://auth.x.ai::b1a00492-073a-47ea-816f-4c329264a828",
            "expires_at": "2026-06-19T13:00:00Z",
            "error_class": None,
            "error_message": None,
        },
    )

    def fake_persist(cfg, event):
        captured["config"] = cfg
        captured["event"] = event
        return True, 1, None, None

    monkeypatch.setattr(loop, "_persist_grok_oidc_auth_observation", fake_persist)

    events = loop.run_due_sidecar_tasks(
        config,
        loop.SidecarTaskState(),
        now_monotonic=100.0,
    )

    refresh_events = [event for event in events if event.get("event") == "grok_oidc_refresh"]
    assert len(refresh_events) == 1
    assert captured["config"] is config
    assert captured["event"]["refreshed"] is True
    assert refresh_events[0]["auth_observation_status"] == "refreshed"
    assert refresh_events[0]["auth_observation_persisted"] is True
    assert refresh_events[0]["auth_observation_inserted_count"] == 1
    assert refresh_events[0]["auth_observation_skip_error_class"] is None
    assert refresh_events[0]["auth_observation_skip_reason"] is None
    assert "access-token" not in json.dumps(refresh_events)


def test_run_due_sidecar_tasks_marks_grok_oidc_auth_persistence_failure(
    monkeypatch,
) -> None:
    config = _grok_oidc_auth_persist_config(
        grok_oidc_force_refresh=True,
        grok_oidc_refresh_interval_seconds=3600.0,
    )

    monkeypatch.setattr(
        loop.grok_oidc_refresh,
        "repair_grok_oidc_auth_file_metadata",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        loop.grok_oidc_refresh,
        "refresh_grok_oidc_auth_file",
        lambda *_args, **_kwargs: {
            "attempted": True,
            "refreshed": True,
            "skipped": False,
            "auth_file": config.grok_oidc_auth_file,
            "scope": "scope",
            "expires_at": "2026-06-19T13:00:00Z",
            "error_class": None,
            "error_message": None,
        },
    )
    monkeypatch.setattr(
        loop,
        "_persist_grok_oidc_auth_observation",
        lambda *_args, **_kwargs: (False, 0, "OperationalError", "connection refused"),
    )

    events = loop.run_due_sidecar_tasks(
        config,
        loop.SidecarTaskState(),
        now_monotonic=100.0,
    )

    refresh_event = next(
        event for event in events if event.get("event") == "grok_oidc_refresh"
    )
    assert refresh_event["auth_observation_status"] == "refreshed"
    assert refresh_event["auth_observation_persisted"] is False
    assert refresh_event["auth_observation_inserted_count"] == 0
    assert refresh_event["auth_observation_skip_error_class"] == "OperationalError"
    assert refresh_event["auth_observation_skip_reason"] == "connection refused"


def _codex_reset_credit_poll_config(**overrides):
    from dataclasses import replace

    config = loop.ProviderStatusLoopConfig(
        apply=True,
        dsn="postgresql://example/db",
        environment="dev",
        interval_seconds=300.0,
        timeout=2.0,
        ping_count=1,
        ping_timeout=2,
        skip_icmp=False,
        once=True,
        setup_schema=False,
        db_lock_timeout_ms=1000,
        db_statement_timeout_ms=5000,
        grok_oidc_refresh_enabled=False,
        codex_oauth_refresh_enabled=False,
        codex_auth_file="/home/zepfu/.codex/auth.json",
        codex_auth_file_source="AAWM_CODEX_AUTH_FILE",
        codex_reset_credit_poll_enabled=True,
        codex_reset_credit_poll_interval_seconds=3600.0,
        codex_reset_credit_poll_http_timeout_seconds=30.0,
        codex_usage_url="https://chatgpt.com/backend-api/wham/rate-limit-reset-credits",
        codex_reset_credit_poll_max_attempts=3,
        codex_reset_credit_poll_retry_backoff_seconds=0.5,
        grok_billing_poll_enabled=False,
    )
    if overrides:
        config = replace(config, **overrides)
    return config


def _codex_reset_credit_payload_snake(**overrides) -> dict:
    payload = {
        "rate_limit_reset_credits": {
            "available_count": 2,
        }
    }
    payload.update(overrides)
    return payload




def _codex_reset_credit_payload_detail(**overrides) -> dict:
    granted_a = datetime(2026, 6, 24, 18, 0, tzinfo=timezone.utc)
    granted_b = datetime(2026, 6, 25, 12, 0, tzinfo=timezone.utc)
    payload = {
        "credits": [
            {
                "id": "credit-visible-1",
                "status": "available",
                "reset_type": "refer_a_friend",
                "granted_at": granted_a.isoformat().replace("+00:00", "Z"),
                "expires_at": "2026-07-24T18:00:00Z",
            },
            {
                "status": "available",
                "reset_type": "refer_a_friend",
                "granted_at": granted_b.isoformat().replace("+00:00", "Z"),
                "expires_at": "2026-07-25T12:00:00Z",
            },
        ]
    }
    payload.update(overrides)
    return payload

def _codex_reset_credit_auth_context(**overrides) -> dict:
    context = {
        "access_token": "access-token-secret",
        "account_id": "acct-openai-primary",
    }
    context.update(overrides)
    return context


def test_parse_codex_reset_credit_available_count_accepts_snake_and_camel() -> None:
    assert (
        loop._parse_codex_reset_credit_available_count(
            _codex_reset_credit_payload_snake()
        )
        == 2
    )
    assert (
        loop._parse_codex_reset_credit_available_count(
            {"rateLimitResetCredits": {"availableCount": 5}}
        )
        == 5
    )


def test_parse_codex_reset_credit_expires_at_ignores_missing_credit_expiry() -> None:
    assert (
        loop._parse_codex_reset_credit_expires_at(
            {"rate_limit_reset_credits": {"available_count": 1}}
        )
        is None
    )


def test_build_codex_reset_credit_request_headers_includes_account_id_without_secrets() -> None:
    headers = loop._build_codex_reset_credit_request_headers(
        _codex_reset_credit_auth_context()
    )

    assert headers["authorization"] == "Bearer access-token-secret"
    assert headers["ChatGPT-Account-Id"] == "acct-openai-primary"
    headers_json = json.dumps(headers)
    assert "refresh_token" not in headers_json
    assert "acct-openai-primary" in headers_json


def test_account_identity_hash_uses_stable_short_sha256_prefix() -> None:
    assert probes.account_identity_hash("acct-openai-primary") == hashlib.sha256(
        b"acct-openai-primary"
    ).hexdigest()[:12]


def test_provider_credit_insert_sql_dedupes_unchanged_snapshots() -> None:
    sql = probes.PROVIDER_CREDIT_OBSERVATIONS_INSERT_SQL
    view_sql = probes.PROVIDER_CREDIT_CURRENT_VIEW_SQL

    assert "latest.available_count IS NOT DISTINCT FROM candidate.available_count" in sql
    assert "latest.granted_at IS NOT DISTINCT FROM candidate.granted_at" in sql
    assert "latest.expires_at IS NOT DISTINCT FROM candidate.expires_at" in sql
    assert "latest.status IS NOT DISTINCT FROM candidate.status" in sql
    assert "latest.credit_identity IS NOT DISTINCT FROM candidate.credit_identity" in sql
    assert "latest.operator_annotation IS NOT DISTINCT FROM candidate.operator_annotation" in sql
    assert "latest.source_url IS NOT DISTINCT FROM candidate.source_url" in sql
    assert "latest.raw_provider_fields IS NOT DISTINCT FROM" not in sql
    assert "credit_identity," in view_sql
    assert "operator_annotation," in view_sql
    assert "granted_at," in view_sql
    assert "detail_credit.credit_identity <> ''" in view_sql


def test_build_codex_reset_credit_observation_keeps_raw_fields_narrow() -> None:
    observation = loop._build_codex_reset_credit_observation(
        _codex_reset_credit_poll_config(),
        observed_at=datetime(2026, 6, 27, 12, 0, tzinfo=timezone.utc),
        response_body={
            "rate_limit_reset_credits": {"available_count": 2},
            "plan": "plus",
            "account_id": "acct-openai-primary",
            "email": "operator@example.com",
        },
        auth_context=_codex_reset_credit_auth_context(),
        status_code=200,
        attempt_count=1,
        retry_count=0,
    )

    assert observation["raw_provider_fields"] == {
        "rate_limit_reset_credits": {"available_count": 2}
    }
    raw_json = json.dumps(observation["raw_provider_fields"])
    assert "plan" not in raw_json
    assert "acct-openai-primary" not in raw_json
    assert "operator@example.com" not in raw_json


def test_insert_provider_credit_observations_returns_changed_rowcount(monkeypatch) -> None:
    fake_conn = _FakeProviderStatusConnection()
    fake_conn.cursor_instance.rowcount = 0
    monkeypatch.setattr(probes.psycopg, "connect", lambda _dsn: fake_conn)

    inserted = probes.insert_provider_credit_observations(
        "postgresql://example/db",
        [
            {
                "observed_at": datetime(2026, 6, 27, 12, 0, tzinfo=timezone.utc),
                "environment": "dev",
                "provider": "openai",
                "account_hash": "abc123def456",
                "credit_family": "codex_rate_limit_reset",
                "credit_type": "reset_credit",
                "credit_identity": "legacy-aggregate",
                "available_count": 2,
                "granted_at": None,
                "status": "available",
                "expires_at": None,
                "raw_provider_fields": {"rate_limit_reset_credits": {"available_count": 2}},
                "evidence": {"signals": ["codex_reset_credit_poll"]},
                "source": "codex_reset_credit_poll",
            }
        ],
    )

    assert inserted == 0
    insert_sql, payload = fake_conn.cursor_instance.execute_calls[3]
    assert insert_sql == probes.PROVIDER_CREDIT_OBSERVATIONS_INSERT_SQL
    assert payload[7] == 2

    fake_conn.cursor_instance.rowcount = 1
    inserted_changed = probes.insert_provider_credit_observations(
        "postgresql://example/db",
        [
            {
                "observed_at": datetime(2026, 6, 27, 13, 0, tzinfo=timezone.utc),
                "environment": "dev",
                "provider": "openai",
                "account_hash": "abc123def456",
                "credit_family": "codex_rate_limit_reset",
                "credit_type": "reset_credit",
                "credit_identity": "legacy-aggregate",
                "available_count": 1,
                "granted_at": None,
                "status": "available",
                "expires_at": None,
                "raw_provider_fields": {"rate_limit_reset_credits": {"available_count": 1}},
                "evidence": {"signals": ["codex_reset_credit_poll"]},
                "source": "codex_reset_credit_poll",
            }
        ],
    )
    assert inserted_changed == 1


def test_loop_config_reads_codex_reset_credit_poll_env_defaults(monkeypatch) -> None:
    monkeypatch.setenv("AAWM_CODEX_RESET_CREDIT_POLL_ENABLED", "1")
    monkeypatch.setenv("AAWM_CODEX_RESET_CREDIT_POLL_INTERVAL_SECONDS", "7200")
    monkeypatch.setenv("AAWM_CODEX_RESET_CREDIT_POLL_HTTP_TIMEOUT_SECONDS", "45")
    monkeypatch.setenv(
        "AAWM_CODEX_USAGE_URL",
        "https://chatgpt.com/backend-api/wham/usage?lane=dev",
    )
    monkeypatch.setenv("AAWM_CODEX_RESET_CREDIT_POLL_MAX_ATTEMPTS", "5")
    monkeypatch.setenv("AAWM_CODEX_RESET_CREDIT_POLL_RETRY_BACKOFF_SECONDS", "1.25")

    config = loop.parse_config([])

    assert config.codex_reset_credit_poll_enabled is True
    assert config.codex_reset_credit_poll_interval_seconds == 7200.0
    assert config.codex_reset_credit_poll_http_timeout_seconds == 45.0
    assert config.codex_usage_url == "https://chatgpt.com/backend-api/wham/usage?lane=dev"
    assert config.codex_reset_credit_poll_max_attempts == 5
    assert config.codex_reset_credit_poll_retry_backoff_seconds == 1.25


def test_run_due_sidecar_tasks_skips_when_codex_reset_credit_poll_disabled(monkeypatch) -> None:
    config = _codex_reset_credit_poll_config(codex_reset_credit_poll_enabled=False)

    monkeypatch.setattr(
        loop,
        "_fetch_codex_reset_credit_payload",
        lambda *_args, **_kwargs: pytest.fail("Codex reset-credit poll should not run"),
    )

    assert loop.run_due_sidecar_tasks(
        config,
        loop.SidecarTaskState(),
        now_monotonic=100.0,
    ) == []


def test_run_due_sidecar_tasks_throttles_codex_reset_credit_poll(monkeypatch) -> None:
    config = _codex_reset_credit_poll_config(apply=False)
    calls = {"fetch": 0}

    monkeypatch.setattr(
        loop,
        "_fetch_codex_reset_credit_payload",
        lambda *_args, **_kwargs: (
            calls.__setitem__("fetch", calls["fetch"] + 1)
            or {
                "status_code": 200,
                "payload": _codex_reset_credit_payload_snake(),
                "auth_context": _codex_reset_credit_auth_context(),
                "attempt_count": 1,
                "retry_count": 0,
                "poll_url": "https://chatgpt.com/backend-api/wham/rate-limit-reset-credits",
            }
        ),
    )

    state = loop.SidecarTaskState()
    first_events = loop.run_due_sidecar_tasks(config, state, now_monotonic=100.0)
    second_events = loop.run_due_sidecar_tasks(config, state, now_monotonic=200.0)
    third_events = loop.run_due_sidecar_tasks(config, state, now_monotonic=3701.0)

    assert calls == {"fetch": 2}
    assert first_events[-1]["event"] == "codex_reset_credit_poll"
    assert first_events[-1]["available_count"] == 2
    assert first_events[-1]["inserted_count"] == 0
    assert second_events == []
    assert third_events[-1]["event"] == "codex_reset_credit_poll"


def test_run_due_sidecar_tasks_emits_codex_reset_credit_poll_event(monkeypatch) -> None:
    config = _codex_reset_credit_poll_config()

    monkeypatch.setattr(
        loop,
        "_fetch_codex_reset_credit_payload",
        lambda *_args, **_kwargs: {
            "status_code": 200,
            "payload": _codex_reset_credit_payload_snake(),
            "auth_context": _codex_reset_credit_auth_context(),
            "attempt_count": 1,
            "retry_count": 0,
            "poll_url": "https://chatgpt.com/backend-api/wham/rate-limit-reset-credits",
        },
    )
    monkeypatch.setattr(
        loop,
        "_persist_codex_reset_credit_observation",
        lambda *_args, **_kwargs: (1, 1),
    )

    events = loop.run_due_sidecar_tasks(
        config,
        loop.SidecarTaskState(),
        now_monotonic=100.0,
    )

    poll_events = [
        event for event in events if event.get("event") == "codex_reset_credit_poll"
    ]
    assert len(poll_events) == 1
    assert poll_events[0]["persisted"] is True
    assert poll_events[0]["available_count"] == 2
    assert poll_events[0]["inserted_count"] == 1
    assert poll_events[0]["status_code"] == 200
    event_json = json.dumps(poll_events)
    assert "access-token-secret" not in event_json
    assert "acct-openai-primary" not in event_json
    assert '"account_id"' not in event_json


def test_compose_wires_codex_reset_credit_poll_defaults() -> None:
    compose_text = Path("/home/zepfu/projects/litellm/docker-compose.dev.yml").read_text()

    assert (
        "AAWM_CODEX_RESET_CREDIT_POLL_ENABLED=${AAWM_CODEX_RESET_CREDIT_POLL_ENABLED:-1}"
        in compose_text
    )
    assert (
        "AAWM_CODEX_RESET_CREDIT_POLL_INTERVAL_SECONDS=${AAWM_CODEX_RESET_CREDIT_POLL_INTERVAL_SECONDS:-3600}"
        in compose_text
    )
    assert (
        "AAWM_CODEX_USAGE_URL=${AAWM_CODEX_USAGE_URL:-https://chatgpt.com/backend-api/wham/rate-limit-reset-credits}"
        in compose_text
    )


def test_resolve_codex_reset_credit_poll_url_maps_legacy_usage_to_detail() -> None:
    config = _codex_reset_credit_poll_config(
        codex_usage_url="https://chatgpt.com/backend-api/wham/usage"
    )
    assert (
        loop._resolve_codex_reset_credit_poll_url(config)
        == probes.DEFAULT_CODEX_RESET_CREDIT_DETAIL_URL
    )


def test_parse_codex_reset_credit_detail_credits_builds_per_credit_rows() -> None:
    config = _codex_reset_credit_poll_config()
    observed_at = datetime(2026, 6, 28, 12, 0, tzinfo=timezone.utc)
    rows = loop._build_codex_reset_credit_observations(
        config,
        observed_at=observed_at,
        response_body=_codex_reset_credit_payload_detail(),
        auth_context=_codex_reset_credit_auth_context(),
        status_code=200,
        attempt_count=1,
        retry_count=0,
        poll_url=probes.DEFAULT_CODEX_RESET_CREDIT_DETAIL_URL,
    )

    assert len(rows) == 2
    assert all(row["status"] == "available" for row in rows)
    assert rows[0]["credit_identity"] == "credit-visible-1"
    assert rows[0]["granted_at"] == datetime(2026, 6, 24, 18, 0, tzinfo=timezone.utc)
    assert rows[1]["credit_identity"] != rows[0]["credit_identity"]
    identity_b = probes.derive_provider_credit_identity(
        account_hash=rows[1]["account_hash"],
        credit_family="codex_rate_limit_reset",
        granted_at=rows[1]["granted_at"],
        expires_at=rows[1]["expires_at"],
        reset_type="refer_a_friend",
        provider_credit_id=None,
    )
    assert rows[1]["credit_identity"] == identity_b
    newest = max(rows, key=lambda row: row["granted_at"])
    assert newest["source_url"] == loop.DEFAULT_CODEX_RESET_CREDIT_LATEST_VISIBLE_SOURCE_URL
    event_json = json.dumps(rows, default=str)
    assert "access-token-secret" not in event_json
    assert "acct-openai-primary" not in event_json


def test_derive_provider_credit_identity_is_stable_without_provider_id() -> None:
    granted = datetime(2026, 6, 24, 18, 0, tzinfo=timezone.utc)
    expires = datetime(2026, 7, 24, 18, 0, tzinfo=timezone.utc)
    first = probes.derive_provider_credit_identity(
        account_hash="abc123def456",
        credit_family="codex_rate_limit_reset",
        granted_at=granted,
        expires_at=expires,
        reset_type="refer_a_friend",
        provider_credit_id=None,
    )
    second = probes.derive_provider_credit_identity(
        account_hash="abc123def456",
        credit_family="codex_rate_limit_reset",
        granted_at=granted,
        expires_at=expires,
        reset_type="refer_a_friend",
        provider_credit_id=None,
    )
    assert first == second
    assert len(first) == 16


def test_apply_provider_credit_seed_metadata_matches_invite_promotion_grant() -> None:
    row = {
        "granted_at": datetime(2026, 6, 24, 21, 53, tzinfo=timezone.utc),
        "expires_at": datetime(2026, 7, 24, 21, 53, tzinfo=timezone.utc),
        "operator_annotation": None,
        "source_url": None,
    }
    updated = probes.apply_provider_credit_seed_metadata(row)
    assert updated["operator_annotation"] == "Invite Promotion"


def test_apply_provider_credit_seed_metadata_matches_visible_invite_promotion_credit() -> None:
    row = {
        "granted_at": datetime(
            2026, 6, 24, 22, 41, 38, 714466, tzinfo=timezone.utc
        ),
        "expires_at": datetime(
            2026, 7, 24, 22, 41, 38, 714466, tzinfo=timezone.utc
        ),
        "operator_annotation": None,
        "source_url": None,
    }
    updated = probes.apply_provider_credit_seed_metadata(row)
    assert updated["operator_annotation"] == "Invite Promotion"


def test_apply_provider_credit_seed_metadata_sets_source_url_for_june_12_credit() -> None:
    row = {
        "granted_at": datetime(2026, 6, 12, 16, 17, tzinfo=timezone.utc),
        "expires_at": datetime(2026, 7, 12, 16, 17, tzinfo=timezone.utc),
        "operator_annotation": None,
        "source_url": None,
    }
    updated = probes.apply_provider_credit_seed_metadata(row)
    assert (
        updated["source_url"]
        == "https://x.com/thsottiaux/status/2065468501750649006"
    )


def test_build_codex_reset_credit_seed_observations_emits_missing_seed_rows() -> None:
    config = _codex_reset_credit_poll_config()
    account_hash = probes.account_identity_hash("acct-openai-primary")
    rows = loop._build_codex_reset_credit_seed_observations(
        config,
        observed_at=datetime(2026, 6, 28, 12, 0, tzinfo=timezone.utc),
        account_hash=account_hash,
        visible_identities=set(),
        status_code=200,
        attempt_count=1,
        retry_count=0,
        poll_url=probes.DEFAULT_CODEX_RESET_CREDIT_DETAIL_URL,
    )

    assert len(rows) == len(probes.CODEX_RESET_CREDIT_SEED_METADATA)
    by_grant = {row["granted_at"]: row for row in rows}
    assert by_grant[datetime(2026, 6, 24, 21, 53, tzinfo=timezone.utc)][
        "status"
    ] == "used"
    assert by_grant[datetime(2026, 6, 24, 21, 53, tzinfo=timezone.utc)][
        "operator_annotation"
    ] == "Invite Promotion"
    assert by_grant[datetime(2026, 6, 12, 16, 17, tzinfo=timezone.utc)][
        "source_url"
    ] == "https://x.com/thsottiaux/status/2065468501750649006"
    assert all(row["available_count"] == 0 for row in rows)
    assert all(row["evidence"]["seed_backfill"] is True for row in rows)


def test_build_codex_reset_credit_seed_observations_skips_visible_seed_identity() -> None:
    config = _codex_reset_credit_poll_config()
    account_hash = probes.account_identity_hash("acct-openai-primary")
    granted_at = datetime(2026, 6, 24, 22, 41, 38, 714466, tzinfo=timezone.utc)
    expires_at = datetime(2026, 7, 24, 22, 41, 38, 714466, tzinfo=timezone.utc)
    visible_identity = probes.derive_provider_credit_identity(
        account_hash=account_hash,
        credit_family="codex_rate_limit_reset",
        granted_at=granted_at,
        expires_at=expires_at,
        reset_type="codex_rate_limits",
        provider_credit_id=None,
    )

    rows = loop._build_codex_reset_credit_seed_observations(
        config,
        observed_at=datetime(2026, 6, 28, 12, 0, tzinfo=timezone.utc),
        account_hash=account_hash,
        visible_identities={visible_identity},
        status_code=200,
        attempt_count=1,
        retry_count=0,
        poll_url=probes.DEFAULT_CODEX_RESET_CREDIT_DETAIL_URL,
    )

    assert visible_identity not in {row["credit_identity"] for row in rows}


def test_build_codex_reset_credit_seed_observations_skips_visible_grant_window() -> None:
    config = _codex_reset_credit_poll_config()
    account_hash = probes.account_identity_hash("acct-openai-primary")
    granted_at = datetime(2026, 6, 24, 22, 41, 38, 714466, tzinfo=timezone.utc)
    expires_at = datetime(2026, 7, 24, 22, 41, 38, 714466, tzinfo=timezone.utc)

    rows = loop._build_codex_reset_credit_seed_observations(
        config,
        observed_at=datetime(2026, 6, 28, 12, 0, tzinfo=timezone.utc),
        account_hash=account_hash,
        visible_identities={"RateLimitResetCredit_provider_id"},
        visible_credit_windows={(granted_at, expires_at)},
        status_code=200,
        attempt_count=1,
        retry_count=0,
        poll_url=probes.DEFAULT_CODEX_RESET_CREDIT_DETAIL_URL,
    )

    assert all(row["granted_at"] != granted_at for row in rows)


def test_visible_past_expiry_credit_is_marked_expired() -> None:
    rows = loop._build_codex_reset_credit_observations(
        _codex_reset_credit_poll_config(),
        observed_at=datetime(2026, 8, 1, 12, 0, tzinfo=timezone.utc),
        response_body={
            "credits": [
                {
                    "status": "available",
                    "reset_type": "codex_rate_limits",
                    "granted_at": "2026-06-24T12:00:00Z",
                    "expires_at": "2026-07-24T12:00:00Z",
                }
            ]
        },
        auth_context=_codex_reset_credit_auth_context(),
        status_code=200,
        attempt_count=1,
        retry_count=0,
        poll_url=probes.DEFAULT_CODEX_RESET_CREDIT_DETAIL_URL,
    )

    assert rows[0]["status"] == "expired"
    assert rows[0]["available_count"] == 0


def test_synthesize_codex_reset_credit_lifecycle_marks_missing_credit_used(
    monkeypatch,
) -> None:
    config = _codex_reset_credit_poll_config()
    observed_at = datetime(2026, 6, 28, 12, 0, tzinfo=timezone.utc)
    missing_identity = "missing-credit-id"
    monkeypatch.setattr(
        loop,
        "_resolve_dsn",
        lambda _config: "postgresql://example/db",
    )
    monkeypatch.setattr(
        probes,
        "load_provider_credit_current_rows",
        lambda *_args, **_kwargs: [
            {
                "credit_identity": missing_identity,
                "credit_type": "reset_credit",
                "status": "available",
                "granted_at": datetime(2026, 6, 24, 21, 53, tzinfo=timezone.utc),
                "expires_at": datetime(2026, 7, 24, 21, 53, tzinfo=timezone.utc),
                "redeem_started_at": None,
                "redeemed_at": None,
                "operator_annotation": "Invite Promotion",
                "source_url": None,
                "raw_provider_fields": {"credit": {"status": "available"}},
                "evidence": {"signals": ["codex_reset_credit_poll"]},
            }
        ],
    )

    rows = loop._synthesize_codex_reset_credit_lifecycle_observations(
        config,
        observed_at=observed_at,
        account_hash=probes.account_identity_hash("acct-openai-primary"),
        visible_identities={"still-visible"},
        status_code=200,
        attempt_count=1,
        retry_count=0,
        poll_url=probes.DEFAULT_CODEX_RESET_CREDIT_DETAIL_URL,
    )

    assert len(rows) == 1
    assert rows[0]["status"] == "used"
    assert rows[0]["credit_identity"] == missing_identity
    assert rows[0]["available_count"] == 0
    assert rows[0]["evidence"]["lifecycle_reason"] == "credit_missing_before_expiry"


def test_synthesize_codex_reset_credit_lifecycle_marks_past_expiry_expired(
    monkeypatch,
) -> None:
    config = _codex_reset_credit_poll_config()
    observed_at = datetime(2026, 8, 1, 12, 0, tzinfo=timezone.utc)
    monkeypatch.setattr(loop, "_resolve_dsn", lambda _config: "postgresql://example/db")
    monkeypatch.setattr(
        probes,
        "load_provider_credit_current_rows",
        lambda *_args, **_kwargs: [
            {
                "credit_identity": "expired-credit",
                "credit_type": "reset_credit",
                "status": "available",
                "granted_at": datetime(2026, 6, 1, 12, 0, tzinfo=timezone.utc),
                "expires_at": datetime(2026, 7, 1, 12, 0, tzinfo=timezone.utc),
                "redeem_started_at": None,
                "redeemed_at": None,
                "operator_annotation": None,
                "source_url": None,
                "raw_provider_fields": {},
                "evidence": {},
            }
        ],
    )

    rows = loop._synthesize_codex_reset_credit_lifecycle_observations(
        config,
        observed_at=observed_at,
        account_hash="abc123def456",
        visible_identities=set(),
        status_code=200,
        attempt_count=1,
        retry_count=0,
        poll_url=probes.DEFAULT_CODEX_RESET_CREDIT_DETAIL_URL,
    )

    assert len(rows) == 1
    assert rows[0]["status"] == "expired"
    assert rows[0]["evidence"]["lifecycle_reason"] == "credit_past_expiry"


def test_parse_codex_reset_credit_credit_entry_prefers_redeemed_at_for_used() -> None:
    parsed = loop._parse_codex_reset_credit_credit_entry(
        {
            "status": "available",
            "granted_at": "2026-06-20T12:00:00Z",
            "expires_at": "2026-07-20T12:00:00Z",
            "redeemed_at": "2026-06-21T10:00:00Z",
        }
    )
    assert parsed["status"] == "used"
    assert parsed["redeemed_at"] == datetime(2026, 6, 21, 10, 0, tzinfo=timezone.utc)
