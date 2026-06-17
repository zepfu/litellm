import json
from io import BytesIO
from argparse import Namespace
from datetime import datetime, timezone
from pathlib import Path
from subprocess import TimeoutExpired
from urllib import error as urllib_error

import pytest

from litellm.integrations import aawm_agent_identity
from scripts import record_provider_status_observations as probes
from scripts import run_provider_status_observations_loop as loop


def _provider_status_row() -> dict:
    return {
        "observed_at": datetime(2026, 5, 14, 12, 0, tzinfo=timezone.utc),
        "environment": "dev",
        "provider": "control",
        "endpoint_key": "control:google.com",
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
        "provider": "control",
        "endpoint_key": "control:google.com",
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
    assert payload[2] == "control"
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
        "LITELLM_XAI_GROK_CLIENT_VERSION",
        "LITELLM_XAI_GROK_CLIENT_IDENTIFIER",
        "LITELLM_XAI_GROK_XAI_TOKEN_AUTH",
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
    compose_text = (Path(__file__).resolve().parents[2] / "docker-compose.dev.yml").read_text()

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
    assert (
        "LITELLM_XAI_GROK_AUTH_FILE=${LITELLM_XAI_GROK_AUTH_FILE:-/home/zepfu/.grok/auth.json}"
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
            "endpoint_key": "control:google.com",
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
            "endpoint_key": "control:google.com",
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
        "billing_url",
        "client_version",
        "model",
        "observation_count",
        "inserted_count",
        "status_code",
        "attempt_count",
        "retry_count",
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

    assert "content-type" not in headers
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
