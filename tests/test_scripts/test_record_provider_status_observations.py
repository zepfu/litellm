from argparse import Namespace
from datetime import datetime, timezone
from pathlib import Path
from subprocess import TimeoutExpired

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
        "AAWM_GROK_OIDC_REFRESH_INTERVAL_SECONDS=${AAWM_GROK_OIDC_REFRESH_INTERVAL_SECONDS:-3600}"
        in compose_text
    )
    assert "AAWM_GROK_OIDC_FORCE_REFRESH=${AAWM_GROK_OIDC_FORCE_REFRESH:-1}" in compose_text


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
        {"provider": "control", "probe_type": "dns", "success": True},
        {"provider": "control", "probe_type": "tls_handshake", "success": False},
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
