from argparse import Namespace
from datetime import datetime, timezone
from subprocess import TimeoutExpired

import pytest

from litellm.integrations import aawm_agent_identity
from scripts import record_provider_status_observations as probes
from scripts import run_provider_status_observations_loop as loop


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

    def fake_insert_observations(dsn, payload_rows):
        inserted["dsn"] = dsn
        inserted["rows"] = payload_rows

    monkeypatch.setattr(loop.probes, "collect_observations", fake_collect_observations)
    monkeypatch.setattr(loop.probes, "_build_dsn", lambda _args: "postgresql://example/db")
    monkeypatch.setattr(loop.probes, "insert_observations", fake_insert_observations)

    summary = loop.run_cycle(config)

    assert inserted == {"dsn": "postgresql://example/db", "rows": rows}
    assert summary["event"] == "provider_status_observations_cycle"
    assert summary["apply"] is True
    assert summary["inserted"] is True
    assert summary["environment"] == "dev"
    assert summary["row_count"] == 2
    assert summary["success_count"] == 1
    assert summary["failure_count"] == 1


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
    )

    monkeypatch.setattr(loop.probes, "collect_observations", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(loop.probes, "_build_dsn", lambda _args: None)

    with pytest.raises(RuntimeError, match="No database DSN found"):
        loop.run_cycle(config)
