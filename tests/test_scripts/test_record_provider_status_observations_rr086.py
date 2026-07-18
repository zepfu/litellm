"""Focused RR-086 coverage for missing/unusable ICMP ping handling."""

from __future__ import annotations

from datetime import datetime, timezone
from subprocess import TimeoutExpired

from scripts import record_provider_status_observations as probes


def _endpoint() -> probes.Endpoint:
    return probes.Endpoint("anthropic", "api.anthropic.com:443", "api.anthropic.com")


def _observed_at() -> datetime:
    return datetime(2026, 7, 17, 18, 0, tzinfo=timezone.utc)


def test_icmp_probe_records_file_not_found_without_raising(monkeypatch) -> None:
    def fake_run(*_args, **_kwargs):
        raise FileNotFoundError(2, "No such file or directory", "ping")

    monkeypatch.setattr(probes.subprocess, "run", fake_run)

    row = probes._icmp_probe(
        _endpoint(),
        environment="dev",
        observed_at=_observed_at(),
        count=1,
        timeout=2,
    )

    assert row["success"] is False
    assert row["probe_type"] == "icmp_ping"
    assert row["error_class"] == "FileNotFoundError"
    assert row["error_message"]
    assert len(row["error_message"]) <= 300
    assert "No such file or directory" in row["error_message"] or "ping" in row["error_message"]
    assert row["total_ms"] is not None


def test_icmp_probe_records_permission_error_without_raising(monkeypatch) -> None:
    def fake_run(*_args, **_kwargs):
        raise PermissionError(13, "Permission denied", "ping")

    monkeypatch.setattr(probes.subprocess, "run", fake_run)

    row = probes._icmp_probe(
        _endpoint(),
        environment="dev",
        observed_at=_observed_at(),
        count=1,
        timeout=2,
    )

    assert row["success"] is False
    assert row["error_class"] == "PermissionError"
    assert row["error_message"]
    assert len(row["error_message"]) <= 300
    assert "Permission denied" in row["error_message"] or "ping" in row["error_message"]
    assert row["total_ms"] is not None


def test_icmp_probe_oserror_message_is_truncated(monkeypatch) -> None:
    long_message = "x" * 500

    def fake_run(*_args, **_kwargs):
        raise OSError(long_message)

    monkeypatch.setattr(probes.subprocess, "run", fake_run)

    row = probes._icmp_probe(
        _endpoint(),
        environment="dev",
        observed_at=_observed_at(),
        count=1,
        timeout=2,
    )

    assert row["success"] is False
    assert row["error_class"] == "OSError"
    assert row["error_message"] == long_message[-300:]
    assert len(row["error_message"]) == 300


def test_collect_observations_continues_when_ping_binary_missing(monkeypatch) -> None:
    def fake_run(*_args, **_kwargs):
        raise FileNotFoundError(2, "No such file or directory", "ping")

    def fake_resolve(endpoint, *, environment, observed_at, timeout):
        row = probes._empty_observation(
            endpoint=endpoint,
            environment=environment,
            observed_at=observed_at,
            probe_type="dns",
        )
        row.update(
            {
                "success": True,
                "address_family": "ipv4",
                "resolved_ip": "1.2.3.4",
                "dns_ms": 1.0,
                "total_ms": 1.0,
            }
        )
        return row, "1.2.3.4", 2

    def fake_tcp(endpoint, *, environment, observed_at, resolved_ip, family, timeout):
        row = probes._empty_observation(
            endpoint=endpoint,
            environment=environment,
            observed_at=observed_at,
            probe_type="tcp_connect",
        )
        row.update(
            {
                "success": True,
                "address_family": "ipv4",
                "resolved_ip": resolved_ip,
                "tcp_ms": 1.0,
                "total_ms": 1.0,
            }
        )
        return row

    def fake_tls(endpoint, *, environment, observed_at, resolved_ip, family, timeout):
        row = probes._empty_observation(
            endpoint=endpoint,
            environment=environment,
            observed_at=observed_at,
            probe_type="tls_handshake",
        )
        row.update(
            {
                "success": True,
                "address_family": "ipv4",
                "resolved_ip": resolved_ip,
                "tls_ms": 1.0,
                "total_ms": 1.0,
            }
        )
        return row

    monkeypatch.setattr(probes.subprocess, "run", fake_run)
    monkeypatch.setattr(probes, "_resolve_host", fake_resolve)
    monkeypatch.setattr(probes, "_tcp_probe", fake_tcp)
    monkeypatch.setattr(probes, "_tls_probe", fake_tls)

    endpoints = (
        probes.Endpoint("anthropic", "api.anthropic.com:443", "api.anthropic.com"),
        probes.Endpoint("openai", "api.openai.com:443", "api.openai.com"),
    )
    rows = probes.collect_observations(
        endpoints,
        environment="dev",
        timeout=1.0,
        ping_count=1,
        ping_timeout=1,
        skip_icmp=False,
    )

    # Each endpoint: icmp + dns + tcp + tls
    assert len(rows) == 8
    icmp_rows = [row for row in rows if row["probe_type"] == "icmp_ping"]
    assert len(icmp_rows) == 2
    assert all(row["success"] is False for row in icmp_rows)
    assert all(row["error_class"] == "FileNotFoundError" for row in icmp_rows)
    assert all(row["error_message"] for row in icmp_rows)
    assert all(len(row["error_message"]) <= 300 for row in icmp_rows)

    non_icmp = [row for row in rows if row["probe_type"] != "icmp_ping"]
    assert len(non_icmp) == 6
    assert all(row["success"] is True for row in non_icmp)


def test_icmp_probe_timeout_path_unchanged(monkeypatch) -> None:
    def fake_run(*_args, **_kwargs):
        raise TimeoutExpired(
            cmd=["ping", "-c", "1", "-W", "2", "api.anthropic.com"],
            timeout=4,
            output="PING api.anthropic.com (160.79.104.10)",
        )

    monkeypatch.setattr(probes.subprocess, "run", fake_run)

    row = probes._icmp_probe(
        _endpoint(),
        environment="dev",
        observed_at=_observed_at(),
        count=1,
        timeout=2,
    )

    assert row["success"] is False
    assert row["error_class"] == "icmp_timeout"
    assert "api.anthropic.com" in row["error_message"]
    assert len(row["error_message"]) <= 300
    assert row["total_ms"] is not None
