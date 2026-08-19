"""CURSOR-008: mocked Cursor Agent monthly usage poller."""

from __future__ import annotations

import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType

import pytest

_REPO = Path(__file__).resolve().parents[3]
_SCRIPT = _REPO / "scripts" / "run_provider_status_observations_loop.py"


def _load() -> ModuleType:
    name = "run_provider_status_observations_loop_cursor_008"
    if name in sys.modules:
        del sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, _SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def loop() -> ModuleType:
    return _load()


def _config(loop: ModuleType, tmp_path: Path, **overrides):
    from dataclasses import replace

    base = loop.ProviderStatusLoopConfig(
        apply=False,
        dsn=None,
        environment="dev",
        interval_seconds=300.0,
        timeout=2.0,
        ping_count=1,
        ping_timeout=2,
        skip_icmp=True,
        once=True,
        setup_schema=False,
        db_lock_timeout_ms=1000,
        db_statement_timeout_ms=5000,
        observability_anomaly_scan_enabled=False,
        observability_anomaly_scan_error_log_dir=str(tmp_path),
        cursor_agent_usage_poll_enabled=True,
    )
    if overrides:
        base = replace(base, **overrides)
    return base


def _camelcase_usage_payload() -> dict:
    return {
        "billingCycleStart": 1754524800,
        "billingCycleEnd": 1757203200,
        "planUsage": {
            "totalSpend": 4200,
            "includedSpend": 1250,
            "remaining": 8750,
            "limit": 10000,
            "autoPercentUsed": 41,
            "apiPercentUsed": 7,
            "totalPercentUsed": 48,
        },
        "user": {"id": "acct-not-for-persistence"},
    }


class _UsageResponse:
    def __init__(self, body: bytes, *, status: int = 200) -> None:
        self.body = body
        self.status = status

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def getcode(self):
        return self.status

    def read(self):
        return self.body


def test_cursor_agent_usage_poll_is_disabled_by_default(loop, monkeypatch) -> None:
    for name in (
        "AAWM_CURSOR_AGENT_USAGE_POLL_ENABLED",
        "AAWM_CURSOR_AGENT_USAGE_POLL_INTERVAL_SECONDS",
        "AAWM_CURSOR_AGENT_USAGE_POLL_HTTP_TIMEOUT_SECONDS",
        "AAWM_CURSOR_AGENT_USAGE_DASHBOARD_URL",
        "CURSOR_AUTH_TOKEN",
        "CURSOR_API_KEY",
        "CURSOR_CLI_KEY",
    ):
        monkeypatch.delenv(name, raising=False)

    config = loop.parse_config([])

    assert config.cursor_agent_usage_poll_enabled is False
    assert config.cursor_agent_usage_poll_interval_seconds == 3600.0
    assert config.cursor_agent_usage_poll_http_timeout_seconds == 30.0
    assert config.cursor_agent_usage_dashboard_url == "https://api2.cursor.sh"
    assert loop.run_due_sidecar_tasks(config, loop.SidecarTaskState(), now_monotonic=1.0) == []


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (
            ["--cursor-agent-usage-poll-interval-seconds", "0"],
            "--cursor-agent-usage-poll-interval-seconds must be greater than 0",
        ),
        (
            ["--cursor-agent-usage-poll-http-timeout-seconds", "0"],
            "--cursor-agent-usage-poll-http-timeout-seconds must be greater than 0",
        ),
        (
            ["--cursor-agent-usage-dashboard-url", ""],
            "--cursor-agent-usage-dashboard-url must not be empty",
        ),
    ],
)
def test_cursor_agent_usage_poll_config_validation(loop, argv, message) -> None:
    with pytest.raises(SystemExit, match=message):
        loop.parse_config(argv)


def test_cursor_agent_usage_payload_maps_monthly_included_spend(loop, tmp_path) -> None:
    config = _config(loop, tmp_path)
    observed_at = datetime(2026, 8, 19, 12, 0, tzinfo=timezone.utc)
    payloads, summary = loop._build_cursor_agent_usage_rate_limit_payloads(
        config,
        observed_at=observed_at,
        response_body=_camelcase_usage_payload(),
    )

    assert summary["telemetry_status"] == "valid"
    assert summary["weekly_grok_bot"] == "unknown"
    assert summary["weekly_grok_bot_quota_key"] is None
    assert len(payloads) == 1
    row = payloads[0]
    assert row[4] == "cursor_agent"
    assert row[6] == "cursor_agent_monthly:cents"
    assert row[7] == "monthly"
    assert row[8] == "cents"
    assert row[11] == 10000.0
    assert row[12] == 1250.0
    assert row[13] == 8750.0
    assert row[18] == "cursor_agent_usage"
    evidence = json.loads(row[17])
    assert evidence["weekly_grok_bot"] == "unknown"
    serialized = json.dumps(row, default=str)
    assert "acct-not-for-persistence" not in serialized
    assert "CURSOR_AUTH_TOKEN" not in serialized


def test_cursor_agent_usage_fetch_uses_mocked_connect_json(
    loop, tmp_path, monkeypatch
) -> None:
    captured = {}

    def fake_urlopen(request, timeout=None):
        captured["url"] = request.full_url
        captured["method"] = request.get_method()
        captured["timeout"] = timeout
        captured["headers"] = {key.lower(): value for key, value in request.header_items()}
        captured["body"] = request.data
        return _UsageResponse(json.dumps(_camelcase_usage_payload()).encode("utf-8"))

    monkeypatch.setenv("CURSOR_AUTH_TOKEN", "stored-access-token")
    monkeypatch.setenv("CURSOR_API_KEY", "raw-api-key")
    monkeypatch.setenv("CURSOR_CLI_KEY", "cli-key-must-be-ignored")
    monkeypatch.setattr(loop.urllib_request, "urlopen", fake_urlopen)

    config = _config(loop, tmp_path)
    fetched = loop._fetch_cursor_agent_usage_payload(config)

    assert fetched["status_code"] == 200
    assert fetched["payload"]["planUsage"]["includedSpend"] == 1250
    assert captured["url"] == (
        "https://api2.cursor.sh/aiserver.v1.DashboardService/GetCurrentPeriodUsage"
    )
    assert captured["method"] == "POST"
    assert captured["headers"]["authorization"] == "Bearer stored-access-token"
    assert captured["headers"]["connect-protocol-version"] == "1"
    assert captured["body"] == b"{}"
    assert "agentn" not in captured["url"]
    assert "/v0/me" not in captured["url"]


def test_cursor_agent_usage_failed_refresh_keeps_last_good_state(
    loop, tmp_path, monkeypatch
) -> None:
    persist_calls = []

    def boom(_config):
        raise loop.CursorAgentUsagePollError(
            "Cursor Agent usage poll failed with HTTP 503.",
            status_code=503,
            telemetry_class="upstream",
            attempt_count=1,
            retry_count=0,
        )

    monkeypatch.setattr(loop, "_fetch_cursor_agent_usage_payload", boom)
    monkeypatch.setattr(
        loop,
        "_persist_cursor_agent_usage_observations",
        lambda *_args, **_kwargs: persist_calls.append(True) or 1,
    )

    config = _config(loop, tmp_path, apply=True)
    event = loop._run_cursor_agent_usage_poll_task(
        config,
        loop.SidecarTaskState(),
        now_monotonic=1.0,
    )

    assert event["event"] == "cursor_agent_usage_poll"
    assert event["persisted"] is False
    assert event["last_good_state_retained"] is True
    assert event["status_code"] == 503
    assert event["weekly_grok_bot"] == "unknown"
    assert persist_calls == []
    assert "stored-access-token" not in json.dumps(event)


def test_cursor_agent_usage_does_not_log_credentials(
    loop, tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("CURSOR_AUTH_TOKEN", "super-secret-access-token")

    def fake_urlopen(request, timeout=None):
        raise loop.urllib_error.HTTPError(
            request.full_url,
            401,
            "Bearer super-secret-access-token rejected",
            {},
            None,
        )

    monkeypatch.setattr(loop.urllib_request, "urlopen", fake_urlopen)
    event = loop._run_cursor_agent_usage_poll_task(
        _config(loop, tmp_path),
        loop.SidecarTaskState(),
        now_monotonic=1.0,
    )
    serialized = json.dumps(event)
    assert "super-secret-access-token" not in serialized
    assert event["last_good_state_retained"] is True
    assert event["telemetry_class"] == "auth"


def test_cursor_agent_usage_missing_account_identity_does_not_persist(
    loop, tmp_path
) -> None:
    payloads, summary = loop._build_cursor_agent_usage_rate_limit_payloads(
        _config(loop, tmp_path),
        observed_at=datetime(2026, 8, 19, 12, 0, tzinfo=timezone.utc),
        response_body={
            "planUsage": {"includedSpend": 1, "limit": 10, "remaining": 9}
        },
    )
    assert payloads == []
    assert summary["telemetry_status"] == "missing_account_identity"


def test_cursor_agent_usage_poll_respects_interval(loop, tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        loop,
        "_fetch_cursor_agent_usage_payload",
        lambda _config: {"status_code": 200, "payload": _camelcase_usage_payload()},
    )
    config = _config(loop, tmp_path, cursor_agent_usage_poll_interval_seconds=3600.0)
    state = loop.SidecarTaskState()
    first = loop._run_cursor_agent_usage_poll_task(config, state, now_monotonic=1.0)
    second = loop._run_cursor_agent_usage_poll_task(config, state, now_monotonic=10.0)
    assert first["observation_count"] == 1
    assert second is None
