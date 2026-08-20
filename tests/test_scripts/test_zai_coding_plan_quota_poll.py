"""Failing contract tests for Z.AI Coding Plan quota polling (ZAI-012).

These tests encode the sidecar poll contract. They must stay red until
``scripts/run_provider_status_observations_loop.py`` grows the named helpers.
Do not call live ``api.z.ai``. Do not print or persist host ``ZAI_KEY``.
"""

from __future__ import annotations

import hashlib
import inspect
import json
from dataclasses import fields
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

import pytest

from scripts import run_provider_status_observations_loop as loop


REPO_ROOT = Path(__file__).resolve().parents[2]
DEV_COMPOSE_PATH = REPO_ROOT / "docker-compose.dev.yml"
WSL_COMPOSE_PATH = REPO_ROOT / "docker-compose.wsl-grok-oidc.yml"

EXPECTED_QUOTA_URL = "https://api.z.ai/api/monitor/usage/quota/limit"
EXPECTED_SUBSCRIPTION_URL = "https://api.z.ai/api/biz/subscription/list"
BARE_USAGE_404_URL = "https://api.z.ai/api/monitor/usage"

FAKE_ZAI_KEY = "test-zai-coding-plan-key"
FAKE_ZAI_CODING_PLAN_API_KEY = "test-zai-coding-plan-alias-key"
FAKE_ORDINARY_ZAI_API_KEY = "ordinary-zai-api-key-must-not-be-used"
CUSTOMER_ID = "56091784974160639"
EXPECTED_ACCOUNT_HASH = hashlib.sha256(
    f"zai_coding_plan|customerId={CUSTOMER_ID}".encode("utf-8")
).hexdigest()

OPENQUOTA_TOKENS_LIMIT_FIXTURE = {
    "code": 200,
    "msg": "Operation successful",
    "data": {
        "limits": [
            {
                "type": "TOKENS_LIMIT",
                "unit": 3,
                "number": 5,
                "percentage": 17,
                "nextResetTime": 1782724971179,
            },
            {
                "type": "TOKENS_LIMIT",
                "unit": 6,
                "number": 1,
                "percentage": 3,
                "nextResetTime": 1783305486997,
            },
            {
                "type": "TIME_LIMIT",
                "unit": 5,
                "number": 1,
                "usage": 1000,
                "currentValue": 0,
                "remaining": 1000,
                "percentage": 0,
                "nextResetTime": 1785292686976,
            },
        ],
        "level": "pro",
    },
    "success": True,
}

LIVE_CREDIT_LIMIT_FIXTURE = {
    "code": 200,
    "msg": "Operation successful",
    "data": {
        "limits": [
            {
                "type": "CREDIT_LIMIT",
                "unit": 3,
                "number": 5,
                "usage": 12000,
                "currentValue": 0,
                "remaining": 12000,
                "percentage": 0,
            },
            {
                "type": "CREDIT_LIMIT",
                "unit": 6,
                "number": 1,
                "usage": 60000,
                "currentValue": 0,
                "remaining": 59999,
                "percentage": 1,
                "nextResetTime": 1787831598997,
            },
        ],
        "level": "pro",
    },
    "success": True,
}

SUBSCRIPTION_FIXTURE = {
    "code": 200,
    "msg": "Operation successful",
    "data": [
        {
            "productName": "GLM Coding Pro",
            "status": "VALID",
            "customerId": CUSTOMER_ID,
            "billingCycle": "quarterly",
            "inCurrentPeriod": True,
        }
    ],
    "success": True,
}

NO_PLAN_FIXTURE = {
    "code": 400,
    "msg": "This account does not have a coding plan subscription",
    "data": None,
    "success": False,
}

MALFORMED_LIMITS_FIXTURE = {
    "code": 200,
    "msg": "Operation successful",
    "data": {
        "limits": [
            {"type": "TOKENS_LIMIT", "unit": 3, "number": 5},
            {"type": "CREDIT_LIMIT", "unit": 6, "number": 1},
        ],
        "level": "pro",
    },
    "success": True,
}


class _FakeHttpResponse:
    def __init__(self, payload: dict, *, status: int = 200) -> None:
        self.status = status
        self._body = json.dumps(payload).encode("utf-8")

    def getcode(self) -> int:
        return self.status

    def read(self) -> bytes:
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None


def _clear_zai_credential_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "ZAI_KEY",
        "ZAI_CODING_PLAN_API_KEY",
        "ZAI_API_KEY",
        "ZHIPU_API_KEY",
        "AAWM_ZAI_CODING_PLAN_QUOTA_POLL_ENABLED",
        "AAWM_ZAI_CODING_PLAN_QUOTA_URL",
        "AAWM_ZAI_CODING_PLAN_SUBSCRIPTION_URL",
    ):
        monkeypatch.delenv(name, raising=False)


def _base_loop_config(**overrides):
    kwargs = {
        "apply": False,
        "dsn": "postgresql://aawm:aawm_dev@pgbouncer:6432/aawm_tristore",
        "environment": "dev",
        "interval_seconds": 300.0,
        "timeout": 2.0,
        "ping_count": 1,
        "ping_timeout": 2,
        "skip_icmp": False,
        "once": True,
        "setup_schema": False,
        "db_lock_timeout_ms": 1000,
        "db_statement_timeout_ms": 5000,
    }
    kwargs.update(overrides)
    return loop.ProviderStatusLoopConfig(**kwargs)


def _zai_coding_plan_quota_poll_config(**overrides):
    values = {"zai_coding_plan_quota_poll_enabled": True}
    values.update(overrides)
    return _base_loop_config(**values)


def _payload_fields(payload: tuple) -> dict:
    return {
        "observed_at": payload[0],
        "client": payload[1],
        "client_version": payload[2],
        "account_hash": payload[3],
        "provider": payload[4],
        "model": payload[5],
        "quota_key": payload[6],
        "quota_period": payload[7],
        "quota_type": payload[8],
        "expected_reset_at": payload[9],
        "remaining_pct": payload[10],
        "quota_limit": payload[11],
        "quota_used": payload[12],
        "quota_remaining": payload[13],
        "raw_provider_fields": json.loads(payload[16]),
        "evidence": json.loads(payload[17]),
        "source": payload[18],
    }


def _build_quota_payloads(usage_payload: dict, subscription_payload=None, **kwargs):
    return loop._build_zai_coding_plan_quota_rate_limit_payloads(
        _zai_coding_plan_quota_poll_config(),
        observed_at=datetime(2026, 8, 20, 12, 0, tzinfo=timezone.utc),
        usage_payload=usage_payload,
        subscription_payload=subscription_payload,
        **kwargs,
    )


def _serialized_payloads(payloads: list[tuple]) -> str:
    blobs = []
    for payload in payloads:
        blobs.append(payload[16] if isinstance(payload[16], str) else json.dumps(payload[16]))
        blobs.append(payload[17] if isinstance(payload[17], str) else json.dumps(payload[17]))
    return "\n".join(blobs)


def _assert_no_secret_material(blob: str, extra: tuple[str, ...] = ()) -> None:
    forbidden = (
        FAKE_ZAI_KEY,
        FAKE_ZAI_CODING_PLAN_API_KEY,
        FAKE_ORDINARY_ZAI_API_KEY,
        CUSTOMER_ID,
        *extra,
    )
    for secret in forbidden:
        assert secret not in blob


def test_default_zai_coding_plan_quota_poll_constant_is_off() -> None:
    assert loop.DEFAULT_ZAI_CODING_PLAN_QUOTA_POLL_ENABLED is False


def test_provider_status_loop_config_defaults_zai_coding_plan_quota_poll_off() -> None:
    field_names = {item.name for item in fields(loop.ProviderStatusLoopConfig)}
    assert "zai_coding_plan_quota_poll_enabled" in field_names
    assert _base_loop_config().zai_coding_plan_quota_poll_enabled is False


def test_cli_and_env_default_zai_coding_plan_quota_poll_off(monkeypatch) -> None:
    _clear_zai_credential_env(monkeypatch)
    help_text = loop._build_parser().format_help()
    config = loop.parse_config([])

    assert "AAWM_ZAI_CODING_PLAN_QUOTA_POLL_ENABLED" in help_text
    assert "--zai-coding-plan-quota-poll-enabled" in help_text
    assert "--no-zai-coding-plan-quota-poll" in help_text
    assert config.zai_coding_plan_quota_poll_enabled is False


def test_env_can_enable_zai_coding_plan_quota_poll(monkeypatch) -> None:
    _clear_zai_credential_env(monkeypatch)
    monkeypatch.setenv("AAWM_ZAI_CODING_PLAN_QUOTA_POLL_ENABLED", "1")

    config = loop.parse_config([])

    assert config.zai_coding_plan_quota_poll_enabled is True


def test_cli_can_disable_zai_coding_plan_quota_poll_when_env_is_on(monkeypatch) -> None:
    _clear_zai_credential_env(monkeypatch)
    monkeypatch.setenv("AAWM_ZAI_CODING_PLAN_QUOTA_POLL_ENABLED", "1")

    config = loop.parse_config(["--no-zai-coding-plan-quota-poll"])

    assert config.zai_coding_plan_quota_poll_enabled is False


def test_disabled_zai_coding_plan_quota_poll_task_returns_none_without_http(
    monkeypatch,
) -> None:
    _clear_zai_credential_env(monkeypatch)
    http_calls: list[object] = []

    def fake_urlopen(request, timeout=None):
        http_calls.append((request, timeout))
        raise AssertionError("disabled Z.AI coding-plan quota poll must not HTTP")

    monkeypatch.setattr(loop.urllib_request, "urlopen", fake_urlopen)
    config = _zai_coding_plan_quota_poll_config(
        zai_coding_plan_quota_poll_enabled=False,
    )

    result = loop._run_zai_coding_plan_quota_poll_task(
        config,
        loop.SidecarTaskState(),
        now_monotonic=100.0,
    )

    assert result is None
    assert http_calls == []


def test_default_zai_coding_plan_quota_urls_use_openquota_paths() -> None:
    assert loop.DEFAULT_ZAI_CODING_PLAN_QUOTA_URL == EXPECTED_QUOTA_URL
    assert loop.DEFAULT_ZAI_CODING_PLAN_SUBSCRIPTION_URL == EXPECTED_SUBSCRIPTION_URL
    assert loop.DEFAULT_ZAI_CODING_PLAN_QUOTA_URL != BARE_USAGE_404_URL
    assert not loop.DEFAULT_ZAI_CODING_PLAN_QUOTA_URL.rstrip("/").endswith(
        "/api/monitor/usage"
    )
    loop_src = (
        REPO_ROOT / "scripts" / "run_provider_status_observations_loop.py"
    ).read_text(encoding="utf-8")
    assert EXPECTED_QUOTA_URL in loop_src
    assert EXPECTED_SUBSCRIPTION_URL in loop_src
    assert f'"{BARE_USAGE_404_URL}"' not in loop_src
    assert f"'{BARE_USAGE_404_URL}'" not in loop_src


def test_credential_order_prefers_zai_key_then_coding_plan_alias(monkeypatch) -> None:
    _clear_zai_credential_env(monkeypatch)
    monkeypatch.setenv("ZAI_KEY", FAKE_ZAI_KEY)
    monkeypatch.setenv("ZAI_CODING_PLAN_API_KEY", FAKE_ZAI_CODING_PLAN_API_KEY)
    monkeypatch.setenv("ZAI_API_KEY", FAKE_ORDINARY_ZAI_API_KEY)

    resolved = loop._resolve_zai_coding_plan_quota_api_key()
    assert resolved == FAKE_ZAI_KEY

    monkeypatch.delenv("ZAI_KEY", raising=False)
    resolved_alias = loop._resolve_zai_coding_plan_quota_api_key()
    assert resolved_alias == FAKE_ZAI_CODING_PLAN_API_KEY


def test_credential_resolver_does_not_reuse_ordinary_zai_api_key(monkeypatch) -> None:
    _clear_zai_credential_env(monkeypatch)
    monkeypatch.setenv("ZAI_API_KEY", FAKE_ORDINARY_ZAI_API_KEY)
    resolver = loop._resolve_zai_coding_plan_quota_api_key
    assert callable(resolver)

    with pytest.raises(Exception) as exc_info:
        resolver()

    assert not isinstance(exc_info.value, AttributeError)
    rendered = str(exc_info.value)
    assert FAKE_ORDINARY_ZAI_API_KEY not in rendered
    telemetry_class = getattr(exc_info.value, "telemetry_class", None)
    if telemetry_class is not None:
        assert telemetry_class == "auth"


def test_enabled_poll_sends_bearer_authorization_and_openquota_urls(
    monkeypatch,
) -> None:
    _clear_zai_credential_env(monkeypatch)
    monkeypatch.setenv("ZAI_KEY", FAKE_ZAI_KEY)
    captured: list[object] = []

    def fake_urlopen(request, timeout=None):
        captured.append(request)
        url = getattr(request, "full_url", None) or getattr(request, "url", "")
        if urlsplit(url).path.endswith("/api/biz/subscription/list"):
            return _FakeHttpResponse(SUBSCRIPTION_FIXTURE)
        if urlsplit(url).path.endswith("/api/monitor/usage/quota/limit"):
            return _FakeHttpResponse(LIVE_CREDIT_LIMIT_FIXTURE)
        raise AssertionError(f"unexpected Z.AI poll URL: {url}")

    monkeypatch.setattr(loop.urllib_request, "urlopen", fake_urlopen)
    monkeypatch.setattr(
        loop,
        "_persist_zai_coding_plan_quota_observations",
        lambda *_args, **_kwargs: 0,
    )
    event = loop._run_zai_coding_plan_quota_poll_task(
        _zai_coding_plan_quota_poll_config(),
        loop.SidecarTaskState(),
        now_monotonic=100.0,
    )

    assert event is not None
    assert event["event"] == "zai_coding_plan_quota_poll"
    assert captured
    urls = [
        getattr(request, "full_url", None) or getattr(request, "url", "")
        for request in captured
    ]
    assert EXPECTED_QUOTA_URL in urls
    assert EXPECTED_SUBSCRIPTION_URL in urls
    assert BARE_USAGE_404_URL not in urls
    authorizations = {
        request.get_header("Authorization") or request.headers.get("Authorization")
        for request in captured
    }
    assert authorizations == {f"Bearer {FAKE_ZAI_KEY}"}
    serialized = json.dumps(event)
    _assert_no_secret_material(serialized)


def test_tokens_limit_fixture_emits_5h_and_weekly_percent_windows() -> None:
    payloads = _build_quota_payloads(
        OPENQUOTA_TOKENS_LIMIT_FIXTURE,
        SUBSCRIPTION_FIXTURE,
    )
    rows = [_payload_fields(payload) for payload in payloads]
    by_period = {row["quota_period"]: row for row in rows if row["quota_type"] != "credits"}
    five_hour = next(row for row in rows if row["quota_period"] == "5h")
    weekly = next(row for row in rows if row["quota_period"] == "7d")

    assert five_hour["remaining_pct"] == 83.0
    assert five_hour["quota_limit"] is None
    assert five_hour["quota_used"] is None
    assert five_hour["quota_remaining"] is None
    assert five_hour["provider"] == "zai_coding_plan"
    assert five_hour["source"] == "zai_coding_plan_quota_poll"
    assert weekly["remaining_pct"] == 97.0
    assert weekly["quota_limit"] is None
    assert weekly["quota_used"] is None
    assert weekly["quota_remaining"] is None
    serialized = _serialized_payloads(payloads)
    assert "invoice" not in serialized.lower()
    assert "usd" not in serialized.lower()
    time_limit_rows = [
        row
        for row in rows
        if row["quota_limit"] == 1000
        and row["quota_used"] == 0
        and row["quota_remaining"] == 1000
    ]
    if time_limit_rows:
        assert time_limit_rows[0]["remaining_pct"] == 100.0
    assert five_hour["quota_period"] == "5h"
    assert weekly["quota_period"] == "7d"
    assert by_period["5h"]["remaining_pct"] == 83.0
    assert by_period["7d"]["remaining_pct"] == 97.0


def test_credit_limit_fixture_emits_absolute_5h_and_weekly_windows() -> None:
    payloads = _build_quota_payloads(
        LIVE_CREDIT_LIMIT_FIXTURE,
        SUBSCRIPTION_FIXTURE,
    )
    rows = [_payload_fields(payload) for payload in payloads]
    five_hour = next(
        row
        for row in rows
        if row["quota_period"] == "5h" and row["quota_type"] == "credits"
    )
    weekly = next(
        row
        for row in rows
        if row["quota_period"] == "7d" and row["quota_type"] == "credits"
    )

    assert five_hour["quota_key"] == "zai_coding_plan_5h:credits"
    assert five_hour["quota_limit"] == 12000
    assert five_hour["quota_used"] == 0
    assert five_hour["quota_remaining"] == 12000
    assert five_hour["remaining_pct"] == 100.0
    assert five_hour["expected_reset_at"] is None
    assert five_hour["provider"] == "zai_coding_plan"
    assert five_hour["source"] == "zai_coding_plan_quota_poll"
    assert weekly["quota_key"] == "zai_coding_plan_7d:credits"
    assert weekly["quota_limit"] == 60000
    assert weekly["quota_used"] == 1
    assert weekly["quota_remaining"] == 59999
    assert weekly["remaining_pct"] == 99.0
    assert weekly["provider"] == "zai_coding_plan"
    assert weekly["source"] == "zai_coding_plan_quota_poll"


def test_account_hash_uses_hashed_customer_id_not_plaintext() -> None:
    payloads = _build_quota_payloads(
        LIVE_CREDIT_LIMIT_FIXTURE,
        SUBSCRIPTION_FIXTURE,
    )
    rows = [_payload_fields(payload) for payload in payloads]
    hashes = {row["account_hash"] for row in rows}

    assert hashes == {EXPECTED_ACCOUNT_HASH}
    serialized = _serialized_payloads(payloads)
    _assert_no_secret_material(serialized)
    assert CUSTOMER_ID not in json.dumps(rows, default=str)
    helper_hash = loop._hash_zai_coding_plan_account_identity(
        subscription_payload=SUBSCRIPTION_FIXTURE,
        api_key=FAKE_ZAI_KEY,
    )
    assert helper_hash == EXPECTED_ACCOUNT_HASH
    assert helper_hash != hashlib.sha256(FAKE_ZAI_KEY.encode("utf-8")).hexdigest()


def test_no_plan_response_does_not_invent_full_remaining_payloads() -> None:
    builder = loop._build_zai_coding_plan_quota_rate_limit_payloads
    assert callable(builder)
    try:
        payloads = _build_quota_payloads(NO_PLAN_FIXTURE, SUBSCRIPTION_FIXTURE)
    except Exception as exc:
        assert not isinstance(exc, AttributeError)
        telemetry_class = getattr(exc, "telemetry_class", None)
        assert telemetry_class in {None, "no_plan", "contract_drift"}
        assert "100" not in str(exc)
        return

    assert payloads == []


def test_malformed_limits_do_not_silently_invent_windows() -> None:
    builder = loop._build_zai_coding_plan_quota_rate_limit_payloads
    assert callable(builder)
    try:
        payloads = _build_quota_payloads(
            MALFORMED_LIMITS_FIXTURE,
            SUBSCRIPTION_FIXTURE,
        )
    except Exception as exc:
        assert not isinstance(exc, AttributeError)
        telemetry_class = getattr(exc, "telemetry_class", None)
        assert telemetry_class in {None, "malformed_telemetry", "contract_drift"}
        return

    rows = [_payload_fields(payload) for payload in payloads]
    assert rows == []
    invented = [
        row
        for row in rows
        if row["remaining_pct"] in {0.0, 100.0}
        or row["quota_limit"] not in {None}
    ]
    assert invented == []


def test_missing_coding_plan_credentials_are_auth_without_http(monkeypatch) -> None:
    _clear_zai_credential_env(monkeypatch)
    monkeypatch.setenv("ZAI_API_KEY", FAKE_ORDINARY_ZAI_API_KEY)
    http_calls: list[object] = []

    def fake_urlopen(request, timeout=None):
        http_calls.append((request, timeout))
        raise AssertionError("missing coding-plan credentials must not HTTP")

    monkeypatch.setattr(loop.urllib_request, "urlopen", fake_urlopen)
    event = loop._run_zai_coding_plan_quota_poll_task(
        _zai_coding_plan_quota_poll_config(),
        loop.SidecarTaskState(),
        now_monotonic=100.0,
    )

    assert event is not None
    assert event["event"] == "zai_coding_plan_quota_poll"
    assert event.get("error_class")
    assert event.get("telemetry_class") == "auth" or "auth" in str(
        event.get("error_class", "")
    ).lower()
    assert http_calls == []
    serialized = json.dumps(event)
    _assert_no_secret_material(serialized)


def test_compose_files_default_zai_coding_plan_quota_poll_off() -> None:
    dev_text = DEV_COMPOSE_PATH.read_text(encoding="utf-8")
    wsl_text = WSL_COMPOSE_PATH.read_text(encoding="utf-8")

    assert "AAWM_ZAI_CODING_PLAN_QUOTA_POLL_ENABLED" in dev_text
    assert "AAWM_ZAI_CODING_PLAN_QUOTA_POLL_ENABLED" in wsl_text
    assert (
        "AAWM_ZAI_CODING_PLAN_QUOTA_POLL_ENABLED=${AAWM_ZAI_CODING_PLAN_QUOTA_POLL_ENABLED:-0}"
        in dev_text
    )
    assert "AAWM_ZAI_CODING_PLAN_QUOTA_POLL_ENABLED=0" in wsl_text
    assert (
        "AAWM_ZAI_CODING_PLAN_QUOTA_POLL_ENABLED=${AAWM_ZAI_CODING_PLAN_QUOTA_POLL_ENABLED:-1}"
        not in dev_text
    )


def test_run_due_sidecar_tasks_registers_zai_coding_plan_quota_poll() -> None:
    source = inspect.getsource(loop.run_due_sidecar_tasks)
    assert "_run_zai_coding_plan_quota_poll_task" in source
    assert '"zai_coding_plan_quota_poll"' in source or "'zai_coding_plan_quota_poll'" in source


def test_run_due_sidecar_tasks_invokes_registered_zai_poll(monkeypatch) -> None:
    called = {}

    def fake_task(config, state, *, now_monotonic):
        called["config"] = config
        called["state"] = state
        called["now_monotonic"] = now_monotonic
        return {
            "event": "zai_coding_plan_quota_poll",
            "attempted": True,
            "skipped": True,
        }

    monkeypatch.setattr(loop, "_run_zai_coding_plan_quota_poll_task", fake_task)
    events = loop.run_due_sidecar_tasks(
        _zai_coding_plan_quota_poll_config(),
        loop.SidecarTaskState(),
        now_monotonic=100.0,
    )
    matching = [
        event for event in events if event.get("event") == "zai_coding_plan_quota_poll"
    ]
    assert matching
    assert called["now_monotonic"] == 100.0
