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
from urllib.parse import parse_qs, urlsplit

import pytest

from litellm.integrations import aawm_agent_identity
from litellm.llms.xai import oauth
from litellm.secret_managers.codex_oauth_inventory import (
    CodexOAuthCredentialRecord,
    CodexOAuthCredentialSnapshot,
    CodexOAuthInventory,
    codex_oauth_account_identity_hash,
)
from scripts import codex_oauth_refresh
from scripts import grok_oidc_refresh
from scripts import kimi_oauth_refresh
from scripts import record_provider_status_observations as probes
from scripts import run_provider_status_observations_loop as loop
from scripts import xai_oauth_refresh


def _build_test_jwt(payload: dict) -> str:
    def encode_part(value: dict) -> str:
        encoded = base64.urlsafe_b64encode(json.dumps(value).encode("utf-8"))
        return encoded.rstrip(b"=").decode("ascii")

    return f"{encode_part({'alg': 'none'})}.{encode_part(payload)}.sig"


def _codex_oauth_record(
    label: str,
    *,
    declaration_order: int,
    root: Path = Path("/home/zepfu/.codex"),
    enabled: bool = True,
) -> CodexOAuthCredentialRecord:
    account_id = f"acct-{label}"
    return CodexOAuthCredentialRecord(
        label=label,
        auth_path=root / f"oauth.{label}.json",
        lock_path=root / f"oauth.{label}.json.lock",
        priority=(declaration_order + 1) * 10,
        weight=1.0,
        enabled=enabled,
        models=("*",),
        expected_account_hash=codex_oauth_account_identity_hash(account_id),
        declaration_order=declaration_order,
    )


def _codex_oauth_inventory(
    *labels: str,
    root: Path = Path("/home/zepfu/.codex"),
) -> CodexOAuthInventory:
    selected_labels = labels or ("account1", "account2")
    return CodexOAuthInventory(
        records=tuple(
            _codex_oauth_record(
                label,
                declaration_order=index,
                root=root,
            )
            for index, label in enumerate(selected_labels)
        )
    )


def _codex_oauth_inventory_env_payload() -> str:
    inventory = _codex_oauth_inventory()
    return json.dumps(
        {
            "schema_version": 1,
            "accounts": [
                {
                    "label": record.label,
                    "auth_path": str(record.auth_path),
                    "lock_path": str(record.lock_path),
                    "priority": record.priority,
                    "weight": record.weight,
                    "enabled": record.enabled,
                    "models": list(record.models),
                    "expected_account_hash": record.expected_account_hash,
                }
                for record in inventory.records
            ],
        }
    )


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


def test_default_endpoints_include_anthropic_and_openai_front_doors() -> None:
    endpoints = {
        (endpoint.provider, endpoint.endpoint_key, endpoint.host)
        for endpoint in probes.DEFAULT_ENDPOINTS
    }

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
        grok_billing_client_version_source="config",
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


def _clear_grok_billing_version_env(monkeypatch) -> None:
    for env_name in (
        *loop.GROK_BILLING_CLIENT_VERSION_ENV_VARS,
        "AAWM_GROK_CLIENT_VERSION_CACHE_PATH",
        "AAWM_GROK_CLIENT_VERSION_CACHE_MAX_AGE_SECONDS",
    ):
        monkeypatch.delenv(env_name, raising=False)


def _write_grok_billing_version_cache(
    path: Path,
    *,
    version: str,
    observed_at=None,
) -> None:
    observed_at = observed_at or datetime.now(timezone.utc)
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "client": "grok-cli",
                "version": version,
                "build": "a1b2c3d4",
                "channel": "stable",
                "source": "installed-grok-cli",
                "observed_at": observed_at.strftime("%Y-%m-%dT%H:%M:%SZ"),
            }
        ),
        encoding="utf-8",
    )


def _grok_billing_request_headers(config):
    return loop._build_grok_billing_request_headers(
        config,
        access_token="access-token-secret",
        identity_headers=_grok_billing_auth_context()["identity_headers"],
    )


def _grok_billing_fetch_result(
    config,
    *,
    payload=None,
    status_code: int = 200,
    attempt_count: int = 1,
    retry_count: int = 0,
):
    request_headers = _grok_billing_request_headers(config)
    return {
        "status_code": status_code,
        "payload": payload or _grok_billing_payload(),
        "attempt_count": attempt_count,
        "retry_count": retry_count,
        "request_headers": request_headers,
    }


def _alibaba_quota_poll_config(**overrides):
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
        db_lock_timeout_ms=1000,
        db_statement_timeout_ms=5000,
        alibaba_quota_poll_enabled=True,
        alibaba_quota_poll_interval_seconds=300.0,
        alibaba_subscription_poll_interval_seconds=21600.0,
        alibaba_quota_poll_http_timeout_seconds=30.0,
        alibaba_quota_gateway_url=loop.DEFAULT_ALIBABA_QUOTA_GATEWAY_URL,
        alibaba_quota_poll_max_attempts=2,
        alibaba_quota_poll_retry_backoff_seconds=0.5,
    )
    if overrides:
        config = replace(config, **overrides)
    return config


ALIBABA_TEST_RAM_KEY = "LTAI5tTestAccessKeyId"
ALIBABA_TEST_RAM_SECRET = "testAccessKeySecretValue"
ALIBABA_TEST_RAM_PRINCIPAL = "ram-principal-test"
ALIBABA_TEST_ACS_DATE = "2026-08-19T12:00:00Z"
ALIBABA_TEST_SIGNATURE_NONCE = "fixed-nonce-00000000-0000-4000-8000-000000000001"
ALIBABA_TEST_EMPTY_BODY_SHA256 = (
    "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
)
ALIBABA_TEST_ACS3_SIGNATURE = (
    "ebe667be26342cb207e85b1e5c9f74acbc2e8065521a7f7b559f7cc8bf35f562"
)


def _set_alibaba_ram_env(
    monkeypatch,
    *,
    access_key_id: str = ALIBABA_TEST_RAM_KEY,
    access_key_secret: str = ALIBABA_TEST_RAM_SECRET,
    principal: str | None = ALIBABA_TEST_RAM_PRINCIPAL,
) -> None:
    monkeypatch.delenv("ALIBABA_WEB_KEY", raising=False)
    monkeypatch.delenv("AAWM_ALIBABA_WEB_AUTH_FILE", raising=False)
    monkeypatch.setenv("ALIBABA_RAM_KEY", access_key_id)
    monkeypatch.setenv("ALIBABA_RAM_SECRET", access_key_secret)
    if principal is None:
        monkeypatch.delenv("ALIBABA_RAM_PRINCIPAL", raising=False)
    else:
        monkeypatch.setenv("ALIBABA_RAM_PRINCIPAL", principal)


def _clear_alibaba_ram_env(monkeypatch) -> None:
    for name in (
        "ALIBABA_RAM_KEY",
        "ALIBABA_RAM_SECRET",
        "ALIBABA_RAM_PRINCIPAL",
        "ALIBABA_WEB_KEY",
        "AAWM_ALIBABA_WEB_AUTH_FILE",
    ):
        monkeypatch.delenv(name, raising=False)


def _alibaba_ram_auth(monkeypatch, **kwargs):
    _set_alibaba_ram_env(monkeypatch, **kwargs)
    return loop._load_alibaba_ram_auth(_alibaba_quota_poll_config())


class _FakeAlibabaHTTPResponse:
    def __init__(self, payload: bytes | str, *, status: int = 200) -> None:
        self.status = status
        self.payload = payload.encode("utf-8") if isinstance(payload, str) else payload

    def getcode(self) -> int:
        return self.status

    def read(self, size: int = -1) -> bytes:
        return self.payload if size < 0 else self.payload[:size]

    def __enter__(self):
        return self

    def __exit__(self, *args) -> None:
        return None


def _alibaba_http_success(payload) -> _FakeAlibabaHTTPResponse:
    body = payload if isinstance(payload, (bytes, str)) else json.dumps(payload)
    return _FakeAlibabaHTTPResponse(body)


def _alibaba_mint_ok(token: str = "cli-access-token-secret") -> tuple[int, str]:
    return 200, json.dumps({"cliAccessToken": token, "Success": True})


def _alibaba_mint_nopermission() -> tuple[int, str]:
    return 200, json.dumps(
        {
            "Success": False,
            "Code": "NoPermission",
            "Message": "RAM secret-looking-denied-detail",
        }
    )


def _install_alibaba_mint(monkeypatch, *responses):
    remaining = list(responses)
    calls: list[dict] = []

    def fake_mint(*, host, path, headers, body, timeout_seconds):
        calls.append(
            {
                "host": host,
                "path": path,
                "headers": dict(headers),
                "body": body,
                "timeout_seconds": timeout_seconds,
            }
        )
        if not remaining:
            raise AssertionError("unexpected extra Alibaba mint call")
        return remaining.pop(0)

    monkeypatch.setattr(loop, "ALIBABA_MINT_HTTP_POST_FN", fake_mint)
    return calls


def _install_alibaba_quota_open(monkeypatch, handler):
    calls: list = []

    def fake_open(request, timeout=None):
        calls.append(request)
        return handler(request, timeout)

    monkeypatch.setattr(loop, "ALIBABA_QUOTA_HTTP_OPEN_FN", fake_open)
    return calls


def _alibaba_gateway_success_handler(
    *,
    usage_payload=None,
    subscription_payload=None,
    reset_cards=None,
):
    usage = (
        _alibaba_weekly_only_usage_payload()
        if usage_payload is None
        else usage_payload
    )
    subscription = (
        _alibaba_subscription_payload()
        if subscription_payload is None
        else subscription_payload
    )
    cards = [] if reset_cards is None else reset_cards

    def handler(request, timeout):
        del timeout
        api_name = parse_qs(urlsplit(request.full_url).query).get("api", [None])[0]
        if api_name == loop.ALIBABA_TOKEN_PLAN_SUBSCRIPTION_API:
            payload = subscription
        elif api_name == loop.ALIBABA_TOKEN_PLAN_USAGE_API:
            payload = usage
        elif api_name == loop.ALIBABA_TOKEN_PLAN_RESET_CARD_LIST_API:
            payload = cards
        else:
            raise AssertionError(f"unexpected Alibaba gateway api {api_name!r}")
        return _alibaba_http_success(_alibaba_console_envelope(payload))

    return handler


def _assert_no_alibaba_secrets(serialized: str, extra: tuple[str, ...] = ()) -> None:
    forbidden = (
        ALIBABA_TEST_RAM_SECRET,
        ALIBABA_TEST_RAM_PRINCIPAL,
        "cli-access-token-secret",
        "console-bearer-secret",
        "stale-bearer-secret",
        "login-ticket-secret",
        "security-token-secret",
        "sec_token",
        "login_aliyunid_ticket",
        "RAM secret-looking-denied-detail",
        *extra,
    )
    for secret in forbidden:
        assert secret not in serialized, secret


def _alibaba_poll_events(events) -> list[dict]:
    return [event for event in events if event.get("event") == "alibaba_quota_poll"]


def _alibaba_console_envelope(data) -> dict:
    return {
        "data": {
            "DataV2": {
                "data": {
                    "code": "SUCCESS",
                    "success": True,
                    "data": data,
                }
            }
        }
    }


def _alibaba_usage_payload() -> dict:
    return {
        "per5HourPercentage": 0.25,
        "per5HourResetTime": 1784686920000,
        "per1WeekPercentage": 0.5,
        "per1WeekResetTime": 1785083160000,
    }


def _alibaba_weekly_only_usage_payload() -> dict:
    return {
        "per1WeekResetTime": 1786299120000,
        "per1WeekPercentage": 1.0,
    }


def _alibaba_subscription_payload() -> dict:
    return {
        "instanceCode": "instance-secret-identifier",
        "specCode": "pro",
        "remainingDays": 29,
        "startTime": 1784592000000,
        "endTime": 1787184000000,
        "status": "VALID",
    }


def _alibaba_reset_card_payload() -> list[dict]:
    return [
        {
            "cardNo": "reset-card-secret-available",
            "cardType": "WEEKLY",
            "effectiveAt": 1785542400000,
            "expiresAt": 1788220800000,
        },
        {
            "cardNo": "reset-card-secret-expired",
            "cardType": "PROMOTIONAL",
            "effectiveAt": 1782864000000,
            "expiresAt": 1785888000000,
        },
    ]


def _grok_billing_payload() -> dict:
    return {
        "config": {
            "creditUsagePercent": 14.539333,
            "productUsage": [
                {"name": "GrokBuild", "usagePercent": 12.507334},
                {"name": "Api", "usagePercent": 2.032},
            ],
            "currentPeriod": {
                "type": "USAGE_PERIOD_TYPE_WEEKLY",
                "start": "2026-07-03T19:54:47.584112+00:00",
                "end": "2026-07-10T19:54:47.584112+00:00",
            },
            "billingPeriodStart": "2026-07-03T19:54:47.584112+00:00",
            "billingPeriodEnd": "2026-07-10T19:54:47.584112+00:00",
        }
    }


def _grok_billing_weekly_fresh_payload() -> dict:
    return {
        "config": {
            "currentPeriod": {
                "type": "USAGE_PERIOD_TYPE_WEEKLY",
                "start": "2026-07-03T19:54:47.584112+00:00",
                "end": "2026-07-10T19:54:47.584112+00:00",
            },
            "billingPeriodStart": "2026-07-03T19:54:47.584112+00:00",
            "billingPeriodEnd": "2026-07-10T19:54:47.584112+00:00",
        }
    }


def _grok_billing_monthly_counter_payload() -> dict:
    return {
        "config": {
            "monthlyLimit": {"val": 150000},
            "used": {"val": 42910},
            "billingPeriodStart": "2026-07-01T00:00:00+00:00",
            "billingPeriodEnd": "2026-08-01T00:00:00+00:00",
        }
    }


def _grok_billing_legacy_monthly_credit_payload() -> dict:
    return {
        "config": {
            "creditUsagePercent": 27.0,
            "productUsage": [
                {"product": "GrokBuild", "usagePercent": 26.0},
                {"product": "Api", "usagePercent": 1.0},
            ],
            "billingPeriodStart": "2026-07-01T00:00:00+00:00",
            "billingPeriodEnd": "2026-08-01T00:00:00+00:00",
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


def test_resolve_grok_sidecar_auth_file_prefers_explicit_non_default_path(tmp_path, monkeypatch) -> None:
    explicit_auth = tmp_path / "explicit-auth.json"
    native_auth = tmp_path / "native-auth.json"
    monkeypatch.delenv("AAWM_GROK_OIDC_AUTH_FILE", raising=False)
    monkeypatch.setenv("LITELLM_XAI_GROK_AUTH_FILE", str(native_auth))

    resolved_path, source = loop._resolve_grok_sidecar_auth_file(str(explicit_auth))

    assert resolved_path == str(explicit_auth)
    assert source == "explicit"


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


@pytest.mark.parametrize(
    ("env_name", "configured_name", "expected_name"),
    (
        (
            "LITELLM_XAI_OAUTH_GROK_AUTH_FILE",
            "legacy-oauth.json",
            "legacy-oauth.json",
        ),
        ("GROK_AUTH_FILE", "grok-auth.json", "grok-auth.json"),
    ),
)
def test_resolve_grok_sidecar_auth_file_supports_each_native_source(
    tmp_path,
    monkeypatch,
    env_name,
    configured_name,
    expected_name,
) -> None:
    for configured_env_name in (
        "AAWM_GROK_OIDC_AUTH_FILE",
        "LITELLM_XAI_GROK_AUTH_FILE",
        "LITELLM_XAI_OAUTH_GROK_AUTH_FILE",
        "GROK_AUTH_FILE",
        "GROK_HOME",
    ):
        monkeypatch.delenv(configured_env_name, raising=False)
    monkeypatch.setenv(env_name, str(tmp_path / configured_name))

    resolved_path, source = loop._resolve_grok_sidecar_auth_file(loop.DEFAULT_GROK_OIDC_AUTH_FILE)

    assert resolved_path == str(tmp_path / expected_name)
    assert source == env_name
    assert str(tmp_path) not in source


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

    monkeypatch.delenv("GROK_HOME")
    monkeypatch.setenv("HOME", str(tmp_path))

    resolved_path, source = loop._resolve_grok_sidecar_auth_file(loop.DEFAULT_GROK_OIDC_AUTH_FILE)

    assert resolved_path == str(tmp_path / ".grok" / "auth.json")
    assert source == "default"


def test_grok_runtime_and_sidecar_resolve_same_conflicting_override(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("AAWM_GROK_OIDC_AUTH_FILE", "~/aawm/auth.json")
    monkeypatch.setenv(
        "LITELLM_XAI_GROK_AUTH_FILE",
        "~/litellm-native/auth.json",
    )
    monkeypatch.setenv(
        "LITELLM_XAI_OAUTH_GROK_AUTH_FILE",
        "~/legacy-oauth/auth.json",
    )
    monkeypatch.setenv("GROK_AUTH_FILE", "~/grok/auth.json")
    monkeypatch.setenv("GROK_HOME", "~/grok-home")

    runtime_path = oauth.default_grok_xai_oauth_auth_path()
    sidecar_path, sidecar_source = loop._resolve_grok_sidecar_auth_file("~/explicit/auth.json")

    expected_path = tmp_path / "aawm" / "auth.json"
    assert runtime_path == expected_path
    assert sidecar_path == str(expected_path)
    assert sidecar_source == "AAWM_GROK_OIDC_AUTH_FILE"
    assert str(expected_path) not in sidecar_source


def test_resolve_grok_billing_client_version_prefers_grok_client_version(monkeypatch) -> None:
    _clear_grok_billing_version_env(monkeypatch)
    monkeypatch.setenv("GROK_CLIENT_VERSION", "0.2.70")
    config = _grok_billing_poll_config(
        grok_billing_client_version=None,
        grok_billing_client_version_source=None,
    )

    resolution = loop._resolve_grok_billing_client_version(config)

    assert resolution.version == "0.2.70"
    assert resolution.source == "GROK_CLIENT_VERSION"


def test_loop_config_does_not_freeze_grok_client_version_env(monkeypatch) -> None:
    _clear_grok_billing_version_env(monkeypatch)
    monkeypatch.setenv("GROK_CLIENT_VERSION", "0.2.70")

    config = loop.parse_config([])
    monkeypatch.setenv("GROK_CLIENT_VERSION", "0.2.71")
    headers = loop._build_grok_billing_request_headers(
        config,
        access_token="access-token-secret",
    )

    assert config.grok_billing_client_version is None
    assert config.grok_billing_client_version_source is None
    assert headers["x-grok-client-version"] == "0.2.71"
    assert headers.version_resolution.source == "GROK_CLIENT_VERSION"


def test_grok_billing_client_version_env_precedence_is_per_request(
    monkeypatch,
) -> None:
    _clear_grok_billing_version_env(monkeypatch)
    config = _grok_billing_poll_config(
        grok_billing_client_version=None,
        grok_billing_client_version_source=None,
    )
    monkeypatch.setenv("AAWM_GROK_BILLING_CLIENT_VERSION", "1.2.3")
    monkeypatch.setenv("LITELLM_XAI_GROK_CLIENT_VERSION", "2.3.4")
    monkeypatch.setenv("GROK_CLIENT_VERSION", "3.4.5")

    first = loop._build_grok_billing_request_headers(
        config,
        access_token="access-token-secret",
    )
    monkeypatch.delenv("AAWM_GROK_BILLING_CLIENT_VERSION")
    second = loop._build_grok_billing_request_headers(
        config,
        access_token="access-token-secret",
    )
    monkeypatch.delenv("LITELLM_XAI_GROK_CLIENT_VERSION")
    third = loop._build_grok_billing_request_headers(
        config,
        access_token="access-token-secret",
    )

    assert first["x-grok-client-version"] == "1.2.3"
    assert (
        first.version_resolution.source
        == "AAWM_GROK_BILLING_CLIENT_VERSION"
    )
    assert second["x-grok-client-version"] == "2.3.4"
    assert (
        second.version_resolution.source
        == "LITELLM_XAI_GROK_CLIENT_VERSION"
    )
    assert third["x-grok-client-version"] == "3.4.5"
    assert third.version_resolution.source == "GROK_CLIENT_VERSION"


@pytest.mark.parametrize(
    ("env_name", "value"),
    [
        ("AAWM_GROK_BILLING_CLIENT_VERSION", ""),
        ("LITELLM_XAI_GROK_CLIENT_VERSION", ""),
        ("GROK_CLIENT_VERSION", ""),
        ("AAWM_GROK_BILLING_CLIENT_VERSION", "v1.2.3"),
        ("LITELLM_XAI_GROK_CLIENT_VERSION", "1.2.beta"),
        ("GROK_CLIENT_VERSION", "1"),
    ],
)
def test_grok_billing_client_version_present_empty_or_invalid_fails_closed(
    monkeypatch,
    env_name,
    value,
) -> None:
    _clear_grok_billing_version_env(monkeypatch)
    monkeypatch.setenv(env_name, value)
    if env_name != "GROK_CLIENT_VERSION":
        monkeypatch.setenv("GROK_CLIENT_VERSION", "9.9.9")
    config = _grok_billing_poll_config(
        grok_billing_client_version=None,
        grok_billing_client_version_source=None,
    )

    with pytest.raises(loop.GrokBillingClientVersionError) as exc_info:
        loop._build_grok_billing_request_headers(
            config,
            access_token="access-token-secret",
        )

    assert exc_info.value.source_metadata == {
        "client_version_source": env_name
    }
    assert env_name in str(exc_info.value)
    if value:
        assert value not in str(exc_info.value)


def test_grok_billing_client_version_falls_back_to_valid_cache(
    tmp_path,
    monkeypatch,
) -> None:
    _clear_grok_billing_version_env(monkeypatch)
    cache_path = tmp_path / "native-client-version.json"
    _write_grok_billing_version_cache(cache_path, version="4.5.6")
    monkeypatch.setenv(
        "AAWM_GROK_CLIENT_VERSION_CACHE_PATH",
        str(cache_path),
    )
    config = _grok_billing_poll_config(
        grok_billing_client_version=None,
        grok_billing_client_version_source=None,
    )

    headers = loop._build_grok_billing_request_headers(
        config,
        access_token="access-token-secret",
    )

    assert headers["x-grok-client-version"] == "4.5.6"
    assert headers["user-agent"] == "grok/4.5.6"
    assert headers.version_resolution.sanitized_metadata() == {
        "client_version_source": "cache",
        "client_version_cache_source": "installed-grok-cli",
        "client_version_cache_path_class": "configured",
    }


@pytest.mark.parametrize(
    ("cache_state", "error_fragment"),
    [
        ("missing", "missing"),
        ("invalid", "valid JSON"),
        ("stale", "stale"),
        ("future", "future"),
    ],
)
def test_grok_billing_attempt_reports_sanitized_cache_failure(
    tmp_path,
    monkeypatch,
    cache_state,
    error_fragment,
) -> None:
    _clear_grok_billing_version_env(monkeypatch)
    cache_path = tmp_path / "secret-cache-location.json"
    now = datetime.now(timezone.utc)
    if cache_state == "invalid":
        cache_path.write_text("secret-cache-contents", encoding="utf-8")
    elif cache_state == "stale":
        _write_grok_billing_version_cache(
            cache_path,
            version="5.6.7",
            observed_at=now - timedelta(minutes=5),
        )
    elif cache_state == "future":
        _write_grok_billing_version_cache(
            cache_path,
            version="5.6.7",
            observed_at=now + timedelta(minutes=5),
        )
    monkeypatch.setenv(
        "AAWM_GROK_CLIENT_VERSION_CACHE_PATH",
        str(cache_path),
    )
    monkeypatch.setenv(
        "AAWM_GROK_CLIENT_VERSION_CACHE_MAX_AGE_SECONDS",
        "60",
    )
    monkeypatch.setattr(
        loop,
        "_load_grok_billing_auth_context",
        lambda _path: _grok_billing_auth_context(),
    )
    monkeypatch.setattr(
        loop.urllib_request,
        "urlopen",
        lambda *_args, **_kwargs: pytest.fail(
            "billing HTTP request must not run without a valid version"
        ),
    )
    config = _grok_billing_poll_config(
        apply=False,
        grok_billing_client_version=None,
        grok_billing_client_version_source=None,
    )

    event = loop.run_due_sidecar_tasks(
        config,
        loop.SidecarTaskState(),
        now_monotonic=100.0,
    )[0]
    encoded_event = json.dumps(event)

    assert event["error_class"] == "GrokBillingClientVersionError"
    assert error_fragment in event["error_message"]
    assert event["client_version_source"] == "cache"
    assert event["client_version_cache_path_class"] == "configured"
    assert str(cache_path) not in encoded_event
    assert "secret-cache-contents" not in encoded_event
    assert "access-token-secret" not in encoded_event


def test_grok_billing_client_version_observes_atomic_inode_replacement(
    tmp_path,
    monkeypatch,
) -> None:
    _clear_grok_billing_version_env(monkeypatch)
    cache_path = tmp_path / "native-client-version.json"
    replacement_path = tmp_path / "replacement.json"
    _write_grok_billing_version_cache(cache_path, version="6.7.8")
    monkeypatch.setenv(
        "AAWM_GROK_CLIENT_VERSION_CACHE_PATH",
        str(cache_path),
    )
    config = _grok_billing_poll_config(
        grok_billing_client_version=None,
        grok_billing_client_version_source=None,
    )

    first = loop._build_grok_billing_request_headers(
        config,
        access_token="access-token-secret",
    )
    _write_grok_billing_version_cache(replacement_path, version="7.8.9")
    os.replace(replacement_path, cache_path)
    second = loop._build_grok_billing_request_headers(
        config,
        access_token="access-token-secret",
    )

    assert first["x-grok-client-version"] == "6.7.8"
    assert second["x-grok-client-version"] == "7.8.9"


def test_grok_billing_cli_version_is_explicit_config_only(monkeypatch) -> None:
    _clear_grok_billing_version_env(monkeypatch)
    config = loop.parse_config(
        ["--grok-billing-client-version", "8.9.10"]
    )
    empty_config = loop.parse_config(
        ["--grok-billing-client-version", ""]
    )

    assert config.grok_billing_client_version == "8.9.10"
    assert config.grok_billing_client_version_source == "cli"
    with pytest.raises(loop.GrokBillingClientVersionError) as exc_info:
        loop._build_grok_billing_request_headers(
            empty_config,
            access_token="access-token-secret",
        )
    assert exc_info.value.source_metadata == {
        "client_version_source": "cli"
    }


def test_grok_billing_has_no_hardcoded_client_version_fallback() -> None:
    script_text = (
        Path(__file__).resolve().parents[2]
        / "scripts/run_provider_status_observations_loop.py"
    ).read_text(encoding="utf-8")
    help_text = loop._build_parser().format_help()

    assert "DEFAULT_GROK_BILLING_CLIENT_VERSION" not in script_text
    assert "0.2.55" not in script_text
    assert "0.2.55" not in help_text


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
        "AAWM_CODEX_QUOTA_DSN",
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
        "AAWM_OBSERVABILITY_ANOMALY_SCAN_STATEMENT_TIMEOUT_MS",
        "AAWM_OBSERVABILITY_ANOMALY_SCAN_ERROR_LOG_DIR",
        "AAWM_PROVIDER_AUTH_HEALTH_POLL_ENABLED",
        "AAWM_PROVIDER_AUTH_HEALTH_POLL_INTERVAL_SECONDS",
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
    assert config.codex_quota_dsn is None
    assert config.require_pgbouncer is False
    assert config.grok_billing_poll_enabled is False
    assert config.grok_billing_poll_interval_seconds == 3600.0
    assert config.grok_billing_poll_http_timeout_seconds == 30.0
    assert (
        config.grok_billing_url
        == "https://cli-chat-proxy.grok.com/v1/billing?format=credits"
    )
    assert config.grok_billing_client_version is None
    assert config.grok_billing_client_version_source is None
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
    assert config.observability_anomaly_scan_statement_timeout_ms == 15000
    assert config.observability_anomaly_scan_error_log_dir == "/app/.analysis"
    assert config.provider_auth_health_poll_enabled is False
    assert config.provider_auth_health_poll_interval_seconds == 3600.0


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


def test_loop_config_uses_dedicated_codex_quota_dsn(monkeypatch) -> None:
    monkeypatch.setenv(
        "AAWM_CODEX_QUOTA_DSN",
        "postgresql://litellm_dev:secret@pgbouncer-litellm-dev:6432/litellm_dev",
    )

    config = loop.parse_config([])

    assert config.codex_quota_dsn == (
        "postgresql://litellm_dev:secret@pgbouncer-litellm-dev:6432/litellm_dev"
    )


def test_provider_status_compose_hardens_sidecar_db_path() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    compose_text = (repo_root / "docker-compose.dev.yml").read_text()
    dockerfile_text = (
        repo_root / "docker/Dockerfile.provider_status_observations"
    ).read_text()

    assert "container_name: aawm-provider-status-observations" in compose_text
    assert "AAWM_DB_HOST=${LITELLM_AAWM_DB_HOST:-pgbouncer-aawm-dev}" in compose_text
    assert "AAWM_DB_PORT=${LITELLM_AAWM_DB_PORT:-6432}" in compose_text
    assert (
        "AAWM_DATABASE_URL=${LITELLM_AAWM_DATABASE_URL:-postgresql://aawm:aawm_dev@pgbouncer-aawm-dev:6432/aawm_tristore?application_name=aawm-provider-status-observations}"
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
    assert "- /home/zepfu/.codex:/home/zepfu/.codex:ro" in compose_text
    assert "- /home/zepfu/.codex:/home/zepfu/.codex" in compose_text
    assert compose_text.count("- *codex-oauth-inventory") == 2
    assert "LITELLM_CODEX_AUTH_FILE=" not in compose_text
    assert "AAWM_CODEX_AUTH_FILE=" not in compose_text
    assert "AAWM_CODEX_LOCK_FILE=" not in compose_text
    for expected_codex_setting in (
        "AAWM_CODEX_OAUTH_REFRESH_ENABLED=${AAWM_CODEX_OAUTH_REFRESH_ENABLED:-1}",
        "AAWM_CODEX_AUTH_FILE_UID=${AAWM_CODEX_AUTH_FILE_UID:-1000}",
        "AAWM_CODEX_AUTH_FILE_GID=${AAWM_CODEX_AUTH_FILE_GID:-1000}",
        "AAWM_CODEX_AUTH_FILE_MODE=${AAWM_CODEX_AUTH_FILE_MODE:-0o600}",
        "AAWM_CODEX_OAUTH_FORCE_REFRESH=${AAWM_CODEX_OAUTH_FORCE_REFRESH:-1}",
    ):
        assert expected_codex_setting in compose_text
    for expected_kimi_usage_setting in (
        "AAWM_KIMI_USAGE_POLL_ENABLED=${AAWM_KIMI_USAGE_POLL_ENABLED:-1}",
        "AAWM_KIMI_USAGE_POLL_INTERVAL_SECONDS=${AAWM_KIMI_USAGE_POLL_INTERVAL_SECONDS:-3600}",
        "AAWM_KIMI_USAGE_POLL_HTTP_TIMEOUT_SECONDS=${AAWM_KIMI_USAGE_POLL_HTTP_TIMEOUT_SECONDS:-30}",
    ):
        assert expected_kimi_usage_setting in compose_text
    assert (
        "AAWM_GROK_BILLING_POLL_ENABLED=${AAWM_GROK_BILLING_POLL_ENABLED:-1}"
        in compose_text
    )
    for expected_cursor_usage_setting in (
        "AAWM_CURSOR_AGENT_USAGE_POLL_ENABLED=${AAWM_CURSOR_AGENT_USAGE_POLL_ENABLED:-0}",
        "AAWM_CURSOR_AGENT_USAGE_POLL_INTERVAL_SECONDS=${AAWM_CURSOR_AGENT_USAGE_POLL_INTERVAL_SECONDS:-3600}",
        "AAWM_CURSOR_AGENT_USAGE_POLL_HTTP_TIMEOUT_SECONDS=${AAWM_CURSOR_AGENT_USAGE_POLL_HTTP_TIMEOUT_SECONDS:-30}",
        "AAWM_CURSOR_AGENT_USAGE_DASHBOARD_URL=${AAWM_CURSOR_AGENT_USAGE_DASHBOARD_URL:-https://api2.cursor.sh}",
    ):
        assert expected_cursor_usage_setting in compose_text
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
        "AAWM_OBSERVABILITY_ANOMALY_SCAN_STATEMENT_TIMEOUT_MS=${AAWM_OBSERVABILITY_ANOMALY_SCAN_STATEMENT_TIMEOUT_MS:-15000}"
        in compose_text
    )
    assert (
        "AAWM_OBSERVABILITY_ANOMALY_SCAN_ERROR_LOG_DIR=${AAWM_OBSERVABILITY_ANOMALY_SCAN_ERROR_LOG_DIR:-/app/.analysis}"
        in compose_text
    )


def test_provider_status_compose_wires_passive_auth_health_disabled_by_default() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    compose_text = (repo_root / "docker-compose.dev.yml").read_text()

    assert (
        "AAWM_PROVIDER_AUTH_HEALTH_POLL_ENABLED=${AAWM_PROVIDER_AUTH_HEALTH_POLL_ENABLED:-0}"
        in compose_text
    )
    assert (
        "AAWM_PROVIDER_AUTH_HEALTH_POLL_INTERVAL_SECONDS=${AAWM_PROVIDER_AUTH_HEALTH_POLL_INTERVAL_SECONDS:-3600}"
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
        "AAWM_XAI_OAUTH_REFRESH_INTERVAL_SECONDS=${AAWM_XAI_OAUTH_REFRESH_INTERVAL_SECONDS:-300}"
        in compose_text
    )
    assert (
        "AAWM_XAI_OAUTH_REFRESH_BUFFER_SECONDS=${AAWM_XAI_OAUTH_REFRESH_BUFFER_SECONDS:-900}"
        in compose_text
    )
    assert (
        "AAWM_XAI_OAUTH_FORCE_REFRESH=${AAWM_XAI_OAUTH_FORCE_REFRESH:-0}"
        in compose_text
    )
    assert (
        "COPY scripts/xai_oauth_refresh.py "
        "/app/scripts/xai_oauth_refresh.py"
    ) in dockerfile_text


def test_validate_xai_oauth_guardrail_rejects_outer_cadence_above_buffer() -> None:
    config = loop.ProviderStatusLoopConfig(
        apply=False,
        dsn=None,
        environment="dev",
        interval_seconds=901.0,
        timeout=2.0,
        ping_count=1,
        ping_timeout=2,
        skip_icmp=True,
        once=True,
        setup_schema=False,
        db_lock_timeout_ms=1000,
        db_statement_timeout_ms=5000,
        xai_oauth_refresh_enabled=True,
        xai_oauth_refresh_interval_seconds=60.0,
        xai_oauth_refresh_buffer_seconds=900,
        xai_oauth_force_refresh=False,
    )

    with pytest.raises(
        SystemExit,
        match=(
            r"--interval-seconds=901 exceeds "
            r"--xai-oauth-refresh-buffer-seconds=900; "
            r"outer eligibility cadence must not exceed the refresh buffer"
        ),
    ):
        loop._validate_xai_oauth_config_args(
            Namespace(
                xai_oauth_scope=config.xai_oauth_scope,
                xai_oauth_refresh_interval_seconds=60.0,
                xai_oauth_refresh_buffer_seconds=900,
                xai_oauth_http_timeout_seconds=30.0,
                xai_oauth_refresh_enabled=True,
                xai_oauth_force_refresh=False,
                interval_seconds=901.0,
            )
        )


@pytest.mark.parametrize(
    ("summary", "invoke_callback", "expected_throttled"),
    [
        (
            {
                "attempted": False,
                "refreshed": False,
                "skipped": True,
                "error_class": None,
                "error_message": None,
            },
            False,
            False,
        ),
        (
            {
                "attempted": True,
                "refreshed": False,
                "skipped": False,
                "error_class": "TimeoutError",
                "error_message": "pre-network failure",
            },
            False,
            False,
        ),
        (
            {
                "attempted": True,
                "refreshed": False,
                "skipped": False,
                "error_class": "HTTPError",
                "error_message": "endpoint rejected",
            },
            True,
            True,
        ),
        (
            {
                "attempted": True,
                "refreshed": True,
                "skipped": False,
                "error_class": None,
                "error_message": None,
            },
            True,
            True,
        ),
    ],
)
def test_oauth_schedule_throttles_only_actual_token_endpoint_attempts(
    summary,
    invoke_callback,
    expected_throttled,
) -> None:
    schedule = loop.OAuthRefreshScheduleState()
    state = {"last_attempt": None}

    def inspect(*, now):
        return {
            "eligibility_checked_at": now().isoformat().replace("+00:00", "Z"),
            "expires_at": "2026-08-13T23:00:00Z",
            "refresh_due_at": "2026-08-13T22:00:00Z",
            "next_refresh_check_at": "2026-08-13T23:05:00Z",
            "eligible": True,
            "credential_health": "fresh",
            "usable": True,
            "error_class": None,
            "error_message": None,
        }

    calls = []

    def refresh(callback):
        calls.append(callback)
        if invoke_callback:
            callback()
        return dict(summary)

    final, _operation, _post, evidence, helper_called = (
        loop._run_oauth_refresh_schedule(
            schedule=schedule,
            last_attempt_monotonic=state["last_attempt"],
            set_last_attempt_monotonic=lambda value: state.__setitem__(
                "last_attempt", value
            ),
            now_monotonic=100.0,
            wall_now=datetime(2026, 8, 13, 22, 30, tzinfo=timezone.utc),
            eligibility_inspector=inspect,
            refresh_call=refresh,
            force=False,
            attempt_interval_seconds=300.0,
            eligibility_cadence_seconds=300.0,
            buffer_seconds=3600.0,
        )
    )

    assert final["usable"] is True
    assert helper_called is True
    assert evidence["actual_attempted"] is invoke_callback
    assert evidence["actual_attempt_count"] == (1 if invoke_callback else 0)
    assert state["last_attempt"] == (100.0 if invoke_callback else None)

    if expected_throttled:
        second = loop._run_oauth_refresh_schedule(
            schedule=schedule,
            last_attempt_monotonic=state["last_attempt"],
            set_last_attempt_monotonic=lambda value: state.__setitem__(
                "last_attempt", value
            ),
            now_monotonic=200.0,
            wall_now=datetime(2026, 8, 13, 22, 31, tzinfo=timezone.utc),
            eligibility_inspector=inspect,
            refresh_call=lambda _callback: pytest.fail(
                "actual endpoint attempt throttle should skip helper call"
            ),
            force=False,
            attempt_interval_seconds=300.0,
            eligibility_cadence_seconds=300.0,
            buffer_seconds=3600.0,
        )
        assert second[4] is False
        assert second[3]["actual_attempted"] is False


def test_oauth_schedule_retries_pre_network_failure_next_outer_cycle() -> None:
    schedule = loop.OAuthRefreshScheduleState()
    actual_attempt = {"value": None}
    helper_calls = []
    eligibility_calls = []

    def inspect(*, now):
        eligibility_calls.append(now())
        return {
            "eligibility_checked_at": now().isoformat().replace("+00:00", "Z"),
            "expires_at": "2026-08-13T23:00:00Z",
            "refresh_due_at": "2026-08-13T22:00:00Z",
            "next_refresh_check_at": "2026-08-13T22:35:00Z",
            "eligible": True,
            "credential_health": "fresh",
            "usable": True,
            "error_class": None,
            "error_message": None,
        }

    def refresh(_callback):
        helper_calls.append(True)
        return {
            "attempted": True,
            "refreshed": False,
            "skipped": False,
            "error_class": "TimeoutError",
            "error_message": "pre-network failure",
        }

    def run_schedule(now_monotonic, wall_now):
        return loop._run_oauth_refresh_schedule(
            schedule=schedule,
            last_attempt_monotonic=actual_attempt["value"],
            set_last_attempt_monotonic=lambda value: actual_attempt.__setitem__(
                "value", value
            ),
            now_monotonic=now_monotonic,
            wall_now=wall_now,
            eligibility_inspector=inspect,
            refresh_call=refresh,
            force=False,
            attempt_interval_seconds=300.0,
            eligibility_cadence_seconds=300.0,
            buffer_seconds=3600.0,
        )

    first = run_schedule(
        100.0,
        datetime(2026, 8, 13, 22, 30, tzinfo=timezone.utc),
    )
    next_outer_cycle = run_schedule(
        101.0,
        datetime(2026, 8, 13, 22, 30, 1, tzinfo=timezone.utc),
    )

    assert first[4] is True
    assert next_outer_cycle[4] is True
    assert first[3]["actual_attempted"] is False
    assert next_outer_cycle[3]["actual_attempted"] is False
    assert first[3]["refresh_result_class"] == "refresh_failed"
    assert next_outer_cycle[3]["refresh_result_class"] == "refresh_failed"
    assert helper_calls == [True, True]
    assert len(eligibility_calls) == 4
    assert actual_attempt["value"] is None
    assert schedule.last_actual_attempt_at is None
    assert schedule.actual_attempt_count == 0


def test_oauth_schedule_reinspects_external_replacement_after_local_failure(
    tmp_path,
) -> None:
    credential_path = tmp_path / "oauth.json"
    due_expiry = datetime(2026, 8, 13, 23, 0, tzinfo=timezone.utc)
    fresh_expiry = datetime(2026, 8, 14, 1, 0, tzinfo=timezone.utc)
    credential_path.write_text(
        json.dumps({"expires_at": due_expiry.isoformat()}),
        encoding="utf-8",
    )
    schedule = loop.OAuthRefreshScheduleState()
    actual_attempt = {"value": None}
    helper_calls = []

    def inspect(*, now):
        observed_at = now()
        expires_at = datetime.fromisoformat(
            json.loads(credential_path.read_text(encoding="utf-8"))["expires_at"]
        )
        refresh_due_at = expires_at - timedelta(hours=1)
        return {
            "eligibility_checked_at": observed_at.isoformat().replace(
                "+00:00", "Z"
            ),
            "expires_at": expires_at.isoformat().replace("+00:00", "Z"),
            "refresh_due_at": refresh_due_at.isoformat().replace("+00:00", "Z"),
            "next_refresh_check_at": refresh_due_at.isoformat().replace(
                "+00:00", "Z"
            ),
            "eligible": observed_at >= refresh_due_at,
            "credential_health": "fresh",
            "usable": observed_at < expires_at,
            "error_class": None,
            "error_message": None,
        }

    def refresh(_callback):
        helper_calls.append(True)
        return {
            "attempted": True,
            "refreshed": False,
            "skipped": False,
            "error_class": "ValueError",
            "error_message": "local request construction failed",
        }

    first = loop._run_oauth_refresh_schedule(
        schedule=schedule,
        last_attempt_monotonic=actual_attempt["value"],
        set_last_attempt_monotonic=lambda value: actual_attempt.__setitem__(
            "value", value
        ),
        now_monotonic=100.0,
        wall_now=datetime(2026, 8, 13, 22, 30, tzinfo=timezone.utc),
        eligibility_inspector=inspect,
        refresh_call=refresh,
        force=False,
        attempt_interval_seconds=300.0,
        eligibility_cadence_seconds=300.0,
        buffer_seconds=3600.0,
    )
    assert first[4] is True
    assert first[3]["actual_attempted"] is False
    assert actual_attempt["value"] is None

    replacement_path = tmp_path / "oauth.replacement.json"
    replacement_path.write_text(
        json.dumps({"expires_at": fresh_expiry.isoformat()}),
        encoding="utf-8",
    )
    os.replace(replacement_path, credential_path)

    second = loop._run_oauth_refresh_schedule(
        schedule=schedule,
        last_attempt_monotonic=actual_attempt["value"],
        set_last_attempt_monotonic=lambda value: actual_attempt.__setitem__(
            "value", value
        ),
        now_monotonic=101.0,
        wall_now=datetime(2026, 8, 13, 22, 30, 1, tzinfo=timezone.utc),
        eligibility_inspector=inspect,
        refresh_call=lambda _callback: pytest.fail(
            "fresh external replacement must be observed before helper invocation"
        ),
        force=False,
        attempt_interval_seconds=300.0,
        eligibility_cadence_seconds=300.0,
        buffer_seconds=3600.0,
    )
    assert second[4] is False
    assert second[3]["refresh_result_class"] == "refresh_not_due"
    assert second[3]["expires_at"] == "2026-08-14T01:00:00Z"
    assert second[3]["actual_attempted"] is False
    assert second[3]["credential_health"] == "fresh"
    assert helper_calls == [True]
    assert actual_attempt["value"] is None


def test_xai_incident_deadline_boundary_does_not_consume_attempt_throttle() -> None:
    issuance = datetime(2026, 8, 12, 17, 18, 43, 881054, tzinfo=timezone.utc)
    expiry = datetime(2026, 8, 12, 23, 18, 43, 881054, tzinfo=timezone.utc)
    assert expiry - issuance == timedelta(hours=6)
    due_boundary = expiry - timedelta(minutes=15)
    assert due_boundary == datetime(
        2026, 8, 12, 23, 3, 43, 881054, tzinfo=timezone.utc
    )

    schedule = loop.OAuthRefreshScheduleState()
    last_attempt = {"value": None}
    calls = []
    eligibility_times = iter(
        (
            datetime(2026, 8, 12, 22, 38, 44, tzinfo=timezone.utc),
            datetime(2026, 8, 12, 23, 3, 44, tzinfo=timezone.utc),
        )
    )

    def inspect(*, now):
        observed = next(eligibility_times)
        return {
            "eligibility_checked_at": observed.isoformat().replace("+00:00", "Z"),
            "expires_at": expiry.isoformat().replace("+00:00", "Z"),
            "refresh_due_at": due_boundary.isoformat().replace("+00:00", "Z"),
            "next_refresh_check_at": due_boundary.isoformat().replace("+00:00", "Z"),
            "eligible": observed >= due_boundary,
            "credential_health": "fresh",
            "usable": True,
            "error_class": None,
            "error_message": None,
        }

    def refresh(callback):
        calls.append(True)
        callback()
        return {
            "attempted": True,
            "refreshed": True,
            "skipped": False,
            "error_class": None,
            "error_message": None,
        }

    first = loop._run_oauth_refresh_schedule(
        schedule=schedule,
        last_attempt_monotonic=None,
        set_last_attempt_monotonic=lambda value: last_attempt.__setitem__(
            "value", value
        ),
        now_monotonic=100.0,
        wall_now=datetime(2026, 8, 12, 22, 38, 44, tzinfo=timezone.utc),
        eligibility_inspector=inspect,
        refresh_call=refresh,
        force=False,
        attempt_interval_seconds=300.0,
        eligibility_cadence_seconds=300.0,
        buffer_seconds=900.0,
    )
    assert first[3]["refresh_result_class"] == "refresh_not_due"
    assert first[3]["actual_attempted"] is False
    assert calls == []

    second = loop._run_oauth_refresh_schedule(
        schedule=schedule,
        last_attempt_monotonic=last_attempt["value"],
        set_last_attempt_monotonic=lambda value: last_attempt.__setitem__(
            "value", value
        ),
        now_monotonic=101.0,
        wall_now=datetime(2026, 8, 12, 23, 3, 44, tzinfo=timezone.utc),
        eligibility_inspector=inspect,
        refresh_call=refresh,
        force=False,
        attempt_interval_seconds=300.0,
        eligibility_cadence_seconds=300.0,
        buffer_seconds=900.0,
    )
    assert second[3]["refresh_result_class"] == "refresh_due"
    assert second[3]["actual_attempted"] is True
    assert second[3]["last_actual_attempt_at"] == "2026-08-12T23:03:44Z"
    assert calls == [True]


def test_xai_restart_state_discards_pre_restart_attempt_throttle(
    monkeypatch,
) -> None:
    config = _xai_oauth_auth_persist_config(
        apply=False,
        interval_seconds=300.0,
        xai_oauth_refresh_interval_seconds=3600.0,
        xai_oauth_refresh_buffer_seconds=900,
    )
    wall_now = datetime(2026, 8, 13, 22, 30, tzinfo=timezone.utc)
    inspections = []
    refresh_calls = []

    def inspect(*_args, now, **_kwargs):
        observed = now()
        inspections.append(observed)
        return {
            "eligibility_checked_at": observed.isoformat().replace("+00:00", "Z"),
            "expires_at": "2026-08-13T22:40:00Z",
            "refresh_due_at": "2026-08-13T22:25:00Z",
            "next_refresh_check_at": "2026-08-13T22:35:00Z",
            "eligible": True,
            "credential_health": "fresh",
            "usable": True,
            "scope": config.xai_oauth_scope,
            "error_class": None,
            "error_message": None,
        }

    def refresh(*_args, on_token_endpoint_attempt, **_kwargs):
        refresh_calls.append(True)
        on_token_endpoint_attempt()
        return {
            "attempted": True,
            "refreshed": True,
            "skipped": False,
            "scope": config.xai_oauth_scope,
            "expires_at": "2026-08-13T23:30:00Z",
            "error_class": None,
            "error_message": None,
        }

    monkeypatch.setattr(
        loop.xai_oauth_refresh,
        "inspect_xai_oauth_refresh_eligibility",
        inspect,
    )
    monkeypatch.setattr(
        loop.xai_oauth_refresh,
        "refresh_xai_oauth_auth_file",
        refresh,
    )

    pre_restart_state = loop.SidecarTaskState(
        xai_oauth_last_attempt_monotonic=100.0
    )
    throttled = loop._run_xai_oauth_refresh_task(
        config,
        pre_restart_state,
        now_monotonic=200.0,
        now_wall=wall_now,
    )
    restarted_state = loop.SidecarTaskState()
    after_restart = loop._run_xai_oauth_refresh_task(
        config,
        restarted_state,
        now_monotonic=200.0,
        now_wall=wall_now,
    )

    assert throttled is not None
    assert throttled["actual_attempted"] is False
    assert after_restart is not None
    assert after_restart["actual_attempted"] is True
    assert after_restart["last_actual_attempt_at"] == "2026-08-13T22:30:00Z"
    assert restarted_state.xai_oauth_last_attempt_monotonic == 200.0
    assert len(inspections) == 3
    assert refresh_calls == [True]


def test_kimi_event_and_observation_use_dynamic_refresh_threshold(
    tmp_path,
) -> None:
    wall_now = datetime(2026, 8, 13, 22, 30, tzinfo=timezone.utc)
    auth_file = tmp_path / "kimi.json"
    auth_file.write_text(
        json.dumps(
            {
                "access_token": "access-token-secret",
                "refresh_token": "refresh-token-secret",
                "expires_at": (wall_now + timedelta(minutes=10)).timestamp(),
                "expires_in": 900,
                "scope": "kimi-code",
            }
        ),
        encoding="utf-8",
    )
    config = _grok_oidc_auth_persist_config(
        apply=False,
        grok_oidc_refresh_enabled=False,
        kimi_oauth_refresh_enabled=True,
        kimi_oauth_auth_file=str(auth_file),
        kimi_oauth_lock_file=str(tmp_path / "kimi.lock"),
        kimi_oauth_refresh_interval_seconds=3600.0,
        kimi_oauth_force_refresh=False,
    )

    event = loop._run_kimi_oauth_refresh_task(
        config,
        loop.SidecarTaskState(),
        now_monotonic=100.0,
        now_wall=wall_now,
    )

    assert event is not None
    assert event["refresh_result_class"] == "refresh_not_due"
    assert event["refresh_threshold_seconds"] == 450.0
    assert event["actual_attempted"] is False
    observation = loop._build_kimi_oauth_auth_observation(config, event)
    assert observation["metadata"]["refresh_threshold_seconds"] == 450.0
    rendered = json.dumps(observation, default=str)
    assert "access-token-secret" not in rendered
    assert "refresh-token-secret" not in rendered


@pytest.mark.parametrize(
    (
        "operation_error_class",
        "expires_at",
        "refresh_due_at",
        "eligible",
        "expected_result",
        "expected_health",
        "usable",
    ),
    [
        (
            None,
            "2026-08-13T23:00:00Z",
            "2026-08-13T23:30:00Z",
            False,
            "refresh_not_due",
            "fresh",
            True,
        ),
        (
            None,
            "2026-08-13T23:00:00Z",
            "2026-08-13T22:00:00Z",
            True,
            "refresh_due",
            "degraded",
            True,
        ),
        (
            "HTTPError",
            "2026-08-13T23:00:00Z",
            "2026-08-13T22:00:00Z",
            True,
            "refresh_failed",
            "degraded",
            True,
        ),
        (
            "HTTPError",
            "2026-08-13T21:00:00Z",
            "2026-08-13T20:00:00Z",
            True,
            "expired",
            "expired",
            False,
        ),
    ],
)
def test_oauth_schedule_result_and_health_classes(
    operation_error_class,
    expires_at,
    refresh_due_at,
    eligible,
    expected_result,
    expected_health,
    usable,
) -> None:
    schedule = loop.OAuthRefreshScheduleState()

    def inspect(*, now):
        return {
            "eligibility_checked_at": now().isoformat().replace("+00:00", "Z"),
            "expires_at": expires_at,
            "refresh_due_at": refresh_due_at,
            "next_refresh_check_at": "2026-08-13T23:05:00Z",
            "eligible": eligible,
            "credential_health": "fresh",
            "usable": usable,
            "error_class": None,
            "error_message": None,
        }

    def refresh(callback):
        callback()
        return {
            "attempted": True,
            "refreshed": False,
            "skipped": False,
            "error_class": operation_error_class,
            "error_message": "refresh_token=secret-value",
        }

    _final, _summary, _post, evidence, _called = loop._run_oauth_refresh_schedule(
        schedule=schedule,
        last_attempt_monotonic=None,
        set_last_attempt_monotonic=lambda _value: None,
        now_monotonic=100.0,
        wall_now=datetime(2026, 8, 13, 22, 30, tzinfo=timezone.utc),
        eligibility_inspector=inspect,
        refresh_call=refresh,
        force=False,
        attempt_interval_seconds=300.0,
        eligibility_cadence_seconds=300.0,
        buffer_seconds=3600.0,
    )
    assert evidence["refresh_result_class"] == expected_result
    assert evidence["credential_health"] == expected_health
    assert evidence["usable"] is usable
    if operation_error_class:
        assert evidence["scheduler_error_message"] is not None
        assert "secret-value" not in evidence["scheduler_error_message"]
        assert "REDACTED" in evidence["scheduler_error_message"]
    else:
        assert evidence["scheduler_error_message"] is None


def test_oauth_schedule_preserves_not_due_credential_after_forced_failure() -> None:
    schedule = loop.OAuthRefreshScheduleState()

    def inspect(*, now):
        return {
            "eligibility_checked_at": now().isoformat().replace("+00:00", "Z"),
            "expires_at": "2026-08-13T23:00:00Z",
            "refresh_due_at": "2026-08-13T22:45:00Z",
            "next_refresh_check_at": "2026-08-13T22:45:00Z",
            "eligible": False,
            "credential_health": "fresh",
            "usable": True,
            "error_class": None,
            "error_message": None,
        }

    def refresh(callback):
        callback()
        return {
            "attempted": True,
            "refreshed": False,
            "skipped": False,
            "error_class": "HTTPError",
            "error_message": "Authorization=Bearer secret-value-123456",
        }

    _final, _summary, _post, evidence, called = loop._run_oauth_refresh_schedule(
        schedule=schedule,
        last_attempt_monotonic=None,
        set_last_attempt_monotonic=lambda _value: None,
        now_monotonic=100.0,
        wall_now=datetime(2026, 8, 13, 22, 30, tzinfo=timezone.utc),
        eligibility_inspector=inspect,
        refresh_call=refresh,
        force=True,
        attempt_interval_seconds=300.0,
        eligibility_cadence_seconds=300.0,
        buffer_seconds=900.0,
    )
    assert called is True
    assert evidence["refresh_result_class"] == "refresh_not_due"
    assert evidence["credential_health"] == "fresh"
    assert evidence["usable"] is True
    assert evidence["scheduler_error_class"] == "HTTPError"
    assert evidence["scheduler_error_message"] is not None
    assert "secret-value-123456" not in evidence["scheduler_error_message"]
    assert "REDACTED" in evidence["scheduler_error_message"]


def test_provider_status_compose_aawm_network_external_name_contract() -> None:
    """Both services reference the logical ``aawm_default`` network, which is
    declared external with a configurable rendered name defaulting to the
    live ``aawm-infrastructure_default`` network."""
    import yaml

    repo_root = Path(__file__).resolve().parents[2]
    compose_text = (repo_root / "docker-compose.dev.yml").read_text()

    # Logical key and external declaration are present in raw text.
    assert "  aawm_default:" in compose_text
    assert "external: true" in compose_text
    assert (
        "name: ${AAWM_INFRASTRUCTURE_NETWORK_NAME:-aawm-infrastructure_default}"
        in compose_text
    )

    # Parse the YAML (substitution syntax is a plain scalar here) and assert
    # the global network contract plus both service references.
    doc = yaml.safe_load(compose_text)
    net = doc["networks"]["aawm_default"]
    assert net["external"] is True
    assert (
        net["name"]
        == "${AAWM_INFRASTRUCTURE_NETWORK_NAME:-aawm-infrastructure_default}"
    )

    services = doc["services"]
    referencing = [
        name
        for name, svc in services.items()
        if "aawm_default" in (svc.get("networks") or [])
    ]
    # Both the litellm-dev proxy and the provider-status sidecar must attach.
    assert "litellm-dev" in referencing
    assert "provider-status-observations" in referencing


def test_provider_status_compose_aawm_network_rendered_default_name() -> None:
    """`docker compose config` with an empty env renders the default external
    network name ``aawm-infrastructure_default`` and keeps it external.

    Skipped when the docker CLI is unavailable so the focused suite stays
    hermetic; the raw-text contract above still guards the declaration.
    """
    import shutil
    import subprocess

    if shutil.which("docker") is None:
        pytest.skip("docker CLI unavailable")

    repo_root = Path(__file__).resolve().parents[2]
    proc = subprocess.run(
        [
            "docker",
            "compose",
            "--env-file",
            "/dev/null",
            "-f",
            "docker-compose.dev.yml",
            "config",
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        timeout=120,
        env={
            **os.environ,
            "AAWM_CODEX_OAUTH_ACCOUNT1_EXPECTED_HASH": "111111111111",
            "AAWM_CODEX_OAUTH_ACCOUNT2_EXPECTED_HASH": "222222222222",
        },
    )
    if proc.returncode != 0:
        pytest.skip(f"docker compose config unavailable: {proc.stderr.strip()}")

    import yaml

    rendered = yaml.safe_load(proc.stdout)
    net = rendered["networks"]["aawm_default"]
    assert net["external"] is True
    assert net["name"] == "aawm-infrastructure_default"

    services = rendered["services"]
    proxy_inventory = services["litellm-dev"]["environment"][
        "LITELLM_CODEX_OAUTH_INVENTORY"
    ]
    sidecar_inventory = services["provider-status-observations"]["environment"][
        "LITELLM_CODEX_OAUTH_INVENTORY"
    ]
    assert proxy_inventory == sidecar_inventory
    sidecar_environment = services["provider-status-observations"]["environment"]
    assert sidecar_environment["AAWM_XAI_OAUTH_REFRESH_INTERVAL_SECONDS"] == "300"
    assert sidecar_environment["AAWM_XAI_OAUTH_REFRESH_BUFFER_SECONDS"] == "900"
    assert sidecar_environment["AAWM_XAI_OAUTH_FORCE_REFRESH"] == "0"
    inventory = json.loads(proxy_inventory)
    assert inventory["schema_version"] == 1
    assert [
        (
            account["label"],
            account["auth_path"],
            account["lock_path"],
            account["priority"],
            account["weight"],
            account["enabled"],
            account["models"],
            account["expected_account_hash"],
        )
        for account in inventory["accounts"]
    ] == [
        (
            "account1",
            "/home/zepfu/.codex/oauth.account1.json",
            "/home/zepfu/.codex/oauth.account1.json.lock",
            10,
            1.0,
            True,
            ["*"],
            "111111111111",
        ),
        (
            "account2",
            "/home/zepfu/.codex/oauth.account2.json",
            "/home/zepfu/.codex/oauth.account2.json.lock",
            20,
            1.0,
            True,
            ["*"],
            "222222222222",
        ),
    ]

    proxy_codex_mount = next(
        volume
        for volume in services["litellm-dev"]["volumes"]
        if volume.get("source") == "/home/zepfu/.codex"
    )
    sidecar_codex_mount = next(
        volume
        for volume in services["provider-status-observations"]["volumes"]
        if volume.get("source") == "/home/zepfu/.codex"
    )
    assert proxy_codex_mount["target"] == "/home/zepfu/.codex"
    assert proxy_codex_mount["read_only"] is True
    assert sidecar_codex_mount["target"] == "/home/zepfu/.codex"
    assert sidecar_codex_mount.get("read_only", False) is False


def test_env_example_documents_aawm_infrastructure_network_name() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    env_text = (repo_root / ".env.example").read_text()
    assert (
        "AAWM_INFRASTRUCTURE_NETWORK_NAME = aawm-infrastructure_default"
        in env_text
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
        kwargs["on_token_endpoint_attempt"]()
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
    assert calls[0][0] == ("/home/zepfu/.grok/auth.json",)
    assert calls[0][1]["buffer_seconds"] == 300
    assert calls[0][1]["force"] is True
    assert calls[0][1]["lock_file"] == "/home/zepfu/.grok/auth.json.lock"
    assert calls[0][1]["http_timeout_seconds"] == 30.0
    assert callable(calls[0][1]["on_token_endpoint_attempt"])
    assert events[0]["event"] == "grok_oidc_refresh"
    assert events[0]["environment"] == "dev"
    assert events[0]["refreshed"] is True
    assert "access-token" not in str(events)
    assert second_events[0]["event"] == "grok_oidc_refresh"
    assert second_events[0]["skipped"] is True
    assert second_events[0]["actual_attempted"] is False
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
    assert events[0] == {
        "event": "grok_oidc_metadata_repair",
        "observed_at": events[0]["observed_at"],
        "environment": "dev",
        "attempted": True,
        "repaired": True,
        "auth_file": "/home/zepfu/.grok/auth.json",
        "error_class": None,
        "error_message": None,
    }
    assert events[1]["event"] == "grok_oidc_refresh"
    assert events[1]["skipped"] is True
    assert events[1]["actual_attempted"] is False


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


def test_loop_config_reads_alibaba_quota_poll_env_defaults(monkeypatch) -> None:
    monkeypatch.setenv("AAWM_ALIBABA_QUOTA_POLL_ENABLED", "1")
    monkeypatch.setenv("AAWM_ALIBABA_QUOTA_POLL_INTERVAL_SECONDS", "600")
    monkeypatch.setenv("AAWM_ALIBABA_SUBSCRIPTION_POLL_INTERVAL_SECONDS", "43200")
    monkeypatch.setenv("AAWM_ALIBABA_QUOTA_POLL_HTTP_TIMEOUT_SECONDS", "45")
    monkeypatch.setenv(
        "AAWM_ALIBABA_QUOTA_GATEWAY_URL",
        "https://bailian-singapore-cs.alibabacloud.com/cli/api.json",
    )
    monkeypatch.setenv("AAWM_ALIBABA_QUOTA_POLL_MAX_ATTEMPTS", "4")
    monkeypatch.setenv("AAWM_ALIBABA_QUOTA_POLL_RETRY_BACKOFF_SECONDS", "1.25")

    config = loop.parse_config([])

    assert config.alibaba_quota_poll_enabled is True
    assert config.alibaba_quota_poll_interval_seconds == 600.0
    assert config.alibaba_subscription_poll_interval_seconds == 43200.0
    assert config.alibaba_quota_poll_http_timeout_seconds == 45.0
    assert config.alibaba_quota_gateway_url == loop.DEFAULT_ALIBABA_QUOTA_GATEWAY_URL
    assert not hasattr(config, "alibaba_web_auth_file")
    assert config.alibaba_quota_poll_max_attempts == 4
    assert config.alibaba_quota_poll_retry_backoff_seconds == 1.25
    assert loop.DEFAULT_ALIBABA_QUOTA_GATEWAY_URL.endswith("/cli/api.json")


def test_alibaba_ram_auth_signs_empty_body_and_builds_bearer_request(
    monkeypatch,
    capsys,
) -> None:
    config = _alibaba_quota_poll_config()
    auth = _alibaba_ram_auth(monkeypatch)
    headers = loop._alibaba_acs3_signed_headers(
        access_key_id=auth["access_key_id"],
        access_key_secret=auth["access_key_secret"],
        host=loop.ALIBABA_TOKEN_PLAN_MINT_HOST,
        pathname=loop.ALIBABA_TOKEN_PLAN_MINT_PATH,
        action=loop.ALIBABA_TOKEN_PLAN_MINT_ACTION,
        version=loop.ALIBABA_TOKEN_PLAN_MINT_VERSION,
        method="POST",
        body="",
        query_string="",
        acs_date=ALIBABA_TEST_ACS_DATE,
        signature_nonce=ALIBABA_TEST_SIGNATURE_NONCE,
    )
    request = loop._build_alibaba_quota_request(
        config,
        api_name=loop.ALIBABA_TOKEN_PLAN_SUBSCRIPTION_API,
        access_token="console-bearer-secret",
    )
    query = parse_qs(urlsplit(request.full_url).query)
    body = parse_qs((request.data or b"").decode("utf-8"))
    params = json.loads(body["params"][0])
    captured = capsys.readouterr()

    assert loop.ALIBABA_TOKEN_PLAN_MINT_HOST.endswith(".aliyuncs.com")
    assert not loop._alibaba_host_is_china(loop.ALIBABA_TOKEN_PLAN_MINT_HOST)
    assert not loop._alibaba_host_is_china(urlsplit(config.alibaba_quota_gateway_url).hostname)
    assert headers["x-acs-action"] == "GenerateCLIAccessToken"
    assert headers["x-acs-version"] == "2026-02-10"
    assert headers["x-acs-content-sha256"] == ALIBABA_TEST_EMPTY_BODY_SHA256
    assert headers["authorization"].startswith("ACS3-HMAC-SHA256 Credential=")
    assert f"Signature={ALIBABA_TEST_ACS3_SIGNATURE}" in headers["authorization"]
    assert ALIBABA_TEST_RAM_SECRET not in headers["authorization"]
    assert urlsplit(request.full_url).path == "/cli/api.json"
    assert query["api"] == [loop.ALIBABA_TOKEN_PLAN_SUBSCRIPTION_API]
    assert query["action"] == ["IntlBroadScopeAspnGateway"]
    assert body["region"] == ["ap-southeast-1"]
    assert set(body) == {"params", "region"}
    assert request.get_header("Authorization") == "Bearer console-bearer-secret"
    assert request.get_header("Cookie") is None
    assert "sec_token" not in body
    assert "_v" not in query
    assert auth["auth_source"] == "ALIBABA_RAM_KEY"
    assert auth["credential_reloaded"] is True
    assert params["Api"] == loop.ALIBABA_TOKEN_PLAN_SUBSCRIPTION_API
    assert params["Data"]["queryInstanceInfoRequest"] == {
        "commodityCode": loop.ALIBABA_TOKEN_PLAN_COMMODITY_CODE
    }
    _assert_no_alibaba_secrets(json.dumps(headers) + captured.out + captured.err)


@pytest.mark.parametrize(
    "env_updates",
    [
        {"ALIBABA_RAM_KEY": "", "ALIBABA_RAM_SECRET": ALIBABA_TEST_RAM_SECRET},
        {"ALIBABA_RAM_KEY": ALIBABA_TEST_RAM_KEY, "ALIBABA_RAM_SECRET": ""},
        {"ALIBABA_RAM_KEY": "bad\nkey", "ALIBABA_RAM_SECRET": ALIBABA_TEST_RAM_SECRET},
    ],
)
def test_alibaba_ram_auth_rejects_incomplete_or_invalid_credentials(
    monkeypatch,
    env_updates,
) -> None:
    _set_alibaba_ram_env(monkeypatch)
    for name, value in env_updates.items():
        if value:
            monkeypatch.setenv(name, value)
        else:
            monkeypatch.delenv(name, raising=False)

    with pytest.raises(loop.AlibabaAuthError):
        loop._load_alibaba_ram_auth(_alibaba_quota_poll_config())


def test_alibaba_china_hosts_are_never_selected() -> None:
    config = _alibaba_quota_poll_config()
    assert loop._alibaba_host_is_china("dashscope.aliyuncs.com") is True
    assert loop._alibaba_host_is_china("token-plan.cn-beijing.maas.aliyuncs.com") is True
    assert loop._alibaba_host_is_china("modelstudio.cn-beijing.aliyuncs.com") is True
    assert loop._alibaba_host_is_china("bailian.console.aliyun.com") is True
    assert loop._alibaba_host_is_china(loop.ALIBABA_TOKEN_PLAN_MINT_HOST) is False
    assert (
        loop._alibaba_host_is_china(
            urlsplit(loop.DEFAULT_ALIBABA_QUOTA_GATEWAY_URL).hostname
        )
        is False
    )
    with pytest.raises(loop.AlibabaAuthError, match="China"):
        loop._alibaba_quota_request_url(
            _alibaba_quota_poll_config(
                alibaba_quota_gateway_url="https://bailian.console.aliyun.com/cli/api.json"
            ),
            api_name=loop.ALIBABA_TOKEN_PLAN_USAGE_API,
        )
    assert "cn-beijing" not in config.alibaba_quota_gateway_url
    assert "aliyun.com" not in urlsplit(config.alibaba_quota_gateway_url).hostname


def test_alibaba_quota_request_never_uses_cookie_or_ticket_file(monkeypatch) -> None:
    config = _alibaba_quota_poll_config()
    auth = _alibaba_ram_auth(monkeypatch)
    request = loop._build_alibaba_quota_request(
        config,
        api_name=loop.ALIBABA_TOKEN_PLAN_USAGE_API,
        access_token="console-bearer-secret",
    )
    body = parse_qs((request.data or b"").decode("utf-8"))
    query = parse_qs(urlsplit(request.full_url).query)

    assert "login_ticket" not in auth
    assert "sec_token" not in auth
    assert "cookie_jar" not in vars(loop.AlibabaConsoleSession("fingerprint"))
    assert request.get_header("Authorization") == "Bearer console-bearer-secret"
    assert request.get_header("Cookie") is None
    assert "sec_token" not in body
    assert "login_aliyunid_ticket" not in body
    assert "_v" not in query
    assert not hasattr(config, "alibaba_web_auth_file")
    assert not hasattr(loop, "_load_alibaba_web_auth")
    assert not hasattr(loop, "_new_alibaba_web_session")
    assert not hasattr(loop, "_bootstrap_alibaba_web_session")
    assert not hasattr(loop, "AlibabaWebSession")


def test_alibaba_quota_payloads_map_consumed_fractions_and_hash_identity() -> None:
    observed_at = datetime(2026, 7, 21, 18, 0, tzinfo=timezone.utc)
    subscription = loop._parse_alibaba_subscription_payload(_alibaba_subscription_payload())

    payloads = loop._build_alibaba_quota_rate_limit_payloads(
        _alibaba_quota_poll_config(),
        observed_at=observed_at,
        usage_payload=_alibaba_usage_payload(),
        subscription=subscription,
    )

    # 2 windows x 6 active models = 12 rows.  The account-wide quota is shared,
    # so all models carry identical remaining_pct per window.
    assert len(payloads) == 12
    assert [payload[6] for payload in payloads] == [
        loop.ALIBABA_TOKEN_PLAN_5H_QUOTA_KEY,
        loop.ALIBABA_TOKEN_PLAN_5H_QUOTA_KEY,
        loop.ALIBABA_TOKEN_PLAN_5H_QUOTA_KEY,
        loop.ALIBABA_TOKEN_PLAN_5H_QUOTA_KEY,
        loop.ALIBABA_TOKEN_PLAN_5H_QUOTA_KEY,
        loop.ALIBABA_TOKEN_PLAN_5H_QUOTA_KEY,
        loop.ALIBABA_TOKEN_PLAN_7D_QUOTA_KEY,
        loop.ALIBABA_TOKEN_PLAN_7D_QUOTA_KEY,
        loop.ALIBABA_TOKEN_PLAN_7D_QUOTA_KEY,
        loop.ALIBABA_TOKEN_PLAN_7D_QUOTA_KEY,
        loop.ALIBABA_TOKEN_PLAN_7D_QUOTA_KEY,
        loop.ALIBABA_TOKEN_PLAN_7D_QUOTA_KEY,
    ]
    assert [payload[10] for payload in payloads] == [
        75.0,
        75.0,
        75.0,
        75.0,
        75.0,
        75.0,
        50.0,
        50.0,
        50.0,
        50.0,
        50.0,
        50.0,
    ]
    assert all(payload[11:14] == (None, None, None) for payload in payloads)
    assert all(payload[4] == loop.ALIBABA_TOKEN_PLAN_PROVIDER for payload in payloads)
    assert all(payload[18] == loop.ALIBABA_TOKEN_PLAN_SOURCE for payload in payloads)
    assert payloads[0][3] == hashlib.sha256(b"alibaba-token-plan|instanceCode=instance-secret-identifier").hexdigest()
    persisted_json = json.dumps(
        [json.loads(payload[16]) for payload in payloads] + [json.loads(payload[17]) for payload in payloads]
    )
    assert "instance-secret-identifier" not in persisted_json
    assert "login-ticket-secret" not in persisted_json
    assert "security-token-secret" not in persisted_json


def _alibaba_auth_envelope(
    *,
    error_code: str = "ConsoleNeedLogin",
    error_msg: str = "session-expired-secret-detail",
) -> dict:
    """HTTP 200 application-level auth envelope (nested DataV2 shape).

    Gateway transport success (top-level successResponse) but inner
    data.DataV2.data.success=false with errorCode/errorMsg.
    """
    return {
        "successResponse": True,
        "data": {
            "DataV2": {
                "data": {
                    "code": "SUCCESS",
                    "success": False,
                    "errorCode": error_code,
                    "errorMsg": error_msg,
                }
            }
        },
    }


def _alibaba_auth_envelope_live(
    *,
    error_code: str = "ConsoleNeedLogin",
    error_msg: str = "session-expired-secret-detail",
) -> dict:
    """HTTP 200 application-level auth envelope (exact live shape).

    The live console gateway places success=false + errorCode/errorMsg
    directly under top-level ``data`` with NO DataV2 wrapper.  This is the
    exact key path observed in production.
    """
    return {
        "successResponse": True,
        "data": {
            "success": False,
            "errorCode": error_code,
            "errorMsg": error_msg,
        },
    }


def test_alibaba_console_envelope_auth_detection_unit() -> None:
    # Known auth envelope: success False + a known login/session signal.
    assert loop._alibaba_console_envelope_is_auth_failure(
        {"success": False, "errorCode": "ConsoleNeedLogin", "errorMsg": "x"}
    ) is True
    # Session-expired signal in errorMsg qualifies.
    assert loop._alibaba_console_envelope_is_auth_failure(
        {"success": False, "errorMsg": "SessionExpired"}
    ) is True
    # Narrow classifier: quota/capacity signals are NOT auth.
    assert loop._alibaba_console_envelope_is_auth_failure(
        {"success": False, "errorCode": "QuotaExceeded", "errorMsg": "capacity"}
    ) is False
    # Internal/server error signals are NOT auth.
    assert loop._alibaba_console_envelope_is_auth_failure(
        {"success": False, "errorCode": "InternalError", "errorMsg": "boom"}
    ) is False
    # Generic error fields with no known auth signal are NOT auth.
    assert loop._alibaba_console_envelope_is_auth_failure(
        {"success": False, "errorCode": "SomeRandomCode", "errorMsg": "whatever"}
    ) is False
    # success True is never an auth failure.
    assert loop._alibaba_console_envelope_is_auth_failure(
        {"success": True, "errorCode": "x"}
    ) is False
    # success False but no errorCode/errorMsg -> not the known auth envelope.
    assert loop._alibaba_console_envelope_is_auth_failure(
        {"success": False}
    ) is False
    # Non-mapping -> False.
    assert loop._alibaba_console_envelope_is_auth_failure(None) is False


def test_extract_alibaba_console_data_classifies_auth_envelope_as_auth() -> None:
    payload = _alibaba_auth_envelope(
        error_code="ConsoleNeedLogin",
        error_msg="login-aliyunid-ticket-expired-secret",
    )
    with pytest.raises(loop.AlibabaQuotaPollError) as exc_info:
        loop._extract_alibaba_console_data(payload, endpoint="usage")
    assert exc_info.value.telemetry_class == "auth"
    assert exc_info.value.status_code == 200
    # Fail-closed: no provider data returned.


def test_extract_alibaba_console_data_classifies_live_auth_envelope_as_auth() -> None:
    """Exact live key path: payload.data.success=false + errorCode/errorMsg
    directly under top-level data, no DataV2 wrapper."""
    payload = _alibaba_auth_envelope_live(
        error_code="ConsoleNeedLogin",
        error_msg="login-aliyunid-ticket-expired-secret",
    )
    with pytest.raises(loop.AlibabaQuotaPollError) as exc_info:
        loop._extract_alibaba_console_data(payload, endpoint="usage")
    assert exc_info.value.telemetry_class == "auth"
    assert exc_info.value.status_code == 200


def test_extract_alibaba_console_data_live_auth_envelope_never_logs_raw_error() -> None:
    """Live-shape auth envelope must never expose raw errorCode/errorMsg."""
    # Auth-matching code so the auth path is exercised; secret-looking msg.
    secret_code = "ConsoleNeedLogin"
    secret_msg = "SECRET_LIVE_login_ticket_value_xyz"
    payload = _alibaba_auth_envelope_live(error_code=secret_code, error_msg=secret_msg)
    with pytest.raises(loop.AlibabaQuotaPollError) as exc_info:
        loop._extract_alibaba_console_data(payload, endpoint="subscription")
    assert exc_info.value.telemetry_class == "auth"
    rendered = str(exc_info.value)
    assert secret_code not in rendered
    assert secret_msg not in rendered


def test_extract_alibaba_console_data_auth_envelope_never_logs_raw_error() -> None:
    # Auth-matching code so the auth path is exercised; secret-looking msg.
    secret_code = "SessionExpired"
    secret_msg = "SECRET_login_ticket_value_xyz"
    payload = _alibaba_auth_envelope(error_code=secret_code, error_msg=secret_msg)
    with pytest.raises(loop.AlibabaQuotaPollError) as exc_info:
        loop._extract_alibaba_console_data(payload, endpoint="subscription")
    assert exc_info.value.telemetry_class == "auth"
    rendered = str(exc_info.value)
    assert secret_code not in rendered
    assert secret_msg not in rendered


def test_extract_alibaba_console_data_quota_exceeded_is_not_auth() -> None:
    """A quota/capacity envelope must remain fail-closed under contract_drift,
    never auth."""
    payload = _alibaba_auth_envelope_live(
        error_code="QuotaExceeded",
        error_msg="plan-capacity-exhausted",
    )
    with pytest.raises(loop.AlibabaQuotaPollError) as exc_info:
        loop._extract_alibaba_console_data(payload, endpoint="usage")
    assert exc_info.value.telemetry_class == "contract_drift"


def test_extract_alibaba_console_data_internal_error_is_not_auth() -> None:
    """An internal/server error envelope must remain fail-closed under
    contract_drift, never auth."""
    payload = _alibaba_auth_envelope_live(
        error_code="InternalError",
        error_msg="upstream-blew-up",
    )
    with pytest.raises(loop.AlibabaQuotaPollError) as exc_info:
        loop._extract_alibaba_console_data(payload, endpoint="usage")
    assert exc_info.value.telemetry_class == "contract_drift"


def test_alibaba_auth_allowlist_natural_language_session_expired_is_auth() -> None:
    """Finding 3: a natural-language errorMsg such as
    'The login session has expired' (no errorCode) must be classified as
    auth via weak+expiry keyword pairing, while raw text is never exposed."""
    assert loop._alibaba_error_text_matches_auth_allowlist(
        {"errorMsg": "The login session has expired"}
    ) is True
    # Full envelope path: live shape, message-only, classified auth.
    payload = _alibaba_auth_envelope_live(
        error_code="",
        error_msg="The login session has expired",
    )
    with pytest.raises(loop.AlibabaQuotaPollError) as exc_info:
        loop._extract_alibaba_console_data(payload, endpoint="usage")
    assert exc_info.value.telemetry_class == "auth"
    rendered = str(exc_info.value)
    assert "The login session has expired" not in rendered


def test_alibaba_auth_allowlist_console_need_login_is_auth() -> None:
    """ConsoleNeedLogin is a strong auth signal regardless of message."""
    assert loop._alibaba_error_text_matches_auth_allowlist(
        {"errorCode": "ConsoleNeedLogin", "errorMsg": "please sign in"}
    ) is True


def test_alibaba_auth_allowlist_quota_exceeded_is_not_auth() -> None:
    """QuotaExceeded must remain non-auth even with a session word present."""
    assert loop._alibaba_error_text_matches_auth_allowlist(
        {"errorCode": "QuotaExceeded", "errorMsg": "session quota exhausted"}
    ) is False


def test_alibaba_auth_allowlist_internal_error_is_not_auth() -> None:
    """InternalError / server envelopes must remain non-auth."""
    assert loop._alibaba_error_text_matches_auth_allowlist(
        {"errorCode": "InternalError", "errorMsg": "upstream server failure"}
    ) is False
    assert loop._alibaba_error_text_matches_auth_allowlist(
        {"errorMsg": "ServiceUnavailable"}
    ) is False


def test_alibaba_auth_allowlist_bare_session_word_is_not_auth() -> None:
    """A bare weak keyword (session) without an expiry/invalid keyword must
    NOT be auth, so quota/server messages mentioning a session stay fail-closed."""
    assert loop._alibaba_error_text_matches_auth_allowlist(
        {"errorMsg": "session quota usage report"}
    ) is False


def test_alibaba_auth_allowlist_notlogin_with_generic_gateway_is_auth() -> None:
    """Regression: live Alibaba envelope has errorCode=NotLogin with errorMsg
    mentioning 'gateway' (the console gateway infrastructure).  The bare
    generic 'gateway' token must NOT suppress the strong notlogin auth signal.
    Matches secret-safe live classification: strong=[notlogin], weak=[login],
    expiry=[], non_auth=[gateway]."""
    # Classifier unit: NotLogin + generic gateway wording -> auth.
    assert loop._alibaba_error_text_matches_auth_allowlist(
        {"errorCode": "NotLogin", "errorMsg": "please login via gateway console"}
    ) is True
    # Full envelope path: live shape, classified as auth not contract_drift.
    payload = _alibaba_auth_envelope_live(
        error_code="NotLogin",
        error_msg="please login via gateway console",
    )
    with pytest.raises(loop.AlibabaQuotaPollError) as exc_info:
        loop._extract_alibaba_console_data(payload, endpoint="usage")
    assert exc_info.value.telemetry_class == "auth"
    assert exc_info.value.status_code == 200
    # Redaction: raw error text never exposed.
    rendered = str(exc_info.value)
    assert "NotLogin" not in rendered
    assert "please login via gateway console" not in rendered


def test_alibaba_auth_allowlist_specific_non_auth_still_overrides_strong() -> None:
    """Specific non-auth guards (badgateway, quota, capacity, internal, server,
    upstream) must remain fail-closed even when a strong auth keyword is also
    present."""
    for non_auth_msg in (
        "badgateway after login",
        "quota exceeded for login session",
        "capacity exhausted unauthorized",
        "internal error unauthorized",
        "server error needlogin",
        "upstream timeout forbidden",
    ):
        assert loop._alibaba_error_text_matches_auth_allowlist(
            {"errorCode": "NotLogin", "errorMsg": non_auth_msg}
        ) is False, f"expected non-auth for: {non_auth_msg}"

def test_extract_alibaba_console_data_genuine_contract_drift_still_drift() -> None:
    # success False but NO errorCode/errorMsg -> genuine contract drift, not auth.
    payload = {
        "data": {
            "DataV2": {
                "data": {"code": "SUCCESS", "success": False}
            }
        }
    }
    with pytest.raises(loop.AlibabaQuotaPollError) as exc_info:
        loop._extract_alibaba_console_data(payload, endpoint="usage")
    assert exc_info.value.telemetry_class == "contract_drift"


def test_extract_alibaba_console_data_unrecognized_shape_is_contract_drift() -> None:
    # Missing inner structure entirely -> contract drift (fail closed).
    with pytest.raises(loop.AlibabaQuotaPollError) as exc_info:
        loop._extract_alibaba_console_data({"data": {}}, endpoint="usage")
    assert exc_info.value.telemetry_class == "contract_drift"


def test_extract_alibaba_console_data_success_returns_provider_data() -> None:
    payload = _alibaba_console_envelope(_alibaba_usage_payload())
    provider_data = loop._extract_alibaba_console_data(payload, endpoint="usage")
    assert provider_data["per5HourPercentage"] == 0.25


def test_alibaba_reset_card_parser_builds_sanitized_credit_rows() -> None:
    reset_cards = _alibaba_reset_card_payload()
    assert loop._extract_alibaba_console_reset_card_list(
        _alibaba_console_envelope(reset_cards),
        endpoint="reset_cards",
    ) == reset_cards
    with pytest.raises(loop.AlibabaQuotaPollError) as exc_info:
        loop._extract_alibaba_console_reset_card_list(
            _alibaba_console_envelope({"cards": reset_cards}),
            endpoint="reset_cards",
        )
    assert exc_info.value.telemetry_class == "contract_drift"

    subscription = loop._parse_alibaba_subscription_payload(_alibaba_subscription_payload())
    rows, available_count = loop._build_alibaba_reset_card_observations(
        _alibaba_quota_poll_config(),
        observed_at=datetime(2026, 8, 11, 12, 0, tzinfo=timezone.utc),
        reset_cards=reset_cards,
        subscription=subscription,
        status_code=200,
        attempt_count=1,
        retry_count=0,
        auth_source="auth_file",
    )

    available, expired = rows
    assert available_count == 1
    assert (
        available["provider"],
        available["credit_family"],
        available["credit_type"],
        available["status"],
        available["available_count"],
        expired["status"],
        expired["available_count"],
    ) == (
        loop.ALIBABA_TOKEN_PLAN_PROVIDER,
        loop.ALIBABA_TOKEN_PLAN_RESET_CARD_CREDIT_FAMILY,
        "manual_reset_card",
        "available",
        1,
        "expired",
        0,
    )
    assert available["raw_provider_fields"] == {
        "parser_version": loop.ALIBABA_TOKEN_PLAN_RESET_CARD_PARSER_VERSION,
        "cardType": "WEEKLY",
        "effectiveAt": 1785542400000,
        "expiresAt": 1788220800000,
    }
    assert available["credit_identity"] == hashlib.sha256(
        (
            f"{subscription['account_hash']}|"
            f"{loop.ALIBABA_TOKEN_PLAN_RESET_CARD_CREDIT_FAMILY}|"
            "reset-card-secret-available"
        ).encode()
    ).hexdigest()
    serialized = json.dumps(rows, default=str)
    for forbidden in (
        "cardNo",
        "reset-card-secret-available",
        "reset-card-secret-expired",
        "instance-secret-identifier",
    ):
        assert forbidden not in serialized


def test_alibaba_reset_card_persistence_dedupes_and_transitions_empty_inventory(
    monkeypatch,
) -> None:
    config = _alibaba_quota_poll_config()
    observed_at = datetime(2026, 8, 11, 12, 0, tzinfo=timezone.utc)
    subscription = loop._parse_alibaba_subscription_payload(_alibaba_subscription_payload())
    observations, available_count = loop._build_alibaba_reset_card_observations(
        config,
        observed_at=observed_at,
        reset_cards=_alibaba_reset_card_payload(),
        subscription=subscription,
        status_code=200,
        attempt_count=1,
        retry_count=0,
    )
    inserted_rows = []
    current_rows = observations
    load_calls = []

    def fake_load(_dsn, **kwargs):
        load_calls.append(kwargs)
        return current_rows

    def fake_insert(_dsn, rows, **_kwargs):
        inserted_rows.extend(rows)
        return 0

    monkeypatch.setattr(probes, "load_provider_credit_current_rows", fake_load)
    monkeypatch.setattr(probes, "insert_provider_credit_observations", fake_insert)
    assert loop._persist_alibaba_reset_card_observations(
        config,
        observed_at=observed_at,
        observations=observations,
        account_hash=subscription["account_hash"],
        available_count=available_count,
        status_code=200,
        attempt_count=1,
        retry_count=0,
    ) == (2, 0)
    assert inserted_rows == observations

    empty_rows, empty_count = loop._build_alibaba_reset_card_observations(
        config,
        observed_at=observed_at,
        reset_cards=[],
        subscription=subscription,
        status_code=200,
        attempt_count=1,
        retry_count=0,
    )
    current_rows = [
        dict(observations[0], status="available"),
        dict(observations[1], status="available"),
        dict(observations[0], credit_identity="already-used", status="used"),
    ]
    lifecycle_rows = loop._synthesize_alibaba_reset_card_lifecycle_observations(
        config,
        observed_at=observed_at,
        account_hash=subscription["account_hash"],
        visible_identities=set(),
        visible_card_count=0,
        available_count=empty_count,
        status_code=200,
        attempt_count=1,
        retry_count=0,
    )
    assert empty_rows == []
    assert empty_count == 0
    assert {
        row["status"]: row["evidence"]["lifecycle_reason"]
        for row in lifecycle_rows
    } == {
        "used": "card_missing_before_expiry",
        "expired": "card_past_expiry",
    }
    assert load_calls[-1]["provider"] == loop.ALIBABA_TOKEN_PLAN_PROVIDER
    assert load_calls[-1]["credit_family"] == loop.ALIBABA_TOKEN_PLAN_RESET_CARD_CREDIT_FAMILY
    assert load_calls[-1]["source"] == loop.ALIBABA_TOKEN_PLAN_RESET_CARD_SOURCE


def test_alibaba_quota_payloads_emit_exact_active_model_identities() -> None:
    import yaml

    repo_root = Path(__file__).resolve().parents[2]
    configured_models = {
        candidate["model"]
        for path in (repo_root / "litellm/proxy/aawm_alias_config").glob("*.yaml")
        for alias in yaml.safe_load(path.read_text(encoding="utf-8")).get(
            "aliases", []
        )
        for candidate in alias.get("candidates", [])
        if candidate.get("provider") == loop.ALIBABA_TOKEN_PLAN_PROVIDER
    }
    assert set(loop.ALIBABA_TOKEN_PLAN_ACTIVE_MODELS) == configured_models

    subscription = loop._parse_alibaba_subscription_payload(_alibaba_subscription_payload())
    payloads = loop._build_alibaba_quota_rate_limit_payloads(
        _alibaba_quota_poll_config(),
        observed_at=datetime(2026, 7, 21, 18, 0, tzinfo=timezone.utc),
        usage_payload=_alibaba_usage_payload(),
        subscription=subscription,
    )
    # Exact truthful model identities for all active alias candidates.
    models = [payload[5] for payload in payloads]
    assert models == list(loop.ALIBABA_TOKEN_PLAN_ACTIVE_MODELS) * 2
    # Shared account-wide quota: identical remaining_pct per window across models.
    by_window = {}
    for payload in payloads:
        by_window.setdefault(payload[6], []).append(payload[10])
    assert by_window[loop.ALIBABA_TOKEN_PLAN_5H_QUOTA_KEY] == [75.0] * 6
    assert by_window[loop.ALIBABA_TOKEN_PLAN_7D_QUOTA_KEY] == [50.0] * 6
    # Shared-quota scope recorded in evidence.
    for payload in payloads:
        evidence = json.loads(payload[17])
        assert "account-wide" in evidence["quota_scope"]
        assert all(
            model in evidence["quota_scope"]
            for model in loop.ALIBABA_TOKEN_PLAN_ACTIVE_MODELS
        )


def test_alibaba_quota_payloads_allow_absent_reset_for_unused_window() -> None:
    usage = _alibaba_usage_payload()
    usage["per5HourPercentage"] = 0.0
    usage.pop("per5HourResetTime")

    payloads = loop._build_alibaba_quota_rate_limit_payloads(
        _alibaba_quota_poll_config(),
        observed_at=datetime(2026, 7, 21, 18, 0, tzinfo=timezone.utc),
        usage_payload=usage,
        subscription=loop._parse_alibaba_subscription_payload(
            _alibaba_subscription_payload()
        ),
    )

    assert payloads[0][9] is None
    assert payloads[0][10] == 100.0
    raw_provider_fields = json.loads(payloads[0][16])
    evidence = json.loads(payloads[0][17])
    assert raw_provider_fields["reset_at_ms"] is None
    assert raw_provider_fields["reset_at_state"] == "absent_unused_window"
    assert "alibaba_token_plan_reset_absent_unused_window" in evidence["signals"]


def test_alibaba_quota_payloads_reject_absent_reset_for_consumed_window() -> None:
    usage = _alibaba_usage_payload()
    usage.pop("per5HourResetTime")

    with pytest.raises(ValueError, match="per5HourResetTime"):
        loop._build_alibaba_quota_rate_limit_payloads(
            _alibaba_quota_poll_config(),
            observed_at=datetime(2026, 7, 21, 18, 0, tzinfo=timezone.utc),
            usage_payload=usage,
            subscription=loop._parse_alibaba_subscription_payload(
                _alibaba_subscription_payload()
            ),
        )


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("per5HourPercentage", -0.01),
        ("per5HourPercentage", 1.01),
        ("per1WeekPercentage", "not-a-number"),
        ("per5HourResetTime", 123),
        ("per1WeekResetTime", "not-a-timestamp"),
    ],
)
def test_alibaba_quota_payloads_reject_invalid_provider_values(
    field_name,
    value,
) -> None:
    usage = _alibaba_usage_payload()
    usage[field_name] = value

    with pytest.raises(ValueError, match="Alibaba"):
        loop._build_alibaba_quota_rate_limit_payloads(
            _alibaba_quota_poll_config(),
            observed_at=datetime(2026, 7, 21, 18, 0, tzinfo=timezone.utc),
            usage_payload=usage,
            subscription=loop._parse_alibaba_subscription_payload(_alibaba_subscription_payload()),
        )


def test_alibaba_subscription_parser_rejects_inactive_plan() -> None:
    payload = _alibaba_subscription_payload()
    payload["status"] = "EXPIRED"

    with pytest.raises(ValueError, match="not active"):
        loop._parse_alibaba_subscription_payload(payload)


def test_alibaba_quota_payloads_emit_weekly_only_for_live_weekly_payload() -> None:
    # Live authenticated usage payload from 2026-08-08: the 5-hour pair is
    # wholly absent; only the weekly window is present.
    usage = {
        "per1WeekResetTime": 1786299120000,
        "per1WeekPercentage": 1.0,
    }

    payloads = loop._build_alibaba_quota_rate_limit_payloads(
        _alibaba_quota_poll_config(),
        observed_at=datetime(2026, 8, 8, 12, 0, tzinfo=timezone.utc),
        usage_payload=usage,
        subscription=loop._parse_alibaba_subscription_payload(
            _alibaba_subscription_payload()
        ),
    )

    # Only the 7-day window, one row per active model identity.
    assert [payload[6] for payload in payloads] == [
        loop.ALIBABA_TOKEN_PLAN_7D_QUOTA_KEY,
        loop.ALIBABA_TOKEN_PLAN_7D_QUOTA_KEY,
        loop.ALIBABA_TOKEN_PLAN_7D_QUOTA_KEY,
        loop.ALIBABA_TOKEN_PLAN_7D_QUOTA_KEY,
        loop.ALIBABA_TOKEN_PLAN_7D_QUOTA_KEY,
        loop.ALIBABA_TOKEN_PLAN_7D_QUOTA_KEY,
    ]
    assert [payload[7] for payload in payloads] == ["7d"] * 6
    assert [payload[5] for payload in payloads] == list(
        loop.ALIBABA_TOKEN_PLAN_ACTIVE_MODELS
    )
    assert [payload[10] for payload in payloads] == [0.0] * 6
    assert all(
        payload[9] == datetime(2026, 8, 9, 18, 12, tzinfo=timezone.utc)
        for payload in payloads
    )
    assert all(payload[18] == loop.ALIBABA_TOKEN_PLAN_SOURCE for payload in payloads)
    for payload in payloads:
        raw_provider_fields = json.loads(payload[16])
        assert raw_provider_fields["window"] == "7d"
        assert raw_provider_fields["remaining_pct"] == 0.0
        assert raw_provider_fields["reset_at_state"] == "valid"


def test_alibaba_quota_payloads_reject_empty_usage_payload() -> None:
    with pytest.raises(ValueError, match="no recognized quota window"):
        loop._build_alibaba_quota_rate_limit_payloads(
            _alibaba_quota_poll_config(),
            observed_at=datetime(2026, 8, 8, 12, 0, tzinfo=timezone.utc),
            usage_payload={},
            subscription=loop._parse_alibaba_subscription_payload(
                _alibaba_subscription_payload()
            ),
        )


def test_alibaba_quota_payloads_reject_partial_5h_window() -> None:
    # Reset key present but percentage key absent is partial, not absent.
    usage = _alibaba_usage_payload()
    usage.pop("per5HourPercentage")

    with pytest.raises(ValueError, match="per5HourPercentage"):
        loop._build_alibaba_quota_rate_limit_payloads(
            _alibaba_quota_poll_config(),
            observed_at=datetime(2026, 8, 8, 12, 0, tzinfo=timezone.utc),
            usage_payload=usage,
            subscription=loop._parse_alibaba_subscription_payload(
                _alibaba_subscription_payload()
            ),
        )


def test_alibaba_acs3_mint_signs_empty_body_on_intl_host(monkeypatch, capsys) -> None:
    mint_calls = _install_alibaba_mint(monkeypatch, _alibaba_mint_ok("console-bearer-secret"))
    token = loop._mint_alibaba_console_access_token(
        _alibaba_quota_poll_config(),
        _alibaba_ram_auth(monkeypatch),
        acs_date=ALIBABA_TEST_ACS_DATE,
        signature_nonce=ALIBABA_TEST_SIGNATURE_NONCE,
    )
    captured = capsys.readouterr()
    request = mint_calls[0]

    assert token == "console-bearer-secret"
    assert request["host"] == "modelstudio.ap-southeast-1.aliyuncs.com"
    assert request["path"] == "/modelstudio/cli/generateAccessToken"
    assert request["body"] == b""
    assert request["headers"]["x-acs-action"] == "GenerateCLIAccessToken"
    assert request["headers"]["x-acs-version"] == "2026-02-10"
    assert request["headers"]["x-acs-content-sha256"] == ALIBABA_TEST_EMPTY_BODY_SHA256
    assert request["headers"]["authorization"].endswith(
        f"Signature={ALIBABA_TEST_ACS3_SIGNATURE}"
    )
    assert not loop._alibaba_host_is_china(request["host"])
    _assert_no_alibaba_secrets(
        json.dumps(request["headers"]) + captured.out + captured.err
    )


@pytest.mark.parametrize(
    ("status_code", "payload"),
    [
        (403, {"Code": "NoPermission", "Message": "denied", "Success": False}),
        (200, {"Code": "NoPermission", "Message": "denied", "Success": False}),
        (403, {"success": False, "errorCode": "Forbidden", "errorMsg": "denied"}),
    ],
)
def test_alibaba_mint_permission_denied_is_auth_and_keeps_last_good(
    monkeypatch,
    status_code,
    payload,
) -> None:
    monkeypatch.setattr(
        loop,
        "ALIBABA_MINT_HTTP_POST_FN",
        lambda **_kwargs: (status_code, json.dumps(payload)),
    )
    auth = _alibaba_ram_auth(monkeypatch)
    last_good_subscription = loop._parse_alibaba_subscription_payload(
        _alibaba_subscription_payload()
    )
    state = loop.SidecarTaskState(
        alibaba_auth_fingerprint=str(auth["credential_fingerprint"]),
        alibaba_quota_last_attempt_monotonic=0.0,
        alibaba_subscription_last_attempt_monotonic=50.0,
        alibaba_subscription_payload=last_good_subscription,
    )

    with pytest.raises(loop.AlibabaQuotaPollError) as exc_info:
        loop._mint_alibaba_console_access_token(
            _alibaba_quota_poll_config(),
            auth,
        )

    events = _alibaba_poll_events(
        loop.run_due_sidecar_tasks(
            _alibaba_quota_poll_config(),
            state,
            now_monotonic=400.0,
        )
    )

    assert exc_info.value.telemetry_class == "auth"
    assert exc_info.value.endpoint == "authentication"
    assert exc_info.value.mint_attempted is True
    assert events[0]["telemetry_class"] == "auth"
    assert events[0]["error_endpoint"] == "authentication"
    assert events[0]["last_good_state_retained"] is True
    assert events[0]["mint_attempted"] is True
    assert events[0]["mint_succeeded"] is False
    assert events[0]["subscription_refreshed"] is False
    assert events[0]["persisted"] is False
    assert state.alibaba_subscription_payload is last_good_subscription
    assert "NoPermission" not in str(exc_info.value)
    _assert_no_alibaba_secrets(str(exc_info.value) + json.dumps(events))


def test_fetch_alibaba_quota_payload_mints_once_and_reuses_cached_bearer(
    monkeypatch,
) -> None:
    config = _alibaba_quota_poll_config()
    auth = _alibaba_ram_auth(monkeypatch)
    mint_calls = _install_alibaba_mint(
        monkeypatch, _alibaba_mint_ok("console-bearer-secret")
    )

    def handler(request, timeout):
        del timeout
        body = parse_qs((request.data or b"").decode("utf-8"))
        assert set(body) == {"params", "region"}
        assert request.get_header("Cookie") is None
        return _alibaba_http_success(_alibaba_console_envelope(_alibaba_usage_payload()))

    gateway_calls = _install_alibaba_quota_open(monkeypatch, handler)
    state = loop.SidecarTaskState()
    session = loop.AlibabaConsoleSession(credential_fingerprint=auth["credential_fingerprint"])

    first = loop._fetch_alibaba_quota_payload(
        config,
        api_name=loop.ALIBABA_TOKEN_PLAN_USAGE_API,
        endpoint="usage",
        auth=auth,
        session=session,
        task_state=state,
    )
    second = loop._fetch_alibaba_quota_payload(
        config,
        api_name=loop.ALIBABA_TOKEN_PLAN_USAGE_API,
        endpoint="usage",
        auth=auth,
        session=session,
        task_state=state,
    )

    assert len(mint_calls) == 1
    assert mint_calls[0]["host"] == loop.ALIBABA_TOKEN_PLAN_MINT_HOST
    assert mint_calls[0]["path"] == loop.ALIBABA_TOKEN_PLAN_MINT_PATH
    assert mint_calls[0]["body"] == b""
    assert [
        request.get_header("Authorization") for request in gateway_calls
    ] == [
        "Bearer console-bearer-secret",
        "Bearer console-bearer-secret",
    ]
    assert first["payload"] == _alibaba_usage_payload()
    assert second["payload"] == _alibaba_usage_payload()
    assert first["mint_attempted"] is True
    assert first["mint_succeeded"] is True
    assert second["mint_attempted"] is False
    assert state.alibaba_access_token == "console-bearer-secret"
    assert session.access_token == "console-bearer-secret"


@pytest.mark.parametrize("failure_kind", ["429", "503", "transport"])
def test_fetch_alibaba_quota_payload_retries_transient_failure(
    monkeypatch,
    failure_kind,
) -> None:
    config = _alibaba_quota_poll_config(
        alibaba_quota_poll_max_attempts=2,
        alibaba_quota_poll_retry_backoff_seconds=0.5,
    )
    attempts = {"count": 0}
    sleeps: list[float] = []

    def fake_urlopen(request, timeout):
        del timeout
        assert request.get_header("Authorization") == "Bearer console-bearer-secret"
        attempts["count"] += 1
        if attempts["count"] == 1:
            if failure_kind == "transport":
                raise urllib_error.URLError("temporary connection failure")
            raise urllib_error.HTTPError(
                config.alibaba_quota_gateway_url,
                int(failure_kind),
                "temporary failure",
                hdrs=None,
                fp=BytesIO(b"{}"),
            )
        return _alibaba_http_success(_alibaba_console_envelope(_alibaba_usage_payload()))

    _install_alibaba_quota_open(monkeypatch, fake_urlopen)
    monkeypatch.setattr(
        loop,
        "ALIBABA_QUOTA_POLL_SLEEP_FN",
        lambda seconds: sleeps.append(seconds),
    )

    fetched = loop._fetch_alibaba_quota_payload(
        config,
        api_name=loop.ALIBABA_TOKEN_PLAN_USAGE_API,
        endpoint="usage",
        auth=_alibaba_ram_auth(monkeypatch),
        session=loop.AlibabaConsoleSession(
            credential_fingerprint="fingerprint",
            access_token="console-bearer-secret",
        ),
    )

    assert fetched["status_code"] == 200
    assert fetched["payload"] == _alibaba_usage_payload()
    assert fetched["attempt_count"] == 2
    assert fetched["retry_count"] == 1
    assert sleeps == [0.5]


def test_fetch_alibaba_quota_payload_gateway_success_uses_cli_api_and_bearer(
    monkeypatch,
) -> None:
    config = _alibaba_quota_poll_config()
    inner = _alibaba_gateway_success_handler(
        usage_payload=_alibaba_weekly_only_usage_payload(),
        reset_cards=[],
    )

    def handler(request, timeout):
        query = parse_qs(urlsplit(request.full_url).query)
        body = parse_qs((request.data or b"").decode("utf-8"))
        request._alibaba_seen = {
            "path": urlsplit(request.full_url).path,
            "api": query["api"][0],
            "authorization": request.get_header("Authorization"),
            "cookie": request.get_header("Cookie"),
            "body_keys": sorted(body),
        }
        return inner(request, timeout)

    calls = _install_alibaba_quota_open(monkeypatch, handler)
    session = loop.AlibabaConsoleSession(
        credential_fingerprint="fingerprint",
        access_token="console-bearer-secret",
    )
    usage = loop._fetch_alibaba_quota_payload(
        config,
        api_name=loop.ALIBABA_TOKEN_PLAN_USAGE_API,
        endpoint="usage",
        auth=_alibaba_ram_auth(monkeypatch),
        session=session,
    )
    subscription = loop._fetch_alibaba_quota_payload(
        config,
        api_name=loop.ALIBABA_TOKEN_PLAN_SUBSCRIPTION_API,
        endpoint="subscription",
        auth=_alibaba_ram_auth(monkeypatch),
        session=session,
    )
    reset_cards = loop._fetch_alibaba_quota_payload(
        config,
        api_name=loop.ALIBABA_TOKEN_PLAN_RESET_CARD_LIST_API,
        endpoint="reset_cards",
        auth=_alibaba_ram_auth(monkeypatch),
        session=session,
    )
    seen = [request._alibaba_seen for request in calls]

    assert [row["path"] for row in seen] == ["/cli/api.json"] * 3
    assert [row["api"] for row in seen] == [
        loop.ALIBABA_TOKEN_PLAN_USAGE_API,
        loop.ALIBABA_TOKEN_PLAN_SUBSCRIPTION_API,
        loop.ALIBABA_TOKEN_PLAN_RESET_CARD_LIST_API,
    ]
    assert {row["authorization"] for row in seen} == {"Bearer console-bearer-secret"}
    assert all(row["cookie"] is None for row in seen)
    assert all(row["body_keys"] == ["params", "region"] for row in seen)
    assert usage["payload"]["per1WeekPercentage"] == 1.0
    assert "per5HourPercentage" not in usage["payload"]
    assert subscription["payload"] == _alibaba_subscription_payload()
    assert reset_cards["payload"] == []


@pytest.mark.parametrize("status_code", [401, 403])
def test_fetch_alibaba_quota_payload_remints_once_on_http_auth_failure(
    monkeypatch,
    status_code,
) -> None:
    config = _alibaba_quota_poll_config(alibaba_quota_poll_max_attempts=3)
    mint_calls = []
    quota_calls = []

    def fake_mint(_config, _auth, **_kwargs):
        mint_calls.append("mint")
        return "console-bearer-secret"

    def fake_urlopen(request, timeout):
        del timeout
        quota_calls.append(request.get_header("Authorization"))
        if len(quota_calls) == 1:
            raise urllib_error.HTTPError(
                config.alibaba_quota_gateway_url,
                status_code,
                "auth failure",
                hdrs=None,
                fp=BytesIO(b"{}"),
            )
        return _alibaba_http_success(_alibaba_console_envelope(_alibaba_usage_payload()))

    monkeypatch.setattr(loop, "_mint_alibaba_console_access_token", fake_mint)
    monkeypatch.setattr(loop, "ALIBABA_QUOTA_HTTP_OPEN_FN", fake_urlopen)
    fetched = loop._fetch_alibaba_quota_payload(
        config,
        api_name=loop.ALIBABA_TOKEN_PLAN_USAGE_API,
        endpoint="usage",
        auth=_alibaba_ram_auth(monkeypatch),
        session=loop.AlibabaConsoleSession(
            credential_fingerprint="fingerprint",
            access_token="stale-bearer-secret",
        ),
    )

    assert mint_calls == ["mint"]
    assert quota_calls == [
        "Bearer stale-bearer-secret",
        "Bearer console-bearer-secret",
    ]
    assert fetched["status_code"] == 200
    assert fetched["attempt_count"] == 2
    assert fetched["refresh_attempted"] is True
    assert fetched["refresh_succeeded"] is True
    assert fetched["mint_succeeded"] is True


def test_fetch_alibaba_notlogined_remints_once_then_succeeds(monkeypatch) -> None:
    config = _alibaba_quota_poll_config(alibaba_quota_poll_max_attempts=1)
    mint_calls = []
    quota_calls = []

    def fake_mint(_config, _auth, **_kwargs):
        mint_calls.append("mint")
        return "console-bearer-secret"

    def fake_urlopen(request, timeout):
        del timeout
        quota_calls.append(request.get_header("Authorization"))
        if len(quota_calls) == 1:
            return _alibaba_http_success(
                _alibaba_auth_envelope_live(
                    error_code="NotLogined",
                    error_msg="please login via gateway console",
                )
            )
        return _alibaba_http_success(_alibaba_console_envelope(_alibaba_usage_payload()))

    monkeypatch.setattr(loop, "_mint_alibaba_console_access_token", fake_mint)
    monkeypatch.setattr(loop, "ALIBABA_QUOTA_HTTP_OPEN_FN", fake_urlopen)
    fetched = loop._fetch_alibaba_quota_payload(
        config,
        api_name=loop.ALIBABA_TOKEN_PLAN_USAGE_API,
        endpoint="usage",
        auth=_alibaba_ram_auth(monkeypatch),
        session=loop.AlibabaConsoleSession(
            credential_fingerprint="fingerprint",
            access_token="stale-bearer-secret",
        ),
    )

    assert mint_calls == ["mint"]
    assert quota_calls == [
        "Bearer stale-bearer-secret",
        "Bearer console-bearer-secret",
    ]
    assert fetched["payload"] == _alibaba_usage_payload()
    assert fetched["refresh_attempted"] is True
    assert fetched["refresh_succeeded"] is True


def test_fetch_alibaba_second_auth_failure_does_not_mint_again(monkeypatch) -> None:
    config = _alibaba_quota_poll_config(alibaba_quota_poll_max_attempts=3)
    mint_calls = []
    quota_calls = []

    def fake_mint(_config, _auth, **_kwargs):
        mint_calls.append("mint")
        return "console-bearer-secret"

    def fake_urlopen(request, timeout):
        del timeout
        quota_calls.append(request.get_header("Authorization"))
        return _alibaba_http_success(
            _alibaba_auth_envelope_live(
                error_code="NotLogined",
                error_msg="please login via gateway console",
            )
        )

    monkeypatch.setattr(loop, "_mint_alibaba_console_access_token", fake_mint)
    monkeypatch.setattr(loop, "ALIBABA_QUOTA_HTTP_OPEN_FN", fake_urlopen)

    with pytest.raises(loop.AlibabaQuotaPollError) as exc_info:
        loop._fetch_alibaba_quota_payload(
            config,
            api_name=loop.ALIBABA_TOKEN_PLAN_USAGE_API,
            endpoint="usage",
            auth=_alibaba_ram_auth(monkeypatch),
            session=loop.AlibabaConsoleSession(
                credential_fingerprint="fingerprint",
                access_token="stale-bearer-secret",
            ),
        )

    assert mint_calls == ["mint"]
    assert quota_calls == [
        "Bearer stale-bearer-secret",
        "Bearer console-bearer-secret",
    ]
    assert exc_info.value.telemetry_class == "auth"
    assert exc_info.value.refresh_attempted is True
    assert exc_info.value.refresh_succeeded is True
    assert exc_info.value.mint_attempted is True
    assert "NotLogined" not in str(exc_info.value)


def test_alibaba_quota_error_preserves_mint_telemetry() -> None:
    exc = loop.AlibabaQuotaPollError(
        "sanitized auth failure",
        status_code=401,
        telemetry_class="auth",
        attempt_count=2,
        retry_count=0,
        endpoint="usage",
        mint_attempted=True,
        mint_succeeded=True,
        refresh_attempted=True,
        refresh_succeeded=True,
    )
    summary = {
        "telemetry_status": None,
        "error_class": None,
        "error_message": None,
        "mint_attempted": False,
        "mint_succeeded": False,
        "refresh_attempted": False,
        "refresh_succeeded": False,
    }

    loop._record_alibaba_poll_failure(summary, exc)

    assert summary["mint_attempted"] is True
    assert summary["mint_succeeded"] is True
    assert summary["refresh_attempted"] is True
    assert summary["refresh_succeeded"] is True
    assert summary["last_good_state_retained"] is True


def test_run_due_sidecar_tasks_schedules_alibaba_quota_inventory(
    monkeypatch,
) -> None:
    config = _alibaba_quota_poll_config()
    _set_alibaba_ram_env(monkeypatch)
    calls: list[str] = []
    reset_fetch_count = 0
    reset_persisted = []

    def fake_fetch(_config, *, api_name, endpoint, auth, session, task_state=None, **kwargs):
        nonlocal reset_fetch_count
        del kwargs
        assert auth["auth_source"] == "ALIBABA_RAM_KEY"
        assert isinstance(session, loop.AlibabaConsoleSession)
        assert isinstance(task_state, loop.SidecarTaskState)
        calls.append(endpoint)
        if api_name == loop.ALIBABA_TOKEN_PLAN_SUBSCRIPTION_API:
            data = _alibaba_subscription_payload()
        elif api_name == loop.ALIBABA_TOKEN_PLAN_RESET_CARD_LIST_API:
            reset_fetch_count += 1
            data = _alibaba_reset_card_payload()
            if reset_fetch_count == 2:
                data[0]["cardNo"] = "reset-card-secret-invalid"
                data[0]["expiresAt"] = "invalid-expiry"
        else:
            data = _alibaba_usage_payload()
        return {
            "status_code": 200,
            "payload": data,
            "attempt_count": 1,
            "retry_count": 0,
            "mint_attempted": endpoint == "subscription",
            "mint_succeeded": endpoint == "subscription",
            "refresh_attempted": False,
            "refresh_succeeded": False,
        }

    monkeypatch.setattr(loop, "_fetch_alibaba_quota_payload", fake_fetch)
    monkeypatch.setattr(
        loop,
        "_persist_alibaba_quota_observations",
        lambda _config, payloads: len(payloads),
    )
    monkeypatch.setattr(
        loop,
        "_persist_alibaba_reset_card_observations",
        lambda _config, **kwargs: reset_persisted.append(kwargs["observations"])
        or (len(kwargs["observations"]),) * 2,
    )

    state = loop.SidecarTaskState()
    first = loop.run_due_sidecar_tasks(config, state, now_monotonic=100.0)
    throttled = loop.run_due_sidecar_tasks(config, state, now_monotonic=200.0)
    usage_only = loop.run_due_sidecar_tasks(config, state, now_monotonic=401.0)
    refreshed = loop.run_due_sidecar_tasks(config, state, now_monotonic=21701.0)

    assert calls == [
        "subscription",
        "usage",
        "reset_cards",
        "usage",
        "reset_cards",
        "subscription",
        "usage",
        "reset_cards",
    ]
    assert throttled == []
    assert (
        first[0]["event"],
        first[0]["subscription_refreshed"],
        first[0]["observation_count"],
        first[0]["inserted_count"],
        first[0]["persisted"],
        first[0]["reset_card_visible_count"],
        first[0]["reset_card_available_count"],
        first[0]["reset_card_observation_count"],
        first[0]["reset_card_inserted_count"],
        first[0]["reset_card_persisted"],
        first[0]["telemetry_status"],
        first[0]["auth_source"],
        first[0]["mint_attempted"],
        first[0]["mint_succeeded"],
        first[0]["token_cached"],
        first[0]["gateway_path"],
    ) == (
        "alibaba_quota_poll",
        True,
        12,
        12,
        True,
        2,
        1,
        2,
        2,
        True,
        "valid",
        "ALIBABA_RAM_KEY",
        True,
        True,
        False,
        "/cli/api.json",
    )
    assert (
        usage_only[0]["subscription_refreshed"],
        usage_only[0]["persisted"],
        usage_only[0]["reset_card_persisted"],
        usage_only[0]["telemetry_class"],
        usage_only[0]["error_endpoint"],
        usage_only[0]["last_good_state_retained"],
    ) == (False, True, False, "malformed_telemetry", "reset_cards", True)
    assert refreshed[0]["subscription_refreshed"] is True
    assert len(reset_persisted) == 2
    serialized = json.dumps(first + usage_only + refreshed)
    _assert_no_alibaba_secrets(
        serialized,
        extra=("instance-secret-identifier", "reset-card-secret-invalid", "invalid-expiry"),
    )


def test_run_due_sidecar_tasks_reuses_cached_alibaba_bearer_across_polls(
    monkeypatch,
) -> None:
    config = _alibaba_quota_poll_config(apply=False)
    auth = _alibaba_ram_auth(monkeypatch)
    mint_calls = _install_alibaba_mint(
        monkeypatch, _alibaba_mint_ok("console-bearer-secret")
    )
    gateway_calls = _install_alibaba_quota_open(
        monkeypatch,
        _alibaba_gateway_success_handler(reset_cards=_alibaba_reset_card_payload()),
    )
    state = loop.SidecarTaskState()
    first = loop.run_due_sidecar_tasks(config, state, now_monotonic=100.0)
    second = loop.run_due_sidecar_tasks(config, state, now_monotonic=401.0)
    apis = [
        parse_qs(urlsplit(request.full_url).query)["api"][0]
        for request in gateway_calls
    ]
    bodies = [
        parse_qs((request.data or b"").decode("utf-8")) for request in gateway_calls
    ]

    assert len(mint_calls) == 1
    assert mint_calls[0]["host"] == loop.ALIBABA_TOKEN_PLAN_MINT_HOST
    assert mint_calls[0]["path"] == loop.ALIBABA_TOKEN_PLAN_MINT_PATH
    assert mint_calls[0]["body"] == b""
    assert apis == [
        loop.ALIBABA_TOKEN_PLAN_SUBSCRIPTION_API,
        loop.ALIBABA_TOKEN_PLAN_USAGE_API,
        loop.ALIBABA_TOKEN_PLAN_RESET_CARD_LIST_API,
        loop.ALIBABA_TOKEN_PLAN_USAGE_API,
        loop.ALIBABA_TOKEN_PLAN_RESET_CARD_LIST_API,
    ]
    assert {
        request.get_header("Authorization") for request in gateway_calls
    } == {"Bearer console-bearer-secret"}
    assert all(request.get_header("Cookie") is None for request in gateway_calls)
    assert all("sec_token" not in body for body in bodies)
    assert state.alibaba_access_token == "console-bearer-secret"
    assert first[0]["mint_attempted"] is True
    assert second[0]["mint_attempted"] is False
    assert first[0]["token_cached"] is True
    assert second[0]["token_cached"] is True
    serialized = json.dumps(first + second)
    _assert_no_alibaba_secrets(serialized)
    assert auth["principal_hash"] is not None


def test_run_due_sidecar_tasks_resets_cached_token_when_ram_fingerprint_changes(
    monkeypatch,
) -> None:
    config = _alibaba_quota_poll_config(apply=False)
    _set_alibaba_ram_env(monkeypatch)
    mint_tokens = iter(["first-bearer-secret", "second-bearer-secret"])
    calls: list[str] = []

    def fake_mint(_config, _auth, **_kwargs):
        return next(mint_tokens)

    def fake_urlopen(request, timeout):
        del timeout
        calls.append(request.get_header("Authorization"))
        api_name = parse_qs(urlsplit(request.full_url).query)["api"][0]
        if api_name == loop.ALIBABA_TOKEN_PLAN_SUBSCRIPTION_API:
            payload = _alibaba_subscription_payload()
        elif api_name == loop.ALIBABA_TOKEN_PLAN_RESET_CARD_LIST_API:
            payload = []
        else:
            payload = _alibaba_usage_payload()
        return _alibaba_http_success(_alibaba_console_envelope(payload))

    monkeypatch.setattr(loop, "_mint_alibaba_console_access_token", fake_mint)
    monkeypatch.setattr(loop, "ALIBABA_QUOTA_HTTP_OPEN_FN", fake_urlopen)
    state = loop.SidecarTaskState()
    first = loop.run_due_sidecar_tasks(config, state, now_monotonic=100.0)
    monkeypatch.setenv("ALIBABA_RAM_KEY", "LTAI5tRotatedAccessKeyId")
    second = loop.run_due_sidecar_tasks(config, state, now_monotonic=401.0)

    assert calls[:3] == ["Bearer first-bearer-secret"] * 3
    assert calls[3:] == ["Bearer second-bearer-secret"] * 3
    assert first[0]["credential_reset"] is False
    assert second[0]["credential_reset"] is True
    assert second[0]["subscription_refreshed"] is True
    assert state.alibaba_access_token == "second-bearer-secret"
    serialized = json.dumps(first + second)
    _assert_no_alibaba_secrets(serialized, extra=("first-bearer-secret", "second-bearer-secret"))


def test_run_due_sidecar_tasks_redacts_alibaba_failure(monkeypatch) -> None:
    config = _alibaba_quota_poll_config()
    _set_alibaba_ram_env(monkeypatch)

    def fake_fetch(_config, *, api_name, endpoint, auth, session, task_state=None, **kwargs):
        del api_name, endpoint, auth, session, task_state, kwargs
        raise ValueError(
            "Authorization=Bearer console-bearer-secret "
            "ALIBABA_RAM_SECRET=testAccessKeySecretValue "
            "password=password-secret mfa=mfa-secret"
        )

    monkeypatch.setattr(loop, "_fetch_alibaba_quota_payload", fake_fetch)
    events = _alibaba_poll_events(
        loop.run_due_sidecar_tasks(
            config,
            loop.SidecarTaskState(),
            now_monotonic=100.0,
        )
    )
    event_json = json.dumps(events)
    assert events[0]["event"] == "alibaba_quota_poll"
    assert events[0]["telemetry_status"] == "degraded"
    assert "REDACTED" in events[0]["error_message"]
    _assert_no_alibaba_secrets(event_json, extra=("password-secret", "mfa-secret"))


def test_run_due_sidecar_tasks_reports_missing_alibaba_auth_without_traceback(
    monkeypatch,
) -> None:
    _clear_alibaba_ram_env(monkeypatch)

    events = _alibaba_poll_events(
        loop.run_due_sidecar_tasks(
            _alibaba_quota_poll_config(),
            loop.SidecarTaskState(),
            now_monotonic=100.0,
        )
    )

    assert events[0]["event"] == "alibaba_quota_poll"
    assert events[0]["telemetry_status"] == "degraded"
    assert events[0]["telemetry_class"] == "auth"
    assert events[0]["error_endpoint"] == "authentication"
    assert events[0]["error_class"] == "AlibabaAuthError"
    assert events[0]["error_message"] == "Alibaba RAM credentials are incomplete."
    assert events[0]["last_good_state_retained"] is True
    assert "traceback" not in events[0]


def test_run_due_sidecar_tasks_clears_stale_subscription_on_refresh_failure(
    monkeypatch,
) -> None:
    config = _alibaba_quota_poll_config()
    _set_alibaba_ram_env(monkeypatch)
    state = loop.SidecarTaskState(
        alibaba_quota_last_attempt_monotonic=100.0,
        alibaba_subscription_last_attempt_monotonic=100.0,
        alibaba_access_token="last-good-bearer-secret",
        alibaba_subscription_payload=loop._parse_alibaba_subscription_payload(
            _alibaba_subscription_payload()
        ),
    )

    def fake_fetch(_config, *, api_name, endpoint, auth, session, task_state=None, **kwargs):
        del api_name, endpoint, auth, session, task_state, kwargs
        raise loop.AlibabaQuotaPollError(
            "Alibaba Token Plan subscription poll failed with HTTP 503.",
            status_code=503,
            telemetry_class="upstream",
            attempt_count=2,
            retry_count=1,
            endpoint="subscription",
        )

    monkeypatch.setattr(loop, "_fetch_alibaba_quota_payload", fake_fetch)
    events = _alibaba_poll_events(
        loop.run_due_sidecar_tasks(config, state, now_monotonic=21700.0)
    )

    assert events[0]["telemetry_status"] == "degraded"
    assert events[0]["last_good_state_retained"] is True
    assert state.alibaba_subscription_payload is None
    assert state.alibaba_access_token == "last-good-bearer-secret"


def test_alibaba_empty_reset_card_list_synthesizes_lifecycle(monkeypatch) -> None:
    config = _alibaba_quota_poll_config()
    observed_at = datetime(2026, 8, 11, 12, 0, tzinfo=timezone.utc)
    subscription = loop._parse_alibaba_subscription_payload(_alibaba_subscription_payload())
    visible_rows, available_count = loop._build_alibaba_reset_card_observations(
        config,
        observed_at=observed_at,
        reset_cards=_alibaba_reset_card_payload(),
        subscription=subscription,
        status_code=200,
        attempt_count=1,
        retry_count=0,
    )
    monkeypatch.setattr(
        probes,
        "load_provider_credit_current_rows",
        lambda *_args, **_kwargs: [
            dict(visible_rows[0], status="available"),
            dict(visible_rows[1], status="available"),
            dict(visible_rows[0], credit_identity="already-used", status="used"),
        ],
    )
    empty_rows, empty_count = loop._build_alibaba_reset_card_observations(
        config,
        observed_at=observed_at,
        reset_cards=[],
        subscription=subscription,
        status_code=200,
        attempt_count=1,
        retry_count=0,
    )
    lifecycle_rows = loop._synthesize_alibaba_reset_card_lifecycle_observations(
        config,
        observed_at=observed_at,
        account_hash=subscription["account_hash"],
        visible_identities=set(),
        visible_card_count=0,
        available_count=empty_count,
        status_code=200,
        attempt_count=1,
        retry_count=0,
    )
    assert empty_rows == []
    assert empty_count == 0
    assert available_count == 1
    assert {
        row["status"]: row["evidence"]["lifecycle_reason"]
        for row in lifecycle_rows
    } == {
        "used": "card_missing_before_expiry",
        "expired": "card_past_expiry",
    }


def test_alibaba_poll_never_invokes_bl_or_china_hosts(monkeypatch) -> None:
    invoked = []
    loop_src = (
        Path(__file__).resolve().parents[2]
        / "scripts/run_provider_status_observations_loop.py"
    ).read_text(encoding="utf-8")

    def fake_run(*args, **kwargs):
        invoked.append((args, kwargs))
        raise AssertionError("bl must not be invoked")

    monkeypatch.setattr(loop, "ALIBABA_MINT_HTTP_POST_FN", lambda **_kwargs: pytest.fail("live mint"))
    monkeypatch.setattr(loop, "ALIBABA_QUOTA_HTTP_OPEN_FN", lambda *_args, **_kwargs: pytest.fail("live gateway"))
    monkeypatch.setattr("subprocess.run", fake_run, raising=False)
    config = _alibaba_quota_poll_config()
    request = loop._build_alibaba_quota_request(
        config,
        api_name=loop.ALIBABA_TOKEN_PLAN_USAGE_API,
        access_token="console-bearer-secret",
    )
    assert invoked == []
    assert "import subprocess" not in loop_src
    assert '["bl"]' not in loop_src
    assert "'bl'" not in loop_src
    assert "/data/api.json" not in loop_src
    assert "login_ticket" not in loop_src
    assert "sec_token" not in loop_src
    assert "AlibabaWebSession" not in loop_src
    assert urlsplit(request.full_url).hostname == "bailian-singapore-cs.alibabacloud.com"
    assert not loop._alibaba_host_is_china(urlsplit(request.full_url).hostname)
    assert not loop._alibaba_host_is_china(loop.ALIBABA_TOKEN_PLAN_MINT_HOST)


def test_persist_alibaba_quota_observations_uses_sidecar_db_path(
    monkeypatch,
) -> None:
    config = _alibaba_quota_poll_config(
        db_lock_timeout_ms=123,
        db_statement_timeout_ms=456,
    )
    subscription = loop._parse_alibaba_subscription_payload(_alibaba_subscription_payload())
    payloads = loop._build_alibaba_quota_rate_limit_payloads(
        config,
        observed_at=datetime(2026, 7, 21, 18, 0, tzinfo=timezone.utc),
        usage_payload=_alibaba_usage_payload(),
        subscription=subscription,
    )
    fake_conn = _FakeProviderStatusConnection()
    monkeypatch.setattr(loop.probes.psycopg, "connect", lambda _dsn: fake_conn)

    inserted_count = loop._persist_alibaba_quota_observations(config, payloads)

    assert inserted_count == 12
    assert fake_conn.cursor_instance.execute_calls[:3] == [
        (
            "SELECT set_config('application_name', %s, false)",
            ("aawm-provider-status-observations-alibaba-quota",),
        ),
        ("SELECT set_config('lock_timeout', %s, true)", ("123ms",)),
        ("SELECT set_config('statement_timeout', %s, true)", ("456ms",)),
    ]
    assert [params[6] for _statement, params in fake_conn.cursor_instance.execute_calls[3:]] == [
        loop.ALIBABA_TOKEN_PLAN_5H_QUOTA_KEY,
        loop.ALIBABA_TOKEN_PLAN_5H_QUOTA_KEY,
        loop.ALIBABA_TOKEN_PLAN_5H_QUOTA_KEY,
        loop.ALIBABA_TOKEN_PLAN_5H_QUOTA_KEY,
        loop.ALIBABA_TOKEN_PLAN_5H_QUOTA_KEY,
        loop.ALIBABA_TOKEN_PLAN_5H_QUOTA_KEY,
        loop.ALIBABA_TOKEN_PLAN_7D_QUOTA_KEY,
        loop.ALIBABA_TOKEN_PLAN_7D_QUOTA_KEY,
        loop.ALIBABA_TOKEN_PLAN_7D_QUOTA_KEY,
        loop.ALIBABA_TOKEN_PLAN_7D_QUOTA_KEY,
        loop.ALIBABA_TOKEN_PLAN_7D_QUOTA_KEY,
        loop.ALIBABA_TOKEN_PLAN_7D_QUOTA_KEY,
    ]


def test_compose_wires_alibaba_quota_poll_defaults() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    compose_text = (repo_root / "docker-compose.dev.yml").read_text()
    start = compose_text.index("\n  provider-status-observations:\n")
    end = compose_text.index("\nnetworks:", start)
    sidecar = compose_text[start:end]

    assert "/home/zepfu/.alibaba" not in sidecar
    assert "AAWM_ALIBABA_WEB_AUTH_FILE" not in sidecar
    assert "ALIBABA_WEB_KEY" not in sidecar
    assert "ALIBABA_RAM_KEY=${ALIBABA_RAM_KEY:-}" in sidecar
    assert "ALIBABA_RAM_SECRET=${ALIBABA_RAM_SECRET:-}" in sidecar
    assert "ALIBABA_RAM_PRINCIPAL=${ALIBABA_RAM_PRINCIPAL:-}" in sidecar
    assert (
        "AAWM_ALIBABA_QUOTA_GATEWAY_URL=${AAWM_ALIBABA_QUOTA_GATEWAY_URL:-"
        "https://bailian-singapore-cs.alibabacloud.com/cli/api.json}"
        in sidecar
    )
    assert "AAWM_ALIBABA_QUOTA_POLL_INTERVAL_SECONDS=${AAWM_ALIBABA_QUOTA_POLL_INTERVAL_SECONDS:-300}" in sidecar
    assert (
        "AAWM_ALIBABA_SUBSCRIPTION_POLL_INTERVAL_SECONDS=${AAWM_ALIBABA_SUBSCRIPTION_POLL_INTERVAL_SECONDS:-21600}"
        in sidecar
    )
    assert "AAWM_ALIBABA_QUOTA_POLL_MAX_ATTEMPTS=${AAWM_ALIBABA_QUOTA_POLL_MAX_ATTEMPTS:-2}" in sidecar
    assert (
        "AAWM_ALIBABA_QUOTA_POLL_RETRY_BACKOFF_SECONDS=${AAWM_ALIBABA_QUOTA_POLL_RETRY_BACKOFF_SECONDS:-0.5}"
        in sidecar
    )
    assert "AAWM_ALIBABA_QUOTA_POLL_ENABLED=${AAWM_ALIBABA_QUOTA_POLL_ENABLED:-0}" in sidecar
    assert "AAWM_ALIBABA_QUOTA_POLL_ENABLED=${AAWM_ALIBABA_QUOTA_POLL_ENABLED:-1}" not in sidecar


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
    assert config.grok_billing_client_version is None
    assert config.grok_billing_client_version_source is None
    headers = loop._build_grok_billing_request_headers(
        config,
        access_token="access-token-secret",
    )
    assert headers["x-grok-client-version"] == "0.2.60"
    assert (
        headers.version_resolution.source
        == "AAWM_GROK_BILLING_CLIENT_VERSION"
    )
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
            or _grok_billing_fetch_result(config)
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
        request_headers=_grok_billing_request_headers(config),
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
    assert summary["user_agent"] == "grok/0.2.55"
    assert summary["client_version_source"] == "config"
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
        lambda *_args, **_kwargs: _grok_billing_fetch_result(config),
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
        lambda *_args, **_kwargs: _grok_billing_fetch_result(config),
    )

    def fake_persist(cfg, *, observed_at, response_body, request_headers=None):
        captured["config"] = cfg
        captured["observed_at"] = observed_at
        captured["response_body"] = response_body
        captured["request_headers"] = request_headers
        return 1, 1

    monkeypatch.setattr(loop, "_persist_grok_billing_observations", fake_persist)

    events = loop.run_due_sidecar_tasks(
        config,
        loop.SidecarTaskState(),
        now_monotonic=100.0,
    )

    assert captured["config"] is config
    assert captured["response_body"] == _grok_billing_payload()
    assert captured["request_headers"]["x-userid"] == "user_123"
    assert captured["request_headers"]["x-grok-client-version"] == "0.2.55"
    assert events[0]["event"] == "grok_billing_poll"
    assert events[0]["persisted"] is True
    assert events[0]["observation_count"] == 1
    assert events[0]["inserted_count"] == 1
    assert events[0]["status_code"] == 200
    assert "access-token" not in json.dumps(events)


def test_grok_billing_reuses_request_headers_for_event_and_persistence(
    monkeypatch,
) -> None:
    config = _grok_billing_poll_config(
        grok_billing_client_version="9.10.11",
    )
    captured = {}

    class CapturedRequest:
        def __init__(self, url, *, headers, method):
            captured["request_headers"] = headers
            self.url = url
            self.method = method

        def get_method(self):
            return self.method

    def fake_urlopen(request, timeout):
        assert request is not None
        assert timeout == 30.0
        return type(
            "Resp",
            (),
            {
                "status": 200,
                "getcode": lambda self: 200,
                "read": lambda self: json.dumps(
                    _grok_billing_payload()
                ).encode(),
                "__enter__": lambda self: self,
                "__exit__": lambda self, *args: None,
            },
        )()

    def fake_persist(
        cfg,
        *,
        observed_at,
        response_body,
        request_headers,
    ):
        assert request_headers is captured["request_headers"]
        payload = loop._build_grok_billing_rate_limit_payload(
            cfg,
            observed_at=observed_at,
            response_body=response_body,
            request_headers=request_headers,
        )
        captured["persisted_payload"] = payload
        return 1, 1

    monkeypatch.setattr(
        loop,
        "_load_grok_billing_auth_context",
        lambda _path: _grok_billing_auth_context(),
    )
    monkeypatch.setattr(loop.urllib_request, "Request", CapturedRequest)
    monkeypatch.setattr(loop.urllib_request, "urlopen", fake_urlopen)
    monkeypatch.setattr(
        loop,
        "_persist_grok_billing_observations",
        fake_persist,
    )

    event = loop.run_due_sidecar_tasks(
        config,
        loop.SidecarTaskState(),
        now_monotonic=100.0,
    )[0]
    evidence = json.loads(captured["persisted_payload"][17])

    assert event["client_version"] == "9.10.11"
    assert event["user_agent"] == "grok/9.10.11"
    assert event["client_version_source"] == "config"
    assert captured["persisted_payload"][2] == "9.10.11"
    assert evidence["request_contract_client_version"] == "9.10.11"
    assert evidence["request_contract_user_agent"] == "grok/9.10.11"
    assert evidence["request_contract_client_version_source"] == "config"


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
        "user_agent",
        "client_version_source",
        "client_version_cache_source",
        "client_version_cache_path_class",
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

    payload = loop._build_grok_billing_rate_limit_payload(
        config,
        observed_at=observed_at,
        response_body=_grok_billing_payload(),
        request_headers=_grok_billing_request_headers(config),
    )

    assert payload[0] == observed_at
    assert payload[1] == "grok-build"
    assert payload[2] == "0.2.55"
    assert payload[3] is None
    assert payload[4] == "xai"
    assert payload[5] == "grok-composer-2.5-fast"
    assert payload[6] == "xai_grok_build_weekly_credits:credits"
    assert payload[7] == "weekly"
    assert payload[8] == "credits"
    assert payload[9] == datetime(2026, 7, 10, 19, 54, 47, 584112, tzinfo=timezone.utc)
    assert payload[10] == pytest.approx(85.460667)
    assert payload[6] != "xai_grok_build_monthly_credits:credits"
    assert payload[11] is None
    assert payload[12] is None
    assert payload[13] is None
    assert payload[14] == datetime(2026, 7, 3, 19, 54, 47, 584112, tzinfo=timezone.utc)
    assert payload[15] == datetime(2026, 7, 10, 19, 54, 47, 584112, tzinfo=timezone.utc)
    raw_provider_fields = json.loads(payload[16])
    assert raw_provider_fields["creditUsagePercent"] == pytest.approx(14.539333)
    assert raw_provider_fields["productUsage"][0]["name"] == "GrokBuild"
    assert raw_provider_fields["quota_unit"] == "grok_billing_credit_usage_percent"
    evidence = json.loads(payload[17])
    assert evidence["signals"] == [
        "grok_billing_payload",
        "grok_billing_weekly_credit",
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
    assert evidence["request_contract_client_version"] == "0.2.55"
    assert evidence["request_contract_user_agent"] == "grok/0.2.55"
    assert evidence["request_contract_client_version_source"] == "config"
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


def test_grok_billing_sidecar_payload_maps_weekly_fresh_period_snapshot() -> None:
    config = _grok_billing_poll_config()
    observed_at = datetime(2026, 7, 3, 20, 0, tzinfo=timezone.utc)
    response_body = _grok_billing_weekly_fresh_payload()

    payload = loop._build_grok_billing_rate_limit_payload(
        config,
        observed_at=observed_at,
        response_body=response_body,
        request_headers=_grok_billing_request_headers(config),
    )

    assert payload[6] == "xai_grok_build_weekly_credits:credits"
    assert payload[7] == "weekly"
    assert payload[8] == "credits"
    assert payload[9] == datetime(2026, 7, 10, 19, 54, 47, 584112, tzinfo=timezone.utc)
    assert payload[10] == 100.0
    assert payload[6] != "xai_grok_build_monthly_credits:credits"
    assert payload[11] is None
    assert payload[12] is None
    assert payload[13] is None
    raw_provider_fields = json.loads(payload[16])
    assert (
        raw_provider_fields["quota_unit"] == "grok_billing_weekly_credit_fresh_period"
    )
    assert "creditUsagePercent" not in raw_provider_fields
    evidence = json.loads(payload[17])
    assert "grok_billing_weekly_fresh_period" in evidence["signals"]
    assert evidence["unit_note"].startswith("Fresh weekly Grok Build credit periods")


def test_grok_billing_sidecar_payload_maps_monthly_counter_snapshot() -> None:
    config = _grok_billing_poll_config()
    observed_at = datetime(2026, 7, 3, 12, 0, tzinfo=timezone.utc)
    response_body = _grok_billing_monthly_counter_payload()

    payload = loop._build_grok_billing_rate_limit_payload(
        config,
        observed_at=observed_at,
        response_body=response_body,
        request_headers=_grok_billing_request_headers(config),
    )

    assert payload[6] == "xai_grok_build_monthly_requests:requests"
    assert payload[7] == "monthly"
    assert payload[8] == "requests"
    assert payload[10] == 71.0
    assert payload[11] == pytest.approx(150000.0)
    assert payload[12] == pytest.approx(42910.0)
    evidence = json.loads(payload[17])
    assert "grok_billing_monthly_counter" in evidence["signals"]


def test_grok_billing_sidecar_payload_keeps_legacy_credit_snapshot_monthly() -> None:
    config = _grok_billing_poll_config()
    observed_at = datetime(2026, 7, 3, 16, 45, tzinfo=timezone.utc)

    payload = loop._build_grok_billing_rate_limit_payload(
        config,
        observed_at=observed_at,
        response_body=_grok_billing_legacy_monthly_credit_payload(),
        request_headers=_grok_billing_request_headers(config),
    )

    assert payload[6] == "xai_grok_build_monthly_credits:credits"
    assert payload[7] == "monthly"
    assert payload[8] == "credits"
    assert payload[9] == datetime(2026, 8, 1, tzinfo=timezone.utc)
    assert payload[10] == pytest.approx(73.0)
    assert payload[11] is None
    assert payload[12] is None
    assert payload[13] is None
    evidence = json.loads(payload[17])
    assert "grok_billing_legacy_monthly_credit" in evidence["signals"]


def test_grok_billing_sidecar_payload_raises_without_usage_or_period() -> None:
    config = _grok_billing_poll_config()
    observed_at = datetime(2026, 6, 30, 12, 0, tzinfo=timezone.utc)

    with pytest.raises(ValueError, match="absolute or percentage quota fields"):
        loop._build_grok_billing_rate_limit_payload(
            config,
            observed_at=observed_at,
            response_body={"config": {}},
            request_headers=_grok_billing_request_headers(config),
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
        lambda *_args, **_kwargs: _grok_billing_fetch_result(
            config,
            payload=period_only_payload,
        ),
    )

    def fake_persist(cfg, *, observed_at, response_body, request_headers=None):
        captured["response_body"] = response_body
        captured["request_headers"] = request_headers
        return 1, 1

    monkeypatch.setattr(loop, "_persist_grok_billing_observations", fake_persist)

    events = loop.run_due_sidecar_tasks(
        config,
        loop.SidecarTaskState(),
        now_monotonic=100.0,
    )

    assert captured["response_body"] == period_only_payload
    assert captured["request_headers"]["x-grok-client-version"] == "0.2.55"
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
        request_headers=_grok_billing_request_headers(config),
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
    assert insert_payload[6] == "xai_grok_build_weekly_credits:credits"
    assert insert_payload[11] is None
    assert insert_payload[12] is None
    assert insert_payload[13] is None
    evidence = json.loads(insert_payload[17])
    assert evidence["request_contract_source"] == "grok_billing_sidecar_poll"
    assert len(evidence["request_contract_fingerprint"]) == 64


def test_grok_billing_persistence_records_sanitized_cache_source(
    tmp_path,
    monkeypatch,
) -> None:
    _clear_grok_billing_version_env(monkeypatch)
    cache_path = tmp_path / "native-client-version.json"
    _write_grok_billing_version_cache(cache_path, version="10.11.12")
    monkeypatch.setenv(
        "AAWM_GROK_CLIENT_VERSION_CACHE_PATH",
        str(cache_path),
    )
    config = _grok_billing_poll_config(
        grok_billing_client_version=None,
        grok_billing_client_version_source=None,
    )
    request_headers = loop._build_grok_billing_request_headers(
        config,
        access_token="access-token-secret",
    )

    payload = loop._build_grok_billing_rate_limit_payload(
        config,
        observed_at=datetime(2026, 7, 28, 12, 0, tzinfo=timezone.utc),
        response_body=_grok_billing_payload(),
        request_headers=request_headers,
    )
    evidence = json.loads(payload[17])
    encoded_evidence = json.dumps(evidence)

    assert payload[2] == "10.11.12"
    assert evidence["request_contract_client_version"] == "10.11.12"
    assert evidence["request_contract_user_agent"] == "grok/10.11.12"
    assert evidence["request_contract_client_version_source"] == "cache"
    assert (
        evidence["request_contract_client_version_cache_source"]
        == "installed-grok-cli"
    )
    assert (
        evidence["request_contract_client_version_cache_path_class"]
        == "configured"
    )
    assert str(cache_path) not in encoded_evidence
    assert "access-token-secret" not in encoded_evidence


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


def test_fetch_grok_billing_payload_does_not_use_managed_xai_oauth(
    monkeypatch,
) -> None:
    native_auth_file = "/native-grok/auth.json"
    managed_auth_file = "/managed-xai/auth.json"
    config = _grok_billing_poll_config(
        grok_oidc_auth_file=native_auth_file,
        xai_oauth_auth_file=managed_auth_file,
    )
    loaded_auth_files = []

    def fake_load_grok_auth(auth_file):
        loaded_auth_files.append(auth_file)
        return _grok_billing_auth_context()

    def managed_auth_read_must_not_run(*_args, **_kwargs):
        raise AssertionError("managed oa_xai OAuth must not serve native billing")

    def fake_urlopen(_request, timeout):
        assert timeout == 30.0
        return type(
            "Resp",
            (),
            {
                "status": 200,
                "getcode": lambda self: 200,
                "read": lambda self: json.dumps(
                    _grok_billing_payload()
                ).encode(),
                "__enter__": lambda self: self,
                "__exit__": lambda self, *args: None,
            },
        )()

    monkeypatch.setattr(
        loop,
        "_load_grok_billing_auth_context",
        fake_load_grok_auth,
    )
    monkeypatch.setattr(
        loop.xai_oauth_refresh,
        "_read_credential_payload",
        managed_auth_read_must_not_run,
    )
    monkeypatch.setattr(loop.urllib_request, "urlopen", fake_urlopen)

    fetched = loop._fetch_grok_billing_payload(config)

    assert fetched["status_code"] == 200
    assert loaded_auth_files == [native_auth_file]
    assert managed_auth_file not in loaded_auth_files


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
        ("SELECT set_config('statement_timeout', %s, true)", ("15000ms",)),
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
    assert "metadata->>'session_history_repository_status', '') <> 'unresolved'" in sql
    assert "metadata->>'session_history_repository_unresolved', 'false') <> 'true'" in sql


def test_observability_git_anomalies_use_parsed_tool_activity_counts() -> None:
    sql = loop.OBSERVABILITY_SESSION_HISTORY_ANOMALY_SQL
    commit_branch = sql.split(
        "'git_commit_activity_not_reflected' AS anomaly_class", 1
    )[1].split("UNION ALL", 1)[0]
    push_branch = sql.split(
        "'git_push_activity_not_reflected' AS anomaly_class", 1
    )[1].split("),\nranked AS", 1)[0]

    assert "'activity_git_commit_command', activity_git_commit_command" in commit_branch
    assert "'activity_git_push_command', activity_git_push_command" in push_branch
    assert "AND activity_git_commit_count > 0" in commit_branch
    assert "AND activity_git_push_count > 0" in push_branch
    assert "OR activity_git_commit_command" not in commit_branch
    assert "OR activity_git_push_command" not in push_branch


def test_observability_rate_limit_anomaly_sql_filters_recent_unscoped_observations() -> None:
    sql = loop.OBSERVABILITY_RATE_LIMIT_ANOMALY_SQL
    assert "observed_at >=" in sql
    assert "account_hash IS NULL" in sql
    assert sql.index("observed_at >=") < sql.index("recent_traffic AS")


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
    expired_access_token = _build_test_jwt({"exp": int(time.time()) - 60})
    auth_path.write_text(
        json.dumps(
            {
                "tokens": {
                    "access_token": expired_access_token,
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
    # Shared sanitizer redacts secret values (access_token=...), not field
    # labels in explanatory messages such as "has no refresh_token".
    err = summary["error_message"] or ""
    assert expired_access_token not in err
    assert "does not contain a refresh_token" in err


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
    # Shared sanitizer redacts secret *values* (access_token=...), not field
    # labels in explanatory messages such as "has no refresh_token".
    err = summary["error_message"] or ""
    assert "expired-xai-token" not in err
    assert "has no refresh_token" in err


@pytest.mark.parametrize("provider_name", ["grok", "codex", "xai", "kimi"])
def test_passive_oauth_health_inspectors_are_read_only_and_offline(
    provider_name,
    tmp_path,
    monkeypatch,
) -> None:
    auth_path = tmp_path / f"{provider_name}-auth.json"
    future_timestamp = time.time() + 3600
    future_iso = (
        datetime.fromtimestamp(future_timestamp, tz=timezone.utc)
        .isoformat()
        .replace("+00:00", "Z")
    )
    if provider_name == "grok":
        module = grok_oidc_refresh
        inspector = module.inspect_grok_oidc_credential_health
        payload = {"access_token": "grok-secret", "expires_at": future_iso}
        inspector_kwargs = {}
    elif provider_name == "codex":
        module = codex_oauth_refresh
        inspector = module.inspect_codex_oauth_credential_health
        payload = {
            "tokens": {
                "access_token": _build_test_jwt({"exp": int(future_timestamp)}),
                "account_id": "acct_test",
            }
        }
        inspector_kwargs = {}
    elif provider_name == "xai":
        module = xai_oauth_refresh
        inspector = module.inspect_xai_oauth_credential_health
        payload = {"access_token": "xai-secret", "expires_at": future_iso}
        inspector_kwargs = {}
    else:
        module = kimi_oauth_refresh
        inspector = module.inspect_kimi_oauth_credential_health
        payload = {
            "access_token": "kimi-secret",
            "expires_at": future_timestamp,
            "scope": "kimi-code",
        }
        inspector_kwargs = {}

    auth_path.write_text(json.dumps(payload), encoding="utf-8")
    auth_path.chmod(0o600)
    before_content = auth_path.read_bytes()
    before_stat = auth_path.stat()
    before_children = sorted(path.name for path in tmp_path.iterdir())

    def fail_network(*_args, **_kwargs):
        raise AssertionError("passive credential inspection must not use the network")

    monkeypatch.setattr(module.urllib_request, "urlopen", fail_network)

    summary = inspector(auth_path, **inspector_kwargs)

    after_stat = auth_path.stat()
    assert summary["health_status"] == "fresh"
    assert summary["refreshed"] is False
    assert auth_path.read_bytes() == before_content
    assert after_stat.st_mtime_ns == before_stat.st_mtime_ns
    assert after_stat.st_mode == before_stat.st_mode
    assert sorted(path.name for path in tmp_path.iterdir()) == before_children


def test_passive_oauth_health_inspection_classifies_expired_and_malformed(
    tmp_path,
) -> None:
    expired_path = tmp_path / "expired-codex.json"
    expired_path.write_text(
        json.dumps(
            {
                "tokens": {
                    "access_token": _build_test_jwt(
                        {"exp": int(time.time()) - 60}
                    )
                }
            }
        ),
        encoding="utf-8",
    )
    malformed_path = tmp_path / "malformed-codex.json"
    malformed_path.write_text("{not-json", encoding="utf-8")

    expired = codex_oauth_refresh.inspect_codex_oauth_credential_health(
        expired_path
    )
    malformed = codex_oauth_refresh.inspect_codex_oauth_credential_health(
        malformed_path
    )

    assert expired["health_status"] == "expired"
    assert expired["error_class"] == "CredentialExpiredError"
    assert malformed["health_status"] == "malformed"
    assert malformed["error_class"] == "ValueError"


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


def test_passive_auth_health_poll_persists_four_sanitized_rows_on_cadence(
    monkeypatch,
) -> None:
    config = _grok_oidc_auth_persist_config(
        grok_oidc_refresh_enabled=False,
        codex_oauth_refresh_enabled=False,
        xai_oauth_refresh_enabled=False,
        kimi_oauth_refresh_enabled=False,
        provider_auth_health_poll_enabled=True,
        provider_auth_health_poll_interval_seconds=3600.0,
    )
    state = loop.SidecarTaskState()
    persisted = []
    future = "2026-07-22T18:00:00Z"

    monkeypatch.setattr(
        loop.grok_oidc_refresh,
        "inspect_grok_oidc_credential_health",
        lambda *_args, **_kwargs: {
            "attempted": True,
            "refreshed": False,
            "skipped": False,
            "auth_file": config.grok_oidc_auth_file,
            "scope": grok_oidc_refresh.DEFAULT_GROK_OIDC_SCOPE,
            "health_status": "fresh",
            "expires_at": future,
            "error_class": None,
            "error_message": None,
        },
    )
    monkeypatch.setattr(
        loop.codex_oauth_refresh,
        "inspect_codex_oauth_credential_health",
        lambda *_args, **_kwargs: {
            "attempted": True,
            "refreshed": False,
            "skipped": False,
            "auth_file": config.codex_auth_file,
            "account_id": "acct_test",
            "health_status": "fresh",
            "expires_at": future,
            "error_class": None,
            "error_message": None,
        },
    )
    monkeypatch.setattr(
        loop.xai_oauth_refresh,
        "inspect_xai_oauth_credential_health",
        lambda *_args, **_kwargs: {
            "attempted": True,
            "refreshed": False,
            "skipped": False,
            "auth_file": config.xai_oauth_auth_file,
            "scope": config.xai_oauth_scope,
            "health_status": "fresh",
            "expires_at": future,
            "error_class": None,
            "error_message": None,
        },
    )
    monkeypatch.setattr(
        loop.kimi_oauth_refresh,
        "inspect_kimi_oauth_credential_health",
        lambda *_args, **_kwargs: {
            "attempted": True,
            "refreshed": False,
            "skipped": False,
            "auth_file": config.kimi_oauth_auth_file,
            "scope": kimi_oauth_refresh.DEFAULT_KIMI_OAUTH_SCOPE,
            "health_status": "fresh",
            "expires_at": future,
            "error_class": None,
            "error_message": None,
        },
    )

    def fake_persist(_config, observation):
        persisted.append(dict(observation))
        return True, 1, None, None

    monkeypatch.setattr(
        loop,
        "_persist_passive_provider_auth_observation",
        fake_persist,
    )

    first_events = loop.run_due_sidecar_tasks(config, state, now_monotonic=100.0)
    early_events = loop.run_due_sidecar_tasks(config, state, now_monotonic=200.0)
    due_events = loop.run_due_sidecar_tasks(config, state, now_monotonic=3700.0)

    assert len(first_events) == 4
    assert early_events == []
    assert len(due_events) == 4
    assert len(persisted) == 8
    assert {row["auth_family"] for row in persisted} == {
        "grok_oidc",
        "codex_oauth",
        "xai_oauth",
        "kimi_oauth",
    }
    assert all(row["source_task"] == "provider_auth_health_poll" for row in persisted)
    assert all(row["metadata"]["passive_read_only"] is True for row in persisted)
    assert all(row["metadata"]["network_calls"] is False for row in persisted)
    assert all(
        row["metadata"]["credential_file_mutated"] is False for row in persisted
    )
    assert all("auth_file" not in event for event in first_events + due_events)


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
        codex_oauth_inventory=_codex_oauth_inventory("account1"),
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
    record = _codex_oauth_inventory("account1").records[0]
    event = {
        "event": "codex_oauth_refresh",
        "observed_at": "2026-06-19T12:00:00Z",
        "environment": "dev",
        "attempted": True,
        "refreshed": True,
        "skipped": False,
        "account_label": record.label,
        "account_hash": record.expected_account_hash,
        "expires_at": "2026-06-19T13:00:00Z",
        "error_class": None,
        "error_message": None,
    }
    event.update(overrides)
    return event


def test_build_codex_oauth_auth_observation_maps_successful_refresh() -> None:
    config = _codex_oauth_auth_persist_config()
    record = config.codex_oauth_inventory.records[0]
    event = _codex_oauth_refresh_sidecar_event()

    observation = loop._build_codex_auth_observation(
        config,
        event,
        record=record,
    )

    assert observation["observed_at"] == datetime(2026, 6, 19, 12, 0, tzinfo=timezone.utc)
    assert observation["environment"] == "dev"
    assert observation["provider"] == "openai"
    assert observation["auth_family"] == "codex_oauth"
    assert observation["credential_scope"] == "account1"
    assert observation["auth_file_hash"] == hashlib.sha256(
        str(record.auth_path).encode("utf-8")
    ).hexdigest()
    assert observation["status"] == "refreshed"
    assert observation["attempted"] is True
    assert observation["refreshed"] is True
    assert observation["skipped"] is False
    assert observation["expires_at"] == datetime(2026, 6, 19, 13, 0, tzinfo=timezone.utc)
    assert observation["last_success_at"] == observation["observed_at"]
    assert observation["source_task"] == "codex_oauth_refresh"
    assert observation["metadata"]["auth_file_source"] == "codex_oauth_inventory"
    assert observation["metadata"]["account_label"] == "account1"
    assert observation["metadata"]["account_hash"] == record.expected_account_hash
    observation_json = json.dumps(observation, default=str)
    assert "refresh_token" not in observation_json
    assert "access_token" not in observation_json
    assert "/home/zepfu/.codex/auth.json" not in observation_json


def test_build_codex_oauth_auth_observation_sanitizes_refresh_failure() -> None:
    config = _codex_oauth_auth_persist_config()
    record = config.codex_oauth_inventory.records[0]
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

    observation = loop._build_codex_auth_observation(
        config,
        event,
        record=record,
    )

    assert observation["status"] == "failed"
    assert observation["last_success_at"] is None
    assert observation["error_class"] == "ValueError"
    assert "REDACTED" in observation["error_message"]
    assert "super-secret" not in json.dumps(observation, default=str)
    assert "leaked-token" not in json.dumps(observation, default=str)


def test_run_due_sidecar_tasks_persists_codex_auth_observation_when_apply_enabled(
    monkeypatch,
    tmp_path,
) -> None:
    wall_now = datetime(2026, 6, 19, 12, 0, tzinfo=timezone.utc)
    inventory = _codex_oauth_inventory("account1", root=tmp_path)
    record = inventory.records[0]
    record.auth_path.write_text(
        json.dumps(
            {
                "tokens": {
                    "access_token": _build_test_jwt(
                        {"exp": int((wall_now + timedelta(minutes=30)).timestamp())}
                    ),
                    "refresh_token": "refresh-account1",
                    "account_id": "acct-account1",
                }
            }
        ),
        encoding="utf-8",
    )
    record.auth_path.chmod(0o600)
    config = _codex_oauth_auth_persist_config(
        codex_force_refresh=True,
        codex_refresh_interval_seconds=3600.0,
        codex_oauth_inventory=inventory,
    )
    captured = {}

    def fake_refresh(selected_record, **kwargs):
        kwargs["on_token_endpoint_attempt"]()
        selected_record.auth_path.write_text(
            json.dumps(
                {
                    "tokens": {
                        "access_token": _build_test_jwt(
                            {"exp": int((wall_now + timedelta(hours=1)).timestamp())}
                        ),
                        "refresh_token": "rotated-refresh-account1",
                        "account_id": "acct-account1",
                    }
                }
            ),
            encoding="utf-8",
        )
        return {
            "attempted": True,
            "refreshed": True,
            "skipped": False,
            "account_label": selected_record.label,
            "account_hash": selected_record.expected_account_hash,
            "expires_at": "2026-06-19T13:00:00Z",
            "error_class": None,
            "error_message": None,
            "error_hint": None,
        }

    monkeypatch.setattr(
        loop.codex_oauth_refresh,
        "refresh_codex_oauth_inventory_record",
        fake_refresh,
    )

    def fake_persist(persist_config, event, *, record=None):
        captured["config"] = persist_config
        captured["event"] = dict(event)
        captured["record"] = record
        return True, 1, None, None

    monkeypatch.setattr(loop, "_persist_codex_auth_observation", fake_persist)

    events = loop.run_due_sidecar_tasks(
        config,
        loop.SidecarTaskState(),
        now_monotonic=100.0,
        now_wall=wall_now,
    )

    refresh_events = [event for event in events if event.get("event") == "codex_oauth_refresh"]
    assert len(refresh_events) == 1
    assert refresh_events[0]["auth_observation_status"] == "refreshed"
    assert refresh_events[0]["auth_observation_persisted"] is True
    assert refresh_events[0]["auth_observation_inserted_count"] == 1
    assert captured["config"] is config
    assert captured["event"]["account_label"] == "account1"
    assert captured["event"]["account_hash"] == record.expected_account_hash
    assert captured["record"] is record


def test_codex_inventory_refresh_isolates_failures_and_keeps_label_timers(
    monkeypatch,
    tmp_path,
) -> None:
    inventory = _codex_oauth_inventory("account1", "account2", root=tmp_path)
    wall_now = datetime(2026, 8, 13, 22, 30, tzinfo=timezone.utc)
    initial_expiry = int((wall_now + timedelta(hours=1)).timestamp())
    refreshed_expiry = int((wall_now + timedelta(hours=2)).timestamp())
    for record in inventory.records:
        record.auth_path.write_text(
            json.dumps(
                {
                    "tokens": {
                        "access_token": _build_test_jwt(
                            {"exp": initial_expiry}
                        ),
                        "refresh_token": f"refresh-{record.label}",
                        "account_id": f"acct-{record.label}",
                    }
                }
            ),
            encoding="utf-8",
        )
        record.auth_path.chmod(0o600)
    config = _codex_oauth_auth_persist_config(
        apply=False,
        codex_oauth_inventory=inventory,
        codex_refresh_buffer_seconds=3600,
    )
    calls = []

    def fake_refresh(record, **kwargs):
        calls.append(record.label)
        if record.label == "account1":
            raise RuntimeError(
                f"failed {record.auth_path} token=secret-token"
            )
        kwargs["on_token_endpoint_attempt"]()
        record.auth_path.write_text(
            json.dumps(
                {
                    "tokens": {
                        "access_token": _build_test_jwt(
                            {"exp": refreshed_expiry}
                        ),
                        "refresh_token": f"rotated-{record.label}",
                        "account_id": f"acct-{record.label}",
                    }
                }
            ),
            encoding="utf-8",
        )
        return {
            "attempted": True,
            "refreshed": True,
            "skipped": False,
            "account_label": record.label,
            "account_hash": record.expected_account_hash,
            "expires_at": "2026-08-14T00:30:00Z",
            "error_class": None,
            "error_message": None,
            "error_hint": None,
        }

    monkeypatch.setattr(
        loop.codex_oauth_refresh,
        "refresh_codex_oauth_inventory_record",
        fake_refresh,
    )

    state = loop.SidecarTaskState()
    events = loop._run_codex_oauth_refresh_task(
        config,
        state,
        now_monotonic=100.0,
        now_wall=wall_now,
    )

    refresh_events = [
        event for event in events if event["event"] == "codex_oauth_refresh"
    ]
    aggregate = next(
        event for event in events if event["event"] == "codex_oauth_refresh_aggregate"
    )
    assert calls == ["account1", "account2"]
    assert [event["account_label"] for event in refresh_events] == [
        "account1",
        "account2",
    ]
    assert aggregate["health"] == "degraded"
    assert aggregate["usable_count"] == 2
    rendered = json.dumps(events)
    assert "/home/zepfu/.codex" not in rendered
    assert "secret-token" not in rendered

    calls.clear()
    second_events = loop._run_codex_oauth_refresh_task(
        config,
        state,
        now_monotonic=400.0,
        now_wall=wall_now + timedelta(minutes=5),
    )
    assert calls == ["account1"]
    second_refresh_events = [
        event for event in second_events if event["event"] == "codex_oauth_refresh"
    ]
    second_by_label = {
        event["account_label"]: event for event in second_refresh_events
    }
    assert second_by_label["account1"]["actual_attempted"] is False
    assert second_by_label["account1"]["refresh_result_class"] == "refresh_failed"
    assert second_by_label["account1"]["credential_health"] == "degraded"
    assert second_by_label["account1"]["auth_observation_status"] == "failed"
    assert second_by_label["account2"]["actual_attempted"] is False
    assert second_by_label["account2"]["refresh_result_class"] == "refresh_not_due"
    assert state.codex_oauth_last_attempt_monotonic_by_label.get("account1") is None
    assert state.codex_oauth_last_attempt_monotonic_by_label["account2"] == 100.0
    second_aggregate = next(
        event
        for event in second_events
        if event["event"] == "codex_oauth_refresh_aggregate"
    )
    assert second_aggregate["health"] == "degraded"

    terminal = loop._codex_account_aggregate_event(
        event_name="codex_oauth_refresh_aggregate",
        config=config,
        records=inventory.records,
        usable_by_label={"account1": False, "account2": False},
        status_by_label={"account1": "failed", "account2": "failed"},
    )
    assert terminal["health"] == "terminal"


def test_codex_oauth_refresh_reinspects_replaced_account_identity(
    monkeypatch,
    tmp_path,
) -> None:
    inventory = _codex_oauth_inventory("account1", root=tmp_path)
    record = inventory.records[0]
    wall_now = datetime(2026, 8, 13, 22, 30, tzinfo=timezone.utc)
    future_expiry = int((wall_now + timedelta(hours=2)).timestamp())
    record.auth_path.write_text(
        json.dumps(
            {
                "tokens": {
                    "access_token": _build_test_jwt({"exp": future_expiry}),
                    "refresh_token": "refresh-account1",
                    "account_id": "acct-account1",
                }
            }
        ),
        encoding="utf-8",
    )
    config = _codex_oauth_auth_persist_config(
        apply=False,
        codex_oauth_inventory=inventory,
        codex_refresh_buffer_seconds=300,
    )
    helper_calls = []

    def fake_refresh(selected_record, **_kwargs):
        helper_calls.append(selected_record.label)
        return {
            "attempted": True,
            "refreshed": False,
            "skipped": False,
            "account_label": selected_record.label,
            "account_hash": selected_record.expected_account_hash,
            "expires_at": None,
            "error_class": "CodexOAuthIdentityMismatchError",
            "error_message": (
                f"Codex OAuth credential identity mismatch for account "
                f"'{selected_record.label}'."
            ),
            "error_hint": None,
        }

    monkeypatch.setattr(
        loop.codex_oauth_refresh,
        "refresh_codex_oauth_inventory_record",
        fake_refresh,
    )
    state = loop.SidecarTaskState()

    first_events = loop._run_codex_oauth_refresh_task(
        config,
        state,
        now_monotonic=100.0,
        now_wall=wall_now,
    )
    first_refresh = next(
        event for event in first_events if event["event"] == "codex_oauth_refresh"
    )
    assert first_refresh["refresh_result_class"] == "refresh_not_due"
    assert first_refresh["usable"] is True
    assert helper_calls == []

    replacement_path = tmp_path / "oauth.account1.replacement.json"
    replacement_path.write_text(
        json.dumps(
            {
                "tokens": {
                    "access_token": _build_test_jwt({"exp": future_expiry}),
                    "refresh_token": "refresh-other-account",
                    "account_id": "acct-other-account",
                }
            }
        ),
        encoding="utf-8",
    )
    os.replace(replacement_path, record.auth_path)

    second_events = loop._run_codex_oauth_refresh_task(
        config,
        state,
        now_monotonic=101.0,
        now_wall=wall_now + timedelta(seconds=1),
    )
    second_refresh = next(
        event for event in second_events if event["event"] == "codex_oauth_refresh"
    )
    assert helper_calls == ["account1"]
    assert second_refresh["refresh_result_class"] != "refresh_not_due"
    assert second_refresh["usable"] is False
    assert second_refresh["actual_attempted"] is False
    assert second_refresh["account_label"] == "account1"
    assert "acct-other-account" not in json.dumps(second_events)


def test_codex_actual_attempt_throttle_retains_failed_degraded_aggregate(
    monkeypatch,
    tmp_path,
) -> None:
    inventory = _codex_oauth_inventory("account1", root=tmp_path)
    record = inventory.records[0]
    record.auth_path.write_text(
        json.dumps(
            {
                "tokens": {
                    "access_token": _build_test_jwt(
                        {
                            "exp": int(
                                datetime(
                                    2026,
                                    8,
                                    13,
                                    23,
                                    0,
                                    tzinfo=timezone.utc,
                                ).timestamp()
                            )
                        }
                    ),
                    "refresh_token": "refresh-account1",
                    "account_id": "acct-account1",
                }
            }
        ),
        encoding="utf-8",
    )
    config = _codex_oauth_auth_persist_config(
        apply=False,
        codex_oauth_inventory=inventory,
        codex_refresh_interval_seconds=300.0,
        codex_refresh_buffer_seconds=3600,
    )
    helper_calls = []

    def fake_refresh(selected_record, **kwargs):
        helper_calls.append(selected_record.label)
        kwargs["on_token_endpoint_attempt"]()
        return {
            "attempted": True,
            "refreshed": False,
            "skipped": False,
            "account_label": selected_record.label,
            "account_hash": selected_record.expected_account_hash,
            "expires_at": "2026-08-13T23:00:00Z",
            "error_class": "HTTPError",
            "error_message": "token endpoint rejected request",
            "error_hint": None,
        }

    monkeypatch.setattr(
        loop.codex_oauth_refresh,
        "refresh_codex_oauth_inventory_record",
        fake_refresh,
    )
    state = loop.SidecarTaskState()
    wall_now = datetime(2026, 8, 13, 22, 30, tzinfo=timezone.utc)

    first_events = loop._run_codex_oauth_refresh_task(
        config,
        state,
        now_monotonic=100.0,
        now_wall=wall_now,
    )
    first_refresh = next(
        event for event in first_events if event["event"] == "codex_oauth_refresh"
    )
    first_aggregate = next(
        event
        for event in first_events
        if event["event"] == "codex_oauth_refresh_aggregate"
    )
    assert first_refresh["refresh_result_class"] == "refresh_failed"
    assert first_refresh["credential_health"] == "degraded"
    assert first_refresh["auth_observation_status"] == "failed"
    assert first_aggregate["health"] == "degraded"
    assert state.codex_oauth_last_attempt_monotonic_by_label["account1"] == 100.0

    second_events = loop._run_codex_oauth_refresh_task(
        config,
        state,
        now_monotonic=101.0,
        now_wall=wall_now + timedelta(seconds=1),
    )
    second_refresh = next(
        event for event in second_events if event["event"] == "codex_oauth_refresh"
    )
    second_aggregate = next(
        event
        for event in second_events
        if event["event"] == "codex_oauth_refresh_aggregate"
    )
    assert helper_calls == ["account1"]
    assert second_refresh["actual_attempted"] is False
    assert second_refresh["refresh_result_class"] == "refresh_failed"
    assert second_refresh["credential_health"] == "degraded"
    assert second_refresh["auth_observation_status"] == "degraded"
    assert second_aggregate["health"] == "degraded"
    assert second_aggregate["accounts"] == [
        {
            "account_label": "account1",
            "account_hash": record.expected_account_hash,
            "status": "degraded",
            "usable": True,
        }
    ]
    assert state.codex_oauth_last_attempt_monotonic_by_label["account1"] == 100.0


def test_codex_inventory_passive_health_isolates_records(monkeypatch) -> None:
    inventory = _codex_oauth_inventory("account1", "account2")
    config = _grok_oidc_auth_persist_config(
        grok_oidc_refresh_enabled=False,
        codex_oauth_inventory=inventory,
        provider_auth_health_poll_enabled=True,
        provider_auth_health_poll_interval_seconds=3600.0,
    )
    inspected = []

    def fake_inspect(record):
        inspected.append(record.label)
        if record.label == "account1":
            return {
                "attempted": True,
                "health_status": "malformed",
                "account_label": record.label,
                "account_hash": record.expected_account_hash,
                "expires_at": None,
                "error_class": "ValueError",
                "error_message": "account1 health inspection failed.",
            }
        return {
            "attempted": True,
            "health_status": "fresh",
            "account_label": record.label,
            "account_hash": record.expected_account_hash,
            "expires_at": "2026-08-08T18:00:00Z",
            "error_class": None,
            "error_message": None,
        }

    monkeypatch.setattr(loop, "_inspect_codex_inventory_record_health", fake_inspect)
    monkeypatch.setattr(
        loop,
        "_persist_codex_passive_auth_observation",
        lambda *_args, **_kwargs: (False, 0, None, "apply_disabled"),
    )
    state = loop.SidecarTaskState(
        provider_auth_health_poll_last_attempt_monotonic=100.0
    )

    events = loop._run_provider_auth_health_poll_task(
        config,
        state,
        now_monotonic=200.0,
    )

    assert inspected == ["account1", "account2"]
    aggregate = next(
        event for event in events if event["event"] == "codex_oauth_health_aggregate"
    )
    assert aggregate["health"] == "degraded"
    assert aggregate["usable_count"] == 1


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

    def fake_xai_refresh(*_args, **kwargs):
        kwargs["on_token_endpoint_attempt"]()
        return {
            "attempted": True,
            "refreshed": True,
            "skipped": False,
            "auth_file": config.xai_oauth_auth_file,
            "scope": config.xai_oauth_scope,
            "expires_at": "2026-06-19T13:00:00Z",
            "error_class": None,
            "error_message": None,
        }

    monkeypatch.setattr(
        loop.xai_oauth_refresh,
        "refresh_xai_oauth_auth_file",
        fake_xai_refresh,
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
        now_wall=datetime(2026, 6, 19, 12, 0, tzinfo=timezone.utc),
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
        "inspect_grok_oidc_refresh_eligibility",
        lambda *_args, now, **_kwargs: {
            "eligibility_checked_at": now().isoformat().replace("+00:00", "Z"),
            "expires_at": "2026-06-19T13:00:00Z",
            "refresh_due_at": "2026-06-19T12:55:00Z",
            "next_refresh_check_at": "2026-06-19T12:55:00Z",
            "eligible": False,
            "credential_health": "fresh",
            "usable": True,
            "error_class": None,
            "error_message": None,
        },
    )

    def fake_grok_refresh(*_args, **kwargs):
        kwargs["on_token_endpoint_attempt"]()
        return {
            "attempted": True,
            "refreshed": True,
            "skipped": False,
            "auth_file": config.grok_oidc_auth_file,
            "scope": "https://auth.x.ai::b1a00492-073a-47ea-816f-4c329264a828",
            "expires_at": "2026-06-19T13:00:00Z",
            "error_class": None,
            "error_message": None,
        }

    monkeypatch.setattr(
        loop.grok_oidc_refresh,
        "refresh_grok_oidc_auth_file",
        fake_grok_refresh,
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
        now_wall=datetime(2026, 6, 19, 12, 0, tzinfo=timezone.utc),
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
        "inspect_grok_oidc_refresh_eligibility",
        lambda *_args, now, **_kwargs: {
            "eligibility_checked_at": now().isoformat().replace("+00:00", "Z"),
            "expires_at": "2026-06-19T13:00:00Z",
            "refresh_due_at": "2026-06-19T12:55:00Z",
            "next_refresh_check_at": "2026-06-19T12:55:00Z",
            "eligible": False,
            "credential_health": "fresh",
            "usable": True,
            "error_class": None,
            "error_message": None,
        },
    )

    def fake_grok_failure_refresh(*_args, **kwargs):
        kwargs["on_token_endpoint_attempt"]()
        return {
            "attempted": True,
            "refreshed": True,
            "skipped": False,
            "auth_file": config.grok_oidc_auth_file,
            "scope": "scope",
            "expires_at": "2026-06-19T13:00:00Z",
            "error_class": None,
            "error_message": None,
        }

    monkeypatch.setattr(
        loop.grok_oidc_refresh,
        "refresh_grok_oidc_auth_file",
        fake_grok_failure_refresh,
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
        now_wall=datetime(2026, 6, 19, 12, 0, tzinfo=timezone.utc),
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
        codex_oauth_inventory=_codex_oauth_inventory("account1"),
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


def _codex_quota_payload(
    *,
    primary_reset: str = "2026-08-08T18:00:00Z",
    secondary_reset: str = "2026-08-15T12:00:00Z",
    model: str | None = None,
) -> dict:
    rate_limits = {
        "limit_id": "codex_bengalfox",
        "limit_name": "GPT-5.3-Codex-Spark",
        "primary": {
            "used_percent": 12.5,
            "window_minutes": 300,
            "resets_at": primary_reset,
        },
        "secondary": {
            "used_percent": 51.0,
            "window_minutes": 10080,
            "resets_at": secondary_reset,
        },
    }
    if model is not None:
        rate_limits["model"] = model
    return {
        "rate_limit_reset_credits": {"available_count": 2},
        "rate_limits": rate_limits,
    }


def _codex_reset_credit_auth_context(**overrides) -> dict:
    record = _codex_oauth_inventory("account1").records[0]
    context = {
        "access_token": "access-token-secret",
        "account_id": "acct-account1",
        "account_label": record.label,
        "account_hash": record.expected_account_hash,
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
    assert headers["ChatGPT-Account-Id"] == "acct-account1"
    headers_json = json.dumps(headers)
    assert "refresh_token" not in headers_json
    assert "acct-account1" in headers_json


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
    monkeypatch.setenv(
        "LITELLM_CODEX_OAUTH_INVENTORY",
        _codex_oauth_inventory_env_payload(),
    )
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


def test_loop_config_loads_codex_inventory_for_passive_health(monkeypatch) -> None:
    monkeypatch.setenv(
        "LITELLM_CODEX_OAUTH_INVENTORY",
        _codex_oauth_inventory_env_payload(),
    )
    monkeypatch.setenv("AAWM_PROVIDER_AUTH_HEALTH_POLL_ENABLED", "1")
    monkeypatch.setenv("AAWM_CODEX_OAUTH_REFRESH_ENABLED", "0")
    monkeypatch.setenv("AAWM_CODEX_RESET_CREDIT_POLL_ENABLED", "0")

    config = loop.parse_config([])

    assert config.codex_oauth_inventory is not None
    assert [
        record.label
        for record in config.codex_oauth_inventory.ordered_records(
            enabled_only=True
        )
    ] == ["account1", "account2"]


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
    first_poll = next(
        event for event in first_events if event["event"] == "codex_reset_credit_poll"
    )
    third_poll = next(
        event for event in third_events if event["event"] == "codex_reset_credit_poll"
    )
    assert first_poll["available_count"] == 2
    assert first_poll["inserted_count"] == 0
    assert second_events == []
    assert third_poll["available_count"] == 2


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


def test_codex_quota_observations_keep_distinct_fresh_windows_without_inventing_model() -> None:
    config = _codex_reset_credit_poll_config(apply=False)
    record = config.codex_oauth_inventory.records[0]
    observed_at = datetime(2026, 8, 8, 12, 0, tzinfo=timezone.utc)

    rows, summary = loop._build_codex_quota_rate_limit_observations(
        config,
        record=record,
        observed_at=observed_at,
        response_body=_codex_quota_payload(),
    )

    assert summary["required_periods_fresh"] is True
    assert summary["period_states"] == {
        "five_hour": "fresh",
        "seven_day": "fresh",
    }
    by_period = {row["quota_period"]: row for row in rows}
    assert by_period["five_hour"]["remaining_pct"] == 87.5
    assert by_period["seven_day"]["remaining_pct"] == 49.0
    assert by_period["five_hour"]["provider_resets_at"] == datetime(
        2026,
        8,
        8,
        18,
        0,
        tzinfo=timezone.utc,
    )
    assert all(row["account_hash"] == record.expected_account_hash for row in rows)
    assert all(row["evidence"]["account_label"] == "account1" for row in rows)
    assert all(row["model"] is None for row in rows)
    assert all(
        row["evidence"]["upstream_scope"] == "codex_bengalfox" for row in rows
    )


def test_codex_quota_stale_and_unknown_windows_are_not_healthy() -> None:
    config = _codex_reset_credit_poll_config(apply=False)
    record = config.codex_oauth_inventory.records[0]
    payload = _codex_quota_payload(model="gpt-5.3-codex-spark")
    payload["rate_limits"]["primary"]["resets_at"] = "2026-08-08T11:00:00Z"
    payload["rate_limits"]["secondary"].pop("resets_at")

    rows, summary = loop._build_codex_quota_rate_limit_observations(
        config,
        record=record,
        observed_at=datetime(2026, 8, 8, 12, 0, tzinfo=timezone.utc),
        response_body=payload,
    )

    assert summary["required_periods_fresh"] is False
    assert summary["period_states"] == {
        "five_hour": "stale",
        "seven_day": "unknown",
    }
    assert summary["fresh_window_count"] == 0
    assert {row["status"] for row in rows} == {"stale", "unknown"}
    assert all(row["remaining_pct"] is None for row in rows)
    assert all(row["used_percentage"] is None for row in rows)
    assert all(row["exhausted"] is False for row in rows)
    assert all(row["model"] == "gpt-5.3-codex-spark" for row in rows)


def test_codex_quota_poll_uses_each_inventory_records_exact_headers(
    monkeypatch,
) -> None:
    inventory = _codex_oauth_inventory("account1", "account2")
    config = _codex_reset_credit_poll_config(
        apply=False,
        codex_oauth_inventory=inventory,
    )
    snapshots = {
        record.label: CodexOAuthCredentialSnapshot(
            record=record,
            account_hash=record.expected_account_hash,
            expires_at=time.time() + 3600,
            access_token=f"token-{record.label}",
            account_id=f"acct-{record.label}",
        )
        for record in inventory.records
    }
    requests = []

    monkeypatch.setattr(
        loop,
        "load_codex_oauth_credential",
        lambda record: snapshots[record.label],
    )

    def fake_urlopen(request, timeout):
        requests.append(
            (
                request.get_header("Authorization"),
                request.get_header("Chatgpt-account-id"),
                timeout,
            )
        )
        return _FakeAlibabaHTTPResponse(json.dumps(_codex_quota_payload()))

    monkeypatch.setattr(loop.urllib_request, "urlopen", fake_urlopen)

    for record in inventory.records:
        loop._fetch_codex_reset_credit_payload(config, record)

    assert requests == [
        ("Bearer token-account1", "acct-account1", 30.0),
        ("Bearer token-account2", "acct-account2", 30.0),
    ]


def test_codex_quota_poll_failure_does_not_suppress_other_account(
    monkeypatch,
) -> None:
    inventory = _codex_oauth_inventory("account1", "account2")
    config = _codex_reset_credit_poll_config(
        apply=False,
        codex_oauth_inventory=inventory,
    )
    calls = []

    def fake_fetch(_config, record):
        calls.append(record.label)
        if record.label == "account1":
            raise RuntimeError("account1 unavailable")
        return {
            "status_code": 200,
            "payload": _codex_quota_payload(
                primary_reset="2026-12-08T18:00:00Z",
                secondary_reset="2026-12-15T12:00:00Z",
            ),
            "auth_context": {
                "access_token": "token-account2",
                "account_id": "acct-account2",
                "account_label": record.label,
                "account_hash": record.expected_account_hash,
            },
            "attempt_count": 1,
            "retry_count": 0,
            "poll_url": probes.DEFAULT_CODEX_RESET_CREDIT_DETAIL_URL,
        }

    monkeypatch.setattr(loop, "_fetch_codex_reset_credit_payload", fake_fetch)

    class _FrozenDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            value = datetime(2026, 8, 8, 12, 0, tzinfo=timezone.utc)
            return value if tz is None else value.astimezone(tz)

    monkeypatch.setattr(loop, "datetime", _FrozenDateTime)
    events = loop._run_codex_reset_credit_poll_task(
        config,
        loop.SidecarTaskState(),
        now_monotonic=100.0,
    )

    assert calls == ["account1", "account2"]
    by_label = {
        event["account_label"]: event
        for event in events
        if event["event"] == "codex_reset_credit_poll"
    }
    assert by_label["account1"]["error_class"] == "RuntimeError"
    assert by_label["account2"]["quota_health"] == "healthy"
    aggregate = next(
        event for event in events if event["event"] == "codex_quota_poll_aggregate"
    )
    assert aggregate["health"] == "degraded"


def test_persist_codex_quota_observations_uses_synchronous_psycopg_insert(
    monkeypatch,
) -> None:
    config = _codex_reset_credit_poll_config(
        apply=True,
        codex_quota_dsn=(
            "postgresql://litellm_dev:secret@pgbouncer-litellm-dev:6432/"
            "litellm_dev?application_name=codex-quota-writer"
        ),
    )
    record = config.codex_oauth_inventory.records[0]
    rows, _summary = loop._build_codex_quota_rate_limit_observations(
        config,
        record=record,
        observed_at=datetime(2026, 8, 8, 12, 0, tzinfo=timezone.utc),
        response_body=_codex_quota_payload(),
    )
    executed: list = []
    connected_dsns: list[str] = []

    class _FakeCursor:
        rowcount = 1

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def execute(self, sql, params=None):
            executed.append((sql, params))

    class _FakeConnection:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def cursor(self):
            return _FakeCursor()

        def rollback(self):
            raise AssertionError("rollback must not run on success")

    monkeypatch.setattr(
        loop.probes.psycopg,
        "connect",
        lambda dsn: connected_dsns.append(dsn) or _FakeConnection(),
    )

    inserted = loop._persist_codex_quota_observations(config, rows)

    assert inserted == 2
    assert connected_dsns == [
        "postgresql://litellm_dev:secret@pgbouncer-litellm-dev:6432/"
        "litellm_dev?application_name=codex-quota-writer"
    ]
    quota_inserts = [
        (sql, params)
        for sql, params in executed
        if "INSERT INTO public.rate_limit_observations" in sql
    ]
    assert len(quota_inserts) == 2
    quota_keys = set()
    for _sql, params in quota_inserts:
        assert params[1] == "codex"
        assert params[3] == record.expected_account_hash
        assert params[4] == "openai"
        assert params[18] == loop.DEFAULT_CODEX_QUOTA_SOURCE
        quota_keys.add(params[6])
    assert quota_keys == {
        "codex_bengalfox:primary",
        "codex_bengalfox:secondary",
    }
    assert {
        row["account_hash"] for row in rows
    } == {record.expected_account_hash}


def test_persist_codex_quota_observations_falls_back_to_general_dsn(
    monkeypatch,
) -> None:
    config = _codex_reset_credit_poll_config(
        apply=True,
        dsn=(
            "postgresql://aawm:secret@pgbouncer-aawm-dev:6432/"
            "aawm_tristore?application_name=general-sidecar"
        ),
        codex_quota_dsn=None,
    )
    record = config.codex_oauth_inventory.records[0]
    rows, _summary = loop._build_codex_quota_rate_limit_observations(
        config,
        record=record,
        observed_at=datetime(2026, 8, 8, 12, 0, tzinfo=timezone.utc),
        response_body=_codex_quota_payload(),
    )
    connected_dsns: list[str] = []

    class _FakeCursor:
        rowcount = 1

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def execute(self, _sql, _params=None):
            return None

    class _FakeConnection:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def cursor(self):
            return _FakeCursor()

    monkeypatch.setattr(
        loop.probes.psycopg,
        "connect",
        lambda dsn: connected_dsns.append(dsn) or _FakeConnection(),
    )

    assert loop._persist_codex_quota_observations(config, rows) == 2
    assert connected_dsns == [
        "postgresql://aawm:secret@pgbouncer-aawm-dev:6432/"
        "aawm_tristore?application_name=general-sidecar"
    ]


def test_persist_codex_quota_observations_propagates_db_write_skipped(
    monkeypatch,
) -> None:
    config = _codex_reset_credit_poll_config(apply=True)
    record = config.codex_oauth_inventory.records[0]
    rows, _summary = loop._build_codex_quota_rate_limit_observations(
        config,
        record=record,
        observed_at=datetime(2026, 8, 8, 12, 0, tzinfo=timezone.utc),
        response_body=_codex_quota_payload(),
    )

    class _FakeCursor:
        rowcount = 0

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def execute(self, sql, params=None):
            if "INSERT INTO public.rate_limit_observations" in sql:
                raise loop.probes.psycopg.errors.LockNotAvailable("lock timeout")

    class _FakeConnection:
        rolled_back = False

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def cursor(self):
            return _FakeCursor()

        def rollback(self):
            self.rolled_back = True

    fake_conn = _FakeConnection()
    monkeypatch.setattr(
        loop.probes.psycopg,
        "connect",
        lambda _dsn: fake_conn,
    )

    with pytest.raises(loop.probes.ProviderStatusDatabaseWriteSkipped):
        loop._persist_codex_quota_observations(config, rows)
    assert fake_conn.rolled_back is True


def test_codex_quota_poll_reports_synchronous_insert_counts(
    monkeypatch,
) -> None:
    config = _codex_reset_credit_poll_config()
    monkeypatch.setattr(
        loop,
        "_fetch_codex_reset_credit_payload",
        lambda *_args, **_kwargs: {
            "status_code": 200,
            "payload": _codex_quota_payload(),
            "auth_context": _codex_reset_credit_auth_context(),
            "attempt_count": 1,
            "retry_count": 0,
            "poll_url": probes.DEFAULT_CODEX_RESET_CREDIT_DETAIL_URL,
        },
    )
    monkeypatch.setattr(
        loop,
        "_persist_codex_reset_credit_observation",
        lambda *_args, **_kwargs: (1, 1),
    )
    monkeypatch.setattr(
        loop,
        "_persist_codex_quota_observations",
        lambda _config, observations: len(observations),
    )

    events = loop._run_codex_reset_credit_poll_task(
        config,
        loop.SidecarTaskState(),
        now_monotonic=100.0,
    )

    poll = next(
        event for event in events if event["event"] == "codex_reset_credit_poll"
    )
    assert poll["credit_inserted_count"] == 1
    assert poll["quota_observation_count"] == 2
    assert poll["quota_inserted_count"] == 2
    assert poll["quota_storage_status"] == "persisted"
    assert poll["inserted_count"] == 1
    assert "quota_accepted_count" not in poll


def test_codex_quota_poll_reports_db_write_failure_disposition(
    monkeypatch,
) -> None:
    config = _codex_reset_credit_poll_config()
    monkeypatch.setattr(
        loop,
        "_fetch_codex_reset_credit_payload",
        lambda *_args, **_kwargs: {
            "status_code": 200,
            "payload": _codex_quota_payload(),
            "auth_context": _codex_reset_credit_auth_context(),
            "attempt_count": 1,
            "retry_count": 0,
            "poll_url": probes.DEFAULT_CODEX_RESET_CREDIT_DETAIL_URL,
        },
    )
    monkeypatch.setattr(
        loop,
        "_persist_codex_reset_credit_observation",
        lambda *_args, **_kwargs: (1, 1),
    )

    def _raise_skipped(_config, _observations):
        raise loop.probes.ProviderStatusDatabaseWriteSkipped(
            error_class="LockNotAvailable",
            message="lock timeout",
        )

    monkeypatch.setattr(loop, "_persist_codex_quota_observations", _raise_skipped)

    events = loop._run_codex_reset_credit_poll_task(
        config,
        loop.SidecarTaskState(),
        now_monotonic=100.0,
    )

    poll = next(
        event for event in events if event["event"] == "codex_reset_credit_poll"
    )
    assert poll["quota_observation_count"] == 2
    assert poll["quota_inserted_count"] == 0
    assert poll["quota_storage_status"] == "db_write_failed"
    assert poll["quota_error_class"] == "ProviderStatusDatabaseWriteSkipped"
    assert poll["quota_health"] == "terminal"


def test_codex_quota_poll_apply_disabled_keeps_quota_unpersisted(
    monkeypatch,
) -> None:
    config = _codex_reset_credit_poll_config(apply=False)
    monkeypatch.setattr(
        loop,
        "_fetch_codex_reset_credit_payload",
        lambda *_args, **_kwargs: {
            "status_code": 200,
            "payload": _codex_quota_payload(),
            "auth_context": _codex_reset_credit_auth_context(),
            "attempt_count": 1,
            "retry_count": 0,
            "poll_url": probes.DEFAULT_CODEX_RESET_CREDIT_DETAIL_URL,
        },
    )
    monkeypatch.setattr(
        loop,
        "_persist_codex_quota_observations",
        lambda *_args, **_kwargs: pytest.fail(
            "persistence must not run when apply is disabled"
        ),
    )

    events = loop._run_codex_reset_credit_poll_task(
        config,
        loop.SidecarTaskState(),
        now_monotonic=100.0,
    )

    poll = next(
        event for event in events if event["event"] == "codex_reset_credit_poll"
    )
    assert poll["quota_observation_count"] == 2
    assert poll["quota_inserted_count"] == 0
    assert poll["quota_storage_status"] == "apply_disabled"


@pytest.mark.parametrize(
    ("sidecar_events", "expected_exit", "expected_status", "optional_status"),
    [
        (
            [
                {
                    "event": "codex_oauth_refresh",
                    "account_label": "account1",
                    "account_hash": "111111111111",
                    "refreshed": False,
                    "skipped": False,
                    "error_class": "CredentialFileLockError",
                }
            ],
            1,
            "failed",
            "healthy",
        ),
        (
            [
                {"event": "grok_oidc_refresh", "skipped": True},
                {
                    "event": "codex_oauth_refresh",
                    "account_label": "account1",
                    "account_hash": "111111111111",
                    "skipped": True,
                },
                {"event": "xai_oauth_refresh", "skipped": True},
            ],
            0,
            "healthy",
            "healthy",
        ),
        (
            [
                {
                    "event": "codex_reset_credit_poll",
                    "error_class": "TimeoutError",
                }
            ],
            0,
            "healthy",
            "degraded",
        ),
    ],
)
def test_main_once_enforces_required_refresh_policy(
    monkeypatch,
    sidecar_events,
    expected_exit,
    expected_status,
    optional_status,
) -> None:
    config = _codex_reset_credit_poll_config(
        apply=False,
        codex_reset_credit_poll_enabled=False,
    )
    emitted = []
    monkeypatch.setattr(loop, "parse_config", lambda _argv: config)
    monkeypatch.setattr(loop, "validate_runtime_guardrails", lambda _config: None)
    monkeypatch.setattr(loop, "run_cycle", lambda _config: {"event": "cycle"})
    monkeypatch.setattr(
        loop,
        "run_due_sidecar_tasks",
        lambda _config, _state: list(sidecar_events),
    )
    monkeypatch.setattr(loop.signal, "signal", lambda *_args: None)
    monkeypatch.setattr(loop, "_emit", emitted.append)

    assert loop.main(["--once"]) == expected_exit

    one_shot = emitted[-1]
    assert one_shot["event"] == "provider_status_sidecar_one_shot_status"
    assert one_shot["status"] == expected_status
    assert one_shot["optional_status"] == optional_status


def test_compose_wires_codex_reset_credit_poll_defaults() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    compose_text = (repo_root / "docker-compose.dev.yml").read_text()

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
