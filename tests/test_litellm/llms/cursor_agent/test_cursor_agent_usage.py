import hashlib
import json
from datetime import datetime, timezone
from unittest.mock import patch

import pytest

from litellm.llms.cursor_agent import common_utils, usage_client
from litellm.llms.cursor_agent.common_utils import (
    CURSOR_AGENT_DASHBOARD_HOST,
    CURSOR_AGENT_TURN_HOST,
    current_period_usage_url,
)
from litellm.llms.cursor_agent.usage import (
    CURSOR_AGENT_GROK_BOT_USAGE_SOURCE_ENV,
    CURSOR_AGENT_MONTHLY_QUOTA_KEY,
    grok_bot_reevaluation_checkpoint,
    hash_cursor_agent_account_identity,
    parse_current_period_usage,
)
from litellm.llms.cursor_agent.usage_client import (
    CursorAgentUsageAuthError,
    resolve_access_token as resolve_usage_access_token,
)


def _camelcase_usage_payload(**overrides):
    payload = {
        "billingCycleStart": 1754524800,
        "billingCycleEnd": 1757203200,
        "planUsage": {
            "totalSpend": 4200,
            "includedSpend": 1250,
            "remaining": 8750,
            "limit": 10000,
            "remainingBonus": False,
            "bonusTooltip": "",
            "autoPercentUsed": 41,
            "apiPercentUsed": 7,
            "totalPercentUsed": 48,
        },
        "spendLimitUsage": {"limitType": "user"},
        "displayThreshold": 80,
        "enabled": True,
        "displayMessage": "staff: this is includedSpend / limit",
        "user": {"id": "acct-not-for-persistence"},
    }
    payload.update(overrides)
    return payload


def test_common_utils_reexports_stdlib_usage_helpers():
    assert common_utils.current_period_usage_url is usage_client.current_period_usage_url
    assert common_utils.build_dashboard_headers is usage_client.build_dashboard_headers
    assert common_utils.build_turn_headers is usage_client.build_turn_headers
    assert common_utils.cursor_agent_user_agent is usage_client.cursor_agent_user_agent
    assert common_utils.resolve_dashboard_api_base is usage_client.resolve_dashboard_api_base
    assert common_utils.CURSOR_AGENT_DASHBOARD_HOST is usage_client.CURSOR_AGENT_DASHBOARD_HOST
    assert common_utils.CURSOR_AGENT_USAGE_PATH is usage_client.CURSOR_AGENT_USAGE_PATH


def test_usage_url_is_dashboard_connect_not_cloud_agents():
    url = current_period_usage_url(None)
    assert url == (
        "https://api2.cursor.sh/aiserver.v1.DashboardService/GetCurrentPeriodUsage"
    )
    assert url.startswith(CURSOR_AGENT_DASHBOARD_HOST)
    assert CURSOR_AGENT_TURN_HOST not in url
    assert "/v0/me" not in url
    assert "api.cursor.com" not in url


def test_parse_camelcase_dump_maps_included_spend_not_percent_fields():
    snapshot = parse_current_period_usage(_camelcase_usage_payload())

    assert snapshot["state"] == "valid_nonzero"
    assert snapshot["quota_used"] == 1250.0
    assert snapshot["quota_limit"] == 10000.0
    assert snapshot["quota_remaining"] == 8750.0
    assert snapshot["quota_period"] == "monthly"
    assert snapshot["quota_key"] == CURSOR_AGENT_MONTHLY_QUOTA_KEY
    assert snapshot["remaining_pct"] == 87.5
    assert snapshot["raw_provider_fields"]["included_spend_cents"] == 1250.0
    assert snapshot["raw_provider_fields"]["total_spend_cents"] == 4200.0
    assert snapshot["raw_provider_fields"]["total_percent_used"] == 48.0
    assert snapshot["raw_provider_fields"]["percent_fields_are_not_total_over_limit"] is True
    assert snapshot["remaining_pct"] != 48.0
    assert snapshot["remaining_pct"] != pytest.approx(100.0 - (4200.0 / 10000.0 * 100.0))
    assert snapshot["billing_period_start_at"] == datetime.fromtimestamp(
        1754524800, tz=timezone.utc
    )
    assert snapshot["billing_period_end_at"] == datetime.fromtimestamp(
        1757203200, tz=timezone.utc
    )


def test_parse_snake_case_proto_aliases():
    snapshot = parse_current_period_usage(
        {
            "billing_cycle_start": "2026-08-01T00:00:00Z",
            "billing_cycle_end": "2026-09-01T00:00:00Z",
            "plan_usage": {
                "included_spend": 0,
                "limit": 5000,
                "remaining": 5000,
            },
            "account_id": "acct-snake",
        }
    )
    assert snapshot["state"] == "valid_zero"
    assert snapshot["quota_used"] == 0.0
    assert snapshot["quota_limit"] == 5000.0
    assert snapshot["quota_remaining"] == 5000.0
    assert snapshot["remaining_pct"] == 100.0


def test_account_identity_is_hashed_never_raw():
    payload = _camelcase_usage_payload()
    snapshot = parse_current_period_usage(payload)
    expected = hashlib.sha256(
        b"cursor-agent-account|user.id=acct-not-for-persistence"
    ).hexdigest()
    assert snapshot["account_hash"] == expected
    serialized = json.dumps(snapshot, default=str)
    assert "acct-not-for-persistence" not in serialized
    assert "user.id" in snapshot["account_identity_fields"]
    hashed, fields = hash_cursor_agent_account_identity(payload)
    assert hashed == expected
    assert "acct-not-for-persistence" not in hashed
    assert fields == ["user.id"]


def test_grok_bot_stays_unknown_even_with_checkpoint_env(monkeypatch):
    monkeypatch.setenv(
        CURSOR_AGENT_GROK_BOT_USAGE_SOURCE_ENV,
        "aiserver.v1.DashboardService/NotAGrokBotRpc",
    )
    checkpoint = grok_bot_reevaluation_checkpoint()
    snapshot = parse_current_period_usage(_camelcase_usage_payload())
    assert checkpoint["status"] == "unknown"
    assert checkpoint["quota_key"] is None
    assert checkpoint["reevaluation_ready"] is True
    assert snapshot["grok_bot"]["status"] == "unknown"
    assert snapshot["grok_bot"]["quota_key"] is None
    assert snapshot["evidence"]["weekly_grok_bot"] == "unknown"
    assert snapshot["evidence"]["weekly_grok_bot_quota_key"] is None
    assert "weekly" not in (snapshot.get("quota_key") or "")
    assert "grok_bot" not in (snapshot.get("quota_key") or "")


def test_missing_plan_usage_is_absent_not_zero():
    snapshot = parse_current_period_usage({"user": {"id": "acct-1"}})
    assert snapshot["state"] == "absent"
    assert snapshot["quota_used"] is None
    assert snapshot["quota_limit"] is None


def test_malformed_included_spend_is_not_persisted_as_zero():
    snapshot = parse_current_period_usage(
        {
            "planUsage": {"includedSpend": "not-cents", "limit": 100},
            "userId": "acct-2",
        }
    )
    assert snapshot["state"] == "malformed"
    assert snapshot["quota_used"] is None


def test_usage_client_resolve_access_token_ignores_cli_key_and_never_exchanges(
    monkeypatch,
):
    monkeypatch.setenv("CURSOR_CLI_KEY", "cli-key-must-be-ignored")
    monkeypatch.delenv("CURSOR_AUTH_TOKEN", raising=False)
    monkeypatch.delenv("CURSOR_API_KEY", raising=False)
    with pytest.raises(CursorAgentUsageAuthError):
        resolve_usage_access_token(allow_exchange=True)

    monkeypatch.setenv("CURSOR_API_KEY", "raw-api-key")
    assert resolve_usage_access_token(allow_exchange=True) == "raw-api-key"

    monkeypatch.setenv("CURSOR_AUTH_TOKEN", "stored-access-token")
    assert resolve_usage_access_token(None, allow_exchange=True) == "stored-access-token"
    assert resolve_usage_access_token("explicit-access") == "explicit-access"


def test_no_cli_subprocess_on_usage_parse():
    with patch("subprocess.Popen") as mock_popen, patch("subprocess.run") as mock_run:
        parse_current_period_usage(_camelcase_usage_payload())
        grok_bot_reevaluation_checkpoint()
        current_period_usage_url(None)
        mock_popen.assert_not_called()
        mock_run.assert_not_called()
