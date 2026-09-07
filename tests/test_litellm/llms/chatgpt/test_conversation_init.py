"""Focused tests for the ChatGPT conversation-init observer."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from urllib import request as urllib_request

import pytest

from litellm.llms.chatgpt.conversation_init import (
    CHATGPT_CONVERSATION_INIT_CLIENT,
    CHATGPT_CONVERSATION_INIT_DEFAULT_URL,
    CHATGPT_CONVERSATION_INIT_PATH,
    CHATGPT_CONVERSATION_INIT_PROVIDER,
    CHATGPT_CONVERSATION_INIT_SOURCE,
    ChatGPTConversationInitCollector,
    ChatGPTConversationInitError,
    OracleBrowserBoundaryUnavailable,
    OracleBrowserConversationInitTransport,
    build_conversation_init_request,
    build_conversation_init_rate_limit_tuples,
    collect_conversation_init_observations,
    collect_conversation_init_snapshot,
    collect_conversation_init_snapshot_from_oracle_browser,
    conversation_init_request_contract,
    hash_chatgpt_conversation_init_account_identity,
    hash_chatgpt_conversation_init_source_identity,
    looks_like_conversation_init_payload,
    parse_conversation_init_observations,
    request_has_conversation_content,
    sanitize_conversation_init_boundary,
    write_conversation_init_snapshot,
)


# ---------------------------------------------------------------------------
# Payload fixtures
# ---------------------------------------------------------------------------

# Real conversation-init responses carry an account identity envelope. Every
# payload fixture includes it so parsing exercises the authenticated path.
_ACCOUNT_ENVELOPE = {"user": {"id": "user-test-0001"}}


def _with_identity(payload):
    return {**payload, **_ACCOUNT_ENVELOPE}


_VALID_PAYLOAD = {
    "type": "conversation_init",
    "default_model_slug": "gpt-6-pro",
    "intended_default_model_slug": "gpt-6-pro",
    "model_limits": [],
    "limits_progress": [
        {
            "feature": "deep_research",
            "remaining": 250,
            "reset_after": "2026-10-05T13:24:28Z",
        },
        {
            "feature": "image_gen",
            "remaining": 1000,
            "reset_after": "2026-09-06T13:24:28Z",
        },
    ],
    "blocked_features": [],
    "atlas_mode_enabled": False,
    "banner_info": {},
}

_PAYLOAD_WITH_MODEL_LIMITS = {
    "type": "conversation_init",
    "default_model_slug": "gpt-6-pro",
    "model_limits": [
        {
            "model_slug": "gpt-6-pro",
            "remaining": 100,
            "limit": 500,
            "reset_after": "2026-09-07T00:00:00Z",
        },
    ],
    "limits_progress": [],
    "blocked_features": [],
}

_PAYLOAD_WITH_BLOCKED = {
    "type": "conversation_init",
    "default_model_slug": "gpt-6-pro",
    "model_limits": [],
    "limits_progress": [],
    "blocked_features": {"voice": True, "code_interpreter": True},
}

_PAYLOAD_WITH_UNKNOWN_KEYS = {
    "type": "conversation_init",
    "default_model_slug": "gpt-6-pro",
    "model_limits": [],
    "limits_progress": [],
    "blocked_features": [],
    "future_capability": {"state": "enabled"},
    "experimental_flag": True,
    "atlas_mode_enabled": False,
    "banner_info": {"some_key": "some_value"},
}


# ---------------------------------------------------------------------------
# Contract tests
# ---------------------------------------------------------------------------

def test_contract_is_post_no_body() -> None:
    contract = conversation_init_request_contract()
    assert contract["method"] == "POST"
    assert contract["body_omitted"] is True
    assert contract["body"] is None
    assert contract["has_model_message"] is False
    assert contract["has_conversation_content"] is False


def test_contract_url_defaults_correctly() -> None:
    contract = conversation_init_request_contract()
    assert contract["url"] == CHATGPT_CONVERSATION_INIT_DEFAULT_URL
    assert contract["path"] == CHATGPT_CONVERSATION_INIT_PATH


# ---------------------------------------------------------------------------
# Payload detection
# ---------------------------------------------------------------------------

def test_recognizes_valid_payload() -> None:
    assert looks_like_conversation_init_payload(_VALID_PAYLOAD) is True


def test_recognizes_payload_with_model_limits() -> None:
    assert looks_like_conversation_init_payload(_PAYLOAD_WITH_MODEL_LIMITS) is True


def test_rejects_non_mapping() -> None:
    assert looks_like_conversation_init_payload("not a dict") is False
    assert looks_like_conversation_init_payload(None) is False
    assert looks_like_conversation_init_payload([]) is False


def test_rejects_irrelevant_mapping() -> None:
    assert looks_like_conversation_init_payload({"foo": "bar"}) is False


# ---------------------------------------------------------------------------
# Account identity hashing
# ---------------------------------------------------------------------------

def test_account_hash_returns_none_for_empty_payload() -> None:
    h, fields = hash_chatgpt_conversation_init_account_identity({})
    assert h is None
    assert fields == []


def test_account_hash_does_not_return_raw_identity() -> None:
    payload = {"user": {"id": "user-12345"}}
    h, fields = hash_chatgpt_conversation_init_account_identity(payload)
    assert h is not None
    assert "user-12345" not in h
    assert "user.id" in fields


def test_source_identity_is_hashed() -> None:
    h = hash_chatgpt_conversation_init_source_identity("/home/user/collector.json")
    assert h is not None
    assert len(h) == 64
    assert "/home/user" not in h


# ---------------------------------------------------------------------------
# Boundary sanitization
# ---------------------------------------------------------------------------

def test_sanitize_strips_secret_keys() -> None:
    raw = {**_VALID_PAYLOAD, "authorization": "Bearer secret-token-12345"}
    sanitized = sanitize_conversation_init_boundary(raw)
    assert sanitized["redacted_field_count"] >= 1
    assert "authorization" not in sanitized.get("payload", {})


def test_sanitize_preserves_payload_structure() -> None:
    sanitized = sanitize_conversation_init_boundary(_with_identity(_VALID_PAYLOAD))
    assert sanitized["payload_state"] == "present"
    assert isinstance(sanitized["payload"], dict)
    assert "default_model_slug" in sanitized["payload"]


def test_sanitize_handles_non_mapping() -> None:
    sanitized = sanitize_conversation_init_boundary("invalid")
    assert sanitized["payload_state"] == "malformed"
    assert sanitized["payload"] is None


def test_sanitize_handles_empty_dict() -> None:
    sanitized = sanitize_conversation_init_boundary({})
    assert sanitized["payload_state"] == "absent"


def test_sanitize_redacts_email_values() -> None:
    raw = {**_VALID_PAYLOAD, "email": "user@example.com"}
    sanitized = sanitize_conversation_init_boundary(raw)
    # Email should be redacted from the payload
    payload = sanitized.get("payload", {})
    assert "email" not in payload


# ---------------------------------------------------------------------------
# PII / privacy tests
# ---------------------------------------------------------------------------

def test_sanitize_redacts_title_field() -> None:
    """Titles, names, and usernames must not leak into raw_provider_fields."""
    raw = {
        "type": "conversation_init",
        "default_model_slug": "gpt-6-pro",
        "model_limits": [
            {"model_slug": "gpt-6-pro", "title": "GPT-6 Pro", "display_name": "Dr. User"},
        ],
        "limits_progress": [],
        "blocked_features": [],
    }
    sanitized = sanitize_conversation_init_boundary(_with_identity(raw))
    # Parse observations
    observed = datetime.now(timezone.utc)
    observations, summary = parse_conversation_init_observations(sanitized, observed_at=observed)
    for obs in observations:
        rpf = obs.get("raw_provider_fields", {})
        projections = rpf.get("entry_projections", {})
        assert "title" not in projections, "title leaked into projections"
        assert "display_name" not in projections, "display_name leaked into projections"


def test_sanitize_redacts_username_field() -> None:
    raw = {
        "type": "conversation_init",
        "default_model_slug": "gpt-6-pro",
        "model_limits": [
            {"model_slug": "gpt-6-pro", "username": "john_doe"},
        ],
        "limits_progress": [],
        "blocked_features": [],
    }
    sanitized = sanitize_conversation_init_boundary(_with_identity(raw))
    observed = datetime.now(timezone.utc)
    observations, _summary = parse_conversation_init_observations(sanitized, observed_at=observed)
    for obs in observations:
        rpf = obs.get("raw_provider_fields", {})
        projections = rpf.get("entry_projections", {})
        assert "username" not in projections, "username leaked into projections"


def test_sanitize_redacts_workspace_field() -> None:
    raw = {
        "type": "conversation_init",
        "default_model_slug": "gpt-6-pro",
        "model_limits": [
            {"model_slug": "gpt-6-pro", "workspace": "My Personal Workspace"},
        ],
        "limits_progress": [],
        "blocked_features": [],
    }
    sanitized = sanitize_conversation_init_boundary(_with_identity(raw))
    observed = datetime.now(timezone.utc)
    observations, _summary = parse_conversation_init_observations(sanitized, observed_at=observed)
    for obs in observations:
        rpf = obs.get("raw_provider_fields", {})
        projections = rpf.get("entry_projections", {})
        assert "workspace" not in projections, "workspace leaked into projections"


def test_sanitize_redacts_feature_note() -> None:
    raw = {
        "type": "conversation_init",
        "default_model_slug": "gpt-6-pro",
        "model_limits": [],
        "limits_progress": [
            {"feature": "deep_research", "remaining": 250, "feature_note": "Priority access granted"},
        ],
        "blocked_features": [],
    }
    sanitized = sanitize_conversation_init_boundary(raw)
    observed = datetime.now(timezone.utc)
    observations, _summary = parse_conversation_init_observations(sanitized, observed_at=observed)
    for obs in observations:
        rpf = obs.get("raw_provider_fields", {})
        projections = rpf.get("entry_projections", {})
        assert "feature_note" not in projections, "feature_note leaked into projections"


def test_sanitize_fails_closed_for_unknown_personal_and_token_fields() -> None:
    raw = _with_identity(
        {
            "type": "conversation_init",
            "default_model_slug": "gpt-6-pro",
            "model_limits": [],
            "limits_progress": [],
            "blocked_features": [],
            "localStorage": {"private_state": "browser-state-secret"},
            "sessionToken": "session-token-secret",
            "titles": ["Private chat with Jane"],
            "usernames": ["jane_doe"],
            "workspace": "Acme Corp",
            "notes": ["private operator note"],
            "unknown_value": "Jane Doe",
            "opaque_value": "opaque-token-secret",
        }
    )
    raw["user"] = {**raw["user"], "name": "Jane Doe"}

    sanitized = sanitize_conversation_init_boundary(raw)
    serialized = json.dumps(sanitized, sort_keys=True)
    persisted_projection = json.dumps(
        {
            "payload": sanitized["payload"],
            "payload_schema": sanitized["payload_schema"],
        },
        sort_keys=True,
    )

    for secret_or_personal_value in (
        "Jane Doe",
        "browser-state-secret",
        "session-token-secret",
        "Private chat with Jane",
        "jane_doe",
        "Acme Corp",
        "private operator note",
        "opaque-token-secret",
    ):
        assert secret_or_personal_value not in serialized
    for sensitive_key in (
        "localStorage",
        "sessionToken",
        "titles",
        "usernames",
        "workspace",
        "notes",
    ):
        assert sensitive_key not in persisted_projection
    assert sanitized["payload"]["user"] == {}
    assert sanitized["redacted_field_count"] >= 7


def test_sanitize_preserves_safe_unknown_nested_values_and_changes_history() -> None:
    def payload_for(state: str, remaining: int) -> dict:
        return _with_identity(
            {
                "type": "conversation_init",
                "default_model_slug": "gpt-6-pro",
                "model_limits": [],
                "limits_progress": [
                    {
                        "feature": "deep_research",
                        "remaining": 5,
                        "future_capability": {
                            "state": state,
                            "remaining": remaining,
                            "enabled": True,
                            "sessionToken": "nested-token-secret",
                        },
                        "future_usage": {
                            "window": "rolling",
                            "used": 1,
                            "status": "active",
                        },
                    }
                ],
                "blocked_features": [],
            }
        )

    first = sanitize_conversation_init_boundary(payload_for("enabled", 4))
    second = sanitize_conversation_init_boundary(payload_for("disabled", 3))
    first_observations, _ = parse_conversation_init_observations(
        first,
        observed_at=datetime.now(timezone.utc),
    )
    second_observations, _ = parse_conversation_init_observations(
        second,
        observed_at=datetime.now(timezone.utc),
    )

    first_snapshot = first_observations[0]["raw_provider_fields"]
    second_snapshot = second_observations[0]["raw_provider_fields"]
    first_feature = next(
        row
        for row in first_observations
        if row["quota_key"] == "chatgpt_conversation_init:feature:deep_research"
    )
    second_feature = next(
        row
        for row in second_observations
        if row["quota_key"] == "chatgpt_conversation_init:feature:deep_research"
    )

    first_projection = first["payload"]["limits_progress"][0]
    assert first_projection["future_capability"] == {
        "state": "enabled",
        "remaining": 4,
        "enabled": True,
    }
    assert first_projection["future_usage"] == {
        "window": "rolling",
        "used": 1,
        "status": "active",
    }
    assert "nested-token-secret" not in json.dumps(first, sort_keys=True)
    assert (
        first_snapshot["top_level_projections"]
        != second_snapshot["top_level_projections"]
    )
    assert (
        first_feature["raw_provider_fields"]["entry_projections"]
        != second_feature["raw_provider_fields"]["entry_projections"]
    )


def test_sanitize_redacts_unknown_strings_inside_safe_contexts() -> None:
    raw = _with_identity(
        {
            "type": "conversation_init",
            "default_model_slug": "gpt-6-pro",
            "model_limits": [],
            "limits_progress": [
                {
                    "feature": "deep_research",
                    "future_capability": {
                        "owner": "jane_doe",
                        "opaque_value": "opaque-token-secret",
                        "state": "enabled",
                        "window": "rolling",
                    },
                }
            ],
            "blocked_features": [],
        }
    )

    sanitized = sanitize_conversation_init_boundary(raw)
    serialized = json.dumps(sanitized, sort_keys=True)
    capability = sanitized["payload"]["limits_progress"][0]["future_capability"]
    observations, _summary = parse_conversation_init_observations(
        sanitized,
        observed_at=datetime.now(timezone.utc),
    )
    persisted = json.dumps(observations, sort_keys=True)

    assert capability == {
        "state": "enabled",
        "window": "rolling",
    }
    assert "jane_doe" not in serialized
    assert "opaque-token-secret" not in serialized
    assert "jane_doe" not in persisted
    assert "opaque-token-secret" not in persisted


def test_sanitize_preserves_numeric_token_counters_but_redacts_token_values() -> None:
    raw = _with_identity(
        {
            "type": "conversation_init",
            "default_model_slug": "gpt-6-pro",
            "model_limits": [],
            "limits_progress": [
                {
                    "feature": "deep_research",
                    "input_tokens": 12,
                    "output_tokens": 34,
                    "token_limit": 100,
                    "sessionToken": "session-token-secret",
                    "token_metadata": {"value": "opaque-token-secret"},
                }
            ],
            "blocked_features": [],
        }
    )

    sanitized = sanitize_conversation_init_boundary(raw)
    serialized = json.dumps(sanitized, sort_keys=True)
    entry = sanitized["payload"]["limits_progress"][0]
    observations, _summary = parse_conversation_init_observations(
        sanitized,
        observed_at=datetime.now(timezone.utc),
    )
    feature = next(
        row
        for row in observations
        if row["quota_key"] == "chatgpt_conversation_init:feature:deep_research"
    )
    projections = feature["raw_provider_fields"]["entry_projections"]

    assert entry["input_tokens"] == 12
    assert entry["output_tokens"] == 34
    assert entry["token_limit"] == 100
    assert projections["input_tokens"] == 12
    assert projections["output_tokens"] == 34
    assert projections["token_limit"] == 100
    assert "sessionToken" not in entry
    assert "token_metadata" not in entry
    assert "session-token-secret" not in serialized
    assert "opaque-token-secret" not in serialized


# ---------------------------------------------------------------------------
# Parsing tests
# ---------------------------------------------------------------------------

def test_parse_limits_progress() -> None:
    sanitized = sanitize_conversation_init_boundary(_with_identity(_VALID_PAYLOAD))
    observed = datetime.now(timezone.utc)
    observations, summary = parse_conversation_init_observations(sanitized, observed_at=observed)

    assert summary["limits_progress_state"] == "present"
    assert "deep_research" in summary["discovered_feature_identities"]
    assert "image_gen" in summary["discovered_feature_identities"]
    assert summary["valid_observation_count"] >= 3  # snapshot + 2 features


def test_parse_empty_model_limits_as_unknown() -> None:
    sanitized = sanitize_conversation_init_boundary(_with_identity(_VALID_PAYLOAD))
    observed = datetime.now(timezone.utc)
    _observations, summary = parse_conversation_init_observations(sanitized, observed_at=observed)

    # Empty model_limits is unknown, not proof of unlimited
    assert summary["model_limits_state"] == "empty_unknown"


def test_parse_model_limits_with_usage() -> None:
    sanitized = sanitize_conversation_init_boundary(_with_identity(_PAYLOAD_WITH_MODEL_LIMITS))
    observed = datetime.now(timezone.utc)
    observations, summary = parse_conversation_init_observations(sanitized, observed_at=observed)

    assert summary["model_limits_state"] == "present"
    assert "gpt-6-pro" in summary["discovered_model_identities"]
    model_obs = [o for o in observations if o.get("quota_type") == "count"]
    assert len(model_obs) >= 1


def test_parse_blocked_features() -> None:
    sanitized = sanitize_conversation_init_boundary(_with_identity(_PAYLOAD_WITH_BLOCKED))
    observed = datetime.now(timezone.utc)
    observations, summary = parse_conversation_init_observations(sanitized, observed_at=observed)

    assert summary["blocked_features_state"] == "present"
    assert "voice" in summary["discovered_blocked_identities"]
    assert "code_interpreter" in summary["discovered_blocked_identities"]


def test_parse_absent_collections() -> None:
    payload = {
        "type": "conversation_init",
        "default_model_slug": "gpt-6-pro",
    }
    sanitized = sanitize_conversation_init_boundary(payload)
    observed = datetime.now(timezone.utc)
    _observations, summary = parse_conversation_init_observations(sanitized, observed_at=observed)

    assert summary["model_limits_state"] == "absent_unknown"
    assert summary["limits_progress_state"] == "absent_unknown"
    assert summary["blocked_features_state"] == "absent_unknown"


def test_parse_preserves_unknown_top_level_keys() -> None:
    sanitized = sanitize_conversation_init_boundary(_with_identity(_PAYLOAD_WITH_UNKNOWN_KEYS))
    observed = datetime.now(timezone.utc)
    observations, summary = parse_conversation_init_observations(sanitized, observed_at=observed)

    snapshot = next((o for o in observations if "snapshot" in o.get("quota_key", "")), None)
    assert snapshot is not None
    unknown = snapshot["raw_provider_fields"]["unknown_top_level_keys"]
    assert "future_capability" in unknown
    assert "experimental_flag" in unknown


# ---------------------------------------------------------------------------
# Default model identity tests
# ---------------------------------------------------------------------------

def test_parse_default_model_identities() -> None:
    sanitized = sanitize_conversation_init_boundary(_with_identity(_VALID_PAYLOAD))
    observed = datetime.now(timezone.utc)
    _observations, summary = parse_conversation_init_observations(sanitized, observed_at=observed)

    assert "gpt-6-pro" in summary["discovered_model_identities"]


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

def test_parse_malformed_payload() -> None:
    sanitized = {"payload_state": "malformed", "payload": None}
    observed = datetime.now(timezone.utc)
    observations, summary = parse_conversation_init_observations(sanitized, observed_at=observed)

    assert observations == []
    assert summary["telemetry_status"] == "malformed"


def test_parse_http_error_payload() -> None:
    sanitized = {
        "status_code": 403,
        "payload": None,
        "payload_state": "absent",
        "request": {},
    }
    observed = datetime.now(timezone.utc)
    observations, summary = parse_conversation_init_observations(sanitized, observed_at=observed)

    assert observations == []
    assert summary["telemetry_status"] == "auth"


def test_parse_non_init_payload() -> None:
    sanitized = sanitize_conversation_init_boundary({"not": "conversation_init"})
    observed = datetime.now(timezone.utc)
    observations, summary = parse_conversation_init_observations(sanitized, observed_at=observed)

    assert observations == []
    assert summary["telemetry_status"] == "malformed"


# ---------------------------------------------------------------------------
# Tuple construction
# ---------------------------------------------------------------------------

def test_build_rate_limit_tuples_structure() -> None:
    observed = datetime.now(timezone.utc)
    observations = [
        {
            "quota_key": "chatgpt_conversation_init:feature:deep_research",
            "quota_period": None,
            "quota_type": "count",
            "model": "deep_research",
            "expected_reset_at": None,
            "remaining_pct": 50.0,
            "quota_limit": 500.0,
            "quota_used": None,
            "quota_remaining": 250.0,
            "billing_period_start_at": None,
            "billing_period_end_at": None,
            "raw_provider_fields": {},
            "evidence": {},
        },
    ]
    tuples = build_conversation_init_rate_limit_tuples(
        observations,
        observed_at=observed,
        account_hash="abc123",
    )
    assert len(tuples) == 1
    t = tuples[0]
    assert t[0] == observed  # observed_at
    assert t[1] == CHATGPT_CONVERSATION_INIT_CLIENT
    assert t[3] == "abc123"  # account_hash
    assert t[4] == CHATGPT_CONVERSATION_INIT_PROVIDER
    assert t[6] == "chatgpt_conversation_init:feature:deep_research"
    assert t[18] == CHATGPT_CONVERSATION_INIT_SOURCE
    assert t[21].startswith("chatgpt-conversation-init-")


# ---------------------------------------------------------------------------
# File collector tests
# ---------------------------------------------------------------------------

def test_collect_observations_from_file(tmp_path: Path) -> None:
    source = tmp_path / "conversation-init.json"
    source.write_text(json.dumps(_VALID_PAYLOAD), encoding="utf-8")

    payloads, summary = collect_conversation_init_observations(
        str(source),
    )
    assert summary["telemetry_status"] == "valid"
    assert summary["valid_observation_count"] >= 3
    assert len(payloads) >= 3


def test_collect_observations_missing_file(tmp_path: Path) -> None:
    source = tmp_path / "nonexistent.json"

    with pytest.raises(ChatGPTConversationInitError) as exc_info:
        collect_conversation_init_observations(str(source))
    assert exc_info.value.telemetry_class == "auth"


def test_collect_observations_invalid_json(tmp_path: Path) -> None:
    source = tmp_path / "bad.json"
    source.write_text("not valid json", encoding="utf-8")

    with pytest.raises(ChatGPTConversationInitError) as exc_info:
        collect_conversation_init_observations(str(source))
    assert exc_info.value.telemetry_class == "malformed_telemetry"


class _FixtureTransport:
    def __init__(self, envelope):
        self.envelope = envelope
        self.requests = []

    def fetch(self, request: urllib_request.Request):
        self.requests.append(request)
        return self.envelope


class _ExplodingTransport:
    def fetch(self, request: urllib_request.Request):
        raise RuntimeError("transport exploded")


class _FakeOraclePage:
    def __init__(self, envelope):
        self.url = "https://chatgpt.com/"
        self.envelope = envelope
        self.evaluate_calls = []

    def evaluate(self, script, url):
        self.evaluate_calls.append((script, url))
        return self.envelope


class _FakeOracleContext:
    def __init__(self, pages):
        self.pages = pages


class _FakeOracleBrowser:
    def __init__(self, pages):
        self.contexts = [_FakeOracleContext(pages)]
        self.disconnect_calls = 0
        self.close_calls = 0

    def disconnect(self):
        self.disconnect_calls += 1

    def close(self):
        self.close_calls += 1
        raise AssertionError("attached Oracle browser must not be closed")


class _FakeOracleChromium:
    def __init__(self, browser):
        self.browser = browser
        self.connect_calls = []
        self.persistent_launch_calls = 0

    def connect_over_cdp(self, endpoint, *, timeout):
        self.connect_calls.append((endpoint, timeout))
        return self.browser

    def launch_persistent_context(self, *args, **kwargs):
        self.persistent_launch_calls += 1
        raise AssertionError("transport must not launch a persistent context")


class _FakeOraclePlaywright:
    def __init__(self, browser):
        self.chromium = _FakeOracleChromium(browser)
        self.stop_calls = 0

    def stop(self):
        self.stop_calls += 1


def test_build_request_has_no_conversation_content() -> None:
    request = conversation_init_request_contract()
    built = urllib_request.Request(request["url"], data=None, method="POST")
    assert request_has_conversation_content(built) is False


def test_oracle_transport_attaches_and_posts_without_storage_or_browser_launch() -> None:
    page = _FakeOraclePage(
        {
            "status_code": 200,
            "payload": _with_identity(_VALID_PAYLOAD),
        }
    )
    browser = _FakeOracleBrowser([page])
    playwright = _FakeOraclePlaywright(browser)
    transport = OracleBrowserConversationInitTransport(
        cdp_endpoint="http://127.0.0.1:9222",
        playwright_factory=lambda: playwright,
    )
    request = build_conversation_init_request()

    response = transport.fetch(request)

    assert response["status_code"] == 200
    assert request.get_method() == "POST"
    assert request.data is None
    assert playwright.chromium.connect_calls == [
        ("http://127.0.0.1:9222", 30000)
    ]
    assert playwright.chromium.persistent_launch_calls == 0
    assert browser.disconnect_calls == 1
    assert browser.close_calls == 0
    assert playwright.stop_calls == 1
    assert len(page.evaluate_calls) == 1
    script, evaluated_url = page.evaluate_calls[0]
    lowered_script = script.lower()
    assert evaluated_url == CHATGPT_CONVERSATION_INIT_DEFAULT_URL
    assert 'method: "post"' in lowered_script
    assert '"body"' not in lowered_script
    for forbidden_browser_access in (
        "document.cookie",
        "localstorage",
        "sessionstorage",
        "indexeddb",
        "storage_state",
        "cookies",
    ):
        assert forbidden_browser_access not in lowered_script


def test_oracle_transport_fails_closed_without_existing_chatgpt_page() -> None:
    page = _FakeOraclePage({"status_code": 200, "payload": _VALID_PAYLOAD})
    page.url = "https://example.test/"
    browser = _FakeOracleBrowser([page])
    playwright = _FakeOraclePlaywright(browser)
    transport = OracleBrowserConversationInitTransport(
        playwright_factory=lambda: playwright,
    )

    with pytest.raises(OracleBrowserBoundaryUnavailable):
        transport.fetch(build_conversation_init_request())

    assert page.evaluate_calls == []
    assert browser.disconnect_calls == 1
    assert browser.close_calls == 0
    assert playwright.stop_calls == 1


def test_oracle_transport_rejects_request_body() -> None:
    browser = _FakeOracleBrowser([])
    playwright = _FakeOraclePlaywright(browser)
    transport = OracleBrowserConversationInitTransport(
        playwright_factory=lambda: playwright,
    )
    request = urllib_request.Request(
        CHATGPT_CONVERSATION_INIT_DEFAULT_URL,
        data=b"{}",
        method="POST",
    )

    with pytest.raises(OracleBrowserBoundaryUnavailable):
        transport.fetch(request)

    assert playwright.chromium.connect_calls == []
    assert browser.disconnect_calls == 0
    assert playwright.stop_calls == 0


def test_public_oracle_browser_entry_point_uses_attach_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "conversation-init.json"
    transport = _FixtureTransport(
        {"status_code": 200, "payload": _with_identity(_VALID_PAYLOAD)}
    )
    monkeypatch.setattr(
        "litellm.llms.chatgpt.conversation_init.build_oracle_browser_conversation_init_transport",
        lambda **kwargs: transport,
    )

    summary = collect_conversation_init_snapshot_from_oracle_browser(
        str(source),
        cdp_endpoint="http://127.0.0.1:9222",
    )

    assert summary["written"] is True
    assert summary["collector_source"] == "oracle_browser_cdp_attach"
    assert summary["browser_boundary"] == "oracle_browser_cdp_attach"
    assert summary["live_authenticated_oracle_browser"] is True
    assert summary["request_body_omitted"] is True
    assert request_has_conversation_content(transport.requests[0]) is False


def test_collector_writes_sanitized_snapshot(tmp_path: Path) -> None:
    source = tmp_path / "conversation-init.json"
    transport = _FixtureTransport(
        {
            "status_code": 200,
            "headers": {"authorization": "Bearer secret-token", "cookie": "session=abc"},
            "payload": _with_identity(_VALID_PAYLOAD),
        }
    )

    summary = ChatGPTConversationInitCollector(transport).collect(str(source))
    assert summary["written"] is True
    assert summary["collector_source"] == "browser_boundary"
    assert summary["live_authenticated_oracle_browser"] is False
    assert summary["request_method"] == "POST"
    assert summary["request_body_omitted"] is True
    assert summary["has_model_message"] is False
    assert len(transport.requests) == 1
    assert request_has_conversation_content(transport.requests[0]) is False

    snapshot = json.loads(source.read_text(encoding="utf-8"))
    snapshot_text = json.dumps(snapshot)
    assert "secret-token" not in snapshot_text
    assert "session=abc" not in snapshot_text
    assert "authorization" not in snapshot_text.lower() or snapshot.get("headers") is None
    assert "headers" not in snapshot
    assert snapshot["status_code"] == 200
    assert snapshot["payload"]["default_model_slug"] == "gpt-6-pro"
    assert snapshot["account_hash"]
    assert snapshot["account_hash"] != "user-test-0001"
    assert "user-test-0001" not in snapshot_text

    payloads, parse_summary = collect_conversation_init_observations(str(source))
    assert parse_summary["telemetry_status"] == "valid"
    assert parse_summary["account_identity_hashed"] is True
    assert len(payloads) >= 3
    raw_fields = json.loads(payloads[0][16])
    assert "observed_at" not in raw_fields


def test_collector_redacts_personal_fields_from_written_snapshot(tmp_path: Path) -> None:
    source = tmp_path / "conversation-init.json"
    payload = _with_identity(
        {
            **_VALID_PAYLOAD,
            "title": "Private chat with Jane",
            "username": "jane.doe",
            "workspace": "Acme Corp",
        }
    )
    transport = _FixtureTransport({"status_code": 200, "payload": payload})

    summary = collect_conversation_init_snapshot(
        str(source),
        transport=transport,
    )
    assert summary["written"] is True
    snapshot_text = source.read_text(encoding="utf-8")
    assert "Private chat with Jane" not in snapshot_text
    assert "jane.doe" not in snapshot_text
    assert "Acme Corp" not in snapshot_text


def test_collector_refuses_symlink_destination(tmp_path: Path) -> None:
    real = tmp_path / "real.json"
    real.write_text("{}", encoding="utf-8")
    link = tmp_path / "conversation-init.json"
    link.symlink_to(real)
    transport = _FixtureTransport(
        {"status_code": 200, "payload": _with_identity(_VALID_PAYLOAD)}
    )

    with pytest.raises(ChatGPTConversationInitError):
        write_conversation_init_snapshot(str(link), {"payload": {}})
    summary = collect_conversation_init_snapshot(str(link), transport=transport)
    assert summary["written"] is False
    assert summary["telemetry_class"] == "malformed_telemetry"
    assert real.read_text(encoding="utf-8") == "{}"


def test_collector_keeps_last_good_snapshot_on_http_error(tmp_path: Path) -> None:
    source = tmp_path / "conversation-init.json"
    good = _FixtureTransport(
        {"status_code": 200, "payload": _with_identity(_VALID_PAYLOAD)}
    )
    first = collect_conversation_init_snapshot(str(source), transport=good)
    assert first["written"] is True
    original = source.read_text(encoding="utf-8")

    failing = _FixtureTransport(
        {"status_code": 403, "payload": {"detail": "Invalid conversation init"}}
    )
    second = collect_conversation_init_snapshot(str(source), transport=failing)
    assert second["written"] is False
    assert second["status_code"] == 403
    assert second["telemetry_class"] == "auth"
    assert second["last_good_state_retained"] is True
    assert source.read_text(encoding="utf-8") == original


def test_collector_keeps_last_good_snapshot_on_transport_failure(tmp_path: Path) -> None:
    source = tmp_path / "conversation-init.json"
    good = _FixtureTransport(
        {"status_code": 200, "payload": _with_identity(_VALID_PAYLOAD)}
    )
    collect_conversation_init_snapshot(str(source), transport=good)
    original = source.read_text(encoding="utf-8")

    second = collect_conversation_init_snapshot(
        str(source),
        transport=_ExplodingTransport(),
    )
    assert second["written"] is False
    assert second["last_good_state_retained"] is True
    assert source.read_text(encoding="utf-8") == original


def test_collector_does_not_claim_live_oracle_browser(tmp_path: Path) -> None:
    source = tmp_path / "conversation-init.json"
    transport = _FixtureTransport(
        {"status_code": 200, "payload": _with_identity(_VALID_PAYLOAD)}
    )
    summary = collect_conversation_init_snapshot(str(source), transport=transport)
    assert summary["live_authenticated_oracle_browser"] is False
