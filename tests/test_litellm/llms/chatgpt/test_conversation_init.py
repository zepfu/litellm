"""Focused tests for the ChatGPT conversation-init observer."""

from __future__ import annotations

import json
import time
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
    classify_native_history_request,
    observe_native_history_from_injected_events,
    hash_chatgpt_conversation_init_canonical_account_id,
    _chatgpt_browser_fetch_enable_params,
    _observe_native_history_oracle_page,
    _observe_native_oracle_init,
    _oracle_browser_history_observation_worker,
    _oracle_browser_safe_error_class,
    _oracle_browser_worker_error_suffix,
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


_TEST_ACCOUNT_ID = "acct-test-bound"
_TEST_PAGE_TARGET_ID = "page-target-1"
_TEST_OWNED_TARGET_ID = "owned-target-1"


def _test_account_hash() -> str:
    hashed = hash_chatgpt_conversation_init_canonical_account_id(_TEST_ACCOUNT_ID)
    assert hashed is not None
    return hashed


class _FakeOracleExpectPage:
    def __init__(self, page):
        self.value = page

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False


class _FakeCdpSession:
    def __init__(self, page):
        self.page = page
        self.handlers = {}
        self.sends = []
        self._bodies: dict[str, str] = {}
        self._streams: dict[str, str] = {}

    def on(self, event, handler):
        self.handlers[event] = handler

    def send(self, method, params=None):
        self.sends.append((method, params or {}))
        remaining_errors = getattr(self.page, "fetch_enable_errors", 0)
        if method == "Fetch.enable" and remaining_errors:
            self.page.fetch_enable_errors = remaining_errors - 1
            raise getattr(self.page, "fetch_enable_error") or RuntimeError(
                "Fetch.enable failed"
            )
        remaining_body_errors = getattr(self.page, "get_response_body_errors", 0)
        if method == "Network.getResponseBody" and remaining_body_errors:
            self.page.get_response_body_errors = remaining_body_errors - 1
            raise getattr(self.page, "get_response_body_error") or RuntimeError(
                "Network.getResponseBody failed"
            )
        if method == "Page.navigate":
            url = (params or {}).get("url") or self.page.url
            self.page.goto_calls.append(url)
            self.page.url = url
            if not getattr(self.page, "skip_goto_emit", False):
                if getattr(self.page, "history_mode", False):
                    if str(url).startswith("https://chatgpt.com/c/"):
                        self.emit_native_history_detail()
                    elif not getattr(self.page, "_native_emit_done", False):
                        self.page._native_emit_done = True
                        self.emit_native_history_index()
                elif not getattr(self.page, "_native_emit_done", False):
                    self.page._native_emit_done = True
                    self.emit_native_init()
            navigate_error = getattr(self.page, "goto_error", None)
            if navigate_error is not None:
                raise navigate_error
            return {}
        if method == "Network.getResponseBody":
            request_id = (params or {}).get("requestId")
            stored = self._bodies.get(str(request_id))
            if stored is not None:
                return {"body": stored, "base64Encoded": False}
            payload = self.page.envelope.get("payload", self.page.envelope)
            return {"body": json.dumps(payload), "base64Encoded": False}
        if method == "Fetch.takeResponseBodyAsStream":
            request_id = (params or {}).get("requestId")
            handle = f"stream-{request_id}"
            self._streams[handle] = self._bodies.get(str(request_id), "{}")
            return {"stream": handle}
        if method == "IO.read":
            handle = (params or {}).get("handle")
            data = self._streams.get(str(handle), "")
            self._streams[handle] = ""
            return {"data": data, "eof": True, "base64Encoded": False}
        if method == "IO.close":
            return {}
        if method == "Target.getTargetInfo":
            target_id = (params or {}).get("targetId") or self.page.target_id
            return {
                "targetInfo": {
                    "targetId": target_id,
                    "browserContextId": "ctx-1",
                }
            }
        if method == "Target.createTarget":
            owned = self.page.context._owned
            if owned is not None:
                owned.url = (params or {}).get("url") or owned.url
                owned.target_id = _TEST_OWNED_TARGET_ID
            return {"targetId": _TEST_OWNED_TARGET_ID}
        if method == "Target.closeTarget":
            return {"success": True}
        return {}

    def emit_native_history(self) -> None:
        self.emit_native_history_index()
        self.emit_native_history_detail()

    def emit_native_history_index(self) -> None:
        headers = {"chatgpt-account-id": _TEST_ACCOUNT_ID}
        paused = self.handlers.get("Fetch.requestPaused")
        will = self.handlers.get("Network.requestWillBeSent")
        extra = self.handlers.get("Network.requestWillBeSentExtraInfo")
        if paused is not None:
            paused(
                {
                    "requestId": "doc-third-party-1",
                    "request": {
                        "url": "https://challenges.cloudflare.com/cdn-cgi/challenge-platform/h/g/orchestrate/jsch/v1",
                        "method": "GET",
                    },
                    "resourceType": "Document",
                }
            )
            paused(
                {
                    "requestId": "doc-www-1",
                    "request": {
                        "url": "https://www.chatgpt.com/",
                        "method": "GET",
                    },
                    "resourceType": "Document",
                    "redirectedRequestId": "doc-home-redirect",
                }
            )
            paused(
                {
                    "requestId": "doc-cf-1",
                    "request": {
                        "url": "https://chatgpt.com/cdn-cgi/challenge-platform/h/g/orchestrate/jsch/v1",
                        "method": "GET",
                    },
                    "resourceType": "Document",
                    "redirectedRequestId": "doc-home-redirect",
                }
            )
            paused(
                {
                    "requestId": "doc-1",
                    "request": {
                        "url": "https://chatgpt.com/",
                        "method": "GET",
                    },
                    "resourceType": "Document",
                }
            )
            paused(
                {
                    "requestId": "doc-spa-1",
                    "request": {
                        "url": "https://chatgpt.com/c/conv-001",
                        "method": "GET",
                    },
                    "resourceType": "Document",
                }
            )
            paused(
                {
                    "requestId": "opt-index",
                    "request": {
                        "url": "https://chatgpt.com/backend-api/conversations?offset=0&limit=100",
                        "method": "OPTIONS",
                    },
                    "resourceType": "XHR",
                }
            )
            paused(
                {
                    "requestId": "post-index",
                    "request": {
                        "url": "https://chatgpt.com/backend-api/conversations",
                        "method": "POST",
                    },
                    "resourceType": "XHR",
                }
            )
            paused(
                {
                    "requestId": "resp-conversation",
                    "networkId": "net-conversation",
                    "request": {
                        "url": "https://chatgpt.com/backend-api/conversation",
                        "method": "POST",
                    },
                    "responseStatusCode": 200,
                    "resourceType": "XHR",
                }
            )
            paused(
                {
                    "requestId": "init-home",
                    "request": {
                        "url": "https://chatgpt.com/backend-api/conversation/init",
                        "method": "POST",
                        "hasPostData": False,
                    },
                    "resourceType": "XHR",
                }
            )
            paused(
                {
                    "requestId": "init-spa",
                    "request": {
                        "url": "https://chatgpt.com/backend-api/f/conversation/init",
                        "method": "POST",
                        "hasPostData": False,
                    },
                    "resourceType": "XHR",
                }
            )
        index_payload = {
            "items": [
                {
                    "id": "conv-001",
                    "create_time": "2026-09-05T10:00:00Z",
                    "update_time": "2026-09-05T12:00:00Z",
                    "surface": "chat",
                }
            ],
            "total": 1,
            "limit": 100,
            "offset": 0,
        }
        detail_payload = {
            "id": "conv-001",
            "mapping": {
                "node-001": {
                    "id": "node-001",
                    "message": {
                        "id": "msg-001",
                        "author": {"role": "user"},
                        "create_time": "2026-09-05T10:00:00Z",
                        "metadata": {
                            "requested_model": "gpt-5.6-astra-pro",
                            "surface": "chat",
                        },
                        "end_turn": True,
                        "status": "finished_successfully",
                    },
                    "parent": None,
                    "children": ["node-002"],
                },
                "node-002": {
                    "id": "node-002",
                    "message": {
                        "id": "msg-002",
                        "author": {"role": "assistant"},
                        "create_time": "2026-09-05T10:00:05Z",
                        "metadata": {
                            "model_slug": "gpt-5.6-astra-pro",
                            "generation_id": "gen-abc-001",
                            "request_id": "req-abc-001",
                            "surface": "chat",
                        },
                        "end_turn": True,
                        "status": "finished_successfully",
                    },
                    "parent": "node-001",
                    "children": [],
                },
            },
        }

        def emit_history_get(
            *,
            fetch_id: str,
            network_id: str,
            url: str,
            payload: dict,
            phase: str,
        ) -> None:
            self._bodies[fetch_id] = json.dumps(payload)
            self._bodies[network_id] = json.dumps(payload)
            request = {
                "url": url,
                "method": "GET",
                "headers": headers,
                "hasPostData": False,
            }
            if phase in {"request", "both"}:
                if paused is not None:
                    paused(
                        {
                            "requestId": fetch_id,
                            "networkId": network_id,
                            "request": request,
                            "resourceType": "XHR",
                        }
                    )
                if will is not None:
                    will({"requestId": network_id, "request": request})
                if extra is not None and not getattr(
                    self.page, "skip_history_extra", False
                ):
                    extra({"requestId": network_id, "headers": headers})
                received = self.handlers.get("Network.responseReceived")
                if received is not None:
                    received(
                        {
                            "requestId": network_id,
                            "type": "XHR",
                            "response": {
                                "url": url,
                                "status": 200,
                                "headers": {"content-type": "application/json"},
                            },
                        }
                    )
                finished = self.handlers.get("Network.loadingFinished")
                if finished is not None:
                    finished({"requestId": network_id})
            if phase in {"response", "both"} and paused is not None:
                paused(
                    {
                        "requestId": fetch_id,
                        "networkId": network_id,
                        "request": request,
                        "responseStatusCode": 200,
                        "responseHeaders": [
                            {"name": "content-type", "value": "application/json"}
                        ],
                        "resourceType": "XHR",
                    }
                )

        self._history_index_payload = index_payload
        self._history_detail_payload = detail_payload
        self._emit_history_get = emit_history_get
        emit_history_get(
            fetch_id="fetch-index",
            network_id="net-index",
            url="https://chatgpt.com/backend-api/conversations?offset=0&limit=100",
            payload=index_payload,
            phase="request",
        )
        emit_history_get(
            fetch_id="fetch-index",
            network_id="net-index",
            url="https://chatgpt.com/backend-api/conversations?offset=0&limit=100",
            payload=index_payload,
            phase="response",
        )

    def emit_native_history_detail(self) -> None:
        if getattr(self, "_history_detail_emitted", False):
            return
        self._history_detail_emitted = True
        emit_history_get = getattr(self, "_emit_history_get", None)
        detail_payload = getattr(self, "_history_detail_payload", None)
        if emit_history_get is None or detail_payload is None:
            self.emit_native_history_index()
            emit_history_get = self._emit_history_get
            detail_payload = self._history_detail_payload
        emit_history_get(
            fetch_id="fetch-detail",
            network_id="net-detail",
            url="https://chatgpt.com/backend-api/conversations/conv-001",
            payload=detail_payload,
            phase="request",
        )
        emit_history_get(
            fetch_id="fetch-detail",
            network_id="net-detail",
            url="https://chatgpt.com/backend-api/conversations/conv-001",
            payload=detail_payload,
            phase="response",
        )

    def emit_native_init(self) -> None:
        headers = {"chatgpt-account-id": _TEST_ACCOUNT_ID}
        init_url = CHATGPT_CONVERSATION_INIT_DEFAULT_URL
        request_id = "req-init-1"
        paused = self.handlers.get("Fetch.requestPaused")
        if paused is not None:
            paused(
                {
                    "requestId": "doc-third-party-1",
                    "request": {
                        "url": "https://challenges.cloudflare.com/cdn-cgi/challenge-platform/h/g/orchestrate/jsch/v1",
                        "method": "GET",
                    },
                    "resourceType": "Document",
                }
            )
            paused(
                {
                    "requestId": "doc-www-1",
                    "request": {
                        "url": "https://www.chatgpt.com/",
                        "method": "GET",
                    },
                    "resourceType": "Document",
                    "redirectedRequestId": "doc-home-redirect",
                }
            )
            paused(
                {
                    "requestId": "doc-cf-1",
                    "request": {
                        "url": "https://chatgpt.com/cdn-cgi/challenge-platform/h/g/orchestrate/jsch/v1",
                        "method": "GET",
                    },
                    "resourceType": "Document",
                    "redirectedRequestId": "doc-home-redirect",
                }
            )
            paused(
                {
                    "requestId": "doc-1",
                    "request": {
                        "url": "https://chatgpt.com/",
                        "method": "GET",
                    },
                    "resourceType": "Document",
                }
            )
            paused(
                {
                    "requestId": "doc-spa-1",
                    "request": {
                        "url": "https://chatgpt.com/c/conv-001",
                        "method": "GET",
                    },
                    "resourceType": "Document",
                }
            )
            paused(
                {
                    "requestId": request_id,
                    "request": {
                        "url": init_url,
                        "method": "POST",
                        "headers": headers,
                        "hasPostData": False,
                    },
                    "resourceType": "XHR",
                }
            )
        will = self.handlers.get("Network.requestWillBeSent")
        if will is not None:
            will(
                {
                    "requestId": request_id,
                    "request": {
                        "url": init_url,
                        "method": "POST",
                        "headers": headers,
                        "hasPostData": False,
                    },
                }
            )
        extra = self.handlers.get("Network.requestWillBeSentExtraInfo")
        if extra is not None:
            extra({"requestId": request_id, "headers": headers})
        received = self.handlers.get("Network.responseReceived")
        if received is not None:
            received(
                {
                    "requestId": request_id,
                    "type": "XHR",
                    "response": {
                        "url": init_url,
                        "status": int(self.page.envelope.get("status_code") or 200),
                        "headers": {},
                    },
                }
            )
        finished = self.handlers.get("Network.loadingFinished")
        if finished is not None:
            finished({"requestId": request_id})


class _FakeOraclePage:
    def __init__(
        self,
        envelope,
        *,
        url: str = "https://chatgpt.com/",
        target_id: str = _TEST_PAGE_TARGET_ID,
    ):
        self.url = url
        self.envelope = envelope
        self.target_id = target_id
        self.evaluate_calls = []
        self.goto_calls = []
        self.context = None
        self._session = None

    def evaluate(self, script, *args):
        self.evaluate_calls.append((script, args[0] if args else None))
        if args:
            return self.envelope
        return False

    def goto(self, url, wait_until=None, timeout=None):
        del wait_until, timeout
        self.goto_calls.append(url)
        self.url = url
        if self._session is not None:
            self._session.sends.append(("Page.goto", {"url": url}))
            if not getattr(self, "skip_goto_emit", False):
                if getattr(self, "history_mode", False):
                    self._session.emit_native_history()
                else:
                    self._session.emit_native_init()
        goto_error = getattr(self, "goto_error", None)
        if goto_error is not None:
            raise goto_error

    def wait_for_timeout(self, _ms):
        if self._session is not None and getattr(self, "emit_on_wait", False):
            self.emit_on_wait = False
            if getattr(self, "history_mode", False):
                self._session.emit_native_history()
            else:
                self._session.emit_native_init()
        return None


class _FakeOracleContext:
    def __init__(self, pages):
        self.pages = list(pages)
        self._owned = None
        for page in self.pages:
            page.context = self

    def new_cdp_session(self, page):
        remaining = getattr(self, "owned_cdp_errors", 0)
        if remaining and page is self._owned:
            self.owned_cdp_errors = remaining - 1
            raise getattr(self, "owned_cdp_error") or RuntimeError(
                "owned CDP session failed"
            )
        session = _FakeCdpSession(page)
        page._session = session
        return session

    def expect_page(self, predicate=None, timeout=None):
        del predicate, timeout
        source = self.pages[0] if self.pages else _FakeOraclePage({})
        owned = _FakeOraclePage(
            source.envelope,
            url="about:blank",
            target_id=_TEST_OWNED_TARGET_ID,
        )
        owned.context = self
        self.pages.append(owned)
        self._owned = owned
        return _FakeOracleExpectPage(owned)


class _FakeOracleBrowser:
    def __init__(self, pages):
        self.contexts = [_FakeOracleContext(pages)]
        self.disconnect_calls = 0
        self.close_calls = 0

    def new_browser_cdp_session(self):
        return _FakeCdpSession(self.contexts[0].pages[0])

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
        page_target_id=_TEST_PAGE_TARGET_ID,
        expected_account_hash=_test_account_hash(),
        playwright_factory=lambda: playwright,
    )
    request = build_conversation_init_request()

    response = transport.fetch(request)

    assert response["status_code"] == 200
    assert request.get_method() == "POST"
    assert request.data is None
    native = response["native_capture"]
    assert native["request_method"] == "POST"
    assert native["request_body_omitted"] is True
    assert native["account_hash"] == _test_account_hash()
    assert native["identity_source"] == "native_request_header"
    serialized = json.dumps(response)
    assert "cookie" not in serialized.lower()
    assert "authorization" not in serialized.lower()
    assert _TEST_ACCOUNT_ID not in serialized
    assert "Bearer" not in serialized
    assert playwright.chromium.persistent_launch_calls == 0
    assert browser.close_calls == 0


def test_oracle_capture_worker_retries_owned_cdp_session_error() -> None:
    class Error(Exception):
        pass

    page = _FakeOraclePage(
        {
            "status_code": 200,
            "payload": _with_identity(_VALID_PAYLOAD),
        }
    )
    browser = _FakeOracleBrowser([page])
    browser.contexts[0].owned_cdp_errors = 1
    browser.contexts[0].owned_cdp_error = Error(
        "Protocol error (Target.attachToTarget): session closed"
    )
    playwright = _FakeOraclePlaywright(browser)
    transport = OracleBrowserConversationInitTransport(
        cdp_endpoint="http://127.0.0.1:9222",
        page_target_id=_TEST_PAGE_TARGET_ID,
        expected_account_hash=_test_account_hash(),
        playwright_factory=lambda: playwright,
        timeout_seconds=2.0,
    )
    response = transport.fetch(build_conversation_init_request())
    assert response["status_code"] == 200
    native = response["native_capture"]
    assert native["account_hash"] == _test_account_hash()
    assert native["request_method"] == "POST"
    serialized = json.dumps(response)
    assert "cookie" not in serialized.lower()
    assert "authorization" not in serialized.lower()
    assert _TEST_ACCOUNT_ID not in serialized


def test_oracle_transport_fails_closed_without_existing_chatgpt_page() -> None:
    page = _FakeOraclePage({"status_code": 200, "payload": _VALID_PAYLOAD})
    page.url = "https://example.test/"
    browser = _FakeOracleBrowser([page])
    playwright = _FakeOraclePlaywright(browser)
    transport = OracleBrowserConversationInitTransport(
        page_target_id=_TEST_PAGE_TARGET_ID,
        expected_account_hash=_test_account_hash(),
        playwright_factory=lambda: playwright,
    )

    with pytest.raises(OracleBrowserBoundaryUnavailable):
        transport.fetch(build_conversation_init_request())

    assert browser.close_calls == 0
    assert playwright.chromium.persistent_launch_calls == 0


def test_oracle_transport_rejects_request_body() -> None:
    browser = _FakeOracleBrowser([])
    playwright = _FakeOraclePlaywright(browser)
    transport = OracleBrowserConversationInitTransport(
        page_target_id=_TEST_PAGE_TARGET_ID,
        expected_account_hash=_test_account_hash(),
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
        {
            "status_code": 200,
            "payload": {
                **_with_identity(_VALID_PAYLOAD),
                "account_id": _TEST_ACCOUNT_ID,
            },
            "native_capture": {
                "account_hash": _test_account_hash(),
                "identity_source": "native_request_header",
                "selector_evidence": "request_and_extra_info",
                "request_response_correlated": True,
                "request_method": "POST",
                "request_body_omitted": True,
                "browser_challenge": False,
            },
        }
    )
    monkeypatch.setattr(
        "litellm.llms.chatgpt.conversation_init.build_oracle_browser_conversation_init_transport",
        lambda **kwargs: transport,
    )

    summary = collect_conversation_init_snapshot_from_oracle_browser(
        str(source),
        cdp_endpoint="http://127.0.0.1:9222",
        page_target_id=_TEST_PAGE_TARGET_ID,
        expected_account_hash=_test_account_hash(),
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


def test_native_history_unknown_get_bootstrap_continues_and_mutations_abort() -> None:
    account_hash = _test_account_hash()
    headers = {"chatgpt-account-id": _TEST_ACCOUNT_ID}
    observation = observe_native_history_from_injected_events(
        [
            {
                "method": "GET",
                "url": "https://chatgpt.com/backend-api/me",
                "resourceType": "XHR",
                "headers": headers,
            },
            {
                "method": "POST",
                "url": "https://chatgpt.com/backend-api/sentinel/chat-requirements",
                "resourceType": "XHR",
                "headers": headers,
            },
            {
                "method": "GET",
                "url": "https://chatgpt.com/backend-api/conversations?offset=0&limit=100",
                "resourceType": "XHR",
                "headers": headers,
                "status": 200,
                "payload": {
                    "items": [
                        {
                            "id": "conv-001",
                            "create_time": "2026-09-05T10:00:00Z",
                            "update_time": "2026-09-05T12:00:00Z",
                            "surface": "chat",
                        }
                    ],
                    "total": 1,
                    "limit": 100,
                    "offset": 0,
                },
            },
            {
                "method": "GET",
                "url": "https://chatgpt.com/backend-api/conversations/conv-001",
                "resourceType": "XHR",
                "headers": headers,
                "status": 200,
                "payload": {
                    "id": "conv-001",
                    "mapping": {
                        "node-001": {
                            "id": "node-001",
                            "message": {
                                "id": "msg-001",
                                "author": {"role": "user"},
                                "create_time": "2026-09-05T10:00:00Z",
                                "metadata": {
                                    "requested_model": "gpt-5.6-astra-pro",
                                    "surface": "chat",
                                },
                                "end_turn": True,
                                "status": "finished_successfully",
                            },
                            "parent": None,
                            "children": ["node-002"],
                        },
                        "node-002": {
                            "id": "node-002",
                            "message": {
                                "id": "msg-002",
                                "author": {"role": "assistant"},
                                "create_time": "2026-09-05T10:00:05Z",
                                "metadata": {
                                    "model_slug": "gpt-5.6-astra-pro",
                                    "generation_id": "gen-abc-001",
                                    "request_id": "req-abc-001",
                                    "surface": "chat",
                                },
                                "end_turn": True,
                                "status": "finished_successfully",
                            },
                            "parent": "node-001",
                            "children": [],
                        },
                    },
                },
            },
        ],
        expected_account_hash=account_hash,
    )
    assert observation["account_identity_verified"] is True
    assert observation["observation_state"] == "history_observed"
    assert observation["request_body_omitted"] is True
    assert observation["blocked_request"]["branch"] == "mutating_method"
    assert observation["blocked_request"]["method"] == "POST"
    pages = observation["history_pages"]
    assert any(page["route_class"] == "modern_history_index" for page in pages)
    assert any(page["route_class"] == "modern_conversation_detail" for page in pages)
    detail = next(
        page for page in pages if page["route_class"] == "modern_conversation_detail"
    )
    assert detail["conversation_id"] == "conv-001"
    assert {item["message_id"] for item in detail["messages"]} == {"msg-001", "msg-002"}
    serialized = json.dumps(observation)
    assert "cookie" not in serialized.lower()
    assert "authorization" not in serialized.lower()
    assert _TEST_ACCOUNT_ID not in serialized
    assert "Bearer" not in serialized

    bootstrap = classify_native_history_request(
        method="GET",
        url="https://chatgpt.com/backend-api/me",
        resource_type="XHR",
    )
    assert bootstrap["action"] == "continue"
    assert bootstrap["stops_observation"] is False
    mutation = classify_native_history_request(
        method="POST",
        url="https://chatgpt.com/backend-api/conversation",
        resource_type="XHR",
    )
    assert mutation["action"] == "fail"
    assert mutation["stops_observation"] is True
    unsafe = classify_native_history_request(
        method="GET",
        url="https://chatgpt.com/backend-api/conversations/conv-001/delete",
        resource_type="XHR",
    )
    assert unsafe["action"] == "fail"
    assert unsafe["stops_observation"] is True
    preflight = classify_native_history_request(
        method="OPTIONS",
        url="https://chatgpt.com/backend-api/conversations?offset=0&limit=100",
        resource_type="XHR",
    )
    assert preflight["action"] == "fail"
    assert preflight["stops_observation"] is False
    create_index = classify_native_history_request(
        method="POST",
        url="https://chatgpt.com/backend-api/conversations",
        resource_type="XHR",
    )
    assert create_index["action"] == "fail"
    assert create_index["stops_observation"] is False


def test_native_history_oracle_observer_admits_index_then_detail() -> None:
    account_hash = _test_account_hash()
    page = _FakeOraclePage(
        {
            "status_code": 200,
            "payload": _with_identity(_VALID_PAYLOAD),
        }
    )
    page.history_mode = True
    _FakeOracleContext([page])
    session = page.context.new_cdp_session(page)
    observation = _observe_native_history_oracle_page(
        page,
        session=session,
        expected_account_hash=account_hash,
        deadline=time.monotonic() + 5.0,
        max_response_bytes=1_000_000,
    )
    assert observation["account_identity_verified"] is True
    assert observation["observation_state"] == "history_observed"
    assert observation.get("failure_reason") is None
    methods = [method for method, _params in session.sends]
    assert "Fetch.enable" in methods
    assert "Page.navigate" in methods
    enable = next(params for method, params in session.sends if method == "Fetch.enable")
    patterns = enable["patterns"]
    url_patterns = [str(pattern.get("urlPattern", "")) for pattern in patterns]
    assert url_patterns
    assert all("/backend-api/" in url for url in url_patterns)
    assert not any(url.endswith("/backend-api/*") for url in url_patterns)
    assert any("/backend-api/conversation" in url for url in url_patterns)
    pages = observation["history_pages"]
    assert any(page["route_class"] == "modern_history_index" for page in pages)
    assert any(page["route_class"] == "modern_conversation_detail" for page in pages)
    fails = [
        params.get("requestId")
        for method, params in session.sends
        if method == "Fetch.failRequest"
    ]
    assert "opt-index" in fails
    assert "post-index" in fails
    assert "resp-conversation" in fails
    continues = [
        params.get("requestId")
        for method, params in session.sends
        if method == "Fetch.continueRequest"
    ]
    assert "init-home" in continues
    assert "init-spa" in continues
    assert "resp-conversation" in fails
    serialized = json.dumps(observation)
    assert "cookie" not in serialized.lower()
    assert "authorization" not in serialized.lower()
    assert _TEST_ACCOUNT_ID not in serialized
    assert "Bearer" not in serialized


def test_native_history_oracle_observer_reads_bodies_without_extrainfo() -> None:
    account_hash = _test_account_hash()
    page = _FakeOraclePage(
        {
            "status_code": 200,
            "payload": _with_identity(_VALID_PAYLOAD),
        }
    )
    page.history_mode = True
    page.skip_history_extra = True
    _FakeOracleContext([page])
    session = page.context.new_cdp_session(page)
    observation = _observe_native_history_oracle_page(
        page,
        session=session,
        expected_account_hash=account_hash,
        deadline=time.monotonic() + 5.0,
        max_response_bytes=1_000_000,
    )
    assert observation["observation_state"] == "history_observed"
    pages = observation["history_pages"]
    assert any(page["route_class"] == "modern_history_index" for page in pages)
    assert any(page["route_class"] == "modern_conversation_detail" for page in pages)
    serialized = json.dumps(observation)
    assert "cookie" not in serialized.lower()
    assert "authorization" not in serialized.lower()
    assert _TEST_ACCOUNT_ID not in serialized


def test_oracle_capture_worker_surfaces_safe_error_class_without_secrets() -> None:
    class _FailingPlaywright:
        class chromium:
            @staticmethod
            def connect_over_cdp(*_args, **_kwargs):
                raise RuntimeError("secret-cookie=abc authorization=Bearer xyz")

        def stop(self):
            return None

    transport = OracleBrowserConversationInitTransport(
        cdp_endpoint="http://127.0.0.1:9222",
        page_target_id=_TEST_PAGE_TARGET_ID,
        expected_account_hash=_test_account_hash(),
        playwright_factory=lambda: _FailingPlaywright(),
        timeout_seconds=2.0,
    )
    with pytest.raises(OracleBrowserBoundaryUnavailable) as raised:
        transport.fetch(build_conversation_init_request())
    message = str(raised.value)
    assert "RuntimeError" in message
    assert "cdp_connect" in message
    assert "cookie" not in message.lower()
    assert "authorization" not in message.lower()
    assert "Bearer" not in message
    assert "xyz" not in message


def test_oracle_browser_safe_error_class_maps_playwright_abort_and_timeout() -> None:
    class Error(Exception):
        pass

    class TimeoutError(Exception):
        pass

    class TargetClosedError(Exception):
        pass

    assert (
        _oracle_browser_safe_error_class(Error("Page.goto: net::ERR_ABORTED"))
        == "net_err_aborted"
    )
    assert (
        _oracle_browser_safe_error_class(
            Error("Page.goto: ERR_ABORTED at https://chatgpt.com/")
        )
        == "net_err_aborted"
    )
    assert (
        _oracle_browser_safe_error_class(TimeoutError("Timeout 20000ms exceeded"))
        == "timeout"
    )
    assert (
        _oracle_browser_safe_error_class(Error("Timeout 20000ms exceeded")) == "timeout"
    )
    assert (
        _oracle_browser_safe_error_class(
            TargetClosedError("Target page, context or browser has been closed")
        )
        == "target_closed"
    )
    leaked = _oracle_browser_safe_error_class(
        Error("secret-cookie=abc authorization=Bearer xyz")
    )
    assert leaked == "Error"
    assert "cookie" not in leaked.lower()
    assert "Bearer" not in leaked
    suffix = _oracle_browser_worker_error_suffix(
        {
            "ok": False,
            "error_class": "Error",
            "error_stage": "cdp_connect",
            "error_message": "secret-cookie=abc authorization=Bearer xyz",
        }
    )
    assert suffix == " (Error at cdp_connect)"
    assert "cookie" not in suffix.lower()
    assert "Bearer" not in suffix
    assert _oracle_browser_worker_error_suffix(
        {"ok": False, "error_class": "Error", "error_stage": "cookie=secret"}
    ) == " (Error)"
    assert (
        _oracle_browser_safe_error_class(
            Error("Execution context was destroyed, most likely because of a navigation.")
        )
        == "execution_context_destroyed"
    )
    assert (
        _oracle_browser_safe_error_class(Error("Navigation interrupted by another navigation"))
        == "navigation_interrupted"
    )
    assert (
        _oracle_browser_safe_error_class(
            Error("Protocol error (Network.getResponseBody): No resource with given identifier found")
        )
        == "response_body_unavailable"
    )
    assert (
        _oracle_browser_safe_error_class(
            Error("Protocol error (Fetch.enable): session closed")
        )
        == "protocol_error"
    )


def test_history_observation_worker_surfaces_safe_error_class_without_secrets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FailingPlaywright:
        class chromium:
            @staticmethod
            def connect_over_cdp(*_args, **_kwargs):
                raise RuntimeError("secret-cookie=abc authorization=Bearer xyz")

        def stop(self):
            return None

    class _Value:
        def __init__(self, value: object = 0) -> None:
            self.value = value

    class _Event:
        def __init__(self, *, set_now: bool = False) -> None:
            self._set = set_now

        def is_set(self) -> bool:
            return self._set

        def set(self) -> None:
            self._set = True

        def wait(self, timeout: object = None) -> bool:
            del timeout
            return self._set

    class _Sender:
        def close(self) -> None:
            return None

    captured: list[dict] = []
    monkeypatch.setattr(
        "litellm.llms.chatgpt.conversation_init._enter_oracle_browser_worker_process_group",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        "litellm.llms.chatgpt.conversation_init._start_playwright_from_factory",
        lambda *_args, **_kwargs: _FailingPlaywright(),
    )
    monkeypatch.setattr(
        "litellm.llms.chatgpt.conversation_init._send_oracle_browser_worker_message",
        lambda _sender, message: captured.append(dict(message)),
    )
    monkeypatch.setattr(
        "litellm.llms.chatgpt.conversation_init._disconnect_attached_browser",
        lambda *_args, **_kwargs: None,
    )
    _oracle_browser_history_observation_worker(
        _Sender(),
        "http://127.0.0.1:9222",
        _TEST_PAGE_TARGET_ID,
        time.monotonic() + 5.0,
        time.monotonic() + 5.0,
        _test_account_hash(),
        1_000_000,
        _Value(0),
        _Value(0),
        "history-observer:test",
        _Value(b""),
        _Value(0),
        "about:blank#oracle-native-history-test",
        _Value(False),
        object(),
        _Event(),
        _Event(),
        _Event(set_now=True),
        _Value(False),
    )
    assert captured
    payload = captured[0]
    assert payload["ok"] is False
    assert payload["result"] is None
    assert payload["error_class"] == "RuntimeError"
    assert payload["error_stage"] == "cdp_connect"
    serialized = json.dumps(payload)
    assert "cookie" not in serialized.lower()
    assert "authorization" not in serialized.lower()
    assert "Bearer" not in serialized
    assert "xyz" not in serialized


def test_native_init_observer_continues_cloudflare_document_and_captures_init() -> None:
    account_hash = _test_account_hash()
    page = _FakeOraclePage(
        {
            "status_code": 200,
            "payload": _with_identity(_VALID_PAYLOAD),
        }
    )
    _FakeOracleContext([page])
    session = page.context.new_cdp_session(page)
    observation = _observe_native_oracle_init(
        page,
        session=session,
        request_url=CHATGPT_CONVERSATION_INIT_DEFAULT_URL,
        expected_account_hash=account_hash,
        deadline=time.monotonic() + 5.0,
    )
    assert observation["status_code"] == 200
    native = observation["native_capture"]
    assert native["account_hash"] == account_hash
    assert native["request_method"] == "POST"
    assert native["request_body_omitted"] is True
    methods = [method for method, _params in session.sends]
    assert "Fetch.enable" in methods
    assert "Page.navigate" in methods
    enable = next(params for method, params in session.sends if method == "Fetch.enable")
    patterns = enable["patterns"]
    url_patterns = [str(pattern.get("urlPattern", "")) for pattern in patterns]
    assert url_patterns
    assert all("/backend-api/" in url for url in url_patterns)
    assert not any(url.endswith("/backend-api/*") for url in url_patterns)
    assert any("/backend-api/conversation/init" in url for url in url_patterns)
    assert not any("*" == pattern.get("urlPattern") for pattern in patterns)
    continues = [
        params["requestId"]
        for method, params in session.sends
        if method == "Fetch.continueRequest"
    ]
    fails = [
        params["requestId"]
        for method, params in session.sends
        if method == "Fetch.failRequest"
    ]
    assert "doc-third-party-1" in continues
    assert "doc-cf-1" in continues
    assert "doc-spa-1" in continues
    assert "req-init-1" in continues
    assert "doc-third-party-1" not in fails
    assert "req-init-1" not in fails
    serialized = json.dumps(observation)
    assert "cookie" not in serialized.lower()
    assert "authorization" not in serialized.lower()
    assert _TEST_ACCOUNT_ID not in serialized
    assert "Bearer" not in serialized


def test_native_init_observer_keeps_capture_after_playwright_goto_error() -> None:
    class Error(Exception):
        pass

    account_hash = _test_account_hash()
    page = _FakeOraclePage(
        {
            "status_code": 200,
            "payload": _with_identity(_VALID_PAYLOAD),
        }
    )
    page.goto_error = Error("Navigation interrupted by another navigation")
    _FakeOracleContext([page])
    session = page.context.new_cdp_session(page)
    observation = _observe_native_oracle_init(
        page,
        session=session,
        request_url=CHATGPT_CONVERSATION_INIT_DEFAULT_URL,
        expected_account_hash=account_hash,
        deadline=time.monotonic() + 5.0,
    )
    assert observation["status_code"] == 200
    native = observation["native_capture"]
    assert native["account_hash"] == account_hash
    assert native["request_method"] == "POST"
    serialized = json.dumps(observation)
    assert "cookie" not in serialized.lower()
    assert "authorization" not in serialized.lower()
    assert _TEST_ACCOUNT_ID not in serialized


def test_native_init_observer_retries_fetch_enable_error_and_captures_init() -> None:
    class Error(Exception):
        pass

    account_hash = _test_account_hash()
    page = _FakeOraclePage(
        {
            "status_code": 200,
            "payload": _with_identity(_VALID_PAYLOAD),
        }
    )
    page.skip_goto_emit = True
    page.emit_on_wait = True
    page.fetch_enable_errors = 1
    page.fetch_enable_error = Error("Protocol error (Fetch.enable): session closed")
    _FakeOracleContext([page])
    session = page.context.new_cdp_session(page)
    observation = _observe_native_oracle_init(
        page,
        session=session,
        request_url=CHATGPT_CONVERSATION_INIT_DEFAULT_URL,
        expected_account_hash=account_hash,
        deadline=time.monotonic() + 5.0,
    )
    assert observation["status_code"] == 200
    native = observation["native_capture"]
    assert native["account_hash"] == account_hash
    assert native["request_method"] == "POST"
    enable_count = sum(1 for method, _params in session.sends if method == "Fetch.enable")
    assert enable_count >= 2
    serialized = json.dumps(observation)
    assert "cookie" not in serialized.lower()
    assert "authorization" not in serialized.lower()
    assert _TEST_ACCOUNT_ID not in serialized


def test_native_init_observer_retries_missing_response_body_and_captures_init() -> None:
    class Error(Exception):
        pass

    account_hash = _test_account_hash()
    page = _FakeOraclePage(
        {
            "status_code": 200,
            "payload": _with_identity(_VALID_PAYLOAD),
        }
    )
    page.get_response_body_errors = 1
    page.get_response_body_error = Error(
        "Protocol error (Network.getResponseBody): No resource with given identifier found"
    )
    _FakeOracleContext([page])
    session = page.context.new_cdp_session(page)
    observation = _observe_native_oracle_init(
        page,
        session=session,
        request_url=CHATGPT_CONVERSATION_INIT_DEFAULT_URL,
        expected_account_hash=account_hash,
        deadline=time.monotonic() + 5.0,
    )
    assert observation["status_code"] == 200
    native = observation["native_capture"]
    assert native["account_hash"] == account_hash
    assert native["request_method"] == "POST"
    body_reads = sum(
        1 for method, _params in session.sends if method == "Network.getResponseBody"
    )
    assert body_reads >= 2
    serialized = json.dumps(observation)
    assert "cookie" not in serialized.lower()
    assert "authorization" not in serialized.lower()
    assert _TEST_ACCOUNT_ID not in serialized


def test_native_init_observer_keeps_capture_after_goto_timeout() -> None:
    class TimeoutError(Exception):
        pass

    account_hash = _test_account_hash()
    page = _FakeOraclePage(
        {
            "status_code": 200,
            "payload": _with_identity(_VALID_PAYLOAD),
        }
    )
    page.goto_error = TimeoutError("Timeout 5000ms exceeded")
    _FakeOracleContext([page])
    session = page.context.new_cdp_session(page)
    observation = _observe_native_oracle_init(
        page,
        session=session,
        request_url=CHATGPT_CONVERSATION_INIT_DEFAULT_URL,
        expected_account_hash=account_hash,
        deadline=time.monotonic() + 5.0,
    )
    assert observation["status_code"] == 200
    native = observation["native_capture"]
    assert native["account_hash"] == account_hash
    assert native["request_method"] == "POST"
    serialized = json.dumps(observation)
    assert "cookie" not in serialized.lower()
    assert "authorization" not in serialized.lower()
    assert _TEST_ACCOUNT_ID not in serialized


def test_native_history_classifier_continues_cloudflare_document_get() -> None:
    cloudflare = classify_native_history_request(
        method="GET",
        url="https://challenges.cloudflare.com/cdn-cgi/challenge-platform/h/g/orchestrate/jsch/v1",
        resource_type="Document",
    )
    assert cloudflare["action"] == "continue"
    assert cloudflare["stops_observation"] is False
    spa = classify_native_history_request(
        method="GET",
        url="https://chatgpt.com/c/conv-001",
        resource_type="Document",
        redirected="doc-home-redirect",
    )
    assert spa["action"] == "continue"
    assert spa["stops_observation"] is False
    mutating_document = classify_native_history_request(
        method="POST",
        url="https://example.test/",
        resource_type="Document",
    )
    assert mutating_document["action"] == "fail"
    assert mutating_document["stops_observation"] is False
    enable = _chatgpt_browser_fetch_enable_params()
    url_patterns = [str(pattern.get("urlPattern", "")) for pattern in enable["patterns"]]
    assert url_patterns
    assert all("/backend-api/" in url for url in url_patterns)
    assert not any(url.endswith("/backend-api/*") for url in url_patterns)
    assert any("/backend-api/conversation/init" in url for url in url_patterns)
    assert not any(pattern.get("urlPattern") == "*" for pattern in enable["patterns"])
