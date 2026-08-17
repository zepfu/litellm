"""CFG-004 criterion 11: Codex OAuth lane-seeding focused tests.

Covers: exact OpenAI lane identity for ``codex_oauth_account_id``,
canonical UUID validation and rejection of JWT/token/non-UUID values,
and unchanged non-OpenAI lane behavior when the descriptor is absent
or supplied.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException

from litellm.proxy.pass_through_endpoints.aawm_alias_routing.cooldown_acceptance import (
    _validate_codex_oauth_account_id,
    resolve_production_cooldown_key,
    resolve_production_lane_key,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fake_request_with_auth(auth_value: str) -> MagicMock:
    """Build a mock Request with only an Authorization header."""
    req = MagicMock()
    req.headers = {"authorization": auth_value}
    return req


def _bare_request() -> MagicMock:
    """Build a mock Request with no relevant headers."""
    req = MagicMock()
    req.headers = {}
    return req


def _openai_candidate() -> dict:
    return {
        "provider": "openai",
        "model": "gpt-5.3-codex",
        "route_family": "codex_openai_chat_completions_adapter",
        "priority": 1,
        "config_epoch_tag": "unrelated-snapshot-hash",
        "cooldown_identity_tag": (
            "alias:basic:openai:gpt-5.3-codex:"
            "codex_openai_chat_completions_adapter"
        ),
    }


def _alibaba_candidate() -> dict:
    return {
        "provider": "alibaba_token_plan",
        "model": "alibaba_token_plan/qwen3.6-flash",
        "route_family": "codex_alibaba_token_plan_chat_completions_adapter",
        "priority": 2,
        "config_epoch_tag": "abc123",
    }


_CANONICAL_UUID = "550e8400-e29b-41d4-a716-446655440000"


# ---------------------------------------------------------------------------
# _validate_codex_oauth_account_id
# ---------------------------------------------------------------------------


class TestValidateCodexOauthAccountId:
    """Validation of the optional codex_oauth_account_id descriptor."""

    def test_none_returns_none(self):
        assert _validate_codex_oauth_account_id(None) is None

    def test_valid_uuid(self):
        assert _validate_codex_oauth_account_id(_CANONICAL_UUID) == _CANONICAL_UUID

    def test_valid_uuid_with_surrounding_whitespace(self):
        assert _validate_codex_oauth_account_id(f"  {_CANONICAL_UUID}  ") == _CANONICAL_UUID

    def test_valid_uppercase_uuid_normalized_to_lowercase(self):
        upper = "550E8400-E29B-41D4-A716-446655440000"
        assert _validate_codex_oauth_account_id(upper) == _CANONICAL_UUID

    def test_valid_mixed_case_uuid_normalized(self):
        mixed = "550e8400-E29B-41d4-A716-446655440000"
        assert _validate_codex_oauth_account_id(mixed) == _CANONICAL_UUID

    def test_rejects_non_string(self):
        with pytest.raises(HTTPException) as exc_info:
            _validate_codex_oauth_account_id(12345)
        assert exc_info.value.status_code == 400
        assert "must be a string" in exc_info.value.detail["message"]

    def test_rejects_empty_string(self):
        with pytest.raises(HTTPException) as exc_info:
            _validate_codex_oauth_account_id("")
        assert exc_info.value.status_code == 400
        assert "nonempty" in exc_info.value.detail["message"]

    def test_rejects_whitespace_only(self):
        with pytest.raises(HTTPException) as exc_info:
            _validate_codex_oauth_account_id("   ")
        assert exc_info.value.status_code == 400
        assert "nonempty" in exc_info.value.detail["message"]

    def test_rejects_non_uuid_alphanumeric(self):
        with pytest.raises(HTTPException) as exc_info:
            _validate_codex_oauth_account_id("org.acct_01-xyz")
        assert exc_info.value.status_code == 400
        assert "canonical UUID" in exc_info.value.detail["message"]

    def test_rejects_uuid_without_hyphens(self):
        with pytest.raises(HTTPException) as exc_info:
            _validate_codex_oauth_account_id("550e8400e29b41d4a716446655440000")
        assert exc_info.value.status_code == 400
        assert "canonical UUID" in exc_info.value.detail["message"]

    def test_rejects_jwt_shaped_token(self):
        jwt = (
            "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9"
            ".eyJzdWIiOiIxMjM0NTY3ODkwIn0"
            ".dozjgNryP4J3jVmNHl0w5N_XgL0n3I9PlFUP0THsR8U"
        )
        with pytest.raises(HTTPException) as exc_info:
            _validate_codex_oauth_account_id(jwt)
        assert exc_info.value.status_code == 400
        assert "canonical UUID" in exc_info.value.detail["message"]

    def test_rejects_bearer_token_value(self):
        with pytest.raises(HTTPException) as exc_info:
            _validate_codex_oauth_account_id("Bearer sk-abc123def456")
        assert exc_info.value.status_code == 400
        assert "canonical UUID" in exc_info.value.detail["message"]

    def test_rejects_sk_prefixed_token(self):
        with pytest.raises(HTTPException) as exc_info:
            _validate_codex_oauth_account_id("sk-abc123def456")
        assert exc_info.value.status_code == 400
        assert "canonical UUID" in exc_info.value.detail["message"]

    def test_rejects_arbitrary_token_string(self):
        with pytest.raises(HTTPException) as exc_info:
            _validate_codex_oauth_account_id("tok_live_abc123xyz789")
        assert exc_info.value.status_code == 400
        assert "canonical UUID" in exc_info.value.detail["message"]

    def test_rejects_control_characters(self):
        with pytest.raises(HTTPException) as exc_info:
            _validate_codex_oauth_account_id("abc\x00def")
        assert exc_info.value.status_code == 400

    def test_rejects_newline(self):
        with pytest.raises(HTTPException) as exc_info:
            _validate_codex_oauth_account_id("abc\ndef")
        assert exc_info.value.status_code == 400

    def test_rejects_uuid_with_extra_segment(self):
        with pytest.raises(HTTPException) as exc_info:
            _validate_codex_oauth_account_id(f"{_CANONICAL_UUID}-extra")
        assert exc_info.value.status_code == 400
        assert "canonical UUID" in exc_info.value.detail["message"]

    def test_rejects_short_uuid(self):
        with pytest.raises(HTTPException) as exc_info:
            _validate_codex_oauth_account_id("550e8400-e29b-41d4-a716")
        assert exc_info.value.status_code == 400
        assert "canonical UUID" in exc_info.value.detail["message"]


# ---------------------------------------------------------------------------
# Exact OpenAI lane identity
# ---------------------------------------------------------------------------


class TestOpenAILaneIdentity:
    """Prove codex_oauth_account_id seeds the exact production lane."""

    def test_lane_key_matches_production_chatgpt_account_format(self):
        """The descriptor must produce the exact ``chatgpt-account:<id>``
        lane key that production ``_resolve_codex_auto_agent_openai_lane_key``
        derives from the ``chatgpt-account-id`` header."""
        candidate = _openai_candidate()

        req = _bare_request()
        acceptance_lane = resolve_production_lane_key(
            req, candidate, codex_oauth_account_id=_CANONICAL_UUID,
        )

        # Production _resolve_codex_auto_agent_openai_lane_key returns
        # f"chatgpt-account:{account_id}" when chatgpt-account-id header
        # is present.  The acceptance descriptor must produce the same key.
        assert acceptance_lane == f"chatgpt-account:{_CANONICAL_UUID}"

    def test_cooldown_key_uses_exact_lane_and_stable_identity(self):
        """Full cooldown key embeds the chatgpt-account lane and stable tag."""
        candidate = _openai_candidate()

        req = _bare_request()
        acceptance_key = resolve_production_cooldown_key(
            req, candidate, codex_oauth_account_id=_CANONICAL_UUID,
        )

        # _codex_auto_agent_candidate_key format:
        # h{cooldown_identity}:{provider}:{model}:{lane_key}
        expected = (
            f"h{candidate['cooldown_identity_tag']}:"
            f"{candidate['provider']}:{candidate['model']}:"
            f"chatgpt-account:{_CANONICAL_UUID}"
        )
        assert acceptance_key == expected
        assert candidate["config_epoch_tag"] not in acceptance_key

    def test_without_descriptor_delegates_to_request_header_resolution(self):
        """When descriptor is absent, lane derives from request headers via
        the production _resolve_codex_auto_agent_openai_cooldown_lane_key."""
        from unittest.mock import patch

        candidate = _openai_candidate()
        req = _bare_request()

        with patch(
            "litellm.proxy.pass_through_endpoints.aawm_alias_routing.cooldown_acceptance._resolve_codex_auto_agent_openai_cooldown_lane_key",
            return_value="auth:abc123hash",
        ) as mock_resolve:
            lane = resolve_production_lane_key(req, candidate)

        mock_resolve.assert_called_once_with(req)
        assert lane == "auth:abc123hash"

    def test_without_descriptor_none_explicit_delegates(self):
        """Explicit None descriptor behaves same as absent (delegates)."""
        from unittest.mock import patch

        candidate = _openai_candidate()
        req = _bare_request()

        with patch(
            "litellm.proxy.pass_through_endpoints.aawm_alias_routing.cooldown_acceptance._resolve_codex_auto_agent_openai_cooldown_lane_key",
            return_value="__default__",
        ):
            lane_absent = resolve_production_lane_key(req, candidate)
            lane_none = resolve_production_lane_key(
                req, candidate, codex_oauth_account_id=None,
            )
        assert lane_absent == lane_none == "__default__"


# ---------------------------------------------------------------------------
# No descriptor leakage
# ---------------------------------------------------------------------------


class TestNoDescriptorLeakage:
    """The account id must not appear in response or audit output."""

    def test_lane_key_does_not_expose_raw_bearer(self):
        """Lane key uses account id, not bearer tokens."""
        candidate = _openai_candidate()
        req = _fake_request_with_auth("Bearer super-secret-token")

        lane = resolve_production_lane_key(
            req, candidate, codex_oauth_account_id=_CANONICAL_UUID,
        )
        assert "super-secret-token" not in lane
        assert "Bearer" not in lane
        assert _CANONICAL_UUID in lane

    def test_cooldown_key_does_not_expose_auth_header(self):
        """Cooldown key must not embed Authorization header values."""
        candidate = _openai_candidate()
        req = _fake_request_with_auth("Bearer sk-secret-value-xyz")

        key = resolve_production_cooldown_key(
            req, candidate, codex_oauth_account_id=_CANONICAL_UUID,
        )
        assert "sk-secret-value-xyz" not in key
        assert "Bearer" not in key


# ---------------------------------------------------------------------------
# Non-OpenAI lane behavior unchanged
# ---------------------------------------------------------------------------


class TestNonOpenAIUnchanged:
    """Descriptor must not alter non-OpenAI provider lanes."""

    def test_alibaba_lane_ignores_descriptor(self):
        candidate = _alibaba_candidate()
        req = _bare_request()

        lane_without = resolve_production_lane_key(req, candidate)
        lane_with = resolve_production_lane_key(
            req, candidate, codex_oauth_account_id=_CANONICAL_UUID,
        )
        assert lane_without == lane_with
        assert "chatgpt-account" not in lane_with

    def test_openrouter_lane_ignores_descriptor(self):
        candidate = {
            "provider": "openrouter",
            "model": "openrouter/auto",
            "route_family": "codex_openrouter_chat_completions_adapter",
            "priority": 3,
            "config_epoch_tag": "abc123",
        }
        req = _bare_request()

        lane_without = resolve_production_lane_key(req, candidate)
        lane_with = resolve_production_lane_key(
            req, candidate, codex_oauth_account_id=_CANONICAL_UUID,
        )
        assert lane_without == lane_with

    def test_kimi_lane_ignores_descriptor(self):
        candidate = {
            "provider": "kimi_code",
            "model": "kimi_code/kimi-k2",
            "route_family": "codex_kimi_code_chat_completions_adapter",
            "priority": 4,
            "config_epoch_tag": "abc123",
        }
        req = _bare_request()

        lane_without = resolve_production_lane_key(req, candidate)
        lane_with = resolve_production_lane_key(
            req, candidate, codex_oauth_account_id=_CANONICAL_UUID,
        )
        assert lane_without == lane_with

    def test_alibaba_cooldown_key_ignores_descriptor(self):
        candidate = _alibaba_candidate()
        req = _bare_request()

        key_without = resolve_production_cooldown_key(req, candidate)
        key_with = resolve_production_cooldown_key(
            req, candidate, codex_oauth_account_id=_CANONICAL_UUID,
        )
        assert key_without == key_with
