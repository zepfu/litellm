"""Focused tests for OpenRouter shared auth resolution (common_utils)."""

from unittest.mock import patch

import pytest

from litellm.llms.openrouter.common_utils import (
    OpenRouterConfigError,
    get_openrouter_auth_headers,
)


# ---------------------------------------------------------------------------
# Exception class contract
# ---------------------------------------------------------------------------


class TestOpenRouterConfigError:
    def test_is_value_error(self):
        err = OpenRouterConfigError("boom")
        assert isinstance(err, ValueError)

    def test_message_preserved(self):
        err = OpenRouterConfigError("some message")
        assert str(err) == "some message"
        assert err.message == "some message"


# ---------------------------------------------------------------------------
# Caller Authorization header: casing and precedence
# ---------------------------------------------------------------------------


class TestCallerHeaderPrecedence:
    def test_standard_casing_wins_over_api_key(self):
        headers = get_openrouter_auth_headers(
            api_key="explicit-key",
            extra_headers={"Authorization": "Bearer caller-tok"},
        )
        assert headers == {"Authorization": "Bearer caller-tok"}

    def test_lowercase_casing_preserved(self):
        headers = get_openrouter_auth_headers(
            api_key="explicit-key",
            extra_headers={"authorization": "Bearer caller-tok"},
        )
        assert headers == {"authorization": "Bearer caller-tok"}

    def test_mixed_casing_preserved(self):
        headers = get_openrouter_auth_headers(
            extra_headers={"AUTHORIZATION": "Bearer tok123"},
        )
        assert headers == {"AUTHORIZATION": "Bearer tok123"}

    def test_bearer_case_insensitive(self):
        headers = get_openrouter_auth_headers(
            extra_headers={"Authorization": "bearer tok-lower"},
        )
        assert headers == {"Authorization": "bearer tok-lower"}

    def test_caller_header_preserves_exact_value(self):
        raw = "Bearer  spaced-token  "
        headers = get_openrouter_auth_headers(
            extra_headers={"Authorization": raw},
        )
        # Original value preserved, not re-serialized
        assert headers["Authorization"] == raw


# ---------------------------------------------------------------------------
# Malformed caller Authorization headers
# ---------------------------------------------------------------------------


class TestMalformedCallerHeaders:
    @pytest.mark.parametrize(
        "value",
        [
            "",
            "   ",
            "\t\n",
        ],
        ids=["empty", "whitespace", "tabs-newlines"],
    )
    def test_empty_or_whitespace_raises(self, value: str):
        with pytest.raises(OpenRouterConfigError):
            get_openrouter_auth_headers(
                api_key="fallback",
                extra_headers={"Authorization": value},
            )

    @pytest.mark.parametrize(
        "value",
        [
            "Basic abc123",
            "Token xyz",
            "just-a-raw-key",
        ],
        ids=["basic-scheme", "token-scheme", "no-scheme"],
    )
    def test_non_bearer_raises(self, value: str):
        with pytest.raises(OpenRouterConfigError):
            get_openrouter_auth_headers(
                api_key="fallback",
                extra_headers={"Authorization": value},
            )

    @pytest.mark.parametrize(
        "value",
        [
            "Bearer ",
            "Bearer   ",
            "Bearer\t",
        ],
        ids=["trailing-space", "trailing-spaces", "trailing-tab"],
    )
    def test_empty_bearer_token_raises(self, value: str):
        with pytest.raises(OpenRouterConfigError):
            get_openrouter_auth_headers(
                api_key="fallback",
                extra_headers={"Authorization": value},
            )

    def test_malformed_does_not_fall_through(self):
        """Even with a valid api_key, malformed caller header must raise."""
        with pytest.raises(OpenRouterConfigError):
            get_openrouter_auth_headers(
                api_key="valid-key",
                extra_headers={"Authorization": ""},
            )

    @pytest.mark.parametrize(
        "value",
        [
            "Bearer two words",
            "Bearer token\twith-tab",
            "Bearer token\nwith-newline",
            "Bearer token\r\nInjected: header",
            "Bearer token\x00suffix",
        ],
    )
    def test_token_whitespace_or_control_characters_rejected(self, value):
        with pytest.raises(OpenRouterConfigError) as exc_info:
            get_openrouter_auth_headers(
                api_key="fallback",
                extra_headers={"Authorization": value},
            )
        assert "token" not in str(exc_info.value).lower().replace(
            "bearer token", ""
        )

    def test_surrounding_whitespace_preserved_for_single_token(self):
        value = "  Bearer single-token  "
        result = get_openrouter_auth_headers(
            extra_headers={"Authorization": value}
        )
        assert result == {"Authorization": value}

    @pytest.mark.parametrize(
        "value",
        [
            "Bearer\ttoken",
            "Bearer\r\ntoken",
            "Bearer token\r\n",
            "Bearer token\t",
            "\tBearer token",
            "\vBearer token",
            "\fBearer token",
            "\x7fBearer token",
        ],
    )
    def test_raw_control_or_non_printable_characters_rejected(self, value):
        with pytest.raises(
            OpenRouterConfigError,
            match="control or non-printable characters",
        ) as exc_info:
            get_openrouter_auth_headers(
                api_key="valid-fallback",
                extra_headers={"Authorization": value},
            )
        assert value not in str(exc_info.value)


# ---------------------------------------------------------------------------
# Source precedence chain (no caller header)
# ---------------------------------------------------------------------------


def _no_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in ("OPENROUTER_API_KEY", "OR_API_KEY", "AAWM_OPENROUTER_API_KEY"):
        monkeypatch.delenv(var, raising=False)


class TestSourcePrecedence:
    def test_explicit_api_key_first(self, monkeypatch: pytest.MonkeyPatch):
        _no_env(monkeypatch)
        with patch("litellm.api_key", "litellm-global"), patch(
            "litellm.openrouter_key", "litellm-or"
        ):
            headers = get_openrouter_auth_headers(api_key="explicit")
        assert headers == {"Authorization": "Bearer explicit"}

    def test_litellm_api_key_over_litellm_openrouter_key(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        _no_env(monkeypatch)
        with patch("litellm.api_key", "litellm-global"), patch(
            "litellm.openrouter_key", "litellm-or"
        ):
            headers = get_openrouter_auth_headers()
        assert headers == {"Authorization": "Bearer litellm-global"}

    def test_litellm_api_key_when_no_openrouter_key(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        _no_env(monkeypatch)
        with patch("litellm.api_key", "litellm-global"), patch(
            "litellm.openrouter_key", None
        ):
            headers = get_openrouter_auth_headers()
        assert headers == {"Authorization": "Bearer litellm-global"}

    def test_env_openrouter_api_key(self, monkeypatch: pytest.MonkeyPatch):
        _no_env(monkeypatch)
        monkeypatch.setenv("OPENROUTER_API_KEY", "env-or")
        with patch("litellm.api_key", None), patch("litellm.openrouter_key", None):
            headers = get_openrouter_auth_headers()
        assert headers == {"Authorization": "Bearer env-or"}

    def test_env_or_api_key(self, monkeypatch: pytest.MonkeyPatch):
        _no_env(monkeypatch)
        monkeypatch.setenv("OR_API_KEY", "env-orapi")
        with patch("litellm.api_key", None), patch("litellm.openrouter_key", None):
            headers = get_openrouter_auth_headers()
        assert headers == {"Authorization": "Bearer env-orapi"}

    def test_env_aawm_openrouter_api_key(self, monkeypatch: pytest.MonkeyPatch):
        _no_env(monkeypatch)
        monkeypatch.setenv("AAWM_OPENROUTER_API_KEY", "env-aawm")
        with patch("litellm.api_key", None), patch("litellm.openrouter_key", None):
            headers = get_openrouter_auth_headers()
        assert headers == {"Authorization": "Bearer env-aawm"}

    def test_env_order_openrouter_before_or(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        _no_env(monkeypatch)
        monkeypatch.setenv("OPENROUTER_API_KEY", "first")
        monkeypatch.setenv("OR_API_KEY", "second")
        monkeypatch.setenv("AAWM_OPENROUTER_API_KEY", "third")
        with patch("litellm.api_key", None), patch("litellm.openrouter_key", None):
            headers = get_openrouter_auth_headers()
        assert headers == {"Authorization": "Bearer first"}

    def test_env_order_or_before_aawm(self, monkeypatch: pytest.MonkeyPatch):
        _no_env(monkeypatch)
        monkeypatch.setenv("OR_API_KEY", "second")
        monkeypatch.setenv("AAWM_OPENROUTER_API_KEY", "third")
        with patch("litellm.api_key", None), patch("litellm.openrouter_key", None):
            headers = get_openrouter_auth_headers()
        assert headers == {"Authorization": "Bearer second"}


# ---------------------------------------------------------------------------
# Blank / whitespace fallthrough
# ---------------------------------------------------------------------------


class TestBlankFallthrough:
    def test_whitespace_api_key_skipped(self, monkeypatch: pytest.MonkeyPatch):
        _no_env(monkeypatch)
        monkeypatch.setenv("OPENROUTER_API_KEY", "env-val")
        with patch("litellm.api_key", None), patch("litellm.openrouter_key", None):
            headers = get_openrouter_auth_headers(api_key="   ")
        assert headers == {"Authorization": "Bearer env-val"}

    def test_whitespace_litellm_keys_skipped(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        _no_env(monkeypatch)
        monkeypatch.setenv("OR_API_KEY", "env-or")
        with patch("litellm.api_key", "  "), patch("litellm.openrouter_key", "\t"):
            headers = get_openrouter_auth_headers()
        assert headers == {"Authorization": "Bearer env-or"}


# ---------------------------------------------------------------------------
# Missing-all failure
# ---------------------------------------------------------------------------


class TestMissingAllFailure:
    def test_raises_when_no_source(self, monkeypatch: pytest.MonkeyPatch):
        _no_env(monkeypatch)
        with patch("litellm.api_key", None), patch("litellm.openrouter_key", None):
            with pytest.raises(OpenRouterConfigError):
                get_openrouter_auth_headers()

    def test_raises_with_all_whitespace(self, monkeypatch: pytest.MonkeyPatch):
        _no_env(monkeypatch)
        with patch("litellm.api_key", " "), patch("litellm.openrouter_key", " "):
            with pytest.raises(OpenRouterConfigError):
                get_openrouter_auth_headers(api_key="  ")

    def test_missing_error_covers_all_source_categories(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        _no_env(monkeypatch)
        with patch("litellm.api_key", None), patch("litellm.openrouter_key", None):
            with pytest.raises(OpenRouterConfigError) as exc_info:
                get_openrouter_auth_headers()

        message = str(exc_info.value)
        assert "request configuration" in message
        assert "LiteLLM configuration" in message
        assert "environment" in message


# ---------------------------------------------------------------------------
# No key-source leakage in error messages
# ---------------------------------------------------------------------------


class TestNoKeyLeakage:
    def test_missing_all_error_does_not_name_sources(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        _no_env(monkeypatch)
        with patch("litellm.api_key", None), patch("litellm.openrouter_key", None):
            with pytest.raises(OpenRouterConfigError) as exc_info:
                get_openrouter_auth_headers()
        msg = str(exc_info.value).lower()
        # Must not name individual source variables in a way that reveals
        # which were checked or their values.
        assert "openrouter_api_key" not in msg
        assert "or_api_key" not in msg
        assert "aawm_openrouter_api_key" not in msg
        assert "litellm.api_key" not in msg
        assert "litellm.openrouter_key" not in msg

    def test_malformed_header_error_does_not_echo_value(self):
        with pytest.raises(OpenRouterConfigError) as exc_info:
            get_openrouter_auth_headers(
                extra_headers={"Authorization": "Basic super-secret-value"},
            )
        msg = str(exc_info.value)
        assert "super-secret-value" not in msg


class TestNonStringCredentials:
    @pytest.mark.parametrize("value", [None, 123, object()])
    def test_non_string_authorization_header_rejected(self, value):
        with pytest.raises(OpenRouterConfigError, match="must be a string"):
            get_openrouter_auth_headers(
                api_key="fallback",
                extra_headers={"Authorization": value},
            )

    def test_non_string_explicit_key_rejected(self):
        with pytest.raises(OpenRouterConfigError, match="must be a string"):
            get_openrouter_auth_headers(api_key=123)

    def test_non_string_global_key_rejected(self):
        with patch("litellm.api_key", {"secret": "value"}), patch(
            "litellm.openrouter_key", None
        ):
            with pytest.raises(OpenRouterConfigError, match="must be a string"):
                get_openrouter_auth_headers()

    def test_non_string_secret_value_rejected(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        _no_env(monkeypatch)
        with patch("litellm.api_key", None), patch(
            "litellm.openrouter_key", None
        ), patch(
            "litellm.secret_managers.main.get_secret_str",
            return_value=["not", "a", "string"],
        ):
            with pytest.raises(OpenRouterConfigError, match="must be a string"):
                get_openrouter_auth_headers()

    def test_non_string_errors_do_not_expose_values(self):
        secret_value = "non-string-secret-marker"
        with patch("litellm.api_key", {secret_value: True}), patch(
            "litellm.openrouter_key", None
        ):
            with pytest.raises(OpenRouterConfigError) as exc_info:
                get_openrouter_auth_headers()
        assert secret_value not in str(exc_info.value)


class TestMalformedConfiguredCredentials:
    @pytest.mark.parametrize(
        "value",
        [
            "two words",
            "token\twith-tab",
            "token\nwith-newline",
            "token\r\nInjected: header",
            "token\x00suffix",
        ],
    )
    def test_explicit_malformed_value_rejected(
        self, monkeypatch: pytest.MonkeyPatch, value
    ):
        _no_env(monkeypatch)
        monkeypatch.setenv("OPENROUTER_API_KEY", "valid-env-fallback")
        with patch("litellm.api_key", "valid-global-fallback"), patch(
            "litellm.openrouter_key", "valid-openrouter-fallback"
        ), pytest.raises(
            OpenRouterConfigError,
            match="one printable token without whitespace or control characters",
        ) as exc_info:
            get_openrouter_auth_headers(api_key=value)
        assert value not in str(exc_info.value)

    def test_explicit_surrounding_whitespace_trimmed(self):
        assert get_openrouter_auth_headers(api_key="  single-token  ") == {
            "Authorization": "Bearer single-token"
        }

    def test_malformed_global_fails_before_lower_priority_source(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        _no_env(monkeypatch)
        monkeypatch.setenv("OPENROUTER_API_KEY", "valid-lower-priority")
        with patch("litellm.api_key", "malformed global"), patch(
            "litellm.openrouter_key", None
        ):
            with pytest.raises(OpenRouterConfigError) as exc_info:
                get_openrouter_auth_headers()
        assert "malformed global" not in str(exc_info.value)

    @pytest.mark.parametrize(
        "env_name",
        [
            "OPENROUTER_API_KEY",
            "OR_API_KEY",
            "AAWM_OPENROUTER_API_KEY",
        ],
    )
    def test_each_env_source_rejects_malformed_value(
        self, monkeypatch: pytest.MonkeyPatch, env_name: str
    ):
        _no_env(monkeypatch)
        monkeypatch.setenv(env_name, "malformed env-value")
        with patch("litellm.api_key", None), patch(
            "litellm.openrouter_key", None
        ):
            with pytest.raises(OpenRouterConfigError) as exc_info:
                get_openrouter_auth_headers()
        assert "malformed env-value" not in str(exc_info.value)

    @pytest.mark.parametrize(
        ("malformed_source", "fallback_source"),
        [
            ("OPENROUTER_API_KEY", "OR_API_KEY"),
            ("OR_API_KEY", "AAWM_OPENROUTER_API_KEY"),
        ],
    )
    def test_malformed_env_fails_before_valid_lower_priority_source(
        self,
        monkeypatch: pytest.MonkeyPatch,
        malformed_source: str,
        fallback_source: str,
    ):
        _no_env(monkeypatch)
        monkeypatch.setenv(malformed_source, "malformed env-value")
        monkeypatch.setenv(fallback_source, "valid-lower-priority")
        with patch("litellm.api_key", None), patch(
            "litellm.openrouter_key", None
        ), pytest.raises(
            OpenRouterConfigError,
            match="one printable token without whitespace or control characters",
        ):
            get_openrouter_auth_headers()

    def test_malformed_openrouter_global_fails_before_environment(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        _no_env(monkeypatch)
        monkeypatch.setenv("OPENROUTER_API_KEY", "valid-lower-priority")
        with patch("litellm.api_key", " "), patch(
            "litellm.openrouter_key", "malformed openrouter-global"
        ):
            with pytest.raises(OpenRouterConfigError):
                get_openrouter_auth_headers()


# ---------------------------------------------------------------------------
# Duplicate Authorization header detection (case-insensitive)
# ---------------------------------------------------------------------------


class TestDuplicateAuthorizationHeaders:
    def test_identical_values_still_rejected(self):
        with pytest.raises(OpenRouterConfigError, match="multiple"):
            get_openrouter_auth_headers(
                api_key="fallback",
                extra_headers={
                    "Authorization": "Bearer tok1",
                    "authorization": "Bearer tok1",
                },
            )

    def test_different_values_rejected(self):
        with pytest.raises(OpenRouterConfigError, match="multiple"):
            get_openrouter_auth_headers(
                api_key="fallback",
                extra_headers={
                    "Authorization": "Bearer tok1",
                    "AUTHORIZATION": "Bearer tok2",
                },
            )

    def test_reversed_order_rejected(self):
        with pytest.raises(OpenRouterConfigError, match="multiple"):
            get_openrouter_auth_headers(
                extra_headers={
                    "authorization": "Bearer a",
                    "Authorization": "Bearer b",
                },
            )

    def test_malformed_plus_valid_rejected(self):
        """Even if one is malformed, duplicate count >1 rejects first."""
        with pytest.raises(OpenRouterConfigError, match="multiple"):
            get_openrouter_auth_headers(
                api_key="fallback",
                extra_headers={
                    "Authorization": "",
                    "AUTHORIZATION": "Bearer valid",
                },
            )

    def test_no_secret_in_duplicate_error(self):
        with pytest.raises(OpenRouterConfigError) as exc_info:
            get_openrouter_auth_headers(
                extra_headers={
                    "Authorization": "Bearer secret-token-abc",
                    "authorization": "Bearer secret-token-abc",
                },
            )
        msg = str(exc_info.value)
        assert "secret-token-abc" not in msg

    def test_single_header_not_rejected(self):
        """Exactly one Authorization key is fine."""
        headers = get_openrouter_auth_headers(
            extra_headers={"authorization": "Bearer only-one"},
        )
        assert headers == {"authorization": "Bearer only-one"}

    def test_non_auth_headers_ignored(self):
        """Multiple non-auth headers do not trigger the check."""
        headers = get_openrouter_auth_headers(
            api_key="key1",
            extra_headers={
                "X-Custom": "a",
                "x-custom": "b",
                "Authorization": "Bearer tok",
            },
        )
        assert headers == {"Authorization": "Bearer tok"}
