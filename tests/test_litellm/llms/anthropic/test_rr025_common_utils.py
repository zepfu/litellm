"""
RR-025: ambient Anthropic credential helpers in common_utils.

Covers restoration of:
- auth_token param on get_anthropic_headers
- AnthropicModelInfo.get_auth_token / get_auth_header
- ANTHROPIC_AUTH_TOKEN fallback in validate_environment
- ANTHROPIC_BASE_URL fallback in get_api_base

Preserves AAWM missing-key messaging and OAuth (sk-ant-oat*) Bearer behavior.
"""

from unittest.mock import patch as mock_patch

import pytest

FAKE_REGULAR_KEY = "sk-ant-api03-fake-regular-key-for-testing-123456789"
FAKE_OAUTH_TOKEN = "sk-ant-oat01-fake-token-for-testing-123456789abcdef"
FAKE_AUTH_TOKEN = "sk-ant-aut01-fake-auth-token-for-testing-123456789"


class TestGetAnthropicHeadersWithAuthToken:
    """Tests for get_anthropic_headers with auth_token parameter."""

    def test_auth_token_uses_bearer_header(self):
        """auth_token should produce Authorization: Bearer header."""
        from litellm.llms.anthropic.common_utils import AnthropicModelInfo

        config = AnthropicModelInfo()
        headers = config.get_anthropic_headers(
            api_key=None,
            auth_token=FAKE_AUTH_TOKEN,
            computer_tool_used=False,
            prompt_caching_set=False,
            pdf_used=False,
            is_vertex_request=False,
        )

        assert headers["authorization"] == f"Bearer {FAKE_AUTH_TOKEN}"
        assert "x-api-key" not in headers
        # auth_token should NOT set OAuth-specific flags
        assert "anthropic-dangerous-direct-browser-access" not in headers

    def test_auth_token_includes_standard_headers(self):
        """auth_token path should include standard Anthropic headers."""
        from litellm.llms.anthropic.common_utils import AnthropicModelInfo

        config = AnthropicModelInfo()
        headers = config.get_anthropic_headers(
            api_key=None,
            auth_token=FAKE_AUTH_TOKEN,
            computer_tool_used=False,
            prompt_caching_set=False,
            pdf_used=False,
            is_vertex_request=False,
        )

        assert headers["anthropic-version"] == "2023-06-01"
        assert headers["accept"] == "application/json"
        assert headers["content-type"] == "application/json"

    def test_api_key_takes_precedence_over_auth_token(self):
        """When both api_key and auth_token are provided, api_key wins."""
        from litellm.llms.anthropic.common_utils import AnthropicModelInfo

        config = AnthropicModelInfo()
        headers = config.get_anthropic_headers(
            api_key=FAKE_REGULAR_KEY,
            auth_token=FAKE_AUTH_TOKEN,
            computer_tool_used=False,
            prompt_caching_set=False,
            pdf_used=False,
            is_vertex_request=False,
        )

        assert headers["x-api-key"] == FAKE_REGULAR_KEY
        assert "authorization" not in headers

    def test_oauth_api_key_still_uses_oauth_headers(self):
        """OAuth api_key still takes Bearer + OAuth beta flags over auth_token."""
        from litellm.llms.anthropic.common_utils import AnthropicModelInfo

        config = AnthropicModelInfo()
        headers = config.get_anthropic_headers(
            api_key=FAKE_OAUTH_TOKEN,
            auth_token=FAKE_AUTH_TOKEN,
            computer_tool_used=False,
            prompt_caching_set=False,
            pdf_used=False,
            is_vertex_request=False,
        )

        assert headers["authorization"] == f"Bearer {FAKE_OAUTH_TOKEN}"
        assert headers["anthropic-dangerous-direct-browser-access"] == "true"
        assert "x-api-key" not in headers
        assert "oauth-2025-04-20" in headers.get("anthropic-beta", "")


class TestValidateEnvironmentAuthToken:
    """Tests for validate_environment with auth_token resolution."""

    def test_auth_token_env_var_produces_bearer_header(self):
        """validate_environment should use Bearer auth when only ANTHROPIC_AUTH_TOKEN is set."""
        from litellm.llms.anthropic.common_utils import AnthropicModelInfo

        config = AnthropicModelInfo()
        with mock_patch.dict(
            "os.environ",
            {"ANTHROPIC_AUTH_TOKEN": FAKE_AUTH_TOKEN},
            clear=True,
        ):
            headers = config.validate_environment(
                headers={},
                model="claude-sonnet-4-5-20250929",
                messages=[{"role": "user", "content": "Hello"}],
                optional_params={},
                litellm_params={},
                api_key=None,
                api_base=None,
            )

        assert headers["authorization"] == f"Bearer {FAKE_AUTH_TOKEN}"
        assert "x-api-key" not in headers
        assert "anthropic-dangerous-direct-browser-access" not in headers

    def test_api_key_param_takes_precedence_over_auth_token_env_var(self):
        """validate_environment should prefer explicit api_key over ANTHROPIC_AUTH_TOKEN."""
        from litellm.llms.anthropic.common_utils import AnthropicModelInfo

        config = AnthropicModelInfo()
        with mock_patch.dict(
            "os.environ",
            {"ANTHROPIC_AUTH_TOKEN": FAKE_AUTH_TOKEN},
            clear=True,
        ):
            headers = config.validate_environment(
                headers={},
                model="claude-sonnet-4-5-20250929",
                messages=[{"role": "user", "content": "Hello"}],
                optional_params={},
                litellm_params={},
                api_key=FAKE_REGULAR_KEY,
                api_base=None,
            )

        assert headers["x-api-key"] == FAKE_REGULAR_KEY
        assert "authorization" not in headers

    def test_raises_when_no_credentials_with_aawm_message(self):
        """Missing credentials keep the AAWM direct-route missing-key message."""
        import litellm
        from litellm.llms.anthropic.common_utils import AnthropicModelInfo

        config = AnthropicModelInfo()
        with mock_patch.dict("os.environ", {}, clear=True):
            with pytest.raises(litellm.AuthenticationError) as exc_info:
                config.validate_environment(
                    headers={},
                    model="claude-sonnet-4-5-20250929",
                    messages=[{"role": "user", "content": "Hello"}],
                    optional_params={},
                    litellm_params={},
                    api_key=None,
                    api_base=None,
                )

        message = exc_info.value.message
        assert "ANTHROPIC_API_KEY" in message
        assert "server-side" in message.lower() or "deployment" in message.lower()
        assert "not Anthropic provider credentials" in message

    def test_resolves_api_key_from_env_when_param_is_none(self):
        """validate_environment should resolve ANTHROPIC_API_KEY from env when api_key param is None."""
        from litellm.llms.anthropic.common_utils import AnthropicModelInfo

        config = AnthropicModelInfo()
        with mock_patch.dict(
            "os.environ",
            {"ANTHROPIC_API_KEY": FAKE_REGULAR_KEY},
            clear=True,
        ):
            headers = config.validate_environment(
                headers={},
                model="claude-sonnet-4-5-20250929",
                messages=[{"role": "user", "content": "Hello"}],
                optional_params={},
                litellm_params={},
                api_key=None,
                api_base=None,
            )

        assert headers["x-api-key"] == FAKE_REGULAR_KEY
        assert "authorization" not in headers

    def test_oauth_api_key_env_still_uses_bearer(self):
        """OAuth token in ANTHROPIC_API_KEY env still uses Bearer via oauth handling."""
        from litellm.llms.anthropic.common_utils import AnthropicModelInfo

        config = AnthropicModelInfo()
        with mock_patch.dict(
            "os.environ",
            {"ANTHROPIC_API_KEY": FAKE_OAUTH_TOKEN},
            clear=True,
        ):
            headers = config.validate_environment(
                headers={},
                model="claude-sonnet-4-5-20250929",
                messages=[{"role": "user", "content": "Hello"}],
                optional_params={},
                litellm_params={},
                api_key=None,
                api_base=None,
            )

        assert headers["authorization"] == f"Bearer {FAKE_OAUTH_TOKEN}"
        assert "x-api-key" not in headers
        assert headers["anthropic-dangerous-direct-browser-access"] == "true"


class TestGetAuthToken:
    """Tests for AnthropicModelInfo.get_auth_token() static method."""

    def test_returns_env_var_value(self):
        from litellm.llms.anthropic.common_utils import AnthropicModelInfo

        with mock_patch.dict(
            "os.environ", {"ANTHROPIC_AUTH_TOKEN": FAKE_AUTH_TOKEN}, clear=True
        ):
            assert AnthropicModelInfo.get_auth_token() == FAKE_AUTH_TOKEN

    def test_returns_none_when_not_set(self):
        from litellm.llms.anthropic.common_utils import AnthropicModelInfo

        with mock_patch.dict("os.environ", {}, clear=True):
            assert AnthropicModelInfo.get_auth_token() is None

    def test_explicit_param_takes_precedence(self):
        from litellm.llms.anthropic.common_utils import AnthropicModelInfo

        explicit_token = "sk-ant-aut01-explicit-token-override-123456789"
        assert AnthropicModelInfo.get_auth_token(explicit_token) == explicit_token


class TestGetAuthHeader:
    """Tests for AnthropicModelInfo.get_auth_header() centralized helper."""

    def test_returns_x_api_key_when_api_key_provided(self):
        from litellm.llms.anthropic.common_utils import AnthropicModelInfo

        result = AnthropicModelInfo.get_auth_header(api_key=FAKE_REGULAR_KEY)
        assert result == {"x-api-key": FAKE_REGULAR_KEY}

    def test_returns_x_api_key_from_env(self):
        from litellm.llms.anthropic.common_utils import AnthropicModelInfo

        with mock_patch.dict(
            "os.environ",
            {"ANTHROPIC_API_KEY": FAKE_REGULAR_KEY},
            clear=True,
        ):
            result = AnthropicModelInfo.get_auth_header()
            assert result == {"x-api-key": FAKE_REGULAR_KEY}

    def test_returns_bearer_from_auth_token_env(self):
        from litellm.llms.anthropic.common_utils import AnthropicModelInfo

        with mock_patch.dict(
            "os.environ",
            {"ANTHROPIC_AUTH_TOKEN": FAKE_AUTH_TOKEN},
            clear=True,
        ):
            result = AnthropicModelInfo.get_auth_header()
            assert result == {"authorization": f"Bearer {FAKE_AUTH_TOKEN}"}

    def test_api_key_takes_precedence_over_auth_token(self):
        from litellm.llms.anthropic.common_utils import AnthropicModelInfo

        with mock_patch.dict(
            "os.environ",
            {
                "ANTHROPIC_API_KEY": FAKE_REGULAR_KEY,
                "ANTHROPIC_AUTH_TOKEN": FAKE_AUTH_TOKEN,
            },
            clear=True,
        ):
            result = AnthropicModelInfo.get_auth_header()
            assert result == {"x-api-key": FAKE_REGULAR_KEY}

    def test_explicit_api_key_overrides_env_auth_token(self):
        from litellm.llms.anthropic.common_utils import AnthropicModelInfo

        with mock_patch.dict(
            "os.environ",
            {"ANTHROPIC_AUTH_TOKEN": FAKE_AUTH_TOKEN},
            clear=True,
        ):
            result = AnthropicModelInfo.get_auth_header(api_key=FAKE_REGULAR_KEY)
            assert result == {"x-api-key": FAKE_REGULAR_KEY}

    def test_returns_none_when_no_credentials(self):
        from litellm.llms.anthropic.common_utils import AnthropicModelInfo

        with mock_patch.dict("os.environ", {}, clear=True):
            result = AnthropicModelInfo.get_auth_header()
            assert result is None

    def test_oauth_token_uses_bearer_not_x_api_key(self):
        from litellm.llms.anthropic.common_utils import AnthropicModelInfo

        result = AnthropicModelInfo.get_auth_header(api_key=FAKE_OAUTH_TOKEN)
        assert result == {"authorization": f"Bearer {FAKE_OAUTH_TOKEN}"}

    def test_oauth_token_from_env_uses_bearer(self):
        from litellm.llms.anthropic.common_utils import AnthropicModelInfo

        with mock_patch.dict(
            "os.environ",
            {"ANTHROPIC_API_KEY": FAKE_OAUTH_TOKEN},
            clear=True,
        ):
            result = AnthropicModelInfo.get_auth_header()
            assert result == {"authorization": f"Bearer {FAKE_OAUTH_TOKEN}"}


class TestGetApiBaseFallbackChain:
    """Tests for AnthropicModelInfo.get_api_base() fallback to ANTHROPIC_BASE_URL."""

    def test_explicit_param_takes_precedence(self):
        from litellm.llms.anthropic.common_utils import AnthropicModelInfo

        assert (
            AnthropicModelInfo.get_api_base("https://explicit.example.com")
            == "https://explicit.example.com"
        )

    def test_defaults_to_anthropic_api(self):
        from litellm.llms.anthropic.common_utils import AnthropicModelInfo

        with mock_patch.dict("os.environ", {}, clear=True):
            assert AnthropicModelInfo.get_api_base() == "https://api.anthropic.com"

    def test_api_base_env_preferred_over_base_url_env(self):
        from litellm.llms.anthropic.common_utils import AnthropicModelInfo

        with mock_patch.dict(
            "os.environ",
            {
                "ANTHROPIC_API_BASE": "https://api-base.example.com",
                "ANTHROPIC_BASE_URL": "https://base-url.example.com",
            },
            clear=True,
        ):
            assert AnthropicModelInfo.get_api_base() == "https://api-base.example.com"

    def test_falls_back_to_base_url_env(self):
        from litellm.llms.anthropic.common_utils import AnthropicModelInfo

        with mock_patch.dict(
            "os.environ",
            {"ANTHROPIC_BASE_URL": "https://base-url.example.com"},
            clear=True,
        ):
            assert AnthropicModelInfo.get_api_base() == "https://base-url.example.com"
