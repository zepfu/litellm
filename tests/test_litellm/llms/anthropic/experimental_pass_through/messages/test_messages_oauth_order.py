"""RR-029: ambient ANTHROPIC_API_KEY OAuth tokens must become Bearer headers."""

from __future__ import annotations

from litellm.llms.anthropic.experimental_pass_through.messages.transformation import (
    AnthropicMessagesConfig,
)

FAKE_OAUTH = "sk-ant-oat01-fake-token-for-testing-123456789abcdef"
FAKE_KEY = "sk-ant-api03-regular-key-for-testing-123456789"


class TestMessagesOAuthAmbientCredentialOrder:
    def setup_method(self):
        self.config = AnthropicMessagesConfig()

    def test_oauth_env_key_becomes_bearer_before_x_api_key(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", FAKE_OAUTH)
        headers, _ = self.config.validate_anthropic_messages_environment(
            headers={},
            model="claude-sonnet-4-20250514",
            messages=[{"role": "user", "content": "hi"}],
            optional_params={},
            litellm_params={},
            api_key=None,
        )
        assert headers.get("authorization") == f"Bearer {FAKE_OAUTH}"
        assert "x-api-key" not in headers

    def test_regular_env_key_uses_x_api_key(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", FAKE_KEY)
        headers, _ = self.config.validate_anthropic_messages_environment(
            headers={},
            model="claude-sonnet-4-20250514",
            messages=[{"role": "user", "content": "hi"}],
            optional_params={},
            litellm_params={},
            api_key=None,
        )
        assert headers.get("x-api-key") == FAKE_KEY
        assert "authorization" not in {k.lower() for k in headers}
