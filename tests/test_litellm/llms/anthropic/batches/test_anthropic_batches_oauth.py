"""
OAuth header handling for Anthropic Batches API.

sk-ant-oat* tokens must use Authorization: Bearer, not x-api-key.
"""

from litellm.llms.anthropic.batches.transformation import AnthropicBatchesConfig

FAKE_OAUTH_TOKEN = "sk-ant-oat01-fake-token-for-testing-123456789abcdef"
FAKE_REGULAR_KEY = "sk-ant-api03-regular-key-for-testing-123456789"


class TestAnthropicBatchesOAuthHeaders:
    def setup_method(self):
        self.config = AnthropicBatchesConfig()

    def test_regular_api_key_uses_x_api_key(self):
        headers = self.config.validate_environment(
            headers={},
            model="claude-3-5-sonnet-20241022",
            messages=[],
            optional_params={},
            litellm_params={},
            api_key=FAKE_REGULAR_KEY,
        )
        assert headers["x-api-key"] == FAKE_REGULAR_KEY
        assert "authorization" not in headers

    def test_oauth_key_uses_bearer_authorization(self):
        headers = self.config.validate_environment(
            headers={},
            model="claude-3-5-sonnet-20241022",
            messages=[],
            optional_params={},
            litellm_params={},
            api_key=FAKE_OAUTH_TOKEN,
        )
        assert headers.get("authorization") == f"Bearer {FAKE_OAUTH_TOKEN}"
        assert "x-api-key" not in headers
        assert "oauth-2025-04-20" in headers.get("anthropic-beta", "")
        assert "message-batches-2024-09-24" in headers.get("anthropic-beta", "")
