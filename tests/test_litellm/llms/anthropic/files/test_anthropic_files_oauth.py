"""
OAuth header handling for Anthropic Files API.

sk-ant-oat* tokens must use Authorization: Bearer, not x-api-key.
"""

from litellm.llms.anthropic.files.handler import AnthropicFilesHandler
from litellm.llms.anthropic.files.transformation import (
    ANTHROPIC_FILES_BETA_HEADER,
    AnthropicFilesConfig,
)

FAKE_OAUTH_TOKEN = "sk-ant-oat01-fake-token-for-testing-123456789abcdef"
FAKE_REGULAR_KEY = "sk-ant-api03-regular-key-for-testing-123456789"


class TestAnthropicFilesOAuthHeaders:
    def setup_method(self):
        self.config = AnthropicFilesConfig()

    def test_regular_api_key_uses_x_api_key(self):
        headers = self.config.validate_environment(
            headers={},
            model="",
            messages=[],
            optional_params={},
            litellm_params={},
            api_key=FAKE_REGULAR_KEY,
        )
        assert headers["x-api-key"] == FAKE_REGULAR_KEY
        assert "authorization" not in headers
        assert headers["anthropic-beta"] == ANTHROPIC_FILES_BETA_HEADER

    def test_oauth_key_uses_bearer_authorization(self):
        headers = self.config.validate_environment(
            headers={},
            model="",
            messages=[],
            optional_params={},
            litellm_params={},
            api_key=FAKE_OAUTH_TOKEN,
        )
        assert headers.get("authorization") == f"Bearer {FAKE_OAUTH_TOKEN}"
        assert "x-api-key" not in headers
        assert "oauth-2025-04-20" in headers.get("anthropic-beta", "")
        assert ANTHROPIC_FILES_BETA_HEADER in headers.get("anthropic-beta", "")


class TestAnthropicFilesHandlerOAuthHeaders:
    """Handler builds headers inline for batch-results file content."""

    def test_oauth_key_uses_bearer_authorization(self):
        from litellm.llms.anthropic.common_utils import optionally_handle_anthropic_oauth

        headers = {
            "accept": "application/json",
            "anthropic-version": "2023-06-01",
            "x-api-key": FAKE_OAUTH_TOKEN,
        }
        headers, _ = optionally_handle_anthropic_oauth(
            headers=headers, api_key=FAKE_OAUTH_TOKEN
        )
        assert headers.get("authorization") == f"Bearer {FAKE_OAUTH_TOKEN}"
        assert "x-api-key" not in headers

    def test_handler_afile_content_applies_oauth(self, monkeypatch):
        """Ensure afile_content passes OAuth headers to the HTTP client."""
        import asyncio
        from unittest.mock import AsyncMock, MagicMock

        handler = AnthropicFilesHandler()
        captured = {}

        async def fake_get(url, headers=None, **kwargs):
            captured["headers"] = headers
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.content = b""
            mock_resp.headers = {}
            mock_resp.request = MagicMock()
            mock_resp.raise_for_status = MagicMock()
            return mock_resp

        mock_client = MagicMock()
        mock_client.get = fake_get

        monkeypatch.setattr(
            "litellm.llms.anthropic.files.handler.get_async_httpx_client",
            lambda **kwargs: mock_client,
        )
        monkeypatch.setattr(
            handler,
            "_transform_anthropic_batch_results_to_openai_format",
            lambda content: content,
        )

        asyncio.run(
            handler.afile_content(
                file_content_request={"file_id": "batch_123"},
                api_key=FAKE_OAUTH_TOKEN,
                api_base="https://api.anthropic.com",
            )
        )

        assert captured["headers"].get("authorization") == f"Bearer {FAKE_OAUTH_TOKEN}"
        assert "x-api-key" not in captured["headers"]
