"""Public auth-failure and resolver flow tests for OpenRouter (D1-539/D1-540).

The focused sync and async cases cover completion, embedding, image generation,
and image edit without duplicating the source-precedence matrix in
test_openrouter_common_utils.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import litellm
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
from litellm.llms.openrouter.common_utils import OpenRouterConfigError


@pytest.fixture(autouse=True)
def _no_keys(monkeypatch):
    """Ensure no ambient key source resolves so the malformed header path is
    exercised deterministically."""
    for var in ("OPENROUTER_API_KEY", "OR_API_KEY", "AAWM_OPENROUTER_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(litellm, "api_key", None, raising=False)
    monkeypatch.setattr(litellm, "openrouter_key", None, raising=False)
    with patch(
        "litellm.secret_managers.main.get_secret_str", return_value=None
    ):
        yield


def _mock_sync_client():
    client = MagicMock()
    return client


def _mock_async_client():
    client = object.__new__(AsyncHTTPHandler)
    client.post = AsyncMock()
    return client


def _run_executor_inline(monkeypatch):
    async def run_inline(executor, func):
        return func()

    monkeypatch.setattr(
        asyncio.get_running_loop(),
        "run_in_executor",
        run_inline,
    )


# ---------------------------------------------------------------------------
# completion / chat
# ---------------------------------------------------------------------------


class TestPublicCompletionAuth:
    def test_sync_completion_malformed_header_raises_config_error(self):
        client = _mock_sync_client()
        with patch(
            "litellm.llms.custom_httpx.llm_http_handler._get_httpx_client",
            return_value=client,
        ):
            with pytest.raises(OpenRouterConfigError):
                litellm.completion(
                    model="openrouter/google/gemini-2.5-flash",
                    messages=[{"role": "user", "content": "hi"}],
                    headers={"Authorization": "Basic not-bearer"},
                    client=client,
                )
        client.post.assert_not_called()

    def test_sync_completion_malformed_explicit_key_no_dispatch(self):
        client = _mock_sync_client()
        with patch(
            "litellm.llms.custom_httpx.llm_http_handler._get_httpx_client",
            return_value=client,
        ):
            with pytest.raises(OpenRouterConfigError):
                litellm.completion(
                    model="openrouter/google/gemini-2.5-flash",
                    messages=[{"role": "user", "content": "hi"}],
                    api_key="malformed explicit-key",
                    client=client,
                )
        client.post.assert_not_called()

    @pytest.mark.parametrize(
        "authorization",
        [
            "Bearer\ttoken",
            "Bearer\r\ntoken",
            "Bearer token\r\n",
            "Bearer token\t",
            "\tBearer token",
            "\x7fBearer token",
        ],
    )
    def test_sync_completion_raw_auth_controls_no_dispatch(
        self, authorization
    ):
        client = _mock_sync_client()
        with patch(
            "litellm.llms.custom_httpx.llm_http_handler._get_httpx_client",
            return_value=client,
        ), pytest.raises(
            OpenRouterConfigError,
            match="control or non-printable characters",
        ):
            litellm.completion(
                model="openrouter/google/gemini-2.5-flash",
                messages=[{"role": "user", "content": "hi"}],
                headers={"Authorization": authorization},
                client=client,
            )
        client.post.assert_not_called()

    @pytest.mark.asyncio
    async def test_async_completion_embedded_whitespace_no_dispatch(
        self, monkeypatch
    ):
        _run_executor_inline(monkeypatch)
        client = _mock_async_client()
        with pytest.raises(OpenRouterConfigError):
            await litellm.acompletion(
                model="openrouter/google/gemini-2.5-flash",
                messages=[{"role": "user", "content": "hi"}],
                headers={"Authorization": "Bearer two words"},
                max_retries=0,
                num_retries=0,
                timeout=1,
                client=client,
            )
        client.post.assert_not_awaited()

    def test_sync_completion_duplicate_header_raises_config_error(self):
        client = _mock_sync_client()
        with patch(
            "litellm.llms.custom_httpx.llm_http_handler._get_httpx_client",
            return_value=client,
        ):
            with pytest.raises(OpenRouterConfigError, match="multiple"):
                litellm.completion(
                    model="openrouter/google/gemini-2.5-flash",
                    messages=[{"role": "user", "content": "hi"}],
                    headers={
                        "Authorization": "Bearer a",
                        "authorization": "Bearer b",
                    },
                    client=client,
                )
        client.post.assert_not_called()


# ---------------------------------------------------------------------------
# embedding
# ---------------------------------------------------------------------------


class TestPublicEmbeddingAuth:
    def test_sync_embedding_malformed_header_raises_config_error(self):
        client = _mock_sync_client()
        with patch(
            "litellm.llms.custom_httpx.llm_http_handler._get_httpx_client",
            return_value=client,
        ):
            with pytest.raises(OpenRouterConfigError):
                litellm.embedding(
                    model="openrouter/text-embedding-3-small",
                    input=["hello"],
                    headers={"Authorization": "not-a-bearer"},
                    client=client,
                )
        client.post.assert_not_called()

    @pytest.mark.asyncio
    async def test_async_embedding_missing_auth_no_dispatch(self, monkeypatch):
        _run_executor_inline(monkeypatch)
        client = _mock_async_client()
        with pytest.raises(OpenRouterConfigError):
            await litellm.aembedding(
                model="openrouter/text-embedding-3-small",
                input=["hello"],
                max_retries=0,
                num_retries=0,
                timeout=1,
                client=client,
            )
        client.post.assert_not_awaited()


# ---------------------------------------------------------------------------
# image generation
# ---------------------------------------------------------------------------


class TestPublicImageGenerationAuth:
    def test_sync_image_generation_malformed_header_raises_config_error(self):
        client = _mock_sync_client()
        with patch(
            "litellm.llms.custom_httpx.llm_http_handler._get_httpx_client",
            return_value=client,
        ):
            with pytest.raises(OpenRouterConfigError):
                litellm.image_generation(
                    model="openrouter/google/gemini-2.5-flash-image",
                    prompt="a cat",
                    extra_headers={"Authorization": "Bearer "},
                    client=client,
                )
        client.post.assert_not_called()

    @pytest.mark.asyncio
    async def test_async_image_generation_malformed_auth_no_dispatch(
        self, monkeypatch
    ):
        _run_executor_inline(monkeypatch)
        client = _mock_async_client()
        with pytest.raises(OpenRouterConfigError):
            await litellm.aimage_generation(
                model="openrouter/google/gemini-2.5-flash-image",
                prompt="a cat",
                extra_headers={"Authorization": "Bearer token\tinjected"},
                max_retries=0,
                num_retries=0,
                timeout=1,
                client=client,
            )
        client.post.assert_not_awaited()


# ---------------------------------------------------------------------------
# image edit
# ---------------------------------------------------------------------------


class TestPublicImageEditAuth:
    def test_sync_image_edit_malformed_header_raises_config_error(self):
        client = _mock_sync_client()
        with patch(
            "litellm.llms.custom_httpx.llm_http_handler._get_httpx_client",
            return_value=client,
        ):
            with pytest.raises(OpenRouterConfigError):
                litellm.image_edit(
                    model="openrouter/google/gemini-2.5-flash-image",
                    image=b"\x89PNG fake",
                    prompt="edit",
                    extra_headers={"Authorization": "Token xyz"},
                    client=client,
                )
        client.post.assert_not_called()

    @pytest.mark.asyncio
    async def test_async_image_edit_malformed_auth_no_dispatch(
        self, monkeypatch
    ):
        _run_executor_inline(monkeypatch)
        client = _mock_async_client()
        with pytest.raises(OpenRouterConfigError):
            await litellm.aimage_edit(
                model="openrouter/google/gemini-2.5-flash-image",
                image=b"\x89PNG fake",
                prompt="edit",
                extra_headers={
                    "Authorization": "Bearer token\r\nInjected: header"
                },
                max_retries=0,
                num_retries=0,
                timeout=1,
                client=client,
            )
        client.post.assert_not_awaited()


# ---------------------------------------------------------------------------
# main.py regression: api_key fallback removed, shared resolver owns it
# ---------------------------------------------------------------------------


class TestMainPyApiKeyResolverRegression:
    """Verify that the duplicated api_key fallback blocks were removed from
    litellm/main.py and that key resolution now happens in the shared
    get_openrouter_auth_headers helper at validate_environment time."""

    def test_no_explicit_key_reaches_handler_as_none(self, monkeypatch):
        """When no api_key is passed, main.py should forward None (or the
        dynamic key) to the handler, not a pre-resolved env fallback."""
        monkeypatch.setenv("OR_API_KEY", "env-or-key")
        captured = {}

        original_validate = (
            litellm.llms.openrouter.chat.transformation.OpenrouterConfig.validate_environment
        )

        def spy_validate(self, headers, model, messages, optional_params,
                        litellm_params, api_key=None, api_base=None):
            captured["api_key"] = api_key
            # Provide a fallback key so the flow completes regardless of
            # the autouse _no_keys fixture patching away env sources.
            return original_validate(
                self, headers, model, messages, optional_params,
                litellm_params, api_key=api_key or "spy-fallback-key",
                api_base=api_base,
            )

        client = _mock_sync_client()
        response = MagicMock()
        response.status_code = 200
        response.json.return_value = {
            "id": "x", "object": "chat.completion", "created": 0,
            "model": "m", "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
        response.headers = {}
        client.post.return_value = response

        with patch(
            "litellm.llms.openrouter.chat.transformation.OpenrouterConfig.validate_environment",
            spy_validate,
        ), patch(
            "litellm.llms.custom_httpx.llm_http_handler._get_httpx_client",
            return_value=client,
        ):
            litellm.completion(
                model="openrouter/google/gemini-2.5-flash",
                messages=[{"role": "user", "content": "hi"}],
                client=client,
            )

        # api_key should be None (not pre-resolved from env) because the
        # fallback block was removed from main.py.
        assert captured["api_key"] is None

    def test_explicit_key_passed_through(self):
        """An explicit api_key must still reach validate_environment."""
        captured = {}

        original_validate = (
            litellm.llms.openrouter.chat.transformation.OpenrouterConfig.validate_environment
        )

        def spy_validate(self, headers, model, messages, optional_params,
                        litellm_params, api_key=None, api_base=None):
            captured["api_key"] = api_key
            return original_validate(
                self, headers, model, messages, optional_params,
                litellm_params, api_key=api_key, api_base=api_base,
            )

        client = _mock_sync_client()
        response = MagicMock()
        response.status_code = 200
        response.json.return_value = {
            "id": "x", "object": "chat.completion", "created": 0,
            "model": "m", "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
        response.headers = {}
        client.post.return_value = response

        with patch(
            "litellm.llms.openrouter.chat.transformation.OpenrouterConfig.validate_environment",
            spy_validate,
        ), patch(
            "litellm.llms.custom_httpx.llm_http_handler._get_httpx_client",
            return_value=client,
        ):
            litellm.completion(
                model="openrouter/google/gemini-2.5-flash",
                messages=[{"role": "user", "content": "hi"}],
                api_key="explicit-key-123",
                client=client,
            )

        assert captured["api_key"] == "explicit-key-123"
