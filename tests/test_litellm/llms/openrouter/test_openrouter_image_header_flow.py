"""Compact image header flow tests for OpenRouter (D1-539/D1-540).

Covers:
- Public generation header forwarding (extra_headers passed to handler).
- Sync/async generation and edit handler pre-validation merge.
- Handler extra_headers precedence over optional_request_params.
- Non-auth header preservation.
- Exactly one Authorization key at transport.
- Malformed late-source auth causing no HTTP post.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from litellm.llms.custom_httpx.llm_http_handler import BaseLLMHTTPHandler
from litellm.llms.openrouter.common_utils import OpenRouterConfigError
from litellm.llms.openrouter.image_edit.transformation import (
    OpenRouterImageEditConfig,
)
from litellm.llms.openrouter.image_generation.transformation import (
    OpenRouterImageGenerationConfig,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_logging():
    m = MagicMock()
    m.pre_call = MagicMock()
    return m


def _mock_sync_client():
    client = MagicMock()
    response = MagicMock()
    response.status_code = 200
    response.json.return_value = {
        "choices": [{"message": {"content": "ok", "images": []}}],
        "usage": {},
    }
    client.post.return_value = response
    return client


def _mock_async_client():
    client = AsyncMock()
    response = MagicMock()
    response.status_code = 200
    response.json.return_value = {
        "choices": [{"message": {"content": "ok", "images": []}}],
        "usage": {},
    }
    client.post.return_value = response
    return client


def _patch_async_client(client):
    return patch(
        "litellm.llms.custom_httpx.llm_http_handler.get_async_httpx_client",
        return_value=client,
    )


def test_openrouter_image_validators_do_not_mutate_input_headers():
    generation_headers = {
        "authorization": "Bearer caller-token",
        "X-Custom": "generation",
    }
    generation_original = dict(generation_headers)
    generation_result = OpenRouterImageGenerationConfig().validate_environment(
        headers=generation_headers,
        model="openrouter/google/gemini-2.5-flash-image",
        messages=[],
        optional_params={},
        litellm_params={},
    )

    edit_headers = {
        "authorization": "Bearer caller-token",
        "X-Custom": "edit",
    }
    edit_original = dict(edit_headers)
    edit_result = OpenRouterImageEditConfig().validate_environment(
        headers=edit_headers,
        model="openrouter/google/gemini-2.5-flash-image",
    )

    assert generation_headers == generation_original
    assert generation_result is not generation_headers
    assert edit_headers == edit_original
    assert edit_result is not edit_headers


# ---------------------------------------------------------------------------
# Image generation handler: pre-validation merge
# ---------------------------------------------------------------------------


class TestImageGenerationHandlerMerge:
    def test_handler_extra_headers_win_over_optional_params(self):
        handler = BaseLLMHTTPHandler()
        config = OpenRouterImageGenerationConfig()
        client = _mock_sync_client()

        with patch(
            "litellm.llms.custom_httpx.llm_http_handler._get_httpx_client",
            return_value=client,
        ):
            handler.image_generation_handler(
                model="openrouter/google/gemini-2.5-flash-image",
                prompt="draw a cat",
                image_generation_provider_config=config,
                image_generation_optional_request_params={
                    "extra_headers": {"Authorization": "Bearer from-params"},
                },
                custom_llm_provider="openrouter",
                litellm_params={},
                logging_obj=_mock_logging(),
                timeout=30,
                extra_headers={"Authorization": "Bearer from-handler"},
                client=client,
            )

        call_kwargs = client.post.call_args
        sent_headers = call_kwargs.kwargs.get("headers") or call_kwargs[1].get("headers")
        auth_keys = [k for k in sent_headers if k.lower() == "authorization"]
        assert len(auth_keys) == 1
        assert sent_headers[auth_keys[0]] == "Bearer from-handler"

    def test_non_auth_headers_preserved(self):
        handler = BaseLLMHTTPHandler()
        config = OpenRouterImageGenerationConfig()
        client = _mock_sync_client()

        with patch(
            "litellm.llms.custom_httpx.llm_http_handler._get_httpx_client",
            return_value=client,
        ):
            handler.image_generation_handler(
                model="openrouter/google/gemini-2.5-flash-image",
                prompt="draw a cat",
                image_generation_provider_config=config,
                image_generation_optional_request_params={
                    "extra_headers": {"X-Custom": "keep-me"},
                },
                custom_llm_provider="openrouter",
                litellm_params={},
                logging_obj=_mock_logging(),
                timeout=30,
                extra_headers={"Authorization": "Bearer tok"},
                client=client,
            )

        call_kwargs = client.post.call_args
        sent_headers = call_kwargs.kwargs.get("headers") or call_kwargs[1].get("headers")
        assert sent_headers.get("X-Custom") == "keep-me"

    def test_malformed_auth_no_http_post(self):
        handler = BaseLLMHTTPHandler()
        config = OpenRouterImageGenerationConfig()
        client = _mock_sync_client()

        with patch(
            "litellm.llms.custom_httpx.llm_http_handler._get_httpx_client",
            return_value=client,
        ):
            with pytest.raises(OpenRouterConfigError):
                handler.image_generation_handler(
                    model="openrouter/google/gemini-2.5-flash-image",
                    prompt="draw a cat",
                    image_generation_provider_config=config,
                    image_generation_optional_request_params={},
                    custom_llm_provider="openrouter",
                    litellm_params={},
                    logging_obj=_mock_logging(),
                    timeout=30,
                    extra_headers={"Authorization": "Basic not-bearer"},
                    client=client,
                )

        client.post.assert_not_called()

    def test_differently_cased_auth_across_sources_rejected(self):
        handler = BaseLLMHTTPHandler()
        config = OpenRouterImageGenerationConfig()
        client = _mock_sync_client()

        with patch(
            "litellm.llms.custom_httpx.llm_http_handler._get_httpx_client",
            return_value=client,
        ):
            with pytest.raises(OpenRouterConfigError, match="multiple"):
                handler.image_generation_handler(
                    model="openrouter/google/gemini-2.5-flash-image",
                    prompt="draw a cat",
                    image_generation_provider_config=config,
                    image_generation_optional_request_params={
                        "extra_headers": {
                            "Authorization": "Bearer from-params"
                        },
                    },
                    custom_llm_provider="openrouter",
                    litellm_params={},
                    logging_obj=_mock_logging(),
                    timeout=30,
                    extra_headers={"authorization": "Bearer from-handler"},
                    client=client,
                )

        client.post.assert_not_called()

    def test_non_openrouter_validator_contract_and_handler_precedence(self):
        handler = BaseLLMHTTPHandler()
        config = MagicMock()
        client = _mock_sync_client()
        validated_headers = {}

        def validate_environment(**kwargs):
            validated_headers.update(kwargs["headers"])
            return dict(kwargs["headers"])

        config.validate_environment.side_effect = validate_environment
        config.get_complete_url.return_value = "https://example.test/images"
        config.transform_image_generation_request.return_value = {}
        config.use_multipart_form_data.return_value = False
        config.transform_image_generation_response.return_value = MagicMock()

        with patch(
            "litellm.llms.custom_httpx.llm_http_handler._get_httpx_client",
            return_value=client,
        ):
            handler.image_generation_handler(
                model="image-model",
                prompt="draw a cat",
                image_generation_provider_config=config,
                image_generation_optional_request_params={
                    "extra_headers": {
                        "X-Optional": "present",
                        "X-Shared": "optional",
                    },
                },
                custom_llm_provider="recraft",
                litellm_params={},
                logging_obj=_mock_logging(),
                timeout=30,
                extra_headers={
                    "X-Handler": "present",
                    "X-Shared": "handler",
                },
                client=client,
            )

        assert validated_headers == {
            "X-Optional": "present",
            "X-Shared": "optional",
        }
        sent_headers = client.post.call_args.kwargs["headers"]
        assert sent_headers == {
            "X-Optional": "present",
            "X-Handler": "present",
            "X-Shared": "handler",
        }


# ---------------------------------------------------------------------------
# Image generation handler async: pre-validation merge
# ---------------------------------------------------------------------------


class TestAsyncImageGenerationHandlerMerge:
    @pytest.mark.asyncio
    async def test_async_handler_extra_headers_win(self):
        handler = BaseLLMHTTPHandler()
        config = OpenRouterImageGenerationConfig()
        client = _mock_async_client()

        with _patch_async_client(client):
            await handler.async_image_generation_handler(
                model="openrouter/google/gemini-2.5-flash-image",
                prompt="draw a cat",
                image_generation_provider_config=config,
                image_generation_optional_request_params={
                    "extra_headers": {"Authorization": "Bearer from-params"},
                },
                custom_llm_provider="openrouter",
                litellm_params={},
                logging_obj=_mock_logging(),
                timeout=30,
                extra_headers={"Authorization": "Bearer from-handler"},
                client=client,
            )

        call_kwargs = client.post.call_args
        sent_headers = call_kwargs.kwargs.get("headers") or call_kwargs[1].get("headers")
        auth_keys = [k for k in sent_headers if k.lower() == "authorization"]
        assert len(auth_keys) == 1
        assert sent_headers[auth_keys[0]] == "Bearer from-handler"


# ---------------------------------------------------------------------------
# Image edit handler: pre-validation merge
# ---------------------------------------------------------------------------


class TestImageEditHandlerMerge:
    def test_sync_edit_merge_and_precedence(self):
        from litellm.types.router import GenericLiteLLMParams

        handler = BaseLLMHTTPHandler()
        config = OpenRouterImageEditConfig()
        client = _mock_sync_client()

        with patch(
            "litellm.llms.custom_httpx.llm_http_handler._get_httpx_client",
            return_value=client,
        ):
            handler.image_edit_handler(
                model="openrouter/google/gemini-2.5-flash-image",
                image=b"\x89PNG fake",
                prompt="edit this",
                image_edit_provider_config=config,
                image_edit_optional_request_params={
                    "extra_headers": {"Authorization": "Bearer from-params"},
                },
                custom_llm_provider="openrouter",
                litellm_params=GenericLiteLLMParams(),
                logging_obj=_mock_logging(),
                timeout=30,
                extra_headers={"Authorization": "Bearer from-handler"},
                client=client,
            )

        call_kwargs = client.post.call_args
        sent_headers = call_kwargs.kwargs.get("headers") or call_kwargs[1].get("headers")
        auth_keys = [k for k in sent_headers if k.lower() == "authorization"]
        assert len(auth_keys) == 1
        assert sent_headers[auth_keys[0]] == "Bearer from-handler"

    def test_sync_edit_malformed_no_post(self):
        from litellm.types.router import GenericLiteLLMParams

        handler = BaseLLMHTTPHandler()
        config = OpenRouterImageEditConfig()
        client = _mock_sync_client()

        with patch(
            "litellm.llms.custom_httpx.llm_http_handler._get_httpx_client",
            return_value=client,
        ):
            with pytest.raises(OpenRouterConfigError):
                handler.image_edit_handler(
                    model="openrouter/google/gemini-2.5-flash-image",
                    image=b"\x89PNG fake",
                    prompt="edit this",
                    image_edit_provider_config=config,
                    image_edit_optional_request_params={},
                    custom_llm_provider="openrouter",
                    litellm_params=GenericLiteLLMParams(),
                    logging_obj=_mock_logging(),
                    timeout=30,
                    extra_headers={"Authorization": ""},
                    client=client,
                )

        client.post.assert_not_called()


# ---------------------------------------------------------------------------
# Async image edit handler: pre-validation merge
# ---------------------------------------------------------------------------


class TestAsyncImageEditHandlerMerge:
    @pytest.mark.asyncio
    async def test_async_edit_merge_and_precedence(self):
        from litellm.types.router import GenericLiteLLMParams

        handler = BaseLLMHTTPHandler()
        config = OpenRouterImageEditConfig()
        client = _mock_async_client()

        with _patch_async_client(client):
            await handler.async_image_edit_handler(
                model="openrouter/google/gemini-2.5-flash-image",
                image=b"\x89PNG fake",
                prompt="edit this",
                image_edit_provider_config=config,
                image_edit_optional_request_params={
                    "extra_headers": {"Authorization": "Bearer from-params"},
                },
                custom_llm_provider="openrouter",
                litellm_params=GenericLiteLLMParams(),
                logging_obj=_mock_logging(),
                timeout=30,
                extra_headers={"Authorization": "Bearer from-handler"},
                client=client,
            )

        call_kwargs = client.post.call_args
        sent_headers = call_kwargs.kwargs.get("headers") or call_kwargs[1].get("headers")
        auth_keys = [k for k in sent_headers if k.lower() == "authorization"]
        assert len(auth_keys) == 1
        assert sent_headers[auth_keys[0]] == "Bearer from-handler"
