"""
Unit tests for OpenRouter embedding transformation logic.
"""
from unittest.mock import MagicMock, patch

import pytest

import litellm
from litellm.llms.openrouter.common_utils import OpenRouterConfigError
from litellm.llms.openrouter.embedding.transformation import (
    OpenrouterEmbeddingConfig,
)


def test_openrouter_embedding_supported_params():
    """Test that supported OpenAI params are correctly defined."""
    config = OpenrouterEmbeddingConfig()
    supported = config.get_supported_openai_params("test-model")

    assert "timeout" in supported
    assert "dimensions" in supported
    assert "encoding_format" in supported
    assert "user" in supported


def test_openrouter_embedding_transform_request():
    """Test request transformation logic."""
    config = OpenrouterEmbeddingConfig()

    # Test with string input
    result = config.transform_embedding_request(
        model="openrouter/google/text-embedding-004",
        input="Hello world",
        optional_params={},
        headers={},
    )

    assert result["model"] == "google/text-embedding-004"
    assert result["input"] == ["Hello world"]

    # Test with list input
    result = config.transform_embedding_request(
        model="google/text-embedding-004",
        input=["Hello", "World"],
        optional_params={"dimensions": 512},
        headers={},
    )

    assert result["model"] == "google/text-embedding-004"
    assert result["input"] == ["Hello", "World"]
    assert result["dimensions"] == 512


def test_openrouter_embedding_validate_environment():
    """Test environment validation and header setup."""
    config = OpenrouterEmbeddingConfig()

    headers = config.validate_environment(
        headers={"Custom-Header": "value"},
        model="test-model",
        messages=[],
        optional_params={},
        litellm_params={},
        api_key="test-api-key",
    )

    # Should include OpenRouter-specific headers
    assert "HTTP-Referer" in headers
    assert "X-Title" in headers
    # Should include Content-Type header
    assert "Content-Type" in headers
    assert headers["Content-Type"] == "application/json"
    # Should include Authorization header
    assert "Authorization" in headers
    assert headers["Authorization"] == "Bearer test-api-key"
    # Should preserve custom headers
    assert headers["Custom-Header"] == "value"


def test_openrouter_embedding_preserves_valid_caller_authorization():
    """Test embedding adapter integration with caller-provided auth."""
    config = OpenrouterEmbeddingConfig()

    headers = config.validate_environment(
        headers={
            "authorization": "Bearer caller-key",
            "Custom-Header": "value",
        },
        model="test-model",
        messages=[],
        optional_params={},
        litellm_params={},
        api_key="explicit-key",
    )

    assert headers["authorization"] == "Bearer caller-key"
    assert "Authorization" not in headers
    assert headers["Custom-Header"] == "value"


def test_openrouter_embedding_missing_auth_prevents_http_dispatch(monkeypatch):
    """Test missing auth fails before the embedding HTTP request."""
    for env_name in (
        "OPENROUTER_API_KEY",
        "OR_API_KEY",
        "AAWM_OPENROUTER_API_KEY",
    ):
        monkeypatch.delenv(env_name, raising=False)
    monkeypatch.setattr(litellm, "api_key", None, raising=False)
    monkeypatch.setattr(litellm, "openrouter_key", None, raising=False)

    client = MagicMock()
    with patch(
        "litellm.secret_managers.main.get_secret_str",
        return_value=None,
    ), patch(
        "litellm.llms.custom_httpx.llm_http_handler._get_httpx_client",
        return_value=client,
    ):
        with pytest.raises(OpenRouterConfigError):
            litellm.embedding(
                model="openrouter/text-embedding-3-small",
                input=["hello"],
                client=client,
            )

    client.post.assert_not_called()


def test_openrouter_embedding_get_complete_url():
    """Test URL construction."""
    config = OpenrouterEmbeddingConfig()

    url = config.get_complete_url(
        api_base="https://openrouter.ai/api/v1",
        api_key="test-key",
        model="test-model",
        optional_params={},
        litellm_params={},
    )

    assert url == "https://openrouter.ai/api/v1/embeddings"

    # Test with trailing slash
    url = config.get_complete_url(
        api_base="https://openrouter.ai/api/v1/",
        api_key="test-key",
        model="test-model",
        optional_params={},
        litellm_params={},
    )

    assert url == "https://openrouter.ai/api/v1/embeddings"


def test_openrouter_embedding_map_params():
    """Test parameter mapping."""
    config = OpenrouterEmbeddingConfig()

    result = config.map_openai_params(
        non_default_params={"dimensions": 512, "timeout": 30, "unsupported": "value"},
        optional_params={},
        model="test-model",
        drop_params=False,
    )

    # Supported params should be included
    assert result["dimensions"] == 512
    assert result["timeout"] == 30
    # Unsupported params should not be included
    assert "unsupported" not in result
