import asyncio
import json
import os
import stat
import sys
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path


import httpx
import pytest
from respx import MockRouter

import litellm
from litellm import Choices, Message, ModelResponse, EmbeddingResponse, Usage
from litellm import completion
from unittest.mock import patch
from litellm.llms.xai.chat.transformation import XAIChatConfig, XAI_API_BASE
from base_llm_unit_tests import BaseReasoningLLMTests, BaseLLMChatTest


def test_xai_chat_config_get_openai_compatible_provider_info():
    config = XAIChatConfig()

    # Test with default values
    api_base, api_key = config._get_openai_compatible_provider_info(
        api_base=None, api_key=None
    )
    assert api_base == XAI_API_BASE
    assert api_key == os.environ.get("XAI_API_KEY")

    # Test with custom API key
    custom_api_key = "test_api_key"
    api_base, api_key = config._get_openai_compatible_provider_info(
        api_base=None, api_key=custom_api_key
    )
    assert api_base == XAI_API_BASE
    assert api_key == custom_api_key

    # Test with custom environment variables for api_base and api_key
    with patch.dict(
        "os.environ",
        {"XAI_API_BASE": "https://env.x.ai/v1", "XAI_API_KEY": "env_api_key"},
    ):
        api_base, api_key = config._get_openai_compatible_provider_info(None, None)
        assert api_base == "https://env.x.ai/v1"
        assert api_key == "env_api_key"


@pytest.mark.asyncio
async def test_xai_oauth_managed_auth_file_loads_scoped_token(tmp_path, monkeypatch):
    from litellm.llms.xai import oauth

    credential_path = tmp_path / "xai-oauth.json"
    credential_path.write_text(
        json.dumps(
            {
                "https://auth.x.ai::b1a00492-073a-47ea-816f-4c329264a828": {
                    "key": "managed-access-token",
                    "refresh_token": "refresh-token",
                    "expires_at": (
                        datetime.now(timezone.utc) + timedelta(hours=1)
                    ).isoformat(),
                    "oidc_client_id": "client-id",
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("LITELLM_XAI_OAUTH_AUTH_FILE", str(credential_path))

    assert await oauth.get_xai_oauth_access_token() == "managed-access-token"


@pytest.mark.asyncio
async def test_xai_oauth_managed_auth_file_is_required(monkeypatch):
    from litellm.llms.xai import oauth

    monkeypatch.delenv("LITELLM_XAI_OAUTH_AUTH_FILE", raising=False)

    with pytest.raises(ValueError, match="LITELLM_XAI_OAUTH_AUTH_FILE"):
        await oauth.get_xai_oauth_access_token()


@pytest.mark.asyncio
async def test_xai_oauth_managed_auth_file_refreshes_near_expiry(
    tmp_path,
    monkeypatch,
):
    from litellm.llms.xai import oauth

    credential_path = tmp_path / "xai-oauth.json"
    credential_path.write_text(
        json.dumps(
            {
                "key": "old-access-token",
                "refresh_token": "old-refresh-token",
                "expires_at": (
                    datetime.now(timezone.utc) + timedelta(seconds=30)
                ).isoformat(),
                "oidc_client_id": "client-id",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("LITELLM_XAI_OAUTH_AUTH_FILE", str(credential_path))
    monkeypatch.setenv("LITELLM_XAI_OAUTH_TOKEN_ENDPOINT", "https://auth.test/token")
    captured = {}

    class FakeAsyncClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, traceback):
            return None

        async def post(self, url, data, headers):
            captured["url"] = url
            captured["data"] = data
            captured["headers"] = headers
            return httpx.Response(
                200,
                json={
                    "access_token": "new-access-token",
                    "refresh_token": "new-refresh-token",
                    "expires_in": 3600,
                    "token_type": "Bearer",
                },
            )

    with patch("litellm.llms.xai.oauth.httpx.AsyncClient", FakeAsyncClient):
        assert await oauth.get_xai_oauth_access_token() == "new-access-token"

    assert captured["url"] == "https://auth.test/token"
    assert captured["data"] == {
        "grant_type": "refresh_token",
        "refresh_token": "old-refresh-token",
        "client_id": "client-id",
    }
    refreshed = json.loads(credential_path.read_text(encoding="utf-8"))
    assert refreshed["key"] == "new-access-token"
    assert refreshed["access_token"] == "new-access-token"
    assert refreshed["refresh_token"] == "new-refresh-token"


def test_xai_oauth_migrate_hermes_provider_tokens(tmp_path, monkeypatch):
    from litellm.llms.xai import oauth

    monkeypatch.delenv("LITELLM_XAI_OAUTH_SCOPE", raising=False)

    source_path = tmp_path / "hermes" / "auth.json"
    target_path = tmp_path / "litellm-owned" / "xai-oauth.json"
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_text(
        json.dumps(
            {
                "providers": {
                    "xai-oauth": {
                        "tokens": {
                            "access_token": "hermes-access-token",
                            "refresh_token": "hermes-refresh-token",
                            "id_token": "hermes-id-token",
                            "token_type": "Bearer",
                            "expires_in": 600,
                        },
                        "discovery": {
                            "token_endpoint": "https://auth.x.ai/oauth2/token"
                        },
                        "last_refresh": "2026-06-01T10:00:00Z",
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    migrated_path = oauth.migrate_hermes_xai_oauth_credential(
        hermes_auth_file=source_path,
        target_auth_file=target_path,
    )

    default_scope = "https://auth.x.ai::b1a00492-073a-47ea-816f-4c329264a828"
    migrated = json.loads(target_path.read_text(encoding="utf-8"))

    assert migrated_path == target_path
    assert list(migrated) == [default_scope]

    record = migrated[default_scope]
    assert record["key"] == "hermes-access-token"
    assert record["access_token"] == "hermes-access-token"
    assert record["refresh_token"] == "hermes-refresh-token"
    assert record["id_token"] == "hermes-id-token"
    assert record["token_type"] == "Bearer"
    assert record["token_endpoint"] == "https://auth.x.ai/oauth2/token"
    assert record["oidc_client_id"] == "b1a00492-073a-47ea-816f-4c329264a828"
    assert record["source"] == "hermes.providers.xai-oauth"
    assert record["expires_at"] == "2026-06-01T10:10:00Z"
    if os.name != "nt":
        assert stat.S_IMODE(target_path.stat().st_mode) == 0o600


def test_xai_oauth_migrate_hermes_credential_pool_falls_back_to_first_usable_record(
    tmp_path,
    monkeypatch,
):
    from litellm.llms.xai import oauth

    monkeypatch.delenv("LITELLM_XAI_OAUTH_SCOPE", raising=False)

    source_path = tmp_path / "hermes" / "auth.json"
    target_path = tmp_path / "litellm-owned" / "xai-oauth.json"
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_text(
        json.dumps(
            {
                "credential_pool": {
                    "xai-oauth": [
                        {
                            "base_url": "https://ignored.example",
                        },
                        {
                            "access_token": "pool-access-token",
                            "refresh_token": "pool-refresh-token",
                            "id_token": "pool-id-token",
                            "token_type": "Bearer",
                            "base_url": "https://chosen.example",
                            "last_refresh": "2026-06-01T11:00:00Z",
                        },
                    ]
                }
            }
        ),
        encoding="utf-8",
    )

    oauth.migrate_hermes_xai_oauth_credential(
        hermes_auth_file=source_path,
        target_auth_file=target_path,
    )

    default_scope = "https://auth.x.ai::b1a00492-073a-47ea-816f-4c329264a828"
    migrated = json.loads(target_path.read_text(encoding="utf-8"))

    assert list(migrated) == [default_scope]

    record = migrated[default_scope]
    assert record["key"] == "pool-access-token"
    assert record["access_token"] == "pool-access-token"
    assert record["refresh_token"] == "pool-refresh-token"
    assert record["id_token"] == "pool-id-token"
    assert record["token_type"] == "Bearer"
    assert record["token_endpoint"] == "https://auth.x.ai/oauth2/token"
    assert record["oidc_client_id"] == "b1a00492-073a-47ea-816f-4c329264a828"
    assert record["source"] == "hermes.credential_pool.xai-oauth"
    assert record["source_base_url"] == "https://chosen.example"
    assert record["source_last_refresh"] == "2026-06-01T11:00:00Z"


def test_xai_oauth_migration_rejects_target_under_hermes(tmp_path):
    from litellm.llms.xai import oauth

    source_path = tmp_path / "source" / "auth.json"
    target_path = tmp_path / ".hermes" / "xai-oauth.json"
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="outside the user's .hermes directory"):
        oauth.migrate_hermes_xai_oauth_credential(
            hermes_auth_file=source_path,
            target_auth_file=target_path,
        )


def test_xai_oauth_migration_rejects_target_source_path(tmp_path):
    from litellm.llms.xai import oauth

    source_path = tmp_path / "hermes" / "auth.json"
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="cannot be the Hermes source file"):
        oauth.migrate_hermes_xai_oauth_credential(
            hermes_auth_file=source_path,
            target_auth_file=source_path,
        )


@pytest.mark.asyncio
async def test_xai_oauth_managed_refresh_requires_refresh_token(
    tmp_path,
    monkeypatch,
):
    from litellm.llms.xai import oauth

    credential_path = tmp_path / "xai-oauth.json"
    credential_path.write_text(
        json.dumps(
            {
                "key": "old-access-token",
                "expires_at": (
                    datetime.now(timezone.utc) + timedelta(seconds=30)
                ).isoformat(),
                "oidc_client_id": "client-id",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("LITELLM_XAI_OAUTH_AUTH_FILE", str(credential_path))

    with pytest.raises(ValueError, match="Reseed or relogin"):
        await oauth.get_xai_oauth_access_token()


@pytest.mark.asyncio
async def test_xai_oauth_managed_auth_file_serializes_refresh(
    tmp_path,
    monkeypatch,
):
    from litellm.llms.xai import oauth

    credential_path = tmp_path / "xai-oauth.json"
    credential_path.write_text(
        json.dumps(
            {
                "key": "old-access-token",
                "refresh_token": "old-refresh-token",
                "expires_at": (
                    datetime.now(timezone.utc) + timedelta(seconds=30)
                ).isoformat(),
                "oidc_client_id": "client-id",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("LITELLM_XAI_OAUTH_AUTH_FILE", str(credential_path))
    monkeypatch.setenv("LITELLM_XAI_OAUTH_TOKEN_ENDPOINT", "https://auth.test/token")
    refresh_count = 0

    class FakeAsyncClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, traceback):
            return None

        async def post(self, url, data, headers):
            nonlocal refresh_count
            refresh_count += 1
            await asyncio.sleep(0)
            return httpx.Response(
                200,
                json={
                    "access_token": "new-access-token",
                    "refresh_token": "new-refresh-token",
                    "expires_in": 3600,
                },
            )

    with patch("litellm.llms.xai.oauth.httpx.AsyncClient", FakeAsyncClient):
        tokens = await asyncio.gather(
            oauth.get_xai_oauth_access_token(),
            oauth.get_xai_oauth_access_token(),
        )

    assert tokens == ["new-access-token", "new-access-token"]
    assert refresh_count == 1


def test_xai_oauth_model_mapping_and_metadata():
    from litellm.llms.xai import oauth

    upstream_model = oauth.resolve_oa_xai_upstream_model("oa_xai/grok-4.3")
    metadata = oauth.build_oa_xai_metadata("oa_xai/grok-4.3", upstream_model)

    assert upstream_model == "xai/grok-4.3"
    assert metadata["auth_mode"] == "oauth"
    assert metadata["credential_family"] == "xai_oauth"
    assert metadata["passthrough_route_family"] == "xai_oauth_api"
    assert metadata["shared_quota_family"] == "xai_grok_subscription"


def test_xai_chat_config_map_openai_params():
    """
    XAI is OpenAI compatible*

    Does not support all OpenAI parameters:
    - max_completion_tokens -> max_tokens

    """
    config = XAIChatConfig()

    # Test mapping of parameters
    non_default_params = {
        "max_completion_tokens": 100,
        "frequency_penalty": 0.5,
        "logit_bias": {"50256": -100},
        "logprobs": 5,
        "messages": [{"role": "user", "content": "Hello"}],
        "model": "xai/grok-beta",
        "n": 2,
        "presence_penalty": 0.2,
        "response_format": {"type": "json_object"},
        "seed": 42,
        "stop": ["END"],
        "stream": True,
        "stream_options": {},
        "temperature": 0.7,
        "tool_choice": "auto",
        "tools": [{"type": "function", "function": {"name": "get_weather"}}],
        "top_logprobs": 3,
        "top_p": 0.9,
        "user": "test_user",
        "unsupported_param": "value",
    }
    optional_params = {}
    model = "xai/grok-beta"

    result = config.map_openai_params(non_default_params, optional_params, model)

    # Assert all supported parameters are present in the result
    assert result["max_tokens"] == 100  # max_completion_tokens -> max_tokens
    assert result["frequency_penalty"] == 0.5
    assert result["logit_bias"] == {"50256": -100}
    assert result["logprobs"] == 5
    assert result["n"] == 2
    assert result["presence_penalty"] == 0.2
    assert result["response_format"] == {"type": "json_object"}
    assert result["seed"] == 42
    assert result["stop"] == ["END"]
    assert result["stream"] is True
    assert result["stream_options"] == {}
    assert result["temperature"] == 0.7
    assert result["tool_choice"] == "auto"
    assert result["tools"] == [
        {"type": "function", "function": {"name": "get_weather"}}
    ]
    assert result["top_logprobs"] == 3
    assert result["top_p"] == 0.9
    assert result["user"] == "test_user"

    # Assert unsupported parameter is not in the result
    assert "unsupported_param" not in result


def test_xai_check_for_stop_in_supported_params():
    supported_params = XAIChatConfig().get_supported_openai_params(
        model="xai/grok-3-mini"
    )
    assert "stop" not in supported_params


@pytest.mark.parametrize("model", ["xai/grok-4", "xai/grok-4-0709"])
def test_xai_grok_4_stop_not_supported(model):
    """
    Test that grok-4 models do not support the stop parameter

    Issue: https://github.com/BerriAI/litellm/issues/12635
    """
    supported_params = XAIChatConfig().get_supported_openai_params(model=model)
    assert "stop" not in supported_params


@pytest.mark.parametrize("model", ["xai/grok-4", "xai/grok-4-0709", "xai/grok-4-latest", "xai/grok-code-fast", "xai/grok-code-fast-1"])
def test_xai_grok_4_frequency_penalty_not_supported(model):
    """
    Test that grok-4 models do not support the frequency_penalty parameter
    """
    supported_params = XAIChatConfig().get_supported_openai_params(model=model)
    assert "frequency_penalty" not in supported_params



def test_xai_message_name_filtering():
    messages = [
        {
            "role": "system",
            "content": "*I press the green button*",
            "name": "example_user",
        },
        {"role": "user", "content": "Hello", "name": "John"},
        {"role": "assistant", "content": "Hello", "name": "Jane"},
    ]
    response = completion(
        model="xai/grok-3-mini-beta",
        messages=messages,
    )
    assert response is not None
    assert response.choices[0].message.content is not None


class TestXAIReasoningEffort(BaseReasoningLLMTests):
    def get_base_completion_call_args(self):
        return {
            "model": "xai/grok-3-mini-beta",
            "messages": [{"role": "user", "content": "Hello"}],
        }


class TestXAIChat(BaseLLMChatTest):
    def get_base_completion_call_args(self):
        return {
            "model": "xai/grok-3-mini-beta",
        }

    def test_tool_call_no_arguments(self, tool_call_no_arguments):
        """Test that tool calls with no arguments is translated correctly. Relevant issue: https://github.com/BerriAI/litellm/issues/6833"""
        pass

    def test_web_search(self):
        """Web search is only supported for Grok 4 family models"""
        from litellm.utils import supports_web_search

        os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
        litellm.model_cost = litellm.get_model_cost_map(url="")

        litellm._turn_on_debug()

        # Use grok-4-1-fast which supports web search
        model = "xai/grok-4-1-fast"

        if not supports_web_search(model, None):
            pytest.skip("Model does not support web search")

        response = completion(
            model=model,
            messages=[
                {"role": "user", "content": "What's the weather like in Boston today?"}
            ],
            web_search_options={},
            max_tokens=100,
        )

        assert response is not None


def test_xai_streaming_with_include_usage():
    """
    Test that xAI streaming correctly handles usage in the last chunk
    when stream_options={"include_usage": True} is set.
    
    xAI sends usage in a chunk with empty choices array, which should be
    handled by XAIChatCompletionStreamingHandler.
    """
    try:
        response = completion(
            model="xai/grok-4-1-fast-non-reasoning",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Say hello in one word"}
            ],
            stream=True,
            stream_options={"include_usage": True},
            max_tokens=10,
        )

        chunks = []
        usage_chunk = None
        
        for chunk in response:
            chunks.append(chunk)
            if hasattr(chunk, "usage") and chunk.usage is not None:
                usage_chunk = chunk
        
        # Verify we got chunks
        assert len(chunks) > 0, "Should receive streaming chunks"
        
        # Verify usage was included in one of the chunks
        assert usage_chunk is not None, "Should receive usage in streaming chunks"
        
        # Verify usage has expected fields
        assert hasattr(usage_chunk.usage, "prompt_tokens"), "Usage should have prompt_tokens"
        assert hasattr(usage_chunk.usage, "completion_tokens"), "Usage should have completion_tokens"
        assert hasattr(usage_chunk.usage, "total_tokens"), "Usage should have total_tokens"
        
        # Verify usage values are positive
        assert usage_chunk.usage.prompt_tokens > 0, "prompt_tokens should be positive"
        assert usage_chunk.usage.completion_tokens > 0, "completion_tokens should be positive"
        assert usage_chunk.usage.total_tokens > 0, "total_tokens should be positive"
        
        print(f"✓ Successfully received usage in streaming chunk: {usage_chunk.usage}")
        
    except Exception as e:
        if "API key" in str(e) or "authentication" in str(e).lower():
            pytest.skip(f"Skipping test due to API key issue: {str(e)}")
        raise
