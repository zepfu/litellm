import pytest
from litellm.proxy.health_check import _update_litellm_params_for_health_check
from litellm.litellm_core_utils.health_check_helpers import HealthCheckHelpers
from unittest.mock import AsyncMock, patch, MagicMock


@pytest.mark.asyncio
async def test_update_litellm_params_max_tokens_default():
    """
    Test that max_tokens defaults to 1 for non-wildcard models.
    """
    model_info = {}
    litellm_params = {"model": "gpt-4"}

    updated_params = _update_litellm_params_for_health_check(model_info, litellm_params)

    assert updated_params["max_tokens"] == 1


@pytest.mark.asyncio
async def test_update_litellm_params_max_tokens_custom():
    """
    Test that max_tokens respects health_check_max_tokens from model_info.
    """
    model_info = {"health_check_max_tokens": 5}
    litellm_params = {"model": "gpt-4"}

    updated_params = _update_litellm_params_for_health_check(model_info, litellm_params)

    assert updated_params["max_tokens"] == 5


@pytest.mark.asyncio
async def test_update_litellm_params_max_tokens_wildcard():
    """
    Test that max_tokens does NOT default to 1 for wildcard models.
    """
    model_info = {}
    litellm_params = {"model": "openai/*"}

    updated_params = _update_litellm_params_for_health_check(model_info, litellm_params)

    # Should not be set to 1
    assert "max_tokens" not in updated_params or updated_params["max_tokens"] != 1


def test_nvidia_embedding_health_check_params_are_embedding_safe():
    model_params = {
        "model": "nvidia_nim/nvidia/nv-embed-v1",
        "api_key": "fake-key",
        "messages": [{"role": "user", "content": "test"}],
        "max_tokens": 1,
    }

    filtered_params = HealthCheckHelpers._get_embedding_health_check_params(
        model_params=model_params,
        custom_llm_provider="nvidia_nim",
    )

    assert "messages" not in filtered_params
    assert "max_tokens" not in filtered_params
    assert filtered_params["encoding_format"] == "float"
    assert filtered_params["input_type"] == "query"


def test_rerank_health_check_params_drop_max_tokens():
    model_params = {
        "model": "nvidia_nim/nvidia/nv-rerankqa-mistral-4b-v3",
        "api_key": "fake-key",
        "messages": [{"role": "user", "content": "test"}],
        "max_tokens": 1,
    }

    filtered_params = HealthCheckHelpers._get_rerank_health_check_params(
        model_params=model_params,
    )

    assert "messages" not in filtered_params
    assert "max_tokens" not in filtered_params


@pytest.mark.asyncio
async def test_ahealth_check_wildcard_models_respects_max_tokens():
    """
    Test that ahealth_check_wildcard_models respects max_tokens if passed,
    otherwise defaults to 10.
    """
    with patch(
        "litellm.litellm_core_utils.llm_request_utils.pick_cheapest_chat_models_from_llm_provider",
        return_value=["gpt-4o-mini"],
    ), patch("litellm.acompletion", new_callable=AsyncMock):
        # Test Case 1: No max_tokens passed, should default to 10
        model_params = {}
        await HealthCheckHelpers.ahealth_check_wildcard_models(
            model="openai/*",
            custom_llm_provider="openai",
            model_params=model_params,
            litellm_logging_obj=MagicMock(),
        )
        assert model_params["max_tokens"] == 10

        # Test Case 2: Custom health_check_max_tokens passed via model_params, should be respected
        model_params = {"max_tokens": 3}
        await HealthCheckHelpers.ahealth_check_wildcard_models(
            model="openai/*",
            custom_llm_provider="openai",
            model_params=model_params,
            litellm_logging_obj=MagicMock(),
        )
        assert model_params["max_tokens"] == 3
