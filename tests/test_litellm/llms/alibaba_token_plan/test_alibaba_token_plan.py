import json

import pytest

import litellm
from litellm.llms.alibaba_token_plan import (
    ALIBABA_TOKEN_PLAN_API_BASE,
    ALIBABA_TOKEN_PLAN_CHAT_COMPLETIONS_URL,
    ALIBABA_TOKEN_PLAN_MODEL_IDS,
    ALIBABA_TOKEN_PLAN_SETTINGS_FILE_ENV,
    AlibabaTokenPlanAuthenticationError,
    AlibabaTokenPlanChatConfig,
)
from litellm.types.utils import LlmProviders
from litellm.utils import ProviderConfigManager


def test_should_register_alibaba_token_plan_provider_and_config() -> None:
    assert LlmProviders.ALIBABA_TOKEN_PLAN.value == "alibaba_token_plan"
    assert "alibaba_token_plan" in litellm.openai_compatible_providers
    assert isinstance(
        litellm.AlibabaTokenPlanChatConfig(),
        AlibabaTokenPlanChatConfig,
    )
    assert isinstance(
        ProviderConfigManager.get_provider_chat_config(
            model="qwen3.8-max-preview",
            provider=LlmProviders.ALIBABA_TOKEN_PLAN,
        ),
        AlibabaTokenPlanChatConfig,
    )


def test_should_admit_only_the_six_token_plan_models() -> None:
    assert ALIBABA_TOKEN_PLAN_MODEL_IDS == {
        "qwen3.8-max-preview",
        "qwen3.7-plus",
        "qwen3.7-max",
        "qwen3.6-flash",
        "deepseek-v4-pro",
        "glm-5.2",
    }


@pytest.mark.parametrize(
    "model",
    (
        "qwen3.8-max",
        "qwen/qwen3.8-max-preview",
        "dashscope/qwen3.8-max-preview",
        "alibaba_token_plan/unknown",
    ),
)
def test_should_reject_non_token_plan_model_ids(model: str) -> None:
    with pytest.raises(ValueError, match="Unsupported Alibaba Token Plan model"):
        AlibabaTokenPlanChatConfig().get_complete_url(
            api_base=None,
            api_key=None,
            model=model,
            optional_params={},
            litellm_params={},
        )


def test_should_use_only_the_canonical_endpoint_and_existing_credential(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ALIBABA_KEY", "existing-token-plan-key")
    config = AlibabaTokenPlanChatConfig()

    api_base, api_key = config._get_openai_compatible_provider_info(
        "https://caller.invalid/v1",
        "caller-supplied-key",
    )
    complete_url = config.get_complete_url(
        api_base="https://caller.invalid/v1",
        api_key="caller-supplied-key",
        model="alibaba_token_plan/qwen3.8-max-preview",
        optional_params={},
        litellm_params={},
    )

    assert api_base == ALIBABA_TOKEN_PLAN_API_BASE
    assert api_key == "existing-token-plan-key"
    assert complete_url == ALIBABA_TOKEN_PLAN_CHAT_COMPLETIONS_URL


def test_should_return_a_bounded_missing_credential_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    monkeypatch.delenv("ALIBABA_KEY", raising=False)
    monkeypatch.setenv(
        ALIBABA_TOKEN_PLAN_SETTINGS_FILE_ENV,
        str(tmp_path / "missing-settings.json"),
    )

    with pytest.raises(AlibabaTokenPlanAuthenticationError) as exc_info:
        AlibabaTokenPlanChatConfig()._get_openai_compatible_provider_info(
            None,
            "caller-supplied-key",
        )

    rendered = str(exc_info.value)
    assert "existing ALIBABA_KEY credential" in rendered
    assert "caller-supplied-key" not in rendered
    assert "reauthor" not in rendered.lower()


def test_should_read_the_existing_qwen_settings_credential_in_place(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    env_key = "QWEN_CUSTOM_TOKEN_PLAN_KEY"
    settings_path = tmp_path / "settings.json"
    settings_path.write_text(
        json.dumps(
            {
                "env": {env_key: "existing-qwen-token-plan-key"},
                "modelProviders": {
                    "openai": [
                        {
                            "id": model_id,
                            "name": model_id,
                            "baseUrl": ALIBABA_TOKEN_PLAN_API_BASE,
                            "envKey": env_key,
                        }
                        for model_id in ALIBABA_TOKEN_PLAN_MODEL_IDS
                    ]
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.delenv("ALIBABA_KEY", raising=False)
    monkeypatch.setenv(ALIBABA_TOKEN_PLAN_SETTINGS_FILE_ENV, str(settings_path))

    api_base, api_key = (
        AlibabaTokenPlanChatConfig()._get_openai_compatible_provider_info(
            None,
            None,
        )
    )

    assert api_base == ALIBABA_TOKEN_PLAN_API_BASE
    assert api_key == "existing-qwen-token-plan-key"


def test_should_send_the_raw_provider_model_with_the_existing_key(
    monkeypatch: pytest.MonkeyPatch,
    respx_mock,
) -> None:
    monkeypatch.setenv("ALIBABA_KEY", "existing-token-plan-key")
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    respx_mock.post(ALIBABA_TOKEN_PLAN_CHAT_COMPLETIONS_URL).respond(
        json={
            "id": "chatcmpl-alibaba",
            "object": "chat.completion",
            "created": 1,
            "model": "qwen3.8-max-preview",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "done"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 3,
                "completion_tokens": 2,
                "total_tokens": 5,
            },
        }
    )

    response = litellm.completion(
        model="alibaba_token_plan/qwen3.8-max-preview",
        messages=[{"role": "user", "content": "hello"}],
        api_base="https://caller.invalid/v1",
        api_key="caller-supplied-key",
    )

    assert response.choices[0].message.content == "done"
    request = respx_mock.calls[0].request
    request_body = json.loads(request.content)
    assert str(request.url) == ALIBABA_TOKEN_PLAN_CHAT_COMPLETIONS_URL
    assert request.headers["Authorization"] == "Bearer existing-token-plan-key"
    assert request_body["model"] == "qwen3.8-max-preview"
    assert "alibaba_token_plan/" not in request.content.decode()
    assert "aawm-sota-alibaba" not in request.content.decode()


def test_should_load_all_token_plan_models_from_the_bundled_catalog(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")

    local_model_cost = litellm.get_model_cost_map(url="unused")

    for model_id in ALIBABA_TOKEN_PLAN_MODEL_IDS:
        model_info = local_model_cost[f"alibaba_token_plan/{model_id}"]
        assert model_info["litellm_provider"] == "alibaba_token_plan"
        assert model_info["mode"] == "chat"
        assert model_info["supports_function_calling"] is True
        assert model_info["supports_parallel_function_calling"] is True
        assert model_info["namespace_tool_function_adapters"]["collaboration"]
