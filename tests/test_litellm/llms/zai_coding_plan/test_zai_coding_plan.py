"""Focused tests for the dedicated Z.AI Coding Plan chat transport."""

from __future__ import annotations

import json

import pytest

import litellm
from litellm.llms.zai_coding_plan import (
    ZAI_CODING_PLAN_API_BASE,
    ZAI_CODING_PLAN_CHAT_COMPLETIONS_URL,
    ZAI_CODING_PLAN_MODEL_IDS,
    ZAI_CODING_PLAN_USER_AGENT_PREFIX,
    ZAICodingPlanApiBaseError,
    ZAICodingPlanAuthenticationError,
    ZAICodingPlanChatConfig,
)
from litellm.types.utils import LlmProviders
from litellm.utils import ProviderConfigManager


def test_should_register_zai_coding_plan_provider_and_config() -> None:
    assert LlmProviders.ZAI_CODING_PLAN.value == "zai_coding_plan"
    assert "zai_coding_plan" in litellm.openai_compatible_providers
    assert isinstance(litellm.ZAICodingPlanChatConfig(), ZAICodingPlanChatConfig)
    assert isinstance(
        ProviderConfigManager.get_provider_chat_config(
            model="glm-5.3",
            provider=LlmProviders.ZAI_CODING_PLAN,
        ),
        ZAICodingPlanChatConfig,
    )


def test_should_keep_supported_model_ids_as_the_documented_coding_plan_set() -> None:
    assert ZAI_CODING_PLAN_MODEL_IDS == {"glm-5.3", "glm-5-turbo", "glm-4.7"}


@pytest.mark.parametrize(
    "model",
    (
        "zai_coding_plan/glm-5.3",
        "zai_coding_plan/glm-5-turbo",
        "zai_coding_plan/glm-4.7",
        "glm-5.3",
        "glm-5-turbo",
        "glm-4.7",
    ),
)
def test_should_admit_documented_coding_plan_model_ids(model: str) -> None:
    url = ZAICodingPlanChatConfig().get_complete_url(
        api_base=None,
        api_key=None,
        model=model,
        optional_params={},
        litellm_params={},
    )
    assert url == ZAI_CODING_PLAN_CHAT_COMPLETIONS_URL
    assert ZAICodingPlanChatConfig._model_id(model) == model.split("/")[-1]


@pytest.mark.parametrize(
    "model",
    (
        "zai_coding_plan/",
        "zai_coding_plan/   ",
        "zai_coding_plan/glm/sub",
        "zai_coding_plan/glm-5.2",
        "zai_coding_plan/glm-5.1",
        "zai/glm-5.3",
        "openai/gpt-5",
        "sota-zai",
        "",
    ),
)
def test_should_reject_unknown_or_malformed_coding_plan_model_ids(model: str) -> None:
    with pytest.raises(ValueError, match="Unsupported Z.AI Coding Plan model"):
        ZAICodingPlanChatConfig().get_complete_url(
            api_base=None,
            api_key=None,
            model=model,
            optional_params={},
            litellm_params={},
        )


def test_should_resolve_provider_to_the_canonical_coding_plan_base(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ZAI_KEY", "coding-plan-key")
    monkeypatch.delenv("ZAI_API_KEY", raising=False)

    model, provider, api_key, api_base = litellm.get_llm_provider(
        model="zai_coding_plan/glm-5.3"
    )

    assert model == "glm-5.3"
    assert provider == "zai_coding_plan"
    assert api_key == "coding-plan-key"
    assert api_base == ZAI_CODING_PLAN_API_BASE
    assert api_base != "https://api.z.ai/api/paas/v4"


def test_should_prefer_zai_key_over_coding_plan_and_zhipu_aliases(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ZAI_KEY", "primary-coding-key")
    monkeypatch.setenv("ZAI_CODING_PLAN_API_KEY", "alias-coding-key")
    monkeypatch.setenv("ZHIPU_API_KEY", "zhipu-coding-key")
    monkeypatch.setenv("ZAI_API_KEY", "ordinary-zai-key")

    _, api_key = ZAICodingPlanChatConfig()._get_openai_compatible_provider_info(
        None,
        "caller-supplied-key",
    )

    assert api_key == "primary-coding-key"


def test_should_use_coding_plan_alias_when_zai_key_is_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("ZAI_KEY", raising=False)
    monkeypatch.setenv("ZAI_CODING_PLAN_API_KEY", "alias-coding-key")
    monkeypatch.setenv("ZHIPU_API_KEY", "zhipu-coding-key")
    monkeypatch.setenv("ZAI_API_KEY", "ordinary-zai-key")

    _, api_key = ZAICodingPlanChatConfig()._get_openai_compatible_provider_info(
        None,
        "caller-supplied-key",
    )

    assert api_key == "alias-coding-key"


def test_should_use_zhipu_key_only_as_coding_plan_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("ZAI_KEY", raising=False)
    monkeypatch.delenv("ZAI_CODING_PLAN_API_KEY", raising=False)
    monkeypatch.setenv("ZHIPU_API_KEY", "zhipu-coding-key")
    monkeypatch.setenv("ZAI_API_KEY", "ordinary-zai-key")

    _, api_key = ZAICodingPlanChatConfig()._get_openai_compatible_provider_info(
        None,
        "caller-supplied-key",
    )

    assert api_key == "zhipu-coding-key"


def test_should_not_silently_reuse_ordinary_zai_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("ZAI_KEY", raising=False)
    monkeypatch.delenv("ZAI_CODING_PLAN_API_KEY", raising=False)
    monkeypatch.delenv("ZHIPU_API_KEY", raising=False)
    monkeypatch.setenv("ZAI_API_KEY", "ordinary-zai-key")

    with pytest.raises(ZAICodingPlanAuthenticationError) as exc_info:
        ZAICodingPlanChatConfig()._get_openai_compatible_provider_info(
            None,
            "ordinary-zai-key",
        )

    rendered = str(exc_info.value)
    assert "ZAI_KEY" in rendered
    assert "ordinary-zai-key" not in rendered
    assert "recharge" not in rendered.lower()


@pytest.mark.parametrize(
    "api_base",
    (
        None,
        ZAI_CODING_PLAN_API_BASE,
        ZAI_CODING_PLAN_API_BASE + "/",
        ZAI_CODING_PLAN_CHAT_COMPLETIONS_URL,
        ZAI_CODING_PLAN_CHAT_COMPLETIONS_URL + "/",
    ),
)
def test_should_accept_canonical_coding_plan_api_base_prefix(api_base: str | None) -> None:
    url = ZAICodingPlanChatConfig().get_complete_url(
        api_base=api_base,
        api_key=None,
        model="zai_coding_plan/glm-5.3",
        optional_params={},
        litellm_params={},
    )
    assert url == ZAI_CODING_PLAN_CHAT_COMPLETIONS_URL


@pytest.mark.parametrize(
    "api_base",
    (
        "",
        "https://api.z.ai/api/paas/v4",
        "https://api.z.ai/api/paas/v4/chat/completions",
        "https://open.bigmodel.cn/api/coding/paas/v4",
        "https://open.bigmodel.cn/api/coding/paas/v4/chat/completions",
        "https://caller.invalid/v1",
        "http://api.z.ai/api/coding/paas/v4",
        "https://api.z.ai.evil/api/coding/paas/v4",
    ),
)
def test_should_reject_ordinary_paas_china_and_hostile_api_bases(api_base: str) -> None:
    with pytest.raises(ZAICodingPlanApiBaseError):
        ZAICodingPlanChatConfig().get_complete_url(
            api_base=api_base,
            api_key=None,
            model="zai_coding_plan/glm-5.3",
            optional_params={},
            litellm_params={},
        )


def test_should_default_thinking_and_map_glm53_reasoning_effort(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ZAI_KEY", "coding-plan-key")
    config = ZAICodingPlanChatConfig()
    params = config.map_openai_params(
        non_default_params={"reasoning_effort": "xhigh"},
        optional_params={},
        model="glm-5.3",
        drop_params=False,
    )
    request = config.transform_request(
        model="glm-5.3",
        messages=[{"role": "user", "content": "hello"}],
        optional_params=dict(params),
        litellm_params={},
        headers={},
    )

    assert request["model"] == "glm-5.3"
    assert "sota-zai" not in json.dumps(request)
    assert request["extra_body"]["thinking"] == {
        "type": "enabled",
        "clear_thinking": False,
    }
    assert request["reasoning_effort"] == "max"


@pytest.mark.parametrize(
    ("inbound", "expected"),
    (
        ("minimal", "low"),
        ("low", "low"),
        ("medium", "high"),
        ("high", "high"),
        ("xhigh", "max"),
        ("max", "max"),
    ),
)
def test_should_map_inbound_reasoning_effort_for_glm53(
    inbound: str,
    expected: str,
) -> None:
    params = ZAICodingPlanChatConfig().map_openai_params(
        non_default_params={"reasoning_effort": inbound},
        optional_params={},
        model="glm-5.3",
        drop_params=False,
    )
    assert params["reasoning_effort"] == expected


def test_should_preserve_reasoning_content_on_tool_continuation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ZAI_KEY", "coding-plan-key")
    messages = [
        {"role": "user", "content": "inspect"},
        {
            "role": "assistant",
            "content": None,
            "reasoning_content": "keep this thinking",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "read_file", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "ok"},
    ]

    request = ZAICodingPlanChatConfig().transform_request(
        model="glm-5.3",
        messages=messages,
        optional_params={"tools": [{"type": "function", "function": {"name": "read_file"}}]},
        litellm_params={},
        headers={},
    )

    assert request["messages"][1]["reasoning_content"] == "keep this thinking"
    assert request["extra_body"]["thinking"] == {
        "type": "enabled",
        "clear_thinking": False,
    }


def test_should_force_stream_usage_and_owned_user_agent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ZAI_KEY", "coding-plan-key")
    config = ZAICodingPlanChatConfig()
    request = config.transform_request(
        model="glm-5.3",
        messages=[{"role": "user", "content": "hello"}],
        optional_params={
            "stream": True,
            "stream_options": {"include_usage": False, "custom_option": "preserve"},
        },
        litellm_params={},
        headers={},
    )
    headers = config.validate_environment(
        headers={
            "User-Agent": "opencode/1.18.19",
            "X-Msh-Device-Id": "not-permitted",
            "x-opencode-client": "not-permitted",
        },
        model="glm-5.3",
        messages=[],
        optional_params={},
        litellm_params={},
        api_key="coding-plan-key",
    )

    assert request["stream_options"] == {
        "include_usage": True,
        "custom_option": "preserve",
    }
    assert headers["Authorization"] == "Bearer coding-plan-key"
    assert headers["User-Agent"].startswith(f"{ZAI_CODING_PLAN_USER_AGENT_PREFIX}/")
    assert "opencode" not in headers["User-Agent"]
    assert not any(name.lower().startswith("x-msh-") for name in headers)
    assert not any(name.lower().startswith("x-opencode-") for name in headers)


def test_should_send_the_raw_glm_id_to_the_coding_plan_url(
    monkeypatch: pytest.MonkeyPatch,
    respx_mock,
) -> None:
    monkeypatch.setenv("ZAI_KEY", "coding-plan-key")
    monkeypatch.setenv("ZAI_API_KEY", "ordinary-zai-key")
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    respx_mock.post(ZAI_CODING_PLAN_CHAT_COMPLETIONS_URL).respond(
        json={
            "id": "chatcmpl-zai-coding",
            "object": "chat.completion",
            "created": 1,
            "model": "glm-5.3",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "done",
                        "reasoning_content": "think first",
                    },
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
        model="zai_coding_plan/glm-5.3",
        messages=[{"role": "user", "content": "hello"}],
        api_key="ordinary-zai-key",
    )

    assert response.choices[0].message.content == "done"
    request = respx_mock.calls[0].request
    request_body = json.loads(request.content)
    assert str(request.url) == ZAI_CODING_PLAN_CHAT_COMPLETIONS_URL
    assert "api/paas/v4" not in str(request.url)
    assert request.headers["Authorization"] == "Bearer coding-plan-key"
    assert request.headers["User-Agent"].startswith(
        f"{ZAI_CODING_PLAN_USER_AGENT_PREFIX}/"
    )
    assert request_body["model"] == "glm-5.3"
    assert request_body["thinking"] == {"type": "enabled", "clear_thinking": False}
    assert "sota-zai" not in request.content.decode()
    assert "zai_coding_plan/" not in request.content.decode()


def test_should_load_coding_plan_models_from_the_bundled_catalog(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    local_model_cost = litellm.get_model_cost_map(url="unused")
    ordinary_glm47 = local_model_cost["zai/glm-4.7"]

    for model_id in ZAI_CODING_PLAN_MODEL_IDS:
        model_info = local_model_cost[f"zai_coding_plan/{model_id}"]
        pricing = model_info["provider_specific_entry"]["zai_coding_plan"][
            "aawm_reference_pricing"
        ]
        assert model_info["litellm_provider"] == "zai_coding_plan"
        assert model_info["mode"] == "chat"
        assert model_info["supports_function_calling"] is True
        assert model_info["supports_reasoning"] is True
        assert pricing["billing_mode"] == "zai_coding_plan_subscription"
        assert pricing["actual_invoice_cost_known"] is False
        assert model_info.get("input_cost_per_token") != ordinary_glm47["input_cost_per_token"]
        assert model_info.get("output_cost_per_token") != ordinary_glm47["output_cost_per_token"]
        assert "input_cost_per_token" not in model_info
        assert "output_cost_per_token" not in model_info
