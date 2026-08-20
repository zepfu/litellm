"""Codex adapter metadata for Z.AI Coding Plan chat completions."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from litellm.llms.zai_coding_plan.adapters.adapter import (
    ZAI_CODING_PLAN_CREDENTIAL_SENTINEL,
    normalize_zai_coding_plan_adapter_model_name,
    normalize_zai_coding_plan_custom_tool_outputs,
    prepare_codex_zai_coding_plan_adapter_route,
)
from litellm.llms.zai_coding_plan.chat.transformation import (
    ZAI_CODING_PLAN_API_BASE,
    ZAI_CODING_PLAN_CHAT_COMPLETIONS_URL,
)
from litellm.proxy.pass_through_endpoints.aawm_alias_routing import adapter_config
from litellm.proxy.pass_through_endpoints.aawm_alias_routing.config_schema import (
    CODEX_ONLY_ROUTE_FAMILIES,
    CODEX_TO_ANTHROPIC_ROUTE_FAMILY,
    REGISTERED_PROVIDERS,
    REGISTERED_ROUTE_FAMILIES,
)


_CODEX_ZAI_ROUTE_FAMILY = "codex_zai_coding_plan_chat_completions_adapter"


def _request() -> MagicMock:
    request = MagicMock()
    request.headers = {}
    request.scope = {}
    return request


@pytest.mark.parametrize(
    ("model", "expected"),
    (
        ("zai_coding_plan/glm-5.3", "zai_coding_plan/glm-5.3"),
        ("zai_coding_plan/glm-5-turbo", "zai_coding_plan/glm-5-turbo"),
        ("zai_coding_plan/glm-4.7", "zai_coding_plan/glm-4.7"),
        ("glm-5.3", None),
        ("zai/glm-5.3", None),
        ("zai_coding_plan/", None),
        ("zai_coding_plan/glm-5.2", None),
        ("zai_coding_plan/glm/sub", None),
        ("sota-zai", None),
    ),
)
def test_should_normalize_only_documented_prefixed_coding_plan_models(
    model: str,
    expected: str | None,
) -> None:
    assert normalize_zai_coding_plan_adapter_model_name(model) == expected


def test_should_register_codex_only_coding_plan_route_family() -> None:
    config = adapter_config.CODEX_ZAI_CODING_PLAN
    assert config.route_family == _CODEX_ZAI_ROUTE_FAMILY
    assert config.custom_llm_provider == "zai_coding_plan"
    assert config.credential_family == "zai_coding_plan"
    assert config.expected_target_family == "zai_coding_plan"
    assert config.target_endpoint_label == (
        "zai_coding_plan:/api/coding/paas/v4/chat/completions"
    )
    assert "zai_coding_plan" in REGISTERED_PROVIDERS
    assert _CODEX_ZAI_ROUTE_FAMILY in REGISTERED_ROUTE_FAMILIES
    assert _CODEX_ZAI_ROUTE_FAMILY in CODEX_ONLY_ROUTE_FAMILIES
    assert _CODEX_ZAI_ROUTE_FAMILY not in CODEX_TO_ANTHROPIC_ROUTE_FAMILY
    assert not hasattr(adapter_config, "ANTHROPIC_ZAI_CODING_PLAN")
    assert not any(
        name.startswith("ANTHROPIC_ZAI") or "anthropic_zai_coding_plan" in name
        for name in dir(adapter_config)
    )


@pytest.mark.asyncio
async def test_should_prepare_codex_adapter_metadata_and_coding_url() -> None:
    plan = await prepare_codex_zai_coding_plan_adapter_route(
        request=_request(),
        adapter_model="zai_coding_plan/glm-5.3",
        prepared_request_body={
            "model": "zai_coding_plan/glm-5.3",
            "input": "hello",
            "reasoning": {"effort": "max"},
            "stream": True,
        },
    )

    completion_kwargs = plan.perform_kwargs["completion_kwargs"]
    metadata = plan.prepared_request_body["litellm_metadata"]
    assert plan.config is adapter_config.CODEX_ZAI_CODING_PLAN
    assert plan.api_key == ZAI_CODING_PLAN_CREDENTIAL_SENTINEL
    assert plan.api_base == ZAI_CODING_PLAN_API_BASE
    assert plan.target_url == ZAI_CODING_PLAN_CHAT_COMPLETIONS_URL
    assert completion_kwargs["model"] == "glm-5.3"
    assert completion_kwargs["custom_llm_provider"] == "zai_coding_plan"
    assert completion_kwargs["num_retries"] == 0
    assert metadata["zai_coding_plan_upstream_model"] == "glm-5.3"
    assert metadata["billing_mode"] == "zai_coding_plan_subscription"
    assert metadata["actual_invoice_cost_known"] is False
    assert f"route:{_CODEX_ZAI_ROUTE_FAMILY}" in metadata["tags"]
    assert adapter_config.CODEX_ZAI_CODING_PLAN.tag_prefix in metadata["tags"]
    assert any(
        span.get("name") == adapter_config.CODEX_ZAI_CODING_PLAN.span_name
        for span in metadata["langfuse_spans"]
    )
    assert "sota-zai" not in json.dumps(completion_kwargs)
    assert "anthropic_zai_coding_plan" not in json.dumps(metadata)


def test_should_normalize_codex_custom_tool_outputs_to_function_shape() -> None:
    request_body = {
        "input": [
            {
                "type": "custom_tool_call_output",
                "call_id": "call_coding_plan",
                "output": "pwd",
            }
        ]
    }

    normalized = normalize_zai_coding_plan_custom_tool_outputs(request_body)

    assert normalized["input"][0]["type"] == "function_call_output"
    assert normalized["input"][0]["call_id"] == "call_coding_plan"
    assert request_body["input"][0]["type"] == "custom_tool_call_output"
