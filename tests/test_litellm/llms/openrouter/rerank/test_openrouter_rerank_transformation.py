import importlib
import json
from unittest.mock import MagicMock

import httpx
import pytest

import litellm
from litellm.types.rerank import RerankResponse
from litellm.utils import ProviderConfigManager


OPENROUTER_RERANK_MODEL = "openrouter/cohere/rerank-4-pro"


def _get_openrouter_rerank_config_class():
    module = importlib.import_module("litellm.llms.openrouter.rerank.transformation")
    return getattr(module, "OpenRouterRerankConfig")


def test_openrouter_rerank_config_is_registered_for_rerank_provider() -> None:
    config_class = _get_openrouter_rerank_config_class()

    config = ProviderConfigManager.get_provider_rerank_config(
        model=OPENROUTER_RERANK_MODEL,
        provider=litellm.LlmProviders.OPENROUTER,
        api_base=None,
        present_version_params=[],
    )

    assert isinstance(config, config_class)


def test_openrouter_rerank_transform_request_strips_provider_prefix() -> None:
    config = _get_openrouter_rerank_config_class()()

    request_body = config.transform_rerank_request(
        model=OPENROUTER_RERANK_MODEL,
        optional_rerank_params={
            "query": "What is OpenRouter?",
            "documents": ["OpenRouter routes LLM calls.", "Unrelated text."],
            "top_n": 1,
            "return_documents": True,
        },
        headers={},
    )

    assert request_body["model"] == "cohere/rerank-4-pro"
    assert request_body["query"] == "What is OpenRouter?"
    assert request_body["documents"] == [
        "OpenRouter routes LLM calls.",
        "Unrelated text.",
    ]
    assert request_body["top_n"] == 1
    assert request_body["return_documents"] is True


def test_openrouter_rerank_uses_openrouter_endpoint_and_headers() -> None:
    config = _get_openrouter_rerank_config_class()()

    assert (
        config.get_complete_url(
            api_base="https://openrouter.ai/api/v1/",
            model=OPENROUTER_RERANK_MODEL,
        )
        == "https://openrouter.ai/api/v1/rerank"
    )

    headers = config.validate_environment(
        headers={"Custom-Header": "value"},
        model=OPENROUTER_RERANK_MODEL,
        api_key="test-openrouter-key",
    )

    assert headers["Authorization"] == "Bearer test-openrouter-key"
    assert headers["Content-Type"] == "application/json"
    assert headers["HTTP-Referer"]
    assert headers["X-Title"]
    assert headers["Custom-Header"] == "value"


def test_openrouter_rerank_transform_response_preserves_provider_usage_cost() -> None:
    config = _get_openrouter_rerank_config_class()()
    response_data = {
        "id": "or-rerank-response-1",
        "results": [
            {
                "index": 0,
                "relevance_score": 0.98,
                "document": {"text": "OpenRouter routes LLM calls."},
            }
        ],
        "meta": {"billed_units": {"search_units": 1}},
        "usage": {"total_tokens": 12, "cost": 0.0042},
    }
    raw_response = MagicMock(spec=httpx.Response)
    raw_response.json.return_value = response_data
    raw_response.status_code = 200
    raw_response.text = json.dumps(response_data)
    raw_response.headers = {}

    result = config.transform_rerank_response(
        model=OPENROUTER_RERANK_MODEL,
        raw_response=raw_response,
        model_response=RerankResponse(),
        logging_obj=MagicMock(),
    )

    assert result.id == "or-rerank-response-1"
    assert result.results[0]["index"] == 0
    assert result._hidden_params["openrouter_usage"]["cost"] == pytest.approx(0.0042)
    assert (
        result._hidden_params["additional_headers"][
            "llm_provider-x-litellm-response-cost"
        ]
        == pytest.approx(0.0042)
    )


def test_openrouter_rerank_transform_response_estimates_total_tokens() -> None:
    config = _get_openrouter_rerank_config_class()()
    response_data = {
        "id": "or-rerank-response-2",
        "results": [{"index": 0, "relevance_score": 0.98}],
        "usage": {"search_units": 1, "cost": 0.0025},
    }
    raw_response = MagicMock(spec=httpx.Response)
    raw_response.json.return_value = response_data
    raw_response.status_code = 200
    raw_response.text = json.dumps(response_data)
    raw_response.headers = {}

    result = config.transform_rerank_response(
        model=OPENROUTER_RERANK_MODEL,
        raw_response=raw_response,
        model_response=RerankResponse(),
        logging_obj=MagicMock(),
        request_data={
            "query": "Which document explains the proxy path?",
            "documents": [
                "Call OpenRouter directly with provider keys.",
                "Call the LiteLLM proxy with attribution headers.",
            ],
        },
    )

    assert result.meta is not None
    billed_units = result.meta["billed_units"]
    assert billed_units["search_units"] == 1
    assert billed_units["total_tokens"] is not None
    assert billed_units["total_tokens"] > 0
