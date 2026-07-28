import json
import os
import sys
from datetime import datetime
from unittest.mock import AsyncMock

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path


import httpx
import pytest
from respx import MockRouter
from unittest.mock import patch, MagicMock, AsyncMock

import litellm
from litellm import Choices, Message, ModelResponse, EmbeddingResponse, Usage
from litellm import completion
from base_rerank_unit_tests import BaseLLMRerankTest
from litellm.llms.nvidia_nim.embed import NvidiaNimEmbeddingConfig
import litellm


def test_completion_nvidia_nim():
    from openai import OpenAI

    litellm.set_verbose = True
    model_name = "nvidia_nim/databricks/dbrx-instruct"
    client = OpenAI(
        api_key="fake-api-key",
    )

    with patch.object(
        client.chat.completions.with_raw_response, "create"
    ) as mock_client:
        try:
            completion(
                model=model_name,
                messages=[
                    {
                        "role": "user",
                        "content": "What's the weather like in Boston today in Fahrenheit?",
                    }
                ],
                presence_penalty=0.5,
                frequency_penalty=0.1,
                client=client,
            )
        except Exception as e:
            print(e)
        # Add any assertions here to check the response

        mock_client.assert_called_once()
        request_body = mock_client.call_args.kwargs

        print("request_body: ", request_body)

        assert request_body["messages"] == [
            {
                "role": "user",
                "content": "What's the weather like in Boston today in Fahrenheit?",
            },
        ]
        assert request_body["model"] == "databricks/dbrx-instruct"
        assert request_body["frequency_penalty"] == 0.1
        assert request_body["presence_penalty"] == 0.5


def test_embedding_nvidia_nim():
    litellm.set_verbose = True
    from openai import OpenAI

    client = OpenAI(
        api_key="fake-api-key",
    )
    with patch.object(client.embeddings.with_raw_response, "create") as mock_client:
        try:
            litellm.embedding(
                model="nvidia_nim/nvidia/nv-embedqa-e5-v5",
                input="What is the meaning of life?",
                input_type="passage",
                dimensions=1024,
                client=client,
            )
        except Exception as e:
            print(e)
        mock_client.assert_called_once()
        request_body = mock_client.call_args.kwargs
        print("request_body: ", request_body)
        assert request_body["input"] == "What is the meaning of life?"
        assert request_body["model"] == "nvidia/nv-embedqa-e5-v5"
        assert request_body["extra_body"]["input_type"] == "passage"
        assert request_body["dimensions"] == 1024
        assert "encoding_format" not in request_body


def test_embedding_nvidia_nim_drops_health_check_max_tokens_from_extra_body():
    litellm.set_verbose = True
    from openai import OpenAI

    client = OpenAI(
        api_key="fake-api-key",
    )
    with patch.object(client.embeddings.with_raw_response, "create") as mock_client:
        try:
            litellm.embedding(
                model="nvidia_nim/nvidia/nv-embed-v1",
                input=["What is the meaning of life?"],
                input_type="query",
                max_tokens=1,
                client=client,
            )
        except Exception as e:
            print(e)
        mock_client.assert_called_once()
        request_body = mock_client.call_args.kwargs
        assert "encoding_format" not in request_body
        assert request_body["extra_body"]["input_type"] == "query"
        assert "max_tokens" not in request_body["extra_body"]


def test_chat_completion_nvidia_nim_with_tools():
    from openai import OpenAI

    litellm.set_verbose = True
    model_name = "nvidia_nim/meta/llama3-70b-instruct"
    client = OpenAI(
        api_key="fake-api-key",
    )

    # Define tools
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the current weather in a given location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA",
                        },
                        "unit": {
                            "type": "string",
                            "enum": ["celsius", "fahrenheit"],
                            "description": "The unit of temperature to use",
                        },
                    },
                    "required": ["location"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_current_time",
                "description": "Get the current time in a given timezone",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "timezone": {
                            "type": "string",
                            "description": "The timezone, e.g. EST, PST",
                        },
                    },
                    "required": ["timezone"],
                },
            },
        },
    ]

    with patch.object(
        client.chat.completions.with_raw_response, "create"
    ) as mock_client:
        try:
            completion(
                model=model_name,
                messages=[
                    {
                        "role": "user",
                        "content": "What's the weather like in Boston today and what time is it in EST?",
                    }
                ],
                tools=tools,
                tool_choice="auto",
                parallel_tool_calls=True,
                temperature=0.7,
                client=client,
            )
        except Exception as e:
            print(e)
        
        # Add assertions to check the request
        mock_client.assert_called_once()
        request_body = mock_client.call_args.kwargs

        print("request_body: ", request_body)

        assert request_body["messages"] == [
            {
                "role": "user",
                "content": "What's the weather like in Boston today and what time is it in EST?",
            },
        ]
        assert request_body["model"] == "meta/llama3-70b-instruct"
        assert request_body["temperature"] == 0.7
        assert request_body["tools"] == tools
        assert request_body["tool_choice"] == "auto"
        assert request_body["parallel_tool_calls"] == True

@pytest.mark.asyncio()
async def test_nvidia_nim_rerank_ranking_endpoint():
    """
    Test that using "nvidia_nim/ranking/<model>" forces the /v1/ranking endpoint.
    
    This allows users to explicitly use the /v1/ranking endpoint for models like
    nvidia/llama-3.2-nv-rerankqa-1b-v2.
    
    Reference: https://build.nvidia.com/nvidia/llama-3_2-nv-rerankqa-1b-v2/deploy
    """
    mock_response = AsyncMock()

    def return_val():
        return {
            "rankings": [
                {"index": 0, "logit": 0.95},
                {"index": 1, "logit": 0.75},
            ],
        }

    mock_response.json = return_val
    mock_response.headers = {"key": "value"}
    mock_response.status_code = 200

    with patch(
        "litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.post",
        return_value=mock_response,
    ) as mock_post:
        # Use "ranking/" prefix to force /v1/ranking endpoint
        response = await litellm.arerank(
            model="nvidia_nim/ranking/nvidia/llama-3.2-nv-rerankqa-1b-v2",
            query="What is the GPU memory bandwidth?",
            documents=["H100 delivers 3TB/s memory bandwidth", "A100 has 2TB/s memory bandwidth"],
            top_n=2,
            api_key="fake-api-key",
        )

        mock_post.assert_called_once()
        
        args_to_api = mock_post.call_args.kwargs["data"]
        _url = mock_post.call_args.kwargs["url"]
        print("url = ", _url)

        # Verify URL is /v1/ranking
        assert _url == "https://ai.api.nvidia.com/v1/ranking"

        # Verify request body structure
        request_data = json.loads(args_to_api)
        print("request_data=", request_data)

        # Query should be an object with 'text' field
        assert request_data["query"] == {"text": "What is the GPU memory bandwidth?"}

        # Documents should be 'passages'
        assert request_data["passages"] == [
            {"text": "H100 delivers 3TB/s memory bandwidth"},
            {"text": "A100 has 2TB/s memory bandwidth"},
        ]

        # Model name in body should NOT have "ranking/" prefix
        assert request_data["model"] == "nvidia/llama-3.2-nv-rerankqa-1b-v2"


@pytest.mark.asyncio()
async def test_nvidia_nim_rerank_uses_hosted_base_when_embedding_base_env_set(
    monkeypatch,
):
    mock_response = AsyncMock()
    mock_response.json = lambda: {"rankings": [{"index": 0, "logit": 0.95}]}
    mock_response.headers = {"key": "value"}
    mock_response.status_code = 200
    monkeypatch.setenv("NVIDIA_NIM_API_BASE", "https://integrate.api.nvidia.com/v1")
    monkeypatch.delenv("NVIDIA_NIM_RERANK_API_BASE", raising=False)

    with patch(
        "litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.post",
        return_value=mock_response,
    ) as mock_post:
        await litellm.arerank(
            model="nvidia_nim/nvidia/llama-3_2-nv-rerankqa-1b-v2",
            query="What is the GPU memory bandwidth?",
            documents=["H100 delivers 3TB/s memory bandwidth"],
            api_key="fake-api-key",
        )

        assert (
            mock_post.call_args.kwargs["url"]
            == "https://ai.api.nvidia.com/v1/retrieval/nvidia/llama-3_2-nv-rerankqa-1b-v2/reranking"
        )


@pytest.mark.asyncio()
async def test_nvidia_nim_rerank_mistral_models_use_shared_hosted_endpoint():
    mock_response = AsyncMock()
    mock_response.json = lambda: {"rankings": [{"index": 0, "logit": 0.95}]}
    mock_response.headers = {"key": "value"}
    mock_response.status_code = 200

    with patch(
        "litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.post",
        return_value=mock_response,
    ) as mock_post:
        await litellm.arerank(
            model="nvidia_nim/nvidia/nv-rerankqa-mistral-4b-v3",
            query="What is the GPU memory bandwidth?",
            documents=["H100 delivers 3TB/s memory bandwidth"],
            api_key="fake-api-key",
        )

        assert (
            mock_post.call_args.kwargs["url"]
            == "https://ai.api.nvidia.com/v1/retrieval/nvidia/reranking"
        )
        request_data = json.loads(mock_post.call_args.kwargs["data"])
        assert request_data["model"] == "nvidia/nv-rerankqa-mistral-4b-v3"

    with patch(
        "litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.post",
        return_value=mock_response,
    ) as mock_post:
        await litellm.arerank(
            model="nvidia_nim/nvidia/rerank-qa-mistral-4b",
            query="What is the GPU memory bandwidth?",
            documents=["H100 delivers 3TB/s memory bandwidth"],
            api_key="fake-api-key",
        )

        assert (
            mock_post.call_args.kwargs["url"]
            == "https://ai.api.nvidia.com/v1/retrieval/nvidia/reranking"
        )
        request_data = json.loads(mock_post.call_args.kwargs["data"])
        assert request_data["model"] == "nv-rerank-qa-mistral-4b:1"


class TestNvidiaNim(BaseLLMRerankTest):
    def get_custom_llm_provider(self) -> litellm.LlmProviders:
        return litellm.LlmProviders.NVIDIA_NIM

    def get_base_rerank_call_args(self) -> dict:
        return {
            "model": "nvidia_nim/nvidia/llama-3_2-nv-rerankqa-1b-v2",
        }
    
    def get_expected_cost(self) -> float:
        """Nvidia NIM rerank models are free (cost = 0.0)"""
        return 0.0


# ---------------------------------------------------------------------------
# D1-541: NvidiaNimEmbeddingConfig instance-local state isolation tests
# ---------------------------------------------------------------------------


class TestNvidiaNimEmbeddingConfigIsolation:
    """Prove constructor state is instance-local, not class-level."""

    def test_two_conflicting_instances_remain_isolated(self):
        a = NvidiaNimEmbeddingConfig(input_type="passage", truncate="END")
        b = NvidiaNimEmbeddingConfig(input_type="query", truncate="NONE")
        assert a.get_config() == {"input_type": "passage", "truncate": "END"}
        assert b.get_config() == {"input_type": "query", "truncate": "NONE"}

    def test_default_after_configured_has_no_stale_values(self):
        _configured = NvidiaNimEmbeddingConfig(
            encoding_format="float", user="u1", input_type="passage", truncate="END"
        )
        default = NvidiaNimEmbeddingConfig()
        assert default.get_config() == {}

    def test_concurrent_construction_no_cross_instance_mutation(self):
        import concurrent.futures

        def make(idx: int):
            return NvidiaNimEmbeddingConfig(input_type=f"type_{idx}")

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            instances = list(pool.map(make, range(64)))

        for idx, inst in enumerate(instances):
            assert inst.get_config() == {"input_type": f"type_{idx}"}

    def test_repeated_get_config_calls_are_stable(self):
        cfg = NvidiaNimEmbeddingConfig(input_type="passage", truncate="END")
        first = cfg.get_config()
        for _ in range(5):
            assert cfg.get_config() == first

    def test_config_names_are_instance_only_and_defaults_are_none(self):
        config_names = {"encoding_format", "user", "input_type", "truncate"}
        assert config_names.isdisjoint(vars(NvidiaNimEmbeddingConfig))

        default = NvidiaNimEmbeddingConfig()
        assert vars(default) == {
            "encoding_format": None,
            "user": None,
            "input_type": None,
            "truncate": None,
        }

    def test_class_get_config_call_remains_compatible(self):
        configured = NvidiaNimEmbeddingConfig(input_type="passage", truncate="END")

        assert NvidiaNimEmbeddingConfig.get_config() == {}
        assert configured.get_config() == {
            "input_type": "passage",
            "truncate": "END",
        }


class TestNvidiaNimEmbeddingConfigMapping:
    """Existing mapping behavior remains intact after instance-local refactor."""

    def test_map_openai_params_routes_input_type_and_truncate_to_extra_body(self):
        cfg = NvidiaNimEmbeddingConfig()
        result = cfg.map_openai_params(
            non_default_params={"input_type": "passage", "truncate": "END"},
            optional_params={},
        )
        assert result["extra_body"]["input_type"] == "passage"
        assert result["extra_body"]["truncate"] == "END"

    def test_map_openai_params_passes_dimensions_top_level(self):
        cfg = NvidiaNimEmbeddingConfig()
        result = cfg.map_openai_params(
            non_default_params={"dimensions": 1024},
            optional_params={},
        )
        assert result["dimensions"] == 1024
        assert "dimensions" not in result.get("extra_body", {})

    def test_map_openai_params_drops_max_tokens(self):
        cfg = NvidiaNimEmbeddingConfig()
        result = cfg.map_openai_params(
            non_default_params={"max_tokens": 1, "input_type": "query"},
            optional_params={},
        )
        assert "max_tokens" not in result
        assert "max_tokens" not in result.get("extra_body", {})
        assert result["extra_body"]["input_type"] == "query"

    def test_map_openai_params_kwargs_forwarded_to_extra_body(self):
        cfg = NvidiaNimEmbeddingConfig()
        result = cfg.map_openai_params(
            non_default_params={},
            optional_params={},
            kwargs={"custom_key": "custom_val", "max_tokens": 99},
        )
        assert result["extra_body"]["custom_key"] == "custom_val"
        assert "max_tokens" not in result["extra_body"]

    def test_get_supported_openai_params(self):
        cfg = NvidiaNimEmbeddingConfig()
        assert cfg.get_supported_openai_params() == [
            "encoding_format",
            "user",
            "dimensions",
        ]


class TestNvidiaNimEmbeddingConfigSubclassDescriptor:
    """Descriptor binding stays correct when the config is subclassed."""

    def test_subclass_class_access_and_instance_isolation(self):
        class Sub(NvidiaNimEmbeddingConfig):
            pass

        # Class-level access returns the empty default.
        assert Sub.get_config() == {}

        # Configured subclass instance exposes only its own state.
        configured = Sub(input_type="passage", truncate="END")
        assert configured.get_config() == {"input_type": "passage", "truncate": "END"}

        # A fresh default instance (subclass and parent) has no stale state.
        assert Sub().get_config() == {}
        assert NvidiaNimEmbeddingConfig().get_config() == {}


class TestNvidiaNimEmbeddingConfigLazySingleton:
    """Public litellm.nvidiaNimEmbeddingConfig lazy lookup compatibility."""

    def test_lazy_singleton_resolves_and_maps_params(self):
        singleton = litellm.nvidiaNimEmbeddingConfig

        # Resolves to a config instance with an empty default config.
        assert isinstance(singleton, NvidiaNimEmbeddingConfig)
        assert singleton.get_config() == {}

        # Basic mapping compatibility is preserved through the public lookup.
        result = singleton.map_openai_params(
            non_default_params={"input_type": "passage", "truncate": "END"},
            optional_params={},
        )
        assert result["extra_body"]["input_type"] == "passage"
        assert result["extra_body"]["truncate"] == "END"
