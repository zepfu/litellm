import json
import os
import sys
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

import litellm

sys.path.insert(
    0, os.path.abspath("../../..")
)  # Adds the parent directory to the system path

from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.proxy.pass_through_endpoints.llm_provider_handlers.openai_passthrough_logging_handler import (
    OpenAIPassthroughLoggingHandler,
    _MODEL_PRICE_DISK_MISS_CACHE,
    _MODEL_PRICE_MAP_CACHE,
    _reset_model_price_lookup_caches,
)
from litellm.proxy.pass_through_endpoints.success_handler import (
    PassThroughEndpointLogging,
)
from litellm.types.passthrough_endpoints.pass_through_endpoints import (
    PassthroughStandardLoggingPayload,
)


class TestOpenAIPassthroughLoggingHandler:
    """Test the OpenAI passthrough logging handler for cost tracking."""

    def setup_method(self):
        """Set up test fixtures"""
        self.start_time = datetime.now()
        self.end_time = datetime.now()
        self.handler = OpenAIPassthroughLoggingHandler()
        
        # Mock OpenAI chat completions response
        self.mock_openai_response = {
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "created": 1677652288,
            "model": "gpt-4o-2024-08-06",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "Hello! How can I help you today?"
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": 20,
                "completion_tokens": 15,
                "total_tokens": 35
            }
        }

    def _create_mock_logging_obj(self) -> LiteLLMLoggingObj:
        """Create a mock logging object"""
        mock_logging_obj = MagicMock()
        mock_logging_obj.model_call_details = {}
        return mock_logging_obj

    def _create_mock_httpx_response(self, response_data: dict = None) -> httpx.Response:
        """Create a mock httpx response"""
        if response_data is None:
            response_data = self.mock_openai_response
            
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.text = json.dumps(response_data)
        mock_response.json.return_value = response_data
        mock_response.headers = {"content-type": "application/json"}
        return mock_response

    def _create_passthrough_logging_payload(self, user: str = "test_user") -> PassthroughStandardLoggingPayload:
        """Create a mock passthrough logging payload"""
        return PassthroughStandardLoggingPayload(
            url="https://api.openai.com/v1/chat/completions",
            request_body={"model": "gpt-4o", "messages": [{"role": "user", "content": "Hello"}]},
            request_method="POST",
        )

    def test_llm_provider_name(self):
        """Test that the handler returns the correct provider name"""
        assert self.handler.llm_provider_name == "openai"

    def test_get_provider_config(self):
        """Test that the handler returns an OpenAI config"""
        handler = OpenAIPassthroughLoggingHandler()
        config = handler.get_provider_config(model="gpt-4o")
        assert config is not None
        # Verify it's an OpenAI config by checking if it has the expected methods
        assert hasattr(config, 'transform_response')

    def test_is_openai_chat_completions_route(self):
        """Test OpenAI chat completions route detection"""
        # Positive cases
        assert OpenAIPassthroughLoggingHandler.is_openai_chat_completions_route("https://api.openai.com/v1/chat/completions") is True
        assert OpenAIPassthroughLoggingHandler.is_openai_chat_completions_route("https://openai.azure.com/v1/chat/completions") is True
        assert OpenAIPassthroughLoggingHandler.is_openai_chat_completions_route("https://openrouter.ai/api/v1/chat/completions") is True
        assert OpenAIPassthroughLoggingHandler.is_openai_chat_completions_route("https://integrate.api.nvidia.com/v1/chat/completions") is True
        assert OpenAIPassthroughLoggingHandler.is_openai_chat_completions_route("https://cli-chat-proxy.grok.com/v1/chat/completions") is True
        
        # Negative cases
        assert OpenAIPassthroughLoggingHandler.is_openai_chat_completions_route("https://api.openai.com/v1/models") is False
        assert OpenAIPassthroughLoggingHandler.is_openai_chat_completions_route("http://localhost:4000/openai/v1/chat/completions") is False
        assert OpenAIPassthroughLoggingHandler.is_openai_chat_completions_route("https://api.anthropic.com/v1/messages") is False
        assert OpenAIPassthroughLoggingHandler.is_openai_chat_completions_route("") is False

    def test_is_openai_image_generation_route(self):
        """Test OpenAI image generation route detection"""
        # Positive cases
        assert OpenAIPassthroughLoggingHandler.is_openai_image_generation_route("https://api.openai.com/v1/images/generations") is True
        assert OpenAIPassthroughLoggingHandler.is_openai_image_generation_route("https://openai.azure.com/v1/images/generations") is True
        
        # Negative cases
        assert OpenAIPassthroughLoggingHandler.is_openai_image_generation_route("https://api.openai.com/v1/chat/completions") is False
        assert OpenAIPassthroughLoggingHandler.is_openai_image_generation_route("https://api.openai.com/v1/images/edits") is False
        assert OpenAIPassthroughLoggingHandler.is_openai_image_generation_route("http://localhost:4000/openai/v1/images/generations") is False
        assert OpenAIPassthroughLoggingHandler.is_openai_image_generation_route("") is False

    def test_is_openai_image_editing_route(self):
        """Test OpenAI image editing route detection"""
        # Positive cases
        assert OpenAIPassthroughLoggingHandler.is_openai_image_editing_route("https://api.openai.com/v1/images/edits") is True
        assert OpenAIPassthroughLoggingHandler.is_openai_image_editing_route("https://openai.azure.com/v1/images/edits") is True
        
        # Negative cases
        assert OpenAIPassthroughLoggingHandler.is_openai_image_editing_route("https://api.openai.com/v1/chat/completions") is False
        assert OpenAIPassthroughLoggingHandler.is_openai_image_editing_route("https://api.openai.com/v1/images/generations") is False
        assert OpenAIPassthroughLoggingHandler.is_openai_image_editing_route("http://localhost:4000/openai/v1/images/edits") is False
        assert OpenAIPassthroughLoggingHandler.is_openai_image_editing_route("") is False

    def test_is_openai_responses_route(self):
        """Test OpenAI responses API route detection"""
        # Positive cases
        assert OpenAIPassthroughLoggingHandler.is_openai_responses_route("https://api.openai.com/v1/responses") is True
        assert OpenAIPassthroughLoggingHandler.is_openai_responses_route("https://openai.azure.com/v1/responses") is True
        assert OpenAIPassthroughLoggingHandler.is_openai_responses_route("https://api.openai.com/responses") is True
        assert OpenAIPassthroughLoggingHandler.is_openai_responses_route("https://openrouter.ai/api/v1/responses") is True
        assert OpenAIPassthroughLoggingHandler.is_openai_responses_route("https://integrate.api.nvidia.com/v1/responses") is True
        assert OpenAIPassthroughLoggingHandler.is_openai_responses_route("https://api.x.ai/v1/responses") is True
        
        # Negative cases
        assert OpenAIPassthroughLoggingHandler.is_openai_responses_route("https://api.openai.com/v1/chat/completions") is False
        assert OpenAIPassthroughLoggingHandler.is_openai_responses_route("https://api.openai.com/v1/images/generations") is False
        assert OpenAIPassthroughLoggingHandler.is_openai_responses_route("http://localhost:4000/openai/v1/responses") is False
        assert OpenAIPassthroughLoggingHandler.is_openai_responses_route("") is False

    @patch('litellm.completion_cost')
    @patch('litellm.litellm_core_utils.litellm_logging.get_standard_logging_object_payload')
    def test_openai_passthrough_handler_success(self, mock_get_standard_logging, mock_completion_cost):
        """Test successful cost tracking for OpenAI chat completions"""
        # Arrange
        mock_completion_cost.return_value = 0.000045
        mock_get_standard_logging.return_value = {"test": "logging_payload"}
        
        mock_httpx_response = self._create_mock_httpx_response()
        mock_logging_obj = self._create_mock_logging_obj()
        passthrough_payload = self._create_passthrough_logging_payload()
        
        kwargs = {
            "passthrough_logging_payload": passthrough_payload,
            "model": "gpt-4o",
        }

        # Act
        result = OpenAIPassthroughLoggingHandler.openai_passthrough_handler(
            httpx_response=mock_httpx_response,
            response_body=self.mock_openai_response,
            logging_obj=mock_logging_obj,
            url_route="https://api.openai.com/v1/chat/completions",
            result="",
            start_time=self.start_time,
            end_time=self.end_time,
            cache_hit=False,
            request_body={"model": "gpt-4o", "messages": [{"role": "user", "content": "Hello"}]},
            **kwargs
        )

        # Assert
        assert result is not None
        assert "result" in result
        assert "kwargs" in result
        assert result["kwargs"]["response_cost"] == 0.000045
        assert result["kwargs"]["model"] == "gpt-4o"
        assert result["kwargs"]["custom_llm_provider"] == "openai"
        
        # Verify cost calculation was called
        mock_completion_cost.assert_called_once()
        
        # Verify logging object was updated
        assert mock_logging_obj.model_call_details["response_cost"] == 0.000045
        assert mock_logging_obj.model_call_details["model"] == "gpt-4o"
        assert mock_logging_obj.model_call_details["custom_llm_provider"] == "openai"

    @patch('litellm.completion_cost')
    def test_openai_passthrough_handler_non_chat_completions(self, mock_completion_cost):
        """Test that non-chat-completions routes fall back to base handler"""
        # Arrange
        mock_httpx_response = self._create_mock_httpx_response()
        mock_logging_obj = self._create_mock_logging_obj()
        passthrough_payload = self._create_passthrough_logging_payload()
        
        kwargs = {
            "passthrough_logging_payload": passthrough_payload,
            "model": "gpt-4o",
        }

        # Act - Use a non-chat-completions route
        result = OpenAIPassthroughLoggingHandler.openai_passthrough_handler(
            httpx_response=mock_httpx_response,
            response_body={"id": "file-123", "object": "file"},
            logging_obj=mock_logging_obj,
            url_route="https://api.openai.com/v1/files",
            result="",
            start_time=self.start_time,
            end_time=self.end_time,
            cache_hit=False,
            request_body={"purpose": "fine-tune"},
            **kwargs
        )

        # Assert - Should fall back to base handler for non-chat-completions
        assert result is not None
        assert "result" in result
        assert "kwargs" in result
        # Cost calculation may be called by the base handler fallback
        # The important thing is that our specific OpenAI handler logic didn't run

    @patch('litellm.completion_cost')
    @patch('litellm.litellm_core_utils.litellm_logging.get_standard_logging_object_payload')
    def test_openai_passthrough_handler_uses_cleaned_request_headers(
        self, mock_get_standard_logging, mock_completion_cost
    ):
        """OpenAI passthrough logging should overwrite proxy_server_request headers with cleaned request headers."""
        mock_completion_cost.return_value = 0.000045
        mock_get_standard_logging.return_value = {"test": "logging_payload"}

        mock_httpx_response = self._create_mock_httpx_response()
        mock_logging_obj = self._create_mock_logging_obj()
        passthrough_payload = PassthroughStandardLoggingPayload(
            url="https://api.openai.com/v1/chat/completions",
            request_body={"model": "gpt-4o", "messages": [{"role": "user", "content": "Hello"}]},
            request_method="POST",
            request_headers={"x-trace-id": "trace-123"},
        )

        kwargs = {
            "passthrough_logging_payload": passthrough_payload,
            "model": "gpt-4o",
            "litellm_params": {
                "proxy_server_request": {
                    "headers": {"authorization": "Bearer raw-token"}
                }
            },
        }

        result = OpenAIPassthroughLoggingHandler.openai_passthrough_handler(
            httpx_response=mock_httpx_response,
            response_body=self.mock_openai_response,
            logging_obj=mock_logging_obj,
            url_route="https://api.openai.com/v1/chat/completions",
            result="",
            start_time=self.start_time,
            end_time=self.end_time,
            cache_hit=False,
            request_body={"model": "gpt-4o", "messages": [{"role": "user", "content": "Hello"}]},
            **kwargs
        )

        proxy_server_request = result["kwargs"]["litellm_params"]["proxy_server_request"]
        assert proxy_server_request["headers"] == {"x-trace-id": "trace-123"}

    def test_is_openai_responses_route_accepts_chatgpt_codex_backend(self):
        assert OpenAIPassthroughLoggingHandler.is_openai_responses_route(
            "https://chatgpt.com/backend-api/codex/responses"
        )

    def test_openai_passthrough_handler_normalizes_responses_api_usage(self):
        codex_response = {
            "id": "resp-codex-usage-test",
            "object": "response",
            "created": 1775863780,
            "model": "gpt-5.4",
            "output": [
                {
                    "type": "function_call",
                    "call_id": "call_123",
                    "id": "call_123",
                    "name": "apply_patch",
                    "arguments": "{}",
                }
            ],
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "usage probe"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "input_tokens": 15517,
                "output_tokens": 27,
                "total_tokens": 15544,
                "input_tokens_details": {"cached_tokens": 3456},
                "output_tokens_details": {"reasoning_tokens": 0, "text_tokens": 27},
            },
        }
        mock_httpx_response = self._create_mock_httpx_response(codex_response)
        mock_logging_obj = self._create_mock_logging_obj()

        result = OpenAIPassthroughLoggingHandler.openai_passthrough_handler(
            httpx_response=mock_httpx_response,
            response_body=codex_response,
            logging_obj=mock_logging_obj,
            url_route="https://chatgpt.com/backend-api/codex/responses",
            result="",
            start_time=self.start_time,
            end_time=self.end_time,
            cache_hit=False,
            request_body={"model": "gpt-5.4", "input": "usage probe"},
            passthrough_logging_payload=PassthroughStandardLoggingPayload(
                url="https://chatgpt.com/backend-api/codex/responses",
                request_body={"model": "gpt-5.4", "input": "usage probe"},
                request_method="POST",
            ),
            litellm_params={},
        )

        usage = result["result"].usage
        standard_logging_object = result["kwargs"]["standard_logging_object"]
        assert usage.prompt_tokens == 15517
        assert usage.completion_tokens == 27
        assert usage.total_tokens == 15544
        assert usage.prompt_tokens_details.cached_tokens == 3456
        assert usage.completion_tokens_details.text_tokens == 27
        assert (
            standard_logging_object["metadata"]["usage_object"][
                "prompt_tokens_details"
            ]["cached_tokens"]
            == 3456
        )
        assert (
            standard_logging_object["response"]["usage"]["prompt_tokens_details"][
                "cached_tokens"
            ]
            == 3456
        )
        assert standard_logging_object["model"] == "gpt-5.4"
        assert standard_logging_object["prompt_tokens"] == 15517
        assert standard_logging_object["completion_tokens"] == 27
        assert standard_logging_object["total_tokens"] == 15544
        metadata = result["kwargs"]["litellm_params"].get("metadata", {})
        assert "langfuse_spans" not in metadata
        assert result["result"]._hidden_params["responses_output"] == codex_response["output"]

    def test_openai_passthrough_handler_preserves_non_streaming_codex_output_items(self):
        codex_response = {
            "id": "resp-codex-tool-test",
            "object": "response",
            "created": 1775863780,
            "model": "gpt-5.4",
            "output": [
                {
                    "type": "local_shell_call",
                    "call_id": "shell_123",
                    "id": "shell_123",
                    "input": {"command": "pwd"},
                }
            ],
            "usage": {
                "input_tokens": 20,
                "output_tokens": 5,
                "total_tokens": 25,
                "input_tokens_details": {"cached_tokens": 0},
                "output_tokens_details": {"reasoning_tokens": 0, "text_tokens": 5},
            },
        }
        mock_httpx_response = self._create_mock_httpx_response(codex_response)
        mock_logging_obj = self._create_mock_logging_obj()

        result = OpenAIPassthroughLoggingHandler.openai_passthrough_handler(
            httpx_response=mock_httpx_response,
            response_body=codex_response,
            logging_obj=mock_logging_obj,
            url_route="https://chatgpt.com/backend-api/codex/responses",
            result="",
            start_time=self.start_time,
            end_time=self.end_time,
            cache_hit=False,
            request_body={"model": "gpt-5.4", "input": "run pwd once"},
            passthrough_logging_payload=PassthroughStandardLoggingPayload(
                url="https://chatgpt.com/backend-api/codex/responses",
                request_body={"model": "gpt-5.4", "input": "run pwd once"},
                request_method="POST",
            ),
            litellm_params={"metadata": {"passthrough_route_family": "codex_responses"}},
        )

        assert result["result"]._hidden_params["responses_output"] == codex_response["output"]

    @patch("litellm.completion_cost")
    def test_openai_passthrough_handler_preserves_reasoning_only_responses_output(
        self, mock_completion_cost
    ):
        mock_completion_cost.return_value = 0.111

        reasoning_only_response = {
            "id": "resp-reasoning-only",
            "object": "response",
            "created_at": 1775863780,
            "model": "gpt-oss-120b",
            "output": [
                {
                    "type": "reasoning",
                    "id": "rs_123",
                    "summary": [
                        {
                            "type": "summary_text",
                            "text": "Inspecting the shell state before answering.",
                        }
                    ],
                }
            ],
            "usage": {
                "input_tokens": 50,
                "output_tokens": 10,
                "total_tokens": 60,
                "input_tokens_details": {"cached_tokens": 0},
                "output_tokens_details": {"reasoning_tokens": 10, "text_tokens": 0},
            },
        }
        mock_httpx_response = self._create_mock_httpx_response(reasoning_only_response)
        mock_logging_obj = self._create_mock_logging_obj()

        result = OpenAIPassthroughLoggingHandler.openai_passthrough_handler(
            httpx_response=mock_httpx_response,
            response_body=reasoning_only_response,
            logging_obj=mock_logging_obj,
            url_route="https://openrouter.ai/api/v1/responses",
            result="",
            start_time=self.start_time,
            end_time=self.end_time,
            cache_hit=False,
            request_body={"model": "gpt-oss-120b", "input": "probe reasoning"},
            passthrough_logging_payload=PassthroughStandardLoggingPayload(
                url="https://openrouter.ai/api/v1/responses",
                request_body={"model": "gpt-oss-120b", "input": "probe reasoning"},
                request_method="POST",
            ),
            custom_llm_provider="openrouter",
            litellm_params={},
        )

        usage = result["result"].usage
        message = result["result"].choices[0].message
        assert message.content == ""
        assert (
            message.reasoning_content
            == "Inspecting the shell state before answering."
        )
        assert usage.prompt_tokens == 50
        assert usage.completion_tokens == 10
        assert usage.total_tokens == 60
        assert result["result"]._hidden_params["responses_output"] == reasoning_only_response["output"]
        assert result["kwargs"]["response_cost"] == 0.111

    @patch("litellm.completion_cost")
    def test_openai_streaming_handler_rebuilds_responses_api_usage(
        self, mock_completion_cost
    ):
        mock_completion_cost.return_value = 0.123

        mock_logging_obj = self._create_mock_logging_obj()
        mock_logging_obj.model_call_details = {
            "custom_llm_provider": "openai",
            "litellm_params": {},
        }

        streaming_chunks = [
            'data: {"type":"response.output_text.delta","item_id":"msg_123","output_index":0,"content_index":0,"delta":"priced codex"}',
            'data: {"type":"response.completed","response":{"id":"resp-codex-stream","object":"response","created_at":1775863780,"status":"completed","model":"gpt-5.4","output":[{"type":"message","id":"msg_123","status":"completed","role":"assistant","content":[{"type":"output_text","text":"priced codex","annotations":[]}]}],"usage":{"input_tokens":15518,"output_tokens":25,"total_tokens":15543,"input_tokens_details":{"cached_tokens":6528},"output_tokens_details":{"reasoning_tokens":0,"text_tokens":25}}}}',
            "data: [DONE]",
        ]

        result = OpenAIPassthroughLoggingHandler._handle_logging_openai_collected_chunks(
            litellm_logging_obj=mock_logging_obj,
            passthrough_success_handler_obj=MagicMock(spec=PassThroughEndpointLogging),
            url_route="https://chatgpt.com/backend-api/codex/responses",
            request_body={"model": "gpt-5.4", "input": "priced codex"},
            endpoint_type="openai",
            start_time=self.start_time,
            all_chunks=streaming_chunks,
            end_time=self.end_time,
        )

        usage = result["result"].usage
        standard_logging_object = result["kwargs"]["standard_logging_object"]
        assert usage.prompt_tokens == 15518
        assert usage.completion_tokens == 25
        assert usage.total_tokens == 15543
        assert usage.prompt_tokens_details.cached_tokens == 6528
        assert usage.completion_tokens_details.text_tokens == 25
        assert (
            standard_logging_object["metadata"]["usage_object"][
                "prompt_tokens_details"
            ]["cached_tokens"]
            == 6528
        )
        assert (
            standard_logging_object["response"]["usage"]["prompt_tokens_details"][
                "cached_tokens"
            ]
            == 6528
        )
        assert standard_logging_object["model"] == "gpt-5.4"
        assert standard_logging_object["prompt_tokens"] == 15518
        assert standard_logging_object["completion_tokens"] == 25
        assert standard_logging_object["total_tokens"] == 15543
        mock_completion_cost.assert_called_once()
        assert mock_completion_cost.call_args.kwargs["call_type"] == "responses"
        assert result["kwargs"]["response_cost"] == 0.123

    @patch("litellm.completion_cost")
    def test_openai_streaming_handler_rebuilds_reasoning_only_responses_output(
        self, mock_completion_cost
    ):
        mock_completion_cost.return_value = 0.222

        mock_logging_obj = self._create_mock_logging_obj()
        mock_logging_obj.model_call_details = {
            "custom_llm_provider": "openrouter",
            "litellm_params": {},
        }

        streaming_chunks = [
            'event: response.created',
            'data: {"type":"response.created","response":{"id":"resp-reasoning-only","object":"response","created_at":1775869900,"status":"in_progress","model":"gpt-oss-120b","output":[]}}',
            'event: response.reasoning_summary_text.delta',
            'data: {"type":"response.reasoning_summary_text.delta","item_id":"rs_123","output_index":0,"summary_index":0,"delta":"Inspecting the shell state before answering."}',
            'event: response.completed',
            'data: {"type":"response.completed","response":{"id":"resp-reasoning-only","object":"response","created_at":1775869900,"status":"completed","model":"gpt-oss-120b","output":[{"type":"reasoning","id":"rs_123","summary":[{"type":"summary_text","text":"Inspecting the shell state before answering."}]}],"usage":{"input_tokens":50,"output_tokens":10,"total_tokens":60,"input_tokens_details":{"cached_tokens":0},"output_tokens_details":{"reasoning_tokens":10,"text_tokens":0}}}}',
            "data: [DONE]",
        ]

        result = OpenAIPassthroughLoggingHandler._handle_logging_openai_collected_chunks(
            litellm_logging_obj=mock_logging_obj,
            passthrough_success_handler_obj=MagicMock(spec=PassThroughEndpointLogging),
            url_route="https://openrouter.ai/api/v1/responses",
            request_body={"model": "gpt-oss-120b", "input": "probe reasoning"},
            endpoint_type="openai",
            start_time=self.start_time,
            all_chunks=streaming_chunks,
            end_time=self.end_time,
        )

        usage = result["result"].usage
        message = result["result"].choices[0].message
        assert message.content == ""
        assert (
            message.reasoning_content
            == "Inspecting the shell state before answering."
        )
        assert usage.prompt_tokens == 50
        assert usage.completion_tokens == 10
        assert usage.total_tokens == 60
        assert result["result"]._hidden_params["responses_output"] == [
            {
                "type": "reasoning",
                "id": "rs_123",
                "summary": [
                    {
                        "type": "summary_text",
                        "text": "Inspecting the shell state before answering.",
                    }
                ],
            }
        ]
        mock_completion_cost.assert_called_once()
        assert mock_completion_cost.call_args.kwargs["call_type"] == "responses"
        assert result["kwargs"]["response_cost"] == 0.222

    @patch("litellm.completion_cost")
    def test_openai_streaming_handler_rebuilds_reasoning_only_content_output(
        self, mock_completion_cost
    ):
        mock_completion_cost.return_value = 0.333

        mock_logging_obj = self._create_mock_logging_obj()
        mock_logging_obj.model_call_details = {
            "custom_llm_provider": "openrouter",
            "litellm_params": {},
        }

        streaming_chunks = [
            'event: response.created',
            'data: {"type":"response.created","response":{"id":"resp-reasoning-content-only","object":"response","created_at":1775869950,"status":"in_progress","model":"gpt-oss-120b","output":[]}}',
            'event: response.completed',
            'data: {"type":"response.completed","response":{"id":"resp-reasoning-content-only","object":"response","created_at":1775869950,"status":"completed","model":"gpt-oss-120b","output":[{"type":"reasoning","id":"rs_456","summary":[],"content":[{"type":"reasoning_text","text":"We need to respond with exactly two words: oss120 smoke."}],"status":"completed"}],"usage":{"input_tokens":40,"output_tokens":8,"total_tokens":48,"input_tokens_details":{"cached_tokens":0},"output_tokens_details":{"reasoning_tokens":8,"text_tokens":0}}}}',
            "data: [DONE]",
        ]

        result = OpenAIPassthroughLoggingHandler._handle_logging_openai_collected_chunks(
            litellm_logging_obj=mock_logging_obj,
            passthrough_success_handler_obj=MagicMock(spec=PassThroughEndpointLogging),
            url_route="https://openrouter.ai/api/v1/responses",
            request_body={"model": "gpt-oss-120b", "input": "probe reasoning content"},
            endpoint_type="openai",
            start_time=self.start_time,
            all_chunks=streaming_chunks,
            end_time=self.end_time,
        )

        usage = result["result"].usage
        message = result["result"].choices[0].message
        assert message.content == ""
        assert (
            message.reasoning_content
            == "We need to respond with exactly two words: oss120 smoke."
        )
        assert usage.prompt_tokens == 40
        assert usage.completion_tokens == 8
        assert usage.total_tokens == 48
        assert result["result"]._hidden_params["responses_output"] == [
            {
                "type": "reasoning",
                "id": "rs_456",
                "summary": [],
                "content": [
                    {
                        "type": "reasoning_text",
                        "text": "We need to respond with exactly two words: oss120 smoke.",
                    }
                ],
                "status": "completed",
            }
        ]
        mock_completion_cost.assert_called_once()
        assert mock_completion_cost.call_args.kwargs["call_type"] == "responses"
        assert result["kwargs"]["response_cost"] == 0.333

    @patch("litellm.completion_cost")
    def test_openai_streaming_handler_estimates_usage_when_completed_is_missing(
        self, mock_completion_cost
    ):
        mock_completion_cost.return_value = 0.444

        mock_logging_obj = self._create_mock_logging_obj()
        mock_logging_obj.model_call_details = {
            "custom_llm_provider": "openrouter",
            "litellm_params": {},
        }

        streaming_chunks = [
            'event: response.created',
            'data: {"type":"response.created","response":{"id":"resp-missing-completed","object":"response","created_at":1775869960,"status":"in_progress","model":"openai/gpt-oss-120b:free","output":[]}}',
            'event: response.output_text.done',
            'data: {"type":"response.output_text.done","item_id":"msg_123","output_index":0,"content_index":0,"text":"oss120 smoke"}',
            "data: [DONE]",
        ]

        result = OpenAIPassthroughLoggingHandler._handle_logging_openai_collected_chunks(
            litellm_logging_obj=mock_logging_obj,
            passthrough_success_handler_obj=MagicMock(spec=PassThroughEndpointLogging),
            url_route="https://openrouter.ai/api/v1/responses",
            request_body={"model": "openai/gpt-oss-120b:free", "input": "probe"},
            endpoint_type="openai",
            start_time=self.start_time,
            all_chunks=streaming_chunks,
            end_time=self.end_time,
        )

        usage = result["result"].usage
        message = result["result"].choices[0].message
        assert message.content == "oss120 smoke"
        assert usage.prompt_tokens >= 1
        assert usage.completion_tokens >= 1
        assert usage.total_tokens == usage.prompt_tokens + usage.completion_tokens
        assert (
            result["result"]._hidden_params[
                "openai_responses_stream_missing_completed"
            ]
            is True
        )
        assert (
            result["result"]._hidden_params[
                "openai_responses_stream_usage_estimated"
            ]
            is True
        )
        mock_completion_cost.assert_called_once()
        assert mock_completion_cost.call_args.kwargs["call_type"] == "responses"
        assert result["kwargs"]["response_cost"] == 0.444


    @patch("litellm.completion_cost")
    def test_openai_streaming_handler_rebuilds_failed_terminal_without_missing_completed_warning(
        self, mock_completion_cost, caplog
    ):
        mock_completion_cost.return_value = 0.0

        mock_logging_obj = self._create_mock_logging_obj()
        mock_logging_obj.model_call_details = {
            "custom_llm_provider": "openai",
            "litellm_params": {},
        }

        streaming_chunks = [
            'event: response.created',
            'data: {"type":"response.created","response":{"id":"resp-failed","status":"in_progress","model":"gpt-5.4","output":[]}}',
            'event: response.output_text.delta',
            'data: {"type":"response.output_text.delta","item_id":"msg_123","output_index":0,"content_index":0,"delta":"partial"}',
            'event: response.failed',
            'data: {"type":"response.failed","response":{"id":"resp-failed","object":"response","status":"failed","model":"gpt-5.4","output":[],"error":{"type":"proxy_stream_terminal_error","message":"Streaming response interrupted after first byte due to upstream read timeout"}}}',
            "data: [DONE]",
        ]

        with caplog.at_level("WARNING"):
            result = OpenAIPassthroughLoggingHandler._handle_logging_openai_collected_chunks(
                litellm_logging_obj=mock_logging_obj,
                passthrough_success_handler_obj=MagicMock(spec=PassThroughEndpointLogging),
                url_route="https://chatgpt.com/backend-api/codex/responses",
                request_body={"model": "gpt-5.4", "input": "probe"},
                endpoint_type="openai",
                start_time=self.start_time,
                all_chunks=streaming_chunks,
                end_time=self.end_time,
            )

        assert result["result"] is not None
        assert result["result"].choices[0].message.content == ""
        assert result["result"]._hidden_params["responses_terminal_event_type"] == "response.failed"
        assert result["result"]._hidden_params["responses_terminal_status"] == "failed"
        assert "timeout" in result["result"]._hidden_params["responses_terminal_error"]
        assert result["kwargs"]["standard_logging_object"]["status"] == "failure"
        assert "timeout" in result["kwargs"]["standard_logging_object"]["error_str"]
        assert "No response.completed event found" not in caplog.text
        assert "No recognized Responses terminal event found" not in caplog.text

    @patch("litellm.completion_cost")
    def test_openai_streaming_handler_rebuilds_incomplete_terminal_as_failure(
        self, mock_completion_cost
    ):
        mock_completion_cost.return_value = 0.0

        mock_logging_obj = self._create_mock_logging_obj()
        mock_logging_obj.model_call_details = {
            "custom_llm_provider": "openai",
            "litellm_params": {},
        }

        streaming_chunks = [
            'data: {"type":"response.incomplete","response":{"id":"resp-incomplete","object":"response","status":"incomplete","model":"gpt-5.4","output":[],"incomplete_details":{"reason":"max_output_tokens"}}}',
            "data: [DONE]",
        ]

        result = OpenAIPassthroughLoggingHandler._handle_logging_openai_collected_chunks(
            litellm_logging_obj=mock_logging_obj,
            passthrough_success_handler_obj=MagicMock(spec=PassThroughEndpointLogging),
            url_route="https://chatgpt.com/backend-api/codex/responses",
            request_body={"model": "gpt-5.4", "input": "probe"},
            endpoint_type="openai",
            start_time=self.start_time,
            all_chunks=streaming_chunks,
            end_time=self.end_time,
        )

        assert result["result"] is not None
        assert result["result"]._hidden_params["responses_terminal_event_type"] == "response.incomplete"
        assert result["result"]._hidden_params["responses_terminal_status"] == "incomplete"
        assert result["kwargs"]["standard_logging_object"]["status"] == "failure"

    @patch("litellm.completion_cost")
    def test_openai_streaming_handler_rebuilds_completed_terminal_failed_status_as_failure(
        self, mock_completion_cost
    ):
        mock_completion_cost.return_value = 0.0

        mock_logging_obj = self._create_mock_logging_obj()
        mock_logging_obj.model_call_details = {
            "custom_llm_provider": "openai",
            "litellm_params": {},
        }

        streaming_chunks = [
            'event: response.created',
            'data: {"type":"response.created","response":{"id":"resp-completed-failed","object":"response","status":"in_progress","model":"gpt-5.4","output":[]}}',
            'event: response.completed',
            'data: {"type":"response.completed","response":{"id":"resp-completed-failed","object":"response","status":"failed","model":"gpt-5.4","output":[],"error":{"type":"proxy_stream_terminal_error","message":"Streaming response failed with credentials Bearer sk-1234567890abcdefghijklmnopqrstuv"}}}',
            "data: [DONE]",
        ]

        result = OpenAIPassthroughLoggingHandler._handle_logging_openai_collected_chunks(
            litellm_logging_obj=mock_logging_obj,
            passthrough_success_handler_obj=MagicMock(spec=PassThroughEndpointLogging),
            url_route="https://chatgpt.com/backend-api/codex/responses",
            request_body={"model": "gpt-5.4", "input": "probe"},
            endpoint_type="openai",
            start_time=self.start_time,
            all_chunks=streaming_chunks,
            end_time=self.end_time,
        )

        assert result["result"] is not None
        assert (
            result["result"]._hidden_params["responses_terminal_event_type"]
            == "response.completed"
        )
        assert result["result"]._hidden_params["responses_terminal_status"] == "failed"
        hidden_error = result["result"]._hidden_params["responses_terminal_error"]
        assert hidden_error is not None
        assert "Bearer sk-" not in hidden_error
        assert "REDACTED" in hidden_error
        assert result["kwargs"]["standard_logging_object"]["status"] == "failure"
        assert "REDACTED" in result["kwargs"]["standard_logging_object"]["error_str"]
        assert "sk-" not in result["kwargs"]["standard_logging_object"]["error_str"]

    @patch("litellm.completion_cost")
    def test_openai_streaming_handler_rebuilds_completed_terminal_incomplete_status_as_failure(
        self, mock_completion_cost
    ):
        mock_completion_cost.return_value = 0.0

        mock_logging_obj = self._create_mock_logging_obj()
        mock_logging_obj.model_call_details = {
            "custom_llm_provider": "openai",
            "litellm_params": {},
        }

        streaming_chunks = [
            'event: response.created',
            'data: {"type":"response.created","response":{"id":"resp-completed-incomplete","object":"response","status":"in_progress","model":"gpt-5.4","output":[]}}',
            'event: response.completed',
            'data: {"type":"response.completed","response":{"id":"resp-completed-incomplete","object":"response","status":"incomplete","model":"gpt-5.4","output":[],"incomplete_details":{"reason":"max_output_tokens","attempt_count":3}}}',
            "data: [DONE]",
        ]

        result = OpenAIPassthroughLoggingHandler._handle_logging_openai_collected_chunks(
            litellm_logging_obj=mock_logging_obj,
            passthrough_success_handler_obj=MagicMock(spec=PassThroughEndpointLogging),
            url_route="https://chatgpt.com/backend-api/codex/responses",
            request_body={"model": "gpt-5.4", "input": "probe"},
            endpoint_type="openai",
            start_time=self.start_time,
            all_chunks=streaming_chunks,
            end_time=self.end_time,
        )

        assert result["result"] is not None
        assert (
            result["result"]._hidden_params["responses_terminal_event_type"]
            == "response.completed"
        )
        assert result["result"]._hidden_params["responses_terminal_status"] == "incomplete"
        assert (
            result["result"]._hidden_params["responses_terminal_incomplete_details"]
            == "max_output_tokens"
        )
        assert "attempt_count" not in str(
            result["result"]._hidden_params["responses_terminal_incomplete_details"]
        )
        assert result["kwargs"]["standard_logging_object"]["status"] == "failure"

    @patch("litellm.completion_cost")
    def test_openai_streaming_handler_returns_none_for_no_terminal_stream(
        self, mock_completion_cost, caplog
    ):
        mock_completion_cost.return_value = 0.0

        mock_logging_obj = self._create_mock_logging_obj()
        mock_logging_obj.model_call_details = {
            "custom_llm_provider": "openai",
            "litellm_params": {},
        }

        streaming_chunks = [
            'event: response.created',
            'data: {"type":"response.created","response":{"id":"resp-truncated","status":"in_progress","model":"gpt-5.4","output":[]}}',
            'event: response.output_text.done',
            'data: {"type":"response.output_text.done","item_id":"msg_123","output_index":0,"content_index":0,"text":"partial only"}',
        ]

        with caplog.at_level("WARNING"):
            result = OpenAIPassthroughLoggingHandler._handle_logging_openai_collected_chunks(
                litellm_logging_obj=mock_logging_obj,
                passthrough_success_handler_obj=MagicMock(spec=PassThroughEndpointLogging),
                url_route="https://chatgpt.com/backend-api/codex/responses",
                request_body={"model": "gpt-5.4", "input": "probe"},
                endpoint_type="openai",
                start_time=self.start_time,
                all_chunks=streaming_chunks,
                end_time=self.end_time,
            )

        assert result["result"] is None
        assert "No recognized Responses terminal event found" in caplog.text

    @patch("litellm.completion_cost")
    def test_openai_streaming_handler_marks_output_text_done_fallback_as_synthesized(
        self, mock_completion_cost
    ):
        mock_completion_cost.return_value = 0.444

        mock_logging_obj = self._create_mock_logging_obj()
        mock_logging_obj.model_call_details = {
            "custom_llm_provider": "openrouter",
            "litellm_params": {},
        }

        streaming_chunks = [
            'event: response.created',
            'data: {"type":"response.created","response":{"id":"resp-missing-completed","object":"response","created_at":1775869960,"status":"in_progress","model":"openai/gpt-oss-120b:free","output":[]}}',
            'event: response.output_text.done',
            'data: {"type":"response.output_text.done","item_id":"msg_123","output_index":0,"content_index":0,"text":"oss120 smoke"}',
            "data: [DONE]",
        ]

        result = OpenAIPassthroughLoggingHandler._handle_logging_openai_collected_chunks(
            litellm_logging_obj=mock_logging_obj,
            passthrough_success_handler_obj=MagicMock(spec=PassThroughEndpointLogging),
            url_route="https://openrouter.ai/api/v1/responses",
            request_body={"model": "openai/gpt-oss-120b:free", "input": "probe"},
            endpoint_type="openai",
            start_time=self.start_time,
            all_chunks=streaming_chunks,
            end_time=self.end_time,
        )

        hidden = result["result"]._hidden_params
        assert hidden["openai_responses_stream_missing_completed"] is True
        assert hidden["openai_responses_stream_usage_estimated"] is True
        assert hidden["openai_responses_stream_synthesized_terminal"] is True
        assert hidden["openai_responses_stream_missing_formal_terminal_event"] is True
        assert result["kwargs"]["standard_logging_object"]["status"] == "success"

    def test_openai_passthrough_handler_backfills_openrouter_responses_usage_and_model(self):
        mock_httpx_response = self._create_mock_httpx_response(
            {
                "id": "resp-openrouter-usage-test",
                "object": "response",
                "created_at": 1775863780,
                "model": "openrouter/free",
                "output": [
                    {
                        "type": "message",
                        "id": "msg_123",
                        "status": "completed",
                        "role": "assistant",
                        "content": [
                            {"type": "output_text", "text": "free smoke", "annotations": []}
                        ],
                    }
                ],
                "usage": {
                    "input_tokens": 95829,
                    "output_tokens": 198,
                    "total_tokens": 96027,
                    "input_tokens_details": {"cached_tokens": 0},
                    "output_tokens_details": {"reasoning_tokens": 0, "text_tokens": 198},
                },
            }
        )
        mock_logging_obj = self._create_mock_logging_obj()
        transformed_response = litellm.ModelResponse(
            model="unknown",
            choices=[
                litellm.Choices(
                    message=litellm.Message(role="assistant", content="free smoke"),
                    finish_reason="stop",
                    index=0,
                )
            ],
            usage=litellm.Usage(prompt_tokens=0, completion_tokens=0, total_tokens=0),
        )
        mock_provider_config = MagicMock()
        mock_provider_config.transform_response.return_value = transformed_response

        with patch.object(
            OpenAIPassthroughLoggingHandler,
            "get_provider_config",
            return_value=mock_provider_config,
        ), patch("litellm.completion_cost", return_value=0.484095):
            result = OpenAIPassthroughLoggingHandler.openai_passthrough_handler(
                httpx_response=mock_httpx_response,
                response_body=mock_httpx_response.json(),
                logging_obj=mock_logging_obj,
                url_route="https://openrouter.ai/api/v1/responses",
                result="",
                start_time=self.start_time,
                end_time=self.end_time,
                cache_hit=False,
                request_body={"model": "openrouter/free", "input": "free smoke"},
                passthrough_logging_payload=PassthroughStandardLoggingPayload(
                    url="https://openrouter.ai/api/v1/responses",
                    request_body={"model": "openrouter/free", "input": "free smoke"},
                    request_method="POST",
                ),
                custom_llm_provider="openrouter",
                litellm_params={},
            )

        usage = result["result"].usage
        standard_logging_object = result["kwargs"]["standard_logging_object"]
        assert result["result"].model == "openrouter/free"
        assert usage.prompt_tokens == 95829
        assert usage.completion_tokens == 198
        assert usage.total_tokens == 96027
        assert standard_logging_object["model"] == "openrouter/free"
        assert standard_logging_object["prompt_tokens"] == 95829
        assert standard_logging_object["completion_tokens"] == 198
        assert standard_logging_object["total_tokens"] == 96027

    def test_openai_passthrough_handler_preserves_unmapped_openrouter_responses_model(self):
        response_body = {
            "id": "resp-openrouter-wildcard-test",
            "object": "response",
            "created_at": 1775863780,
            "model": "openrouter/owl-alpha",
            "output": [
                {
                    "type": "message",
                    "id": "msg_123",
                    "status": "completed",
                    "role": "assistant",
                    "content": [
                        {"type": "output_text", "text": "OK", "annotations": []}
                    ],
                }
            ],
            "usage": {
                "input_tokens": 98,
                "output_tokens": 2,
                "total_tokens": 100,
                "input_tokens_details": {"cached_tokens": 0},
                "output_tokens_details": {"reasoning_tokens": 0},
                "cost": 0,
            },
        }
        mock_httpx_response = self._create_mock_httpx_response(response_body)
        mock_logging_obj = self._create_mock_logging_obj()

        with patch("litellm.completion_cost", side_effect=Exception("unmapped model")):
            result = OpenAIPassthroughLoggingHandler.openai_passthrough_handler(
                httpx_response=mock_httpx_response,
                response_body=response_body,
                logging_obj=mock_logging_obj,
                url_route="https://openrouter.ai/api/v1/responses",
                result="",
                start_time=self.start_time,
                end_time=self.end_time,
                cache_hit=False,
                request_body={"model": "openrouter/owl-alpha", "input": "Reply OK"},
                passthrough_logging_payload=PassthroughStandardLoggingPayload(
                    url="https://openrouter.ai/api/v1/responses",
                    request_body={"model": "openrouter/owl-alpha", "input": "Reply OK"},
                    request_method="POST",
                ),
                custom_llm_provider="openrouter",
                litellm_params={
                    "metadata": {
                        "passthrough_route_family": (
                            "anthropic_openrouter_responses_adapter"
                        ),
                        "anthropic_adapter_original_model": "openrouter/owl-alpha",
                        "anthropic_adapter_model": "owl-alpha",
                    }
                },
            )

        usage = result["result"].usage
        standard_logging_object = result["kwargs"]["standard_logging_object"]
        assert result["result"].model == "openrouter/owl-alpha"
        assert usage.prompt_tokens == 98
        assert usage.completion_tokens == 2
        assert usage.total_tokens == 100
        assert result["kwargs"]["response_cost"] == 0.0
        assert standard_logging_object["model"] == "openrouter/owl-alpha"
        assert standard_logging_object["prompt_tokens"] == 98
        assert standard_logging_object["completion_tokens"] == 2
        metadata = result["kwargs"]["litellm_params"]["metadata"]
        assert metadata["anthropic_adapter_original_model"] == "openrouter/owl-alpha"

    @patch("litellm.completion_cost")
    def test_openai_streaming_handler_rebuilds_codex_stream_with_empty_output(
        self, mock_completion_cost
    ):
        mock_completion_cost.return_value = 0.456

        mock_logging_obj = self._create_mock_logging_obj()
        litellm_params = {"metadata": {"passthrough_route_family": "codex_responses"}}
        mock_logging_obj.model_call_details = {
            "custom_llm_provider": "openai",
            "litellm_params": litellm_params,
        }

        streaming_chunks = [
            'event: response.created',
            'data: {"type":"response.created","response":{"id":"resp-codex-stream","object":"response","created_at":1775868809,"status":"in_progress","model":"gpt-5.4","output":[]}}',
            'event: response.output_item.added',
            'data: {"type":"response.output_item.added","output_index":0,"item":{"type":"function_call","call_id":"call_pwd","id":"fc_pwd","name":"Bash","arguments":""}}',
            'event: response.function_call_arguments.done',
            'data: {"type":"response.function_call_arguments.done","item_id":"fc_pwd","output_index":0,"arguments":"{\\"command\\":\\"pwd\\"}"}',
            'event: response.output_text.delta',
            'data: {"type":"response.output_text.delta","item_id":"msg_123","output_index":1,"content_index":0,"delta":"langfuse "}',
            'event: response.output_text.delta',
            'data: {"type":"response.output_text.delta","item_id":"msg_123","output_index":1,"content_index":0,"delta":"fixed"}',
            'event: response.completed',
            'data: {"type":"response.completed","response":{"id":"resp-codex-stream","object":"response","created_at":1775868809,"status":"completed","model":"gpt-5.4","output":[],"usage":{"input_tokens":15519,"output_tokens":29,"total_tokens":15548,"input_tokens_details":{"cached_tokens":6528},"output_tokens_details":{"reasoning_tokens":19}}}}',
        ]

        result = OpenAIPassthroughLoggingHandler._handle_logging_openai_collected_chunks(
            litellm_logging_obj=mock_logging_obj,
            passthrough_success_handler_obj=MagicMock(spec=PassThroughEndpointLogging),
            url_route="https://chatgpt.com/backend-api/codex/responses",
            request_body={"model": "gpt-5.4", "input": "langfuse fixed"},
            endpoint_type="openai",
            start_time=self.start_time,
            all_chunks=streaming_chunks,
            end_time=self.end_time,
        )

        usage = result["result"].usage
        standard_logging_object = result["kwargs"]["standard_logging_object"]
        metadata = result["kwargs"]["litellm_params"]["metadata"]
        assert result["result"].choices[0].message.content == "langfuse fixed"
        assert usage.prompt_tokens == 15519
        assert usage.completion_tokens == 29
        assert usage.total_tokens == 15548
        assert usage.prompt_tokens_details.cached_tokens == 6528
        assert standard_logging_object["model"] == "gpt-5.4"
        assert standard_logging_object["prompt_tokens"] == 15519
        assert standard_logging_object["completion_tokens"] == 29
        assert standard_logging_object["total_tokens"] == 15548
        assert result["result"]._hidden_params["responses_output"] == [
            {
                "type": "function_call",
                "call_id": "call_pwd",
                "id": "fc_pwd",
                "name": "Bash",
                "arguments": '{"command":"pwd"}',
            }
        ]
        assert "response.output_item.added" in metadata["responses_stream_event_types"]
        assert "response.function_call_arguments.done" in metadata["responses_stream_event_types"]
        assert metadata["responses_stream_event_counts"]["response.output_item.added"] == 1
        assert metadata["responses_stream_tool_call_count"] == 1
        assert metadata["responses_stream_tool_names"] == ["Bash"]
        assert metadata["responses_stream_tool_state"] == [
            {
                "type": "function_call",
                "name": "Bash",
                "call_id": "call_pwd",
                "arguments_hash": "d66a53fedbf412beeadb3868ece33ec9e7e20e9aa0224b75a70c5655b8ca2e2c",
                "arguments_size_bytes": 17,
            }
        ]
        metadata_json = json.dumps(metadata)
        assert '{"command":"pwd"}' not in metadata_json
        assert "super-secret-token" not in metadata_json
        assert "API_KEY=leaked" not in metadata_json
        assert '"arguments":' not in metadata_json
        mock_completion_cost.assert_called_once()
        assert mock_completion_cost.call_args.kwargs["call_type"] == "responses"
        assert result["kwargs"]["response_cost"] == 0.456

    @patch("litellm.completion_cost")
    def test_openai_streaming_handler_rebuilds_codex_stream_with_missing_output(
        self, mock_completion_cost
    ):
        mock_completion_cost.return_value = 0.457

        mock_logging_obj = self._create_mock_logging_obj()
        litellm_params = {"metadata": {"passthrough_route_family": "codex_responses"}}
        mock_logging_obj.model_call_details = {
            "custom_llm_provider": "openai",
            "litellm_params": litellm_params,
        }

        streaming_chunks = [
            'event: response.created',
            'data: {"type":"response.created","response":{"id":"resp-codex-stream","object":"response","created_at":1775868809,"status":"in_progress","model":"gpt-5.4","output":[]}}',
            'event: response.output_item.added',
            'data: {"type":"response.output_item.added","output_index":0,"item":{"type":"function_call","call_id":"call_pwd","id":"fc_pwd","name":"Bash","arguments":""}}',
            'event: response.function_call_arguments.done',
            'data: {"type":"response.function_call_arguments.done","item_id":"fc_pwd","output_index":0,"arguments":"{\\"command\\":\\"pwd\\"}"}',
            'event: response.output_text.delta',
            'data: {"type":"response.output_text.delta","item_id":"msg_123","output_index":1,"content_index":0,"delta":"langfuse "}',
            'event: response.output_text.delta',
            'data: {"type":"response.output_text.delta","item_id":"msg_123","output_index":1,"content_index":0,"delta":"fixed"}',
            'event: response.completed',
            'data: {"type":"response.completed","response":{"id":"resp-codex-stream","object":"response","created_at":1775868809,"status":"completed","model":"gpt-5.4","usage":{"input_tokens":15519,"output_tokens":29,"total_tokens":15548,"input_tokens_details":{"cached_tokens":6528},"output_tokens_details":{"reasoning_tokens":19}}}}',
        ]

        result = OpenAIPassthroughLoggingHandler._handle_logging_openai_collected_chunks(
            litellm_logging_obj=mock_logging_obj,
            passthrough_success_handler_obj=MagicMock(spec=PassThroughEndpointLogging),
            url_route="https://chatgpt.com/backend-api/codex/responses",
            request_body={"model": "gpt-5.4", "input": "langfuse fixed"},
            endpoint_type="openai",
            start_time=self.start_time,
            all_chunks=streaming_chunks,
            end_time=self.end_time,
            kwargs={"litellm_params": litellm_params},
        )

        usage = result["result"].usage
        standard_logging_object = result["kwargs"]["standard_logging_object"]
        metadata = result["kwargs"]["litellm_params"]["metadata"]
        assert result["result"].choices[0].message.content == "langfuse fixed"
        assert usage.prompt_tokens == 15519
        assert usage.completion_tokens == 29
        assert usage.total_tokens == 15548
        assert usage.prompt_tokens_details.cached_tokens == 6528
        assert standard_logging_object["model"] == "gpt-5.4"
        assert standard_logging_object["prompt_tokens"] == 15519
        assert standard_logging_object["completion_tokens"] == 29
        assert standard_logging_object["total_tokens"] == 15548
        assert "cannot parse chunks" not in json.dumps(
            standard_logging_object, default=str
        )
        assert result["result"]._hidden_params["responses_output"] == [
            {
                "type": "function_call",
                "call_id": "call_pwd",
                "id": "fc_pwd",
                "name": "Bash",
                "arguments": '{"command":"pwd"}',
            }
        ]
        spans = metadata["langfuse_spans"]
        assert spans[0]["name"] == "codex.usage_normalize"
        assert spans[0]["metadata"]["streaming"] is True
        assert spans[0]["metadata"]["call_type"] == "responses"
        assert spans[0]["metadata"]["total_tokens"] == 15548
        mock_completion_cost.assert_called_once()
        assert mock_completion_cost.call_args.kwargs["call_type"] == "responses"
        assert result["kwargs"]["response_cost"] == 0.457

    @patch("litellm.completion_cost")
    def test_openai_streaming_handler_records_compact_tool_state_without_raw_arguments(
        self, mock_completion_cost
    ):
        mock_completion_cost.return_value = 0.458

        mock_logging_obj = self._create_mock_logging_obj()
        litellm_params = {"metadata": {"passthrough_route_family": "codex_responses"}}
        mock_logging_obj.model_call_details = {
            "custom_llm_provider": "openai",
            "litellm_params": litellm_params,
        }

        streaming_chunks = [
            'event: response.created',
            'data: {"type":"response.created","response":{"id":"resp-codex-tools","object":"response","created_at":1775868809,"status":"in_progress","model":"gpt-5.4","output":[]}}',
            'event: response.output_item.added',
            'data: {"type":"response.output_item.added","output_index":0,"item":{"type":"function_call","call_id":"call_secret","id":"fc_secret","name":"Bash","arguments":""}}',
            'event: response.function_call_arguments.done',
            'data: {"type":"response.function_call_arguments.done","item_id":"fc_secret","output_index":0,"arguments":"{\\"command\\":\\"export SECRET=super-secret-token\\"}"}',
            'event: response.output_item.added',
            'data: {"type":"response.output_item.added","output_index":1,"item":{"type":"local_shell_call","call_id":"shell_1","id":"shell_1","name":"local_shell","input":{"command":"ls -la"}}}',
            'event: response.output_item.added',
            'data: {"type":"response.output_item.added","output_index":2,"item":{"type":"apply_patch_call","call_id":"patch_1","id":"patch_1","name":"apply_patch","patch":"*** Begin Patch\\n*** Update File: secret.py\\n+API_KEY=leaked\\n*** End Patch"}}',
            'event: response.completed',
            'data: {"type":"response.completed","response":{"id":"resp-codex-tools","object":"response","created_at":1775868809,"status":"completed","model":"gpt-5.4","output":[],"usage":{"input_tokens":10,"output_tokens":3,"total_tokens":13}}}',
        ]

        result = OpenAIPassthroughLoggingHandler._handle_logging_openai_collected_chunks(
            litellm_logging_obj=mock_logging_obj,
            passthrough_success_handler_obj=MagicMock(spec=PassThroughEndpointLogging),
            url_route="https://chatgpt.com/backend-api/codex/responses",
            request_body={"model": "gpt-5.4", "input": "tool metadata"},
            endpoint_type="openai",
            start_time=self.start_time,
            all_chunks=streaming_chunks,
            end_time=self.end_time,
            kwargs={"litellm_params": litellm_params},
        )

        metadata = result["kwargs"]["litellm_params"]["metadata"]
        assert metadata["responses_stream_tool_call_count"] == 3
        assert metadata["responses_stream_tool_names"] == [
            "Bash",
            "local_shell",
            "apply_patch",
        ]
        assert metadata["responses_stream_tool_state"] == [
            {
                "type": "function_call",
                "name": "Bash",
                "call_id": "call_secret",
                "arguments_hash": "df9c4812e948b70ee537d3f817c84993b2143f34812af8af0cc95c37d62977b4",
                "arguments_size_bytes": 46,
            },
            {
                "type": "local_shell_call",
                "name": "local_shell",
                "call_id": "shell_1",
                "arguments_hash": "1df8bccaec747dc615b50678f35bf5b51756a45f9b2b77b247c7a617fde58b3e",
                "arguments_size_bytes": 20,
            },
            {
                "type": "apply_patch_call",
                "name": "apply_patch",
                "call_id": "patch_1",
                "arguments_hash": "9dfab7fdeb6aa76caa1b593d61845ec82375dd4f4470a7334bce0250d74771a5",
                "arguments_size_bytes": 72,
            },
        ]
        metadata_json = json.dumps(metadata)
        assert "super-secret-token" not in metadata_json
        assert "API_KEY=leaked" not in metadata_json
        assert '"arguments"' not in metadata_json

    def test_openai_passthrough_handler_adds_codex_usage_normalize_span(self):
        codex_response = {
            "id": "resp-codex-span-test",
            "object": "response",
            "created": 1775863780,
            "model": "gpt-5.4",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "usage probe"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "input_tokens": 100,
                "output_tokens": 20,
                "total_tokens": 120,
                "input_tokens_details": {"cached_tokens": 10},
                "output_tokens_details": {"reasoning_tokens": 0, "text_tokens": 20},
            },
        }
        mock_httpx_response = self._create_mock_httpx_response(codex_response)
        mock_logging_obj = self._create_mock_logging_obj()

        result = OpenAIPassthroughLoggingHandler.openai_passthrough_handler(
            httpx_response=mock_httpx_response,
            response_body=codex_response,
            logging_obj=mock_logging_obj,
            url_route="https://chatgpt.com/backend-api/codex/responses",
            result="",
            start_time=self.start_time,
            end_time=self.end_time,
            cache_hit=False,
            request_body={"model": "gpt-5.4", "input": "usage probe"},
            passthrough_logging_payload=PassthroughStandardLoggingPayload(
                url="https://chatgpt.com/backend-api/codex/responses",
                request_body={"model": "gpt-5.4", "input": "usage probe"},
                request_method="POST",
            ),
            litellm_params={"metadata": {"passthrough_route_family": "codex_responses"}},
        )

        spans = result["kwargs"]["litellm_params"]["metadata"]["langfuse_spans"]
        assert spans[0]["name"] == "codex.usage_normalize"
        assert spans[0]["metadata"]["streaming"] is False
        assert spans[0]["metadata"]["call_type"] == "responses"
        assert spans[0]["metadata"]["total_tokens"] == 120

    @patch('litellm.completion_cost')
    @patch('litellm.litellm_core_utils.litellm_logging.get_standard_logging_object_payload')
    def test_openai_passthrough_handler_with_user_tracking(self, mock_get_standard_logging, mock_completion_cost):
        """Test cost tracking with user information"""
        # Arrange
        mock_completion_cost.return_value = 0.000123
        mock_get_standard_logging.return_value = {"test": "logging_payload"}
        
        mock_httpx_response = self._create_mock_httpx_response()
        mock_logging_obj = self._create_mock_logging_obj()
        
        # Create payload with user information
        passthrough_payload = PassthroughStandardLoggingPayload(
            url="https://api.openai.com/v1/chat/completions",
            request_body={
                "model": "gpt-4o", 
                "messages": [{"role": "user", "content": "Hello"}],
                "user": "test_user_123"
            },
            request_method="POST",
        )
        
        kwargs = {
            "passthrough_logging_payload": passthrough_payload,
            "model": "gpt-4o",
        }

        # Act
        result = OpenAIPassthroughLoggingHandler.openai_passthrough_handler(
            httpx_response=mock_httpx_response,
            response_body=self.mock_openai_response,
            logging_obj=mock_logging_obj,
            url_route="https://api.openai.com/v1/chat/completions",
            result="",
            start_time=self.start_time,
            end_time=self.end_time,
            cache_hit=False,
            request_body={"model": "gpt-4o", "messages": [{"role": "user", "content": "Hello"}], "user": "test_user_123"},
            **kwargs
        )

        # Assert
        assert result is not None
        assert "result" in result
        assert "kwargs" in result
        assert result["kwargs"]["response_cost"] == 0.000123
        
        # Verify user information is included in litellm_params
        assert "litellm_params" in result["kwargs"]
        assert "proxy_server_request" in result["kwargs"]["litellm_params"]
        assert "body" in result["kwargs"]["litellm_params"]["proxy_server_request"]
        assert result["kwargs"]["litellm_params"]["proxy_server_request"]["body"]["user"] == "test_user_123"

    @patch('litellm.completion_cost')
    def test_openai_passthrough_handler_cost_calculation_error(self, mock_completion_cost):
        """Test error handling in cost calculation"""
        # Arrange
        mock_completion_cost.side_effect = Exception("Cost calculation failed")
        
        mock_httpx_response = self._create_mock_httpx_response()
        mock_logging_obj = self._create_mock_logging_obj()
        passthrough_payload = self._create_passthrough_logging_payload()
        
        kwargs = {
            "passthrough_logging_payload": passthrough_payload,
            "model": "gpt-4o",
        }

        # Act
        result = OpenAIPassthroughLoggingHandler.openai_passthrough_handler(
            httpx_response=mock_httpx_response,
            response_body=self.mock_openai_response,
            logging_obj=mock_logging_obj,
            url_route="https://api.openai.com/v1/chat/completions",
            result="",
            start_time=self.start_time,
            end_time=self.end_time,
            cache_hit=False,
            request_body={"model": "gpt-4o", "messages": [{"role": "user", "content": "Hello"}]},
            **kwargs
        )

        # Assert - Should fall back to base handler when cost calculation fails
        assert result is not None
        assert "result" in result
        assert "kwargs" in result

    def test_build_complete_streaming_response(self):
        """Test the streaming response builder (placeholder implementation)"""
        # This is a placeholder method that returns None for now
        result = self.handler._build_complete_streaming_response(
            all_chunks=["chunk1", "chunk2"],
            litellm_logging_obj=self._create_mock_logging_obj(),
            model="gpt-4o",
            url_route="https://api.openai.com/v1/chat/completions",
        )
        
        assert result is None  # Placeholder implementation

    @patch('litellm.completion_cost')
    @patch('litellm.litellm_core_utils.litellm_logging.get_standard_logging_object_payload')
    def test_different_models_cost_tracking(self, mock_get_standard_logging, mock_completion_cost):
        """Test cost tracking for different OpenAI models"""
        # Arrange
        mock_get_standard_logging.return_value = {"test": "logging_payload"}
        
        test_cases = [
            ("gpt-4o", 0.000045),
            ("gpt-4o-mini", 0.000015),
            ("gpt-3.5-turbo", 0.000002),
        ]
        
        for model, expected_cost in test_cases:
            mock_completion_cost.return_value = expected_cost
            
            mock_httpx_response = self._create_mock_httpx_response()
            mock_httpx_response.json.return_value = {
                **self.mock_openai_response,
                "model": model
            }
            
            mock_logging_obj = self._create_mock_logging_obj()
            passthrough_payload = self._create_passthrough_logging_payload()
            
            kwargs = {
                "passthrough_logging_payload": passthrough_payload,
                "model": model,
            }

            # Act
            result = OpenAIPassthroughLoggingHandler.openai_passthrough_handler(
                httpx_response=mock_httpx_response,
                response_body={**self.mock_openai_response, "model": model},
                logging_obj=mock_logging_obj,
                url_route="https://api.openai.com/v1/chat/completions",
                result="",
                start_time=self.start_time,
                end_time=self.end_time,
                cache_hit=False,
                request_body={"model": model, "messages": [{"role": "user", "content": "Hello"}]},
                **kwargs
            )

            # Assert
            assert result is not None
            assert "result" in result
            assert "kwargs" in result
            assert result["kwargs"]["response_cost"] == expected_cost
            assert result["kwargs"]["model"] == model
            assert result["kwargs"]["custom_llm_provider"] == "openai"

    def test_static_methods(self):
        """Test that static methods work correctly"""
        # Test static method calls
        assert (
            OpenAIPassthroughLoggingHandler.is_openai_chat_completions_route(
                "https://api.openai.com/v1/chat/completions"
            )
            is True
        )
        # Test instance method
        handler = OpenAIPassthroughLoggingHandler()
        assert handler.get_provider_config("gpt-4o") is not None

    @patch('litellm.completion_cost')
    @patch('litellm.litellm_core_utils.litellm_logging.get_standard_logging_object_payload')
    def test_azure_passthrough_tags_metadata_model_provider(self, mock_get_standard_logging, mock_completion_cost):
        """Test that tags, metadata, model, and custom_llm_provider are preserved for Azure passthrough in UI"""
        # Arrange
        mock_completion_cost.return_value = 0.000045
        mock_get_standard_logging.return_value = {"test": "logging_payload"}
        
        mock_httpx_response = self._create_mock_httpx_response()
        mock_logging_obj = self._create_mock_logging_obj()
        
        # Create payload with metadata tags
        passthrough_payload = PassthroughStandardLoggingPayload(
            url="https://openai.azure.com/v1/chat/completions",
            request_body={
                "model": "gpt-4o",
                "messages": [{"role": "user", "content": "Hello"}]
            },
            request_method="POST",
        )
        
        # Set up kwargs with existing litellm_params containing metadata tags
        kwargs = {
            "passthrough_logging_payload": passthrough_payload,
            "model": "gpt-4o",
            "custom_llm_provider": "azure",  # Azure passthrough
            "litellm_params": {
                "metadata": {
                    "tags": ["production", "azure-deployment"],
                    "user_id": "user_123"
                },
                "proxy_server_request": {
                    "body": {
                        "user": "test_user"
                    }
                }
            }
        }

        # Act
        result = OpenAIPassthroughLoggingHandler.openai_passthrough_handler(
            httpx_response=mock_httpx_response,
            response_body=self.mock_openai_response,
            logging_obj=mock_logging_obj,
            url_route="https://openai.azure.com/v1/chat/completions",
            result="",
            start_time=self.start_time,
            end_time=self.end_time,
            cache_hit=False,
            request_body={"model": "gpt-4o", "messages": [{"role": "user", "content": "Hello"}]},
            **kwargs
        )

        # Assert - Verify tags, model, and custom_llm_provider are preserved
        assert result is not None
        assert "kwargs" in result
        
        # Verify model and custom_llm_provider are set correctly
        assert result["kwargs"]["model"] == "gpt-4o"
        assert result["kwargs"]["custom_llm_provider"] == "azure"  # Should preserve Azure, not default to "openai"
        assert result["kwargs"]["response_cost"] == 0.000045
        
        # Verify metadata tags are preserved in litellm_params
        assert "litellm_params" in result["kwargs"]
        assert "metadata" in result["kwargs"]["litellm_params"]
        assert "tags" in result["kwargs"]["litellm_params"]["metadata"]
        assert result["kwargs"]["litellm_params"]["metadata"]["tags"] == ["production", "azure-deployment"]
        assert result["kwargs"]["litellm_params"]["metadata"]["user_id"] == "user_123"
        
        # Verify logging object has correct values for UI display
        assert mock_logging_obj.model_call_details["model"] == "gpt-4o"
        assert mock_logging_obj.model_call_details["custom_llm_provider"] == "azure"
        assert mock_logging_obj.model_call_details["response_cost"] == 0.000045
        
        # Verify cost calculation was called with correct custom_llm_provider
        mock_completion_cost.assert_called_once()
        call_args = mock_completion_cost.call_args
        assert call_args[1]["custom_llm_provider"] == "azure"

    @patch('litellm.completion_cost')
    @patch('litellm.litellm_core_utils.litellm_logging.get_standard_logging_object_payload')
    @patch('litellm.proxy.pass_through_endpoints.llm_provider_handlers.openai_passthrough_logging_handler.OpenAIPassthroughLoggingHandler.get_provider_config')
    def test_responses_api_cost_tracking(self, mock_get_provider_config, mock_get_standard_logging, mock_completion_cost):
        """Test cost tracking for responses API route"""
        # Arrange
        mock_completion_cost.return_value = 0.000050
        mock_get_standard_logging.return_value = {"test": "logging_payload"}
        
        # Mock the provider config's transform_response to return a valid ModelResponse
        from litellm import ModelResponse
        mock_model_response = ModelResponse(
            id="resp_abc123",
            model="gpt-4o-2024-08-06",
            choices=[{
                "message": {
                    "role": "assistant",
                    "content": "Hello! How can I help you today?"
                }
            }],
            usage={
                "prompt_tokens": 20,
                "completion_tokens": 15,
                "total_tokens": 35
            }
        )
        
        mock_provider_config = MagicMock()
        mock_provider_config.transform_response.return_value = mock_model_response
        mock_get_provider_config.return_value = mock_provider_config
        
        # Mock responses API response
        mock_responses_response = {
            "id": "resp_abc123",
            "object": "response",
            "created": 1677652288,
            "model": "gpt-4o-2024-08-06",
            "output": [
                {
                    "type": "text",
                    "text": "Hello! How can I help you today?"
                }
            ],
            "usage": {
                "input_tokens": 20,
                "output_tokens": 15
            }
        }
        
        mock_httpx_response = self._create_mock_httpx_response(mock_responses_response)
        mock_logging_obj = self._create_mock_logging_obj()
        passthrough_payload = self._create_passthrough_logging_payload()
        
        kwargs = {
            "passthrough_logging_payload": passthrough_payload,
            "model": "gpt-4o",
            "custom_llm_provider": "openai",
        }

        # Act
        result = OpenAIPassthroughLoggingHandler.openai_passthrough_handler(
            httpx_response=mock_httpx_response,
            response_body=mock_responses_response,
            logging_obj=mock_logging_obj,
            url_route="https://api.openai.com/v1/responses",
            result="",
            start_time=self.start_time,
            end_time=self.end_time,
            cache_hit=False,
            request_body={"model": "gpt-4o", "input": "Tell me about AI"},
            **kwargs
        )

        # Assert
        assert result is not None
        assert "result" in result
        assert "kwargs" in result
        assert result["kwargs"]["response_cost"] == 0.000050
        assert result["kwargs"]["model"] == "gpt-4o"
        assert result["kwargs"]["custom_llm_provider"] == "openai"
        
        # Verify cost calculation was called with responses call type
        mock_completion_cost.assert_called_once()
        call_args = mock_completion_cost.call_args
        assert call_args[1]["call_type"] == "responses"
        assert call_args[1]["model"] == "gpt-4o"
        assert call_args[1]["custom_llm_provider"] == "openai"
        
        # Verify logging object was updated
        assert mock_logging_obj.model_call_details["response_cost"] == 0.000050
        assert mock_logging_obj.model_call_details["model"] == "gpt-4o"
        assert mock_logging_obj.model_call_details["custom_llm_provider"] == "openai"


class TestOpenAIPassthroughIntegration:
    """Integration tests for OpenAI passthrough cost tracking"""

    def setup_method(self):
        """Set up test fixtures"""
        self.handler = PassThroughEndpointLogging()
        self.start_time = datetime.now()
        self.end_time = datetime.now()

    def _create_mock_logging_obj(self) -> LiteLLMLoggingObj:
        """Create a mock logging object"""
        mock_logging_obj = MagicMock()
        mock_logging_obj.model_call_details = {}
        return mock_logging_obj

    def _create_mock_httpx_response(self, response_data: dict = None) -> httpx.Response:
        """Create a mock httpx response"""
        if response_data is None:
            response_data = {"id": "test", "choices": [{"message": {"content": "Hello"}}]}
            
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.text = json.dumps(response_data)
        mock_response.json.return_value = response_data
        mock_response.headers = {"content-type": "application/json"}
        return mock_response

    def _create_passthrough_logging_payload(self, user: str = "test_user") -> PassthroughStandardLoggingPayload:
        """Create a mock passthrough logging payload"""
        return PassthroughStandardLoggingPayload(
            url="https://api.openai.com/v1/chat/completions",
            request_body={"model": "gpt-4o", "messages": [{"role": "user", "content": "Hello"}]},
            request_method="POST",
        )

    def test_is_openai_route_detection(self):
        """Test OpenAI route detection in the main success handler"""
        # Positive cases
        assert (
            self.handler.is_openai_route("https://api.openai.com/v1/chat/completions")
            is True
        )
        assert (
            self.handler.is_openai_route(
                "https://openai.azure.com/v1/chat/completions"
            )
            is True
        )
        assert self.handler.is_openai_route("https://api.openai.com/v1/models") is True
        assert (
            self.handler.is_openai_route("https://openrouter.ai/api/v1/responses")
            is True
        )
        assert self.handler.is_openai_route("https://api.x.ai/v1/responses") is True

        # Negative cases
        assert (
            self.handler.is_openai_route(
                "http://localhost:4000/openai/v1/chat/completions"
            )
            is False
        )
        assert (
            self.handler.is_openai_route("https://api.anthropic.com/v1/messages")
            is False
        )
        assert (
            self.handler.is_openai_route("https://api.assemblyai.com/v2/transcript")
            is False
        )
        assert self.handler.is_openai_route("") is False

    @patch('litellm.proxy.pass_through_endpoints.llm_provider_handlers.openai_passthrough_logging_handler.OpenAIPassthroughLoggingHandler.openai_passthrough_handler')
    @pytest.mark.asyncio
    async def test_success_handler_calls_openai_handler(self, mock_openai_handler):
        """Test that the success handler calls our OpenAI handler for OpenAI routes"""
        # Arrange
        mock_openai_handler.return_value = {
            "result": {"id": "chatcmpl-123"},
            "kwargs": {
                "response_cost": 0.000045,
                "model": "gpt-4o",
                "custom_llm_provider": "openai"
            }
        }
        
        mock_httpx_response = MagicMock(spec=httpx.Response)
        mock_httpx_response.text = '{"id": "chatcmpl-123", "choices": [{"message": {"content": "Hello"}}]}'
        
        mock_logging_obj = AsyncMock()
        mock_logging_obj.model_call_details = {}
        mock_logging_obj.async_success_handler = AsyncMock()
        mock_logging_obj.get_combined_callback_list = MagicMock(return_value=[])
        
        passthrough_payload = PassthroughStandardLoggingPayload(
            url="https://api.openai.com/v1/chat/completions",
            request_body={"model": "gpt-4o", "messages": [{"role": "user", "content": "Hello"}]},
            request_method="POST",
        )

        # Act
        result = await self.handler.pass_through_async_success_handler(
            httpx_response=mock_httpx_response,
            response_body={"id": "chatcmpl-123", "choices": [{"message": {"content": "Hello"}}]},
            logging_obj=mock_logging_obj,
            url_route="https://api.openai.com/v1/chat/completions",
            result="",
            start_time=datetime.now(),
            end_time=datetime.now(),
            cache_hit=False,
            request_body={"model": "gpt-4o", "messages": [{"role": "user", "content": "Hello"}]},
            passthrough_logging_payload=passthrough_payload,
        )

        # Assert
        mock_openai_handler.assert_called_once()
        # The success handler returns None on success, which is expected
        assert result is None

    @pytest.mark.asyncio
    async def test_success_handler_falls_back_for_non_openai_routes(self):
        """Test that non-OpenAI routes don't call our handler"""
        # Arrange
        mock_httpx_response = MagicMock(spec=httpx.Response)
        mock_httpx_response.text = '{"status": "success"}'
        mock_httpx_response.headers = {"content-type": "application/json"}
        
        mock_logging_obj = MagicMock()
        mock_logging_obj.model_call_details = {}
        
        passthrough_payload = PassthroughStandardLoggingPayload(
            url="https://api.anthropic.com/v1/messages",
            request_body={"model": "claude-3-sonnet", "messages": [{"role": "user", "content": "Hello"}]},
            request_method="POST",
        )

        # Mock the _handle_logging method to capture calls
        self.handler._handle_logging = AsyncMock()

        # Act
        await self.handler.pass_through_async_success_handler(
            httpx_response=mock_httpx_response,
            response_body={"status": "success"},
            logging_obj=mock_logging_obj,
            url_route="https://api.anthropic.com/v1/messages",
            result="",
            start_time=datetime.now(),
            end_time=datetime.now(),
            cache_hit=False,
            request_body={"model": "claude-3-sonnet", "messages": [{"role": "user", "content": "Hello"}]},
            passthrough_logging_payload=passthrough_payload,
        )

        # Assert - Should call the base handler, not our OpenAI handler
        self.handler._handle_logging.assert_called_once()

    @patch('litellm.cost_calculator.default_image_cost_calculator')
    def test_calculate_image_generation_cost(self, mock_image_cost_calculator):
        """Test image generation cost calculation"""
        # Arrange
        mock_image_cost_calculator.return_value = 0.040
        model = "dall-e-3"
        response_body = {
            "data": [
                {
                    "url": "https://example.com/image1.png",
                    "revised_prompt": "A beautiful sunset over the ocean"
                }
            ]
        }
        request_body = {
            "model": "dall-e-3",
            "prompt": "A beautiful sunset over the ocean",
            "n": 1,
            "size": "1024x1024",
            "quality": "standard"
        }

        # Act
        cost = OpenAIPassthroughLoggingHandler._calculate_image_generation_cost(
            model=model,
            response_body=response_body,
            request_body=request_body,
        )

        # Assert
        assert cost == 0.040
        mock_image_cost_calculator.assert_called_once_with(
            model=model,
            custom_llm_provider="openai",
            quality="standard",
            n=1,
            size="1024x1024",
            optional_params=request_body,
        )

    @patch('litellm.cost_calculator.default_image_cost_calculator')
    def test_calculate_image_editing_cost(self, mock_image_cost_calculator):
        """Test image editing cost calculation"""
        # Arrange
        mock_image_cost_calculator.return_value = 0.020
        model = "dall-e-2"
        response_body = {
            "data": [
                {
                    "url": "https://example.com/edited_image.png",
                    "revised_prompt": "A beautiful sunset over the ocean with added clouds"
                }
            ]
        }
        request_body = {
            "model": "dall-e-2",
            "prompt": "Add clouds to the sky",
            "n": 1,
            "size": "1024x1024"
        }

        # Act
        cost = OpenAIPassthroughLoggingHandler._calculate_image_editing_cost(
            model=model,
            response_body=response_body,
            request_body=request_body,
        )

        # Assert
        assert cost == 0.020
        mock_image_cost_calculator.assert_called_once_with(
            model=model,
            custom_llm_provider="openai",
            quality=None,  # Image editing doesn't have quality parameter
            n=1,
            size="1024x1024",
            optional_params=request_body,
        )

    def test_cost_calculation_preservation(self):
        """Test that manually calculated costs are preserved and not overridden."""
        # Create a logging object
        logging_obj = LiteLLMLoggingObj(
            model="dall-e-3",
            messages=[{"role": "user", "content": "Generate an image"}],
            stream=False,
            call_type="pass_through_endpoint",
            start_time=self.start_time,
            litellm_call_id="test_123",
            function_id="test_fn",
        )
        
        # Set a manually calculated cost in model_call_details
        test_cost = 0.040000
        logging_obj.model_call_details["response_cost"] = test_cost
        logging_obj.model_call_details["model"] = "dall-e-3"
        logging_obj.model_call_details["custom_llm_provider"] = "openai"
        
        # Create an ImageResponse with cost in _hidden_params
        from litellm.types.utils import ImageResponse
        image_response = ImageResponse(
            data=[{"url": "https://example.com/image.png"}],
            model="dall-e-3",
        )
        image_response._hidden_params = {"response_cost": test_cost}
        
        # Test the _response_cost_calculator method
        calculated_cost = logging_obj._response_cost_calculator(result=image_response)
        
        assert calculated_cost == test_cost, f"Expected {test_cost}, got {calculated_cost}"

    @patch('litellm.cost_calculator.default_image_cost_calculator')
    def test_openai_passthrough_handler_image_generation(self, mock_image_cost_calculator):
        """Test successful cost tracking for OpenAI image generation"""
        # Arrange
        mock_image_cost_calculator.return_value = 0.040
        
        mock_image_response = {
            "data": [
                {
                    "url": "https://example.com/image1.png",
                    "revised_prompt": "A beautiful sunset over the ocean"
                }
            ]
        }
        
        mock_httpx_response = self._create_mock_httpx_response(mock_image_response)
        mock_logging_obj = self._create_mock_logging_obj()
        passthrough_payload = self._create_passthrough_logging_payload()
        
        kwargs = {
            "passthrough_logging_payload": passthrough_payload,
            "model": "dall-e-3",
        }

        request_body = {
            "model": "dall-e-3",
            "prompt": "A beautiful sunset over the ocean",
            "n": 1,
            "size": "1024x1024",
            "quality": "standard"
        }

        # Act
        result = OpenAIPassthroughLoggingHandler.openai_passthrough_handler(
            httpx_response=mock_httpx_response,
            response_body=mock_image_response,
            logging_obj=mock_logging_obj,
            url_route="https://api.openai.com/v1/images/generations",
            result="",
            start_time=self.start_time,
            end_time=self.end_time,
            cache_hit=False,
            request_body=request_body,
            **kwargs
        )

        # Assert
        assert result is not None
        assert "result" in result
        assert "kwargs" in result
        assert result["kwargs"]["response_cost"] == 0.040
        assert result["kwargs"]["model"] == "dall-e-3"
        assert result["kwargs"]["custom_llm_provider"] == "openai"
        
        # Verify cost calculation was called
        mock_image_cost_calculator.assert_called_once()
        
        # Verify logging object was updated
        assert mock_logging_obj.model_call_details["response_cost"] == 0.040
        assert mock_logging_obj.model_call_details["model"] == "dall-e-3"
        assert mock_logging_obj.model_call_details["custom_llm_provider"] == "openai"

    @patch('litellm.cost_calculator.default_image_cost_calculator')
    def test_openai_passthrough_handler_image_editing(self, mock_image_cost_calculator):
        """Test successful cost tracking for OpenAI image editing"""
        # Arrange
        mock_image_cost_calculator.return_value = 0.020
        
        mock_image_response = {
            "data": [
                {
                    "url": "https://example.com/edited_image.png",
                    "revised_prompt": "A beautiful sunset over the ocean with added clouds"
                }
            ]
        }
        
        mock_httpx_response = self._create_mock_httpx_response(mock_image_response)
        mock_logging_obj = self._create_mock_logging_obj()
        passthrough_payload = self._create_passthrough_logging_payload()
        
        kwargs = {
            "passthrough_logging_payload": passthrough_payload,
            "model": "dall-e-2",
        }

        request_body = {
            "model": "dall-e-2",
            "prompt": "Add clouds to the sky",
            "n": 1,
            "size": "1024x1024"
        }

        # Act
        result = OpenAIPassthroughLoggingHandler.openai_passthrough_handler(
            httpx_response=mock_httpx_response,
            response_body=mock_image_response,
            logging_obj=mock_logging_obj,
            url_route="https://api.openai.com/v1/images/edits",
            result="",
            start_time=self.start_time,
            end_time=self.end_time,
            cache_hit=False,
            request_body=request_body,
            **kwargs
        )

        # Assert
        assert result is not None
        assert "result" in result
        assert "kwargs" in result
        assert result["kwargs"]["response_cost"] == 0.020
        assert result["kwargs"]["model"] == "dall-e-2"
        assert result["kwargs"]["custom_llm_provider"] == "openai"
        
        # Verify cost calculation was called
        mock_image_cost_calculator.assert_called_once()
        
        # Verify logging object was updated
        assert mock_logging_obj.model_call_details["response_cost"] == 0.020
        assert mock_logging_obj.model_call_details["model"] == "dall-e-2"
        assert mock_logging_obj.model_call_details["custom_llm_provider"] == "openai"


class TestModelPriceLookupCache:
    """RR-055: on-disk model price map and negative lookup caches."""

    def setup_method(self):
        _reset_model_price_lookup_caches()

    def teardown_method(self):
        _reset_model_price_lookup_caches()

    def test_lookup_reads_and_parses_each_price_source_once_across_hits(
        self, tmp_path, monkeypatch
    ):
        """Repeated hits must reuse the parsed map, not re-read/re-parse JSON."""
        package_root = tmp_path / "litellm_pkg"
        package_root.mkdir()
        package_file = package_root / "__init__.py"
        package_file.write_text("# stub litellm package path\n", encoding="utf-8")

        primary = package_root / "model_prices_and_context_window.json"
        bundled = package_root / "bundled_model_prices_and_context_window_fallback.json"
        parent = package_root.parent / "model_prices_and_context_window.json"
        # Put the hit only in the last source so the first lookup must open every
        # source once, proving map-level memoization across repeated hits.
        primary.write_text("{}", encoding="utf-8")
        bundled.write_text("{}", encoding="utf-8")
        parent.write_text(
            json.dumps(
                {
                    "openrouter/test-model-hit": {
                        "input_cost_per_token": 0.001,
                        "output_cost_per_token": 0.002,
                    }
                }
            ),
            encoding="utf-8",
        )

        monkeypatch.setattr(litellm, "__file__", str(package_file))
        monkeypatch.setattr(litellm, "model_cost", {}, raising=False)

        read_counts = {"count": 0}
        original_read_text = type(primary).read_text

        def counting_read_text(self, *args, **kwargs):
            read_counts["count"] += 1
            return original_read_text(self, *args, **kwargs)

        monkeypatch.setattr(type(primary), "read_text", counting_read_text)

        first = OpenAIPassthroughLoggingHandler._lookup_model_price_info(
            "openrouter/test-model-hit",
            "openrouter",
        )
        second = OpenAIPassthroughLoggingHandler._lookup_model_price_info(
            "openrouter/test-model-hit",
            "openrouter",
        )
        third = OpenAIPassthroughLoggingHandler._lookup_model_price_info(
            "openrouter/test-model-hit",
            "openrouter",
        )

        assert first == {
            "input_cost_per_token": 0.001,
            "output_cost_per_token": 0.002,
        }
        assert second is first
        assert third is first
        # One parse/read per source path (primary + bundled + parent), not per request.
        assert read_counts["count"] == 3
        assert len(_MODEL_PRICE_MAP_CACHE) == 3

    def test_lookup_caches_negative_disk_misses_without_rereads(
        self, tmp_path, monkeypatch
    ):
        """Known-absent models must skip subsequent on-disk map scans."""
        package_root = tmp_path / "litellm_pkg"
        package_root.mkdir()
        package_file = package_root / "__init__.py"
        package_file.write_text("# stub litellm package path\n", encoding="utf-8")

        primary = package_root / "model_prices_and_context_window.json"
        bundled = package_root / "bundled_model_prices_and_context_window_fallback.json"
        parent = package_root.parent / "model_prices_and_context_window.json"
        primary.write_text(json.dumps({"other-model": {"input_cost_per_token": 1}}), encoding="utf-8")
        bundled.write_text("{}", encoding="utf-8")
        parent.write_text("{}", encoding="utf-8")

        monkeypatch.setattr(litellm, "__file__", str(package_file))
        monkeypatch.setattr(litellm, "model_cost", {}, raising=False)

        read_counts = {"count": 0}
        original_read_text = type(primary).read_text

        def counting_read_text(self, *args, **kwargs):
            read_counts["count"] += 1
            return original_read_text(self, *args, **kwargs)

        monkeypatch.setattr(type(primary), "read_text", counting_read_text)

        missing = "totally-unmapped-model-rr055"
        first = OpenAIPassthroughLoggingHandler._lookup_model_price_info(
            missing,
            "openrouter",
        )
        second = OpenAIPassthroughLoggingHandler._lookup_model_price_info(
            missing,
            "openrouter",
        )
        third = OpenAIPassthroughLoggingHandler._lookup_model_price_info(
            missing,
            "openrouter",
        )

        assert first is None
        assert second is None
        assert third is None
        # Initial miss reads each source once; later misses use negative cache.
        assert read_counts["count"] == 3
        assert missing in _MODEL_PRICE_DISK_MISS_CACHE
        assert f"openrouter/{missing}" in _MODEL_PRICE_DISK_MISS_CACHE

    def test_lookup_prefers_runtime_model_cost_over_disk_cache(
        self, tmp_path, monkeypatch
    ):
        """Runtime litellm.model_cost remains the first source and is not cached away."""
        package_root = tmp_path / "litellm_pkg"
        package_root.mkdir()
        package_file = package_root / "__init__.py"
        package_file.write_text("# stub litellm package path\n", encoding="utf-8")

        primary = package_root / "model_prices_and_context_window.json"
        primary.write_text(
            json.dumps(
                {
                    "runtime-vs-disk": {
                        "input_cost_per_token": 0.1,
                        "output_cost_per_token": 0.2,
                    }
                }
            ),
            encoding="utf-8",
        )
        (package_root / "bundled_model_prices_and_context_window_fallback.json").write_text(
            "{}",
            encoding="utf-8",
        )
        (package_root.parent / "model_prices_and_context_window.json").write_text(
            "{}",
            encoding="utf-8",
        )

        monkeypatch.setattr(litellm, "__file__", str(package_file))
        monkeypatch.setattr(
            litellm,
            "model_cost",
            {
                "runtime-vs-disk": {
                    "input_cost_per_token": 9.0,
                    "output_cost_per_token": 8.0,
                }
            },
            raising=False,
        )

        result = OpenAIPassthroughLoggingHandler._lookup_model_price_info(
            "runtime-vs-disk",
            None,
        )
        assert result == {
            "input_cost_per_token": 9.0,
            "output_cost_per_token": 8.0,
        }
        # Runtime hit must not populate on-disk caches.
        assert _MODEL_PRICE_MAP_CACHE == {}
        assert _MODEL_PRICE_DISK_MISS_CACHE == set()

    def test_reset_helper_clears_process_lifetime_caches(self, tmp_path, monkeypatch):
        package_root = tmp_path / "litellm_pkg"
        package_root.mkdir()
        package_file = package_root / "__init__.py"
        package_file.write_text("# stub litellm package path\n", encoding="utf-8")
        (package_root / "model_prices_and_context_window.json").write_text(
            "{}",
            encoding="utf-8",
        )
        (package_root / "bundled_model_prices_and_context_window_fallback.json").write_text(
            "{}",
            encoding="utf-8",
        )
        (package_root.parent / "model_prices_and_context_window.json").write_text(
            "{}",
            encoding="utf-8",
        )
        monkeypatch.setattr(litellm, "__file__", str(package_file))
        monkeypatch.setattr(litellm, "model_cost", {}, raising=False)

        OpenAIPassthroughLoggingHandler._lookup_model_price_info(
            "never-exists-rr055",
            None,
        )
        assert _MODEL_PRICE_MAP_CACHE
        assert "never-exists-rr055" in _MODEL_PRICE_DISK_MISS_CACHE

        _reset_model_price_lookup_caches()
        assert _MODEL_PRICE_MAP_CACHE == {}
        assert _MODEL_PRICE_DISK_MISS_CACHE == set()




if __name__ == "__main__":
    pytest.main([__file__])
