import json
import os
import sys
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

sys.path.insert(0, os.path.abspath("../../.."))  # Adds the parent directory to the system path

from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.proxy.pass_through_endpoints.llm_provider_handlers.gemini_passthrough_logging_handler import (
    GeminiPassthroughLoggingHandler,
)
from litellm.proxy.pass_through_endpoints.success_handler import (
    PassThroughEndpointLogging,
)
from litellm.types.passthrough_endpoints.pass_through_endpoints import (
    PassthroughStandardLoggingPayload,
)


class TestGeminiPassthroughLoggingHandler:
    """Test the Gemini passthrough logging handler for native Gemini / Google AI Studio."""

    def setup_method(self):
        """Set up test fixtures"""
        self.start_time = datetime.now()
        self.end_time = datetime.now()
        self.handler = GeminiPassthroughLoggingHandler()

        # Native Gemini generateContent response
        self.mock_gemini_response = {
            "candidates": [
                {
                    "content": {"parts": [{"text": "Hello! How can I help you today?"}], "role": "model"},
                    "finishReason": "STOP",
                    "index": 0,
                    "safetyRatings": [
                        {"category": "HARM_CATEGORY_HARASSMENT", "probability": "NEGLIGIBLE"},
                        {"category": "HARM_CATEGORY_HATE_SPEECH", "probability": "NEGLIGIBLE"},
                        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "probability": "NEGLIGIBLE"},
                        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "probability": "NEGLIGIBLE"},
                    ],
                }
            ],
            "usageMetadata": {
                "promptTokenCount": 10,
                "candidatesTokenCount": 8,
                "totalTokenCount": 18,
                "cachedContentTokenCount": 4,
                "cacheWriteInputTokens": 3,
                "cacheWriteInputTokenCount": 2,
                "cacheCreationInputTokens": 1,
            },
        }

    def _create_mock_httpx_response(self) -> httpx.Response:
        """Create a mock httpx.Response for testing"""
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.text = json.dumps(self.mock_gemini_response)
        mock_response.json.return_value = self.mock_gemini_response
        mock_response.headers = {"content-type": "application/json"}
        return mock_response

    def _create_mock_logging_obj(self) -> LiteLLMLoggingObj:
        """Create a mock logging object for testing"""
        mock_logging_obj = MagicMock(spec=LiteLLMLoggingObj)
        mock_logging_obj.model_call_details = {}
        mock_logging_obj.optional_params = {}
        mock_logging_obj.litellm_call_id = "test-call-id-123"
        return mock_logging_obj

    def _create_passthrough_logging_payload(self) -> PassthroughStandardLoggingPayload:
        """Create a mock passthrough logging payload for testing"""
        return PassthroughStandardLoggingPayload(
            url="https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent",
            request_body={"contents": [{"parts": [{"text": "Hello"}]}]},
            request_method="POST",
        )

    # ------------------------------------------------------------------
    # Route identification
    # ------------------------------------------------------------------

    def test_is_gemini_route(self):
        """Test that Gemini routes are correctly identified"""
        handler = PassThroughEndpointLogging()

        assert (
            handler.is_gemini_route(
                "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent",
                custom_llm_provider="gemini",
            )
            is True
        )
        assert (
            handler.is_gemini_route(
                "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:streamGenerateContent",
                custom_llm_provider="gemini",
            )
            is True
        )
        # Non-Gemini endpoints must not match
        assert (
            handler.is_gemini_route(
                "https://api.openai.com/v1/chat/completions",
                custom_llm_provider="openai",
            )
            is False
        )

    # ------------------------------------------------------------------
    # Model extraction
    # ------------------------------------------------------------------

    def test_extract_model_from_url(self):
        """Test that model is correctly extracted from Gemini URLs"""
        model = GeminiPassthroughLoggingHandler.extract_model_from_url(
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"
        )
        assert model == "gemini-1.5-flash"

        model = GeminiPassthroughLoggingHandler.extract_model_from_url(
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-pro:streamGenerateContent"
        )
        assert model == "gemini-1.5-pro"

    def test_extract_model_from_request_body_fallback(self):
        """When the URL has no /models/ segment, fall back to request_body."""
        model = GeminiPassthroughLoggingHandler.extract_model_from_url(
            "https://generativelanguage.googleapis.com/v1beta:generateContent",
            request_body={"model": "gemini-2.5-flash"},
        )
        assert model == "gemini-2.5-flash"

    # ------------------------------------------------------------------
    # SSE / stream payload parsing helpers
    # ------------------------------------------------------------------

    def test_parse_stream_payload_json_objects_dict(self):
        result = GeminiPassthroughLoggingHandler._parse_stream_payload_json_objects(
            '{"candidates": []}'
        )
        assert result == [{"candidates": []}]

    def test_parse_stream_payload_json_objects_list(self):
        result = GeminiPassthroughLoggingHandler._parse_stream_payload_json_objects(
            '[{"a": 1}, {"b": 2}]'
        )
        assert result == [{"a": 1}, {"b": 2}]

    def test_parse_stream_payload_json_objects_done(self):
        assert GeminiPassthroughLoggingHandler._parse_stream_payload_json_objects("[DONE]") == []

    def test_extract_sse_data_payloads(self):
        chunk = "data: {\"a\": 1}\n\ndata: {\"b\": 2}\n\n"
        payloads = GeminiPassthroughLoggingHandler._extract_sse_data_payloads(chunk)
        assert payloads == ['{"a": 1}', '{"b": 2}']

    def test_parse_stream_chunk_json_objects_sse(self):
        chunk = "data: {\"candidates\": []}\n\ndata: [DONE]\n\n"
        result = GeminiPassthroughLoggingHandler._parse_stream_chunk_json_objects(chunk)
        assert result == [{"candidates": []}]

    # ------------------------------------------------------------------
    # Usage extraction
    # ------------------------------------------------------------------

    def test_extract_usage_object_from_response_body(self):
        usage = GeminiPassthroughLoggingHandler._extract_usage_object_from_response_body(
            self.mock_gemini_response
        )
        assert usage is not None
        assert usage["promptTokenCount"] == 10
        assert usage["totalTokenCount"] == 18

    def test_extract_usage_object_from_response_body_missing(self):
        assert GeminiPassthroughLoggingHandler._extract_usage_object_from_response_body(
            {"candidates": []}
        ) is None

    def test_extract_usage_object_from_response_body_list(self):
        """The last usage-bearing list item supplies usageMetadata."""
        body = [
            {
                "candidates": [{"content": {"parts": [{"text": "Hi"}]}}],
                "usageMetadata": {
                    "promptTokenCount": 2,
                    "candidatesTokenCount": 1,
                    "totalTokenCount": 3,
                },
            },
            {
                "candidates": [{"content": {"parts": [{"text": " there"}]}}],
                "usageMetadata": {
                    "promptTokenCount": 7,
                    "candidatesTokenCount": 3,
                    "totalTokenCount": 10,
                },
            },
        ]
        usage = GeminiPassthroughLoggingHandler._extract_usage_object_from_response_body(body)
        assert usage is not None
        assert usage["promptTokenCount"] == 7
        assert usage["totalTokenCount"] == 10

    def test_extract_usage_object_from_response_body_list_trailing_malformed_usage(
        self,
    ):
        """A malformed final usageMetadata field overrides earlier valid usage."""
        body = [
            {
                "usageMetadata": {
                    "promptTokenCount": 7,
                    "candidatesTokenCount": 3,
                    "totalTokenCount": 10,
                }
            },
            {"candidates": [{"content": {"parts": [{"text": "Hi"}]}}]},
            {"usageMetadata": "malformed"},
        ]

        assert (
            GeminiPassthroughLoggingHandler._extract_usage_object_from_response_body(
                body
            )
            is None
        )

    def test_extract_usage_object_from_response_body_list_no_usage(self):
        """List without any usageMetadata returns None."""
        assert GeminiPassthroughLoggingHandler._extract_usage_object_from_response_body(
            [{"candidates": []}]
        ) is None

    def test_extract_usage_object_from_response_body_non_dict(self):
        """Non-dict, non-list values return None."""
        assert GeminiPassthroughLoggingHandler._extract_usage_object_from_response_body(
            "not a dict"
        ) is None

    def test_extract_usage_object_from_stream_chunks(self):
        chunks = [
            "data: " + json.dumps({"candidates": [{"content": {"parts": [{"text": "Hi"}]}}]}) + "\n\n",
            "data: " + json.dumps({
                "candidates": [{"content": {"parts": [{"text": " there"}]}}],
                "usageMetadata": {"promptTokenCount": 5, "candidatesTokenCount": 3, "totalTokenCount": 8},
            }) + "\n\n",
            "data: [DONE]\n\n",
        ]
        usage = GeminiPassthroughLoggingHandler._extract_usage_object_from_stream_chunks(chunks)
        assert usage is not None
        assert usage["totalTokenCount"] == 8

    # ------------------------------------------------------------------
    # Non-streaming handler (generateContent)
    # ------------------------------------------------------------------

    @patch("litellm.completion_cost")
    @patch("litellm.litellm_core_utils.litellm_logging.get_standard_logging_object_payload")
    def test_gemini_passthrough_handler_success(self, mock_get_standard_logging, mock_completion_cost):
        """Test successful cost tracking for Gemini generateContent endpoint"""
        mock_completion_cost.return_value = 0.000045
        mock_get_standard_logging.return_value = {"test": "logging_payload"}

        mock_httpx_response = self._create_mock_httpx_response()
        mock_logging_obj = self._create_mock_logging_obj()
        passthrough_payload = self._create_passthrough_logging_payload()

        kwargs = {
            "passthrough_logging_payload": passthrough_payload,
            "model": "gemini-1.5-flash",
        }

        result = GeminiPassthroughLoggingHandler.gemini_passthrough_handler(
            httpx_response=mock_httpx_response,
            response_body=self.mock_gemini_response,
            logging_obj=mock_logging_obj,
            url_route="https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent",
            result="",
            start_time=self.start_time,
            end_time=self.end_time,
            cache_hit=False,
            request_body={"contents": [{"parts": [{"text": "Hello"}]}]},
            **kwargs,
        )

        assert result is not None
        assert "result" in result
        assert "kwargs" in result
        assert result["kwargs"]["response_cost"] == 0.000045
        assert result["kwargs"]["model"] == "gemini-1.5-flash"
        assert result["kwargs"]["custom_llm_provider"] == "gemini"
        assert result["kwargs"]["litellm_params"]["metadata"]["usage_object"] == {
            "promptTokenCount": 10,
            "candidatesTokenCount": 8,
            "totalTokenCount": 18,
            "cachedContentTokenCount": 4,
            "cacheWriteInputTokens": 3,
            "cacheWriteInputTokenCount": 2,
            "cacheCreationInputTokens": 1,
        }

        mock_completion_cost.assert_called_once()
        assert mock_completion_cost.call_args.kwargs["model"] == "gemini-1.5-flash"
        assert mock_completion_cost.call_args.kwargs["custom_llm_provider"] == "gemini"

        assert mock_logging_obj.model_call_details["response_cost"] == 0.000045
        assert mock_logging_obj.model_call_details["model"] == "gemini-1.5-flash"
        assert mock_logging_obj.model_call_details["custom_llm_provider"] == "gemini"

    @patch("litellm.completion_cost")
    def test_gemini_passthrough_handler_streaming(self, mock_completion_cost):
        """Test cost tracking for Gemini streaming endpoint (list response_body)"""
        mock_completion_cost.return_value = 0.000030

        mock_chunks = [
            {"candidates": [{"content": {"parts": [{"text": "Hello"}]}}]},
            {
                "candidates": [{"content": {"parts": [{"text": " there!"}]}}],
                "usageMetadata": {
                    "promptTokenCount": 10,
                    "candidatesTokenCount": 2,
                    "totalTokenCount": 12,
                    "cachedContentTokenCount": 6,
                    "cacheWriteInputTokens": 5,
                    "cacheWriteInputTokenCount": 4,
                    "cacheCreationInputTokens": 3,
                },
            },
        ]

        mock_httpx_response = self._create_mock_httpx_response()
        mock_logging_obj = self._create_mock_logging_obj()
        passthrough_payload = self._create_passthrough_logging_payload()

        kwargs = {
            "passthrough_logging_payload": passthrough_payload,
            "model": "gemini-1.5-flash",
        }

        result = GeminiPassthroughLoggingHandler.gemini_passthrough_handler(
            httpx_response=mock_httpx_response,
            response_body=mock_chunks,
            logging_obj=mock_logging_obj,
            url_route="https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent",
            result="",
            start_time=self.start_time,
            end_time=self.end_time,
            cache_hit=False,
            request_body={"contents": [{"parts": [{"text": "Hello"}]}]},
            **kwargs,
        )

        assert result is not None
        assert result["kwargs"]["response_cost"] == 0.000030
        assert result["kwargs"]["model"] == "gemini-1.5-flash"
        assert result["kwargs"]["custom_llm_provider"] == "gemini"
        assert result["kwargs"]["litellm_params"]["metadata"]["usage_object"] == {
            "promptTokenCount": 10,
            "candidatesTokenCount": 2,
            "totalTokenCount": 12,
            "cachedContentTokenCount": 6,
            "cacheWriteInputTokens": 5,
            "cacheWriteInputTokenCount": 4,
            "cacheCreationInputTokens": 3,
        }
        mock_completion_cost.assert_called_once()

    def test_gemini_passthrough_handler_non_gemini_route(self):
        """Test that non-Gemini routes return None"""
        mock_httpx_response = self._create_mock_httpx_response()
        mock_logging_obj = self._create_mock_logging_obj()
        passthrough_payload = self._create_passthrough_logging_payload()

        kwargs = {
            "passthrough_logging_payload": passthrough_payload,
            "model": "gpt-4o",
        }

        result = GeminiPassthroughLoggingHandler.gemini_passthrough_handler(
            httpx_response=mock_httpx_response,
            response_body=self.mock_gemini_response,
            logging_obj=mock_logging_obj,
            url_route="https://api.openai.com/v1/chat/completions",
            result="",
            start_time=self.start_time,
            end_time=self.end_time,
            cache_hit=False,
            request_body={"model": "gpt-4o", "messages": [{"role": "user", "content": "Hello"}]},
            **kwargs,
        )

        assert result is not None
        assert result["result"] is None
        assert "kwargs" in result

    # ------------------------------------------------------------------
    # Streaming: _build_complete_streaming_response with native chunks
    # ------------------------------------------------------------------

    def test_build_complete_streaming_response_native_sse_chunks(self):
        """Native Gemini SSE data frames are assembled into a ModelResponse."""
        mock_logging_obj = self._create_mock_logging_obj()
        chunk1 = "data: " + json.dumps({
            "candidates": [
                {
                    "content": {"parts": [{"text": "Hello"}], "role": "model"},
                    "index": 0,
                }
            ],
        })
        chunk2 = "data: " + json.dumps({
            "candidates": [
                {
                    "content": {"parts": [{"text": " world"}], "role": "model"},
                    "finishReason": "STOP",
                    "index": 0,
                }
            ],
            "usageMetadata": {
                "promptTokenCount": 5,
                "candidatesTokenCount": 2,
                "totalTokenCount": 7,
            },
        })

        response = GeminiPassthroughLoggingHandler._build_complete_streaming_response(
            all_chunks=[chunk1, chunk2, "data: [DONE]"],
            litellm_logging_obj=mock_logging_obj,
            model="gemini-2.0-flash",
            url_route="https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:streamGenerateContent",
        )

        assert response is not None
        assert response.model == "gemini-2.0-flash"
        assert "Hello" in response.choices[0].message.content
        assert response.usage.prompt_tokens == 5
        assert response.usage.completion_tokens == 2

    def test_build_complete_streaming_response_native_dict_chunks(self):
        """Plain JSON dict chunks (no SSE framing) are handled."""
        mock_logging_obj = self._create_mock_logging_obj()
        chunk = json.dumps({
            "candidates": [
                {
                    "content": {"parts": [{"text": "plain dict"}], "role": "model"},
                    "finishReason": "STOP",
                    "index": 0,
                }
            ],
            "usageMetadata": {
                "promptTokenCount": 3,
                "candidatesTokenCount": 2,
                "totalTokenCount": 5,
            },
        })

        response = GeminiPassthroughLoggingHandler._build_complete_streaming_response(
            all_chunks=[chunk],
            litellm_logging_obj=mock_logging_obj,
            model="gemini-1.5-pro",
            url_route="https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-pro:generateContent",
        )

        assert response is not None
        assert response.model == "gemini-1.5-pro"
        assert response.usage.prompt_tokens == 3

    def test_build_complete_streaming_response_non_generate_content_returns_none(self):
        """URLs without generateContent/streamGenerateContent return None."""
        mock_logging_obj = self._create_mock_logging_obj()
        response = GeminiPassthroughLoggingHandler._build_complete_streaming_response(
            all_chunks=['{"candidates": []}'],
            litellm_logging_obj=mock_logging_obj,
            model="gemini-1.5-flash",
            url_route="https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:predictLongRunning",
        )
        assert response is None

    # ------------------------------------------------------------------
    # Streaming: collected chunks handler
    # ------------------------------------------------------------------

    @patch("litellm.completion_cost")
    def test_handle_logging_gemini_collected_chunks_stores_usage_object(
        self, mock_completion_cost
    ):
        """Native Gemini SSE chunks produce usage_object in kwargs metadata."""
        mock_completion_cost.return_value = 0.000030
        mock_logging_obj = self._create_mock_logging_obj()
        chunk = "data: " + json.dumps({
            "candidates": [
                {
                    "content": {"parts": [{"text": "gemini streamed"}], "role": "model"},
                    "finishReason": "STOP",
                    "index": 0,
                }
            ],
            "usageMetadata": {
                "promptTokenCount": 14,
                "candidatesTokenCount": 11,
                "totalTokenCount": 25,
                "cachedContentTokenCount": 8,
                "cacheWriteInputTokens": 7,
                "cacheWriteInputTokenCount": 6,
                "cacheCreationInputTokens": 5,
                "candidatesTokensDetails": [
                    {"modality": "THOUGHT", "tokenCount": 5},
                    {"modality": "TEXT", "tokenCount": 6},
                ],
            },
        })

        result = GeminiPassthroughLoggingHandler._handle_logging_gemini_collected_chunks(
            litellm_logging_obj=mock_logging_obj,
            passthrough_success_handler_obj=PassThroughEndpointLogging(),
            url_route="https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:streamGenerateContent",
            request_body={},
            endpoint_type=MagicMock(),
            start_time=self.start_time,
            all_chunks=[chunk, "data: [DONE]"],
            model="gemini-2.0-flash",
            end_time=self.end_time,
        )

        usage = result["kwargs"]["litellm_params"]["metadata"]["usage_object"]
        assert usage["totalTokenCount"] == 25
        assert usage["candidatesTokensDetails"][0]["modality"] == "THOUGHT"
        assert usage["cachedContentTokenCount"] == 8
        assert usage["cacheWriteInputTokens"] == 7
        assert usage["cacheWriteInputTokenCount"] == 6
        assert usage["cacheCreationInputTokens"] == 5

    # ------------------------------------------------------------------
    # Iterator wiring
    # ------------------------------------------------------------------

    def test_build_complete_response_from_gemini_stream_chunks_uses_required_iterator_args(self):
        mock_logging_obj = self._create_mock_logging_obj()
        chunk = "data: " + json.dumps({
            "candidates": [
                {
                    "content": {"parts": [{"text": "gemini streamed"}], "role": "model"},
                    "finishReason": "STOP",
                    "index": 0,
                }
            ],
            "usageMetadata": {
                "promptTokenCount": 10,
                "candidatesTokenCount": 2,
                "totalTokenCount": 12,
            },
        })

        with patch(
            "litellm.proxy.pass_through_endpoints.llm_provider_handlers.gemini_passthrough_logging_handler.GeminiModelResponseIterator"
        ) as mock_iterator_cls:
            mock_iterator = MagicMock()
            mock_stream_chunk = MagicMock()
            mock_iterator._common_chunk_parsing_logic.return_value = mock_stream_chunk
            mock_iterator.pending_model_response_chunks = []
            mock_iterator.chunk_type = "valid_json"
            mock_iterator.accumulated_json = ""
            mock_iterator_cls.return_value = mock_iterator

            with patch(
                "litellm.proxy.pass_through_endpoints.llm_provider_handlers.gemini_passthrough_logging_handler.litellm.stream_chunk_builder",
                return_value=MagicMock(),
            ) as mock_stream_chunk_builder:
                result = GeminiPassthroughLoggingHandler._build_complete_response_from_gemini_stream_chunks(
                    all_chunks=[chunk, "data: [DONE]"],
                    litellm_logging_obj=mock_logging_obj,
                )

        assert result is not None
        mock_iterator_cls.assert_called_once_with(
            streaming_response=None,
            sync_stream=False,
            logging_obj=mock_logging_obj,
        )
        assert mock_stream_chunk_builder.call_count == 1
        assert len(mock_stream_chunk_builder.call_args.kwargs["chunks"]) >= 1

    # ------------------------------------------------------------------
    # Cost lookup
    # ------------------------------------------------------------------

    def test_get_cost_lookup_model_provider_native_gemini(self):
        assert GeminiPassthroughLoggingHandler._get_cost_lookup_model_provider(
            model="gemini-1.5-flash",
            custom_llm_provider="gemini",
        ) == ("gemini-1.5-flash", "gemini")

    # ------------------------------------------------------------------
    # Success handler integration
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    @patch(
        "litellm.proxy.pass_through_endpoints.llm_provider_handlers.gemini_passthrough_logging_handler.litellm.completion_cost",
        return_value=0.000050,
    )
    async def test_pass_through_success_handler_gemini_routing(self, mock_completion_cost):
        """Test that the success handler correctly routes Gemini requests to the Gemini handler"""
        handler = PassThroughEndpointLogging()
        mock_logging_obj = self._create_mock_logging_obj()
        handler._handle_logging = AsyncMock()
        mock_response = self._create_mock_httpx_response()
        passthrough_logging_payload = self._create_passthrough_logging_payload()

        result = await handler.pass_through_async_success_handler(
            httpx_response=mock_response,
            response_body=self.mock_gemini_response,
            logging_obj=mock_logging_obj,
            url_route="https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent",
            result="",
            start_time=self.start_time,
            end_time=self.end_time,
            cache_hit=False,
            request_body={"contents": [{"parts": [{"text": "Hello"}]}]},
            passthrough_logging_payload=passthrough_logging_payload,
            custom_llm_provider="gemini",
        )

        assert result is None
        assert mock_logging_obj.model_call_details["response_cost"] == 0.000050
        assert mock_logging_obj.model_call_details["model"] == "gemini-2.0-flash"
        assert mock_logging_obj.model_call_details["custom_llm_provider"] == "gemini"

        handler._handle_logging.assert_called_once()
        call_kwargs = handler._handle_logging.call_args[1]
        assert call_kwargs["response_cost"] == 0.000050
        assert call_kwargs["model"] == "gemini-2.0-flash"
        assert call_kwargs["custom_llm_provider"] == "gemini"

    # ------------------------------------------------------------------
    # Video (predictLongRunning)
    # ------------------------------------------------------------------

    @patch("litellm.completion_cost")
    def test_veo3_passthrough_cost_tracking(self, mock_completion_cost):
        """Test Veo3 video generation cost tracking for passthrough requests"""
        expected_cost = 0.35 * 8.0  # $2.80
        mock_completion_cost.return_value = expected_cost

        mock_veo_response = {"name": "operations/1234567890123456789"}

        mock_httpx_response = MagicMock(spec=httpx.Response)
        mock_httpx_response.status_code = 200
        mock_httpx_response.json.return_value = mock_veo_response
        mock_httpx_response.headers = {"content-type": "application/json"}

        mock_logging_obj = self._create_mock_logging_obj()

        request_body = {
            "instances": [{"prompt": "A close up of two people staring at a cryptic drawing on a wall,"}],
            "parameters": {"durationSeconds": 8},
        }

        kwargs = {
            "passthrough_logging_payload": PassthroughStandardLoggingPayload(
                url="https://generativelanguage.googleapis.com/v1beta/models/veo-2.0-generate-001:predictLongRunning",
                request_body=request_body,
                request_method="POST",
            ),
        }

        result = GeminiPassthroughLoggingHandler.gemini_passthrough_handler(
            httpx_response=mock_httpx_response,
            response_body=mock_veo_response,
            logging_obj=mock_logging_obj,
            url_route="https://generativelanguage.googleapis.com/v1beta/models/veo-2.0-generate-001:predictLongRunning",
            result="",
            start_time=self.start_time,
            end_time=self.end_time,
            cache_hit=False,
            request_body=request_body,
            **kwargs,
        )

        assert result is not None
        assert result["kwargs"]["response_cost"] == expected_cost
        assert result["kwargs"]["model"] == "veo-2.0-generate-001"
        assert result["kwargs"]["custom_llm_provider"] == "gemini"

        mock_completion_cost.assert_called_once()
        call_args = mock_completion_cost.call_args
        assert call_args.kwargs.get("call_type") == "create_video"
        assert call_args.kwargs.get("custom_llm_provider") == "gemini"
        assert call_args.kwargs.get("model") == "veo-2.0-generate-001"

        video_response = result["result"]
        assert hasattr(video_response, "_hidden_params")
        assert video_response._hidden_params.get("response_cost") == expected_cost

        assert mock_logging_obj.model_call_details["response_cost"] == expected_cost
        assert mock_logging_obj.model_call_details["model"] == "veo-2.0-generate-001"
        assert mock_logging_obj.model_call_details["custom_llm_provider"] == "gemini"
