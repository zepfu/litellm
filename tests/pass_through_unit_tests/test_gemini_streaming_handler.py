import os
import sys
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, os.path.abspath("../.."))

from litellm.proxy.pass_through_endpoints.streaming_handler import (
    PassThroughStreamingHandler,
)
from litellm.types.passthrough_endpoints.pass_through_endpoints import EndpointType


@pytest.mark.asyncio
@patch(
    "litellm.proxy.pass_through_endpoints.streaming_handler.GeminiPassthroughLoggingHandler._handle_logging_gemini_collected_chunks"
)
@patch(
    "litellm.proxy.pass_through_endpoints.streaming_handler.VertexPassthroughLoggingHandler._handle_logging_vertex_collected_chunks"
)
async def test_route_streaming_logging_to_handler_uses_gemini_handler_for_gemini_provider(
    mock_vertex_handler,
    mock_gemini_handler,
):
    mock_gemini_handler.return_value = {
        "result": {"response": "gemini"},
        "kwargs": {"custom_llm_provider": "gemini", "model": "gemini-3-flash-preview"},
    }

    litellm_logging_obj = MagicMock()
    litellm_logging_obj.async_success_handler = AsyncMock()
    litellm_logging_obj._should_run_sync_callbacks_for_async_calls.return_value = False

    await PassThroughStreamingHandler._route_streaming_logging_to_handler(
        litellm_logging_obj=litellm_logging_obj,
        passthrough_success_handler_obj=MagicMock(),
        url_route="https://cloudcode-pa.googleapis.com/v1internal:generateContent",
        request_body={"model": "gemini-3-flash-preview"},
        endpoint_type=EndpointType.VERTEX_AI,
        start_time=datetime.now(),
        raw_bytes=[b'{"response":{"candidates":[{"content":{"parts":[{"text":"gemini routed"}]}}]}}'],
        end_time=datetime.now(),
        custom_llm_provider="gemini",
        success_handler_kwargs={
            "litellm_params": {
                "proxy_server_request": {
                    "headers": {
                        "langfuse_trace_name": "gemini",
                        "langfuse_trace_user_id": "litellm",
                    }
                }
            },
            "call_type": "pass_through_endpoint",
            "litellm_call_id": "call-123",
        },
    )

    mock_gemini_handler.assert_called_once()
    mock_vertex_handler.assert_not_called()
    litellm_logging_obj.async_success_handler.assert_awaited_once()
    async_call_kwargs = litellm_logging_obj.async_success_handler.await_args.kwargs
    assert (
        async_call_kwargs["litellm_params"]["proxy_server_request"]["headers"][
            "langfuse_trace_name"
        ]
        == "gemini"
    )
    assert async_call_kwargs["call_type"] == "pass_through_endpoint"
    assert async_call_kwargs["litellm_call_id"] == "call-123"
    assert async_call_kwargs["custom_llm_provider"] == "gemini"
