import json
from datetime import datetime
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import httpx

from litellm.integrations.aawm_agent_identity import _build_session_history_record
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.proxy.pass_through_endpoints.llm_provider_handlers.gemini_passthrough_logging_handler import (
    GeminiPassthroughLoggingHandler,
)
from litellm.proxy.pass_through_endpoints.success_handler import (
    PassThroughEndpointLogging,
)
from litellm.types.passthrough_endpoints.pass_through_endpoints import EndpointType
from litellm.types.passthrough_endpoints.pass_through_endpoints import (
    PassthroughStandardLoggingPayload,
)


START_TIME = datetime(2026, 1, 1, 12, 0, 0)
END_TIME = datetime(2026, 1, 1, 12, 0, 1)
AAWM_METADATA = {
    "session_id": "session-123",
    "aawm_session_id": "session-123",
    "aawm_agent_id": "agent-456",
}
REQUEST_HEADERS = {
    "x-claude-code-session-id": "session-123",
    "x-aawm-agent-id": "agent-456",
}


def _mock_logging_obj() -> LiteLLMLoggingObj:
    logging_obj = MagicMock(spec=LiteLLMLoggingObj)
    logging_obj.litellm_call_id = "aawm-contract-call-id"
    logging_obj.litellm_trace_id = "aawm-contract-trace-id"
    logging_obj.cost_breakdown = None
    logging_obj.model_call_details = {}
    logging_obj.optional_params = {}
    return logging_obj


def _mock_httpx_response(response_body: Dict[str, Any], url: str) -> httpx.Response:
    return httpx.Response(
        status_code=200,
        headers={"content-type": "application/json"},
        content=json.dumps(response_body).encode("utf-8"),
        request=httpx.Request("POST", url),
    )


def _passthrough_payload(
    url: str,
    request_body: Dict[str, Any],
) -> PassthroughStandardLoggingPayload:
    return PassthroughStandardLoggingPayload(
        url=url,
        request_method="POST",
        request_body=request_body,
        request_headers=REQUEST_HEADERS,
    )


def _assert_aawm_contract(
    *,
    normalized: Dict[str, Any],
    logging_obj: LiteLLMLoggingObj,
    expected_model: str,
    expected_provider: str,
    expected_cost: float,
) -> None:
    assert normalized["result"] is not None
    assert getattr(normalized["result"], "model", None) == expected_model

    kwargs = normalized["kwargs"]
    assert kwargs["model"] == expected_model
    assert kwargs["custom_llm_provider"] == expected_provider
    assert kwargs["response_cost"] == expected_cost
    assert kwargs["passthrough_logging_payload"]["request_headers"] == REQUEST_HEADERS
    assert logging_obj.model_call_details["model"] == expected_model
    assert logging_obj.model_call_details["custom_llm_provider"] == expected_provider
    assert logging_obj.model_call_details["response_cost"] == expected_cost

    litellm_params = kwargs["litellm_params"]
    for key, value in AAWM_METADATA.items():
        assert litellm_params["metadata"][key] == value
    assert litellm_params["proxy_server_request"]["headers"] == REQUEST_HEADERS


def _normalize_passthrough_payload(
    *,
    url: str,
    request_body: Dict[str, Any],
    response_body: Dict[str, Any],
    custom_llm_provider: str | None = None,
) -> Dict[str, Any]:
    logging_obj = _mock_logging_obj()
    normalized = PassThroughEndpointLogging().normalize_llm_passthrough_logging_payload(
        httpx_response=_mock_httpx_response(response_body, url),
        response_body=response_body,
        request_body=request_body,
        logging_obj=logging_obj,
        url_route=url,
        result="",
        start_time=START_TIME,
        end_time=END_TIME,
        cache_hit=False,
        custom_llm_provider=custom_llm_provider,
        passthrough_logging_payload=_passthrough_payload(url, request_body),
        litellm_params={"metadata": AAWM_METADATA.copy()},
    )
    return {
        "result": normalized["standard_logging_response_object"],
        "kwargs": normalized["kwargs"],
        "logging_obj": logging_obj,
    }


def test_gemini_non_streaming_preserves_request_headers_for_aawm_metadata() -> None:
    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"
    request_body = {"contents": [{"parts": [{"text": "hello"}]}]}
    response_body = {
        "candidates": [
            {
                "content": {
                    "parts": [{"text": "hello from gemini"}],
                    "role": "model",
                },
                "finishReason": "STOP",
                "index": 0,
            }
        ],
        "usageMetadata": {
            "promptTokenCount": 5,
            "candidatesTokenCount": 4,
            "totalTokenCount": 9,
        },
    }

    with patch("litellm.completion_cost", return_value=0.000009):
        normalized = _normalize_passthrough_payload(
            url=url,
            request_body=request_body,
            response_body=response_body,
            custom_llm_provider="gemini",
        )

    assert (
        normalized["kwargs"]["litellm_params"]["proxy_server_request"]["headers"]
        == REQUEST_HEADERS
    )


def test_gemini_normalized_result_kwargs_include_aawm_logging_contract() -> None:
    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"
    request_body = {"contents": [{"parts": [{"text": "hello"}]}]}
    response_body = {
        "candidates": [
            {
                "content": {
                    "parts": [{"text": "hello from gemini"}],
                    "role": "model",
                },
                "finishReason": "STOP",
                "index": 0,
            }
        ],
        "usageMetadata": {
            "promptTokenCount": 5,
            "candidatesTokenCount": 4,
            "totalTokenCount": 9,
        },
    }

    with patch("litellm.completion_cost", return_value=0.000009):
        normalized = _normalize_passthrough_payload(
            url=url,
            request_body=request_body,
            response_body=response_body,
            custom_llm_provider="gemini",
        )

    _assert_aawm_contract(
        normalized=normalized,
        logging_obj=normalized["logging_obj"],
        expected_model="gemini-1.5-flash",
        expected_provider="gemini",
        expected_cost=0.000009,
    )


def test_openai_normalized_result_kwargs_include_aawm_logging_contract() -> None:
    url = "https://api.openai.com/v1/chat/completions"
    request_body = {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "hello"}],
    }
    response_body = {
        "id": "chatcmpl-aawm-contract",
        "object": "chat.completion",
        "created": 1775863780,
        "model": "gpt-4o",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "hello from openai"},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 5,
            "completion_tokens": 4,
            "total_tokens": 9,
        },
    }

    with patch("litellm.completion_cost", return_value=0.000009):
        normalized = _normalize_passthrough_payload(
            url=url,
            request_body=request_body,
            response_body=response_body,
        )

    _assert_aawm_contract(
        normalized=normalized,
        logging_obj=normalized["logging_obj"],
        expected_model="gpt-4o",
        expected_provider="openai",
        expected_cost=0.000009,
    )
    assert normalized["kwargs"]["standard_logging_object"]["response"]["id"] == (
        "chatcmpl-aawm-contract"
    )


def test_openai_responses_normalized_payload_preserves_standard_logging_object() -> None:
    url = "https://api.openai.com/v1/responses"
    request_body = {"model": "gpt-5.4", "input": "hello"}
    response_body = {
        "id": "resp_aawm_contract",
        "object": "response",
        "created_at": 1775863780,
        "model": "gpt-5.4",
        "output": [
            {
                "type": "message",
                "id": "msg_aawm_contract",
                "status": "completed",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": "hello from responses",
                        "annotations": [],
                    }
                ],
            }
        ],
        "usage": {
            "input_tokens": 7,
            "output_tokens": 5,
            "total_tokens": 12,
            "input_tokens_details": {"cached_tokens": 3},
        },
    }

    with patch("litellm.completion_cost", return_value=0.000012):
        normalized = _normalize_passthrough_payload(
            url=url,
            request_body=request_body,
            response_body=response_body,
        )

    _assert_aawm_contract(
        normalized=normalized,
        logging_obj=normalized["logging_obj"],
        expected_model="gpt-5.4",
        expected_provider="openai",
        expected_cost=0.000012,
    )
    standard_logging_object = normalized["kwargs"]["standard_logging_object"]
    assert standard_logging_object["response"]["id"] == "resp_aawm_contract"
    assert (
        standard_logging_object["response"]["usage"]["prompt_tokens_details"][
            "cached_tokens"
        ]
        == 3
    )


def test_anthropic_normalized_result_kwargs_include_aawm_logging_contract() -> None:
    url = "https://api.anthropic.com/v1/messages"
    request_body = {
        "model": "claude-3-5-sonnet-20240620",
        "max_tokens": 16,
        "messages": [{"role": "user", "content": "hello"}],
    }
    response_body = {
        "id": "msg_aawm_contract",
        "type": "message",
        "role": "assistant",
        "model": "claude-3-5-sonnet-20240620",
        "content": [{"type": "text", "text": "hello from anthropic"}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": 5, "output_tokens": 4},
    }

    with patch("litellm.completion_cost", return_value=0.000009):
        normalized = _normalize_passthrough_payload(
            url=url,
            request_body=request_body,
            response_body=response_body,
        )

    _assert_aawm_contract(
        normalized=normalized,
        logging_obj=normalized["logging_obj"],
        expected_model="claude-3-5-sonnet-20240620",
        expected_provider="anthropic",
        expected_cost=0.000009,
    )


def test_gemini_streaming_preserves_headers_and_usage_for_aawm_session_history() -> None:
    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:streamGenerateContent"
    request_body = {"contents": [{"parts": [{"text": "hello"}]}]}
    chunks = [
        'data: {"candidates":[{"content":{"parts":[{"text":"hello"}],"role":"model"},"finishReason":"STOP","index":0}],"usageMetadata":{"promptTokenCount":5,"candidatesTokenCount":4,"totalTokenCount":9,"cachedContentTokenCount":2,"cacheWriteInputTokens":1}}\n\n'
    ]
    logging_obj = _mock_logging_obj()

    with patch("litellm.completion_cost", return_value=0.000009):
        normalized = GeminiPassthroughLoggingHandler._handle_logging_gemini_collected_chunks(
            litellm_logging_obj=logging_obj,
            passthrough_success_handler_obj=PassThroughEndpointLogging(),
            url_route=url,
            request_body=request_body,
            endpoint_type=EndpointType.GENERIC,
            start_time=START_TIME,
            all_chunks=chunks,
            model="gemini-1.5-flash",
            end_time=END_TIME,
            kwargs={
                "passthrough_logging_payload": _passthrough_payload(url, request_body),
                "litellm_params": {"metadata": AAWM_METADATA.copy()},
            },
        )

    _assert_aawm_contract(
        normalized=normalized,
        logging_obj=logging_obj,
        expected_model="gemini-1.5-flash",
        expected_provider="gemini",
        expected_cost=0.000009,
    )
    record = _build_session_history_record(
        kwargs=normalized["kwargs"],
        result=normalized["result"],
        start_time=START_TIME.isoformat(),
        end_time=END_TIME.isoformat(),
    )
    assert record is not None
    assert record["cache_read_input_tokens"] == 2
    assert record["cache_creation_input_tokens"] == 1
