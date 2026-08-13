"""Focused mocked coverage for the Cohere Codex candidate adapter."""

from __future__ import annotations

import json
import importlib
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from fastapi.responses import Response
from fastapi.responses import StreamingResponse

from litellm.proxy.pass_through_endpoints.aawm_adapter_runtime import (
    codex_candidate_calls,
)
from litellm.proxy.pass_through_endpoints.aawm_alias_routing import (
    adapter_driver,
    candidate_loop,
)
from litellm.proxy.pass_through_endpoints.providers.cohere import runtime as cohere_runtime


COHERE_TARGET = "https://api.cohere.com/v2/chat"
COHERE_KEY = "cohere-test-key"


def _identity_adapter(body):
    return body, []


def _identity_drop(body):
    return body, []


def _response_from_payload(payload):
    return Response(
        content=json.dumps(payload),
        media_type="application/json",
    )


@pytest.fixture
def cohere_host(monkeypatch):
    requested_secrets = []
    transform = MagicMock()
    transform.transform_responses_api_request_to_chat_completion_request.return_value = {
        "model": "command-r-plus",
        "messages": [{"role": "user", "content": "translated"}],
    }
    transform.transform_chat_completion_response_to_responses_api_response.return_value = {
        "id": "resp_cohere_1",
        "object": "response",
        "status": "completed",
    }
    responses_transformation = importlib.import_module(
        "litellm.responses.litellm_completion_transformation.transformation"
    )
    monkeypatch.setattr(
        responses_transformation,
        "LiteLLMCompletionResponsesConfig",
        transform,
    )

    def get_secret(name):
        requested_secrets.append(name)
        if name == cohere_runtime.COHERE_CANONICAL_API_KEY_ENV_VAR:
            return COHERE_KEY
        return None

    monkeypatch.setattr(
        cohere_runtime,
        "_runtime_dependencies",
        cohere_runtime.CohereRuntimeDependencies(
            get_secret=get_secret,
            clean_secret_string=lambda value: value.strip() if value else None,
            log_debug=MagicMock(),
        ),
    )

    completion = AsyncMock(return_value=SimpleNamespace(id="chatcmpl_cohere_1"))
    egress = MagicMock()
    host = {
        "__builtins__": __builtins__,
        "httpx": httpx,
        "cast": cast,
        "ResponsesAPIOptionalRequestParams": dict,
        "_aawm_adapter_driver": adapter_driver,
        "Response": Response,
        "StreamingResponse": StreamingResponse,
        "litellm": SimpleNamespace(acompletion=completion),
        "_CODEX_AUTO_AGENT_COHERE_PROVIDER": "cohere",
        "_aawm_adapter_config": SimpleNamespace(
            CODEX_COHERE_CHAT_COMPLETIONS=SimpleNamespace(
                route_family="codex_cohere_chat_completions",
                tag_prefix="codex-cohere",
                target_endpoint_label="cohere-chat-v2",
                span_name="codex-cohere-chat-completions",
                credential_family="cohere",
                expected_target_family="cohere",
            )
        ),
        "_adapt_codex_custom_tools_to_functions_from_request_body": _identity_adapter,
        "_adapt_codex_namespace_tools_to_functions_from_request_body": _identity_adapter,
        "_apply_codex_tool_description_patches_to_request_body": _identity_adapter,
        "_drop_unsupported_codex_hosted_tools_from_request_body": _identity_drop,
        "_drop_unsupported_codex_input_items_from_request_body": _identity_drop,
        "_drop_tool_choice_without_tools_from_request_body": _identity_drop,
        "HttpPassThroughEndpointHelpers": SimpleNamespace(
            validate_outgoing_egress=egress
        ),
        "_annotate_request_scope_for_adapted_access_log": MagicMock(),
        "_get_proxy_shared_aiohttp_session": MagicMock(return_value="shared-session"),
        "_build_responses_response_from_adapter_response": MagicMock(
            side_effect=_response_from_payload
        ),
        "LiteLLMCompletionStreamingIterator": MagicMock(),
        "_responses_sse_from_iterator": MagicMock(),
    }
    codex_candidate_calls.install(host)
    host["LiteLLMCompletionStreamingIterator"] = MagicMock()
    host["_responses_sse_from_iterator"] = MagicMock()
    return SimpleNamespace(
        host=host,
        completion=completion,
        egress=egress,
        requested_secrets=requested_secrets,
        transform=transform,
    )


def _request_body(*, stream=False):
    return {
        "model": "cohere/command-r-plus",
        "input": [
            {"role": "user", "content": [{"type": "input_text", "text": "Find it"}]},
            {
                "type": "function_call_output",
                "call_id": "call_cohere_1",
                "output": '{"result":"found"}',
            },
        ],
        "tools": [
            {
                "type": "function",
                "name": "lookup",
                "description": "Look up a value",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                },
            }
        ],
        "stream": stream,
    }


async def _prepare(cohere, *, stream=False, body=None):
    return await cohere.host[
        "_prepare_codex_cohere_chat_completions_adapter_route"
    ](
        request=MagicMock(),
        prepared_request_body=body or _request_body(stream=stream),
        adapter_model=" cohere/command-r-plus ",
        use_alias_candidate_probe=False,
    )


async def _perform(cohere, plan, *, body=None):
    return await cohere.host[
        "_perform_codex_cohere_chat_completions_adapter_call"
    ](
        request=MagicMock(),
        prepared_request_body=body or _request_body(
            stream=plan.client_requested_stream
        ),
        config=plan.config,
        adapter_model="cohere/command-r-plus",
        target_url=plan.target_url,
        api_key=plan.api_key,
        api_base=plan.api_base,
        client_requested_stream=plan.client_requested_stream,
        **plan.perform_kwargs,
    )


@pytest.mark.asyncio
async def test_should_prepare_valid_cohere_candidate_and_map_upstream_model(cohere_host):
    plan = await _prepare(cohere_host)

    assert plan.prepared_request_body["model"] == "cohere/command-r-plus"
    assert plan.perform_kwargs["upstream_model"] == "command-r-plus"
    assert plan.perform_kwargs["completion_kwargs"]["model"] == "command-r-plus"
    assert plan.api_key == COHERE_KEY
    assert str(plan.target_url) == COHERE_TARGET


@pytest.mark.asyncio
async def test_should_send_responses_input_tools_and_tool_results_to_transformer(
    cohere_host,
):
    body = _request_body()
    await _prepare(cohere_host, body=body)

    transform_call = cohere_host.transform.transform_responses_api_request_to_chat_completion_request.call_args
    assert transform_call.kwargs["model"] == "command-r-plus"
    assert transform_call.kwargs["input"] == body["input"]
    assert transform_call.kwargs["custom_llm_provider"] == "cohere"
    assert transform_call.kwargs["responses_api_request"]["tools"] == body["tools"]
    assert transform_call.kwargs["responses_api_request"]["stream"] is False
    assert transform_call.kwargs["input"][1]["call_id"] == "call_cohere_1"


@pytest.mark.asyncio
async def test_should_validate_native_target_canonical_key_and_cohere_egress(
    cohere_host,
):
    plan = await _prepare(cohere_host)

    assert cohere_host.requested_secrets == [
        cohere_runtime.COHERE_API_BASE_ENV_VAR,
        cohere_runtime.COHERE_CANONICAL_API_KEY_ENV_VAR,
    ]
    assert str(plan.target_url) == COHERE_TARGET
    egress_kwargs = cohere_host.egress.call_args.kwargs
    egress_url = egress_kwargs.get("url", egress_kwargs.get("target_url"))
    assert str(egress_url) == COHERE_TARGET
    assert egress_kwargs["credential_family"] == "cohere"
    assert egress_kwargs["expected_target_family"] == "cohere"


@pytest.mark.asyncio
async def test_should_call_acompletion_with_cohere_credentials_and_empty_proxy_headers(
    cohere_host,
):
    plan = await _prepare(cohere_host)
    await _perform(cohere_host, plan)

    call = cohere_host.completion.await_args
    assert call.kwargs["custom_llm_provider"] == "cohere"
    assert call.kwargs["api_base"] == COHERE_TARGET
    assert call.kwargs["api_key"] == COHERE_KEY
    assert call.kwargs["proxy_server_request"]["headers"] == {}


@pytest.mark.asyncio
async def test_should_convert_nonstream_chat_response_through_responses_seam(cohere_host):
    plan = await _prepare(cohere_host)
    response = await _perform(cohere_host, plan)

    assert response.status_code == 200
    assert json.loads(response.body) == {
        "id": "resp_cohere_1",
        "object": "response",
        "status": "completed",
    }
    cohere_host.transform.transform_chat_completion_response_to_responses_api_response.assert_called_once_with(
        chat_completion_response=cohere_host.completion.return_value,
        request_input=plan.perform_kwargs["request_input"],
        responses_api_request=plan.perform_kwargs["responses_api_request"],
    )


@pytest.mark.asyncio
async def test_should_wire_stream_iterator_and_sse_without_losing_terminal_fields(
    cohere_host,
    monkeypatch,
):
    stream_response = {
        "function_call": {"call_id": "call_cohere_1", "name": "lookup"},
        "usage": {"input_tokens": 4, "output_tokens": 3},
        "finish_reason": "tool_calls",
        "terminal": True,
    }
    completion_stream = SimpleNamespace(chunks=[stream_response])
    iterator = object()
    sse_chunks = [
        b'data: {"type":"response.output_item.added","call_id":"call_cohere_1"}\n\n',
        b'data: {"type":"response.completed","usage":{"input_tokens":4,"output_tokens":3},"finish_reason":"tool_calls"}\n\n',
    ]
    cohere_host.completion.return_value = completion_stream
    cohere_host.host["LiteLLMCompletionStreamingIterator"].return_value = iterator
    cohere_host.host["_responses_sse_from_iterator"].return_value = sse_chunks
    streaming_iterator_module = importlib.import_module(
        "litellm.responses.litellm_completion_transformation.streaming_iterator"
    )
    monkeypatch.setattr(
        streaming_iterator_module,
        "LiteLLMCompletionStreamingIterator",
        cohere_host.host["LiteLLMCompletionStreamingIterator"],
    )
    perform_globals = cohere_host.host[
        "_perform_codex_cohere_chat_completions_adapter_call"
    ].__globals__
    perform_globals["LiteLLMCompletionStreamingIterator"] = cohere_host.host[
        "LiteLLMCompletionStreamingIterator"
    ]
    perform_globals["_responses_sse_from_iterator"] = cohere_host.host[
        "_responses_sse_from_iterator"
    ]

    body = _request_body(stream=True)
    plan = await _prepare(cohere_host, stream=True, body=body)
    response = await _perform(cohere_host, plan, body=body)

    iterator_call = cohere_host.host["LiteLLMCompletionStreamingIterator"].call_args.kwargs
    assert iterator_call["litellm_custom_stream_wrapper"] is completion_stream
    assert iterator_call["request_input"] == body["input"]
    assert iterator_call["responses_api_request"]["tools"] == body["tools"]
    assert iterator_call["responses_api_request"]["stream"] is True
    assert iterator_call["custom_llm_provider"] == "cohere"
    cohere_host.host["_responses_sse_from_iterator"].assert_called_once_with(iterator)
    assert isinstance(response, StreamingResponse)
    assert response.media_type == "text/event-stream"
    assert list(sse_chunks) == [
        b'data: {"type":"response.output_item.added","call_id":"call_cohere_1"}\n\n',
        b'data: {"type":"response.completed","usage":{"input_tokens":4,"output_tokens":3},"finish_reason":"tool_calls"}\n\n',
    ]


@pytest.mark.asyncio
async def test_should_raise_sanitized_missing_credential_before_provider_call(cohere_host):
    cohere_host.requested_secrets.clear()
    cohere_runtime._runtime_dependencies = cohere_runtime.CohereRuntimeDependencies(
        get_secret=lambda name: None,
        clean_secret_string=lambda value: value.strip() if value else None,
        log_debug=MagicMock(),
    )

    with pytest.raises(cohere_runtime.CohereMissingCredentialError) as exc_info:
        await _prepare(cohere_host)

    message = exc_info.value.message
    assert "COHERE_API_KEY" in message
    assert COHERE_KEY not in message
    assert "provider-secret-sentinel" not in message
    assert "key=" not in message.lower()
    assert "bearer" not in message.lower()
    assert cohere_host.completion.await_count == 0


@pytest.mark.parametrize(
    ("classification_name", "failure_class", "expected"),
    [
        ("cohere_timeout_connectivity", "transient", "upstream_timeout"),
        ("auth_failure", "auth", "provider_terminal_error"),
        ("quota_failure", "quota_exhausted", "usage_limit_reached"),
        ("rate_failure", "rate_limit", "rate_limited"),
        ("model_failure", "model_unavailable", "candidate_unavailable"),
        ("other_4xx", "provider_4xx_other", "provider_terminal_error"),
        ("provider_5xx", "provider_5xx", "provider_terminal_error"),
        ("transient_failure", "transient", "provider_terminal_error"),
    ],
)
def test_should_map_codex_cohere_candidate_failures(
    monkeypatch,
    classification_name,
    failure_class,
    expected,
):
    classifier = MagicMock(
        return_value=SimpleNamespace(
            name=classification_name,
            failure_class=failure_class,
            cooldown_scope="candidate",
            advance_fresh_candidate=True,
        )
    )
    monkeypatch.setattr(candidate_loop, "classify_cohere_failure", classifier)
    monkeypatch.setattr(
        candidate_loop._error_signals,
        "_extract_adapter_exception_status_code",
        lambda exc: 503,
    )

    result = candidate_loop._classify_codex_cohere_candidate_failure(
        RuntimeError("provider failure"),
        candidate={
            "provider": "cohere",
            "route_family": "codex_cohere_chat_completions_adapter",
        },
        is_codex_alias=True,
    )

    assert result == expected
    classifier.assert_called_once()


@pytest.mark.parametrize(
    ("candidate", "is_codex_alias"),
    [
        (None, True),
        ({"provider": "openai", "route_family": "codex_cohere_chat_completions_adapter"}, True),
        ({"provider": "cohere", "route_family": "other_route"}, True),
        ({"provider": "cohere", "route_family": "codex_cohere_chat_completions_adapter"}, False),
    ],
)
def test_should_strictly_gate_codex_cohere_candidate_failure_classification(
    monkeypatch,
    candidate,
    is_codex_alias,
):
    classifier = MagicMock()
    monkeypatch.setattr(candidate_loop, "classify_cohere_failure", classifier)

    assert (
        candidate_loop._classify_codex_cohere_candidate_failure(
            RuntimeError("provider failure"),
            candidate=candidate,
            is_codex_alias=is_codex_alias,
        )
        is None
    )
    classifier.assert_not_called()
