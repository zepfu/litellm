import asyncio
import json
from unittest.mock import MagicMock

import pytest
import pytest_asyncio
from starlette.responses import Response, StreamingResponse

import litellm
from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER
from litellm.llms.alibaba_token_plan.adapters.adapter import (
    ALIBABA_TOKEN_PLAN_CREDENTIAL_SENTINEL,
    normalize_alibaba_token_plan_adapter_model_name,
    prepare_anthropic_alibaba_token_plan_adapter_route,
)
from litellm.llms.alibaba_token_plan.chat.transformation import (
    ALIBABA_TOKEN_PLAN_API_BASE,
    ALIBABA_TOKEN_PLAN_CHAT_COMPLETIONS_URL,
)
from litellm.proxy.pass_through_endpoints.aawm_alias_routing import adapter_config
from litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints import (
    _handle_codex_alibaba_token_plan_adapter_route,
    _prepare_codex_alibaba_token_plan_adapter_route,
    _resolve_anthropic_alibaba_token_plan_adapter_model,
    _resolve_codex_alibaba_token_plan_adapter_model,
)


_ALLOWED_MODELS = {
    "alibaba_token_plan/qwen3.8-max-preview",
    "alibaba_token_plan/qwen3.7-plus",
    "alibaba_token_plan/qwen3.7-max",
    "alibaba_token_plan/qwen3.6-flash",
    "alibaba_token_plan/deepseek-v4-pro",
    "alibaba_token_plan/glm-5.2",
}


@pytest_asyncio.fixture(autouse=True)
async def _cleanup_alibaba_adapter_async_resources(monkeypatch):
    def _close_logging_coroutine(async_coroutine, metadata=None):
        _ = metadata
        close = getattr(async_coroutine, "close", None)
        if callable(close):
            close()

    monkeypatch.setattr(
        GLOBAL_LOGGING_WORKER,
        "ensure_initialized_and_enqueue",
        _close_logging_coroutine,
    )
    yield
    await GLOBAL_LOGGING_WORKER.stop()
    await litellm.close_litellm_async_clients()


def _request() -> MagicMock:
    request = MagicMock()
    request.headers = {}
    request.scope = {}
    return request


def _collaboration_namespace_tool() -> dict:
    return {
        "type": "namespace",
        "name": "collaboration",
        "description": "Spawn and manage sub-agents.",
        "tools": [
            {
                "type": "function",
                "name": "spawn_agent",
                "description": "Spawn a child agent.",
                "strict": False,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "task_name": {"type": "string"},
                        "message": {"type": "string"},
                    },
                    "required": ["task_name", "message"],
                    "additionalProperties": False,
                },
            },
            {
                "type": "function",
                "name": "wait_agent",
                "description": "Wait for child activity.",
                "strict": False,
                "parameters": {
                    "type": "object",
                    "properties": {"timeout_ms": {"type": "integer"}},
                    "additionalProperties": False,
                },
            },
        ],
    }


async def _stream_text(response: StreamingResponse) -> str:
    chunks: list[bytes] = []
    body_iterator = response.body_iterator
    try:
        async for chunk in body_iterator:
            chunks.append(chunk if isinstance(chunk, bytes) else chunk.encode())
    finally:
        aclose = getattr(body_iterator, "aclose", None)
        if callable(aclose):
            await aclose()
        for _ in range(16):
            await asyncio.sleep(0)
    return b"".join(chunks).decode()


@pytest.mark.parametrize(
    ("model", "expected"),
    (
        ("alibaba_token_plan/qwen3.8-max-preview", "alibaba_token_plan/qwen3.8-max-preview"),
        ("alibaba_token_plan/qwen3.7-plus", "alibaba_token_plan/qwen3.7-plus"),
        ("alibaba_token_plan/qwen3.7-max", "alibaba_token_plan/qwen3.7-max"),
        ("alibaba_token_plan/qwen3.6-flash", "alibaba_token_plan/qwen3.6-flash"),
        ("alibaba_token_plan/deepseek-v4-pro", "alibaba_token_plan/deepseek-v4-pro"),
        ("alibaba_token_plan/glm-5.2", "alibaba_token_plan/glm-5.2"),
        ("qwen3.8-max-preview", None),
        ("dashscope/qwen3.8-max-preview", None),
        ("alibaba_token_plan/unknown", None),
        ("aawm-sota-alibaba", None),
    ),
)
def test_should_normalize_only_exact_token_plan_adapter_models(
    model: str,
    expected: str | None,
) -> None:
    assert (
        normalize_alibaba_token_plan_adapter_model_name(
            model,
            allowed_models=_ALLOWED_MODELS,
        )
        == expected
    )


@pytest.mark.asyncio
async def test_should_prepare_codex_collaboration_with_raw_upstream_model() -> None:
    task_payload = "Run two parallel command batches and report the marker."
    plan = await _prepare_codex_alibaba_token_plan_adapter_route(
        request=_request(),
        adapter_model="alibaba_token_plan/qwen3.8-max-preview",
        prepared_request_body={
            "model": "alibaba_token_plan/qwen3.8-max-preview",
            "input": [
                {
                    "type": "agent_message",
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": (
                                "Message Type: NEW_TASK\n" "Task name: /root/child_a\n" "Sender: /root\n" "Payload:\n"
                            ),
                        },
                        {
                            "type": "encrypted_content",
                            "encrypted_content": task_payload,
                        },
                    ],
                }
            ],
            "tools": [_collaboration_namespace_tool()],
            "parallel_tool_calls": True,
            "stream": True,
        },
    )

    completion_kwargs = plan.perform_kwargs["completion_kwargs"]
    metadata = plan.prepared_request_body["litellm_metadata"]
    assert plan.config is adapter_config.CODEX_ALIBABA_TOKEN_PLAN
    assert plan.api_key == ALIBABA_TOKEN_PLAN_CREDENTIAL_SENTINEL
    assert plan.api_base == ALIBABA_TOKEN_PLAN_API_BASE
    assert plan.target_url == ALIBABA_TOKEN_PLAN_CHAT_COMPLETIONS_URL
    assert completion_kwargs["model"] == "qwen3.8-max-preview"
    assert completion_kwargs["custom_llm_provider"] == "alibaba_token_plan"
    assert completion_kwargs["num_retries"] == 0
    assert completion_kwargs["parallel_tool_calls"] is True
    assert [tool["function"]["name"] for tool in completion_kwargs["tools"]] == [
        "spawn_agent",
        "wait_agent",
    ]
    assert task_payload in str(completion_kwargs["messages"])
    assert metadata["alibaba_token_plan_codex_agent_task_payload_restored"] is True
    assert metadata["alibaba_token_plan_upstream_model"] == "qwen3.8-max-preview"
    assert "aawm-sota-alibaba" not in json.dumps(completion_kwargs)


@pytest.mark.asyncio
async def test_should_prepare_anthropic_with_raw_upstream_model() -> None:
    plan = await prepare_anthropic_alibaba_token_plan_adapter_route(
        request=_request(),
        adapter_model="alibaba_token_plan/qwen3.7-max",
        prepared_request_body={
            "model": "alibaba_token_plan/qwen3.7-max",
            "max_tokens": 64,
            "messages": [{"role": "user", "content": "hello"}],
            "tools": [{"name": "Bash", "input_schema": {"type": "object"}}],
            "parallel_tool_calls": True,
            "stream": False,
        },
    )

    assert plan.config is adapter_config.ANTHROPIC_ALIBABA_TOKEN_PLAN
    assert plan.api_key == ALIBABA_TOKEN_PLAN_CREDENTIAL_SENTINEL
    assert plan.api_base == ALIBABA_TOKEN_PLAN_API_BASE
    assert plan.perform_kwargs["custom_llm_provider"] == "alibaba_token_plan"
    assert plan.perform_kwargs["model_for_upstream"] == "qwen3.7-max"
    assert plan.perform_kwargs["extra_handler_kwargs"]["parallel_tool_calls"] is True
    assert plan.prepared_request_body["messages"] == [{"role": "user", "content": "hello"}]
    assert plan.prepared_request_body["litellm_metadata"]["alibaba_token_plan_upstream_model"] == "qwen3.7-max"
    assert (
        "route:anthropic_alibaba_token_plan_chat_completions_adapter"
        in plan.prepared_request_body["litellm_metadata"]["tags"]
    )
    assert "aawm-sota-alibaba" not in json.dumps(plan.perform_kwargs)


def test_should_require_prefixed_direct_adapter_models() -> None:
    assert (
        _resolve_codex_alibaba_token_plan_adapter_model(
            {"model": "alibaba_token_plan/qwen3.8-max-preview"},
            "/v1/responses",
        )
        == "alibaba_token_plan/qwen3.8-max-preview"
    )
    assert (
        _resolve_anthropic_alibaba_token_plan_adapter_model(
            {"model": "alibaba_token_plan/qwen3.7-max"},
            "/v1/messages",
        )
        == "alibaba_token_plan/qwen3.7-max"
    )
    assert (
        _resolve_codex_alibaba_token_plan_adapter_model(
            {"model": "qwen3.8-max-preview"},
            "/v1/responses",
        )
        is None
    )
    assert (
        _resolve_anthropic_alibaba_token_plan_adapter_model(
            {"model": "aawm-sota-alibaba"},
            "/v1/messages",
        )
        is None
    )


@pytest.mark.asyncio
async def test_should_run_codex_nonstream_without_sending_the_alias(
    monkeypatch: pytest.MonkeyPatch,
    respx_mock,
) -> None:
    monkeypatch.setenv("ALIBABA_KEY", "existing-token-plan-key")
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    spawn_arguments = {
        "task_name": "child_a",
        "message": "Inspect the release documentation.",
    }
    respx_mock.post(ALIBABA_TOKEN_PLAN_CHAT_COMPLETIONS_URL).respond(
        json={
            "id": "chatcmpl-alibaba",
            "object": "chat.completion",
            "created": 1,
            "model": "qwen3.8-max-preview",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_spawn",
                                "type": "function",
                                "function": {
                                    "name": "spawn_agent",
                                    "arguments": json.dumps(spawn_arguments),
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {
                "prompt_tokens": 12,
                "completion_tokens": 7,
                "total_tokens": 19,
            },
        }
    )

    response = await _handle_codex_alibaba_token_plan_adapter_route(
        endpoint="/v1/responses",
        request=_request(),
        fastapi_response=MagicMock(spec=Response),
        user_api_key_dict=MagicMock(),
        prepared_request_body={
            "model": "alibaba_token_plan/qwen3.8-max-preview",
            "input": "dispatch the child",
            "tools": [_collaboration_namespace_tool()],
            "stream": False,
            "litellm_metadata": {
                "model_alias": "aawm-sota-alibaba",
                "inbound_model_alias": "aawm-sota-alibaba",
            },
        },
        adapter_model="alibaba_token_plan/qwen3.8-max-preview",
    )

    upstream_request = respx_mock.calls[0].request
    upstream_body = json.loads(upstream_request.content)
    assert upstream_request.headers["Authorization"] == ("Bearer existing-token-plan-key")
    assert upstream_body["model"] == "qwen3.8-max-preview"
    assert [tool["function"]["name"] for tool in upstream_body["tools"]] == [
        "spawn_agent",
        "wait_agent",
    ]
    assert "aawm-sota-alibaba" not in upstream_request.content.decode()

    response_body = json.loads(response.body)
    function_call = next(item for item in response_body["output"] if item["type"] == "function_call")
    assert function_call["name"] == "spawn_agent"
    assert function_call["namespace"] == "collaboration"
    assert json.loads(function_call["arguments"]) == spawn_arguments


@pytest.mark.asyncio
async def test_should_restore_collaboration_namespace_in_codex_stream(
    monkeypatch: pytest.MonkeyPatch,
    respx_mock,
) -> None:
    monkeypatch.setenv("ALIBABA_KEY", "existing-token-plan-key")
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    stream_body = "\n\n".join(
        [
            'data: {"id":"chatcmpl-alibaba","object":"chat.completion.chunk","created":1,"model":"qwen3.8-max-preview","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_spawn","type":"function","function":{"name":"spawn_agent","arguments":"{\\"task_name\\":\\"child_a\\",\\"message\\":\\"inspect\\"}"}}]},"finish_reason":null}]}',
            'data: {"id":"chatcmpl-alibaba","object":"chat.completion.chunk","created":1,"model":"qwen3.8-max-preview","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls","usage":{"prompt_tokens":5,"completion_tokens":4,"total_tokens":9}}]}',
            "data: [DONE]",
            "",
        ]
    )
    respx_mock.post(ALIBABA_TOKEN_PLAN_CHAT_COMPLETIONS_URL).respond(
        content=stream_body.encode(),
        headers={"content-type": "text/event-stream"},
    )

    response = await _handle_codex_alibaba_token_plan_adapter_route(
        endpoint="/v1/responses",
        request=_request(),
        fastapi_response=MagicMock(spec=Response),
        user_api_key_dict=MagicMock(),
        prepared_request_body={
            "model": "alibaba_token_plan/qwen3.8-max-preview",
            "input": "dispatch the child",
            "tools": [_collaboration_namespace_tool()],
            "stream": True,
        },
        adapter_model="alibaba_token_plan/qwen3.8-max-preview",
    )

    assert isinstance(response, StreamingResponse)
    event_text = await _stream_text(response)
    assert '"spawn_agent"' in event_text
    assert '"collaboration"' in event_text
    assert "response.function_call_arguments.done" in event_text
    upstream_body = json.loads(respx_mock.calls[0].request.content)
    assert upstream_body["model"] == "qwen3.8-max-preview"
    assert "alibaba_token_plan/" not in respx_mock.calls[0].request.content.decode()
