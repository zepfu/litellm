import asyncio
import json
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
import pytest_asyncio
from fastapi import HTTPException
from starlette.responses import Response, StreamingResponse

import litellm

from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER
from litellm.llms.kimi_code.adapters.adapter import (
    KIMI_CODE_MANAGED_CREDENTIAL_SENTINEL,
    normalize_kimi_code_chat_completions_adapter_model_name,
    prepare_anthropic_kimi_chat_completions_adapter_route,
    prepare_codex_kimi_chat_completions_adapter_route,
)
from litellm.llms.kimi_code.chat.transformation import (
    KIMI_CODE_API_BASE,
    KIMI_CODE_CHAT_COMPLETIONS_URL,
    KIMI_CODE_CREDENTIAL_PATH_ENV,
)
from litellm.proxy import proxy_server
from litellm.proxy.pass_through_endpoints.aawm_alias_routing import adapter_config
from litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints import (
    _handle_anthropic_kimi_chat_completions_adapter_route,
    _handle_codex_kimi_chat_completions_adapter_route,
    _prepare_codex_kimi_chat_completions_adapter_route,
    _resolve_anthropic_kimi_chat_completions_adapter_model,
    _resolve_codex_kimi_chat_completions_adapter_model,
)
from litellm.responses.litellm_completion_transformation.transformation import (
    LiteLLMCompletionResponsesConfig,
)
from litellm.utils import _invalidate_model_cost_lowercase_map, get_model_info


_ALLOWED_MODELS = {
    "kimi_code/k3",
    "kimi_code/k3-low",
    "kimi_code/k3-high",
    "kimi_code/k3-max",
    "kimi_code/kimi-for-coding",
    "kimi_code/kimi-for-coding-highspeed",
}


@pytest.fixture(autouse=True)
def _use_current_kimi_code_model_metadata(monkeypatch):
    model_cost = dict(litellm.model_cost)
    model_cost.update(
        {
            "kimi_code/k3": {
                "default_reasoning_effort": "high",
                "litellm_provider": "kimi_code",
                "max_input_tokens": 1048576,
                "mode": "chat",
                "supports_high_reasoning_effort": True,
                "supports_low_reasoning_effort": True,
                "supports_max_reasoning_effort": True,
                "supports_reasoning": True,
                "supports_vision": True,
            },
            "kimi_code/kimi-for-coding": {
                "litellm_provider": "kimi_code",
                "max_input_tokens": 262144,
                "mode": "chat",
                "supports_reasoning": True,
                "supports_vision": True,
            },
            "kimi_code/kimi-for-coding-highspeed": {
                "litellm_provider": "kimi_code",
                "max_input_tokens": 262144,
                "mode": "chat",
                "supports_reasoning": True,
                "supports_vision": True,
            },
        }
    )
    monkeypatch.setattr(litellm, "model_cost", model_cost)
    _invalidate_model_cost_lowercase_map()
    get_model_info.cache_clear()
    yield
    _invalidate_model_cost_lowercase_map()
    get_model_info.cache_clear()


@pytest_asyncio.fixture(autouse=True)
async def _cleanup_kimi_adapter_async_resources(monkeypatch):
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


def _write_credentials(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "access_token": "current-access-token",
                "expires_at": time.time() + 300,
            }
        ),
        encoding="utf-8",
    )


def _request() -> MagicMock:
    request = MagicMock()
    request.headers = {}
    request.scope = {}
    return request


async def _stream_text(response: StreamingResponse) -> str:
    chunks = []
    body_iterator = response.body_iterator
    try:
        async for chunk in body_iterator:
            chunks.append(chunk if isinstance(chunk, bytes) else chunk.encode())
    finally:
        aclose = getattr(body_iterator, "aclose", None)
        if callable(aclose):
            await aclose()
        # OpenAI's AsyncStream closes nested SSE/httpx async generators on
        # successive loop turns after the outer response iterator is closed.
        for _ in range(16):
            await asyncio.sleep(0)
    return b"".join(chunks).decode()


@pytest.mark.parametrize(
    ("model", "expected"),
    (
        ("kimi_code/k3", "kimi_code/k3"),
        ("kimi_code/k3-low", "kimi_code/k3-low"),
        ("kimi_code/k3-high", "kimi_code/k3-high"),
        ("kimi_code/k3-max", "kimi_code/k3-max"),
        ("kimi_code/kimi-for-coding", "kimi_code/kimi-for-coding"),
        ("k3", None),
        ("moonshot/k3", None),
        ("kimi_code/k3-preview", None),
    ),
)
def test_should_normalize_only_managed_kimi_direct_models(model, expected):
    assert normalize_kimi_code_chat_completions_adapter_model_name(model, allowed_models=_ALLOWED_MODELS) == expected


@pytest.mark.asyncio
async def test_should_translate_responses_tools_continuation_and_k3_effort():
    plan = await prepare_codex_kimi_chat_completions_adapter_route(
        request=MagicMock(),
        adapter_model="kimi_code/k3",
        prepared_request_body={
            "model": "kimi_code/k3",
            "input": [
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": "inspect"}],
                },
                {
                    "type": "function_call_output",
                    "call_id": "call_read",
                    "output": "result",
                },
            ],
            "tools": [
                {
                    "type": "function",
                    "name": "read_file",
                    "parameters": {
                        "type": "object",
                        "properties": {"path": {"type": "string"}},
                    },
                }
            ],
            "parallel_tool_calls": True,
            "reasoning": {"effort": "medium"},
            "stream": True,
            "litellm_metadata": {"trace_id": "trace-kimi"},
        },
    )

    completion_kwargs = plan.perform_kwargs["completion_kwargs"]
    assert plan.config is adapter_config.CODEX_KIMI_CHAT_COMPLETIONS
    assert plan.target_url == KIMI_CODE_CHAT_COMPLETIONS_URL
    assert plan.api_base == KIMI_CODE_API_BASE
    assert plan.api_key == KIMI_CODE_MANAGED_CREDENTIAL_SENTINEL
    assert plan.client_requested_stream is True
    assert completion_kwargs["model"] == "k3"
    assert completion_kwargs["custom_llm_provider"] == "kimi_code"
    assert completion_kwargs["num_retries"] == 0
    assert completion_kwargs["reasoning_effort"] == "high"
    assert completion_kwargs["parallel_tool_calls"] is True
    assert completion_kwargs["tools"][0]["function"]["name"] == "read_file"
    assert completion_kwargs["metadata"]["trace_id"] == "trace-kimi"
    assert completion_kwargs["messages"][-1]["role"] == "tool"
    assert completion_kwargs["messages"][-1]["tool_call_id"] == "call_read"
    metadata = plan.prepared_request_body["litellm_metadata"]
    assert metadata["passthrough_route_family"] == ("codex_kimi_chat_completions_adapter")
    assert metadata["kimi_code_adapter_key"] == "kimi_code/k3"
    assert metadata["kimi_code_upstream_model"] == "k3"
    assert adapter_config.CODEX_KIMI_CHAT_COMPLETIONS.tag_prefix in metadata["tags"]
    assert metadata["langfuse_spans"][-1]["name"] == (adapter_config.CODEX_KIMI_CHAT_COMPLETIONS.span_name)


@pytest.mark.asyncio
async def test_should_use_responses_session_history_for_continuation(monkeypatch):
    restored_assistant_message = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "exec_command:0",
                "type": "function",
                "function": {
                    "name": "exec_command",
                    "arguments": '{"cmd":"pwd"}',
                },
            }
        ],
    }
    tool_result_message = {
        "role": "tool",
        "tool_call_id": "exec_command:0",
        "content": "/workspace",
    }
    empty_assistant_message = {"role": "assistant", "content": []}
    later_user_message = {"role": "user", "content": "continue"}
    session_handler = AsyncMock(
        return_value={
            "model": "k3",
            "messages": [
                restored_assistant_message,
                later_user_message,
                tool_result_message,
                empty_assistant_message,
            ],
            "custom_llm_provider": "kimi_code",
            "metadata": {"trace_id": "trace-kimi-continuation"},
            "litellm_trace_id": "session-trace-kimi",
            "stream": True,
        }
    )
    monkeypatch.setattr(
        LiteLLMCompletionResponsesConfig,
        "async_responses_api_session_handler",
        session_handler,
    )

    plan = await prepare_codex_kimi_chat_completions_adapter_route(
        request=MagicMock(),
        adapter_model="kimi_code/k3",
        prepared_request_body={
            "model": "kimi_code/k3",
            "input": "continue",
            "previous_response_id": "resp_prior",
            "stream": True,
            "litellm_metadata": {"trace_id": "trace-kimi-continuation"},
        },
    )

    session_handler.assert_awaited_once()
    handler_kwargs = session_handler.await_args.kwargs
    assert handler_kwargs["previous_response_id"] == "resp_prior"
    assert handler_kwargs["litellm_completion_request"]["stream"] is True
    assert handler_kwargs["litellm_completion_request"]["metadata"]["trace_id"] == "trace-kimi-continuation"
    completion_kwargs = plan.perform_kwargs["completion_kwargs"]
    assert completion_kwargs["messages"] == [
        {
            "role": "assistant",
            "tool_calls": restored_assistant_message["tool_calls"],
        },
        tool_result_message,
        later_user_message,
    ]
    assert completion_kwargs["messages"][1] is tool_result_message
    assert "content" not in completion_kwargs["messages"][0]
    assert completion_kwargs["messages"][0]["tool_calls"][0]["function"]["arguments"] == '{"cmd":"pwd"}'
    assert sum(message["role"] == "assistant" for message in completion_kwargs["messages"]) == 1
    assert completion_kwargs["metadata"]["trace_id"] == "trace-kimi-continuation"
    assert completion_kwargs["metadata"]["kimi_code_chat_message_shape_sanitized"] is True
    assert completion_kwargs["metadata"]["kimi_code_chat_message_shape_removed_empty_message_count"] == 1
    assert completion_kwargs["metadata"]["kimi_code_chat_message_shape_stripped_tool_call_content_count"] == 1
    assert completion_kwargs["litellm_trace_id"] == "session-trace-kimi"
    assert plan.client_requested_stream is True


@pytest.mark.asyncio
async def test_should_preserve_custom_tool_outputs_before_kimi_filtering(
    monkeypatch,
):
    assistant_message = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "exec_command:0",
                "type": "function",
                "function": {
                    "name": "exec_command",
                    "arguments": '{"cmd":"pwd"}',
                },
            },
            {
                "id": "exec_command:1",
                "type": "function",
                "function": {
                    "name": "exec_command",
                    "arguments": '{"cmd":"git status --short"}',
                },
            },
        ],
    }

    async def session_handler(
        *,
        previous_response_id,
        litellm_completion_request,
    ):
        assert previous_response_id == "resp_live_kimi_continuation"
        assert litellm_completion_request["messages"] == [
            {
                "role": "tool",
                "tool_call_id": "exec_command:1",
                "content": "clean",
            },
            {
                "role": "tool",
                "tool_call_id": "exec_command:0",
                "content": "/workspace",
            },
        ]
        return {
            **litellm_completion_request,
            "messages": [
                assistant_message,
                {"role": "user", "content": "continue"},
                *litellm_completion_request["messages"],
            ],
            "litellm_trace_id": "trace-live-kimi-continuation",
        }

    monkeypatch.setattr(
        LiteLLMCompletionResponsesConfig,
        "async_responses_api_session_handler",
        AsyncMock(side_effect=session_handler),
    )

    plan = await _prepare_codex_kimi_chat_completions_adapter_route(
        request=MagicMock(),
        adapter_model="kimi_code/k3-high",
        prepared_request_body={
            "model": "kimi_code/k3-high",
            "input": [
                {
                    "type": "custom_tool_call_output",
                    "call_id": "exec_command:1",
                    "output": "clean",
                },
                {
                    "type": "custom_tool_call_output",
                    "call_id": "exec_command:0",
                    "output": "/workspace",
                },
            ],
            "previous_response_id": "resp_live_kimi_continuation",
            "stream": True,
        },
    )

    completion_kwargs = plan.perform_kwargs["completion_kwargs"]
    assert completion_kwargs["messages"] == [
        {
            "role": "assistant",
            "tool_calls": assistant_message["tool_calls"],
        },
        {
            "role": "tool",
            "tool_call_id": "exec_command:0",
            "content": "/workspace",
        },
        {
            "role": "tool",
            "tool_call_id": "exec_command:1",
            "content": "clean",
        },
        {"role": "user", "content": "continue"},
    ]
    assert completion_kwargs["litellm_trace_id"] == ("trace-live-kimi-continuation")
    assert completion_kwargs["num_retries"] == 0
    assert all(
        item.get("type") != "custom_tool_call_output"
        for item in plan.prepared_request_body["input"]
        if isinstance(item, dict)
    )


@pytest.mark.asyncio
async def test_should_use_full_codex_replay_without_prisma_session_database(
    monkeypatch,
):
    monkeypatch.setattr(proxy_server, "prisma_client", None)

    plan = await _prepare_codex_kimi_chat_completions_adapter_route(
        request=MagicMock(),
        adapter_model="kimi_code/k3-high",
        prepared_request_body={
            "model": "kimi_code/k3-high",
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": "Run pwd and report the result.",
                        }
                    ],
                },
                {
                    "type": "function_call",
                    "call_id": "exec_command:0",
                    "name": "exec_command",
                    "arguments": '{"cmd":"pwd"}',
                },
                {
                    "type": "custom_tool_call_output",
                    "call_id": "exec_command:0",
                    "output": "/workspace",
                },
            ],
            "previous_response_id": "chatcmpl-prior-kimi-response",
            "tools": [
                {
                    "type": "function",
                    "name": "exec_command",
                    "description": "Run a shell command.",
                    "parameters": {
                        "type": "object",
                        "properties": {"cmd": {"type": "string"}},
                        "required": ["cmd"],
                    },
                }
            ],
        },
    )

    assert plan.perform_kwargs["completion_kwargs"]["messages"] == [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Run pwd and report the result.",
                }
            ],
        },
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "exec_command:0",
                    "type": "function",
                    "function": {
                        "name": "exec_command",
                        "arguments": '{"cmd":"pwd"}',
                    },
                    "index": 0,
                }
            ],
        },
        {
            "role": "tool",
            "content": "/workspace",
            "tool_call_id": "exec_command:0",
        },
    ]


@pytest.mark.asyncio
async def test_should_order_multiple_kimi_tool_results_by_assistant_call_order(
    monkeypatch,
):
    assistant_message = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "call_first",
                "type": "function",
                "function": {"name": "first_tool", "arguments": '{"first":1}'},
            },
            {
                "id": "call_second",
                "type": "function",
                "function": {"name": "second_tool", "arguments": '{"second":2}'},
            },
        ],
    }
    first_result = {
        "role": "tool",
        "tool_call_id": "call_first",
        "content": "first",
    }
    second_result = {
        "role": "tool",
        "tool_call_id": "call_second",
        "content": "second",
    }
    later_message = {"role": "user", "content": "after tools"}
    session_handler = AsyncMock(
        return_value={
            "model": "k3",
            "messages": [
                assistant_message,
                second_result,
                later_message,
                first_result,
            ],
            "custom_llm_provider": "kimi_code",
        }
    )
    monkeypatch.setattr(
        LiteLLMCompletionResponsesConfig,
        "async_responses_api_session_handler",
        session_handler,
    )

    plan = await prepare_codex_kimi_chat_completions_adapter_route(
        request=MagicMock(),
        adapter_model="kimi_code/k3-max",
        prepared_request_body={
            "model": "kimi_code/k3-max",
            "input": "continue",
            "previous_response_id": "resp_two_tools",
        },
    )

    messages = plan.perform_kwargs["completion_kwargs"]["messages"]
    assert messages == [
        {
            "role": "assistant",
            "tool_calls": assistant_message["tool_calls"],
        },
        first_result,
        second_result,
        later_message,
    ]
    assert messages[0]["tool_calls"] == assistant_message["tool_calls"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("messages", "error_match"),
    (
        (
            [
                {
                    "role": "tool",
                    "tool_call_id": "exec_command:0",
                    "content": "orphaned",
                }
            ],
            "tool result for unknown tool_call_id 'exec_command:0'",
        ),
        (
            [
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "exec_command:0",
                            "type": "function",
                            "function": {
                                "name": "exec_command",
                                "arguments": "{}",
                            },
                        }
                    ],
                }
            ],
            "missing tool results for tool_call_id\\(s\\): 'exec_command:0'",
        ),
    ),
)
async def test_should_reject_invalid_kimi_tool_result_history(monkeypatch, messages, error_match):
    monkeypatch.setattr(
        LiteLLMCompletionResponsesConfig,
        "async_responses_api_session_handler",
        AsyncMock(
            return_value={
                "model": "k3",
                "messages": messages,
                "custom_llm_provider": "kimi_code",
            }
        ),
    )

    with pytest.raises(ValueError, match=error_match):
        await prepare_codex_kimi_chat_completions_adapter_route(
            request=MagicMock(),
            adapter_model="kimi_code/k3-high",
            prepared_request_body={
                "model": "kimi_code/k3-high",
                "input": "continue",
                "previous_response_id": "resp_invalid_tools",
            },
        )


@pytest.mark.asyncio
async def test_should_preserve_anthropic_tools_and_remove_only_k2_effort_control():
    request_body = {
        "model": "kimi_code/kimi-for-coding",
        "max_tokens": 64,
        "messages": [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "call_1",
                        "name": "Read",
                        "input": {"path": "a"},
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "call_1",
                        "content": "ok",
                    }
                ],
            },
        ],
        "thinking": {"type": "enabled", "budget_tokens": 1024},
        "tools": [{"name": "Read", "input_schema": {"type": "object"}}],
        "parallel_tool_calls": True,
        "litellm_metadata": {"trace_id": "trace-anthropic"},
    }
    plan = await prepare_anthropic_kimi_chat_completions_adapter_route(
        request=MagicMock(),
        adapter_model="kimi_code/kimi-for-coding",
        prepared_request_body=request_body,
    )

    assert plan.target_url == KIMI_CODE_CHAT_COMPLETIONS_URL
    assert plan.api_key == KIMI_CODE_MANAGED_CREDENTIAL_SENTINEL
    assert plan.perform_kwargs["custom_llm_provider"] == "kimi_code"
    assert plan.prepared_request_body["messages"] == request_body["messages"]
    assert plan.prepared_request_body["tools"] == request_body["tools"]
    assert "thinking" not in plan.prepared_request_body
    assert "Bash" not in str(plan.prepared_request_body)
    assert "opencode" not in str(plan.prepared_request_body).lower()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("adapter_key", "upstream_model", "forced_effort"),
    (
        ("kimi_code/k3-low", "k3", "low"),
        ("kimi_code/k3-high", "k3", "high"),
        ("kimi_code/k3-max", "k3", "max"),
        ("kimi_code/kimi-for-coding", "kimi-for-coding", None),
        (
            "kimi_code/kimi-for-coding-highspeed",
            "kimi-for-coding-highspeed",
            None,
        ),
    ),
)
async def test_should_preserve_adapter_key_and_send_exact_upstream_model(adapter_key, upstream_model, forced_effort):
    codex_plan = await prepare_codex_kimi_chat_completions_adapter_route(
        request=MagicMock(),
        adapter_model=adapter_key,
        prepared_request_body={
            "model": adapter_key,
            "input": "hello",
            "reasoning": {"effort": "low"},
        },
    )
    anthropic_plan = await prepare_anthropic_kimi_chat_completions_adapter_route(
        request=MagicMock(),
        adapter_model=adapter_key,
        prepared_request_body={
            "model": adapter_key,
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 64,
            "thinking": {"type": "enabled", "budget_tokens": 1024},
        },
    )

    completion_kwargs = codex_plan.perform_kwargs["completion_kwargs"]
    assert completion_kwargs["model"] == upstream_model
    assert codex_plan.prepared_request_body["litellm_metadata"]["kimi_code_adapter_candidate"] == adapter_key
    assert anthropic_plan.perform_kwargs["model_for_upstream"] == upstream_model
    assert (
        anthropic_plan.prepared_request_body["litellm_metadata"]["passthrough_route_family"]
        == "anthropic_kimi_chat_completions_adapter"
    )
    if forced_effort is None:
        assert "reasoning_effort" not in completion_kwargs
        assert "reasoning_effort" not in anthropic_plan.perform_kwargs["extra_handler_kwargs"]
    else:
        assert completion_kwargs["reasoning_effort"] == forced_effort
        assert anthropic_plan.perform_kwargs["extra_handler_kwargs"]["reasoning_effort"] == forced_effort


def test_should_require_prefixed_direct_adapter_keys():
    assert (
        _resolve_codex_kimi_chat_completions_adapter_model({"model": "kimi_code/k3-high"}, "/v1/responses")
        == "kimi_code/k3-high"
    )
    assert (
        _resolve_anthropic_kimi_chat_completions_adapter_model({"model": "kimi_code/k3-max"}, "/v1/messages")
        == "kimi_code/k3-max"
    )
    assert _resolve_codex_kimi_chat_completions_adapter_model({"model": "k3"}, "/v1/responses") is None
    assert _resolve_anthropic_kimi_chat_completions_adapter_model({"model": "k3"}, "/v1/messages") is None


@pytest.mark.asyncio
async def test_should_run_codex_nonstream_through_hot_read_kimi_provider(monkeypatch, tmp_path, respx_mock):
    credentials_path = tmp_path / "kimi-code.json"
    _write_credentials(credentials_path)
    monkeypatch.setenv(KIMI_CODE_CREDENTIAL_PATH_ENV, str(credentials_path))
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    order = []

    def upstream(request):
        assert order == ["annotated"]
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-kimi",
                "object": "chat.completion",
                "created": 1,
                "model": "k3",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "reasoning": "inspect",
                            "content": "done",
                            "tool_calls": [
                                {
                                    "id": "call_read",
                                    "type": "function",
                                    "function": {
                                        "name": "read_file",
                                        "arguments": '{"path":"a"}',
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
                    "prompt_tokens_details": {"cached_tokens": 4},
                    "completion_tokens_details": {"reasoning_tokens": 3},
                },
            },
        )

    respx_mock.post(KIMI_CODE_CHAT_COMPLETIONS_URL).mock(side_effect=upstream)
    with (
        patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints._annotate_request_scope_for_adapted_access_log",
            side_effect=lambda *_args: (
                order.append("annotated") if not order else None
            ),
        ),
        patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints._emit_adapted_route_access_log",
        ) as emit_route_log,
        patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints._record_adapted_completed_route_rollup_turn",
        ) as record_completed_turn,
    ):
        response = await _handle_codex_kimi_chat_completions_adapter_route(
            endpoint="/v1/responses",
            request=_request(),
            fastapi_response=MagicMock(spec=Response),
            user_api_key_dict=MagicMock(),
            prepared_request_body={
                "model": "kimi_code/k3-high",
                "input": "inspect",
                "tools": [
                    {
                        "type": "function",
                        "name": "read_file",
                        "parameters": {"type": "object"},
                    }
                ],
                "stream": False,
                "litellm_metadata": {"trace_id": "trace-codex"},
            },
            adapter_model="kimi_code/k3-high",
        )

    assert emit_route_log.call_count == 1
    assert emit_route_log.call_args.kwargs["target_url"] == (
        KIMI_CODE_CHAT_COMPLETIONS_URL
    )
    assert emit_route_log.call_args.kwargs["request_body"]["model"] == (
        "kimi_code/k3-high"
    )
    assert record_completed_turn.call_count == 1
    response_text = response.body.decode()
    assert "inspect" in response_text
    assert "done" in response_text
    assert "call_read" in response_text
    assert "read_file" in response_text
    request = respx_mock.calls[0].request
    upstream_body = json.loads(request.content)
    assert upstream_body["model"] == "k3"
    assert upstream_body["thinking"]["effort"] == "high"
    assert request.headers["Authorization"] == "Bearer current-access-token"
    assert KIMI_CODE_MANAGED_CREDENTIAL_SENTINEL not in request.headers["Authorization"]
    assert "opencode" not in json.dumps(upstream_body).lower()


@pytest.mark.asyncio
async def test_should_adapt_apply_patch_and_restore_custom_tool_calls_for_kimi(monkeypatch, tmp_path, respx_mock):
    credentials_path = tmp_path / "kimi-code.json"
    _write_credentials(credentials_path)
    monkeypatch.setenv(KIMI_CODE_CREDENTIAL_PATH_ENV, str(credentials_path))
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    patch_text = "*** Begin Patch\n*** End Patch"

    respx_mock.post(KIMI_CODE_CHAT_COMPLETIONS_URL).respond(
        json={
            "id": "chatcmpl-kimi",
            "object": "chat.completion",
            "created": 1,
            "model": "k3",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_apply_patch",
                                "type": "function",
                                "function": {
                                    "name": "apply_patch",
                                    "arguments": json.dumps({"input": patch_text}),
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

    response = await _handle_codex_kimi_chat_completions_adapter_route(
        endpoint="/v1/responses",
        request=_request(),
        fastapi_response=MagicMock(spec=Response),
        user_api_key_dict=MagicMock(),
        prepared_request_body={
            "model": "kimi_code/k3-high",
            "input": [
                {"role": "user", "content": "apply the patch"},
                {
                    "type": "custom_tool_call",
                    "call_id": "call_prior_apply_patch",
                    "name": "apply_patch",
                    "input": patch_text,
                },
                {
                    "type": "custom_tool_call_output",
                    "call_id": "call_prior_apply_patch",
                    "output": "Exit code: 0",
                },
            ],
            "tools": [
                {
                    "type": "custom",
                    "name": "apply_patch",
                    "description": "Apply a patch to files in the workspace.",
                },
                {"type": "custom", "name": "exec_command"},
                {
                    "type": "function",
                    "name": "read_file",
                    "parameters": {"type": "object", "properties": {}},
                },
                {"type": "tool_search", "name": "tool_search"},
                {"type": "namespace", "name": "functions"},
                {"type": "web_search", "name": "web_search"},
                {"type": "image_generation"},
            ],
            "tool_choice": {"type": "custom", "name": "apply_patch"},
            "stream": False,
        },
        adapter_model="kimi_code/k3-high",
    )

    upstream_body = json.loads(respx_mock.calls[0].request.content)
    assert all(tool["type"] == "function" for tool in upstream_body["tools"])
    assert [tool["function"]["name"] for tool in upstream_body["tools"]] == ["apply_patch", "read_file"]
    upstream_body_json = json.dumps(upstream_body)
    for unsupported_tool_name in (
        "exec_command",
        "tool_search",
        "namespace",
        "web_search",
        "image_generation",
    ):
        assert unsupported_tool_name not in upstream_body_json
    assert upstream_body["messages"][-1]["role"] == "tool"
    assert upstream_body["messages"][-1]["tool_call_id"] == "call_prior_apply_patch"

    response_body = json.loads(response.body)
    custom_tool_call = next(item for item in response_body["output"] if item["type"] == "custom_tool_call")
    assert custom_tool_call == {
        "type": "custom_tool_call",
        "id": "call_apply_patch",
        "call_id": "call_apply_patch",
        "name": "apply_patch",
        "input": patch_text,
        "status": "completed",
    }


@pytest.mark.asyncio
async def test_should_emit_codex_stream_terminal_usage_and_tool_events(monkeypatch, tmp_path, respx_mock):
    credentials_path = tmp_path / "kimi-code.json"
    _write_credentials(credentials_path)
    monkeypatch.setenv(KIMI_CODE_CREDENTIAL_PATH_ENV, str(credentials_path))
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    stream_body = "\n\n".join(
        [
            'data: {"id":"chatcmpl-kimi","object":"chat.completion.chunk","created":1,"model":"k3","choices":[{"index":0,"delta":{"reasoning":"inspect","content":"working","tool_calls":[{"index":0,"id":"call_apply_patch","type":"function","function":{"name":"apply_patch","arguments":"{\\"input\\":\\"*** Begin Patch\\\\n*** End Patch\\"}"}}]},"finish_reason":null}]}',
            'data: {"id":"chatcmpl-kimi","object":"chat.completion.chunk","created":1,"model":"k3","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls","usage":{"prompt_tokens":5,"completion_tokens":4,"total_tokens":9}}]}',
            "data: [DONE]",
            "",
        ]
    )
    respx_mock.post(KIMI_CODE_CHAT_COMPLETIONS_URL).respond(
        content=stream_body.encode(),
        headers={"content-type": "text/event-stream"},
    )

    with (
        patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints._emit_adapted_route_access_log",
        ) as emit_route_log,
        patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints._record_adapted_completed_route_rollup_turn",
        ) as record_completed_turn,
    ):
        response = await _handle_codex_kimi_chat_completions_adapter_route(
            endpoint="/v1/responses",
            request=_request(),
            fastapi_response=MagicMock(spec=Response),
            user_api_key_dict=MagicMock(),
            prepared_request_body={
                "model": "kimi_code/k3-max",
                "input": "inspect",
                "tools": [
                    {
                        "type": "custom",
                        "name": "apply_patch",
                        "description": "Apply a patch to files in the workspace.",
                    },
                    {"type": "custom", "name": "exec_command"},
                ],
                "stream": True,
            },
            adapter_model="kimi_code/k3-max",
        )

        assert emit_route_log.call_count == 1
        assert record_completed_turn.call_count == 0
        event_text = await _stream_text(response)
        assert record_completed_turn.call_count == 1

    assert "response.completed" in event_text
    assert "call_apply_patch" in event_text
    assert "apply_patch" in event_text
    assert "custom_tool_call" in event_text
    assert "response.function_call_arguments.done" not in event_text
    assert "inspect" in event_text
    assert "working" in event_text
    assert "input_tokens" in event_text
    upstream_body = json.loads(respx_mock.calls[0].request.content)
    assert all(tool["type"] == "function" for tool in upstream_body["tools"])
    assert "exec_command" not in json.dumps(upstream_body)


@pytest.mark.asyncio
async def test_should_run_anthropic_nonstream_tool_continuation_with_variant_effort(monkeypatch, tmp_path, respx_mock):
    credentials_path = tmp_path / "kimi-code.json"
    _write_credentials(credentials_path)
    monkeypatch.setenv(KIMI_CODE_CREDENTIAL_PATH_ENV, str(credentials_path))
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    order = []

    def upstream(request):
        assert order == ["annotated"]
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-kimi",
                "object": "chat.completion",
                "created": 1,
                "model": "k3",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "reasoning": "continue",
                            "content": "result",
                            "tool_calls": [
                                {
                                    "id": "call_next",
                                    "type": "function",
                                    "function": {
                                        "name": "Write",
                                        "arguments": '{"path":"b"}',
                                    },
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
                "usage": {
                    "prompt_tokens": 8,
                    "completion_tokens": 6,
                    "total_tokens": 14,
                },
            },
        )

    respx_mock.post(KIMI_CODE_CHAT_COMPLETIONS_URL).mock(side_effect=upstream)
    with patch(
        "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints._annotate_request_scope_for_adapted_access_log",
        side_effect=lambda *_args: order.append("annotated"),
    ):
        response = await _handle_anthropic_kimi_chat_completions_adapter_route(
            endpoint="/v1/messages",
            request=_request(),
            fastapi_response=MagicMock(spec=Response),
            user_api_key_dict=MagicMock(),
            prepared_request_body={
                "model": "kimi_code/k3-low",
                "max_tokens": 64,
                "messages": [
                    {"role": "user", "content": "read"},
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "tool_use",
                                "id": "call_read",
                                "name": "Read",
                                "input": {"path": "a"},
                            }
                        ],
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": "call_read",
                                "content": "contents",
                            }
                        ],
                    },
                ],
                "tools": [
                    {"name": "Read", "input_schema": {"type": "object"}},
                    {"name": "Write", "input_schema": {"type": "object"}},
                ],
                "parallel_tool_calls": True,
                "stream": False,
            },
            adapter_model="kimi_code/k3-low",
        )

    response_text = response.body.decode()
    assert "thinking" in response_text
    assert "continue" in response_text
    assert "tool_use" in response_text
    assert "call_next" in response_text
    request = respx_mock.calls[0].request
    upstream_body = json.loads(request.content)
    assert upstream_body["model"] == "k3"
    assert upstream_body["thinking"]["effort"] == "low"
    assert upstream_body["parallel_tool_calls"] is True
    assert upstream_body["messages"][-1]["role"] == "tool"
    assert upstream_body["messages"][-1]["tool_call_id"] == "call_read"
    assert request.headers["Authorization"] == "Bearer current-access-token"
    assert not any("opencode" in key.lower() for key in request.headers)


@pytest.mark.asyncio
async def test_should_emit_anthropic_streaming_thinking_tool_and_terminal_events(monkeypatch, tmp_path, respx_mock):
    credentials_path = tmp_path / "kimi-code.json"
    _write_credentials(credentials_path)
    monkeypatch.setenv(KIMI_CODE_CREDENTIAL_PATH_ENV, str(credentials_path))
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    stream_body = "\n\n".join(
        [
            'data: {"id":"chatcmpl-kimi","object":"chat.completion.chunk","created":1,"model":"k3","choices":[{"index":0,"delta":{"reasoning":"think","content":"work","tool_calls":[{"index":0,"id":"call_write","type":"function","function":{"name":"Write","arguments":"{}"}}]},"finish_reason":null}]}',
            'data: {"id":"chatcmpl-kimi","object":"chat.completion.chunk","created":1,"model":"k3","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls","usage":{"prompt_tokens":7,"completion_tokens":5,"total_tokens":12}}]}',
            "data: [DONE]",
            "",
        ]
    )
    respx_mock.post(KIMI_CODE_CHAT_COMPLETIONS_URL).respond(
        content=stream_body.encode(),
        headers={"content-type": "text/event-stream"},
    )

    response = await _handle_anthropic_kimi_chat_completions_adapter_route(
        endpoint="/v1/messages",
        request=_request(),
        fastapi_response=MagicMock(spec=Response),
        user_api_key_dict=MagicMock(),
        prepared_request_body={
            "model": "kimi_code/k3-max",
            "max_tokens": 64,
            "messages": [{"role": "user", "content": "work"}],
            "tools": [{"name": "Write", "input_schema": {"type": "object"}}],
            "stream": True,
        },
        adapter_model="kimi_code/k3-max",
    )

    event_text = await _stream_text(response)
    assert "thinking_delta" in event_text
    assert "think" in event_text
    assert "work" in event_text
    assert "tool_use" in event_text
    assert "call_write" in event_text
    assert "message_delta" in event_text
    assert "input_tokens" in event_text
    assert "output_tokens" in event_text
    assert "message_stop" in event_text


class _KimiAdapterFailure(RuntimeError):
    def __init__(self, *, status_code: int, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.message = message
        self.error_code = "token_expired" if status_code == 401 else "upstream_error"
        self.headers = {"x-trace-id": "trace-safe"}


@pytest.mark.asyncio
@pytest.mark.parametrize("use_alias_candidate_probe", (False, True))
async def test_should_return_one_bounded_400_for_invalid_kimi_request_shape(
    use_alias_candidate_probe,
):
    raw_provider_detail = "text content is empty: raw-provider-detail"
    upstream_failure = _KimiAdapterFailure(
        status_code=400,
        message=raw_provider_detail,
    )
    completion_mock = AsyncMock(side_effect=upstream_failure)

    with (
        patch("litellm.acompletion", new=completion_mock),
        pytest.raises(HTTPException) as caught,
    ):
        await _handle_codex_kimi_chat_completions_adapter_route(
            endpoint="/v1/responses",
            request=_request(),
            fastapi_response=MagicMock(spec=Response),
            user_api_key_dict=MagicMock(),
            prepared_request_body={
                "model": "kimi_code/k3-high",
                "input": "hello",
                "stream": False,
            },
            adapter_model="kimi_code/k3-high",
            use_alias_candidate_probe=use_alias_candidate_probe,
        )

    assert caught.value.status_code == 400
    assert caught.value.detail == {
        "error": {
            "message": "Managed Kimi Code rejected the request shape.",
            "type": "invalid_request_error",
            "code": "kimi_code_invalid_request",
        }
    }
    assert raw_provider_detail not in json.dumps(caught.value.detail)
    assert completion_mock.await_count == 1
    assert not hasattr(caught.value, "kimi_code_probe_failure_metadata")


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", (False, True))
async def test_should_return_bounded_401_for_direct_managed_auth_failure(
    stream,
):
    raw_secret = "raw-expired-token-detail"
    upstream_failure = _KimiAdapterFailure(
        status_code=401,
        message=f"credential expired: {raw_secret}",
    )

    with (
        patch(
            "litellm.acompletion",
            new=AsyncMock(side_effect=upstream_failure),
        ),
        patch(
            "scripts.kimi_oauth_refresh.refresh_kimi_oauth_auth_file",
        ) as refresh_mock,
        pytest.raises(HTTPException) as caught,
    ):
        await _handle_codex_kimi_chat_completions_adapter_route(
            endpoint="/v1/responses",
            request=_request(),
            fastapi_response=MagicMock(spec=Response),
            user_api_key_dict=MagicMock(),
            prepared_request_body={
                "model": "kimi_code/k3-high",
                "input": "hello",
                "stream": stream,
            },
            adapter_model="kimi_code/k3-high",
        )

    assert caught.value.status_code == 401
    assert caught.value.detail == {
        "error": {
            "message": (
                "Managed Kimi Code authentication requires the shared "
                "credential to be refreshed."
            ),
            "type": "authentication_error",
            "code": "kimi_code_auth_refresh_required",
        }
    }
    assert raw_secret not in json.dumps(caught.value.detail)
    refresh_mock.assert_not_called()


@pytest.mark.asyncio
async def test_should_preserve_alias_probe_auth_metadata_and_original_failure():
    upstream_failure = _KimiAdapterFailure(
        status_code=401,
        message="credential expired: raw-provider-detail",
    )

    with (
        patch(
            "litellm.acompletion",
            new=AsyncMock(side_effect=upstream_failure),
        ),
        pytest.raises(_KimiAdapterFailure) as caught,
    ):
        await _handle_codex_kimi_chat_completions_adapter_route(
            endpoint="/v1/responses",
            request=_request(),
            fastapi_response=MagicMock(spec=Response),
            user_api_key_dict=MagicMock(),
            prepared_request_body={
                "model": "kimi_code/k3-high",
                "input": "hello",
                "stream": False,
            },
            adapter_model="kimi_code/k3-high",
            use_alias_candidate_probe=True,
        )

    assert caught.value is upstream_failure
    assert upstream_failure.kimi_code_probe_failure_metadata == {
        "kind": "refresh_required_auth",
        "scope": "managed_account",
        "upstream_id": "k3",
        "metadata_gate": "none",
        "status_code": 401,
        "trace_id": "trace-safe",
        "reset_reason": "refresh_required",
    }
    assert "raw-provider-detail" not in json.dumps(
        upstream_failure.kimi_code_probe_failure_metadata
    )


@pytest.mark.asyncio
async def test_should_leave_unrelated_direct_failure_unchanged():
    upstream_failure = _KimiAdapterFailure(
        status_code=500,
        message="unrelated internal failure",
    )

    with (
        patch(
            "litellm.acompletion",
            new=AsyncMock(side_effect=upstream_failure),
        ),
        pytest.raises(_KimiAdapterFailure) as caught,
    ):
        await _handle_codex_kimi_chat_completions_adapter_route(
            endpoint="/v1/responses",
            request=_request(),
            fastapi_response=MagicMock(spec=Response),
            user_api_key_dict=MagicMock(),
            prepared_request_body={
                "model": "kimi_code/k3-high",
                "input": "hello",
                "stream": False,
            },
            adapter_model="kimi_code/k3-high",
        )

    assert caught.value is upstream_failure
    assert not hasattr(upstream_failure, "kimi_code_probe_failure_metadata")
