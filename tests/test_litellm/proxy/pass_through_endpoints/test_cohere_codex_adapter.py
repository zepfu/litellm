"""Focused mocked coverage for the Cohere Codex candidate adapter."""

from __future__ import annotations

import copy
import json
import importlib
import re
from pathlib import Path
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from fastapi import HTTPException
from fastapi.responses import Response
from fastapi.responses import StreamingResponse

from litellm.exceptions import UnsupportedParamsError
from litellm.proxy._types import ProxyException

from litellm.proxy.pass_through_endpoints.aawm_adapter_runtime import (
    codex_candidate_calls,
)
from litellm.proxy.pass_through_endpoints import llm_passthrough_endpoints as lpe
from litellm.proxy.pass_through_endpoints.aawm_alias_routing import (
    adapter_driver,
    candidate_loop,
)
from litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints import (
    _adapt_codex_custom_tools_to_functions_from_request_body,
    _drop_unsupported_codex_hosted_tools_from_request_body,
    _drop_unsupported_codex_input_items_from_request_body,
)
from litellm.proxy.pass_through_endpoints.providers.cohere import runtime as cohere_runtime


COHERE_TARGET = "https://api.cohere.com/v2/chat"
COHERE_KEY = "cohere-test-key"


def _identity_adapter(body):
    return body, []


def _identity_drop(body):
    return body, []


def _response_from_payload(payload, **_kwargs):
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


def _bind_live_codex_tool_policy(cohere):
    cohere.host.update(
        {
            "_adapt_codex_custom_tools_to_functions_from_request_body": (
                _adapt_codex_custom_tools_to_functions_from_request_body
            ),
            "_drop_unsupported_codex_hosted_tools_from_request_body": (
                _drop_unsupported_codex_hosted_tools_from_request_body
            ),
            "_drop_unsupported_codex_input_items_from_request_body": (
                _drop_unsupported_codex_input_items_from_request_body
            ),
        }
    )


def _transform_cohere_request(**kwargs):
    responses_api_request = kwargs["responses_api_request"]
    transformed = {
        "model": kwargs["model"],
        "messages": [{"role": "user", "content": "translated"}],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool.get("description", ""),
                    "parameters": tool["parameters"],
                    "strict": False,
                },
            }
            for tool in responses_api_request.get("tools", [])
        ],
    }
    tool_choice = responses_api_request.get("tool_choice")
    if isinstance(tool_choice, dict) and tool_choice.get("type") == "function":
        transformed["tool_choice"] = {
            "type": "function",
            "function": {"name": tool_choice["name"]},
        }
    return transformed


async def _prepare(
    cohere,
    *,
    stream=False,
    body=None,
    adapter_model=" cohere/command-r-plus ",
):
    return await cohere.host[
        "_prepare_codex_cohere_chat_completions_adapter_route"
    ](
        request=MagicMock(),
        prepared_request_body=body or _request_body(stream=stream),
        adapter_model=adapter_model,
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


def test_should_declare_direct_cohere_hosted_tool_capabilities_in_catalogs():
    for catalog_path in (
        "model_prices_and_context_window.json",
        "litellm/bundled_model_prices_and_context_window_fallback.json",
    ):
        catalog = json.loads(Path(catalog_path).read_text(encoding="utf-8"))
        row = catalog["cohere/north-mini-code-1-0"]

        assert row["custom_tool_function_adapters"] == ["apply_patch"]
        assert row["unsupported_hosted_tools"] == ["custom", "tool_search"]
        assert row["unsupported_input_item_types"] == [
            "custom_tool_call",
            "custom_tool_call_output",
        ]


@pytest.mark.asyncio
async def test_should_adapt_direct_cohere_apply_patch_and_preserve_tool_continuation(
    cohere_host,
):
    _bind_live_codex_tool_policy(cohere_host)
    patch_text = "*** Begin Patch\n*** End Patch"
    body = {
        "model": "cohere/north-mini-code-1-0",
        "input": [
            {
                "role": "user",
                "content": [{"type": "input_text", "text": "Apply this patch"}],
            },
            {
                "type": "custom_tool_call",
                "id": "ctc_apply_patch",
                "status": "completed",
                "call_id": "call_apply_patch",
                "name": "apply_patch",
                "input": patch_text,
            },
            {
                "type": "custom_tool_call_output",
                "call_id": "call_apply_patch",
                "output": "Exit code: 0",
            },
            {
                "type": "custom_tool_call",
                "call_id": "call_unsupported",
                "name": "exec_command",
                "input": "pwd",
            },
            {
                "type": "custom_tool_call_output",
                "call_id": "call_unsupported",
                "output": "/workspace",
            },
        ],
        "tools": [
            {
                "type": "custom",
                "name": "apply_patch",
                "description": "Apply a patch to files in the workspace.",
            },
            {
                "type": "custom",
                "name": "exec_command",
                "description": "Run a command in the workspace.",
            },
        ],
        "tool_choice": {"type": "custom", "name": "apply_patch"},
        "stream": False,
    }
    original_body = copy.deepcopy(body)
    cohere_host.transform.transform_responses_api_request_to_chat_completion_request.side_effect = (
        _transform_cohere_request
    )

    plan = await _prepare(
        cohere_host,
        body=body,
        adapter_model=" cohere/north-mini-code-1-0 ",
    )

    assert body == original_body
    assert plan.prepared_request_body is not body
    assert plan.prepared_request_body["tools"] == [
        {
            "type": "function",
            "name": "apply_patch",
            "description": "Apply a patch to files in the workspace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "input": {
                        "type": "string",
                        "description": (
                            "Raw input for the client-hosted custom tool. For "
                            "apply_patch this must be the complete patch text."
                        ),
                    }
                },
                "required": ["input"],
                "additionalProperties": False,
            },
        }
    ]
    assert plan.prepared_request_body["tool_choice"] == {
        "type": "function",
        "name": "apply_patch",
    }
    assert plan.perform_kwargs["request_input"] == [
        body["input"][0],
        {
            "type": "function_call",
            "id": "ctc_apply_patch",
            "status": "completed",
            "call_id": "call_apply_patch",
            "name": "apply_patch",
            "arguments": json.dumps(
                {"input": patch_text},
                ensure_ascii=False,
                separators=(",", ":"),
            ),
        },
        {
            "type": "function_call_output",
            "call_id": "call_apply_patch",
            "output": "Exit code: 0",
        },
    ]
    assert plan.perform_kwargs["completion_kwargs"]["tools"] == [
        {
            "type": "function",
            "function": {
                "name": "apply_patch",
                "description": "Apply a patch to files in the workspace.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "input": {
                            "type": "string",
                            "description": (
                                "Raw input for the client-hosted custom tool. For "
                                "apply_patch this must be the complete patch text."
                            ),
                        }
                    },
                    "required": ["input"],
                    "additionalProperties": False,
                },
            },
        }
    ]
    assert plan.perform_kwargs["completion_kwargs"]["tool_choice"] == {
        "type": "function",
        "function": {"name": "apply_patch"},
    }
    metadata = plan.prepared_request_body["litellm_metadata"]
    assert metadata["codex_custom_tool_function_adapter_names"] == ["apply_patch"]
    assert metadata["codex_unsupported_hosted_tools_removed"] == [
        {"type": "custom", "index": 1, "name": "exec_command"}
    ]
    assert metadata["codex_unsupported_input_items_removed"] == [
        {"type": "custom_tool_call", "index": 3},
        {"type": "custom_tool_call_output", "index": 4},
    ]


@pytest.mark.asyncio
async def test_should_drop_direct_cohere_tool_search_before_completion_transform(
    cohere_host,
):
    _bind_live_codex_tool_policy(cohere_host)
    body = {
        "model": "cohere/north-mini-code-1-0",
        "input": [
            {
                "role": "user",
                "content": [{"type": "input_text", "text": "Find and patch it"}],
            }
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
            },
            {
                "type": "custom",
                "name": "apply_patch",
                "description": "Apply a patch to files in the workspace.",
                "format": {
                    "type": "grammar",
                    "syntax": "lark",
                    "definition": "start: /.+/",
                },
            },
            {"type": "tool_search", "name": "tool_search"},
        ],
        "tool_choice": {"type": "tool_search", "name": "tool_search"},
        "stream": False,
    }
    original_body = copy.deepcopy(body)
    cohere_host.transform.transform_responses_api_request_to_chat_completion_request.side_effect = (
        _transform_cohere_request
    )

    plan = await _prepare(
        cohere_host,
        body=body,
        adapter_model=" cohere/north-mini-code-1-0 ",
    )

    assert body == original_body
    assert plan.prepared_request_body is not body
    assert plan.prepared_request_body["tools"] == [
        body["tools"][0],
        {
            "type": "function",
            "name": "apply_patch",
            "description": "Apply a patch to files in the workspace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "input": {
                        "type": "string",
                        "description": (
                            "Raw input for the client-hosted custom tool. For "
                            "apply_patch this must be the complete patch text."
                        ),
                    }
                },
                "required": ["input"],
                "additionalProperties": False,
            },
        },
    ]
    assert "tool_search" not in {tool["type"] for tool in plan.prepared_request_body["tools"]}
    assert "tool_choice" not in plan.prepared_request_body
    assert "tool_choice" not in plan.perform_kwargs["responses_api_request"]

    completion_tools = plan.perform_kwargs["completion_kwargs"]["tools"]
    assert completion_tools == [
        {
            "type": "function",
            "function": {
                "name": "lookup",
                "description": "Look up a value",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "apply_patch",
                "description": "Apply a patch to files in the workspace.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "input": {
                            "type": "string",
                            "description": (
                                "Raw input for the client-hosted custom tool. For "
                                "apply_patch this must be the complete patch text."
                            ),
                        }
                    },
                    "required": ["input"],
                    "additionalProperties": False,
                },
            },
        },
    ]
    assert all(tool["type"] == "function" for tool in completion_tools)
    assert "tool_search" not in {tool["function"]["name"] for tool in completion_tools}
    assert "tool_choice" not in plan.perform_kwargs["completion_kwargs"]

    metadata = plan.prepared_request_body["litellm_metadata"]
    assert metadata["codex_unsupported_hosted_tools_removed"] == [
        {"type": "tool_search", "index": 2, "name": "tool_search"}
    ]
    assert metadata["codex_unsupported_hosted_tool_choice_removed"] == {
        "type": "tool_search",
        "name": "tool_search",
    }


@pytest.mark.asyncio
async def test_should_prepare_valid_cohere_candidate_and_map_upstream_model(cohere_host):
    plan = await _prepare(cohere_host)

    assert plan.prepared_request_body["model"] == "cohere/command-r-plus"
    assert plan.perform_kwargs["upstream_model"] == "command-r-plus"
    assert plan.perform_kwargs["completion_kwargs"]["model"] == "command-r-plus"
    assert plan.api_key == COHERE_KEY
    assert str(plan.target_url) == COHERE_TARGET


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.asyncio
async def test_cohere_route_preparation_round_trips_stock_namespace_identity(
    cohere_host, streaming
):
    body = {
        "model": "cohere/command-r-plus",
        "stream": streaming,
        "input": [
            {"role": "user", "content": "wait for collaboration"},
            {
                "type": "function_call",
                "name": "wait_agent",
                "namespace": "functions.collaboration",
                "call_id": "call_wait_1",
                "arguments": '{"timeout_ms":3}',
            },
            {
                "type": "function_call_output",
                "namespace": "functions.collaboration",
                "call_id": "call_wait_1",
                "output": "wait completed",
            },
        ],
        "tools": [
            {
                "type": "namespace",
                "name": "functions.collaboration",
                "tools": [
                    {
                        "type": "function",
                        "name": "wait_agent",
                        "parameters": {
                            "type": "object",
                            "properties": {"timeout_ms": {"type": "integer"}},
                        },
                    }
                ],
            }
        ],
    }
    plan = await _prepare(cohere_host, stream=streaming, body=body)
    advertised_tool = plan.perform_kwargs["responses_api_request"]["tools"][0]
    wire_name = advertised_tool["name"]
    assert re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", wire_name)
    assert plan.perform_kwargs["request_input"][1]["name"] == wire_name
    assert plan.perform_kwargs["request_input"][1]["call_id"] == "call_wait_1"
    assert plan.perform_kwargs["request_input"][1]["arguments"] == '{"timeout_ms":3}'
    assert plan.perform_kwargs["request_input"][2]["call_id"] == "call_wait_1"
    assert plan.perform_kwargs["request_input"][2]["output"] == "wait completed"

    provider_item = {
        "type": "function_call",
        "name": wire_name,
        "call_id": "call_wait_1",
        "arguments": '{"timeout_ms":3}',
    }
    if not streaming:
        restored, count = lpe._restore_adapted_namespace_tool_calls_in_response_body(
            {"output": [provider_item]},
            request_body=body,
            adapter_model="cohere/command-r-plus",
        )
        assert count == 1
        assert restored["output"][0] == {
            **provider_item,
            "name": "wait_agent",
            "namespace": "functions.collaboration",
        }
        return

    from litellm.proxy.pass_through_endpoints.aawm_adapter_runtime import (
        tool_call_restore,
    )

    events = [
        {
            "type": "response.output_item.done",
            "item": provider_item,
        },
        {
            "type": "response.function_call_arguments.done",
            "name": wire_name,
            "call_id": "call_wait_1",
            "arguments": '{"timeout_ms":3}',
        },
        {
            "type": "response.completed",
            "response": {"output": [provider_item]},
        },
    ]

    async def upstream_events():
        for event in events:
            event_type = event["type"]
            yield f"event: {event_type}\ndata: {json.dumps(event)}\n\n".encode()

    upstream = StreamingResponse(upstream_events(), media_type="text/event-stream")
    restored_stream = tool_call_restore._restore_adapted_namespace_tool_calls_in_streaming_response(
        upstream,
        request_body=body,
        adapter_model="cohere/command-r-plus",
    )
    stream_chunks = [chunk async for chunk in restored_stream.body_iterator]
    stream_bytes = b"".join(
        chunk.encode() if isinstance(chunk, str) else chunk for chunk in stream_chunks
    )
    restored_events = [
        json.loads(line[6:])
        for line in stream_bytes.decode().splitlines()
        if line.startswith("data: ")
    ]
    assert restored_events[0]["item"]["name"] == "wait_agent"
    assert restored_events[0]["item"]["namespace"] == "functions.collaboration"
    assert restored_events[0]["item"]["call_id"] == "call_wait_1"
    assert restored_events[1]["name"] == "wait_agent"
    assert restored_events[1]["namespace"] == "functions.collaboration"
    assert restored_events[1]["arguments"] == '{"timeout_ms":3}'
    assert restored_events[2]["response"]["output"][0]["name"] == "wait_agent"
    assert restored_events[2]["response"]["output"][0]["namespace"] == (
        "functions.collaboration"
    )


@pytest.mark.parametrize("namespaced_first", [False, True])
@pytest.mark.parametrize("include_bare_ops", [False, True])
@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.asyncio
async def test_cohere_route_preparation_preserves_flat_and_bare_schemas(
    cohere_host, namespaced_first, include_bare_ops, streaming
):
    namespaced_tool = {
        "type": "function",
        "namespace": "ops",
        "name": "run",
        "parameters": {
            "type": "object",
            "properties": {"n": {"type": "integer"}},
        },
    }
    bare_run_tool = {
        "type": "function",
        "name": "run",
        "parameters": {
            "type": "object",
            "properties": {"n": {"type": "number"}},
        },
    }
    bare_ops_tool = {
        "type": "function",
        "name": "ops",
        "parameters": {
            "type": "object",
            "properties": {"n": {"type": "number"}},
        },
    }
    bare_tools = [bare_run_tool, *([bare_ops_tool] if include_bare_ops else [])]
    tools = [namespaced_tool, *bare_tools] if namespaced_first else [*bare_tools, namespaced_tool]
    body = {
        "model": "cohere/command-r-plus",
        "input": "return tool calls",
        "tools": tools,
        "stream": streaming,
    }
    plan = await _prepare(cohere_host, stream=streaming, body=body)
    emitted = plan.perform_kwargs["responses_api_request"]["tools"]
    namespaced_index = 0 if namespaced_first else len(emitted) - 1
    namespaced_wire_name = emitted[namespaced_index]["name"]
    emitted_bare_names = [
        tool["name"] for index, tool in enumerate(emitted) if index != namespaced_index
    ]
    assert all(re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name) for name in emitted_bare_names)
    assert namespaced_wire_name not in emitted_bare_names

    provider_calls = [
        {
            "type": "function_call",
            "name": namespaced_wire_name,
            "call_id": "call_namespaced",
            "arguments": '{"n":3.0}',
        }
    ]
    bare_indexes = [index for index in range(len(emitted)) if index != namespaced_index]
    provider_calls.extend(
        [
            {
                "type": "function_call",
                "name": emitted[bare_indexes[0]]["name"],
                "call_id": "call_bare_run",
                "arguments": '{"n":3.0}',
            },
            *(
                [
                    {
                        "type": "function_call",
                        "name": emitted[bare_indexes[1]]["name"],
                        "call_id": "call_bare_ops",
                        "arguments": '{"n":3.0}',
                    }
                ]
                if include_bare_ops
                else []
            ),
        ]
    )

    def assert_restored_calls(items):
        restored_by_call_id = {item["call_id"]: item for item in items}
        expectations = [("call_bare_run", "run", None, float)]
        if include_bare_ops:
            expectations.append(("call_bare_ops", "ops", None, float))
        expectations.append(("call_namespaced", "run", "ops", int))
        for call_id, name, namespace, value_type in expectations:
            item = restored_by_call_id[call_id]
            assert (item["name"], item.get("namespace")) == (name, namespace)
            value = json.loads(item["arguments"])["n"]
            if value_type is int:
                assert isinstance(value, int) and not isinstance(value, bool)
            else:
                assert isinstance(value, float) and item["arguments"] == '{"n":3.0}'

    if not streaming:
        restored, _ = lpe._restore_adapted_namespace_tool_calls_in_response_body(
            {"output": provider_calls},
            request_body=body,
            adapter_model="cohere/command-r-plus",
        )
        assert_restored_calls(restored["output"])
        return

    from litellm.proxy.pass_through_endpoints.aawm_adapter_runtime import (
        tool_call_restore,
    )

    events = [
        {
            "type": "response.output_item.done",
            "item": call,
        }
        for call in provider_calls
    ]
    events.extend(
        {
            "type": "response.function_call_arguments.done",
            "name": call["name"],
            "call_id": call["call_id"],
            "arguments": call["arguments"],
        }
        for call in provider_calls
    )
    events.append(
        {
            "type": "response.completed",
            "response": {"output": provider_calls},
        }
    )

    async def upstream_events():
        for event in events:
            event_type = event["type"]
            yield f"event: {event_type}\ndata: {json.dumps(event)}\n\n".encode()

    upstream = StreamingResponse(upstream_events(), media_type="text/event-stream")
    restored_stream = tool_call_restore._restore_adapted_namespace_tool_calls_in_streaming_response(
        upstream,
        request_body=body,
        adapter_model="cohere/command-r-plus",
    )
    stream_chunks = [chunk async for chunk in restored_stream.body_iterator]
    stream_bytes = b"".join(
        chunk.encode() if isinstance(chunk, str) else chunk for chunk in stream_chunks
    )
    restored_events = [
        json.loads(line[6:])
        for line in stream_bytes.decode().splitlines()
        if line.startswith("data: ")
    ]
    item_events = [
        event["item"]
        for event in restored_events
        if event["type"] == "response.output_item.done"
    ]
    args_events = [
        event
        for event in restored_events
        if event["type"] == "response.function_call_arguments.done"
    ]
    terminal = next(
        event["response"]["output"]
        for event in restored_events
        if event["type"] == "response.completed"
    )
    assert_restored_calls(item_events)
    assert_restored_calls(args_events)
    assert_restored_calls(terminal)


@pytest.mark.asyncio
async def test_cohere_route_preparation_rejects_bare_only_duplicate_before_egress(
    cohere_host,
):
    duplicate_tool = {
        "type": "function",
        "name": "run",
        "parameters": {"type": "object", "properties": {}},
    }
    body = {
        "model": "cohere/command-r-plus",
        "input": "run",
        "tools": [duplicate_tool, copy.deepcopy(duplicate_tool)],
    }
    error = None
    try:
        await _prepare(cohere_host, body=body)
    except UnsupportedParamsError as exc:
        error = exc
    assert (
        isinstance(error, UnsupportedParamsError),
        cohere_host.egress.call_count,
        cohere_host.completion.await_count,
    ) == (True, 0, 0)


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
async def test_should_strip_cohere_tool_strict_without_mutating_transformed_tools(
    cohere_host,
):
    schema = {
        "type": "object",
        "properties": {
            "strict": {
                "type": "string",
                "minLength": 1,
            },
            "query": {
                "type": "string",
                "minLength": 2,
                "maxLength": 80,
            },
        },
        "required": ["strict", "query"],
        "additionalProperties": False,
    }
    transformed_completion_kwargs = {
        "model": "command-r-plus",
        "messages": [{"role": "user", "content": "translated"}],
        "tools": [
            {
                "type": "function",
                "strict": False,
                "function": {
                    "name": "lookup",
                    "description": "Look up a value",
                    "strict": False,
                    "parameters": schema,
                },
            },
            {
                "type": "function",
                "name": "respond",
                "description": "Return a value",
                "strict": False,
                "parameters": copy.deepcopy(schema),
            },
        ],
        "tool_choice": {
            "type": "function",
            "function": {"name": "lookup"},
        },
    }
    original_completion_kwargs = copy.deepcopy(transformed_completion_kwargs)
    cohere_host.transform.transform_responses_api_request_to_chat_completion_request.return_value = (
        transformed_completion_kwargs
    )

    plan = await _prepare(cohere_host)
    bound_tools = plan.perform_kwargs["completion_kwargs"]["tools"]

    assert bound_tools == [
        {
            "type": "function",
            "function": {
                "name": "lookup",
                "description": "Look up a value",
                "parameters": schema,
            },
        },
        {
            "type": "function",
            "name": "respond",
            "description": "Return a value",
            "parameters": schema,
        },
    ]
    assert plan.perform_kwargs["completion_kwargs"]["tool_choice"] == (
        original_completion_kwargs["tool_choice"]
    )
    assert transformed_completion_kwargs == original_completion_kwargs
    assert bound_tools is not transformed_completion_kwargs["tools"]
    assert bound_tools[0]["function"]["parameters"]["properties"]["strict"] == {
        "type": "string",
        "minLength": 1,
    }
    assert bound_tools[0]["function"]["parameters"]["required"] == [
        "strict",
        "query",
    ]
    assert bound_tools[0]["function"]["parameters"]["properties"]["query"] == {
        "type": "string",
        "minLength": 2,
        "maxLength": 80,
    }


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
            "model": "cohere/command-r",
            "route_family": "codex_cohere_chat_completions_adapter",
        },
        is_codex_alias=True,
        attempted_provider_call=True,
    )

    assert result == expected
    classifier.assert_called_once()
    assert classifier.call_args.kwargs["attempted_provider_call"] is True
    assert classifier.call_args.kwargs["route_family"] == (
        "codex_cohere_chat_completions_adapter"
    )
    assert classifier.call_args.kwargs["selected_upstream_model"] == "command-r"


@pytest.mark.parametrize(
    ("status_code", "detail"),
    [
        (404, "404 page not found"),
        (404, "model not found"),
        (400, "invalid request: route is not valid"),
    ],
)
def test_should_keep_unattributed_cohere_4xx_as_terminal(
    status_code,
    detail,
):
    exc = RuntimeError("Cohere request failed")
    exc.status_code = status_code
    exc.detail = {"message": detail}

    result = candidate_loop._classify_codex_cohere_candidate_failure(
        exc,
        candidate={
            "provider": "cohere",
            "model": "cohere/command-r",
            "route_family": "codex_cohere_chat_completions_adapter",
        },
        is_codex_alias=True,
        attempted_provider_call=True,
    )

    assert result == "provider_terminal_error"


def test_should_map_structured_model_bound_cohere_404_to_candidate_unavailable():
    exc = RuntimeError("Cohere request failed")
    exc.status_code = 404
    exc.detail = {"error": {"message": "model 'command-r' not found"}}
    exc._aawm_provider_returned = True

    result = candidate_loop._classify_codex_cohere_candidate_failure(
        exc,
        candidate={
            "provider": "cohere",
            "model": "cohere/command-r",
            "route_family": "codex_cohere_chat_completions_adapter",
        },
        is_codex_alias=True,
        attempted_provider_call=True,
    )

    assert result == "candidate_unavailable"


@pytest.mark.parametrize(
    ("error_model", "attempted_provider_call", "provider_returned"),
    [
        ("command-r", True, False),
        ("command-r", False, True),
        ("different-model", True, True),
    ],
)
def test_should_keep_cohere_model_rejection_terminal_without_exact_evidence(
    error_model,
    attempted_provider_call,
    provider_returned,
):
    exc = RuntimeError("Cohere request failed")
    exc.status_code = 404
    exc.detail = {"error": {"message": f"model '{error_model}' not found"}}
    if provider_returned:
        exc._aawm_provider_returned = True

    result = candidate_loop._classify_codex_cohere_candidate_failure(
        exc,
        candidate={
            "provider": "cohere",
            "model": "cohere/command-r",
            "route_family": "codex_cohere_chat_completions_adapter",
        },
        is_codex_alias=True,
        attempted_provider_call=attempted_provider_call,
    )

    assert result == "provider_terminal_error"


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


def test_should_wrap_unclassified_probe_failure_as_proxy_exception() -> None:
    raw = RuntimeError("openrouter 422")
    wrapped = candidate_loop._proxy_exception_for_unclassified_probe_failure(raw)
    assert isinstance(wrapped, ProxyException)
    assert wrapped is not raw
    assert "openrouter 422" in wrapped.message
    assert wrapped.code == "500"
    assert wrapped.type == "internal_server_error"

    http_exc = HTTPException(status_code=422, detail="already http")
    assert (
        candidate_loop._proxy_exception_for_unclassified_probe_failure(http_exc)
        is http_exc
    )
    proxy_exc = ProxyException(
        message="already proxy",
        type="invalid_request_error",
        param="model",
        code=400,
    )
    assert (
        candidate_loop._proxy_exception_for_unclassified_probe_failure(proxy_exc)
        is proxy_exc
    )


@pytest.mark.parametrize("provider_status", [400, 401, 422, 429, 500, 502, 529])
def test_should_preserve_recognized_probe_failure_statuses(provider_status) -> None:
    raw = RuntimeError("upstream probe failed")
    raw.status_code = provider_status  # type: ignore[attr-defined]

    wrapped = candidate_loop._proxy_exception_for_unclassified_probe_failure(raw)

    assert isinstance(wrapped, ProxyException)
    assert wrapped.code == str(provider_status)


def test_should_map_statusless_probe_failure_to_internal_server_error() -> None:
    raw = RuntimeError("adapter crashed without a provider status")

    wrapped = candidate_loop._proxy_exception_for_unclassified_probe_failure(raw)

    assert isinstance(wrapped, ProxyException)
    assert wrapped.code == "500"
    assert wrapped.type == "internal_server_error"
    assert "adapter crashed without a provider status" in wrapped.message
