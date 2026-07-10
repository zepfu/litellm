from types import SimpleNamespace

import pytest

from litellm.llms.anthropic.experimental_pass_through.responses_adapters.streaming_iterator import (
    AnthropicResponsesEmptySuccessError,
    AnthropicResponsesProviderError,
    AnthropicResponsesStreamWrapper,
)


async def _make_stream(*events):
    for event in events:
        yield event


@pytest.mark.asyncio
async def test_stream_wrapper_emits_mcp_tool_use_and_result_blocks():
    response_obj = SimpleNamespace(
        status="completed",
        usage=SimpleNamespace(
            input_tokens=10,
            output_tokens=5,
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
        ),
        output=[],
    )
    events = [
        SimpleNamespace(type="response.created"),
        SimpleNamespace(
            type="response.output_item.added",
            item=SimpleNamespace(
                type="mcp_call",
                id="mcp_123",
                name="search_docs",
                server_label="docs-mcp",
            ),
        ),
        SimpleNamespace(
            type="response.mcp_call_arguments.delta",
            item_id="mcp_123",
            delta='{"query":"adapter"}',
        ),
        SimpleNamespace(
            type="response.mcp_call.completed",
            item=SimpleNamespace(
                type="mcp_call",
                id="mcp_123",
                output="Found adapter docs",
                error=None,
            ),
        ),
        SimpleNamespace(type="response.completed", response=response_obj),
    ]

    wrapper = AnthropicResponsesStreamWrapper(
        responses_stream=_make_stream(*events),
        model="gpt-5.4",
    )
    chunks = [chunk async for chunk in wrapper]

    assert [chunk["type"] for chunk in chunks] == [
        "message_start",
        "content_block_start",
        "content_block_delta",
        "content_block_stop",
        "content_block_start",
        "content_block_stop",
        "message_delta",
        "message_stop",
    ]
    assert chunks[1]["content_block"] == {
        "type": "mcp_tool_use",
        "id": "mcp_123",
        "name": "search_docs",
        "server_name": "docs-mcp",
        "input": {},
    }
    assert chunks[2]["delta"] == {
        "type": "input_json_delta",
        "partial_json": '{"query":"adapter"}',
    }
    assert chunks[4]["content_block"] == {
        "type": "mcp_tool_result",
        "tool_use_id": "mcp_123",
        "is_error": False,
        "content": [{"type": "text", "text": "Found adapter docs"}],
    }
    assert chunks[6]["delta"]["stop_reason"] == "end_turn"
    assert chunks[6]["usage"] == {"input_tokens": 10, "output_tokens": 5}


@pytest.mark.asyncio
async def test_stream_wrapper_failed_response_raises_provider_error():
    response_obj = SimpleNamespace(
        status="failed",
        error={"message": "provider failed"},
        usage=SimpleNamespace(input_tokens=10, output_tokens=0),
        output=[],
    )
    wrapper = AnthropicResponsesStreamWrapper(
        responses_stream=_make_stream(
            SimpleNamespace(type="response.created"),
            SimpleNamespace(type="response.failed", response=response_obj),
        ),
        model="gpt-5.4",
    )

    with pytest.raises(AnthropicResponsesProviderError) as exc_info:
        async for _ in wrapper:
            pass

    assert "failed response" in str(exc_info.value)
    assert "provider failed" in str(exc_info.value)


@pytest.mark.asyncio
async def test_stream_wrapper_incomplete_event_emits_message_delta_max_tokens():
    response_obj = SimpleNamespace(
        id="resp_incomplete",
        status="incomplete",
        model="gpt-5.4",
        usage=SimpleNamespace(
            input_tokens=12,
            output_tokens=4,
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
        ),
        output=[],
        incomplete_details={"reason": "max_output_tokens"},
    )
    wrapper = AnthropicResponsesStreamWrapper(
        responses_stream=_make_stream(
            SimpleNamespace(type="response.created"),
            SimpleNamespace(
                type="response.output_text.done",
                item_id="msg_incomplete",
                output_index=0,
                text="adapter incomplete stream",
            ),
            SimpleNamespace(type="response.incomplete", response=response_obj),
        ),
        model="gpt-5.4",
    )

    chunks = [chunk async for chunk in wrapper]

    assert [chunk["type"] for chunk in chunks] == [
        "message_start",
        "content_block_start",
        "content_block_delta",
        "message_delta",
        "message_stop",
    ]
    assert chunks[3]["delta"]["stop_reason"] == "max_tokens"


@pytest.mark.asyncio
async def test_stream_wrapper_seeds_function_call_input_when_added_item_has_dict_arguments():
    response_obj = SimpleNamespace(
        status="completed",
        usage=SimpleNamespace(
            input_tokens=12,
            output_tokens=7,
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
        ),
        output=[
            {
                "type": "function_call",
                "id": "fc_123",
                "call_id": "call_fc_123",
                "name": "Bash",
                "arguments": {"command": "pwd", "description": "show cwd"},
            }
        ],
    )
    events = [
        SimpleNamespace(type="response.created"),
        SimpleNamespace(
            type="response.output_item.added",
            item=SimpleNamespace(
                type="function_call",
                id="fc_123",
                call_id="call_fc_123",
                name="Bash",
                arguments={"command": "pwd", "description": "show cwd"},
            ),
        ),
        SimpleNamespace(
            type="response.output_item.done",
            item=SimpleNamespace(
                type="function_call",
                id="fc_123",
                call_id="call_fc_123",
                name="Bash",
                arguments={"command": "pwd", "description": "show cwd"},
            ),
        ),
        SimpleNamespace(type="response.completed", response=response_obj),
    ]

    wrapper = AnthropicResponsesStreamWrapper(
        responses_stream=_make_stream(*events),
        model="gpt-5.3-codex-spark",
    )
    chunks = [chunk async for chunk in wrapper]

    assert chunks[1]["content_block"] == {
        "type": "tool_use",
        "id": "call_fc_123",
        "name": "Bash",
        "input": {"command": "pwd", "description": "show cwd"},
    }
    assert [chunk["type"] for chunk in chunks] == [
        "message_start",
        "content_block_start",
        "content_block_stop",
        "message_delta",
        "message_stop",
    ]


@pytest.mark.asyncio
async def test_stream_wrapper_emits_done_delta_when_function_call_arguments_arrive_only_at_done():
    response_obj = SimpleNamespace(
        status="completed",
        usage=SimpleNamespace(
            input_tokens=12,
            output_tokens=7,
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
        ),
        output=[
            {
                "type": "function_call",
                "id": "fc_done",
                "call_id": "call_fc_done",
                "name": "Bash",
                "arguments": {"command": "git status"},
            }
        ],
    )
    events = [
        SimpleNamespace(type="response.created"),
        SimpleNamespace(
            type="response.output_item.added",
            item=SimpleNamespace(
                type="function_call",
                id="fc_done",
                call_id="call_fc_done",
                name="Bash",
                arguments=None,
            ),
        ),
        SimpleNamespace(
            type="response.output_item.done",
            item=SimpleNamespace(
                type="function_call",
                id="fc_done",
                call_id="call_fc_done",
                name="Bash",
                arguments={"command": "git status"},
            ),
        ),
        SimpleNamespace(type="response.completed", response=response_obj),
    ]

    wrapper = AnthropicResponsesStreamWrapper(
        responses_stream=_make_stream(*events),
        model="gpt-5.3-codex-spark",
    )
    chunks = [chunk async for chunk in wrapper]

    assert [chunk["type"] for chunk in chunks] == [
        "message_start",
        "content_block_start",
        "content_block_delta",
        "content_block_stop",
        "message_delta",
        "message_stop",
    ]
    assert chunks[1]["content_block"] == {
        "type": "tool_use",
        "id": "call_fc_done",
        "name": "Bash",
        "input": {},
    }
    assert chunks[2]["delta"] == {
        "type": "input_json_delta",
        "partial_json": '{"command": "git status"}',
    }


@pytest.mark.asyncio
async def test_stream_wrapper_omits_empty_read_pages_argument_in_complete_json_delta():
    response_obj = SimpleNamespace(
        status="completed",
        usage=SimpleNamespace(
            input_tokens=12,
            output_tokens=7,
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
        ),
        output=[
            {
                "type": "function_call",
                "id": "fc_read",
                "call_id": "call_fc_read",
                "name": "Read",
                "arguments": {"file_path": "/tmp/example.py", "pages": ""},
            }
        ],
    )
    events = [
        SimpleNamespace(type="response.created"),
        SimpleNamespace(
            type="response.output_item.added",
            item=SimpleNamespace(
                type="function_call",
                id="fc_read",
                call_id="call_fc_read",
                name="Read",
                arguments=None,
            ),
        ),
        SimpleNamespace(
            type="response.function_call_arguments.delta",
            item_id="fc_read",
            delta='{"file_path": "/tmp/example.py", "pages": ""}',
        ),
        SimpleNamespace(
            type="response.output_item.done",
            item=SimpleNamespace(
                type="function_call",
                id="fc_read",
                call_id="call_fc_read",
                name="Read",
                arguments={"file_path": "/tmp/example.py", "pages": ""},
            ),
        ),
        SimpleNamespace(type="response.completed", response=response_obj),
    ]

    wrapper = AnthropicResponsesStreamWrapper(
        responses_stream=_make_stream(*events),
        model="gpt-5.5",
    )
    chunks = [chunk async for chunk in wrapper]

    assert chunks[1]["content_block"] == {
        "type": "tool_use",
        "id": "call_fc_read",
        "name": "Read",
        "input": {},
    }
    assert chunks[2]["delta"] == {
        "type": "input_json_delta",
        "partial_json": '{"file_path":"/tmp/example.py"}',
    }


@pytest.mark.asyncio
async def test_stream_wrapper_preserves_unrelated_empty_strings_in_read_delta():
    response_obj = SimpleNamespace(
        status="completed",
        usage=SimpleNamespace(
            input_tokens=12,
            output_tokens=7,
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
        ),
        output=[
            {
                "type": "function_call",
                "id": "fc_read_note",
                "call_id": "call_fc_read_note",
                "name": "Read",
                "arguments": {"file_path": "/tmp/example.py", "note": ""},
            }
        ],
    )
    events = [
        SimpleNamespace(type="response.created"),
        SimpleNamespace(
            type="response.output_item.added",
            item=SimpleNamespace(
                type="function_call",
                id="fc_read_note",
                call_id="call_fc_read_note",
                name="Read",
                arguments=None,
            ),
        ),
        SimpleNamespace(
            type="response.function_call_arguments.delta",
            item_id="fc_read_note",
            delta='{"file_path": "/tmp/example.py", "note": ""}',
        ),
        SimpleNamespace(
            type="response.output_item.done",
            item=SimpleNamespace(
                type="function_call",
                id="fc_read_note",
                call_id="call_fc_read_note",
                name="Read",
                arguments={"file_path": "/tmp/example.py", "note": ""},
            ),
        ),
        SimpleNamespace(type="response.completed", response=response_obj),
    ]

    wrapper = AnthropicResponsesStreamWrapper(
        responses_stream=_make_stream(*events),
        model="gpt-5.5",
    )
    chunks = [chunk async for chunk in wrapper]

    assert chunks[2]["delta"] == {
        "type": "input_json_delta",
        "partial_json": '{"file_path": "/tmp/example.py", "note": ""}',
    }


@pytest.mark.asyncio
async def test_stream_wrapper_omits_empty_read_pages_argument_from_split_deltas():
    response_obj = SimpleNamespace(
        status="completed",
        usage=SimpleNamespace(
            input_tokens=12,
            output_tokens=7,
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
        ),
        output=[
            {
                "type": "function_call",
                "id": "fc_read_split",
                "call_id": "call_fc_read_split",
                "name": "Read",
                "arguments": {"file_path": "/tmp/example.py", "pages": ""},
            }
        ],
    )
    events = [
        SimpleNamespace(type="response.created"),
        SimpleNamespace(
            type="response.output_item.added",
            item=SimpleNamespace(
                type="function_call",
                id="fc_read_split",
                call_id="call_fc_read_split",
                name="Read",
                arguments=None,
            ),
        ),
        SimpleNamespace(
            type="response.function_call_arguments.delta",
            item_id="fc_read_split",
            delta='{"file_path": "/tmp/example.py"',
        ),
        SimpleNamespace(
            type="response.function_call_arguments.delta",
            item_id="fc_read_split",
            delta=', "pages": ""}',
        ),
        SimpleNamespace(
            type="response.output_item.done",
            item=SimpleNamespace(
                type="function_call",
                id="fc_read_split",
                call_id="call_fc_read_split",
                name="Read",
                arguments={"file_path": "/tmp/example.py", "pages": ""},
            ),
        ),
        SimpleNamespace(type="response.completed", response=response_obj),
    ]

    wrapper = AnthropicResponsesStreamWrapper(
        responses_stream=_make_stream(*events),
        model="gpt-5.5",
    )
    chunks = [chunk async for chunk in wrapper]

    assert [chunk["type"] for chunk in chunks] == [
        "message_start",
        "content_block_start",
        "content_block_delta",
        "content_block_stop",
        "message_delta",
        "message_stop",
    ]
    assert chunks[2]["delta"] == {
        "type": "input_json_delta",
        "partial_json": '{"file_path":"/tmp/example.py"}',
    }


@pytest.mark.asyncio
async def test_stream_wrapper_emits_arguments_from_done_when_delta_is_absent():
    response_obj = SimpleNamespace(
        status="completed",
        usage=SimpleNamespace(
            input_tokens=12,
            output_tokens=7,
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
        ),
        output=[
            {
                "type": "function_call",
                "id": "fc_pwd",
                "call_id": "call_pwd",
                "name": "Bash",
                "arguments": '{"command":"pwd"}',
            }
        ],
    )
    events = [
        SimpleNamespace(type="response.created"),
        SimpleNamespace(
            type="response.output_item.added",
            item=SimpleNamespace(
                type="function_call",
                id="fc_pwd",
                call_id="call_pwd",
                name="Bash",
                arguments="",
            ),
        ),
        SimpleNamespace(
            type="response.function_call_arguments.done",
            item_id="fc_pwd",
            output_index=0,
            arguments='{"command":"pwd"}',
        ),
        SimpleNamespace(
            type="response.output_item.done",
            item=SimpleNamespace(
                type="function_call",
                id="fc_pwd",
                call_id="call_pwd",
                name="Bash",
            ),
        ),
        SimpleNamespace(type="response.completed", response=response_obj),
    ]

    wrapper = AnthropicResponsesStreamWrapper(
        responses_stream=_make_stream(*events),
        model="gpt-5.4",
    )
    chunks = [chunk async for chunk in wrapper]

    assert [chunk["type"] for chunk in chunks] == [
        "message_start",
        "content_block_start",
        "content_block_delta",
        "content_block_stop",
        "message_delta",
        "message_stop",
    ]
    assert chunks[2]["delta"] == {
        "type": "input_json_delta",
        "partial_json": '{"command":"pwd"}',
    }


@pytest.mark.asyncio
async def test_stream_wrapper_maps_codex_exec_command_deltas_back_to_bash():
    response_obj = SimpleNamespace(
        status="completed",
        usage=SimpleNamespace(
            input_tokens=12,
            output_tokens=7,
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
        ),
        output=[
            {
                "type": "function_call",
                "id": "fc_pwd",
                "call_id": "call_pwd",
                "name": "exec_command",
                "arguments": {
                    "cmd": "pwd",
                    "login": True,
                    "tty": False,
                    "yield_time_ms": 1000,
                    "max_output_tokens": 200,
                },
            }
        ],
    )
    events = [
        SimpleNamespace(type="response.created"),
        SimpleNamespace(
            type="response.output_item.added",
            item=SimpleNamespace(
                type="function_call",
                id="fc_pwd",
                call_id="call_pwd",
                name="exec_command",
                arguments=None,
            ),
        ),
        SimpleNamespace(
            type="response.function_call_arguments.delta",
            item_id="fc_pwd",
            delta='{"cmd":"p',
        ),
        SimpleNamespace(
            type="response.function_call_arguments.delta",
            item_id="fc_pwd",
            delta='wd","login":true,"tty":false,"yield_time_ms":1000,"max_output_tokens":200}',
        ),
        SimpleNamespace(
            type="response.output_item.done",
            item=SimpleNamespace(
                type="function_call",
                id="fc_pwd",
                call_id="call_pwd",
                name="exec_command",
                arguments={
                    "cmd": "pwd",
                    "login": True,
                    "tty": False,
                    "yield_time_ms": 1000,
                    "max_output_tokens": 200,
                },
            ),
        ),
        SimpleNamespace(type="response.completed", response=response_obj),
    ]

    wrapper = AnthropicResponsesStreamWrapper(
        responses_stream=_make_stream(*events),
        model="gpt-5.3-codex-spark",
        use_codex_native_tools=True,
    )
    chunks = [chunk async for chunk in wrapper]

    assert [chunk["type"] for chunk in chunks] == [
        "message_start",
        "content_block_start",
        "content_block_delta",
        "content_block_stop",
        "message_delta",
        "message_stop",
    ]
    assert chunks[1]["content_block"] == {
        "type": "tool_use",
        "id": "call_pwd",
        "name": "Bash",
        "input": {},
    }
    assert chunks[2]["delta"] == {
        "type": "input_json_delta",
        "partial_json": '{"command":"pwd"}',
    }


@pytest.mark.asyncio
async def test_stream_wrapper_estimates_message_start_usage_for_adapter_visibility():
    response_obj = SimpleNamespace(
        status="completed",
        usage=SimpleNamespace(
            input_tokens=21,
            output_tokens=9,
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
        ),
        output=[],
    )
    events = [
        SimpleNamespace(type="response.created"),
        SimpleNamespace(
            type="response.output_text.done",
            item_id="msg_provider_usage",
            output_index=0,
            text="provider usage smoke",
        ),
        SimpleNamespace(type="response.completed", response=response_obj),
    ]

    request_body = {
        "model": "grok-composer-2.5-fast",
        "input": "probe",
        "tools": [
            {
                "type": "function",
                "name": "Bash",
                "parameters": {
                    "type": "object",
                    "properties": {"command": {"type": "string"}},
                },
            }
        ],
    }
    wrapper = AnthropicResponsesStreamWrapper(
        responses_stream=_make_stream(*events),
        model="grok-composer-2.5-fast",
        request_body=request_body,
    )
    chunks = [chunk async for chunk in wrapper]

    message_start = chunks[0]
    message_start_usage = message_start["message"]["usage"]
    assert message_start_usage["input_tokens"] > 0
    assert message_start_usage["output_tokens"] == 0
    assert message_start_usage["cache_creation_input_tokens"] == 0
    assert message_start_usage["cache_read_input_tokens"] == 0
    assert "token_count" not in message_start_usage
    assert "cost" not in message_start_usage

    message_delta = next(chunk for chunk in chunks if chunk["type"] == "message_delta")
    assert message_delta["usage"] == {"input_tokens": 21, "output_tokens": 9}
    assert "token_count" not in message_delta["usage"]
    assert "cost" not in message_delta["usage"]
    assert request_body["litellm_metadata"][
        "anthropic_adapter_message_start_usage_source"
    ] == "estimated"
    assert (
        request_body["litellm_metadata"][
            "anthropic_adapter_message_start_usage_estimated"
        ]
        is True
    )
    assert request_body["litellm_metadata"][
        "anthropic_adapter_client_visible_usage_source"
    ] == "provider_reported"


@pytest.mark.asyncio
async def test_stream_wrapper_estimates_message_start_usage_before_tool_use():
    response_obj = SimpleNamespace(
        status="completed",
        usage=SimpleNamespace(
            input_tokens=34,
            output_tokens=12,
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
        ),
        output=[
            {
                "type": "function_call",
                "id": "fc_usage",
                "call_id": "call_fc_usage",
                "name": "Bash",
                "arguments": {"command": "pwd"},
            }
        ],
    )
    events = [
        SimpleNamespace(type="response.created"),
        SimpleNamespace(
            type="response.output_item.added",
            item=SimpleNamespace(
                type="function_call",
                id="fc_usage",
                call_id="call_fc_usage",
                name="Bash",
                arguments={"command": "pwd"},
            ),
        ),
        SimpleNamespace(
            type="response.output_item.done",
            item=SimpleNamespace(
                type="function_call",
                id="fc_usage",
                call_id="call_fc_usage",
                name="Bash",
                arguments={"command": "pwd"},
            ),
        ),
        SimpleNamespace(type="response.completed", response=response_obj),
    ]

    request_body = {
        "model": "grok-composer-2.5-fast",
        "input": "use the Bash tool",
        "tools": [
            {
                "type": "function",
                "name": "Bash",
                "parameters": {
                    "type": "object",
                    "properties": {"command": {"type": "string"}},
                    "required": ["command"],
                },
            }
        ],
    }
    wrapper = AnthropicResponsesStreamWrapper(
        responses_stream=_make_stream(*events),
        model="grok-composer-2.5-fast",
        request_body=request_body,
    )
    chunks = [chunk async for chunk in wrapper]

    assert chunks[0]["type"] == "message_start"
    assert chunks[0]["message"]["usage"]["input_tokens"] > 0
    assert "cost" not in chunks[0]["message"]["usage"]
    assert "token_count" not in chunks[0]["message"]["usage"]
    assert chunks[1]["content_block"] == {
        "type": "tool_use",
        "id": "call_fc_usage",
        "name": "Bash",
        "input": {"command": "pwd"},
    }
    message_delta = next(chunk for chunk in chunks if chunk["type"] == "message_delta")
    assert message_delta["delta"]["stop_reason"] == "tool_use"
    assert message_delta["usage"] == {"input_tokens": 34, "output_tokens": 12}


@pytest.mark.asyncio
async def test_stream_wrapper_preserves_provider_usage_on_completed_event():
    response_obj = SimpleNamespace(
        status="completed",
        usage=SimpleNamespace(
            input_tokens=21,
            output_tokens=9,
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
        ),
        output=[],
    )
    events = [
        SimpleNamespace(type="response.created"),
        SimpleNamespace(
            type="response.output_text.done",
            item_id="msg_provider_usage",
            output_index=0,
            text="provider usage smoke",
        ),
        SimpleNamespace(type="response.completed", response=response_obj),
    ]

    request_body = {"model": "openai/gpt-oss-120b:free", "input": "probe"}
    wrapper = AnthropicResponsesStreamWrapper(
        responses_stream=_make_stream(*events),
        model="openai/gpt-oss-120b:free",
        request_body=request_body,
    )
    chunks = [chunk async for chunk in wrapper]

    message_delta = next(chunk for chunk in chunks if chunk["type"] == "message_delta")
    assert message_delta["usage"] == {"input_tokens": 21, "output_tokens": 9}
    assert request_body["litellm_metadata"][
        "anthropic_adapter_client_visible_usage_source"
    ] == "provider_reported"
    assert (
        request_body["litellm_metadata"][
            "anthropic_adapter_client_visible_usage_estimated"
        ]
        is False
    )


@pytest.mark.asyncio
async def test_stream_wrapper_preserves_zero_provider_usage_on_completed_event():
    response_obj = SimpleNamespace(
        status="completed",
        usage=SimpleNamespace(
            input_tokens=0,
            output_tokens=0,
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
        ),
        output=[],
    )
    events = [
        SimpleNamespace(type="response.created"),
        SimpleNamespace(
            type="response.output_text.done",
            item_id="msg_zero_provider_usage",
            output_index=0,
            text="provider zero usage smoke",
        ),
        SimpleNamespace(type="response.completed", response=response_obj),
    ]

    request_body = {"model": "openai/gpt-oss-120b:free", "input": "probe"}
    wrapper = AnthropicResponsesStreamWrapper(
        responses_stream=_make_stream(*events),
        model="openai/gpt-oss-120b:free",
        request_body=request_body,
    )
    chunks = [chunk async for chunk in wrapper]

    message_delta = next(chunk for chunk in chunks if chunk["type"] == "message_delta")
    assert message_delta["usage"] == {"input_tokens": 0, "output_tokens": 0}
    assert request_body["litellm_metadata"][
        "anthropic_adapter_client_visible_usage_source"
    ] == "provider_reported"
    assert (
        request_body["litellm_metadata"][
            "anthropic_adapter_client_visible_usage_estimated"
        ]
        is False
    )


@pytest.mark.asyncio
async def test_stream_wrapper_estimates_usage_on_completed_event_when_provider_usage_missing():
    response_obj = SimpleNamespace(
        status="completed",
        usage=None,
        output=[],
    )
    events = [
        SimpleNamespace(type="response.created"),
        SimpleNamespace(
            type="response.output_text.done",
            item_id="msg_missing_usage",
            output_index=0,
            text="estimated usage smoke",
        ),
        SimpleNamespace(type="response.completed", response=response_obj),
    ]

    request_body = {"model": "openai/gpt-oss-120b:free", "input": "probe"}
    wrapper = AnthropicResponsesStreamWrapper(
        responses_stream=_make_stream(*events),
        model="openai/gpt-oss-120b:free",
        request_body=request_body,
    )
    chunks = [chunk async for chunk in wrapper]

    message_delta = next(chunk for chunk in chunks if chunk["type"] == "message_delta")
    assert message_delta["usage"]["input_tokens"] >= 1
    assert message_delta["usage"]["output_tokens"] >= 1
    assert request_body["litellm_metadata"][
        "anthropic_adapter_client_visible_usage_source"
    ] == "estimated"
    assert (
        request_body["litellm_metadata"][
            "anthropic_adapter_client_visible_usage_estimated"
        ]
        is True
    )


@pytest.mark.asyncio
async def test_stream_wrapper_synthesizes_stop_and_usage_when_completed_is_missing():
    events = [
        SimpleNamespace(type="response.created"),
        SimpleNamespace(
            type="response.output_text.done",
            item_id="msg_missing_completed",
            output_index=0,
            text="oss120 smoke",
        ),
    ]

    wrapper = AnthropicResponsesStreamWrapper(
        responses_stream=_make_stream(*events),
        model="openai/gpt-oss-120b:free",
        request_body={"model": "openai/gpt-oss-120b:free", "input": "probe"},
    )
    chunks = [chunk async for chunk in wrapper]

    assert [chunk["type"] for chunk in chunks] == [
        "message_start",
        "content_block_start",
        "content_block_delta",
        "message_delta",
        "message_stop",
    ]
    assert chunks[2]["delta"] == {
        "type": "text_delta",
        "text": "oss120 smoke",
    }
    assert chunks[3]["usage"]["input_tokens"] >= 1
    assert chunks[3]["usage"]["output_tokens"] >= 1


@pytest.mark.asyncio
async def test_stream_wrapper_rejects_empty_success_when_enabled():
    response_obj = SimpleNamespace(
        id="resp_empty",
        status="completed",
        model="google/gemma-4-31b-it:free",
        usage=SimpleNamespace(
            input_tokens=0,
            output_tokens=0,
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
        ),
        output=[],
    )
    events = [
        SimpleNamespace(type="response.created"),
        SimpleNamespace(type="response.completed", response=response_obj),
    ]

    wrapper = AnthropicResponsesStreamWrapper(
        responses_stream=_make_stream(*events),
        model="google/gemma-4-31b-it:free",
        request_body={
            "model": "google/gemma-4-31b-it:free",
            "stream": True,
        },
        reject_empty_success=True,
    )

    with pytest.raises(AnthropicResponsesEmptySuccessError) as exc_info:
        [chunk async for chunk in wrapper]

    assert "empty successful response" in str(exc_info.value)
    assert "resp_empty" in str(exc_info.value)
