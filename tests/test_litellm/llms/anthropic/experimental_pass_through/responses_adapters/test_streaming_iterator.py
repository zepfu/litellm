from types import SimpleNamespace

import pytest

from litellm.llms.anthropic.experimental_pass_through.responses_adapters.streaming_iterator import (
    AnthropicResponsesEmptySuccessError,
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
