from types import SimpleNamespace

import pytest

from litellm.llms.anthropic.experimental_pass_through.responses_adapters.streaming_iterator import (
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
