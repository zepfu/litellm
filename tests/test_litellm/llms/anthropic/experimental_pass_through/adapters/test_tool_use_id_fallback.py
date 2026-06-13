from litellm.llms.anthropic.experimental_pass_through.adapters.transformation import (
    LiteLLMAnthropicMessagesAdapter,
)
from litellm.types.llms.openai import ChatCompletionAssistantToolCall
from litellm.types.utils import Choices, Function, Message, ModelResponse, Usage


def test_translate_anthropic_messages_to_openai_repairs_tool_use_id_from_result():
    adapter = LiteLLMAnthropicMessagesAdapter()

    result = adapter.translate_anthropic_messages_to_openai(
        messages=[
            {"role": "user", "content": "Read a file."},
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "I will read it."},
                    {
                        "type": "tool_use",
                        "name": "Read",
                        "input": {"file_path": "/tmp/example.py"},
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_read_1",
                        "content": "alpha",
                    }
                ],
            },
        ]
    )

    assert result[1]["tool_calls"][0]["id"] == "toolu_read_1"
    assert result[2]["tool_call_id"] == "toolu_read_1"


def test_translate_anthropic_messages_to_openai_pairs_tool_result_after_intervening_message():
    adapter = LiteLLMAnthropicMessagesAdapter()

    result = adapter.translate_anthropic_messages_to_openai(
        messages=[
            {"role": "user", "content": "Read a file."},
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "I will read it."},
                    {
                        "type": "tool_use",
                        "name": "Read",
                        "input": {"file_path": "/tmp/example.py"},
                    },
                ],
            },
            {"role": "user", "content": [{"type": "text", "text": "continue"}]},
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_read_late",
                        "content": "alpha",
                    }
                ],
            },
        ]
    )

    tool_messages = [message for message in result if message.get("role") == "tool"]
    assert result[1]["tool_calls"][0]["id"] == "toolu_read_late"
    assert tool_messages[0]["tool_call_id"] == "toolu_read_late"


def test_translate_openai_response_generates_missing_tool_use_id():
    response = ModelResponse(
        id="test-id",
        choices=[
            Choices(
                index=0,
                finish_reason="tool_calls",
                message=Message(
                    role="assistant",
                    content=None,
                    tool_calls=[
                        ChatCompletionAssistantToolCall(
                            id="",
                            type="function",
                            function=Function(
                                name="read_file",
                                arguments='{"file_path": "/tmp/example.py"}',
                            ),
                        )
                    ],
                ),
            )
        ],
        model="gemini-3.1-pro-preview",
        usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
    )

    adapter = LiteLLMAnthropicMessagesAdapter()
    result = adapter.translate_openai_response_to_anthropic(
        response=response,
        tool_name_mapping={"read_file": "Read"},
    )

    tool_use_blocks = [c for c in result["content"] if c.get("type") == "tool_use"]
    assert len(tool_use_blocks) == 1
    assert tool_use_blocks[0]["id"].startswith("call_")
    assert tool_use_blocks[0]["name"] == "Read"
