from litellm.litellm_core_utils.prompt_templates.factory import (
    convert_to_gemini_tool_call_invoke,
)


def test_convert_to_gemini_tool_call_invoke_preserves_claude_tool_call_id():
    parts = convert_to_gemini_tool_call_invoke(
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "toolu_read_1",
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "arguments": '{"file_path": "/tmp/a.txt"}',
                    },
                }
            ],
        },
        model="claude-sonnet-4-6",
    )

    assert parts == [
        {
            "function_call": {
                "name": "read_file",
                "args": {"file_path": "/tmp/a.txt"},
                "id": "toolu_read_1",
            }
        }
    ]


def test_convert_to_gemini_tool_call_invoke_keeps_gemini_payload_without_id():
    parts = convert_to_gemini_tool_call_invoke(
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "toolu_read_1",
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "arguments": '{"file_path": "/tmp/a.txt"}',
                    },
                }
            ],
        },
        model="gemini-3-flash-preview",
    )

    assert parts[0]["function_call"] == {
        "name": "read_file",
        "args": {"file_path": "/tmp/a.txt"},
    }
    assert "id" not in parts[0]["function_call"]
