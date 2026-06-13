from litellm.litellm_core_utils.prompt_templates.factory import (
    convert_to_gemini_tool_call_invoke,
)
from litellm.llms.vertex_ai.gemini.vertex_and_google_ai_studio_gemini import (
    VertexGeminiConfig,
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


def test_vertex_transform_parts_preserves_function_call_id():
    _, tools, next_index = VertexGeminiConfig._transform_parts(
        [
            {
                "functionCall": {
                    "name": "read_file",
                    "args": {"file_path": "/tmp/a.txt"},
                    "id": "toolu_read_1",
                }
            }
        ],
        cumulative_tool_call_idx=0,
        is_function_call=False,
    )

    assert next_index == 1
    assert tools == [
        {
            "id": "toolu_read_1",
            "type": "function",
            "function": {
                "name": "read_file",
                "arguments": '{"file_path": "/tmp/a.txt"}',
            },
            "index": 0,
        }
    ]


def test_vertex_transform_parts_generates_function_call_id_when_missing():
    _, tools, next_index = VertexGeminiConfig._transform_parts(
        [
            {
                "functionCall": {
                    "name": "read_file",
                    "args": {"file_path": "/tmp/a.txt"},
                }
            }
        ],
        cumulative_tool_call_idx=0,
        is_function_call=False,
    )

    assert next_index == 1
    assert tools is not None
    assert len(tools) == 1
    assert tools[0]["id"].startswith("call_")
    assert tools[0]["function"]["name"] == "read_file"
