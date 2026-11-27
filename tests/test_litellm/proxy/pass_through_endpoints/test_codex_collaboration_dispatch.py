"""Stock Codex tool declarations must reach collaboration normalization."""

from copy import deepcopy

import pytest
from fastapi import HTTPException

from litellm.proxy.pass_through_endpoints.aawm_adapter_runtime.codex_collaboration_dispatch import (
    normalize_codex_collaboration_dispatch_body,
)


@pytest.fixture
def declaration():
    return {
        "type": "additional_tools",
        "role": "developer",
        "tools": [
            {
                "type": "namespace",
                "name": "collaboration",
                "tools": [
                    {
                        "type": "function",
                        "name": "spawn_agent",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "task_name": {"type": "string"},
                                "message": {"type": "string", "encrypted": True},
                            },
                            "required": ["task_name", "message"],
                        },
                    }
                ],
            }
        ],
    }


@pytest.mark.parametrize("declaration_index", [0, 1])
def test_should_normalize_declaration_after_initial_user_message(
    declaration, declaration_index
):
    message = {
        "type": "message",
        "role": "user",
        "content": [{"type": "input_text", "text": "Inspect the renderer."}],
    }
    body = {"input": [message]}
    body["input"].insert(declaration_index, declaration)
    original = deepcopy(body)

    result = normalize_codex_collaboration_dispatch_body(body)

    assert result["input"] == [message]
    schema = result["tools"][0]["tools"][0]["parameters"]["properties"]["message"]
    assert "encrypted" not in schema
    assert '{"cfg047":1,"encoding":"text"' in schema["description"]
    assert body == original


def test_should_normalize_latest_accumulated_declaration(declaration):
    stale = {
        "type": "additional_tools",
        "id": "at_stale",
        "role": "developer",
        "tools": [],
    }
    latest = {
        "type": "additional_tools",
        "id": "at_latest",
        "role": "developer",
        "tools": [declaration["tools"][0]],
    }
    user_message = {
        "type": "message",
        "role": "user",
        "content": [{"type": "input_text", "text": "Continue with the same tools."}],
    }
    body = {
        "model": "codex-auto-agent",
        "tools": [{"type": "web_search", "external_web_access": True}],
        "input": [stale, user_message, latest],
    }

    result = normalize_codex_collaboration_dispatch_body(body)

    assert result["input"] == [user_message]
    assert result["tools"][0] == {"type": "web_search", "external_web_access": True}
    assert result["tools"][1]["name"] == declaration["tools"][0]["name"]
    schema = (
        result["tools"][1]["tools"][0]["parameters"]["properties"]["message"]
    )
    assert "encrypted" not in schema
    assert '{"cfg047":1,"encoding":"text"' in schema["description"]

    repeated = normalize_codex_collaboration_dispatch_body(result)

    assert repeated == result


def _catalog_notice(kind, text):
    return {
        "type": "message",
        "role": "developer",
        "content": [{"type": "input_text", "text": text}],
        "internal_chat_message_metadata_passthrough": {
            "content_item_kinds": [kind],
        },
    }


def test_should_keep_execution_tools_after_single_collaboration_tool_update(declaration):
    delta = deepcopy(declaration)
    spawn = delta["tools"][0]["tools"][0]
    spawn["parameters"]["properties"]["task_name"]["description"] = "Updated guidance."
    declaration["tools"][0]["tools"].append(
        {"type": "function", "name": "wait_agent", "parameters": {}}
    )
    declaration["tools"].extend(
        [
            {
                "type": "namespace",
                "name": "functions",
                "tools": [
                    {"type": "function", "name": name, "parameters": {}}
                    for name in ("exec", "wait")
                ],
            },
            {
                "type": "namespace",
                "name": "clock",
                "tools": [{"type": "function", "name": "sleep", "parameters": {}}],
            },
        ]
    )
    notice = _catalog_notice(
        "tools.incremental_update",
        "This is an incremental tools update. Previously declared tools remain "
        "available for direct calls unless explicitly marked unavailable. If a tool "
        "is redefined here, its latest definition replaces the earlier one.",
    )
    body = {"input": [declaration, notice, delta]}
    original = deepcopy(body)

    result = normalize_codex_collaboration_dispatch_body(body)

    assert result["input"] == [notice]
    collaboration, functions, clock = result["tools"]
    assert [tool["name"] for tool in collaboration["tools"]] == [
        "spawn_agent", "wait_agent"
    ]
    properties = collaboration["tools"][0]["parameters"]["properties"]
    assert properties["task_name"]["description"] == "Updated guidance."
    assert '{"cfg047":1,"encoding":"text"' in properties["message"]["description"]
    assert functions == declaration["tools"][1]
    assert clock == declaration["tools"][2]
    assert normalize_codex_collaboration_dispatch_body(result) == result
    assert body == original


def test_should_apply_stock_catalog_withdrawals_before_later_declarations(declaration):
    exec_tool = {"type": "function", "name": "exec", "parameters": {}}
    wait_tool = {"type": "function", "name": "wait", "parameters": {}}
    functions = {"type": "namespace", "name": "functions", "tools": [exec_tool, wait_tool]}
    declaration["tools"].append(functions)
    notice = _catalog_notice(
        "tools.removed_definition",
        "The following namespaces are no longer available. Do not call tools in "
        "them unless those tools are declared in a later update:\n- collaboration\n\n"
        "The following tools are no longer available. Do not call them:\n- functions.exec",
    )
    reinstated = {
        "type": "additional_tools",
        "role": "developer",
        "tools": [deepcopy(declaration["tools"][0])],
    }
    declaration["tools"][0]["tools"].append(
        {"type": "function", "name": "wait_agent", "parameters": {}}
    )

    result = normalize_codex_collaboration_dispatch_body(
        {"input": [declaration, notice, reinstated]}
    )

    assert result["input"] == [notice]
    assert result["tools"][0] == {**functions, "tools": [wait_tool]}
    assert result["tools"][1]["name"] == "collaboration"
    assert [tool["name"] for tool in result["tools"][1]["tools"]] == ["spawn_agent"]


def test_should_use_current_standard_tools_after_forking_responses_lite(declaration):
    message = {"type": "message", "role": "user", "content": []}
    collaboration = deepcopy(declaration["tools"][0])
    declaration["tools"].append(
        {"type": "function", "name": "parent_tool", "parameters": {"type": "object"}}
    )
    current_tools = [
        {
            "type": "function",
            "name": "exec_command",
            "parameters": {
                "type": "object",
                "properties": {"cmd": {"type": "string"}},
                "required": ["cmd"],
            },
        },
        collaboration,
    ]
    body = {"input": [declaration, message], "tools": current_tools}
    original = deepcopy(body)

    result = normalize_codex_collaboration_dispatch_body(body)

    assert result["input"] == [message]
    assert [tool["name"] for tool in result["tools"]] == [
        "exec_command",
        "collaboration",
    ]
    assert result["tools"][0] == current_tools[0]
    schema = result["tools"][1]["tools"][0]["parameters"]["properties"]["message"]
    assert "encrypted" not in schema
    assert '{"cfg047":1,"encoding":"text"' in schema["description"]
    assert body == original


@pytest.mark.parametrize(
    "tools",
    [
        [{"type": "custom", "name": "unsupported"}],
        [
            {"type": "function", "name": "exec_command", "parameters": {}},
            {"type": "unknown"},
        ],
    ],
)
def test_should_reject_conflicting_top_level_tools(declaration, tools):
    body = {
        "input": [{"type": "message", "role": "user", "content": []}, declaration],
        "tools": tools,
    }

    with pytest.raises(HTTPException) as raised:
        normalize_codex_collaboration_dispatch_body(body)

    assert raised.value.status_code == 409
    assert raised.value.detail["error"]["code"] == "aawm_codex_assignment_unreadable"


@pytest.mark.parametrize("message_type", ["NEW_TASK", "MESSAGE"])
@pytest.mark.parametrize(
    "payload",
    [
        "gAAAAABn" + "A" * 200,
        "aawm_erp:eyJwcm92aWRlciI6Im9wZW5haSJ9;gAAAAABn" + "A" * 200,
        "litellm_enc:eyJtb2RlbF9pZCI6ImdwdC02LjEifQ==;gAAAAABn" + "A" * 200,
    ],
)
def test_should_preserve_provider_owned_opaque_agent_message(message_type, payload):
    """Native Codex ciphertext must reach its route without proxy reinterpretation."""
    body = {
        "input": [
            {
                "type": "agent_message",
                "author": "/root",
                "recipient": "/root/child",
                "content": [
                    {
                        "type": "input_text",
                        "text": (
                            f"Message Type: {message_type}\n"
                            "Task name: /root/child\n"
                            "Sender: /root\n"
                            "Payload:\n"
                        ),
                    },
                    {"type": "encrypted_content", "encrypted_content": payload},
                ],
            }
        ]
    }

    result = normalize_codex_collaboration_dispatch_body(body)

    assert result is body
    assert result["input"][0]["content"][1]["encrypted_content"] == payload


@pytest.mark.parametrize("message_type", ["NEW_TASK", "MESSAGE"])
def test_should_preserve_provider_owned_opaque_single_part_agent_message(message_type):
    payload = "aawm_erp:eyJwcm92aWRlciI6Im9wZW5haSJ9;gAAAAABn" + "A" * 200
    body = {
        "input": [
            {
                "type": "agent_message",
                "author": "/root",
                "recipient": "/root/child",
                "content": [
                    {
                        "type": "input_text",
                        "text": (
                            f"Message Type: {message_type}\n"
                            "Task name: /root/child\n"
                            "Sender: /root\n"
                            f"Payload:\n{payload}"
                        ),
                    }
                ],
            }
        ]
    }

    result = normalize_codex_collaboration_dispatch_body(body)

    assert result is body
    assert result["input"][0]["content"][0]["text"].endswith(payload)


@pytest.mark.parametrize("message_type", ["NEW_TASK", "MESSAGE"])
def test_should_materialize_readable_plaintext_agent_message(message_type):
    """The stock collaboration layer sends readable text in encrypted_content."""
    payload = "Review the current published revision and report any required remediation."
    header = (
        f"Message Type: {message_type}\n"
        "Task name: /root/child\n"
        "Sender: /root\n"
        "Payload:\n"
    )
    body = {
        "input": [
            {
                "type": "agent_message",
                "author": "/root",
                "recipient": "/root/child",
                "content": [
                    {"type": "input_text", "text": header},
                    {"type": "encrypted_content", "encrypted_content": payload},
                ],
            }
        ]
    }

    result = normalize_codex_collaboration_dispatch_body(body)

    assert result is not body
    assert result["input"][0]["content"] == [
        {"type": "input_text", "text": header + payload}
    ]
