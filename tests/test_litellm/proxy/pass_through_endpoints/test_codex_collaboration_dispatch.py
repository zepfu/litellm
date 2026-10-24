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
def test_should_normalize_declaration_after_initial_user_message(declaration, declaration_index):
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


@pytest.mark.parametrize("conflict", ["duplicate", "top_level_tools"])
def test_should_reject_ambiguous_declarations(declaration, conflict):
    body = {"input": [{"type": "message", "role": "user", "content": []}, declaration]}
    if conflict == "duplicate":
        body["input"].append(deepcopy(declaration))
    else:
        body["tools"] = []

    with pytest.raises(HTTPException) as raised:
        normalize_codex_collaboration_dispatch_body(body)

    assert raised.value.status_code == 409
    assert raised.value.detail["error"]["code"] == "aawm_codex_assignment_unreadable"
