"""OPENAI-007: direct OpenAI legacy-history function_call id normalization."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from litellm.proxy.pass_through_endpoints.aawm_adapter_runtime import (
    codex_candidate_calls,
)
from litellm.proxy.pass_through_endpoints.aawm_adapter_runtime.direct_openai_function_call_history import (
    normalize_direct_openai_legacy_function_call_history_ids,
)
from litellm.responses.litellm_completion_transformation.function_call_identity import (
    is_native_responses_function_call_item_id,
    resolve_responses_function_call_identity,
)

# Incident tool/call id that arrived as a non-native function_call item id.
INCIDENT_CALL_ID = "tool_LFt0kECrDyUHYJhFZ2Mh84fc"
NATIVE_FC_ID = "fc_685c42deefc0819a822b6936faaa30be0c76bc1491ab6619"
NATIVE_FC_UUID_ID = "fc_1fe70e2a-a596-45ef-b72c-9b8567c460e5"


def _expected_item_id(call_id: str) -> str:
    item_id, preserved_call_id = resolve_responses_function_call_identity(call_id)
    assert preserved_call_id == call_id
    return item_id


def test_helper_rewrites_only_malformed_function_call_item_ids():
    call_id = INCIDENT_CALL_ID
    expected_item_id = _expected_item_id(call_id)
    nested_unrelated = {
        "type": "message",
        "id": "msg_nested_should_stay",
        "content": [
            {
                "type": "output_text",
                "text": "x",
                "id": "nested_id_must_not_change",
            }
        ],
    }
    body = {
        "model": "gpt-5",
        "input": [
            {
                "type": "function_call",
                "id": call_id,  # malformed: provider tool id collapsed into item id
                "call_id": call_id,
                "name": "Bash",
                "arguments": "{}",
            },
            {
                "type": "function_call",
                "id": NATIVE_FC_ID,
                "call_id": "call_native_hex",
                "name": "Read",
                "arguments": "{}",
            },
            {
                "type": "function_call",
                "id": NATIVE_FC_UUID_ID,
                "call_id": "call_native_uuid",
                "name": "Write",
                "arguments": "{}",
            },
            {
                "type": "function_call_output",
                "id": call_id,  # must not rewrite outputs
                "call_id": call_id,
                "output": "ok",
            },
            nested_unrelated,
            {
                "type": "function_call",
                "id": "fc_1",  # short non-native placeholder
                "call_id": " call_ws_preserve ",
                "name": "Exec",
                "arguments": "{}",
            },
        ],
        "metadata": {"id": "top_level_meta_id_untouched"},
    }

    out = normalize_direct_openai_legacy_function_call_history_ids(
        body
    )

    assert out is not body
    assert out["metadata"] == body["metadata"]
    assert out["metadata"]["id"] == "top_level_meta_id_untouched"

    items = out["input"]
    assert items[0]["type"] == "function_call"
    assert items[0]["call_id"] == call_id
    assert items[0]["id"] == expected_item_id
    assert items[0]["id"] != items[0]["call_id"]
    assert is_native_responses_function_call_item_id(items[0]["id"])

    # Valid native fc_* ids preserved (contiguous hex and UUID-shaped).
    assert items[1]["id"] == NATIVE_FC_ID
    assert items[1]["call_id"] == "call_native_hex"
    assert items[2]["id"] == NATIVE_FC_UUID_ID
    assert items[2]["call_id"] == "call_native_uuid"

    # function_call_output keeps its own id/call_id untouched.
    assert items[3] == body["input"][3]
    assert items[3]["id"] == call_id
    assert items[3]["call_id"] == call_id

    # Unrelated item types and nested ids are untouched.
    assert items[4] is nested_unrelated or items[4] == nested_unrelated
    assert items[4]["id"] == "msg_nested_should_stay"
    assert items[4]["content"][0]["id"] == "nested_id_must_not_change"

    # Non-native short id rewritten from call_id; call_id byte-for-byte.
    spaced_call_id = " call_ws_preserve "
    assert items[5]["call_id"] == spaced_call_id
    assert items[5]["id"] == _expected_item_id(spaced_call_id)

    # Original request body is not mutated in place.
    assert body["input"][0]["id"] == call_id


def test_helper_noop_when_no_function_call_rewrite_needed():
    body = {
        "input": [
            {
                "type": "function_call",
                "id": NATIVE_FC_ID,
                "call_id": "call_ok",
                "name": "Bash",
                "arguments": "{}",
            },
            {"type": "message", "role": "user", "content": "hi"},
        ]
    }
    out = normalize_direct_openai_legacy_function_call_history_ids(
        body
    )
    assert out is body



def _install_native_openai_host(pass_through: Any) -> dict[str, Any]:
    host: dict[str, Any] = {"__builtins__": __builtins__}
    host["pass_through_request"] = pass_through
    host["BaseOpenAIPassThroughHandler"] = MagicMock()
    host["BaseOpenAIPassThroughHandler"]._assemble_headers = MagicMock(
        return_value={"Authorization": "Bearer test"}
    )
    host["_AAWM_ALIAS_CANDIDATE_RETRYABLE_UPSTREAM_STATUS_CODES_DEFAULT"] = [429, 500]
    host["litellm"] = MagicMock()
    host["litellm"].LlmProviders.OPENAI.value = "openai"
    host["_codex_native_openai_candidate_unavailable_detail"] = MagicMock(
        return_value=None
    )
    host["_raise_codex_native_openai_auto_agent_candidate_unavailable"] = MagicMock()
    from fastapi import Request
    from fastapi.responses import Response

    host["Request"] = Request
    host["Response"] = Response
    host["Optional"] = __import__("typing").Optional
    host["Any"] = Any
    host["dict"] = dict
    host["str"] = str
    host["bool"] = bool
    host["list"] = list
    codex_candidate_calls.install(host)
    return host


@pytest.mark.asyncio
async def test_direct_native_openai_request_normalizes_before_pass_through():
    """Incident path: rewrite malformed function_call id before pass_through."""
    call_id = INCIDENT_CALL_ID
    expected_item_id = _expected_item_id(call_id)
    request_body: dict[str, Any] = {
        "model": "gpt-5.1",
        "input": [
            {
                "type": "function_call",
                "id": call_id,
                "call_id": call_id,
                "name": "Bash",
                "arguments": "{\"command\":\"pwd\"}",
            },
            {
                "type": "function_call_output",
                "id": "should_not_change",
                "call_id": call_id,
                "output": "/tmp",
            },
            {
                "type": "function_call",
                "id": NATIVE_FC_UUID_ID,
                "call_id": "call_keep_native",
                "name": "Read",
                "arguments": "{}",
            },
        ],
    }
    original_input = [dict(item) for item in request_body["input"]]

    captured: dict[str, Any] = {}
    mock_response = MagicMock(name="pass_through_response")

    async def _fake_pass_through_request(**kwargs: Any):
        captured.update(kwargs)
        return mock_response

    host = _install_native_openai_host(AsyncMock(side_effect=_fake_pass_through_request))
    fn = host["_perform_codex_auto_agent_native_openai_request"]

    response = await fn(
        request=MagicMock(),
        fastapi_response=MagicMock(),
        user_api_key_dict=MagicMock(),
        target_url="https://api.openai.com/v1/responses",
        api_key="sk-test",
        forward_headers=False,
        request_body=request_body,
    )

    assert response is mock_response
    host["pass_through_request"].assert_awaited_once()
    custom_body = captured["custom_body"]
    assert custom_body is not request_body
    assert custom_body["input"][0]["id"] == expected_item_id

    fc_item = custom_body["input"][0]
    assert fc_item["type"] == "function_call"
    assert fc_item["call_id"] == call_id
    assert fc_item["id"] == expected_item_id
    assert is_native_responses_function_call_item_id(fc_item["id"])

    output_item = custom_body["input"][1]
    assert output_item["type"] == "function_call_output"
    assert output_item["id"] == "should_not_change"
    assert output_item["call_id"] == call_id

    native_item = custom_body["input"][2]
    assert native_item["id"] == NATIVE_FC_UUID_ID
    assert native_item["call_id"] == "call_keep_native"

    # Caller-supplied body input entries keep pre-normalization values.
    assert request_body["input"][0]["id"] == original_input[0]["id"] == call_id
    assert captured["custom_llm_provider"] == "openai"
    assert captured["expected_target_family"] == "openai"
