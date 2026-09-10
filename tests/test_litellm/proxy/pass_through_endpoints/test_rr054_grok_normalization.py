from copy import deepcopy
import json
from typing import Any, Optional

import pytest

from litellm.llms.anthropic.experimental_pass_through.providers.grok import (
    normalization,
)
from litellm.proxy.pass_through_endpoints.providers.xai.request_prep import (
    _coerce_grok_native_function_call_arguments_value,
    _get_anthropic_grok_normalization_runtime,
    _rewrite_grok_native_unsupported_input_items_from_request_body,
    _sanitize_grok_native_function_call_arguments_request_body,
)


@pytest.mark.parametrize(
    "arguments_value,expected_arguments,expected_reason",
    [
        ({"cmd": "pwd"}, {"cmd": "pwd"}, None),
        (None, {}, "missing"),
        ("", {}, "empty"),
        ('{"cmd":"pwd"}', {"cmd": "pwd"}, "parsed_json_string"),
        ("[]", {}, "non_object_json"),
        ("not-json", {}, "invalid_json"),
        (["pwd"], {}, "unsupported_type"),
        (7, {}, "unsupported_type"),
    ],
)
def test_should_preserve_argument_coercion_delegate_parity(
    arguments_value: object,
    expected_arguments: dict[str, Any],
    expected_reason: Optional[str],
) -> None:
    package_result = normalization.coerce_function_call_arguments_value(
        arguments_value
    )
    delegate_result = _coerce_grok_native_function_call_arguments_value(
        arguments_value
    )

    assert package_result == delegate_result
    assert package_result == (expected_arguments, expected_reason)


def test_should_preserve_function_call_argument_sanitization_delegate_parity() -> None:
    request_body: dict[str, Any] = {
        "model": "grok-composer-2.5-fast",
        "input": [
            {"type": "message", "role": "user", "content": "continue"},
            {
                "type": "function_call",
                "name": " exec_command ",
                "call_id": " call_1 ",
                "arguments": '{"cmd":"pwd"}',
            },
            {
                "type": "function_call",
                "name": "broken",
                "call_id": "call_2",
                "arguments": "not-json",
            },
            {
                "type": "function_call_output",
                "call_id": "call_2",
                "output": "failed",
            },
        ],
    }

    package_result = normalization.sanitize_function_call_arguments_request_body(
        deepcopy(request_body)
    )
    delegate_result = _sanitize_grok_native_function_call_arguments_request_body(
        deepcopy(request_body)
    )

    assert package_result == delegate_result
    updated_body, changes = package_result
    assert updated_body["input"][1]["arguments"] == {"cmd": "pwd"}
    assert updated_body["input"][2]["arguments"] == {}
    assert changes == [
        {
            "type": "function_call",
            "index": 1,
            "call_id": "call_1",
            "name": "exec_command",
            "reason": "parsed_json_string",
        },
        {
            "type": "function_call",
            "index": 2,
            "call_id": "call_2",
            "name": "broken",
            "reason": "invalid_json",
        },
    ]


@pytest.mark.parametrize("anthropic_adapter", [False, True])
def test_should_preserve_unsupported_input_rewrite_delegate_parity(
    anthropic_adapter: bool,
) -> None:
    request_body: dict[str, Any] = {
        "model": "grok-composer-2.5-fast",
        "input": [
            {"type": "message", "role": "user", "content": "continue"},
            {
                "type": "function_call",
                "name": "exec_command",
                "call_id": "call_1",
                "arguments": {"cmd": "pwd"},
            },
            {
                "type": "function_call_output",
                "call_id": "call_1",
                "output": {"status": "ok"},
            },
        ],
    }
    if anthropic_adapter:
        request_body["litellm_metadata"] = {
            "route_family": "anthropic_grok_native_responses_adapter"
        }

    package_result = normalization.rewrite_unsupported_input_items_from_request_body(
        _get_anthropic_grok_normalization_runtime(),
        deepcopy(request_body),
    )
    delegate_result = _rewrite_grok_native_unsupported_input_items_from_request_body(
        deepcopy(request_body)
    )

    assert package_result == delegate_result
    updated_body, rewritten_items = package_result
    assert [item["role"] for item in updated_body["input"][1:]] == [
        "assistant",
        "user",
    ]
    assert [item["type"] for item in rewritten_items] == [
        "function_call",
        "function_call_output",
    ]
    if anthropic_adapter:
        assert "Correlation ref:" not in updated_body["input"][1]["content"]
        assert "call_id_hash" in rewritten_items[0]
        assert "call_id" not in rewritten_items[0]
    else:
        assert "Correlation ref: call_1" in updated_body["input"][1]["content"]
        assert rewritten_items[0]["call_id"] == "call_1"


def test_grok_history_rewrite_dialect_is_detected_but_not_repaired() -> None:
    """LiteLLM flattens native function_call history into Tool label text.

    Today's Grok Build dumps used that same dialect as assistant output.
    Repair refuses any block that still has the Context note prefix, so the
    text stays native instead of becoming a structured function_call.
    """
    from litellm.integrations.aawm_agent_quality_rules import (
        is_malformed_grok_literal_tool_label_transcript_text,
    )
    from litellm.proxy.pass_through_endpoints.aawm_adapter_runtime.request_build import (
        _repair_grok_composer_literal_tool_calls_in_text,
    )
    from litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints import (
        _is_codex_auto_agent_malformed_tool_call_text_output,
        _try_repair_codex_auto_agent_grok_native_composer_literal_tool_call_response_body,
    )

    rewritten = normalization.format_function_call_input_message(
        {
            "name": "read_file",
            "call_id": "call-c9d1aa12-e8c4-4c32-8e16-e7e3c8c9d0e4-0",
            "arguments": {"target_file": "/tmp/x.py", "limit": 20},
        }
    )
    assert rewritten.splitlines()[0] == (
        "[Context note - prior assistant step; not an executable tool invocation]"
    )
    assert "Tool label: read_file" in rewritten
    assert "Correlation ref: call-c9d1aa12-e8c4-4c32-8e16-e7e3c8c9d0e4-0" in rewritten
    assert "Input payload:" in rewritten

    echoed = (
        "HEAD still has the dummy 5-byte Connect envelope. I'll inspect the file.\n"
        + rewritten
    )
    advertised = {
        "read_file": {
            "type": "object",
            "properties": {
                "target_file": {"type": "string"},
                "limit": {"type": "integer"},
            },
            "additionalProperties": True,
        }
    }
    leftover, items = _repair_grok_composer_literal_tool_calls_in_text(
        echoed,
        advertised_tools=advertised,
    )
    assert leftover is None
    assert items == []
    assert is_malformed_grok_literal_tool_label_transcript_text(echoed) is True

    without_note = "\n".join(
        line
        for line in echoed.splitlines()
        if "Context note" not in line
    )
    leftover_ok, items_ok = _repair_grok_composer_literal_tool_calls_in_text(
        without_note,
        advertised_tools=advertised,
    )
    assert items_ok and items_ok[0]["name"] == "read_file"
    assert leftover_ok is not None
    assert "Tool label:" not in leftover_ok

    response_body = {
        "id": "resp_today_dump",
        "object": "response",
        "status": "completed",
        "model": "grok-4.6",
        "output": [
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": echoed}],
            }
        ],
    }
    request_body = {
        "model": "grok-4.6",
        "tools": [
            {
                "type": "function",
                "name": "read_file",
                "parameters": advertised["read_file"],
            }
        ],
    }
    assert _is_codex_auto_agent_malformed_tool_call_text_output(response_body) is True
    assert (
        _try_repair_codex_auto_agent_grok_native_composer_literal_tool_call_response_body(
            response_body,
            request_body=request_body,
        )
        is None
    )


def test_grok_4_6_model_info_rewrites_function_call_history() -> None:
    from pathlib import Path

    prices_path = Path(__file__).resolve().parents[4] / "model_prices_and_context_window.json"
    prices = json.loads(prices_path.read_text(encoding="utf-8"))
    grok_46 = prices["xai/grok-4.6"]
    assert "function_call" in grok_46["rewrite_input_item_types"]
    assert "function_call_output" in grok_46["rewrite_input_item_types"]
    assert "native_responses_tool_history" in grok_46["provider_specific_entry"]["xai"][
        "capabilities"
    ]
