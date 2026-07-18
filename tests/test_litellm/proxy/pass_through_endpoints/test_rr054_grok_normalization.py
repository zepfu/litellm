from copy import deepcopy
from typing import Any, Optional

import pytest

from litellm.llms.anthropic.experimental_pass_through.providers.grok import (
    normalization,
)
from litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints import (
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
