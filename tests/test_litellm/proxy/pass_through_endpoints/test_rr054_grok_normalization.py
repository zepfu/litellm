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


def _direct_grok_passthrough_request(*, model: str) -> Any:
    from unittest.mock import MagicMock

    from fastapi import Request

    request = MagicMock(spec=Request)
    request.headers = {
        "content-type": "application/json",
        "x-grok-model-override": model,
        "x-grok-session-id": "session_xai046",
    }
    request.query_params = {}
    request.scope = {}
    return request


def _typed_tool_history_request_body(*, model: str) -> dict[str, Any]:
    return {
        "model": model,
        "input": [
            {"type": "message", "role": "user", "content": "continue"},
            {
                "type": "function_call",
                "name": "read_file",
                "call_id": "call-c9d1aa12-e8c4-4c32-8e16-e7e3c8c9d0e4-0",
                "arguments": {"target_file": "/tmp/x.py", "limit": 20},
            },
            {
                "type": "function_call_output",
                "call_id": "call-c9d1aa12-e8c4-4c32-8e16-e7e3c8c9d0e4-0",
                "output": {"status": "ok"},
            },
        ],
    }


def test_grok_history_preserves_typed_tool_items_for_grok_46() -> None:
    """Direct /grok/v1 request-prep keeps grok-4.6 typed tool history.

    Commit 79b0abb0c3 asserted flattening into Context-note / Tool label
    messages. That characterization is inverted: the shipped Grok CLI
    chat-proxy path must leave function_call / function_call_output items
    as those types and must not synthesize the rewrite dialect into input.
    Residual model-authored Context-note dumps are XAI-047.
    """
    from litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints import (
        _prepare_grok_request_body_for_passthrough,
    )

    prepared = _prepare_grok_request_body_for_passthrough(
        request=_direct_grok_passthrough_request(model="grok-4.6"),
        request_body=_typed_tool_history_request_body(model="grok-4.6"),
    )
    input_items = prepared["input"]
    assert [item.get("type") for item in input_items] == [
        "message",
        "function_call",
        "function_call_output",
    ]
    assert input_items[1]["name"] == "read_file"
    assert input_items[1]["call_id"] == (
        "call-c9d1aa12-e8c4-4c32-8e16-e7e3c8c9d0e4-0"
    )
    assert input_items[2]["call_id"] == (
        "call-c9d1aa12-e8c4-4c32-8e16-e7e3c8c9d0e4-0"
    )
    rendered = json.dumps(prepared["input"])
    assert (
        "[Context note - prior assistant step; not an executable tool invocation]"
        not in rendered
    )
    assert "Tool label:" not in rendered
    assert "Correlation ref:" not in rendered
    assert "Input payload:" not in rendered


def test_grok_model_capability_preserves_grok_46_and_flattens_composer_history() -> None:
    """grok-4.6 CLI passthrough preserves history; composer still flattens.

    Commit 79b0abb0c3 treated rewrite_input_item_types as the product
    contract. Invert: native_responses_tool_history on the Grok CLI
    chat-proxy path must not flatten grok-4.6 history even while the
    catalog still lists rewrite types for composer models.
    """
    from pathlib import Path

    from litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints import (
        _prepare_grok_request_body_for_passthrough,
    )

    prices_path = (
        Path(__file__).resolve().parents[4] / "model_prices_and_context_window.json"
    )
    prices = json.loads(prices_path.read_text(encoding="utf-8"))
    grok_46 = prices["xai/grok-4.6"]
    assert "native_responses_tool_history" in grok_46["provider_specific_entry"]["xai"][
        "capabilities"
    ]

    preserved = _prepare_grok_request_body_for_passthrough(
        request=_direct_grok_passthrough_request(model="grok-4.6"),
        request_body=_typed_tool_history_request_body(model="grok-4.6"),
    )
    assert [item.get("type") for item in preserved["input"]] == [
        "message",
        "function_call",
        "function_call_output",
    ]
    assert "Tool label:" not in json.dumps(preserved["input"])

    flattened = _prepare_grok_request_body_for_passthrough(
        request=_direct_grok_passthrough_request(model="grok-composer-2.5-fast"),
        request_body=_typed_tool_history_request_body(model="grok-composer-2.5-fast"),
    )
    assert [item.get("type") for item in flattened["input"]] == [
        "message",
        "message",
        "message",
    ]
    assert flattened["input"][1]["role"] == "assistant"
    assert "Tool label: read_file" in flattened["input"][1]["content"]
    assert flattened["input"][2]["role"] == "user"
