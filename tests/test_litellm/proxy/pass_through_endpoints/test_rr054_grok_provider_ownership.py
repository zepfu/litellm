"""RR-054 Grok request-normalization ownership tests.

Strict finding #1 assigns provider request shaping to
``providers/<provider>/normalization.py``. The proxy module may retain
compatibility names, but those names must be thin delegates rather than
duplicate Grok sanitization or input-rewrite implementations.
"""

from __future__ import annotations

import ast
from pathlib import Path

from litellm.llms.anthropic.experimental_pass_through.providers.grok import (
    normalization,
)
from litellm.proxy.pass_through_endpoints import llm_passthrough_endpoints as lpe


OWNER_PATH = Path(normalization.__file__).resolve()
PROXY_PATH = Path(lpe.__file__).resolve()

NORMALIZATION_OWNED_FUNCTIONS = {
    "coerce_function_call_arguments_value",
    "sanitize_function_call_arguments_request_body",
    "sanitize_function_call_arguments_in_place",
    "stringify_input_item_value",
    "format_function_call_input_message",
    "format_function_call_output_input_message",
    "rewrite_input_item_for_model_input",
    "is_anthropic_responses_adapter_body",
    "add_input_item_rewrite_logging_metadata",
    "rewrite_unsupported_input_items_from_request_body",
    "rewrite_unsupported_input_items_in_place",
}

PROXY_DELEGATES = {
    "_coerce_grok_native_function_call_arguments_value": (
        "_anthropic_grok_normalization.coerce_function_call_arguments_value"
    ),
    "_sanitize_grok_native_function_call_arguments_request_body": (
        "_anthropic_grok_normalization."
        "sanitize_function_call_arguments_request_body"
    ),
    "_sanitize_grok_native_function_call_arguments_in_place": (
        "_anthropic_grok_normalization.sanitize_function_call_arguments_in_place"
    ),
    "_stringify_grok_native_input_item_value": (
        "_anthropic_grok_normalization.stringify_input_item_value"
    ),
    "_format_grok_native_function_call_input_message": (
        "_anthropic_grok_normalization.format_function_call_input_message"
    ),
    "_format_grok_native_function_call_output_input_message": (
        "_anthropic_grok_normalization.format_function_call_output_input_message"
    ),
    "_rewrite_grok_native_input_item_for_model_input": (
        "_anthropic_grok_normalization.rewrite_input_item_for_model_input"
    ),
    "_is_anthropic_grok_native_responses_adapter_body": (
        "_anthropic_grok_normalization.is_anthropic_responses_adapter_body"
    ),
    "_add_grok_native_input_item_rewrite_logging_metadata": (
        "_anthropic_grok_normalization.add_input_item_rewrite_logging_metadata"
    ),
    "_rewrite_grok_native_unsupported_input_items_from_request_body": (
        "_anthropic_grok_normalization."
        "rewrite_unsupported_input_items_from_request_body"
    ),
    "_rewrite_grok_native_unsupported_input_items_in_place": (
        "_anthropic_grok_normalization.rewrite_unsupported_input_items_in_place"
    ),
}


def _parse(path: Path) -> ast.Module:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    assert isinstance(tree, ast.Module)
    return tree


def _function(tree: ast.Module, name: str) -> ast.FunctionDef | ast.AsyncFunctionDef:
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node
    raise AssertionError(f"{name} must be defined in {tree}")


def _calls(function: ast.FunctionDef | ast.AsyncFunctionDef) -> set[str]:
    calls: set[str] = set()
    for node in ast.walk(function):
        if not isinstance(node, ast.Call):
            continue
        current = node.func
        parts: list[str] = []
        while isinstance(current, ast.Attribute):
            parts.append(current.attr)
            current = current.value
        if isinstance(current, ast.Name):
            parts.append(current.id)
        if parts:
            calls.add(".".join(reversed(parts)))
    return calls


def _non_docstring_statements(
    function: ast.FunctionDef | ast.AsyncFunctionDef,
) -> list[ast.stmt]:
    return [
        statement
        for statement in function.body
        if not (
            isinstance(statement, ast.Expr)
            and isinstance(statement.value, ast.Constant)
            and isinstance(statement.value.value, str)
        )
    ]


def test_rr054_grok_normalization_defines_request_shaping_owners() -> None:
    owner_tree = _parse(OWNER_PATH)

    for function_name in NORMALIZATION_OWNED_FUNCTIONS:
        _function(owner_tree, function_name)


def test_rr054_grok_normalization_owns_sanitization_algorithms() -> None:
    owner_tree = _parse(OWNER_PATH)

    coerce = _function(owner_tree, "coerce_function_call_arguments_value")
    assert "json.loads" in _calls(coerce)
    assert any(isinstance(node, ast.If) for node in ast.walk(coerce))

    sanitize = _function(
        owner_tree,
        "sanitize_function_call_arguments_request_body",
    )
    assert "coerce_function_call_arguments_value" in _calls(sanitize)
    assert any(isinstance(node, ast.For) for node in ast.walk(sanitize))


def test_rr054_grok_normalization_owns_input_rewrite_algorithms() -> None:
    owner_tree = _parse(OWNER_PATH)

    rewrite = _function(
        owner_tree,
        "rewrite_unsupported_input_items_from_request_body",
    )
    assert any(isinstance(node, ast.For) for node in ast.walk(rewrite))
    assert "hashlib.sha256" in _calls(rewrite)
    assert "rewrite_input_item_for_model_input" in _calls(rewrite)
    assert "add_input_item_rewrite_logging_metadata" in _calls(rewrite)
    rewrite_literals = {
        node.value
        for node in ast.walk(rewrite)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }
    assert {"input", "model"} <= rewrite_literals


def test_rr054_grok_normalization_helpers_delegate_within_owner() -> None:
    owner_tree = _parse(OWNER_PATH)
    expected_calls = {
        "sanitize_function_call_arguments_in_place": (
            "sanitize_function_call_arguments_request_body"
        ),
        "format_function_call_input_message": "stringify_input_item_value",
        "format_function_call_output_input_message": "stringify_input_item_value",
        "rewrite_input_item_for_model_input": (
            "format_function_call_input_message"
        ),
        "add_input_item_rewrite_logging_metadata": "runtime.merge_metadata",
        "rewrite_unsupported_input_items_in_place": (
            "rewrite_unsupported_input_items_from_request_body"
        ),
    }

    for function_name, expected_call in expected_calls.items():
        function = _function(owner_tree, function_name)
        assert expected_call in _calls(function)


def test_rr054_grok_proxy_request_shaping_names_are_thin_delegates() -> None:
    proxy_tree = _parse(PROXY_PATH)

    for function_name, expected_call in PROXY_DELEGATES.items():
        function = _function(proxy_tree, function_name)
        assert expected_call in _calls(function), (
            f"{function_name} must delegate to {expected_call}; "
            f"calls={sorted(_calls(function))}"
        )
        statements = _non_docstring_statements(function)
        assert len(statements) == 1, (
            f"{function_name} retains Grok request-shaping logic "
            f"({len(statements)} statements)"
        )
