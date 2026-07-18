"""RR-054 Grok Composer parsing and repair ownership tests."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from litellm.llms.anthropic.experimental_pass_through.providers.grok import (
    composer_repair,
)


OWNER_PATH = Path(composer_repair.__file__).resolve()
PROXY_PATH = (
    OWNER_PATH.parents[5]
    / "proxy"
    / "pass_through_endpoints"
    / "llm_passthrough_endpoints.py"
)

OWNER_FUNCTIONS = {
    "literal_tool_block_strip_start",
    "parse_literal_tool_label_blocks",
    "parse_literal_tool_payload_json",
    "sanitize_literal_tool_arguments",
    "escape_unescaped_newlines_in_json_payload",
    "build_repaired_function_call_output_item",
    "dedupe_repaired_call_id",
    "repair_literal_tool_calls_in_text",
    "response_body_has_literal_tool_label_blocks",
    "repair_literal_tool_calls_in_message_item",
    "try_repair_literal_tool_call_response_body",
}

COMPATIBILITY_DELEGATES = {
    "_grok_composer_literal_tool_block_strip_start": (
        "_anthropic_grok_composer_repair.literal_tool_block_strip_start"
    ),
    "_parse_grok_composer_literal_tool_label_blocks": (
        "_anthropic_grok_composer_repair.parse_literal_tool_label_blocks"
    ),
    "_parse_grok_composer_literal_tool_payload_json": (
        "_anthropic_grok_composer_repair.parse_literal_tool_payload_json"
    ),
    "_sanitize_grok_composer_literal_tool_arguments": (
        "_anthropic_grok_composer_repair.sanitize_literal_tool_arguments"
    ),
    "_build_repaired_grok_composer_function_call_output_item": (
        "_anthropic_grok_composer_repair.build_repaired_function_call_output_item"
    ),
    "_dedupe_repaired_grok_composer_call_id": (
        "_anthropic_grok_composer_repair.dedupe_repaired_call_id"
    ),
    "_repair_grok_composer_literal_tool_calls_in_text": (
        "_anthropic_grok_composer_repair.repair_literal_tool_calls_in_text"
    ),
    "_response_body_has_grok_composer_literal_tool_label_blocks": (
        "_anthropic_grok_composer_repair."
        "response_body_has_literal_tool_label_blocks"
    ),
    "_repair_grok_composer_literal_tool_calls_in_message_item": (
        "_anthropic_grok_composer_repair.repair_literal_tool_calls_in_message_item"
    ),
    "_try_repair_codex_auto_agent_grok_native_composer_literal_tool_call_response_body": (
        "_anthropic_grok_composer_repair."
        "try_repair_literal_tool_call_response_body"
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
    raise AssertionError(f"{name} must be defined")


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


def test_rr054_grok_composer_module_defines_real_owner_functions() -> None:
    owner_tree = _parse(OWNER_PATH)

    for function_name in OWNER_FUNCTIONS:
        _function(owner_tree, function_name)


def test_rr054_grok_composer_module_owns_substantive_parsing_and_repair() -> None:
    owner_tree = _parse(OWNER_PATH)

    parse_blocks = _function(owner_tree, "parse_literal_tool_label_blocks")
    assert "runtime.decode_json_prefix" in _calls(parse_blocks)
    assert any(isinstance(node, ast.For) for node in ast.walk(parse_blocks))

    parse_payload = _function(owner_tree, "parse_literal_tool_payload_json")
    assert "json.JSONDecoder" in _calls(parse_payload)
    assert "escape_unescaped_newlines_in_json_payload" in _calls(parse_payload)

    repair_text = _function(owner_tree, "repair_literal_tool_calls_in_text")
    repair_text_calls = _calls(repair_text)
    assert {
        "parse_literal_tool_label_blocks",
        "parse_literal_tool_payload_json",
        "runtime.validate_tool_arguments",
        "runtime.strip_text_spans",
        "runtime.is_malformed_composer_literal_text",
    } <= repair_text_calls
    assert any(isinstance(node, ast.For) for node in ast.walk(repair_text))

    repair_response = _function(
        owner_tree,
        "try_repair_literal_tool_call_response_body",
    )
    assert {
        "runtime.build_advertised_function_tools_index",
        "runtime.is_malformed_tool_call_text_output",
        "repair_literal_tool_calls_in_message_item",
        "response_body_has_literal_tool_label_blocks",
    } <= _calls(repair_response)
    assert any(isinstance(node, ast.For) for node in ast.walk(repair_response))


def test_rr054_grok_composer_owner_does_not_back_import_god_file() -> None:
    owner_tree = _parse(OWNER_PATH)

    imported_modules: set[str] = set()
    for node in ast.walk(owner_tree):
        if isinstance(node, ast.Import):
            imported_modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_modules.add(node.module)

    assert not any(
        module.endswith("llm_passthrough_endpoints") for module in imported_modules
    )


@pytest.mark.parametrize(
    ("function_name", "expected_call"),
    COMPATIBILITY_DELEGATES.items(),
)
def test_rr054_grok_composer_god_file_names_are_exact_thin_delegates(
    function_name: str,
    expected_call: str,
) -> None:
    proxy_tree = _parse(PROXY_PATH)

    function = _function(proxy_tree, function_name)
    calls = _calls(function)
    assert expected_call in calls, (
        f"{function_name} must delegate to {expected_call}; "
        f"calls={sorted(calls)}"
    )
    statements = _non_docstring_statements(function)
    assert len(statements) == 1, (
        f"{function_name} retains Grok Composer logic "
        f"({len(statements)} statements)"
    )
