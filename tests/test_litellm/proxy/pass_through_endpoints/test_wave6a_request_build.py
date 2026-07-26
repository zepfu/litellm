"""Wave 6A Author A: request_build extraction ownership + behavior tests.

Enforces the behavior-preserving extraction contract from
``llm_passthrough_endpoints.py`` into:

- ``aawm_adapter_runtime/request_build.py``
    Response-body inspection, malformed tool-call intake context,
    Grok Composer literal-tool repair delegation, JSON schema validation,
    and custom tool function adapter helpers.

Structural ownership tests verify AST presence, dependency isolation,
fixed golden behavior, and production facade identity.
"""

from __future__ import annotations

import ast
import importlib
import json
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# God-module import (always available)
# ---------------------------------------------------------------------------
from litellm.proxy.pass_through_endpoints import llm_passthrough_endpoints as lpe

GOD_PATH = Path(lpe.__file__).resolve()

# ---------------------------------------------------------------------------
# Target module
# ---------------------------------------------------------------------------
TARGET_IMPORT_PATH = "litellm.proxy.pass_through_endpoints.aawm_adapter_runtime.request_build"

# ---------------------------------------------------------------------------
# Symbol inventory
# ---------------------------------------------------------------------------
REQUEST_BUILD_SYMBOLS: set[str] = {
    "_responses_output_item_has_meaningful_content",
    "_is_empty_success_responses_body",
    "_is_failed_responses_body",
    "_build_malformed_tool_call_intake_context",
    "_get_anthropic_grok_composer_repair_runtime",
    "_grok_composer_literal_tool_block_strip_start",
    "_parse_grok_composer_literal_tool_label_blocks",
    "_parse_grok_composer_literal_tool_payload_json",
    "_sanitize_grok_composer_literal_tool_arguments",
    "_escape_unescaped_newlines_in_json_payload",
    "_strip_text_spans",
    "_build_advertised_openai_function_tools_index",
    "_json_schema_value_matches_type",
    "_validate_tool_arguments_against_openai_parameters",
    "_build_repaired_grok_composer_function_call_output_item",
    "_dedupe_repaired_grok_composer_call_id",
    "_repair_grok_composer_literal_tool_calls_in_text",
    "_response_body_has_grok_composer_literal_tool_label_blocks",
    "_repair_grok_composer_literal_tool_calls_in_message_item",
    "_try_repair_codex_auto_agent_grok_native_composer_literal_tool_call_response_body",
    "_advertised_custom_tool_function_adapter_names",
    "_parse_adapted_custom_tool_function_arguments",
}

# Module must expose install()
REQUIRED_MODULE_ATTRIBUTES: set[str] = {"install", "_HOST_FUNCTION_NAMES"}


# ---------------------------------------------------------------------------
# AST helpers
# ---------------------------------------------------------------------------


def _parse_module_ast(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _top_level_function_names(tree: ast.Module) -> set[str]:
    return {
        node.name
        for node in ast.iter_child_nodes(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def _top_level_assign_names(tree: ast.Module) -> set[str]:
    names: set[str] = set()
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    names.add(target.id)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
    return names


# ---------------------------------------------------------------------------
# Structural ownership tests
# ---------------------------------------------------------------------------


class TestStructuralOwnership:
    """Verify the target module exists and contains the expected symbols."""

    @pytest.fixture(autouse=True)
    def _load_target(self):
        self.target_mod = importlib.import_module(TARGET_IMPORT_PATH)
        self.target_path = Path(self.target_mod.__file__).resolve()
        self.target_tree = _parse_module_ast(self.target_path)

    def test_module_importable(self):
        assert self.target_mod is not None

    def test_all_symbols_present_as_functions(self):
        fn_names = _top_level_function_names(self.target_tree)
        missing = REQUEST_BUILD_SYMBOLS - fn_names
        assert not missing, f"Missing function defs in request_build.py: {missing}"

    def test_install_function_present(self):
        fn_names = _top_level_function_names(self.target_tree)
        assert "install" in fn_names

    def test_host_function_names_tuple_matches_inventory(self):
        host_names = getattr(self.target_mod, "_HOST_FUNCTION_NAMES", ())
        assert set(host_names) == REQUEST_BUILD_SYMBOLS

    def test_no_god_module_import_at_module_scope(self):
        """request_build must NOT import llm_passthrough_endpoints."""
        source = self.target_path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert "llm_passthrough_endpoints" not in alias.name
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    assert "llm_passthrough_endpoints" not in node.module

    def test_no_import_of_excluded_scopes(self):
        """Must not import SSE, stream collection, endpoint predicates, etc."""
        source = self.target_path.read_text(encoding="utf-8")
        excluded_patterns = [
            "StreamingResponse",
            "EventSourceResponse",
            "sse_",
            "_stream_collect",
            "_validate_payload",
            "_restore_tool_calls",
        ]
        for pattern in excluded_patterns:
            assert pattern not in source, f"Excluded pattern found: {pattern}"


# ---------------------------------------------------------------------------
# Dependency isolation tests
# ---------------------------------------------------------------------------


class TestDependencyIsolation:
    """Verify host-global seams are declared via TYPE_CHECKING only."""

    @pytest.fixture(autouse=True)
    def _load_target(self):
        self.target_mod = importlib.import_module(TARGET_IMPORT_PATH)
        self.target_path = Path(self.target_mod.__file__).resolve()
        self.source = self.target_path.read_text(encoding="utf-8")

    def test_type_checking_guard_present(self):
        assert "if TYPE_CHECKING:" in self.source

    def test_host_functions_declared_in_type_checking_block(self):
        """Key host functions should be declared under TYPE_CHECKING."""
        expected_decls = [
            "_extract_auto_agent_alias_metadata_value",
            "_extract_auto_agent_alias_agent_dispatch_fields",
            "_extract_auto_agent_alias_incoming_endpoint",
            "_extract_auto_agent_alias_session_id",
            "_extract_passthrough_repository",
            "_get_request_header_or_passthrough_alias",
            "is_malformed_composer_call_literal_text",
            "_tool_definition_name",
            "_tool_definition_parameters",
            "_normalize_openai_function_tool_parameters",
            "_get_custom_tool_function_adapter_names_for_model",
            "_normalize_low_cardinality_tag_value",
            "_get_openai_tool_type",
            "_get_openai_tool_name",
        ]
        # Extract TYPE_CHECKING block
        tc_start = self.source.index("if TYPE_CHECKING:")
        # Find end: next top-level non-indented line after the block
        lines = self.source[tc_start:].split("\n")
        tc_block_lines = [lines[0]]
        for line in lines[1:]:
            if line and not line[0].isspace() and not line.startswith("#"):
                break
            tc_block_lines.append(line)
        tc_block = "\n".join(tc_block_lines)
        for decl in expected_decls:
            assert decl in tc_block, f"Missing TYPE_CHECKING declaration: {decl}"

    def test_runtime_imports_are_stdlib_or_typing_only(self):
        """Non-TYPE_CHECKING imports must be stdlib/typing only."""
        tree = ast.parse(self.source)
        allowed_modules = {
            "__future__", "json", "re", "functools", "typing", "types",
        }
        in_type_checking = False
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.If):
                # Detect TYPE_CHECKING guard
                test = node.test
                if isinstance(test, ast.Name) and test.id == "TYPE_CHECKING":
                    in_type_checking = True
                    continue
            if in_type_checking:
                # After the TYPE_CHECKING block, reset when we hit a non-If top-level
                if not isinstance(node, ast.If):
                    in_type_checking = False
                else:
                    continue
            if isinstance(node, ast.Import):
                for alias in node.names:
                    root = alias.name.split(".")[0]
                    assert root in allowed_modules, f"Unexpected runtime import: {alias.name}"
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    root = node.module.split(".")[0]
                    assert root in allowed_modules, f"Unexpected runtime import: {node.module}"


# ---------------------------------------------------------------------------
# Behavior tests: pure functions (no host-global deps)
# ---------------------------------------------------------------------------


class TestResponsesOutputItemHasMeaningfulContent:
    """Behavior tests for _responses_output_item_has_meaningful_content."""

    @pytest.fixture(autouse=True)
    def _fn(self):
        mod = importlib.import_module(TARGET_IMPORT_PATH)
        self.fn = mod._responses_output_item_has_meaningful_content

    def test_non_dict_returns_false(self):
        assert self.fn("string") is False
        assert self.fn(None) is False
        assert self.fn(42) is False

    def test_function_call_returns_true(self):
        assert self.fn({"type": "function_call", "name": "foo"}) is True

    def test_mcp_call_returns_true(self):
        assert self.fn({"type": "mcp_call"}) is True

    def test_message_with_text_content(self):
        item = {"type": "message", "content": [{"type": "output_text", "text": "hello"}]}
        assert self.fn(item) is True

    def test_message_with_empty_text(self):
        item = {"type": "message", "content": [{"type": "output_text", "text": "  "}]}
        assert self.fn(item) is False

    def test_message_with_no_content(self):
        assert self.fn({"type": "message", "content": []}) is False
        assert self.fn({"type": "message"}) is False

    def test_reasoning_with_summary(self):
        item = {"type": "reasoning", "summary": [{"text": "thinking..."}]}
        assert self.fn(item) is True

    def test_reasoning_empty_summary(self):
        assert self.fn({"type": "reasoning", "summary": []}) is False

    def test_unknown_type_returns_false(self):
        assert self.fn({"type": "web_search_call"}) is False


class TestIsEmptySuccessResponsesBody:
    """Behavior tests for _is_empty_success_responses_body."""

    @pytest.fixture(autouse=True)
    def _fn(self):
        mod = importlib.import_module(TARGET_IMPORT_PATH)
        self.fn = mod._is_empty_success_responses_body

    def test_empty_output_completed(self):
        assert self.fn({"status": "completed", "output": []}) is True

    def test_none_status_empty_output(self):
        assert self.fn({"output": []}) is True

    def test_failed_status_not_empty(self):
        assert self.fn({"status": "failed", "output": []}) is False

    def test_in_progress_not_empty(self):
        assert self.fn({"status": "in_progress", "output": []}) is False

    def test_output_with_function_call(self):
        body = {"status": "completed", "output": [{"type": "function_call"}]}
        assert self.fn(body) is False

    def test_output_text_present(self):
        body = {"status": "completed", "output": [], "output_text": "hello"}
        assert self.fn(body) is False

    def test_output_text_whitespace_only(self):
        body = {"status": "completed", "output": [], "output_text": "   "}
        assert self.fn(body) is True


class TestIsFailedResponsesBody:
    """Behavior tests for _is_failed_responses_body."""

    @pytest.fixture(autouse=True)
    def _fn(self):
        mod = importlib.import_module(TARGET_IMPORT_PATH)
        self.fn = mod._is_failed_responses_body

    def test_failed_status(self):
        assert self.fn({"status": "failed"}) is True

    def test_error_present(self):
        assert self.fn({"error": {"message": "boom"}}) is True

    def test_completed_no_error(self):
        assert self.fn({"status": "completed"}) is False

    def test_empty_body(self):
        assert self.fn({}) is False


class TestEscapeUnescapedNewlines:
    """Behavior tests for _escape_unescaped_newlines_in_json_payload."""

    @pytest.fixture(autouse=True)
    def _fn(self):
        mod = importlib.import_module(TARGET_IMPORT_PATH)
        self.fn = mod._escape_unescaped_newlines_in_json_payload

    def test_empty_string(self):
        assert self.fn("") == ""

    def test_newline_inside_json_string_escaped(self):
        payload = '{"key": "line1\nline2"}'
        result = self.fn(payload)
        assert "\n" not in result
        assert "\\n" in result

    def test_newline_outside_json_string_preserved(self):
        payload = '{"key": "val"}\n{"key2": "val2"}'
        result = self.fn(payload)
        # The newline between objects is outside a string, preserved
        assert result == payload

    def test_already_escaped_newline_not_double_escaped(self):
        payload = '{"key": "line1\\nline2"}'
        result = self.fn(payload)
        assert result == payload

    def test_non_string_passthrough(self):
        assert self.fn(None) is None  # type: ignore[arg-type]


class TestStripTextSpans:
    """Behavior tests for _strip_text_spans."""

    @pytest.fixture(autouse=True)
    def _fn(self):
        mod = importlib.import_module(TARGET_IMPORT_PATH)
        self.fn = mod._strip_text_spans

    def test_no_spans(self):
        assert self.fn("hello world", []) == "hello world"

    def test_single_span(self):
        assert self.fn("hello world", [(5, 11)]) == "hello"

    def test_overlapping_spans_merged(self):
        text = "abcdefghij"
        result = self.fn(text, [(0, 3), (2, 5)])
        assert result == "fghij"

    def test_multiple_non_overlapping(self):
        text = "abcdefghij"
        result = self.fn(text, [(0, 2), (5, 7)])
        assert result == "cdehij"


class TestJsonSchemaValueMatchesType:
    """Behavior tests for _json_schema_value_matches_type."""

    @pytest.fixture(autouse=True)
    def _fn(self):
        mod = importlib.import_module(TARGET_IMPORT_PATH)
        self.fn = mod._json_schema_value_matches_type

    def test_string(self):
        assert self.fn("hello", "string") is True
        assert self.fn(42, "string") is False

    def test_integer(self):
        assert self.fn(42, "integer") is True
        assert self.fn(True, "integer") is False
        assert self.fn(3.14, "integer") is False

    def test_number(self):
        assert self.fn(3.14, "number") is True
        assert self.fn(42, "number") is True
        assert self.fn(True, "number") is False

    def test_boolean(self):
        assert self.fn(True, "boolean") is True
        assert self.fn(1, "boolean") is False

    def test_array(self):
        assert self.fn([1, 2], "array") is True
        assert self.fn("no", "array") is False

    def test_object(self):
        assert self.fn({"a": 1}, "object") is True
        assert self.fn([], "object") is False

    def test_null(self):
        assert self.fn(None, "null") is True
        assert self.fn("", "null") is False

    def test_unknown_type_always_true(self):
        assert self.fn("anything", "custom_type") is True


class TestValidateToolArguments:
    """Behavior tests for _validate_tool_arguments_against_openai_parameters."""

    @pytest.fixture(autouse=True)
    def _fn(self):
        mod = importlib.import_module(TARGET_IMPORT_PATH)
        self.fn = mod._validate_tool_arguments_against_openai_parameters

    def test_non_dict_arguments(self):
        result = self.fn(tool_name="t", arguments="str", parameters={})
        assert result == "tool_arguments_not_object"

    def test_unsupported_root_type(self):
        result = self.fn(tool_name="t", arguments={}, parameters={"type": "array"})
        assert result == "tool_schema_unsupported_root_type"

    def test_missing_required_field(self):
        params = {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}
        result = self.fn(tool_name="t", arguments={}, parameters=params)
        assert result == "missing_required_argument:name"

    def test_valid_arguments(self):
        params = {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}
        result = self.fn(tool_name="t", arguments={"name": "foo"}, parameters=params)
        assert result is None

    def test_additional_properties_false_unknown_key(self):
        params = {"type": "object", "properties": {"a": {"type": "string"}}, "additionalProperties": False}
        result = self.fn(tool_name="t", arguments={"a": "x", "b": "y"}, parameters=params)
        assert result == "unknown_argument:b"

    def test_type_mismatch(self):
        params = {"type": "object", "properties": {"count": {"type": "integer"}}}
        result = self.fn(tool_name="t", arguments={"count": "not_int"}, parameters=params)
        assert result == "argument_type_mismatch:count"

    def test_union_type_match(self):
        params = {"type": "object", "properties": {"val": {"type": ["string", "integer"]}}}
        result = self.fn(tool_name="t", arguments={"val": 42}, parameters=params)
        assert result is None

    def test_union_type_mismatch(self):
        params = {"type": "object", "properties": {"val": {"type": ["string", "integer"]}}}
        result = self.fn(tool_name="t", arguments={"val": [1]}, parameters=params)
        assert result == "argument_type_mismatch:val"


class TestParseAdaptedCustomToolFunctionArguments:
    """Behavior tests for _parse_adapted_custom_tool_function_arguments."""

    @pytest.fixture(autouse=True)
    def _fn(self):
        mod = importlib.import_module(TARGET_IMPORT_PATH)
        self.fn = mod._parse_adapted_custom_tool_function_arguments

    def test_non_string_arguments(self):
        result, error = self.fn(42)
        assert result is None
        assert error == "arguments_not_string"

    def test_invalid_json(self):
        result, error = self.fn("not json{{{")
        assert result is None
        assert error == "arguments_not_json"

    def test_not_object(self):
        result, error = self.fn('"just a string"')
        assert result is None
        assert error == "arguments_not_object"

    def test_not_exact_input_object(self):
        result, error = self.fn(json.dumps({"input": "x", "extra": "y"}))
        assert result is None
        assert error == "arguments_not_exact_input_object"

    def test_input_not_string(self):
        result, error = self.fn(json.dumps({"input": 123}))
        assert result is None
        assert error == "input_not_string"

    def test_valid_input(self):
        result, error = self.fn(json.dumps({"input": "hello world"}))
        assert result == "hello world"
        assert error is None


# ---------------------------------------------------------------------------
# Golden parity: god-module functions still callable and match behavior
# ---------------------------------------------------------------------------


class TestGoldenParity:
    """Verify god-module symbols remain callable (pre-extraction parity)."""

    def test_god_module_exposes_same_object_facades(self):
        mod = importlib.import_module(TARGET_IMPORT_PATH)
        for name in REQUEST_BUILD_SYMBOLS:
            assert getattr(lpe, name) is getattr(mod, name)


# ---------------------------------------------------------------------------
# Install mechanism tests
# ---------------------------------------------------------------------------


class TestInstallMechanism:
    """Verify production initialization binds the compatibility facades."""

    def test_production_initialization_populates_god_module(self):
        mod = importlib.import_module(TARGET_IMPORT_PATH)
        for name in REQUEST_BUILD_SYMBOLS:
            assert getattr(lpe, name) is getattr(mod, name)

    def test_production_facades_use_god_module_globals(self):
        mod = importlib.import_module(TARGET_IMPORT_PATH)
        for name in REQUEST_BUILD_SYMBOLS:
            facade = getattr(mod, name)
            function = getattr(facade, "__wrapped__", facade)
            assert function.__globals__ is vars(lpe)
