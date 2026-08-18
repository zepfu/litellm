"""Wave 6A Author C tool_call_restore module-local tests.

AST structure, dependency isolation, and behavior tests for the extracted
tool-call restore symbols plus the repaired-output-ID host seam.
"""

from __future__ import annotations

import ast
import asyncio
import json
from pathlib import Path
from typing import Any, Optional

import pytest

from litellm.proxy.pass_through_endpoints import llm_passthrough_endpoints as lpe
from litellm.proxy.pass_through_endpoints.aawm_adapter_runtime import (
    sse as sse_mod,
    tool_call_restore as mod,
)

MODULE_PATH = Path(mod.__file__).resolve()

EXPECTED_SYMBOLS = (
    "_restore_adapted_custom_tool_calls_in_response_body",
    "_advertised_namespace_tool_function_adapter_map",
    "_advertised_namespace_tool_argument_schemas",
    "_schema_requires_integer",
    "_is_finite_integral_float",
    "_repair_namespace_argument_value",
    "_repair_adapted_namespace_tool_arguments",
    "_attach_namespace_argument_repair_metadata",
    "_restore_adapted_namespace_tool_call_item",
    "_restore_adapted_namespace_tool_calls_in_response_body",
    "_adapted_custom_tool_stream_state_keys",
    "_remember_adapted_custom_tool_stream_state",
    "_get_adapted_custom_tool_stream_state",
    "_restore_adapted_custom_tool_calls_in_stream_event_payload",
    "_restore_adapted_custom_tool_calls_in_sse_event_block",
    "_restore_adapted_custom_tool_calls_in_streaming_response",
    "_restore_adapted_namespace_tool_calls_in_stream_event_payload",
    "_restore_adapted_namespace_tool_calls_in_sse_event_block",
    "_restore_adapted_namespace_tool_calls_in_streaming_response",
    "_raise_codex_auto_agent_malformed_adapted_custom_tool_call",
)

EXPECTED_SEAMS = {"_responses_repaired_output_item_id"}


# ---------------------------------------------------------------------------
# Helpers - stub host dependencies injected into module namespace
# ---------------------------------------------------------------------------


def _stub_normalize_low_cardinality_tag_value(value: Any) -> Optional[str]:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        cleaned = value.strip().lower()
        return cleaned or None
    return None


def _stub_parse_adapted_custom_tool_function_arguments(
    arguments: Any,
) -> tuple[Optional[str], Optional[str]]:
    if not isinstance(arguments, str):
        return None, "arguments_not_string"
    try:
        parsed = json.loads(arguments)
    except (TypeError, ValueError):
        return None, "arguments_not_json"
    if not isinstance(parsed, dict):
        return None, "arguments_not_object"
    if set(parsed) != {"input"}:
        return None, "arguments_not_exact_input_object"
    raw_input = parsed.get("input")
    if not isinstance(raw_input, str):
        return None, "input_not_string"
    return raw_input, None


def _stub_responses_repaired_output_item_id(
    item: dict[str, Any],
    index: int,
) -> str:
    for key in ("id", "call_id"):
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return f"item_{index}"


@pytest.fixture()
def host_deps(monkeypatch: pytest.MonkeyPatch) -> None:
    """Inject stub dependencies into the live production host namespace."""
    monkeypatch.setattr(
        lpe,
        "_normalize_low_cardinality_tag_value",
        _stub_normalize_low_cardinality_tag_value,
        raising=False,
    )
    monkeypatch.setattr(
        lpe,
        "_parse_adapted_custom_tool_function_arguments",
        _stub_parse_adapted_custom_tool_function_arguments,
        raising=False,
    )
    monkeypatch.setattr(
        lpe,
        "_advertised_custom_tool_function_adapter_names",
        lambda request_body, *, adapter_model: {"my_tool"} if request_body else set(),
        raising=False,
    )
    monkeypatch.setattr(
        lpe,
        "_get_namespace_tool_function_adapter_names_for_model",
        lambda model: {"ns_adapter": {"tool_a"}} if model == "test-model" else {},
        raising=False,
    )
    monkeypatch.setattr(
        lpe,
        "_adapt_codex_namespace_tool_definitions",
        lambda tools, *, adapter_names: (
            None,
            [{"name": "tool_a", "namespace": "ns_adapter"}],
            [],
        ),
        raising=False,
    )
    monkeypatch.setattr(
        lpe,
        "_build_failed_responses_diagnostic",
        lambda *, response_body, adapter, adapter_model, stream_event_summaries=None: {
            "adapter": adapter,
            "adapter_model": adapter_model,
        },
        raising=False,
    )
    monkeypatch.setattr(
        lpe,
        "_responses_repaired_output_item_id",
        _stub_responses_repaired_output_item_id,
        raising=False,
    )


# ===========================================================================
# SECTION 1 - AST structure
# ===========================================================================


class TestASTStructure:
    """All extracted restore symbols must be top-level function defs."""

    def test_all_symbols_present(self) -> None:
        tree = ast.parse(MODULE_PATH.read_text())
        top_level_names = {
            node.name
            for node in ast.iter_child_nodes(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        for sym in EXPECTED_SYMBOLS:
            assert sym in top_level_names, f"Missing symbol: {sym}"

    def test_install_present(self) -> None:
        tree = ast.parse(MODULE_PATH.read_text())
        top_level_names = {
            node.name
            for node in ast.iter_child_nodes(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        assert "install" in top_level_names

    def test_host_function_names_tuple_matches(self) -> None:
        assert set(mod._HOST_FUNCTION_NAMES) == set(EXPECTED_SYMBOLS)

    def test_host_seams_not_locally_owned(self) -> None:
        tree = ast.parse(MODULE_PATH.read_text())
        top_level_names = {
            node.name
            for node in ast.iter_child_nodes(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        assert not EXPECTED_SEAMS & top_level_names

    def test_no_god_module_import_at_module_scope(self) -> None:
        tree = ast.parse(MODULE_PATH.read_text())
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert "llm_passthrough_endpoints" not in alias.name
            elif isinstance(node, ast.ImportFrom):
                assert node.module is None or "llm_passthrough_endpoints" not in node.module


# ===========================================================================
# SECTION 2 - Dependency isolation
# ===========================================================================


class TestDependencyIsolation:
    """Module must not import the god module at runtime."""

    def test_no_runtime_god_import(self) -> None:
        source = MODULE_PATH.read_text()
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                if node.module and "llm_passthrough_endpoints" in node.module:
                    pytest.fail(f"Runtime import of god module at line {node.lineno}")
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if "llm_passthrough_endpoints" in alias.name:
                        pytest.fail(f"Runtime import of god module at line {node.lineno}")

    def test_module_imports_only_allowed(self) -> None:
        """Top-level imports should be stdlib, fastapi, litellm.proxy._types, types."""
        tree = ast.parse(MODULE_PATH.read_text())
        allowed_modules = {
            "__future__", "codecs", "json", "typing", "types",
            "fastapi.responses", "litellm.proxy._types",
        }
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                assert node.module in allowed_modules or node.module.startswith("typing"), (
                    f"Unexpected import: {node.module}"
                )


# ===========================================================================
# SECTION 3 - Custom tool response body restore
# ===========================================================================


class TestCustomToolResponseBody:
    """_restore_adapted_custom_tool_calls_in_response_body behavior."""

    def test_no_adapted_names_passthrough(self, host_deps: None) -> None:
        body = {"output": [{"type": "function_call", "name": "other"}]}
        result, count, err = mod._restore_adapted_custom_tool_calls_in_response_body(
            body, request_body=None, adapter_model="m",
        )
        assert result is body
        assert count == 0
        assert err is None

    def test_restores_custom_tool_call(self, host_deps: None) -> None:
        body = {
            "output": [
                {
                    "type": "function_call",
                    "name": "My_Tool",
                    "arguments": json.dumps({"input": "hello"}),
                    "call_id": "c1",
                },
            ],
        }
        result, count, err = mod._restore_adapted_custom_tool_calls_in_response_body(
            body, request_body={"tools": []}, adapter_model="m",
        )
        assert count == 1
        assert err is None
        item = result["output"][0]
        assert item["type"] == "custom_tool_call"
        assert item["input"] == "hello"
        assert item["status"] == "completed"
        assert "arguments" not in item

    def test_malformed_arguments_returns_error(self, host_deps: None) -> None:
        body = {
            "output": [
                {"type": "function_call", "name": "my_tool", "arguments": "not-json"},
            ],
        }
        result, count, err = mod._restore_adapted_custom_tool_calls_in_response_body(
            body, request_body={"tools": []}, adapter_model="m",
        )
        assert count == 0
        assert err is not None
        assert err["reason"] == "arguments_not_json"
        assert result is body

    def test_non_list_output_passthrough(self, host_deps: None) -> None:
        body = {"output": "not-a-list"}
        result, count, err = mod._restore_adapted_custom_tool_calls_in_response_body(
            body, request_body={"tools": []}, adapter_model="m",
        )
        assert result is body
        assert count == 0


# ===========================================================================
# SECTION 4 - Namespace tool restore
# ===========================================================================


class TestNamespaceToolRestore:
    """Namespace tool item and response body restore."""

    def test_restore_item_adds_namespace(self, host_deps: None) -> None:
        item = {"type": "function_call", "name": "tool_a"}
        restored, count = mod._restore_adapted_namespace_tool_call_item(
            item, namespace_by_name={"tool_a": "ns_adapter"},
        )
        assert count == 1
        assert restored["namespace"] == "ns_adapter"
        assert restored is not item

    def test_restore_item_skips_existing_namespace(self, host_deps: None) -> None:
        item = {"type": "function_call", "name": "tool_a", "namespace": "existing"}
        restored, count = mod._restore_adapted_namespace_tool_call_item(
            item, namespace_by_name={"tool_a": "ns_adapter"},
        )
        assert count == 0
        assert restored is item

    def test_restore_item_skips_non_function_call(self, host_deps: None) -> None:
        item = {"type": "message", "name": "tool_a"}
        restored, count = mod._restore_adapted_namespace_tool_call_item(
            item, namespace_by_name={"tool_a": "ns_adapter"},
        )
        assert count == 0
        assert restored is item

    def test_response_body_restore(self, host_deps: None) -> None:
        body = {
            "output": [
                {"type": "function_call", "name": "tool_a"},
                {"type": "message"},
            ],
        }
        result, count = mod._restore_adapted_namespace_tool_calls_in_response_body(
            body, request_body={"tools": []}, adapter_model="test-model",
        )
        assert count == 1
        assert result["output"][0]["namespace"] == "ns_adapter"
        assert result["output"][1] == {"type": "message"}

    def test_custom_vs_namespace_separation(self, host_deps: None) -> None:
        """Custom restore and namespace restore operate independently."""
        body = {
            "output": [
                {
                    "type": "function_call",
                    "name": "my_tool",
                    "arguments": json.dumps({"input": "x"}),
                },
            ],
        }
        # Custom restore transforms type
        custom_result, custom_count, _ = mod._restore_adapted_custom_tool_calls_in_response_body(
            body, request_body={"tools": []}, adapter_model="m",
        )
        assert custom_count == 1
        assert custom_result["output"][0]["type"] == "custom_tool_call"

        # Namespace restore does NOT transform the same item (name not in ns map)
        ns_result, ns_count = mod._restore_adapted_namespace_tool_calls_in_response_body(
            body, request_body={"tools": []}, adapter_model="test-model",
        )
        assert ns_count == 0
        assert ns_result is body


# ===========================================================================
# SECTION 4B - Schema-driven integer argument repair
# ===========================================================================


_WAIT_AGENT_INTEGER_SCHEMA = {
    "type": "object",
    "properties": {"timeout_ms": {"type": "integer"}},
    "additionalProperties": False,
}


class TestNamespaceArgumentRepair:
    """Convert only finite integral floats where the advertised schema is integer."""

    def test_repairs_integral_float_timeout_ms(self) -> None:
        repaired, fields = mod._repair_adapted_namespace_tool_arguments(
            json.dumps({"timeout_ms": 120000.0}),
            _WAIT_AGENT_INTEGER_SCHEMA,
        )
        assert json.loads(repaired) == {"timeout_ms": 120000}
        assert isinstance(json.loads(repaired)["timeout_ms"], int)
        assert fields == ["timeout_ms"]

    def test_leaves_number_schema_float_unchanged(self) -> None:
        arguments = json.dumps({"timeout_ms": 120000.0})
        repaired, fields = mod._repair_adapted_namespace_tool_arguments(
            arguments,
            {
                "type": "object",
                "properties": {"timeout_ms": {"type": "number"}},
            },
        )
        assert repaired == arguments
        assert fields == []

    def test_leaves_nonconvertible_values_unchanged(self) -> None:
        payload = {
            "flag": True,
            "count": 3,
            "label": "120000.0",
            "timeout_ms": 120000.5,
            "unknown": 1.0,
        }
        repaired, fields = mod._repair_adapted_namespace_tool_arguments(
            json.dumps(payload),
            {
                "type": "object",
                "properties": {
                    "flag": {"type": "boolean"},
                    "count": {"type": "integer"},
                    "label": {"type": "string"},
                    "timeout_ms": {"type": "integer"},
                },
            },
        )
        assert json.loads(repaired) == payload
        assert fields == []

    def test_leaves_nan_and_infinity_unchanged(self) -> None:
        payload = {"timeout_ms": float("nan")}
        repaired, fields = mod._repair_adapted_namespace_tool_arguments(
            payload,
            _WAIT_AGENT_INTEGER_SCHEMA,
        )
        assert repaired is payload
        assert fields == []
        payload = {"timeout_ms": float("inf")}
        repaired, fields = mod._repair_adapted_namespace_tool_arguments(
            payload,
            _WAIT_AGENT_INTEGER_SCHEMA,
        )
        assert repaired is payload
        assert fields == []

    def test_leaves_incomplete_json_string_unchanged(self) -> None:
        arguments = '{"timeout_ms": 120000.'
        repaired, fields = mod._repair_adapted_namespace_tool_arguments(
            arguments,
            _WAIT_AGENT_INTEGER_SCHEMA,
        )
        assert repaired == arguments
        assert fields == []

    def test_restore_item_repairs_existing_namespace(self, host_deps: None) -> None:
        item = {
            "type": "function_call",
            "name": "wait_agent",
            "namespace": "collaboration",
            "call_id": "call_wait",
            "arguments": json.dumps({"timeout_ms": 120000.0}),
        }
        records: list[dict[str, Any]] = []
        restored, count = mod._restore_adapted_namespace_tool_call_item(
            item,
            namespace_by_name={"wait_agent": "collaboration"},
            schema_by_name={"wait_agent": _WAIT_AGENT_INTEGER_SCHEMA},
            repair_records=records,
        )
        assert count == 1
        assert restored["namespace"] == "collaboration"
        assert restored["call_id"] == "call_wait"
        assert json.loads(restored["arguments"]) == {"timeout_ms": 120000}
        assert records == [{"name": "wait_agent", "fields": ["timeout_ms"]}]

    def test_response_body_repairs_timeout_and_records_bounded_metadata(
        self, host_deps: None
    ) -> None:
        request_body = {
            "tools": [
                {
                    "type": "namespace",
                    "name": "ns_adapter",
                    "tools": [
                        {
                            "type": "function",
                            "name": "tool_a",
                            "parameters": _WAIT_AGENT_INTEGER_SCHEMA,
                        }
                    ],
                }
            ]
        }
        body = {
            "output": [
                {
                    "type": "function_call",
                    "name": "tool_a",
                    "call_id": "call_wait",
                    "arguments": json.dumps({"timeout_ms": 120000.0}),
                }
            ]
        }
        result, count = mod._restore_adapted_namespace_tool_calls_in_response_body(
            body, request_body=request_body, adapter_model="test-model",
        )
        assert count == 1
        restored_item = result["output"][0]
        assert restored_item["namespace"] == "ns_adapter"
        assert restored_item["call_id"] == "call_wait"
        assert json.loads(restored_item["arguments"]) == {"timeout_ms": 120000}
        metadata = result["litellm_metadata"]
        assert metadata["codex_namespace_tool_argument_repair_count"] == 1
        assert metadata["codex_namespace_tool_argument_repair_names"] == ["tool_a"]
        assert metadata["codex_namespace_tool_argument_repair_fields"] == ["timeout_ms"]
        assert "120000" not in json.dumps(metadata)

    def test_stream_delta_is_not_rewritten(self, host_deps: None) -> None:
        payload = {
            "type": "response.function_call_arguments.delta",
            "name": "tool_a",
            "delta": '{"timeout_ms": 120000.',
        }
        result, count = mod._restore_adapted_namespace_tool_calls_in_stream_event_payload(
            payload,
            namespace_by_name={"tool_a": "ns_adapter"},
            schema_by_name={"tool_a": _WAIT_AGENT_INTEGER_SCHEMA},
        )
        assert count == 0
        assert result is payload
        assert result["delta"] == '{"timeout_ms": 120000.'

    def test_stream_arguments_done_is_repaired(self, host_deps: None) -> None:
        payload = {
            "type": "response.function_call_arguments.done",
            "name": "tool_a",
            "call_id": "call_wait",
            "arguments": json.dumps({"timeout_ms": 120000.0}),
        }
        result, count = mod._restore_adapted_namespace_tool_calls_in_stream_event_payload(
            payload,
            namespace_by_name={"tool_a": "ns_adapter"},
            schema_by_name={"tool_a": _WAIT_AGENT_INTEGER_SCHEMA},
        )
        assert count == 1
        assert result["call_id"] == "call_wait"
        assert json.loads(result["arguments"]) == {"timeout_ms": 120000}


# ===========================================================================
# SECTION 5 - Stream state keys and identity
# ===========================================================================


class TestStreamState:
    """Request-scoped stream state identity and late correlation."""

    def test_state_keys_from_item_and_payload(self) -> None:
        keys = mod._adapted_custom_tool_stream_state_keys(
            {"item_id": "i1", "output_index": 2},
            item={"call_id": "c1", "id": "id1"},
        )
        assert "id:c1" in keys
        assert "id:id1" in keys
        assert "id:i1" in keys
        assert "output:2" in keys

    def test_state_keys_deduped(self) -> None:
        keys = mod._adapted_custom_tool_stream_state_keys(
            {"call_id": "c1"},
            item={"call_id": "c1"},
        )
        assert keys.count("id:c1") == 1

    def test_remember_and_get_state(self) -> None:
        state_by_key: dict[str, dict[str, Any]] = {}
        item = {"call_id": "c1", "name": "my_tool"}
        state = mod._remember_adapted_custom_tool_stream_state(
            state_by_key,
            event_payload={"output_index": 0},
            item=item,
        )
        assert state["name"] == "my_tool"
        assert state["arguments"] == ""

        # Retrieve by output_index
        found = mod._get_adapted_custom_tool_stream_state(
            state_by_key, {"output_index": 0},
        )
        assert found is state

        # Retrieve by call_id
        found2 = mod._get_adapted_custom_tool_stream_state(
            state_by_key, {"call_id": "c1"},
        )
        assert found2 is state

    def test_state_identity_per_request(self) -> None:
        """Two separate state_by_key dicts are independent."""
        state_a: dict[str, dict[str, Any]] = {}
        state_b: dict[str, dict[str, Any]] = {}
        item = {"call_id": "c1", "name": "t"}
        mod._remember_adapted_custom_tool_stream_state(
            state_a, event_payload={}, item=item,
        )
        assert mod._get_adapted_custom_tool_stream_state(state_a, {"call_id": "c1"}) is not None
        assert mod._get_adapted_custom_tool_stream_state(state_b, {"call_id": "c1"}) is None


# ===========================================================================
# SECTION 6 - Stream event payload restore (custom)
# ===========================================================================


class TestCustomStreamEventPayload:
    """_restore_adapted_custom_tool_calls_in_stream_event_payload."""

    def test_output_item_added_restores(self, host_deps: None) -> None:
        state: dict[str, dict[str, Any]] = {}
        payload = {
            "type": "response.output_item.added",
            "item": {"type": "function_call", "name": "my_tool", "call_id": "c1"},
        }
        result, count = mod._restore_adapted_custom_tool_calls_in_stream_event_payload(
            payload,
            request_body={"tools": []},
            adapter_model="m",
            adapted_names={"my_tool"},
            state_by_key=state,
        )
        assert count == 1
        assert result is not None
        assert result["item"]["type"] == "custom_tool_call"
        assert result["item"]["input"] == ""
        # State was remembered
        assert mod._get_adapted_custom_tool_stream_state(state, {"call_id": "c1"}) is not None

    def test_delta_accumulates_and_returns_none(self, host_deps: None) -> None:
        state: dict[str, dict[str, Any]] = {}
        # Seed state
        mod._remember_adapted_custom_tool_stream_state(
            state,
            event_payload={"call_id": "c1"},
            item={"call_id": "c1", "name": "my_tool"},
        )
        payload = {
            "type": "response.function_call_arguments.delta",
            "call_id": "c1",
            "delta": '{"input":',
        }
        result, count = mod._restore_adapted_custom_tool_calls_in_stream_event_payload(
            payload,
            request_body={"tools": []},
            adapter_model="m",
            adapted_names={"my_tool"},
            state_by_key=state,
        )
        assert result is None
        assert count == 1
        st = mod._get_adapted_custom_tool_stream_state(state, {"call_id": "c1"})
        assert st["arguments"] == '{"input":'

    def test_late_correlation_done_event(self, host_deps: None) -> None:
        """Delta accumulation followed by done produces restored payload."""
        state: dict[str, dict[str, Any]] = {}
        mod._remember_adapted_custom_tool_stream_state(
            state,
            event_payload={"call_id": "c1"},
            item={"call_id": "c1", "name": "my_tool"},
        )
        # Accumulate deltas
        for delta in ['{"inp', 'ut": "he', 'llo"}']:
            mod._restore_adapted_custom_tool_calls_in_stream_event_payload(
                {
                    "type": "response.function_call_arguments.delta",
                    "call_id": "c1",
                    "delta": delta,
                },
                request_body={"tools": []},
                adapter_model="m",
                adapted_names={"my_tool"},
                state_by_key=state,
            )
        # Done event
        done_payload = {
            "type": "response.function_call_arguments.done",
            "call_id": "c1",
            "arguments": '{"input": "hello"}',
        }
        result, count = mod._restore_adapted_custom_tool_calls_in_stream_event_payload(
            done_payload,
            request_body={"tools": []},
            adapter_model="m",
            adapted_names={"my_tool"},
            state_by_key=state,
        )
        assert count == 1
        assert result is not None
        assert result["type"] == "response.custom_tool_call_input.done"
        assert result["input"] == "hello"
        assert "arguments" not in result

    def test_unrelated_event_passthrough(self, host_deps: None) -> None:
        state: dict[str, dict[str, Any]] = {}
        payload = {"type": "response.completed", "response": {"status": "completed"}}
        result, count = mod._restore_adapted_custom_tool_calls_in_stream_event_payload(
            payload,
            request_body={"tools": []},
            adapter_model="m",
            adapted_names={"my_tool"},
            state_by_key=state,
        )
        assert result is payload
        assert count == 0


# ===========================================================================
# SECTION 7 - SSE event block restore (custom)
# ===========================================================================


class TestCustomSSEEventBlock:
    """_restore_adapted_custom_tool_calls_in_sse_event_block."""

    def test_non_data_block_passthrough(self, host_deps: None) -> None:
        block = "event: ping\n"
        result, count = mod._restore_adapted_custom_tool_calls_in_sse_event_block(
            block,
            request_body={"tools": []},
            adapter_model="m",
            adapted_names={"my_tool"},
            state_by_key={},
        )
        assert result == block
        assert count == 0

    def test_done_marker_passthrough(self, host_deps: None) -> None:
        block = "data: [DONE]"
        result, count = mod._restore_adapted_custom_tool_calls_in_sse_event_block(
            block,
            request_body={"tools": []},
            adapter_model="m",
            adapted_names={"my_tool"},
            state_by_key={},
        )
        assert result == block
        assert count == 0

    def test_invalid_json_passthrough(self, host_deps: None) -> None:
        block = "data: {invalid"
        result, count = mod._restore_adapted_custom_tool_calls_in_sse_event_block(
            block,
            request_body={"tools": []},
            adapter_model="m",
            adapted_names={"my_tool"},
            state_by_key={},
        )
        assert result == block
        assert count == 0

    def test_delta_event_returns_none(self, host_deps: None) -> None:
        """Delta events are suppressed (return None) to avoid forwarding raw args."""
        state: dict[str, dict[str, Any]] = {}
        mod._remember_adapted_custom_tool_stream_state(
            state,
            event_payload={"call_id": "c1"},
            item={"call_id": "c1", "name": "my_tool"},
        )
        payload = {
            "type": "response.function_call_arguments.delta",
            "call_id": "c1",
            "delta": "x",
        }
        block = f"event: response.function_call_arguments.delta\ndata: {json.dumps(payload)}"
        result, count = mod._restore_adapted_custom_tool_calls_in_sse_event_block(
            block,
            request_body={"tools": []},
            adapter_model="m",
            adapted_names={"my_tool"},
            state_by_key=state,
        )
        assert result is None
        assert count == 1

    def test_repeated_data_lines_and_comment_passthrough(self, host_deps: None) -> None:
        payload_line_one = (
            '{"type":"response.output_item.added","item":{"type":"function_call",'
            '"name":"my_tool",'
        )
        payload_line_two = '"call_id":"my_tool_call","output_index":0}}'
        block = (
            ": keep-alive\r\n"
            "event: response.output_item.added\r\n"
            f"data: {payload_line_one}\r\n"
            f"data: {payload_line_two}\r\n"
            "\r\n"
        )

        result, count = mod._restore_adapted_custom_tool_calls_in_sse_event_block(
            block,
            request_body={"tools": []},
            adapter_model="m",
            adapted_names={"my_tool"},
            state_by_key={},
        )
        assert count == 1
        assert result is not None
        assert ": keep-alive" in result
        assert "custom_tool_call" in result
        assert result.count("data: ") == 1


# ===========================================================================
# SECTION 8 - Namespace stream event payload
# ===========================================================================


@pytest.mark.usefixtures("host_deps")
class TestNamespaceStreamEventPayload:
    """_restore_adapted_namespace_tool_calls_in_stream_event_payload."""

    def test_item_namespace_restored(self) -> None:
        payload = {
            "type": "response.output_item.added",
            "item": {"type": "function_call", "name": "tool_a"},
        }
        result, count = mod._restore_adapted_namespace_tool_calls_in_stream_event_payload(
            payload, namespace_by_name={"tool_a": "ns_adapter"},
        )
        assert count == 1
        assert result["item"]["namespace"] == "ns_adapter"
        assert result is not payload

    def test_response_output_restored(self) -> None:
        payload = {
            "type": "response.completed",
            "response": {
                "output": [{"type": "function_call", "name": "tool_a"}],
            },
        }
        result, count = mod._restore_adapted_namespace_tool_calls_in_stream_event_payload(
            payload, namespace_by_name={"tool_a": "ns_adapter"},
        )
        assert count == 1
        assert result["response"]["output"][0]["namespace"] == "ns_adapter"

    def test_no_match_passthrough(self) -> None:
        payload = {"type": "response.completed", "response": {"output": []}}
        result, count = mod._restore_adapted_namespace_tool_calls_in_stream_event_payload(
            payload, namespace_by_name={"tool_a": "ns_adapter"},
        )
        assert count == 0
        assert result is payload


# ===========================================================================
# SECTION 9 - Namespace SSE event block
# ===========================================================================


@pytest.mark.usefixtures("host_deps")
class TestNamespaceSSEEventBlock:
    """_restore_adapted_namespace_tool_calls_in_sse_event_block."""

    def test_restores_namespace_in_sse(self) -> None:
        payload = {
            "type": "response.output_item.added",
            "item": {"type": "function_call", "name": "tool_a"},
        }
        block = f"event: response.output_item.added\ndata: {json.dumps(payload)}"
        result, count = mod._restore_adapted_namespace_tool_calls_in_sse_event_block(
            block, namespace_by_name={"tool_a": "ns_adapter"},
        )
        assert count == 1
        assert "ns_adapter" in result

    def test_no_match_passthrough(self) -> None:
        block = "data: [DONE]"
        result, count = mod._restore_adapted_namespace_tool_calls_in_sse_event_block(
            block, namespace_by_name={"tool_a": "ns_adapter"},
        )
        assert result == block
        assert count == 0

    def test_repeated_data_lines_and_comment_passthrough(self) -> None:
        payload_line_one = (
            '{"type":"response.output_item.added","item":{"type":"function_call",'
            '"name":"tool_a",'
        )
        payload_line_two = '"output_index":0}}'
        block = (
            ": keep-alive\r\n"
            "event: response.output_item.added\r\n"
            f"data: {payload_line_one}\r\n"
            f"data: {payload_line_two}\r\n"
            "\r\n"
        )

        result, count = mod._restore_adapted_namespace_tool_calls_in_sse_event_block(
            block,
            namespace_by_name={"tool_a": "ns_adapter"},
        )
        assert count == 1
        assert result is not None
        assert ": keep-alive" in result
        assert "ns_adapter" in result
        assert result.count("data: ") == 1


# ===========================================================================
# SECTION 10 - Malformed tool call error
# ===========================================================================


class TestMalformedToolCall:
    """_raise_codex_auto_agent_malformed_adapted_custom_tool_call."""

    def test_raises_proxy_exception(self, host_deps: None) -> None:
        from litellm.proxy._types import ProxyException

        with pytest.raises(ProxyException) as exc_info:
            mod._raise_codex_auto_agent_malformed_adapted_custom_tool_call(
                response_body={"output": []},
                adapter_model="m",
                adapter="test_adapter",
                adapter_label="Test",
                adapter_error={"name": "my_tool", "reason": "bad"},
            )
        exc = exc_info.value
        assert "invalid" in exc.message
        detail = getattr(exc, "detail")
        assert detail["error"]["code"] == "aawm_auto_agent_malformed_tool_call_text"
        assert detail["error"]["status"] == "RESPONSES_MALFORMED_TOOL_CALL"
        assert detail["diagnostic"]["custom_tool_function_adapter_error"]["reason"] == "bad"

    def test_includes_stream_event_summaries(self, host_deps: None) -> None:
        from litellm.proxy._types import ProxyException

        summaries = [{"type": "delta", "index": 0}]
        with pytest.raises(ProxyException) as exc_info:
            mod._raise_codex_auto_agent_malformed_adapted_custom_tool_call(
                response_body={"output": []},
                adapter_model="m",
                adapter="test_adapter",
                adapter_label="Test",
                adapter_error={"name": "x", "reason": "y"},
                stream_event_summaries=summaries,
            )
        detail = getattr(exc_info.value, "detail")
        assert detail["diagnostic"]["custom_tool_function_adapter_error"]["reason"] == "y"


# ===========================================================================
# SECTION 11 - Repaired output item id
# ===========================================================================


class TestRepairedOutputItemId:
    """The canonical SSE-owned repaired output item ID host seam."""

    def test_prefers_id(self) -> None:
        assert sse_mod._responses_repaired_output_item_id({"id": "abc", "call_id": "def"}, 0) == "abc"

    def test_falls_back_to_call_id(self) -> None:
        assert sse_mod._responses_repaired_output_item_id({"call_id": "def"}, 3) == "def"

    def test_falls_back_to_index(self) -> None:
        assert sse_mod._responses_repaired_output_item_id({}, 7) == "item_7"

    def test_strips_whitespace(self) -> None:
        assert sse_mod._responses_repaired_output_item_id({"id": "  x  "}, 0) == "x"

    def test_skips_empty_string(self) -> None:
        assert sse_mod._responses_repaired_output_item_id({"id": "  ", "call_id": "y"}, 0) == "y"


# ===========================================================================
# SECTION 12 - Streaming response wrapper (custom)
# ===========================================================================


class TestCustomStreamingResponse:
    """_restore_adapted_custom_tool_calls_in_streaming_response."""

    def test_no_adapted_names_returns_original(self, host_deps: None) -> None:
        from fastapi.responses import StreamingResponse

        async def gen():
            yield b"data: {}\n\n"

        resp = StreamingResponse(gen(), media_type="text/event-stream")
        result = mod._restore_adapted_custom_tool_calls_in_streaming_response(
            resp, request_body=None, adapter_model="m",
        )
        assert result is resp

    def test_wraps_response_when_adapted(self, host_deps: None) -> None:
        from fastapi.responses import StreamingResponse

        async def gen():
            yield b"data: {}\n\n"

        resp = StreamingResponse(gen(), media_type="text/event-stream")
        result = mod._restore_adapted_custom_tool_calls_in_streaming_response(
            resp, request_body={"tools": []}, adapter_model="m",
        )
        assert result is not resp
        assert result.media_type == "text/event-stream"

    def test_end_to_end_stream_restore(self, host_deps: None) -> None:
        """Full SSE stream with added/delta/done produces restored output."""
        from fastapi.responses import StreamingResponse

        added = {
            "type": "response.output_item.added",
            "item": {"type": "function_call", "name": "my_tool", "call_id": "c1"},
            "output_index": 0,
        }
        delta = {
            "type": "response.function_call_arguments.delta",
            "call_id": "c1",
            "delta": '{"input": "hi"}',
            "output_index": 0,
        }
        done = {
            "type": "response.function_call_arguments.done",
            "call_id": "c1",
            "arguments": '{"input": "hi"}',
            "output_index": 0,
        }

        async def gen():
            yield f"event: response.output_item.added\ndata: {json.dumps(added)}\n\n".encode()
            yield f"event: response.function_call_arguments.delta\ndata: {json.dumps(delta)}\n\n".encode()
            yield f"event: response.function_call_arguments.done\ndata: {json.dumps(done)}\n\n".encode()

        resp = StreamingResponse(gen(), media_type="text/event-stream")
        result = mod._restore_adapted_custom_tool_calls_in_streaming_response(
            resp, request_body={"tools": []}, adapter_model="m",
        )

        async def collect():
            chunks = []
            async for chunk in result.body_iterator:
                chunks.append(chunk)
            return chunks

        chunks = asyncio.run(collect())
        combined = "".join(chunks)
        # The added event should have custom_tool_call type
        assert "custom_tool_call" in combined
        # The done event should have response.custom_tool_call_input.done
        assert "response.custom_tool_call_input.done" in combined
        # Delta should be suppressed (None return)
        assert "function_call_arguments.delta" not in combined

    def test_stream_restore_with_crlf_split_and_utf8_boundary_and_final_unterminated_block(self, host_deps: None) -> None:
        from fastapi.responses import StreamingResponse

        payload = {
            "type": "response.output_item.added",
            "item": {
                "type": "function_call",
                "name": "my_tool",
                "call_id": "tool🚀",
            },
            "output_index": 0,
        }
        event_block = (
            ": keep-alive\r\n"
            "event: response.output_item.added\r\n"
            f"data: {json.dumps(payload, ensure_ascii=False)}\r"
        )
        event_bytes = event_block.encode("utf-8")
        rocket_bytes = "🚀".encode("utf-8")
        split_boundary = event_bytes.index(b"\r") + 1
        emoji_at = event_bytes.index(rocket_bytes) if rocket_bytes in event_bytes else split_boundary
        split_emoji = min(len(event_bytes), emoji_at + 2)

        async def gen():
            yield event_bytes[:split_boundary]
            yield event_bytes[split_boundary:split_emoji]
            yield event_bytes[split_emoji:]

        resp = StreamingResponse(gen(), media_type="text/event-stream")
        result = mod._restore_adapted_custom_tool_calls_in_streaming_response(
            resp, request_body={"tools": []}, adapter_model="m",
        )

        async def collect():
            chunks = []
            async for chunk in result.body_iterator:
                chunks.append(chunk)
            return chunks

        chunks = asyncio.run(collect())
        combined = "".join(chunks)
        assert "custom_tool_call" in combined
        assert "tool🚀" in combined
        # final unterminated block should not be double-terminated
        assert not combined.endswith("\n\n")
        assert ": keep-alive" in combined


# ===========================================================================
# SECTION 13 - Namespace streaming response wrapper
# ===========================================================================


class TestNamespaceStreamingResponse:
    """_restore_adapted_namespace_tool_calls_in_streaming_response."""

    def test_no_namespace_returns_original(self, host_deps: None) -> None:
        from fastapi.responses import StreamingResponse

        async def gen():
            yield b"data: {}\n\n"

        resp = StreamingResponse(gen(), media_type="text/event-stream")
        result = mod._restore_adapted_namespace_tool_calls_in_streaming_response(
            resp, request_body=None, adapter_model="m",
        )
        assert result is resp

    def test_wraps_when_namespace_present(self, host_deps: None) -> None:
        from fastapi.responses import StreamingResponse

        async def gen():
            yield b"data: {}\n\n"

        resp = StreamingResponse(gen(), media_type="text/event-stream")
        result = mod._restore_adapted_namespace_tool_calls_in_streaming_response(
            resp, request_body={"tools": []}, adapter_model="test-model",
        )
        assert result is not resp

    def test_stream_restore_preserves_crlf_comment_and_final_unterminated_block(self, host_deps: None) -> None:
        from fastapi.responses import StreamingResponse

        payload = {
            "type": "response.output_item.added",
            "item": {
                "type": "function_call",
                "name": "tool_a",
            },
            "output_index": 0,
        }
        event_block = (
            ": keep-alive\r\n"
            "event: response.output_item.added\r\n"
            f"data: {json.dumps(payload)}\r"
        )
        event_bytes = event_block.encode("utf-8")
        split_boundary = event_bytes.index(b"\r") + 1

        async def gen():
            yield event_bytes[:split_boundary]
            yield event_bytes[split_boundary:]

        resp = StreamingResponse(gen(), media_type="text/event-stream")
        result = mod._restore_adapted_namespace_tool_calls_in_streaming_response(
            resp, request_body={"tools": []}, adapter_model="test-model",
        )

        async def collect():
            chunks = []
            async for chunk in result.body_iterator:
                chunks.append(chunk)
            return chunks

        chunks = asyncio.run(collect())
        combined = "".join(chunks)
        assert ": keep-alive" in combined
        assert "ns_adapter" in combined
        assert not combined.endswith("\n\n")
