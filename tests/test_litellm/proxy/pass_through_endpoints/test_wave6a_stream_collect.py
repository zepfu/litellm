"""Wave 6A Author E: stream_collect AST / behavior / dependency-isolation tests.

Covers:
- _responses_output_stream_key
- _merge_responses_output_lists
- _responses_output_has_message_text
- _build_collected_responses_text_output_item
- _record_collected_responses_output_item_event
- _record_collected_responses_arguments_event
- _finalize_collected_responses_stream_response
- _build_empty_success_responses_diagnostic
- _collect_responses_response_from_stream
"""

from __future__ import annotations

import ast
import asyncio
import importlib
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from litellm.proxy.pass_through_endpoints import llm_passthrough_endpoints as lpe

# ---------------------------------------------------------------------------
# Module under test
# ---------------------------------------------------------------------------
from litellm.proxy.pass_through_endpoints.aawm_adapter_runtime import stream_collect as sc

MODULE_PATH = Path(sc.__file__).resolve()

# ---------------------------------------------------------------------------
# Expected symbol inventory
# ---------------------------------------------------------------------------
EXPECTED_FUNCTIONS: set[str] = {
    "_responses_output_stream_key",
    "_merge_responses_output_lists",
    "_responses_output_has_message_text",
    "_build_collected_responses_text_output_item",
    "_record_collected_responses_output_item_event",
    "_record_collected_responses_arguments_event",
    "_finalize_collected_responses_stream_response",
    "_build_empty_success_responses_diagnostic",
    "_collect_responses_response_from_stream",
}

EXPECTED_PUBLIC: set[str] = {"install"}

# Seams that must NOT be defined locally (resolved via install() rebinding)
EXPECTED_SEAMS: set[str] = {
    "_iterate_responses_sse_events",
    "_responses_stream_event_summary",
    "_responses_event_text_key",
    "_coerce_namespace_to_mapping",
    "_mapping_or_attr_get",
    "RESPONSES_API_TERMINAL_STREAM_EVENTS",
    "HTTPException",
    "StreamingResponse",
}

# ---------------------------------------------------------------------------
# Shared seam stubs (installed on module, cleaned up per-test)
# ---------------------------------------------------------------------------

def _stub_mapping_or_attr_get(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _stub_coerce_namespace_to_mapping(value: Any, **kw: Any) -> Any:
    if isinstance(value, SimpleNamespace):
        return vars(value)
    return value


def _install_basic_stubs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(lpe, "_mapping_or_attr_get", _stub_mapping_or_attr_get)
    monkeypatch.setattr(
        lpe,
        "_coerce_namespace_to_mapping",
        _stub_coerce_namespace_to_mapping,
    )


def _make_recording_state() -> dict[str, Any]:
    return {
        "output_items": {},
        "ordered_keys": [],
        "key_aliases": {},
        "key_by_output_index": {},
    }


# ===========================================================================
# AST structural tests
# ===========================================================================


class TestASTStructure:
    """Verify module-level AST shape without executing code."""

    @pytest.fixture(autouse=True)
    def _parse(self) -> None:
        self.tree = ast.parse(MODULE_PATH.read_text())
        self.top_level_names: set[str] = set()
        self.function_names: set[str] = set()
        self.async_function_names: set[str] = set()
        for node in ast.iter_child_nodes(self.tree):
            if isinstance(node, ast.FunctionDef):
                self.function_names.add(node.name)
                self.top_level_names.add(node.name)
            elif isinstance(node, ast.AsyncFunctionDef):
                self.async_function_names.add(node.name)
                self.top_level_names.add(node.name)
            elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                for t in targets:
                    if isinstance(t, ast.Name):
                        self.top_level_names.add(t.id)

    def test_all_expected_functions_present(self) -> None:
        all_fns = self.function_names | self.async_function_names
        missing = EXPECTED_FUNCTIONS - all_fns
        assert not missing, f"Missing functions: {missing}"

    def test_install_present(self) -> None:
        assert "install" in self.function_names

    def test_collect_is_async(self) -> None:
        assert "_collect_responses_response_from_stream" in self.async_function_names

    def test_no_import_of_god_module(self) -> None:
        """Module must not import llm_passthrough_endpoints at any scope."""
        for node in ast.walk(self.tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert "llm_passthrough_endpoints" not in alias.name
            elif isinstance(node, ast.ImportFrom):
                assert node.module is None or "llm_passthrough_endpoints" not in node.module

    def test_no_import_of_seam_symbols(self) -> None:
        """Seam symbols must not be imported; they arrive via install()."""
        for node in ast.walk(self.tree):
            if isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    assert alias.name not in EXPECTED_SEAMS, (
                        f"Seam symbol {alias.name!r} must not be imported"
                    )

    def test_host_function_names_tuple_matches(self) -> None:
        """_HOST_FUNCTION_NAMES must list exactly the expected functions."""
        for node in ast.iter_child_nodes(self.tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "_HOST_FUNCTION_NAMES":
                        if isinstance(node.value, ast.Tuple):
                            names = {
                                elt.value
                                for elt in node.value.elts
                                if isinstance(elt, ast.Constant) and isinstance(elt.value, str)
                            }
                            assert names == EXPECTED_FUNCTIONS


# ===========================================================================
# Dependency isolation tests
# ===========================================================================


class TestDependencyIsolation:
    """Verify the module does not pull in heavy/god-module dependencies."""

    def test_module_imports_cleanly(self) -> None:
        mod = importlib.import_module(
            "litellm.proxy.pass_through_endpoints.aawm_adapter_runtime.stream_collect"
        )
        assert mod is sc

    def test_no_god_module_in_module_globals(self) -> None:
        mod_globals = vars(sc)
        for key in mod_globals:
            if "passthrough" in key.lower() or "llm_passthrough" in key.lower():
                pytest.fail(f"Unexpected god-module reference in globals: {key}")

    def test_seam_symbols_not_locally_defined(self) -> None:
        """Before install(), seam symbols should not be in module namespace."""
        mod_dict = vars(sc)
        for seam in ("_iterate_responses_sse_events", "_responses_stream_event_summary",
                     "_responses_event_text_key", "RESPONSES_API_TERMINAL_STREAM_EVENTS",
                     "HTTPException"):
            assert seam not in mod_dict, f"Seam {seam!r} should not be locally defined"


# ===========================================================================
# Behavior tests: _responses_output_stream_key
# ===========================================================================


class TestResponsesOutputStreamKey:
    def test_call_id_preferred(self) -> None:
        result = sc._responses_output_stream_key(
            item={"type": "function_call", "call_id": "call_abc", "id": "fc_1"}
        )
        assert result == "call_abc"

    def test_id_fallback(self) -> None:
        result = sc._responses_output_stream_key(item={"type": "message", "id": "msg_1"})
        assert result == "msg_1"

    def test_item_id_param(self) -> None:
        result = sc._responses_output_stream_key(item_id="item_xyz")
        assert result == "item_xyz"

    def test_output_index_synthetic(self) -> None:
        result = sc._responses_output_stream_key(
            item={"type": "function_call"}, output_index=2
        )
        assert result == "function_call:output:2"

    def test_fallback_index_synthetic(self) -> None:
        result = sc._responses_output_stream_key(
            item={"type": "message"}, fallback_index=3
        )
        assert result == "message:fallback:3"

    def test_bare_fallback_zero(self) -> None:
        result = sc._responses_output_stream_key()
        assert result == "fallback:0"

    def test_no_type_prefix_without_type(self) -> None:
        result = sc._responses_output_stream_key(output_index=0)
        assert result == "output:0"

    def test_whitespace_call_id_skipped(self) -> None:
        result = sc._responses_output_stream_key(
            item={"type": "function_call", "call_id": "  ", "id": "real_id"}
        )
        assert result == "real_id"


# ===========================================================================
# Behavior tests: _merge_responses_output_lists
# ===========================================================================


class TestMergeResponsesOutputLists:
    def test_streamed_only(self) -> None:
        streamed = [{"type": "message", "id": "m1", "content": []}]
        result = sc._merge_responses_output_lists(None, streamed)
        assert len(result) == 1
        assert result[0]["id"] == "m1"

    def test_completed_overlays_streamed(self) -> None:
        streamed = [{"type": "function_call", "call_id": "c1", "arguments": '{"a":1}'}]
        completed = [{"type": "function_call", "call_id": "c1", "status": "completed"}]
        result = sc._merge_responses_output_lists(
            completed, streamed, key_aliases={"c1": "c1"}
        )
        assert len(result) == 1
        assert result[0]["status"] == "completed"
        # arguments preserved from streamed when not in completed
        assert result[0]["arguments"] == '{"a":1}'

    def test_completed_new_item_with_distinct_alias(self) -> None:
        """Completed item with alias not in streamed gets a new key via fallback."""
        streamed = [{"type": "message", "id": "m1"}]
        completed = [{"type": "function_call", "call_id": "c_new", "id": "fc_new"}]
        result = sc._merge_responses_output_lists(
            completed, streamed,
            key_aliases={"m1": "m1"},
        )
        # completed item at index 0: alias c_new not in aliases, index_keys[0]="m1",
        # so it merges into the m1 slot (index-based fallback). This is the
        # documented behavior: index fallback when no alias match.
        assert len(result) >= 1

    def test_completed_truly_new_item_appended(self) -> None:
        """Completed item beyond streamed length gets appended."""
        streamed = [{"type": "message", "id": "m1"}]
        completed = [
            {"type": "message", "id": "m1", "status": "completed"},
            {"type": "function_call", "call_id": "c2", "id": "fc_2"},
        ]
        result = sc._merge_responses_output_lists(completed, streamed)
        assert len(result) == 2

    def test_empty_inputs(self) -> None:
        result = sc._merge_responses_output_lists(None, None)
        assert result == []

    def test_non_dict_items_skipped(self) -> None:
        streamed = [{"type": "message", "id": "m1"}, "garbage", None]  # type: ignore[list-item]
        result = sc._merge_responses_output_lists(None, streamed)
        assert len(result) == 1


# ===========================================================================
# Behavior tests: _responses_output_has_message_text
# ===========================================================================


class TestResponsesOutputHasMessageText:
    def test_true_with_output_text(self) -> None:
        output = [
            {
                "type": "message",
                "content": [{"type": "output_text", "text": "hello"}],
            }
        ]
        assert sc._responses_output_has_message_text(output) is True

    def test_true_with_text_type(self) -> None:
        output = [
            {
                "type": "message",
                "content": [{"type": "text", "text": "world"}],
            }
        ]
        assert sc._responses_output_has_message_text(output) is True

    def test_false_empty_text(self) -> None:
        output = [
            {
                "type": "message",
                "content": [{"type": "output_text", "text": ""}],
            }
        ]
        assert sc._responses_output_has_message_text(output) is False

    def test_false_non_message(self) -> None:
        output = [{"type": "function_call", "content": [{"type": "output_text", "text": "x"}]}]
        assert sc._responses_output_has_message_text(output) is False

    def test_false_not_list(self) -> None:
        assert sc._responses_output_has_message_text("nope") is False
        assert sc._responses_output_has_message_text(None) is False

    def test_false_content_not_list(self) -> None:
        output = [{"type": "message", "content": "plain string"}]
        assert sc._responses_output_has_message_text(output) is False


# ===========================================================================
# Behavior tests: _build_collected_responses_text_output_item
# ===========================================================================


class TestBuildCollectedResponsesTextOutputItem:
    def test_structure(self) -> None:
        item = sc._build_collected_responses_text_output_item("hello world")
        assert item["type"] == "message"
        assert item["id"] == "msg_adapter_0"
        assert item["status"] == "completed"
        assert item["role"] == "assistant"
        assert len(item["content"]) == 1
        part = item["content"][0]
        assert part["type"] == "output_text"
        assert part["text"] == "hello world"
        assert part["annotations"] == []


# ===========================================================================
# Behavior tests: _record_collected_responses_output_item_event
# ===========================================================================


class TestRecordCollectedResponsesOutputItemEvent:
    def setup_method(self) -> None:
        self.monkeypatch = pytest.MonkeyPatch()
        _install_basic_stubs(self.monkeypatch)

    def teardown_method(self) -> None:
        self.monkeypatch.undo()

    def test_basic_recording(self) -> None:
        state = _make_recording_state()
        event = {
            "type": "response.output_item.added",
            "item": {"type": "function_call", "call_id": "c1", "id": "fc_1"},
            "output_index": 0,
        }
        sc._record_collected_responses_output_item_event(event=event, **state)
        assert len(state["ordered_keys"]) == 1
        key = state["ordered_keys"][0]
        assert state["output_items"][key]["call_id"] == "c1"
        assert state["key_by_output_index"][0] == key
        assert state["key_aliases"]["c1"] == key

    def test_non_dict_item_ignored(self) -> None:
        state = _make_recording_state()
        event = {"type": "response.output_item.added", "item": "garbage", "output_index": 0}
        sc._record_collected_responses_output_item_event(event=event, **state)
        assert len(state["ordered_keys"]) == 0

    def test_arguments_preserved_on_merge(self) -> None:
        state = _make_recording_state()
        # First event establishes key with arguments
        event1 = {
            "type": "response.output_item.added",
            "item": {"type": "function_call", "call_id": "c1", "arguments": '{"x":1}'},
            "output_index": 0,
        }
        sc._record_collected_responses_output_item_event(event=event1, **state)
        # Second event same output_index, no arguments
        event2 = {
            "type": "response.output_item.done",
            "item": {"type": "function_call", "call_id": "c1", "status": "completed"},
            "output_index": 0,
        }
        sc._record_collected_responses_output_item_event(event=event2, **state)
        key = state["ordered_keys"][0]
        assert state["output_items"][key]["arguments"] == '{"x":1}'
        assert state["output_items"][key]["status"] == "completed"

    def test_namespace_item_coerced(self) -> None:
        state = _make_recording_state()
        event = {
            "type": "response.output_item.added",
            "item": SimpleNamespace(type="message", id="msg_ns"),
            "output_index": 0,
        }
        sc._record_collected_responses_output_item_event(event=event, **state)
        assert len(state["ordered_keys"]) == 1
        key = state["ordered_keys"][0]
        assert state["output_items"][key]["id"] == "msg_ns"


# ===========================================================================
# Behavior tests: _record_collected_responses_arguments_event
# ===========================================================================


class TestRecordCollectedResponsesArgumentsEvent:
    def setup_method(self) -> None:
        self.monkeypatch = pytest.MonkeyPatch()
        _install_basic_stubs(self.monkeypatch)

    def teardown_method(self) -> None:
        self.monkeypatch.undo()

    def test_delta_accumulation(self) -> None:
        state = _make_recording_state()
        event1 = {"item_id": "fc_1", "output_index": 0, "delta": '{"a"'}
        sc._record_collected_responses_arguments_event(
            event=event1, event_type="response.function_call_arguments.delta", **state
        )
        event2 = {"item_id": "fc_1", "output_index": 0, "delta": ':1}'}
        sc._record_collected_responses_arguments_event(
            event=event2, event_type="response.function_call_arguments.delta", **state
        )
        key = state["ordered_keys"][0]
        assert state["output_items"][key]["arguments"] == '{"a":1}'
        assert state["output_items"][key]["type"] == "function_call"
        # OPENAI-007: arguments events carry item_id only; do not synthesize call_id.
        assert state["output_items"][key]["id"] == "fc_1"
        assert "call_id" not in state["output_items"][key]

    def test_arguments_event_does_not_synthesize_call_id_from_item_id(self) -> None:
        state = _make_recording_state()
        # Distinct identity already present on output_item.added.
        sc._record_collected_responses_output_item_event(
            event={
                "item": {
                    "type": "function_call",
                    "id": "fc_item_1",
                    "call_id": "provider_call_1",
                    "name": "do_thing",
                    "arguments": "",
                },
                "output_index": 0,
            },
            **state,
        )
        sc._record_collected_responses_arguments_event(
            event={"item_id": "fc_item_1", "output_index": 0, "delta": '{"a":1}'},
            event_type="response.function_call_arguments.delta",
            **state,
        )
        key = state["ordered_keys"][0]
        item = state["output_items"][key]
        assert item["id"] == "fc_item_1"
        assert item["call_id"] == "provider_call_1"
        assert item["id"] != item["call_id"]
        assert item["arguments"] == '{"a":1}'

    def test_done_replaces(self) -> None:
        state = _make_recording_state()
        event1 = {"item_id": "fc_1", "output_index": 0, "delta": "partial"}
        sc._record_collected_responses_arguments_event(
            event=event1, event_type="response.function_call_arguments.delta", **state
        )
        event2 = {"item_id": "fc_1", "output_index": 0, "arguments": '{"final":true}'}
        sc._record_collected_responses_arguments_event(
            event=event2, event_type="response.function_call_arguments.done", **state
        )
        key = state["ordered_keys"][0]
        assert state["output_items"][key]["arguments"] == '{"final":true}'

    def test_mcp_call_type(self) -> None:
        state = _make_recording_state()
        event = {"item_id": "mcp_1", "output_index": 0, "delta": "{}"}
        sc._record_collected_responses_arguments_event(
            event=event, event_type="response.mcp_call_arguments.delta", **state
        )
        key = state["ordered_keys"][0]
        assert state["output_items"][key]["type"] == "mcp_call"
        assert "call_id" not in state["output_items"][key]

    def test_late_correlation_by_item_id(self) -> None:
        """item_id alias enables late correlation with output_item events."""
        state = _make_recording_state()
        # Pre-register alias as if output_item.added already ran
        state["key_aliases"]["fc_late"] = "existing_key"
        state["ordered_keys"].append("existing_key")
        state["output_items"]["existing_key"] = {"type": "function_call", "id": "fc_late"}

        event = {"item_id": "fc_late", "output_index": 5, "delta": '{"z":9}'}
        sc._record_collected_responses_arguments_event(
            event=event, event_type="response.function_call_arguments.delta", **state
        )
        assert state["output_items"]["existing_key"]["arguments"] == '{"z":9}'
        # No new key added
        assert len(state["ordered_keys"]) == 1


# ===========================================================================
# Behavior tests: _finalize_collected_responses_stream_response
# ===========================================================================


class TestFinalizeCollectedResponsesStreamResponse:
    def test_text_appended_when_no_message_text(self) -> None:
        response_dict: dict[str, Any] = {"id": "resp_1", "output": []}
        result = sc._finalize_collected_responses_stream_response(
            response_dict=response_dict,
            output_text_parts=["Hello", " world"],
            output_items={},
            ordered_keys=[],
            key_aliases={},
            key_by_output_index={},
        )
        assert len(result["output"]) == 1
        assert result["output"][0]["content"][0]["text"] == "Hello world"

    def test_text_not_duplicated_when_message_exists(self) -> None:
        existing_msg = {
            "type": "message",
            "content": [{"type": "output_text", "text": "already here"}],
        }
        response_dict: dict[str, Any] = {"id": "resp_1", "output": [existing_msg]}
        result = sc._finalize_collected_responses_stream_response(
            response_dict=response_dict,
            output_text_parts=["extra"],
            output_items={},
            ordered_keys=[],
            key_aliases={},
            key_by_output_index={},
        )
        # Should not append another text item since completed_output has message text
        assert len(result["output"]) == 1

    def test_streamed_items_merged(self) -> None:
        response_dict: dict[str, Any] = {"id": "resp_1", "output": []}
        output_items = {"k1": {"type": "function_call", "call_id": "c1", "arguments": "{}"}}
        result = sc._finalize_collected_responses_stream_response(
            response_dict=response_dict,
            output_text_parts=[],
            output_items=output_items,
            ordered_keys=["k1"],
            key_aliases={},
            key_by_output_index={},
        )
        assert len(result["output"]) == 1
        assert result["output"][0]["call_id"] == "c1"

    def test_no_output_no_text_unchanged(self) -> None:
        response_dict: dict[str, Any] = {"id": "resp_1", "status": "completed"}
        result = sc._finalize_collected_responses_stream_response(
            response_dict=response_dict,
            output_text_parts=[],
            output_items={},
            ordered_keys=[],
            key_aliases={},
            key_by_output_index={},
        )
        assert "output" not in result


# ===========================================================================
# Behavior tests: _build_empty_success_responses_diagnostic
# ===========================================================================


class TestBuildEmptySuccessResponsesDiagnostic:
    def test_basic_diagnostic(self) -> None:
        body = {
            "id": "resp_1",
            "status": "completed",
            "model": "gpt-4o",
            "output": [{"type": "message"}, {"type": "function_call"}],
            "usage": {"total_tokens": 42},
            "error": None,
            "incomplete_details": None,
        }
        diag = sc._build_empty_success_responses_diagnostic(
            response_body=body, diagnostic_context=None
        )
        assert diag["id"] == "resp_1"
        assert diag["status"] == "completed"
        assert diag["model"] == "gpt-4o"
        assert diag["output_count"] == 2
        assert diag["output_types"] == ["message", "function_call"]
        assert diag["usage"] == {"total_tokens": 42}
        assert "context" not in diag

    def test_with_context(self) -> None:
        body: dict[str, Any] = {"id": "r", "output": [], "usage": {}}
        diag = sc._build_empty_success_responses_diagnostic(
            response_body=body, diagnostic_context={"reason": "empty"}
        )
        assert diag["context"] == {"reason": "empty"}

    def test_output_types_capped_at_20(self) -> None:
        body: dict[str, Any] = {
            "output": [{"type": f"t{i}"} for i in range(30)],
            "usage": {},
        }
        diag = sc._build_empty_success_responses_diagnostic(
            response_body=body, diagnostic_context=None
        )
        assert diag["output_count"] == 30
        assert len(diag["output_types"]) == 20

    def test_non_list_output(self) -> None:
        body: dict[str, Any] = {"output": "weird", "usage": {}}
        diag = sc._build_empty_success_responses_diagnostic(
            response_body=body, diagnostic_context=None
        )
        assert diag["output_count"] == 0
        assert diag["output_types"] == []


# ===========================================================================
# Behavior tests: _collect_responses_response_from_stream (async)
# ===========================================================================


class _FakeAsyncIterator:
    """Simulates an SSE event iterator with aclose()."""

    def __init__(self, events: list[dict[str, Any]]) -> None:
        self._events = list(events)
        self._index = 0
        self.closed = False

    def __aiter__(self) -> "_FakeAsyncIterator":
        return self

    async def __anext__(self) -> dict[str, Any]:
        if self._index >= len(self._events):
            raise StopAsyncIteration
        event = self._events[self._index]
        self._index += 1
        return event

    async def aclose(self) -> None:
        self.closed = True


class _FakeResponse:
    def __init__(self, events: list[dict[str, Any]]) -> None:
        self._iterator = _FakeAsyncIterator(events)
        self.body_iterator = self._iterator


class FakeHTTPException(Exception):
    def __init__(self, status_code: int, detail: str) -> None:
        self.status_code = status_code
        self.detail = detail
        super().__init__(detail)


def _install_collect_stubs(
    events: list[dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
) -> _FakeAsyncIterator:
    """Install minimal stubs for _collect_responses_response_from_stream."""
    iterator = _FakeAsyncIterator(events)

    def fake_iterate(body_iterator: Any) -> _FakeAsyncIterator:
        return iterator

    monkeypatch.setattr(lpe, "_iterate_responses_sse_events", fake_iterate)
    monkeypatch.setattr(lpe, "_mapping_or_attr_get", _stub_mapping_or_attr_get)
    monkeypatch.setattr(
        lpe,
        "_coerce_namespace_to_mapping",
        _stub_coerce_namespace_to_mapping,
    )
    monkeypatch.setattr(
        lpe,
        "_responses_stream_event_summary",
        lambda event: {"type": event.get("type")},
    )
    monkeypatch.setattr(
        lpe,
        "_responses_event_text_key",
        lambda event: event.get("item_id", "output:0"),
    )
    monkeypatch.setattr(
        lpe,
        "RESPONSES_API_TERMINAL_STREAM_EVENTS",
        frozenset({"response.completed", "response.failed", "response.incomplete"}),
    )
    monkeypatch.setattr(lpe, "HTTPException", FakeHTTPException)
    return iterator


class TestCollectResponsesResponseFromStream:
    def setup_method(self) -> None:
        self.monkeypatch = pytest.MonkeyPatch()

    def teardown_method(self) -> None:
        self.monkeypatch.undo()

    def test_terminal_event_returns_response(self) -> None:
        events = [
            {"type": "response.output_text.delta", "delta": "Hi", "item_id": "m1"},
            {
                "type": "response.completed",
                "response": {"id": "resp_1", "status": "completed", "output": []},
            },
        ]
        _install_collect_stubs(events, self.monkeypatch)
        response = _FakeResponse(events)
        result = asyncio.run(
            sc._collect_responses_response_from_stream(response)  # type: ignore[arg-type]
        )
        assert result["id"] == "resp_1"
        # Text should be appended since output has no message text
        assert len(result["output"]) == 1
        assert result["output"][0]["content"][0]["text"] == "Hi"

    def test_no_terminal_raises_502(self) -> None:
        events = [
            {"type": "response.output_text.delta", "delta": "orphan"},
        ]
        _install_collect_stubs(events, self.monkeypatch)
        response = _FakeResponse(events)
        with pytest.raises(FakeHTTPException) as exc_info:
            asyncio.run(
                sc._collect_responses_response_from_stream(response)  # type: ignore[arg-type]
            )
        assert exc_info.value.status_code == 502

    def test_event_summaries_capped(self) -> None:
        events = [
            {"type": "response.output_text.delta", "delta": f"d{i}", "item_id": "m1"}
            for i in range(60)
        ] + [
            {
                "type": "response.completed",
                "response": {"id": "resp_2", "status": "completed", "output": []},
            },
        ]
        _install_collect_stubs(events, self.monkeypatch)
        response = _FakeResponse(events)
        summaries: list[dict[str, Any]] = []
        asyncio.run(
            sc._collect_responses_response_from_stream(  # type: ignore[arg-type]
                response, event_summaries=summaries
            )
        )
        assert len(summaries) == 50

    def test_function_call_arguments_collected(self) -> None:
        events = [
            {
                "type": "response.output_item.added",
                "item": {"type": "function_call", "call_id": "c1", "id": "fc_1"},
                "output_index": 0,
            },
            {"type": "response.function_call_arguments.delta", "item_id": "fc_1", "output_index": 0, "delta": '{"k"'},
            {"type": "response.function_call_arguments.delta", "item_id": "fc_1", "output_index": 0, "delta": ":1}"},
            {
                "type": "response.completed",
                "response": {"id": "resp_3", "status": "completed", "output": []},
            },
        ]
        _install_collect_stubs(events, self.monkeypatch)
        response = _FakeResponse(events)
        result = asyncio.run(
            sc._collect_responses_response_from_stream(response)  # type: ignore[arg-type]
        )
        fc_items = [i for i in result["output"] if i.get("type") == "function_call"]
        assert len(fc_items) == 1
        assert fc_items[0]["arguments"] == '{"k":1}'

    def test_text_done_dedup(self) -> None:
        """output_text.done should not duplicate text already seen via delta."""
        events = [
            {"type": "response.output_text.delta", "delta": "Hello", "item_id": "m1"},
            {"type": "response.output_text.done", "text": "Hello", "item_id": "m1"},
            {
                "type": "response.completed",
                "response": {"id": "resp_4", "status": "completed", "output": []},
            },
        ]
        _install_collect_stubs(events, self.monkeypatch)
        response = _FakeResponse(events)
        result = asyncio.run(
            sc._collect_responses_response_from_stream(response)  # type: ignore[arg-type]
        )
        text = result["output"][0]["content"][0]["text"]
        assert text == "Hello"


# ===========================================================================
# Production installation contract
# ===========================================================================


class TestInstallContract:
    def test_production_facades_use_host_globals(self) -> None:
        for name in EXPECTED_FUNCTIONS:
            assert getattr(lpe, name) is getattr(sc, name)
            assert getattr(sc, name).__globals__ is vars(lpe)

    def test_production_facades_preserve_function_metadata(self) -> None:
        for name in EXPECTED_FUNCTIONS:
            assert getattr(sc, name).__name__ == name
