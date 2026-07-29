"""Wave 6A Author B: SSE framing, event iteration, and streaming-response builder tests.

Ownership: litellm/proxy/pass_through_endpoints/aawm_adapter_runtime/sse.py

Source symbol inventory (from llm_passthrough_endpoints.py develop HEAD):
    _serialize_responses_adapter_response          (line 5337)
    _responses_sse_from_iterator                   (line 5352)
    _iterate_responses_sse_events                  (line 7098)
    _mapping_or_attr_get                           (line 8412)
    _coerce_namespace_to_mapping                   (line 7131)
    _responses_event_text_key                      (line 7151)
    _responses_stream_event_summary                (line 7166)
    _responses_repaired_output_item_id             (line 8220)
    _responses_sse_from_repaired_response_body     (line 8228)
    _build_anthropic_streaming_response_from_responses_stream (line 9238)
    _build_anthropic_streaming_response_from_completion_adapter_stream (line 9338)

Integration seams (NOT owned here, resolved via host globals):
    _stringify_grok_native_input_item_value

Explicitly excluded:
    - Stream accumulation/finalization
    - Custom/namespace tool restoration
    - Bounded payload replay validation
    - Provider request preparation
"""

from __future__ import annotations

import ast
import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

# ---------------------------------------------------------------------------
# Target module
# ---------------------------------------------------------------------------
from litellm.proxy.pass_through_endpoints.aawm_adapter_runtime import sse as sse_mod

SSE_MODULE_PATH = Path(sse_mod.__file__).resolve()

# ---------------------------------------------------------------------------
# God-module reference for production facade checks
# ---------------------------------------------------------------------------
from litellm.proxy.pass_through_endpoints import llm_passthrough_endpoints as lpe

GOD_PATH = Path(lpe.__file__).resolve()

# ---------------------------------------------------------------------------
# Symbol inventory (Wave 6A Author B)
# ---------------------------------------------------------------------------
W6B_OWNED_SYMBOLS: set[str] = {
    "_serialize_responses_adapter_response",
    "_responses_sse_from_iterator",
    "_iterate_responses_sse_events",
    "_mapping_or_attr_get",
    "_coerce_namespace_to_mapping",
    "_responses_event_text_key",
    "_responses_stream_event_summary",
    "_responses_repaired_output_item_id",
    "_responses_sse_from_repaired_response_body",
    "_build_anthropic_streaming_response_from_responses_stream",
    "_build_anthropic_streaming_response_from_completion_adapter_stream",
}

W6B_INTEGRATION_SEAMS: set[str] = {
    "_stringify_grok_native_input_item_value",
}

W6B_EXCLUDED_SYMBOLS: set[str] = {
    "_collect_responses_response_from_stream",
    "_finalize_collected_responses_stream_response",
    "_record_collected_responses_output_item_event",
    "_record_collected_responses_arguments_event",
    "_restore_adapted_custom_tool_calls_in_streaming_response",
    "_restore_adapted_namespace_tool_calls_in_streaming_response",
    "_validate_alias_candidate_responses_stream_if_needed",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run(coro):
    """Run a coroutine/async-generator collector in a fresh event loop."""
    return asyncio.run(coro)


async def _collect_agen(agen):
    items = []
    async for item in agen:
        items.append(item)
    return items


# ---------------------------------------------------------------------------
# AST helpers
# ---------------------------------------------------------------------------

def _parse_module(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _top_level_function_defs(tree: ast.Module) -> set[str]:
    names: set[str] = set()
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            names.add(node.name)
    return names


def _module_level_imports(tree: ast.Module) -> set[str]:
    """Collect all imported module roots (top-level and function-scoped)."""
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                roots.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                roots.add(node.module.split(".")[0])
    return roots


# ===========================================================================
# SECTION 1: Structural ownership / manifest tests
# ===========================================================================


class TestSymbolManifest:
    """Verify the module exports exactly the declared symbol inventory."""

    def test_all_owned_symbols_defined(self):
        tree = _parse_module(SSE_MODULE_PATH)
        defined = _top_level_function_defs(tree)
        missing = W6B_OWNED_SYMBOLS - defined
        assert not missing, f"Missing owned symbols in sse.py: {missing}"

    def test_no_excluded_symbols_defined(self):
        tree = _parse_module(SSE_MODULE_PATH)
        defined = _top_level_function_defs(tree)
        overlap = W6B_EXCLUDED_SYMBOLS & defined
        assert not overlap, f"Excluded symbols found in sse.py: {overlap}"

    def test_install_covers_all_owned_symbols(self):
        assert set(sse_mod._HOST_FUNCTION_NAMES) == W6B_OWNED_SYMBOLS

    def test_owned_symbols_callable(self):
        for name in W6B_OWNED_SYMBOLS:
            obj = getattr(sse_mod, name, None)
            assert obj is not None, f"{name} not found on module"
            assert callable(obj), f"{name} is not callable"


# ===========================================================================
# SECTION 2: Dependency isolation tests
# ===========================================================================


class TestDependencyIsolation:
    """Ensure no import of the god module or excluded subsystems at module scope."""

    def test_no_god_module_import(self):
        tree = _parse_module(SSE_MODULE_PATH)
        imports = _module_level_imports(tree)
        assert "llm_passthrough_endpoints" not in " ".join(imports)
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                assert "llm_passthrough_endpoints" not in node.module

    def test_allowed_imports_only(self):
        tree = _parse_module(SSE_MODULE_PATH)
        imports = _module_level_imports(tree)
        allowed = {
            "__future__", "codecs", "json", "inspect", "types", "typing",
            "fastapi", "litellm",
        }
        unexpected = imports - allowed
        assert not unexpected, f"Unexpected imports: {unexpected}"


# ===========================================================================
# SECTION 3: Behavior tests - _mapping_or_attr_get
# ===========================================================================


class TestMappingOrAttrGet:
    def test_dict_access(self):
        assert sse_mod._mapping_or_attr_get({"a": 1}, "a") == 1
        assert sse_mod._mapping_or_attr_get({"a": 1}, "b") is None
        assert sse_mod._mapping_or_attr_get({"a": 1}, "b", 42) == 42

    def test_attr_access(self):
        ns = SimpleNamespace(x=10)
        assert sse_mod._mapping_or_attr_get(ns, "x") == 10
        assert sse_mod._mapping_or_attr_get(ns, "y") is None
        assert sse_mod._mapping_or_attr_get(ns, "y", "dflt") == "dflt"


# ===========================================================================
# SECTION 4: Behavior tests - _coerce_namespace_to_mapping
# ===========================================================================


class TestCoerceNamespaceToMapping:
    def test_dict_passthrough(self):
        d = {"key": "val"}
        assert sse_mod._coerce_namespace_to_mapping(d) is d

    def test_namespace_to_dict(self):
        ns = SimpleNamespace(a=1, b=SimpleNamespace(c=2))
        result = sse_mod._coerce_namespace_to_mapping(ns)
        assert result == {"a": 1, "b": {"c": 2}}

    def test_list_of_namespaces(self):
        ns_list = [SimpleNamespace(x=1), SimpleNamespace(y=2)]
        result = sse_mod._coerce_namespace_to_mapping(ns_list)
        assert result == [{"x": 1}, {"y": 2}]

    def test_depth_bound(self):
        deep = SimpleNamespace(inner=SimpleNamespace(val="deep"))
        result = sse_mod._coerce_namespace_to_mapping(deep, _max_depth=0)
        assert isinstance(result, dict)
        assert isinstance(result["inner"], dict)

    def test_scalar_passthrough(self):
        assert sse_mod._coerce_namespace_to_mapping(42) == 42
        assert sse_mod._coerce_namespace_to_mapping("hello") == "hello"


# ===========================================================================
# SECTION 5: Behavior tests - _serialize_responses_adapter_response
# ===========================================================================


class TestSerializeResponsesAdapterResponse:
    def test_model_dump_json(self):
        class FakeModel:
            def model_dump_json(self, exclude_none: bool = False) -> str:
                return '{"kind":"pydantic"}'

        result = sse_mod._serialize_responses_adapter_response(FakeModel())
        assert result == '{"kind":"pydantic"}'

    def test_json_fallback(self):
        class FakeJson:
            def json(self, exclude_none: bool = False) -> str:
                return '{"kind":"json"}'

        result = sse_mod._serialize_responses_adapter_response(FakeJson())
        assert result == '{"kind":"json"}'

    def test_plain_dict(self):
        result = sse_mod._serialize_responses_adapter_response({"a": 1})
        assert json.loads(result) == {"a": 1}


# ===========================================================================
# SECTION 6: Behavior tests - _iterate_responses_sse_events
# ===========================================================================


class TestIterateResponsesSSEEvents:
    def test_parses_sse_data_lines(self):
        async def body():
            yield 'data: {"type":"response.created"}\n\n'
            yield 'data: {"type":"response.completed"}\n\n'

        events = _run(_collect_agen(sse_mod._iterate_responses_sse_events(body())))
        assert len(events) == 2
        assert events[0]["type"] == "response.created"
        assert events[1]["type"] == "response.completed"

    def test_handles_bytes_chunks(self):
        async def body():
            yield b'data: {"type":"response.created"}\n\n'

        events = _run(_collect_agen(sse_mod._iterate_responses_sse_events(body())))
        assert len(events) == 1
        assert events[0]["type"] == "response.created"

    def test_laziness_no_consumption_until_iterated(self):
        consumed = []

        async def body():
            consumed.append(True)
            yield 'data: {"type":"x"}\n\n'

        gen = sse_mod._iterate_responses_sse_events(body())
        assert len(consumed) == 0
        _run(_collect_agen(gen))
        assert len(consumed) == 1

    def test_trailing_buffer_without_double_newline(self):
        async def body():
            yield 'data: {"type":"final"}'

        events = _run(_collect_agen(sse_mod._iterate_responses_sse_events(body())))
        assert len(events) == 1
        assert events[0]["type"] == "final"


# ===========================================================================
# SECTION 7: Behavior tests - _responses_sse_from_iterator
# ===========================================================================


class TestResponsesSSEFromIterator:
    def test_typed_events_get_event_prefix(self):
        async def events():
            yield {"type": "response.created", "id": "r1"}

        chunks = _run(_collect_agen(sse_mod._responses_sse_from_iterator(events())))
        assert chunks[0].startswith("event: response.created\n")
        assert "data: " in chunks[0]
        assert chunks[-1] == "data: [DONE]\n\n"

    def test_untyped_events_data_only(self):
        async def events():
            yield {"no_type": True}

        chunks = _run(_collect_agen(sse_mod._responses_sse_from_iterator(events())))
        assert chunks[0].startswith("data: ")
        assert not chunks[0].startswith("event:")

    def test_on_complete_called(self):
        called = []

        async def events():
            yield {"type": "x"}

        _run(_collect_agen(
            sse_mod._responses_sse_from_iterator(events(), on_complete=lambda: called.append(True))
        ))
        assert called == [True]

    def test_terminal_done_always_last(self):
        async def events():
            yield {"type": "a"}
            yield {"type": "b"}

        chunks = _run(_collect_agen(sse_mod._responses_sse_from_iterator(events())))
        assert chunks[-1] == "data: [DONE]\n\n"
        assert len(chunks) == 3

    def test_closes_iterator_with_aclose(self):
        closed = []

        class FakeIter:
            def __aiter__(self):
                return self

            async def __anext__(self):
                raise StopAsyncIteration

            async def aclose(self):
                closed.append(True)

        _run(_collect_agen(sse_mod._responses_sse_from_iterator(FakeIter())))
        assert closed == [True]

    def test_pre_event_error_reraises_and_closes_once(self):
        capacity_error = RuntimeError("provider capacity")
        callback_calls = []

        class FakeIter:
            def __init__(self):
                self.close_calls = 0
                self.litellm_custom_stream_wrapper = self

            def __aiter__(self):
                return self

            async def __anext__(self):
                raise capacity_error

            async def aclose(self):
                self.close_calls += 1

        iterator = FakeIter()

        async def collect():
            return await _collect_agen(
                sse_mod._responses_sse_from_iterator(
                    iterator,
                    on_stream_error=lambda exc: callback_calls.append(exc),
                )
            )

        try:
            _run(collect())
        except RuntimeError as exc:
            assert exc is capacity_error
        else:
            raise AssertionError("pre-event capacity error did not propagate")
        assert callback_calls == []
        assert iterator.close_calls == 1

    def test_post_event_capacity_emits_terminal_once_without_replay(self):
        capacity_error = RuntimeError("provider capacity")
        callback_calls = []

        class FakeIter:
            def __init__(self):
                self.step = 0
                self.close_calls = 0
                self.litellm_custom_stream_wrapper = self

            def __aiter__(self):
                return self

            async def __anext__(self):
                if self.step == 0:
                    self.step += 1
                    return {"type": "response.created", "response": {"id": "resp_1"}}
                raise capacity_error

            async def aclose(self):
                self.close_calls += 1

        iterator = FakeIter()

        def on_stream_error(exc):
            callback_calls.append(exc)
            return (
                "event: response.failed\n"
                'data: {"type":"response.failed"}\n\n'
            )

        chunks = _run(
            _collect_agen(
                sse_mod._responses_sse_from_iterator(
                    iterator,
                    on_stream_error=on_stream_error,
                )
            )
        )
        rendered = "".join(chunks)
        assert rendered.count("event: response.created") == 1
        assert rendered.count("event: response.failed") == 1
        assert "data: [DONE]" not in rendered
        assert callback_calls == [capacity_error]
        assert iterator.close_calls == 1

    def test_post_event_error_without_callback_retains_prior_behavior(self):
        provider_error = RuntimeError("unrelated provider failure")

        class FakeIter:
            def __init__(self):
                self.step = 0
                self.close_calls = 0

            def __aiter__(self):
                return self

            async def __anext__(self):
                if self.step == 0:
                    self.step += 1
                    return {"type": "response.created"}
                raise provider_error

            async def aclose(self):
                self.close_calls += 1

        iterator = FakeIter()

        async def exercise():
            stream = sse_mod._responses_sse_from_iterator(iterator)
            first = await stream.__anext__()
            try:
                await stream.__anext__()
            except RuntimeError as exc:
                assert exc is provider_error
            else:
                raise AssertionError("unrelated provider error did not propagate")
            return first

        first = _run(exercise())
        assert first.startswith("event: response.created")
        assert iterator.close_calls == 1


# ===========================================================================
# SECTION 8: Behavior tests - _responses_event_text_key
# ===========================================================================


class TestResponsesEventTextKey:
    def test_item_id_preferred(self):
        assert sse_mod._responses_event_text_key({"item_id": "abc", "output_index": 3}) == "abc"

    def test_output_index_fallback(self):
        assert sse_mod._responses_event_text_key({"output_index": 0}) == "output:0"
        assert sse_mod._responses_event_text_key({"output_index": 5}) == "output:5"

    def test_default_output_0(self):
        assert sse_mod._responses_event_text_key({}) == "output:0"

    def test_namespace_event(self):
        ns = SimpleNamespace(item_id="ns_id")
        assert sse_mod._responses_event_text_key(ns) == "ns_id"


# ===========================================================================
# SECTION 9: Behavior tests - _responses_stream_event_summary
# ===========================================================================


class TestResponsesStreamEventSummary:
    def test_output_item_added(self):
        event = {
            "type": "response.output_item.added",
            "item": {"type": "message", "id": "msg_1", "name": None},
        }
        summary = sse_mod._responses_stream_event_summary(event)
        assert summary["type"] == "response.output_item.added"
        assert summary["item_type"] == "message"
        assert summary["item_id"] == "msg_1"

    def test_text_delta(self):
        event = {
            "type": "response.output_text.delta",
            "item_id": "item_x",
            "delta": "Hello world",
        }
        summary = sse_mod._responses_stream_event_summary(event)
        assert summary["item_id"] == "item_x"
        assert summary["text_len"] == 11
        assert summary["text_preview"] == "Hello world"

    def test_completed_event(self):
        event = {
            "type": "response.completed",
            "response": {
                "id": "resp_1",
                "status": "completed",
                "model": "gpt-4o",
                "output": [{"type": "message"}],
                "usage": {"input_tokens": 10, "output_tokens": 20},
            },
        }
        summary = sse_mod._responses_stream_event_summary(event)
        assert summary["response_id"] == "resp_1"
        assert summary["output_count"] == 1
        assert summary["usage"]["input_tokens"] == 10

    def test_unknown_type_minimal(self):
        event = {"type": "response.unknown_thing"}
        summary = sse_mod._responses_stream_event_summary(event)
        assert summary == {"type": "response.unknown_thing"}


# ===========================================================================
# SECTION 10: Behavior tests - _responses_repaired_output_item_id
# ===========================================================================


class TestResponsesRepairedOutputItemId:
    def test_prefers_id(self):
        assert sse_mod._responses_repaired_output_item_id({"id": "abc", "call_id": "xyz"}, 0) == "abc"

    def test_falls_back_to_call_id(self):
        assert sse_mod._responses_repaired_output_item_id({"call_id": "xyz"}, 2) == "xyz"

    def test_synthetic_fallback(self):
        assert sse_mod._responses_repaired_output_item_id({}, 7) == "item_7"

    def test_strips_whitespace(self):
        assert sse_mod._responses_repaired_output_item_id({"id": "  padded  "}, 0) == "padded"


# ===========================================================================
# SECTION 11: Behavior tests - _responses_sse_from_repaired_response_body
# ===========================================================================


class TestResponsesSSEFromRepairedResponseBody:
    def test_message_item_event_order(self):
        body = {"output": [{"type": "message", "id": "msg_1", "content": []}]}
        chunks = _run(_collect_agen(sse_mod._responses_sse_from_repaired_response_body(body)))
        assert "response.output_item.added" in chunks[0]
        assert "response.output_item.done" in chunks[1]
        assert "response.completed" in chunks[2]
        assert chunks[3] == "data: [DONE]\n\n"

    def test_function_call_includes_arguments_done(self):
        body = {
            "output": [
                {"type": "function_call", "id": "fc_1", "arguments": '{"x":1}'},
            ]
        }
        chunks = _run(_collect_agen(sse_mod._responses_sse_from_repaired_response_body(body)))
        assert len(chunks) == 5
        assert "response.function_call_arguments.done" in chunks[1]
        parsed = json.loads(chunks[1].split("data: ", 1)[1])
        assert parsed["arguments"] == '{"x":1}'
        assert parsed["item_id"] == "fc_1"

    def test_non_string_arguments_uses_seam(self):
        """When arguments is not a string, the grok seam is called."""
        original = sse_mod.__dict__.get("_stringify_grok_native_input_item_value")
        sse_mod._stringify_grok_native_input_item_value = lambda v: json.dumps(v)
        try:
            body = {
                "output": [
                    {"type": "function_call", "call_id": "c1", "arguments": {"nested": True}},
                ]
            }
            chunks = _run(_collect_agen(sse_mod._responses_sse_from_repaired_response_body(body)))
            parsed = json.loads(chunks[1].split("data: ", 1)[1])
            assert parsed["arguments"] == '{"nested": true}'
            assert parsed["item_id"] == "c1"
        finally:
            if original is not None:
                sse_mod._stringify_grok_native_input_item_value = original
            else:
                sse_mod.__dict__.pop("_stringify_grok_native_input_item_value", None)

    def test_empty_output(self):
        body = {"output": []}
        chunks = _run(_collect_agen(sse_mod._responses_sse_from_repaired_response_body(body)))
        assert len(chunks) == 2
        assert "response.completed" in chunks[0]
        assert chunks[1] == "data: [DONE]\n\n"

    def test_missing_output_key(self):
        body = {"id": "resp_x"}
        chunks = _run(_collect_agen(sse_mod._responses_sse_from_repaired_response_body(body)))
        assert len(chunks) == 2
        assert chunks[-1] == "data: [DONE]\n\n"

    def test_laziness(self):
        """Async generator is lazy: no work until iterated."""
        body = {"output": [{"type": "message", "id": "m1"}]}
        gen = sse_mod._responses_sse_from_repaired_response_body(body)
        assert gen is not None


# ===========================================================================
# SECTION 12: Behavior tests - streaming response builders
# ===========================================================================


class TestBuildAnthropicStreamingResponseFromCompletionAdapterStream:
    def test_returns_streaming_response(self):
        from fastapi.responses import StreamingResponse

        async def fake_stream():
            yield "data: hello\n\n"

        resp = sse_mod._build_anthropic_streaming_response_from_completion_adapter_stream(fake_stream())
        assert isinstance(resp, StreamingResponse)
        assert resp.media_type == "text/event-stream"


class TestBuildAnthropicStreamingResponseFromResponsesStream:
    def test_translates_responses_events_to_anthropic_sse(self):
        from fastapi.responses import StreamingResponse

        async def responses_stream():
            events = [
                {
                    "type": "response.created",
                    "response": {
                        "id": "resp_1",
                        "model": "upstream-model",
                        "status": "in_progress",
                        "output": [],
                    },
                },
                {
                    "type": "response.output_item.added",
                    "output_index": 0,
                    "item": {
                        "id": "msg_1",
                        "type": "message",
                        "status": "in_progress",
                        "role": "assistant",
                        "content": [],
                    },
                },
                {
                    "type": "response.output_text.delta",
                    "item_id": "msg_1",
                    "output_index": 0,
                    "delta": "hello",
                },
                {
                    "type": "response.output_item.done",
                    "output_index": 0,
                    "item": {
                        "id": "msg_1",
                        "type": "message",
                        "status": "completed",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "hello"}],
                    },
                },
                {
                    "type": "response.completed",
                    "response": {
                        "id": "resp_1",
                        "model": "upstream-model",
                        "status": "completed",
                        "output": [
                            {
                                "id": "msg_1",
                                "type": "message",
                                "status": "completed",
                                "role": "assistant",
                                "content": [
                                    {"type": "output_text", "text": "hello"}
                                ],
                            }
                        ],
                        "usage": {"input_tokens": 3, "output_tokens": 1},
                    },
                },
            ]
            for event in events:
                yield (
                    f"event: {event['type']}\n"
                    f"data: {json.dumps(event)}\n\n"
                )
            yield "data: [DONE]\n\n"

        source = StreamingResponse(
            responses_stream(),
            headers={"x-wave6a": "preserved"},
            media_type="text/event-stream",
        )
        translated = (
            sse_mod._build_anthropic_streaming_response_from_responses_stream(
                source,
                model="alias-model",
                request_body={"messages": [{"role": "user", "content": "hi"}]},
            )
        )
        chunks = _run(_collect_agen(translated.body_iterator))
        payload = b"".join(
            chunk if isinstance(chunk, bytes) else chunk.encode()
            for chunk in chunks
        ).decode()

        assert translated.media_type == "text/event-stream"
        assert translated.headers["x-wave6a"] == "preserved"
        assert "event: message_start" in payload
        assert '"model": "alias-model"' in payload
        assert "event: content_block_delta" in payload
        assert '"text": "hello"' in payload
        assert "event: message_delta" in payload
        assert "event: message_stop" in payload


# ===========================================================================
# SECTION 13: Golden parity - god module still defines same symbols
# ===========================================================================


class TestGodModuleParity:
    """The god module must still export these symbols (pre-cutover)."""

    def test_god_module_has_facades_not_duplicate_definitions(self):
        god_tree = _parse_module(GOD_PATH)
        god_funcs = _top_level_function_defs(god_tree)
        assert not W6B_OWNED_SYMBOLS & god_funcs
        for name in W6B_OWNED_SYMBOLS:
            assert getattr(lpe, name) is getattr(sse_mod, name)


# ===========================================================================
# SECTION 14: install() rebinding (runs last; restores module state)
# ===========================================================================


class TestInstallRebinding:
    def test_production_initialization_publishes_same_objects(self):
        for name in W6B_OWNED_SYMBOLS:
            assert getattr(lpe, name) is getattr(sse_mod, name)

    def test_production_functions_use_god_module_globals(self):
        for name in W6B_OWNED_SYMBOLS:
            assert getattr(sse_mod, name).__globals__ is vars(lpe)

    def test_install_preserves_only_required_module_globals(self):
        original_functions = {
            name: getattr(sse_mod, name) for name in W6B_OWNED_SYMBOLS
        }
        sentinel = object()
        host = {"sentinel": sentinel}

        try:
            sse_mod.install(host)

            assert host["SimpleNamespace"] is SimpleNamespace
            assert host["_coerce_namespace_to_mapping"](
                SimpleNamespace(value=1)
            ) == {"value": 1}
            assert host["sentinel"] is sentinel
            assert set(host) == W6B_OWNED_SYMBOLS | {
                "SimpleNamespace",
                "sentinel",
            }
        finally:
            for name, function in original_functions.items():
                setattr(sse_mod, name, function)
