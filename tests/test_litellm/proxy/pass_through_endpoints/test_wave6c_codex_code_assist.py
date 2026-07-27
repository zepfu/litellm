"""Module-local coverage for the Wave 6C Google Code Assist extraction."""

from __future__ import annotations

import ast
import asyncio
import inspect
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi import Request

import litellm
from litellm.llms.anthropic.experimental_pass_through.providers.google import (
    process_cache,
)
from litellm.proxy.pass_through_endpoints.providers.google import (
    codex_code_assist as mod,
)

MODULE_PATH = Path(mod.__file__).resolve()


@pytest.fixture(autouse=True)
def reset_runtime() -> Iterator[None]:
    previous_runtime = mod._RUNTIME
    process_cache._codex_google_code_assist_tool_call_name_cache.clear()
    process_cache._codex_google_code_assist_tool_call_arguments_cache.clear()
    mod.configure(host_globals={})
    yield
    process_cache._codex_google_code_assist_tool_call_name_cache.clear()
    process_cache._codex_google_code_assist_tool_call_arguments_cache.clear()
    mod._RUNTIME = previous_runtime


def _completion_message_tool_call_ids(message: Any) -> set[str]:
    if not isinstance(message, dict):
        return set()
    calls = message.get("tool_calls")
    if not isinstance(calls, list):
        return set()
    return {
        call["id"]
        for call in calls
        if isinstance(call, dict) and isinstance(call.get("id"), str)
    }


def _completion_message_has_tool_result(message: Any) -> bool:
    return isinstance(message, dict) and message.get("role") == "tool"


def test_module_has_no_god_module_import() -> None:
    tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            assert node.module is None or "llm_passthrough_endpoints" not in node.module
        elif isinstance(node, ast.Import):
            assert all("llm_passthrough_endpoints" not in alias.name for alias in node.names)


def test_schema_union_sanitization_preserves_annotations_and_nullable() -> None:
    mod.configure(
        host_globals={
            "_sanitize_openai_object_schema_properties": lambda schema: 0,
        }
    )
    schema = {
        "description": "target",
        "anyOf": [{"type": "string", "enum": ["a", "b"]}, {"type": "null"}],
    }

    fixes = mod._sanitize_google_code_assist_tool_schema(schema)

    assert fixes == 1
    assert schema == {
        "type": "string",
        "enum": ["a", "b"],
        "description": "target",
        "nullable": True,
    }


def test_anthropic_tool_replay_converts_use_and_result_blocks() -> None:
    body = {
        "messages": [
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "checking"},
                    {
                        "type": "tool_use",
                        "id": "call_1",
                        "name": "Read",
                        "input": {"path": "a.py"},
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "call_1",
                        "content": [{"type": "text", "text": "contents"}],
                    }
                ],
            },
        ]
    }

    updated, changes = mod._normalize_codex_google_code_assist_anthropic_tool_replay(
        body
    )

    assert updated["messages"][0]["tool_calls"] == [
        {
            "id": "call_1",
            "type": "function",
            "function": {"name": "Read", "arguments": '{"path": "a.py"}'},
        }
    ]
    assert updated["messages"][1] == {
        "role": "tool",
        "tool_call_id": "call_1",
        "content": "contents",
    }
    assert changes["google_adapter_codex_converted_anthropic_tool_use_count"] == 1
    assert changes["google_adapter_codex_converted_anthropic_tool_result_count"] == 1


def test_cache_callbacks_use_canonical_process_cache_and_repair_tool_pair() -> None:
    now = 100.0
    mod.configure(
        host_globals={
            "_completion_message_has_tool_result": _completion_message_has_tool_result,
            "_completion_message_tool_call_ids": _completion_message_tool_call_ids,
        },
        monotonic=lambda: now,
    )
    mod._remember_codex_google_code_assist_tool_call_name(
        "call_1",
        "Read",
        '{"path":"a.py"}',
        scope_key="session-a",
    )

    assert (
        mod._RUNTIME.tool_call_name_cache
        is process_cache._codex_google_code_assist_tool_call_name_cache
    )
    assert (
        mod._RUNTIME.tool_call_arguments_cache
        is process_cache._codex_google_code_assist_tool_call_arguments_cache
    )

    repaired, changes = mod._ensure_codex_google_code_assist_tool_results_have_calls(
        {
            "messages": [
                {"role": "assistant", "content": ""},
                {"role": "tool", "tool_call_id": "call_1", "content": "ok"},
            ]
        },
        scope_key="session-a",
    )

    assert repaired["messages"][0]["tool_calls"][0]["function"] == {
        "name": "Read",
        "arguments": '{"path":"a.py"}',
    }
    assert changes["google_adapter_codex_repaired_missing_tool_call_count"] == 1


def test_request_construction_builds_chat_kwargs_and_drops_non_function_tools() -> None:
    mod.configure(
        host_globals={
            "_CODEX_GOOGLE_CODE_ASSIST_DEFAULT_MAX_TOKENS": 32768,
            "_dedupe_sorted_str_list": lambda values: sorted(set(values)),
            "_merge_litellm_metadata": lambda body, **_: body,
        }
    )
    prepared, dropped = mod._drop_codex_google_code_assist_non_function_tools(
        {
            "model": "read",
            "input": "hello",
            "tools": [
                {"type": "web_search_preview"},
                {
                    "type": "function",
                    "name": "Read",
                    "parameters": {"type": "object"},
                },
            ],
        }
    )
    completion_kwargs, request_input, responses_request = (
        mod._build_codex_google_code_assist_completion_kwargs(
            prepared,
            adapter_model="gemini-2.5-pro",
        )
    )

    assert dropped == ["web_search_preview"]
    assert prepared["tools"] == [
        {
            "type": "function",
            "name": "Read",
            "parameters": {"type": "object"},
        }
    ]
    assert request_input == "hello"
    assert responses_request["tools"] == prepared["tools"]
    assert completion_kwargs["max_tokens"] == 32768


def test_duplicate_tool_result_annotation_preserves_parallel_ids() -> None:
    contents: list[dict[str, Any]] = [
        {
            "role": "user",
            "parts": [
                {"functionResponse": {"name": "Read", "response": {"content": "a"}}},
                {"functionResponse": {"name": "Read", "response": {"content": "b"}}},
            ],
        }
    ]

    annotated = mod._annotate_google_code_assist_duplicate_tool_response_parts(
        contents,
        [("Read", "call_1"), ("Read", "call_2")],
        annotate_function_response_id=True,
    )

    assert annotated == 2
    first, second = contents[0]["parts"]
    assert first["functionResponse"]["id"] == "call_1"
    assert first["functionResponse"]["response"]["tool_use_id"] == "call_1"
    assert second["functionResponse"]["id"] == "call_2"
    assert second["functionResponse"]["response"]["tool_use_id"] == "call_2"


def test_sse_translation_unwraps_each_google_response_event() -> None:
    class Logger:
        def info(self, *args: Any, **kwargs: Any) -> None:
            pass

        def debug(self, *args: Any, **kwargs: Any) -> None:
            pass

        def exception(self, *args: Any, **kwargs: Any) -> None:
            pass

    mod.configure(
        host_globals={
            "_clean_codex_auth_value": lambda value: value,
            "_get_google_adapter_post_tool_cooldown_seconds": lambda: 0.0,
            "_get_google_adapter_rate_limit_key": lambda model: str(model),
            "_google_code_assist_unwrapped_chunk_contains_tool_call": lambda chunk: False,
            "_set_google_adapter_cooldown": lambda *args, **kwargs: None,
            "verbose_proxy_logger": Logger(),
        }
    )

    async def body_iterator() -> Any:
        yield (
            b'data: {"response":{"candidates":[{"content":{"parts":[{"text":"hi"}]}}]},'
            b'"traceId":"trace-1"}\n\n'
        )

    async def collect() -> list[str]:
        return [
            chunk
            async for chunk in mod._iterate_google_code_assist_unwrapped_stream(
                body_iterator()
            )
        ]

    chunks = asyncio.run(collect())
    assert len(chunks) == 1
    payload = json.loads(chunks[0].removeprefix("data: ").strip())
    assert payload["responseId"] == "trace-1"
    assert payload["candidates"][0]["content"]["parts"][0]["text"] == "hi"


def test_request_builder_delegation_preserves_explicit_scope_callback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, Any] = {}

    async def fake_builder(**kwargs: Any) -> Any:
        observed.update(kwargs)
        return ({}, {}, [], {}, {}, {})

    monkeypatch.setattr(
        mod._request_assembly,
        "_build_google_code_assist_request_from_completion_kwargs",
        fake_builder,
    )
    request = Request({"type": "http", "headers": []})

    result = asyncio.run(
        mod._build_google_code_assist_request_from_completion_kwargs(
            completion_kwargs={"messages": []},
            adapter_model="gemini-2.5-pro",
            project="project-a",
            request=request,
            completion_kwargs_are_openai_chat=True,
            scope_key="session-a",
        )
    )

    assert result == ({}, {}, [], {}, {}, {})
    assert observed["scope_key"] == "session-a"
    assert observed["completion_kwargs_are_openai_chat"] is True


def test_install_publishes_owned_symbols_without_copying_cache_state() -> None:
    host: dict[str, object] = {}
    mod.install(host)

    assert host["_remember_codex_google_code_assist_tool_call_name"] is (
        mod._remember_codex_google_code_assist_tool_call_name
    )
    assert (
        mod._RUNTIME.tool_call_name_cache
        is process_cache._codex_google_code_assist_tool_call_name_cache
    )


# ---------------------------------------------------------------------------
# Parity pins: sync/asyncgen/signature alignment with the god module
# ---------------------------------------------------------------------------


def test_iterate_unwrapped_stream_is_plain_def_not_asyncgen() -> None:
    """The facade must be a plain def returning the provider async generator,
    not an async-generator function, to match the god module call-site."""
    fn = mod._iterate_google_code_assist_unwrapped_stream
    assert not inspect.isasyncgenfunction(fn), (
        "_iterate_google_code_assist_unwrapped_stream must be a plain def"
    )
    assert callable(fn)


def test_iterate_unwrapped_stream_returns_async_generator() -> None:
    """Calling the plain def must return an async generator object."""
    mod.configure(
        host_globals={
            "_clean_codex_auth_value": lambda v: v,
            "_get_google_adapter_post_tool_cooldown_seconds": lambda: 0.0,
            "_get_google_adapter_rate_limit_key": lambda m: str(m),
            "_google_code_assist_unwrapped_chunk_contains_tool_call": lambda c: False,
            "_set_google_adapter_cooldown": lambda *a, **kw: None,
            "verbose_proxy_logger": type("L", (), {"info": lambda s, *a, **kw: None, "debug": lambda s, *a, **kw: None, "exception": lambda s, *a, **kw: None})(),
        }
    )

    async def body_iter() -> Any:
        yield b'data: {"response":{"candidates":[]},"traceId":"t"}\n\n'

    result = mod._iterate_google_code_assist_unwrapped_stream(body_iter())
    assert inspect.isasyncgen(result) or hasattr(result, "__anext__")


def test_prepare_adapter_request_signature_parity() -> None:
    sig = inspect.signature(mod._prepare_codex_google_code_assist_adapter_request)
    params = sig.parameters
    assert params["adapter_provider"].default == litellm.LlmProviders.GEMINI.value
    assert inspect.iscoroutinefunction(mod._prepare_codex_google_code_assist_adapter_request)


def test_build_completion_kwargs_return_annotation_parity() -> None:
    hints = mod._build_codex_google_code_assist_completion_kwargs.__annotations__
    assert "ResponsesAPIOptionalRequestParams" in hints.get("return", "")


def test_build_codex_streaming_response_adapter_request_annotation() -> None:
    hints = mod._build_codex_streaming_response_from_google_code_assist_stream.__annotations__
    assert hints.get("adapter_request") is not None
    # With from __future__ import annotations, annotations are strings
    assert "SimpleNamespace" in str(hints["adapter_request"])


def test_resolve_scope_key_request_body_annotation() -> None:
    hints = mod._resolve_codex_google_code_assist_tool_call_scope_key.__annotations__
    assert "Payload" in str(hints.get("request_body", ""))


def test_install_publishes_all_owned_names_to_host_globals() -> None:
    host: dict[str, object] = {}
    mod.install(host)
    for name in mod._OWNED_FUNCTION_NAMES:
        assert name in host, f"install() must publish {name}"
        assert host[name] is getattr(mod, name)
    # configure must have been called with the same host dict
    assert mod._RUNTIME.host_globals is host


def test_install_honors_late_host_max_size_monkeypatch_for_fifo() -> None:
    """After install(), a late change to the host module's
    ``_CODEX_GOOGLE_CODE_ASSIST_TOOL_CALL_NAME_CACHE_MAX_SIZE`` must drive FIFO
    eviction without reconfiguration (production compatibility obligation)."""
    host: dict[str, object] = {}
    mod.install(host)

    # Late host-global monkeypatch, mimicking a test patching lpe.<CONST>.
    host["_CODEX_GOOGLE_CODE_ASSIST_TOOL_CALL_NAME_CACHE_MAX_SIZE"] = 2

    mod._remember_codex_google_code_assist_tool_call_name("a", "A")
    mod._remember_codex_google_code_assist_tool_call_name("b", "B")
    mod._remember_codex_google_code_assist_tool_call_name("c", "C")

    assert mod._lookup_codex_google_code_assist_tool_call_name("a") is None
    assert mod._lookup_codex_google_code_assist_tool_call_name("b") == "B"
    assert mod._lookup_codex_google_code_assist_tool_call_name("c") == "C"

    # Default max size is restored when the host constant is removed.
    del host["_CODEX_GOOGLE_CODE_ASSIST_TOOL_CALL_NAME_CACHE_MAX_SIZE"]
    assert mod._resolve_cache_max_size() == mod._RUNTIME.cache_max_size
