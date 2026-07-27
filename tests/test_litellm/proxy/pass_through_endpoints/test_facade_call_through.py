"""Live Wave 6A facade call-through tests."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Iterator
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import Request
from fastapi.responses import StreamingResponse

from litellm.llms.anthropic.experimental_pass_through.providers.google import (
    process_cache,
)
from litellm.proxy.pass_through_endpoints import llm_passthrough_endpoints as lpe
from litellm.proxy.pass_through_endpoints.aawm_adapter_runtime import (
    payload_validation,
    request_build,
    sse,
    stream_collect,
    tool_call_restore,
)
from litellm.proxy.pass_through_endpoints.providers.google import (
    codex_code_assist as gca,
)


def test_request_build_facade_uses_live_metadata_helper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, ...]] = []

    def metadata_value(body: dict[str, Any], *keys: str) -> str | None:
        calls.append(keys)
        return "repository-live" if "repository" in keys else None

    monkeypatch.setattr(lpe, "_extract_auto_agent_alias_metadata_value", metadata_value)
    context = request_build._build_malformed_tool_call_intake_context(
        None,
        {},
        adapter="wave6a",
    )

    assert request_build._build_malformed_tool_call_intake_context is (
        lpe._build_malformed_tool_call_intake_context
    )
    assert context["repository"] == "repository-live"
    assert calls


def test_sse_facade_uses_live_event_type_lookup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def events() -> AsyncIterator[dict[str, int]]:
        yield {"value": 1}

    def event_type(obj: Any, key: str, default: Any = None) -> Any:
        if key == "type":
            return "response.wave6a"
        return default

    monkeypatch.setattr(lpe, "_mapping_or_attr_get", event_type)

    async def collect() -> list[str]:
        return [
            chunk
            async for chunk in sse._responses_sse_from_iterator(events())
        ]

    chunks = asyncio.run(collect())
    assert sse._responses_sse_from_iterator is lpe._responses_sse_from_iterator
    assert chunks[0].startswith("event: response.wave6a\n")


def test_tool_restore_facade_uses_live_advertised_tool_lookup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def advertised(
        request_body: dict[str, Any] | None,
        *,
        adapter_model: str,
    ) -> set[str]:
        calls.append(adapter_model)
        return set()

    monkeypatch.setattr(
        lpe,
        "_advertised_custom_tool_function_adapter_names",
        advertised,
    )
    body = {"output": [{"type": "function_call", "name": "unchanged"}]}
    restored, count, error = (
        tool_call_restore._restore_adapted_custom_tool_calls_in_response_body(
            body,
            request_body={"tools": []},
            adapter_model="live-model",
        )
    )

    assert restored is body
    assert count == 0
    assert error is None
    assert calls == ["live-model"]


def test_stream_collect_facade_uses_canonical_sse_lookup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = lpe._mapping_or_attr_get
    calls: list[str] = []

    def recording_lookup(obj: Any, key: str, default: Any = None) -> Any:
        calls.append(key)
        return original(obj, key, default)

    monkeypatch.setattr(lpe, "_mapping_or_attr_get", recording_lookup)
    output_items: dict[str, dict[str, Any]] = {}
    ordered_keys: list[str] = []
    key_aliases: dict[str, str] = {}
    key_by_output_index: dict[int, str] = {}
    stream_collect._record_collected_responses_output_item_event(
        event={
            "item": {"type": "message", "id": "message-live"},
            "output_index": 0,
        },
        output_items=output_items,
        ordered_keys=ordered_keys,
        key_aliases=key_aliases,
        key_by_output_index=key_by_output_index,
    )

    assert stream_collect._record_collected_responses_output_item_event is (
        lpe._record_collected_responses_output_item_event
    )
    assert output_items["message-live"]["type"] == "message"
    assert calls


def test_payload_validation_facade_uses_live_malformed_text_detector(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def malformed(text: str) -> bool:
        calls.append(text)
        return True

    monkeypatch.setattr(lpe, "is_malformed_composer_call_literal_text", malformed)
    body = {
        "output": [
            {
                "type": "message",
                "content": [{"type": "output_text", "text": "live marker"}],
            }
        ]
    }

    assert (
        payload_validation._is_codex_auto_agent_malformed_tool_call_text_output(
            body
        )
        is True
    )
    assert payload_validation._is_codex_auto_agent_malformed_tool_call_text_output is (
        lpe._is_codex_auto_agent_malformed_tool_call_text_output
    )
    assert calls == ["live marker"]


# ---------------------------------------------------------------------------
# Wave 6C: same-object facade identity + late host-global dependency resolution
# ---------------------------------------------------------------------------


@pytest.fixture()
def wave6c_lpe_runtime() -> Iterator[None]:
    """Bind moved Google Code Assist callers to the live facade namespace."""
    previous_runtime = gca._RUNTIME
    process_cache._codex_google_code_assist_tool_call_name_cache.clear()
    process_cache._codex_google_code_assist_tool_call_arguments_cache.clear()
    gca.configure(host_globals=lpe.__dict__)
    yield
    process_cache._codex_google_code_assist_tool_call_name_cache.clear()
    process_cache._codex_google_code_assist_tool_call_arguments_cache.clear()
    gca._RUNTIME = previous_runtime


def test_wave6c_cache_remember_lookup_late_host_max_size(
    wave6c_lpe_runtime: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert lpe._remember_codex_google_code_assist_tool_call_name is (
        gca._remember_codex_google_code_assist_tool_call_name
    )
    assert lpe._lookup_codex_google_code_assist_tool_call_name is (
        gca._lookup_codex_google_code_assist_tool_call_name
    )

    monkeypatch.setattr(
        lpe,
        "_CODEX_GOOGLE_CODE_ASSIST_TOOL_CALL_NAME_CACHE_MAX_SIZE",
        2,
    )
    gca._remember_codex_google_code_assist_tool_call_name("k1", "Alpha")
    gca._remember_codex_google_code_assist_tool_call_name("k2", "Beta")
    gca._remember_codex_google_code_assist_tool_call_name("k3", "Gamma")

    assert gca._lookup_codex_google_code_assist_tool_call_name("k1") is None
    assert gca._lookup_codex_google_code_assist_tool_call_name("k2") == "Beta"
    assert gca._lookup_codex_google_code_assist_tool_call_name("k3") == "Gamma"


def test_wave6c_request_preparation_uses_late_host_token_loader(
    wave6c_lpe_runtime: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert lpe._prepare_codex_google_code_assist_adapter_request is (
        gca._prepare_codex_google_code_assist_adapter_request
    )

    class ExpectedTokenLoad(Exception):
        pass

    calls: list[str] = []

    async def late_token_loader() -> str:
        calls.append("loader")
        raise ExpectedTokenLoad("late lpe token loader")

    monkeypatch.setattr(
        lpe,
        "_load_valid_local_google_oauth_access_token",
        late_token_loader,
    )

    with pytest.raises(ExpectedTokenLoad, match="late lpe token loader"):
        asyncio.run(
            gca._prepare_codex_google_code_assist_adapter_request(
                request=Request({"type": "http", "headers": []}),
                prepared_request_body={},
                adapter_model="gemini-2.5-pro",
            )
        )

    assert calls == ["loader"]


def test_wave6c_stream_tool_restore_late_host_remember(
    wave6c_lpe_runtime: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert lpe._restore_google_adapter_tool_call_names is (
        gca._restore_google_adapter_tool_call_names
    )

    remembered: list[tuple[str, str, str | None]] = []

    def late_remember(
        tool_call_id: str,
        name: str,
        arguments: str | None,
        *,
        scope_key: str | None = None,
    ) -> None:
        remembered.append((tool_call_id, name, scope_key))

    monkeypatch.setattr(
        lpe,
        "_remember_codex_google_code_assist_tool_call_name",
        late_remember,
    )

    fn = SimpleNamespace(name="adapted_1", arguments='{"p":1}')
    tool_call = SimpleNamespace(id="call_77", function=fn)
    message = SimpleNamespace(tool_calls=[tool_call])
    choice = SimpleNamespace(message=message)
    response_obj = SimpleNamespace(choices=[choice])

    result = gca._restore_google_adapter_tool_call_names(
        response_obj,
        {"adapted_1": "OriginalTool"},
        scope_key="scope-6c",
    )

    assert result is response_obj
    assert fn.name == "OriginalTool"
    assert remembered == [("call_77", "OriginalTool", "scope-6c")]


def test_wave6c_response_collection_uses_late_host_model_normalizer(
    wave6c_lpe_runtime: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert lpe._collect_google_code_assist_response_from_stream is (
        gca._collect_google_code_assist_response_from_stream
    )

    class ExpectedNormalize(Exception):
        pass

    calls: list[str] = []

    def late_normalize(model: str) -> str:
        calls.append(model)
        raise ExpectedNormalize("late lpe model normalizer")

    async def body_iter() -> AsyncIterator[bytes]:
        yield b'{"candidates":[]}'

    monkeypatch.setattr(
        lpe,
        "_normalize_google_completion_adapter_model_name",
        late_normalize,
    )

    with pytest.raises(ExpectedNormalize, match="late lpe model normalizer"):
        asyncio.run(
            gca._collect_google_code_assist_response_from_stream(
                response=StreamingResponse(body_iter()),
                adapter_model="gemini-raw-6c",
                tool_name_mapping={},
                logging_obj=None,
            )
        )

    assert calls == ["gemini-raw-6c"]
