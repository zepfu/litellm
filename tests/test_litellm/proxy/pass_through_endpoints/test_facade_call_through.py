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


# ---------------------------------------------------------------------------
# Wave 6D: request-policy facade call-through tests
# ---------------------------------------------------------------------------

from litellm.proxy.pass_through_endpoints.aawm_request_policy import (
    alias_guidance,
    observability_metadata,
    persisted_output,
)


def test_wave6d_persisted_output_estimate_facade_call_through() -> None:
    """_estimate_google_content_text_chars facade resolves through host globals."""
    assert lpe._estimate_google_content_text_chars is (
        persisted_output._estimate_google_content_text_chars
    )
    block = {"parts": [{"text": "abc"}, {"text": "de"}]}
    assert lpe._estimate_google_content_text_chars(block) == 5
    assert persisted_output._estimate_google_content_text_chars(block) == 5
    assert lpe._estimate_google_content_text_chars("not a dict") == 0


def test_wave6d_persisted_output_expansion_enabled_facade(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Expansion-enabled facade reads env through host globals at call time."""
    assert lpe._is_claude_persisted_output_expansion_enabled is (
        persisted_output._is_claude_persisted_output_expansion_enabled
    )
    monkeypatch.setenv("LITELLM_EXPAND_CLAUDE_PERSISTED_OUTPUT", "1")
    assert lpe._is_claude_persisted_output_expansion_enabled() is True
    monkeypatch.setenv("LITELLM_EXPAND_CLAUDE_PERSISTED_OUTPUT", "0")
    assert lpe._is_claude_persisted_output_expansion_enabled() is False


def test_wave6d_observability_merge_metadata_facade_call_through() -> None:
    """_merge_litellm_metadata facade is same-object and behavior-compatible."""
    assert lpe._merge_litellm_metadata is (
        observability_metadata._merge_litellm_metadata
    )
    body: dict[str, Any] = {"model": "test"}
    result = lpe._merge_litellm_metadata(
        body,
        tags_to_add=["wave6d"],
        extra_fields={"trace_id": "t1"},
    )
    assert result["litellm_metadata"]["tags"] == ["wave6d"]
    assert result["litellm_metadata"]["trace_id"] == "t1"
    assert "litellm_metadata" not in body  # original unchanged


def test_wave6d_observability_iter_anthropic_text_fragments_facade() -> None:
    """_iter_anthropic_text_fragments facade yields text from Anthropic shapes."""
    assert lpe._iter_anthropic_text_fragments is (
        observability_metadata._iter_anthropic_text_fragments
    )
    fragments = list(lpe._iter_anthropic_text_fragments("plain string"))
    assert fragments == ["plain string"]
    # Non-text dicts are walked recursively; only type=="text" dicts yield text
    fragments = list(
        lpe._iter_anthropic_text_fragments(
            [{"type": "text", "text": "hello"}, "tail"]
        )
    )
    assert fragments == ["hello", "tail"]


def test_wave6d_observability_extract_agent_tenant_facade() -> None:
    """_extract_claude_agent_and_tenant_from_request_body facade call-through."""
    assert lpe._extract_claude_agent_and_tenant_from_request_body is (
        observability_metadata._extract_claude_agent_and_tenant_from_request_body
    )
    body = {
        "system": "You are 'alibaba' and you are working on the 'litellm' project."
    }
    agent, tenant = lpe._extract_claude_agent_and_tenant_from_request_body(body)
    assert agent == "alibaba"
    assert tenant == "litellm"


def test_wave6d_observability_detect_post_rewrite_context_files_facade() -> None:
    """_detect_claude_post_rewrite_context_files facade call-through."""
    assert lpe._detect_claude_post_rewrite_context_files is (
        observability_metadata._detect_claude_post_rewrite_context_files
    )
    body: dict[str, Any] = {"system": "see MEMORY.md for details"}
    result = lpe._detect_claude_post_rewrite_context_files(body)
    assert "MEMORY.md" in result


def test_wave6d_alias_guidance_prevention_facade_call_through() -> None:
    """Prevention guidance facade is same-object and appends to instructions."""
    assert lpe._append_codex_auto_agent_prevention_guidance_to_instructions is (
        alias_guidance._append_codex_auto_agent_prevention_guidance_to_instructions
    )
    result = lpe._append_codex_auto_agent_prevention_guidance_to_instructions(
        "base instructions"
    )
    assert result.startswith("base instructions")
    assert len(result) > len("base instructions")


def test_wave6d_alias_guidance_read_agent_model_facade() -> None:
    """_is_aawm_read_agent_alias_model facade call-through."""
    assert lpe._is_aawm_read_agent_alias_model is (
        alias_guidance._is_aawm_read_agent_alias_model
    )
    from litellm.proxy.pass_through_endpoints.aawm_alias_routing.policy import (
        CODEX_AAWM_READ_ALIAS,
    )

    assert lpe._is_aawm_read_agent_alias_model(CODEX_AAWM_READ_ALIAS) is True
    assert lpe._is_aawm_read_agent_alias_model("gpt-4o") is False


def test_wave6d_observability_get_nested_str_value_facade() -> None:
    """_get_nested_str_value facade is observability-owned, not control-plane."""
    from litellm.proxy.pass_through_endpoints import (
        aawm_claude_control_plane as cp,
    )

    assert lpe._get_nested_str_value is (
        observability_metadata._get_nested_str_value
    )
    # Control plane has its own distinct copy
    assert cp._get_nested_str_value is not lpe._get_nested_str_value
    # Both work
    nested = {"x": {"y": {"z": "deep"}}}
    assert lpe._get_nested_str_value(nested, ("x", "y", "z")) == "deep"
    assert cp._get_nested_str_value(nested, ("x", "y", "z")) == "deep"
    assert lpe._get_nested_str_value(nested, ("x", "missing")) is None
