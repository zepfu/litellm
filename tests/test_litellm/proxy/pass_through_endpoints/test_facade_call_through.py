"""Live Wave 6A facade call-through tests."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Iterator
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import Request, Response
from fastapi.responses import StreamingResponse

from litellm.llms.anthropic.experimental_pass_through.providers.google import (
    process_cache,
)
from litellm.proxy.pass_through_endpoints import llm_passthrough_endpoints as lpe
from litellm.proxy.pass_through_endpoints.aawm_adapter_runtime import (
    anthropic_adapter_calls,
    anthropic_dispatch,
    codex_candidate_calls,
    codex_dispatch,
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


# ---------------------------------------------------------------------------
# Wave 6E: request-policy facade call-through tests
# ---------------------------------------------------------------------------

from litellm.proxy.pass_through_endpoints.aawm_request_policy import (
    anthropic_body_prep,
    claude_prompt_replacement,
    codex_tool_policy,
)


def test_wave6e_codex_tool_policy_pure_facade_call_through() -> None:
    """Pure codex_tool_policy functions are same-object on lpe."""
    assert lpe._get_openai_tool_name is codex_tool_policy.get_openai_tool_name
    assert lpe._get_openai_tool_type is codex_tool_policy.get_openai_tool_type
    assert lpe._get_openai_tool_name({"name": "bash"}) == "bash"
    assert lpe._get_openai_tool_type({"type": "function"}) == "function"


def test_wave6e_codex_spawn_agent_description_facade() -> None:
    """patch_codex_spawn_agent_description_text facade is same-object."""
    assert lpe._patch_codex_spawn_agent_description_text is (
        codex_tool_policy.patch_codex_spawn_agent_description_text
    )
    result, count = lpe._patch_codex_spawn_agent_description_text("no match")
    assert result == "no match"
    assert count == 0


def test_wave6e_codex_callbacks_bound_facade_call_through() -> None:
    """Callback-bound facade delegates to codex_tool_policy with live callbacks."""
    result = lpe._get_unsupported_hosted_tool_types_for_model("gpt-4o")
    assert isinstance(result, (list, set))


def test_wave6e_codex_drop_tool_choice_facade_call_through() -> None:
    """drop_tool_choice_without_tools_from_request_body facade call-through."""
    body: dict[str, Any] = {"tool_choice": "auto", "model": "gpt-4o"}
    result = lpe._drop_tool_choice_without_tools_from_request_body(body)
    # Returns (updated_body, removed_tool_choice) tuple
    assert isinstance(result, tuple)
    assert isinstance(result[0], dict)


def test_wave6e_codex_grok_native_facade_call_through() -> None:
    """Grok-native facades are callable through lpe."""
    body: dict[str, Any] = {"model": "claude-3", "messages": []}
    result = lpe._is_anthropic_grok_native_responses_adapter_body(body)
    assert isinstance(result, bool)


def test_wave6e_claude_prompt_replacement_facade_identity() -> None:
    """All 14 claude_prompt_replacement facades are same-object."""
    for symbol in (
        "_parse_claude_code_version",
        "_resolve_claude_auto_memory_template_path",
        "_load_claude_context_replacement_template",
        "_load_claude_prompt_patch_manifest",
        "_extract_markdown_section",
        "_render_claude_auto_memory_replacement",
        "_replace_claude_auto_memory_section_in_text",
        "_replace_claude_system_prompt_override_in_value",
        "_add_claude_system_prompt_override_logging_metadata",
        "_replace_claude_system_prompt_in_anthropic_request_body",
        "_apply_claude_prompt_patches_in_text",
        "_replace_claude_prompt_patches_in_value",
        "_add_claude_prompt_patch_logging_metadata",
        "_apply_claude_prompt_patches_to_anthropic_request_body",
    ):
        assert getattr(lpe, symbol) is getattr(claude_prompt_replacement, symbol), (
            f"{symbol}: facade identity mismatch"
        )


def test_wave6e_claude_parse_version_facade_call_through() -> None:
    """_parse_claude_code_version facade call-through."""
    assert lpe._parse_claude_code_version("2.1.110") == (2, 1, 110)
    assert lpe._parse_claude_code_version(None) is None


def test_wave6e_claude_extract_markdown_section_facade() -> None:
    """_extract_markdown_section facade call-through."""
    md = "## auto memory\nsome memory content\n## Environment\nenv stuff"
    result = lpe._extract_markdown_section(md, "auto memory")
    assert "some memory content" in result


def test_wave6e_anthropic_body_prep_facade_identity() -> None:
    """All 11 anthropic_body_prep facades are same-object."""
    for symbol in (
        "_get_openai_adapter_claude_context_char_cap",
        "_detect_openai_adapter_claude_context_markers",
        "_select_openai_adapter_context_summary_lines",
        "_build_openai_adapter_compacted_claude_context_block",
        "_compact_openai_adapter_claude_context_text",
        "_compact_openai_adapter_claude_context_value",
        "_add_openai_adapter_claude_context_compaction_logging_metadata",
        "_compact_openai_adapter_claude_context_in_anthropic_request_body",
        "_validate_anthropic_tool_blocks_for_passthrough",
        "_repair_anthropic_tool_use_ids_for_passthrough",
        "_prepare_anthropic_request_body_for_passthrough",
    ):
        assert getattr(lpe, symbol) is getattr(anthropic_body_prep, symbol), (
            f"{symbol}: facade identity mismatch"
        )


def test_wave6e_body_prep_detect_markers_facade_call_through() -> None:
    """_detect_openai_adapter_claude_context_markers facade call-through."""
    markers = lpe._detect_openai_adapter_claude_context_markers("see CLAUDE.md for details")
    assert "claude-md" in markers


def test_wave6e_body_prep_validate_tool_blocks_facade_call_through() -> None:
    """_validate_anthropic_tool_blocks_for_passthrough facade call-through."""
    body: dict[str, Any] = {
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "hi"}]}
        ]
    }
    lpe._validate_anthropic_tool_blocks_for_passthrough(body)


def test_wave6e_body_prep_char_cap_facade_call_through(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """_get_openai_adapter_claude_context_char_cap facade reads env."""
    monkeypatch.delenv("AAWM_OPENAI_ADAPTER_CLAUDE_CONTEXT_CHAR_CAP", raising=False)
    assert lpe._get_openai_adapter_claude_context_char_cap() == 1200
    monkeypatch.setenv("AAWM_OPENAI_ADAPTER_CLAUDE_CONTEXT_CHAR_CAP", "500")
    assert lpe._get_openai_adapter_claude_context_char_cap() == 500


# ---------------------------------------------------------------------------
# Wave 6B XAI OAuth compatibility-binding regression
# ---------------------------------------------------------------------------


def test_wave6b_xai_oauth_god_module_patch_reaches_production_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Patching lpe.get_grok_native_oauth_access_token must intercept the
    actual runtime call after the Wave 6B extraction.

    The configured production runtime holds a late-binding lambda that looks
    up ``get_grok_native_oauth_access_token`` in the god-module globals at call
    time, so a monkeypatch of the compatibility facade reaches
    ``_prepare_grok_native_oauth_passthrough_request`` through the live
    runtime rather than the frozen module-scope import.
    """
    from litellm.proxy.pass_through_endpoints.providers.xai import (
        request_prep as xai_request_prep,
    )

    runtime = xai_request_prep._require_runtime()

    calls: list[str] = []

    async def fake_token() -> str:
        calls.append("token")
        return "patched-oauth-token"

    monkeypatch.setattr(
        lpe, "get_grok_native_oauth_access_token", fake_token
    )

    result = asyncio.run(runtime.get_grok_native_oauth_access_token())

    assert result == "patched-oauth-token"
    assert calls == ["token"]


# ---------------------------------------------------------------------------
# Wave 6F: adapter-call facades and live dispatch gates
# ---------------------------------------------------------------------------


def test_wave6f_adapter_call_facades_are_same_object() -> None:
    synthetic_candidate_host: dict[str, Any] = {
        "__builtins__": __builtins__,
    }
    synthetic_dispatch_host: dict[str, Any] = {
        "__builtins__": __builtins__,
    }
    codex_candidate_calls.install(synthetic_candidate_host)
    codex_dispatch.install(synthetic_dispatch_host)

    for symbol in anthropic_adapter_calls._EXTRACTED_FUNCTION_NAMES:
        assert getattr(lpe, symbol) is getattr(
            anthropic_adapter_calls,
            symbol,
        )
    for symbol in codex_candidate_calls._HOST_FUNCTION_NAMES:
        assert getattr(lpe, symbol) is getattr(
            codex_candidate_calls,
            symbol,
        )
    assert lpe.try_dispatch_codex_request is (
        codex_dispatch.try_dispatch_codex_request
    )
    assert lpe.try_dispatch_anthropic_adapter is not (
        anthropic_dispatch.try_dispatch_anthropic_adapter
    )


def test_wave6f_codex_dispatch_uses_live_host_handler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    expected = Response(content=b"codex-dispatched")

    async def handle_alias(**kwargs: Any) -> Response:
        calls.append(kwargs["prepared_request_body"]["model"])
        return expected

    monkeypatch.setattr(
        lpe,
        "_resolve_codex_auto_agent_alias_model",
        lambda body, *, endpoint: "aawm-codex-agent-auto",
    )
    monkeypatch.setattr(
        lpe,
        "_apply_codex_auto_agent_prevention_guidance_to_request_body",
        lambda body: (body, []),
    )
    monkeypatch.setattr(
        lpe,
        "_apply_aawm_read_agent_guidance_to_request_body",
        lambda body, *, alias_model, target_field: (body, []),
    )
    monkeypatch.setattr(
        lpe,
        "_prepare_request_body_for_passthrough_observability",
        lambda *, request, request_body: request_body,
    )
    monkeypatch.setattr(lpe, "_safe_set_request_parsed_body", lambda *args: None)
    monkeypatch.setattr(lpe, "_handle_codex_auto_agent_alias_route", handle_alias)

    body = {"model": "aawm-codex-agent-auto"}
    result = asyncio.run(
        lpe.try_dispatch_codex_request(
            endpoint="/v1/responses",
            request=Request(
                {
                    "type": "http",
                    "method": "POST",
                    "path": "/v1/responses",
                    "headers": [],
                }
            ),
            request_body=body,
            prepared_request_body=body,
            fastapi_response=Response(),
            user_api_key_dict=SimpleNamespace(),
            target_url="https://chatgpt.com/backend-api/codex/responses",
            api_key=None,
            forward_headers=False,
        )
    )

    assert result is expected
    assert calls == ["aawm-codex-agent-auto"]


def test_wave6f_anthropic_dispatch_uses_live_host_handler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    expected = Response(content=b"anthropic-adapted")

    async def handle_openai(**kwargs: Any) -> Response:
        calls.append(kwargs["adapter_model"])
        return expected

    monkeypatch.setattr(
        lpe,
        "_resolve_anthropic_xai_oauth_adapter_model",
        lambda body, *, endpoint: None,
    )
    monkeypatch.setattr(
        lpe,
        "_resolve_anthropic_grok_native_oauth_adapter_model",
        lambda body, *, endpoint: None,
    )
    monkeypatch.setattr(
        lpe,
        "_resolve_anthropic_openai_responses_adapter_model",
        lambda body, *, endpoint: "gpt-5",
    )
    monkeypatch.setattr(
        lpe,
        "_handle_anthropic_openai_responses_adapter_route",
        handle_openai,
    )

    result = asyncio.run(
        lpe.try_dispatch_anthropic_adapter(
            endpoint="/v1/messages",
            request=Request(
                {
                    "type": "http",
                    "method": "POST",
                    "path": "/anthropic/v1/messages",
                    "headers": [],
                }
            ),
            fastapi_response=Response(),
            user_api_key_dict=SimpleNamespace(),
            prepared_request_body={"model": "openai/gpt-5"},
        )
    )

    assert result is expected
    assert calls == ["gpt-5"]


def test_wave6f_codex_candidate_facade_uses_live_host_global(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class ExpectedLookup(Exception):
        pass

    def live_openrouter_key() -> str:
        raise ExpectedLookup("live OpenRouter key lookup")

    monkeypatch.setattr(lpe, "_get_openrouter_api_key", live_openrouter_key)

    with pytest.raises(ExpectedLookup, match="live OpenRouter key lookup"):
        asyncio.run(
            lpe._perform_codex_auto_agent_openrouter_completion_request(
                request=Request(
                    {
                        "type": "http",
                        "method": "POST",
                        "path": "/v1/responses",
                        "headers": [],
                    }
                ),
                adapter_model="openai/gpt-5",
                request_body={"model": "openai/gpt-5"},
            )
        )


def test_wave6f_anthropic_execution_facades_use_live_host_namespace() -> None:
    for symbol in (
        "_perform_anthropic_responses_adapter_pass_through",
        "_perform_normalized_anthropic_completion_adapter_stream",
        "_perform_anthropic_completion_adapter_messages_call",
        "_finalize_anthropic_responses_adapter_upstream_response",
        "_finalize_anthropic_responses_adapter_from_config",
        "_finalize_anthropic_completion_adapter_response",
    ):
        facade = getattr(lpe, symbol)
        assert facade is getattr(anthropic_adapter_calls, symbol)
        assert facade.__globals__ is lpe.__dict__
