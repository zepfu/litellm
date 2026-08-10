"""Wave 6F module-local tests for anthropic_adapter_calls.py.

Pins signatures, async parity, representative success/failure/stream
behavior, and verifies no god-module import at module scope.
"""

from __future__ import annotations

import inspect
from types import SimpleNamespace
from typing import Any, Optional
from unittest.mock import MagicMock, patch

import pytest
from fastapi import Response
from fastapi.responses import StreamingResponse

from litellm.proxy.pass_through_endpoints.aawm_adapter_runtime import (
    anthropic_adapter_calls as mod,
)


# ---------------------------------------------------------------------------
# No god-module import at module scope
# ---------------------------------------------------------------------------


class TestNoGodImport:
    def test_no_god_module_import_at_module_scope(self):
        """The extracted module must not import llm_passthrough_endpoints."""
        source = inspect.getsource(mod)
        assert "from litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints" not in source
        assert "import litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints" not in source

    def test_god_module_not_in_module_globals(self):
        """No reference to the god module object in module namespace."""
        for name, obj in vars(mod).items():
            if hasattr(obj, "__module__") and isinstance(obj.__module__, str):
                assert "llm_passthrough_endpoints" not in obj.__module__, (
                    f"{name} references god module"
                )


# ---------------------------------------------------------------------------
# Signature pinning
# ---------------------------------------------------------------------------


class TestSignaturePinning:
    """Pin public signatures so integrator reconciliation catches drift."""

    def test_decode_http_response_body(self):
        sig = inspect.signature(mod._decode_http_response_body)
        assert list(sig.parameters) == ["body"]

    def test_build_adapted_route_rollup_kwargs(self):
        sig = inspect.signature(mod._build_adapted_route_rollup_kwargs)
        assert list(sig.parameters) == ["litellm_metadata"]

    def test_emit_adapted_route_access_log(self):
        sig = inspect.signature(mod._emit_adapted_route_access_log)
        params = list(sig.parameters)
        assert params == ["request", "target_url", "request_body", "rollup_kwargs", "adapter_label", "provider_bound_body"]

    def test_record_adapted_completed_route_rollup_turn(self):
        sig = inspect.signature(mod._record_adapted_completed_route_rollup_turn)
        params = list(sig.parameters)
        assert params == ["rollup_kwargs", "adapter_label"]

    def test_record_adapted_completed_route_rollup_after_stream(self):
        sig = inspect.signature(mod._record_adapted_completed_route_rollup_after_stream)
        params = list(sig.parameters)
        assert params == ["response", "rollup_kwargs", "adapter_label"]

    def test_normalize_openai_function_tool_parameters(self):
        sig = inspect.signature(mod._normalize_openai_function_tool_parameters)
        assert list(sig.parameters) == ["parameters"]

    def test_sanitize_openai_object_schema_properties(self):
        sig = inspect.signature(mod._sanitize_openai_object_schema_properties)
        assert list(sig.parameters) == ["schema_node"]

    def test_normalize_openai_function_tool_schemas(self):
        sig = inspect.signature(mod._normalize_openai_function_tool_schemas)
        assert list(sig.parameters) == ["translated_body"]

    def test_get_openai_adapter_function_tool_names(self):
        sig = inspect.signature(mod._get_openai_adapter_function_tool_names)
        assert list(sig.parameters) == ["request_body"]

    def test_apply_responses_adapter_parallel_instruction_policy(self):
        sig = inspect.signature(mod._apply_responses_adapter_parallel_instruction_policy)
        params = list(sig.parameters)
        assert params == ["request_body", "tag_prefix", "metadata_prefix", "span_name"]

    def test_apply_openai_adapter_parallel_instruction_policy(self):
        sig = inspect.signature(mod._apply_openai_adapter_parallel_instruction_policy)
        assert list(sig.parameters) == ["request_body"]

    def test_apply_openrouter_adapter_parallel_instruction_policy(self):
        sig = inspect.signature(mod._apply_openrouter_adapter_parallel_instruction_policy)
        assert list(sig.parameters) == ["request_body"]

    def test_get_latest_adapter_user_prompt_text(self):
        sig = inspect.signature(mod._get_latest_adapter_user_prompt_text)
        assert list(sig.parameters) == ["request_body"]

    def test_prompt_explicitly_requests_bash_tool(self):
        sig = inspect.signature(mod._prompt_explicitly_requests_bash_tool)
        assert list(sig.parameters) == ["prompt_text"]

    def test_maybe_force_explicit_bash_tool_choice_for_responses_adapter(self):
        sig = inspect.signature(mod._maybe_force_explicit_bash_tool_choice_for_responses_adapter)
        assert list(sig.parameters) == ["request_body", "translated_body"]

    def test_apply_forced_bash_tool_choice_for_responses_adapter(self):
        sig = inspect.signature(mod._apply_forced_bash_tool_choice_for_responses_adapter)
        assert list(sig.parameters) == ["request_body", "translated_body"]

    def test_maybe_force_explicit_bash_tool_choice_for_completion_adapter(self):
        sig = inspect.signature(mod._maybe_force_explicit_bash_tool_choice_for_completion_adapter)
        assert list(sig.parameters) == ["request_body"]

    def test_responses_request_contains_mcp_tools(self):
        sig = inspect.signature(mod._responses_request_contains_mcp_tools)
        assert list(sig.parameters) == ["request_body"]

    def test_coerce_mapping_to_namespace(self):
        sig = inspect.signature(mod._coerce_mapping_to_namespace)
        params = list(sig.parameters)
        assert params == ["value", "_depth", "_max_depth"]

    def test_drop_anthropic_grok_native_prior_function_call_replay(self):
        sig = inspect.signature(mod._drop_anthropic_grok_native_prior_function_call_replay)
        assert list(sig.parameters) == ["request_body"]

    def test_build_anthropic_response_from_responses_response(self):
        sig = inspect.signature(mod._build_anthropic_response_from_responses_response)
        params = list(sig.parameters)
        assert "response_body" in params
        assert "reject_empty_success" in params
        assert "use_codex_native_tools" in params

    def test_build_completion_adapter_metadata(self):
        sig = inspect.signature(mod._build_completion_adapter_metadata)
        assert list(sig.parameters) == ["request_body"]

    def test_copy_translated_anthropic_adapter_response_headers(self):
        sig = inspect.signature(mod._copy_translated_anthropic_adapter_response_headers)
        params = list(sig.parameters)
        assert params == ["translated_response", "upstream_response"]

    def test_get_anthropic_adapter_access_log_target_label(self):
        sig = inspect.signature(mod._get_anthropic_adapter_access_log_target_label)
        assert list(sig.parameters) == ["target_url"]

    def test_annotate_request_scope_for_adapted_access_log(self):
        sig = inspect.signature(mod._annotate_request_scope_for_adapted_access_log)
        assert list(sig.parameters) == ["request", "target_url"]

    def test_serialize_anthropic_adapter_response(self):
        sig = inspect.signature(mod._serialize_anthropic_adapter_response)
        assert list(sig.parameters) == ["response_obj"]

    def test_build_anthropic_response_from_completion_adapter_response(self):
        sig = inspect.signature(mod._build_anthropic_response_from_completion_adapter_response)
        assert list(sig.parameters) == ["response_obj"]

    def test_get_anthropic_adapter_openai_target_base(self):
        sig = inspect.signature(mod._get_anthropic_adapter_openai_target_base)
        params = list(sig.parameters)
        assert params == ["request", "prefer_chatgpt_codex_backend"]

    def test_get_anthropic_adapter_openrouter_api_key(self):
        sig = inspect.signature(mod._get_anthropic_adapter_openrouter_api_key)
        assert list(sig.parameters) == []

    def test_get_anthropic_adapter_nvidia_api_key(self):
        sig = inspect.signature(mod._get_anthropic_adapter_nvidia_api_key)
        assert list(sig.parameters) == []

    def test_get_anthropic_adapter_nvidia_target_base(self):
        sig = inspect.signature(mod._get_anthropic_adapter_nvidia_target_base)
        assert list(sig.parameters) == []

    def test_get_anthropic_adapter_openrouter_target_base(self):
        sig = inspect.signature(mod._get_anthropic_adapter_openrouter_target_base)
        assert list(sig.parameters) == []

    def test_resolve_anthropic_openai_responses_adapter_auth_context(self):
        sig = inspect.signature(mod._resolve_anthropic_openai_responses_adapter_auth_context)
        assert list(sig.parameters) == ["request"]

    def test_build_anthropic_responses_adapter_request_body(self):
        sig = inspect.signature(mod._build_anthropic_responses_adapter_request_body)
        params = list(sig.parameters)
        assert "request_body" in params
        assert "adapter_model" in params

    def test_prepare_anthropic_completion_adapter_request_body(self):
        sig = inspect.signature(mod._prepare_anthropic_completion_adapter_request_body)
        params = list(sig.parameters)
        assert "prepared_request_body" in params
        assert "adapter_model" in params

    def test_apply_anthropic_responses_adapter_common_request_policies(self):
        sig = inspect.signature(mod._apply_anthropic_responses_adapter_common_request_policies)
        params = list(sig.parameters)
        assert "prepared_request_body" in params
        assert "translated_request_body" in params

    def test_apply_anthropic_responses_adapter_policies_from_config(self):
        sig = inspect.signature(mod._apply_anthropic_responses_adapter_policies_from_config)
        params = list(sig.parameters)
        assert "prepared_request_body" in params
        assert "translated_request_body" in params
        assert "config" in params

    def test_finalize_anthropic_responses_adapter_upstream_response(self):
        sig = inspect.signature(mod._finalize_anthropic_responses_adapter_upstream_response)
        params = list(sig.parameters)
        assert "upstream_response" in params
        assert "request" in params
        assert "adapter_model" in params

    def test_finalize_anthropic_responses_adapter_from_config(self):
        sig = inspect.signature(mod._finalize_anthropic_responses_adapter_from_config)
        params = list(sig.parameters)
        assert "config" in params
        assert "upstream_response" in params

    def test_perform_anthropic_responses_adapter_pass_through(self):
        sig = inspect.signature(mod._perform_anthropic_responses_adapter_pass_through)
        params = list(sig.parameters)
        assert "config" in params
        assert "request" in params
        assert "translated_request_body" in params
        assert "pass_through_fn" in params

    def test_perform_normalized_anthropic_completion_adapter_stream(self):
        sig = inspect.signature(mod._perform_normalized_anthropic_completion_adapter_stream)
        params = list(sig.parameters)
        assert "handler" in params
        assert "handler_call_kwargs" in params
        assert "completion_stream_normalizer" in params

    def test_is_anthropic_messages_response(self):
        sig = inspect.signature(mod._is_anthropic_messages_response)
        assert list(sig.parameters) == ["value"]

    def test_finalize_anthropic_completion_adapter_response(self):
        sig = inspect.signature(mod._finalize_anthropic_completion_adapter_response)
        params = list(sig.parameters)
        assert "completion_response" in params
        assert "stream_flag" in params
        assert "fake_stream" in params

    def test_perform_anthropic_completion_adapter_messages_call(self):
        sig = inspect.signature(mod._perform_anthropic_completion_adapter_messages_call)
        params = list(sig.parameters)
        assert "config" in params
        assert "request" in params
        assert "prepared_request_body" in params
        assert "adapter_model" in params
        assert "target_url" in params
        assert "api_key" in params
        assert "api_base" in params

    def test_add_route_family_logging_metadata(self):
        sig = inspect.signature(mod._add_route_family_logging_metadata)
        assert list(sig.parameters) == ["request_body", "route_family"]

    def test_add_codex_native_tool_alias_adapter_metadata(self):
        sig = inspect.signature(mod._add_codex_native_tool_alias_adapter_metadata)
        params = list(sig.parameters)
        assert params == ["adapter_tags", "adapter_extra_fields", "enabled"]


# ---------------------------------------------------------------------------
# Async parity
# ---------------------------------------------------------------------------


class TestAsyncParity:
    """Verify async functions are coroutines and sync functions are not."""

    _ASYNC_NAMES = (
        "_resolve_anthropic_openai_responses_adapter_auth_context",
        "_finalize_anthropic_responses_adapter_upstream_response",
        "_finalize_anthropic_responses_adapter_from_config",
        "_perform_anthropic_responses_adapter_pass_through",
        "_perform_normalized_anthropic_completion_adapter_stream",
        "_perform_anthropic_completion_adapter_messages_call",
    )

    _SYNC_NAMES = (
        "_decode_http_response_body",
        "_build_adapted_route_rollup_kwargs",
        "_emit_adapted_route_access_log",
        "_record_adapted_completed_route_rollup_turn",
        "_record_adapted_completed_route_rollup_after_stream",
        "_normalize_openai_function_tool_parameters",
        "_sanitize_openai_object_schema_properties",
        "_normalize_openai_function_tool_schemas",
        "_get_openai_adapter_function_tool_names",
        "_apply_responses_adapter_parallel_instruction_policy",
        "_apply_openai_adapter_parallel_instruction_policy",
        "_apply_openrouter_adapter_parallel_instruction_policy",
        "_get_latest_adapter_user_prompt_text",
        "_prompt_explicitly_requests_bash_tool",
        "_maybe_force_explicit_bash_tool_choice_for_responses_adapter",
        "_apply_forced_bash_tool_choice_for_responses_adapter",
        "_maybe_force_explicit_bash_tool_choice_for_completion_adapter",
        "_responses_request_contains_mcp_tools",
        "_coerce_mapping_to_namespace",
        "_drop_anthropic_grok_native_prior_function_call_replay",
        "_build_anthropic_response_from_responses_response",
        "_build_completion_adapter_metadata",
        "_copy_translated_anthropic_adapter_response_headers",
        "_get_anthropic_adapter_access_log_target_label",
        "_annotate_request_scope_for_adapted_access_log",
        "_serialize_anthropic_adapter_response",
        "_build_anthropic_response_from_completion_adapter_response",
        "_get_anthropic_adapter_openai_target_base",
        "_get_anthropic_adapter_openrouter_api_key",
        "_get_anthropic_adapter_nvidia_api_key",
        "_get_anthropic_adapter_nvidia_target_base",
        "_get_anthropic_adapter_openrouter_target_base",
        "_build_anthropic_responses_adapter_request_body",
        "_prepare_anthropic_completion_adapter_request_body",
        "_apply_anthropic_responses_adapter_common_request_policies",
        "_apply_anthropic_responses_adapter_policies_from_config",
        "_is_anthropic_messages_response",
        "_finalize_anthropic_completion_adapter_response",
        "_add_route_family_logging_metadata",
        "_add_codex_native_tool_alias_adapter_metadata",
    )

    def test_async_functions_are_coroutines(self):
        for name in self._ASYNC_NAMES:
            fn = getattr(mod, name)
            assert inspect.iscoroutinefunction(fn), f"{name} should be async"

    def test_sync_functions_are_not_coroutines(self):
        for name in self._SYNC_NAMES:
            fn = getattr(mod, name)
            assert not inspect.iscoroutinefunction(fn), f"{name} should be sync"


# ---------------------------------------------------------------------------
# Representative behavior: success paths
# ---------------------------------------------------------------------------


class TestDecodeHttpResponseBody:
    def test_bytes_decode(self):
        assert mod._decode_http_response_body(b"hello") == "hello"

    def test_invalid_utf8_replaced(self):
        result = mod._decode_http_response_body(b"\xff\xfe")
        assert isinstance(result, str)


class TestBuildAdaptedRouteRollupKwargs:
    def test_wraps_metadata(self):
        result = mod._build_adapted_route_rollup_kwargs({"key": "val"})
        assert result == {"litellm_params": {"metadata": {"key": "val"}}}

    def test_copies_dict(self):
        original = {"a": 1}
        result = mod._build_adapted_route_rollup_kwargs(original)
        original["b"] = 2
        assert "b" not in result["litellm_params"]["metadata"]


class TestNormalizeOpenAIFunctionToolParameters:
    def test_non_dict_returns_default(self):
        assert mod._normalize_openai_function_tool_parameters(None) == {"type": "object", "properties": {}}

    def test_adds_type_and_properties(self):
        result = mod._normalize_openai_function_tool_parameters({"foo": "bar"})
        assert result["type"] == "object"
        assert result["properties"] == {}

    def test_preserves_existing_type(self):
        result = mod._normalize_openai_function_tool_parameters({"type": "object", "properties": {"x": {}}})
        assert result["type"] == "object"
        assert result["properties"] == {"x": {}}


class TestSanitizeOpenAIObjectSchemaProperties:
    def test_fixes_missing_properties(self):
        node = {"type": "object"}
        count = mod._sanitize_openai_object_schema_properties(node)
        assert count == 1
        assert node["properties"] == {}

    def test_no_fix_needed(self):
        node = {"type": "object", "properties": {"a": {}}}
        count = mod._sanitize_openai_object_schema_properties(node)
        assert count == 0


class TestNormalizeOpenAIFunctionToolSchemas:
    def test_normalizes_function_tools(self):
        body = {"tools": [{"type": "function", "parameters": None}]}
        mod._normalize_openai_function_tool_schemas(body)
        assert body["tools"][0]["parameters"]["type"] == "object"

    def test_ignores_non_function_tools(self):
        body = {"tools": [{"type": "mcp", "parameters": None}]}
        mod._normalize_openai_function_tool_schemas(body)
        assert body["tools"][0]["parameters"] is None

    def test_no_tools(self):
        body: dict[str, Any] = {}
        mod._normalize_openai_function_tool_schemas(body)
        assert "tools" not in body


class TestGetOpenAIAdapterFunctionToolNames:
    def test_extracts_names(self):
        body = {"tools": [{"type": "function", "name": "Bash"}, {"type": "mcp"}]}
        assert mod._get_openai_adapter_function_tool_names(body) == ["Bash"]

    def test_empty(self):
        assert mod._get_openai_adapter_function_tool_names({}) == []


class TestParallelInstructionPolicy:
    def test_no_parallel_flag(self):
        body = {"instructions": "hello"}
        result, changes = mod._apply_responses_adapter_parallel_instruction_policy(
            body, tag_prefix="t", metadata_prefix="m", span_name="s"
        )
        assert result is body
        assert changes == {}

    def test_applies_policy(self):
        body = {
            "parallel_tool_calls": True,
            "instructions": "do stuff",
            "tools": [
                {"type": "function", "name": "Bash"},
                {"type": "function", "name": "Read"},
            ],
        }
        result, changes = mod._apply_responses_adapter_parallel_instruction_policy(
            body, tag_prefix="t", metadata_prefix="m", span_name="s"
        )
        assert "m_parallel_instruction_policy_applied" in changes
        assert result["instructions"].startswith(mod._OPENAI_ADAPTER_PARALLEL_FUNCTION_TOOL_INSTRUCTIONS)

    def test_openai_variant(self):
        body = {
            "parallel_tool_calls": True,
            "instructions": "do stuff",
            "tools": [
                {"type": "function", "name": "Bash"},
                {"type": "function", "name": "Read"},
            ],
        }
        result, changes = mod._apply_openai_adapter_parallel_instruction_policy(body)
        assert "openai_adapter_parallel_instruction_policy_applied" in changes

    def test_openrouter_variant(self):
        body = {
            "parallel_tool_calls": True,
            "instructions": "do stuff",
            "tools": [
                {"type": "function", "name": "Bash"},
                {"type": "function", "name": "Read"},
            ],
        }
        result, changes = mod._apply_openrouter_adapter_parallel_instruction_policy(body)
        assert "openrouter_adapter_parallel_instruction_policy_applied" in changes


class TestBashToolChoice:
    def test_prompt_detection(self):
        assert mod._prompt_explicitly_requests_bash_tool("use the Bash Tool") is True
        assert mod._prompt_explicitly_requests_bash_tool("hello") is False
        assert mod._prompt_explicitly_requests_bash_tool(None) is False

    def test_get_latest_user_prompt(self):
        body = {"messages": [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "reply"},
            {"role": "user", "content": "second"},
        ]}
        assert mod._get_latest_adapter_user_prompt_text(body) == "second"

    def test_force_responses_adapter(self):
        request_body = {"messages": [{"role": "user", "content": "use the bash tool"}]}
        translated_body = {"tools": [{"type": "function", "name": "Bash"}]}
        changes = mod._maybe_force_explicit_bash_tool_choice_for_responses_adapter(request_body, translated_body)
        assert changes == {"forced_explicit_bash_tool_choice": "Bash"}
        assert translated_body["tool_choice"] == {"type": "function", "name": "Bash"}

    def test_force_completion_adapter(self):
        request_body = {
            "messages": [{"role": "user", "content": "run the bash command"}],
            "tools": [{"name": "Bash"}],
        }
        changes = mod._maybe_force_explicit_bash_tool_choice_for_completion_adapter(request_body)
        assert changes == {"forced_explicit_bash_tool_choice": "Bash"}
        assert request_body["tool_choice"] == {"type": "tool", "name": "Bash"}


class TestMCPTools:
    def test_contains_mcp(self):
        assert mod._responses_request_contains_mcp_tools({"tools": [{"type": "mcp"}]}) is True

    def test_no_mcp(self):
        assert mod._responses_request_contains_mcp_tools({"tools": [{"type": "function"}]}) is False

    def test_no_tools(self):
        assert mod._responses_request_contains_mcp_tools({}) is False


class TestCoerceMappingToNamespace:
    def test_dict_to_namespace(self):
        result = mod._coerce_mapping_to_namespace({"a": {"b": 1}})
        assert isinstance(result, SimpleNamespace)
        assert isinstance(result.a, SimpleNamespace)
        assert result.a.b == 1

    def test_list_preserved(self):
        result = mod._coerce_mapping_to_namespace([{"x": 1}])
        assert isinstance(result, list)
        assert isinstance(result[0], SimpleNamespace)

    def test_depth_bound(self):
        deep: dict[str, Any] = {"a": 1}
        result = mod._coerce_mapping_to_namespace(deep, _depth=100)
        assert result == deep  # returned as-is


class TestGrokReplayDrop:
    def test_drops_function_calls_and_outputs(self):
        body = {
            "input": [
                {"type": "function_call", "call_id": "c1", "name": "Bash"},
                {"type": "function_call_output", "call_id": "c1"},
                {"type": "message", "content": "keep"},
            ]
        }
        updated, dropped = mod._drop_anthropic_grok_native_prior_function_call_replay(body)
        assert len(dropped) == 2
        assert len(updated["input"]) == 1
        assert updated["input"][0]["type"] == "message"

    def test_no_input(self):
        body: dict[str, Any] = {}
        updated, dropped = mod._drop_anthropic_grok_native_prior_function_call_replay(body)
        assert updated is body
        assert dropped == []


class TestBuildCompletionAdapterMetadata:
    def test_mirrors_trace_fields(self):
        body = {
            "metadata": {"existing": True},
            "litellm_metadata": {
                "session_id": "s1",
                "trace_id": "t1",
                "agent_name": "worker",
                "tags": ["tag1"],
            },
        }
        result = mod._build_completion_adapter_metadata(body)
        assert result["session_id"] == "s1"
        assert result["agent_name"] == "worker"
        assert "tag1" in result["tags"]
        assert result["existing"] is True

    def test_no_litellm_metadata(self):
        body = {"metadata": {"a": 1}}
        result = mod._build_completion_adapter_metadata(body)
        assert result == {"a": 1}


class TestCopyResponseHeaders:
    def test_copies_non_hop_headers(self):
        upstream = Response(content="ok")
        upstream.headers["x-custom"] = "val"
        upstream.headers["content-length"] = "2"
        translated = Response(content="translated")
        mod._copy_translated_anthropic_adapter_response_headers(
            translated_response=translated, upstream_response=upstream
        )
        assert translated.headers["x-custom"] == "val"
        assert translated.headers.get("content-length") != "2"


class TestAccessLogTargetLabel:
    def test_basic_url(self):
        label = mod._get_anthropic_adapter_access_log_target_label("https://api.openai.com/v1/responses?x=1")
        assert label == "api.openai.com/v1/responses?x=1"

    def test_no_query(self):
        label = mod._get_anthropic_adapter_access_log_target_label("https://example.com/path")
        assert label == "example.com/path"


class TestSerializeResponse:
    def test_model_dump_json(self):
        obj = MagicMock()
        obj.model_dump_json.return_value = '{"a":1}'
        assert mod._serialize_anthropic_adapter_response(obj) == '{"a":1}'

    def test_json_fallback(self):
        obj = MagicMock(spec=[])
        obj.json = MagicMock(return_value='{"b":2}')
        assert mod._serialize_anthropic_adapter_response(obj) == '{"b":2}'

    def test_json_dumps_fallback(self):
        assert mod._serialize_anthropic_adapter_response({"c": 3}) == '{"c": 3}'


class TestBuildCompletionAdapterResponse:
    def test_returns_json_response(self):
        obj = MagicMock()
        obj.model_dump_json.return_value = '{"ok":true}'
        resp = mod._build_anthropic_response_from_completion_adapter_response(obj)
        assert isinstance(resp, Response)
        assert resp.media_type == "application/json"


class TestIsAnthropicMessagesResponse:
    def test_dict_is_true(self):
        assert mod._is_anthropic_messages_response({}) is True

    def test_non_dict_is_false(self):
        assert mod._is_anthropic_messages_response("str") is False
        assert mod._is_anthropic_messages_response(None) is False


class TestAddRouteFamilyLoggingMetadata:
    def test_adds_metadata(self):
        body: dict[str, Any] = {}
        result = mod._add_route_family_logging_metadata(body, "anthropic_openai_responses_adapter")
        lm = result.get("litellm_metadata", {})
        assert lm.get("passthrough_route_family") == "anthropic_openai_responses_adapter"

    def test_empty_family_noop(self):
        body: dict[str, Any] = {"x": 1}
        result = mod._add_route_family_logging_metadata(body, "")
        assert result is body


class TestCodexNativeToolMetadata:
    def test_enabled(self):
        tags: list[str] = []
        fields: dict[str, Any] = {}
        mod._add_codex_native_tool_alias_adapter_metadata(tags, fields, enabled=True)
        assert "anthropic-openai-codex-native-tools" in tags
        assert fields["anthropic_adapter_codex_native_tool_aliases"] is True

    def test_disabled(self):
        tags: list[str] = []
        fields: dict[str, Any] = {}
        mod._add_codex_native_tool_alias_adapter_metadata(tags, fields, enabled=False)
        assert tags == []
        assert fields == {}


def _clean_secret_string_stub(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    cleaned = value.strip()
    if len(cleaned) >= 2 and cleaned[0] == cleaned[-1] and cleaned[0] in {'"', "'"}:
        cleaned = cleaned[1:-1].strip()
    return cleaned or None


class TestNvidiaTargetBase:
    @pytest.fixture(autouse=True)
    def _inject_stubs(self):
        setattr(mod, "_clean_secret_string", _clean_secret_string_stub)
        setattr(mod, "_get_first_secret_value", lambda *a: None)
        yield

    def test_default(self):
        with patch.dict("os.environ", {}, clear=True):
            result = mod._get_anthropic_adapter_nvidia_target_base()
        assert result == "https://integrate.api.nvidia.com"

    def test_env_override(self):
        with patch.dict("os.environ", {"NVIDIA_NIM_API_BASE": "https://custom.example.com/v1"}):
            result = mod._get_anthropic_adapter_nvidia_target_base()
        assert result == "https://custom.example.com"


class TestOpenAITargetBase:
    @pytest.fixture(autouse=True)
    def _inject_stubs(self):
        # Production install() rebinds extracted functions so their
        # __globals__ is the host module namespace, not this extracted
        # module's dict.  Patch through the live globals the rebound
        # function actually uses.
        live_globals = mod._get_anthropic_adapter_openai_target_base.__globals__
        key = "_anthropic_adapter_request_uses_codex_native_auth"
        original = live_globals.get(key)
        live_globals[key] = lambda req: False
        self._live_globals = live_globals
        self._key = key
        self._original = original
        yield
        if original is not None:
            live_globals[key] = original
        else:
            live_globals.pop(key, None)

    def test_default(self):
        mock_request = MagicMock()
        with patch.dict("os.environ", {}, clear=True):
            result = mod._get_anthropic_adapter_openai_target_base(mock_request)
        assert result == "https://api.openai.com/"

    def test_chatgpt_backend(self):
        mock_request = MagicMock()
        self._live_globals[self._key] = lambda req: True
        with patch.dict("os.environ", {}, clear=True):
            result = mod._get_anthropic_adapter_openai_target_base(mock_request)
        assert result != "https://api.openai.com/"


# ---------------------------------------------------------------------------
# Representative behavior: failure paths
# ---------------------------------------------------------------------------


class TestFailurePaths:
    def test_finalize_completion_non_stream(self):
        """Non-stream finalize builds a JSON response."""
        mock_response = MagicMock()
        mock_response.model_dump_json.return_value = '{"content":[]}'
        result = mod._finalize_anthropic_completion_adapter_response(
            completion_response=mock_response,
            stream_flag=False,
            fake_stream=False,
            rollup_kwargs={"litellm_params": {"metadata": {}}},
            adapter_label="test",
        )
        assert isinstance(result, Response)

    def test_finalize_completion_fake_stream_requires_dict(self):
        """Fake stream with non-dict response raises TypeError."""
        with pytest.raises(TypeError, match="Fake Anthropic streaming"):
            mod._finalize_anthropic_completion_adapter_response(
                completion_response="not-a-dict",
                stream_flag=True,
                fake_stream=True,
                rollup_kwargs={"litellm_params": {"metadata": {}}},
                adapter_label="test",
            )


# ---------------------------------------------------------------------------
# Representative behavior: stream path
# ---------------------------------------------------------------------------


class TestStreamBehavior:
    def test_record_rollup_after_stream_wraps_iterator(self):
        """Stream rollup wraps the body_iterator."""
        async def _gen():
            yield b"chunk1"
            yield b"chunk2"

        response = StreamingResponse(_gen(), media_type="text/event-stream")
        result = mod._record_adapted_completed_route_rollup_after_stream(
            response,
            {"litellm_params": {"metadata": {}}},
            adapter_label="test",
        )
        assert result is response
        assert result.body_iterator is not _gen()


# ---------------------------------------------------------------------------
# install() contract
# ---------------------------------------------------------------------------


class TestInstall:
    @pytest.fixture(autouse=True)
    def _restore_module(self):
        """Save and restore module-level function bindings around install() tests."""
        saved = {}
        for name in mod._EXTRACTED_FUNCTION_NAMES:
            saved[name] = getattr(mod, name, None)
        yield
        for name, obj in saved.items():
            if obj is not None:
                setattr(mod, name, obj)

    def test_install_publishes_functions_to_host(self):
        host: dict[str, Any] = dict(vars(mod))
        mod.install(host)
        for name in mod._EXTRACTED_FUNCTION_NAMES:
            assert name in host, f"install() missing function {name}"
            assert callable(host[name])

    def test_install_publishes_constants_to_host(self):
        host: dict[str, Any] = dict(vars(mod))
        mod.install(host)
        for name in mod._EXTRACTED_CONSTANT_NAMES:
            assert name in host, f"install() missing constant {name}"

    def test_install_rebinds_globals(self):
        host: dict[str, Any] = dict(vars(mod))
        host["sentinel"] = True
        mod.install(host)
        fn = host["_decode_http_response_body"]
        assert fn.__globals__ is host

    def test_install_produces_callable_rebound(self):
        host: dict[str, Any] = dict(vars(mod))
        mod.install(host)
        fn = host["_decode_http_response_body"]
        assert fn.__name__ == "_decode_http_response_body"
        assert fn(b"x") == "x"


# ---------------------------------------------------------------------------
# Constants pinning
# ---------------------------------------------------------------------------


class TestConstants:
    def test_retryable_status_codes(self):
        assert 429 not in mod._AAWM_ALIAS_CANDIDATE_RETRYABLE_UPSTREAM_STATUS_CODES
        assert 500 in mod._AAWM_ALIAS_CANDIDATE_RETRYABLE_UPSTREAM_STATUS_CODES

    def test_retryable_default_includes_429(self):
        assert 429 in mod._AAWM_ALIAS_CANDIDATE_RETRYABLE_UPSTREAM_STATUS_CODES_DEFAULT

    def test_walk_max_depth(self):
        assert mod._AAWM_REQUEST_BODY_WALK_MAX_DEPTH == 64

    def test_nvidia_env_vars(self):
        assert "NVIDIA_NIM_API_KEY" in mod._ANTHROPIC_ADAPTER_NVIDIA_API_KEY_ENV_VARS

    def test_parallel_instructions_content(self):
        assert "Parallel tool calls" in mod._OPENAI_ADAPTER_PARALLEL_FUNCTION_TOOL_INSTRUCTIONS


# ---------------------------------------------------------------------------
# D1-521: access logging receives final provider_bound_body
# ---------------------------------------------------------------------------


class TestD1521ProviderBoundBodyBoundary:
    """Access logging must receive translated completion kwargs and alias label."""

    @staticmethod
    def _patch_live_emit(monkeypatch, captured: dict[str, Any]) -> None:
        """Patch emit through rebound install() globals, not only module attrs."""
        live_globals = mod._emit_adapted_route_access_log.__globals__

        def _capture(**kwargs):
            captured.clear()
            captured.update(kwargs)

        monkeypatch.setitem(live_globals, "emit_aawm_route_access_log", _capture)
        # Keep module-local path covered when functions are not rebound.
        monkeypatch.setattr(mod, "emit_aawm_route_access_log", _capture, raising=False)

    @staticmethod
    def _patch_live_callable(monkeypatch, func_name: str, value) -> None:
        live_globals = getattr(mod, func_name).__globals__
        monkeypatch.setitem(live_globals, func_name, value)
        monkeypatch.setattr(mod, func_name, value, raising=False)

    def test_emit_adapted_route_access_log_forwards_provider_bound_body(self, monkeypatch):
        captured: dict[str, Any] = {}
        self._patch_live_emit(monkeypatch, captured)

        request_body = {"model": "alias-model", "reasoning_effort": "xhigh"}
        provider_bound_body = {
            "model": "upstream-model",
            "reasoning_effort": "high",
            "messages": [],
        }
        rollup_kwargs = {"litellm_params": {"metadata": {"model_alias": "alias-model"}}}

        mod._emit_adapted_route_access_log(
            request=MagicMock(),
            target_url="https://example.test/v1/chat/completions",
            request_body=request_body,
            rollup_kwargs=rollup_kwargs,
            adapter_label="test-adapter",
            provider_bound_body=provider_bound_body,
        )

        assert captured["request_body"] is request_body
        assert captured["provider_bound_body"] is provider_bound_body
        assert captured["provider_bound_body"]["reasoning_effort"] == "high"
        assert captured["kwargs"] is rollup_kwargs

    @pytest.mark.asyncio
    async def test_completion_messages_call_logs_prepared_completion_kwargs(self, monkeypatch):
        prepared_request_body = {
            "model": "alias-model",
            "max_tokens": 16,
            "messages": [{"role": "user", "content": "hi"}],
            "reasoning_effort": "xhigh",
            "litellm_metadata": {"model_alias": "alias-model"},
            "stream": False,
        }
        translated_completion_kwargs = {
            "model": "upstream-model",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 16,
            "reasoning_effort": "high",
            "custom_llm_provider": "openrouter",
        }
        captured_emit: dict[str, Any] = {}
        acompletion_kwargs: dict[str, Any] = {}

        class _Handler:
            @staticmethod
            def _prepare_completion_kwargs(**kwargs):
                return dict(translated_completion_kwargs), {"tool": "mapped"}

            @staticmethod
            def _transform_completion_response(completion_response, **kwargs):
                return completion_response

        async def _fake_acompletion(**kwargs):
            acompletion_kwargs.update(kwargs)
            return SimpleNamespace(model_dump_json=lambda **_k: '{"type":"message","content":[]}')

        monkeypatch.setattr(
            "litellm.llms.anthropic.experimental_pass_through.adapters.handler."
            "LiteLLMMessagesToCompletionTransformationHandler",
            _Handler,
        )
        # acompletion is resolved via the rebound function globals after install().
        live_globals = mod._perform_anthropic_completion_adapter_messages_call.__globals__
        litellm_obj = live_globals.get("litellm", mod.litellm)
        monkeypatch.setattr(litellm_obj, "acompletion", _fake_acompletion)
        self._patch_live_emit(monkeypatch, captured_emit)
        self._patch_live_callable(
            monkeypatch,
            "_annotate_request_scope_for_adapted_access_log",
            lambda *a, **k: None,
        )
        self._patch_live_callable(
            monkeypatch,
            "_build_adapted_route_rollup_kwargs",
            lambda metadata: {"litellm_params": {"metadata": dict(metadata or {})}},
        )
        self._patch_live_callable(
            monkeypatch,
            "_finalize_anthropic_completion_adapter_response",
            lambda **kwargs: Response(content=b'{"ok":true}', media_type="application/json"),
        )

        result = await mod._perform_anthropic_completion_adapter_messages_call(
            config=SimpleNamespace(adapter_label="OpenRouter", custom_llm_provider="openrouter"),
            request=MagicMock(headers={}),
            prepared_request_body=prepared_request_body,
            adapter_model="alias-model",
            target_url="https://openrouter.ai/api/v1/chat/completions",
            api_key="sk-test",
            api_base="https://openrouter.ai/api/v1",
            client_requested_stream=False,
            model_for_upstream="upstream-model",
        )

        assert isinstance(result, Response)
        assert captured_emit["request_body"] is prepared_request_body
        assert captured_emit["request_body"]["model"] == "alias-model"
        assert captured_emit["provider_bound_body"] == translated_completion_kwargs
        assert acompletion_kwargs == translated_completion_kwargs
