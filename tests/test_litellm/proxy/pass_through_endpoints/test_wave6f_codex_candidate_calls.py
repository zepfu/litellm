"""Wave 6F module-local tests for codex_candidate_calls.py.

Pins: symbol inventory, signatures/async parity, representative dispatch
success/failure/stream behavior, callback ordering, and absence of god import.
"""

from __future__ import annotations

import ast
import inspect
import pathlib
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import unittest.mock

import pytest

from litellm.proxy.pass_through_endpoints.aawm_adapter_runtime import (
    codex_candidate_calls,
)

# ── Symbol inventory ────────────────────────────────────────────────

EXPECTED_PUBLIC_SYMBOLS = frozenset(
    {
        "install",
        "_HOST_FUNCTION_NAMES",
        "_perform_codex_auto_agent_alias_candidate_request",
        "_perform_codex_auto_agent_native_openai_request",
        "_perform_codex_auto_agent_grok_native_responses_request",
        "_perform_codex_auto_agent_oa_xai_responses_request",
        "_validate_codex_auto_agent_openrouter_responses_stream",
        "_perform_codex_auto_agent_openrouter_responses_request",
        "_perform_codex_auto_agent_openrouter_completion_request",
        "_prepare_codex_kimi_chat_completions_adapter_route",
        "_perform_codex_kimi_chat_completions_adapter_call",
        "_handle_codex_kimi_chat_completions_adapter_route",
        "_prepare_codex_alibaba_token_plan_adapter_route",
        "_perform_codex_alibaba_token_plan_adapter_call",
        "_handle_codex_alibaba_token_plan_adapter_route",
        "_handle_codex_opencode_zen_adapter_route",
        "_consume_opencode_zen_tools_mode_header",
        "_build_opencode_zen_completion_call_kwargs",
        "_perform_opencode_zen_completion_call",
        "_prepare_opencode_zen_direct_observability_metadata",
        "_prepare_opencode_zen_known_free_logging",
        "_opencode_zen_callback_headers",
        # D1-574 OpenCode direct 429
        "_opencode_zen_direct_safe_retry_after",
        "_maybe_raise_opencode_zen_direct_rate_limit",
        "_opencode_zen_direct_stream_terminal_error",
        "_OPENCODE_ZEN_DIRECT_429_ERROR_CLASSES",
        "_OPENCODE_ZEN_DIRECT_RETRY_AFTER_CEILING_SECONDS",
        "_OPENCODE_ZEN_DIRECT_PEEK_MAX_BYTES",
    }
)


class TestSymbolInventory:
    """Pin the exact public symbol set of the module."""

    def test_expected_symbols_present(self):
        for name in EXPECTED_PUBLIC_SYMBOLS:
            assert hasattr(codex_candidate_calls, name), f"Missing symbol: {name}"

    def test_host_function_names_matches(self):
        """_HOST_FUNCTION_NAMES must match the extractable function symbols."""
        expected_fns = EXPECTED_PUBLIC_SYMBOLS - {"install", "_HOST_FUNCTION_NAMES"}
        assert set(codex_candidate_calls._HOST_FUNCTION_NAMES) == expected_fns
        assert (
            sum(
                callable(getattr(codex_candidate_calls, name))
                for name in codex_candidate_calls._HOST_FUNCTION_NAMES
            )
            == 23
        )


# ── Signatures and async parity ─────────────────────────────────────

ASYNC_FUNCTIONS = frozenset(
    {
        "_perform_codex_auto_agent_alias_candidate_request",
        "_perform_codex_auto_agent_native_openai_request",
        "_perform_codex_auto_agent_grok_native_responses_request",
        "_perform_codex_auto_agent_oa_xai_responses_request",
        "_validate_codex_auto_agent_openrouter_responses_stream",
        "_perform_codex_auto_agent_openrouter_responses_request",
        "_perform_codex_auto_agent_openrouter_completion_request",
        "_prepare_codex_kimi_chat_completions_adapter_route",
        "_perform_codex_kimi_chat_completions_adapter_call",
        "_handle_codex_kimi_chat_completions_adapter_route",
        "_prepare_codex_alibaba_token_plan_adapter_route",
        "_perform_codex_alibaba_token_plan_adapter_call",
        "_handle_codex_alibaba_token_plan_adapter_route",
        "_handle_codex_opencode_zen_adapter_route",
        "_perform_opencode_zen_completion_call",
    }
)


class TestSignaturesAndAsyncParity:
    """All extracted functions must be async and keyword-only."""

    @pytest.mark.parametrize("fn_name", sorted(ASYNC_FUNCTIONS))
    def test_is_coroutine_function(self, fn_name: str):
        fn = getattr(codex_candidate_calls, fn_name)
        assert inspect.iscoroutinefunction(fn), f"{fn_name} must be async"

    @pytest.mark.parametrize("fn_name", sorted(ASYNC_FUNCTIONS))
    def test_keyword_only_params(self, fn_name: str):
        """All params after the first positional (if any) must be keyword-only."""
        fn = getattr(codex_candidate_calls, fn_name)
        sig = inspect.signature(fn)
        params = list(sig.parameters.values())
        # _validate_codex_auto_agent_openrouter_responses_stream has one positional
        if fn_name == "_validate_codex_auto_agent_openrouter_responses_stream":
            assert params[0].kind == inspect.Parameter.POSITIONAL_OR_KEYWORD
            kw_params = params[1:]
        else:
            kw_params = params
        for p in kw_params:
            assert p.kind == inspect.Parameter.KEYWORD_ONLY, (
                f"{fn_name}: param '{p.name}' must be keyword-only"
            )

    def test_install_is_sync(self):
        assert not inspect.iscoroutinefunction(codex_candidate_calls.install)


# ── Absence of god import ───────────────────────────────────────────


class TestNoGodImport:
    """Module must not import llm_passthrough_endpoints at module scope."""

    def test_no_god_module_import_in_ast(self):
        src_path = pathlib.Path(codex_candidate_calls.__file__)
        tree = ast.parse(src_path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert "llm_passthrough_endpoints" not in alias.name, (
                        f"God import found: import {alias.name}"
                    )
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                assert "llm_passthrough_endpoints" not in module, (
                    f"God import found: from {module}"
                )


# ── Install seam ────────────────────────────────────────────────────


class TestInstallSeam:
    """install() must rebind all functions into host_globals."""

    def test_install_populates_host_globals(self):
        host: dict[str, Any] = {}
        codex_candidate_calls.install(host)
        for name in codex_candidate_calls._HOST_FUNCTION_NAMES:
            assert name in host, f"install() did not publish {name}"
            # Constants like _OPENCODE_ZEN_DIRECT_429_ERROR_CLASSES are not callable
            if not name.startswith("_OPENCODE_ZEN_DIRECT_"):
                assert callable(host[name])

    def test_install_rebinds_globals(self):
        host: dict[str, Any] = {"__builtins__": __builtins__}
        codex_candidate_calls.install(host)
        fn = host["_perform_codex_auto_agent_alias_candidate_request"]
        assert fn.__globals__ is host


# ── Dispatch behavior ───────────────────────────────────────────────


class TestDispatchBehavior:
    """Representative dispatch success/failure via the candidate dispatcher."""

    @pytest.fixture()
    def host_globals(self):
        """Provide a host namespace with stubs for host-global lookups."""
        host: dict[str, Any] = {"__builtins__": __builtins__}
        # Stub all host-global dependencies
        host["_dispatch_auto_agent_alias_candidate_request"] = AsyncMock()
        host["_CODEX_AUTO_AGENT_OPENCODE_PROVIDER"] = "opencode"
        host["_CODEX_AUTO_AGENT_KIMI_CODE_PROVIDER"] = "kimi_code"
        host["_CODEX_AUTO_AGENT_ALIBABA_TOKEN_PLAN_PROVIDER"] = "alibaba_token_plan"
        host["_CODEX_AUTO_AGENT_OPENROUTER_PROVIDER"] = "openrouter"
        host["_CODEX_AUTO_AGENT_XAI_PROVIDER"] = "xai"
        codex_candidate_calls.install(host)
        return host

    @pytest.mark.asyncio
    async def test_dispatch_success_delegates_to_dispatch_fn(self, host_globals):
        """Successful dispatch calls _dispatch_auto_agent_alias_candidate_request."""
        mock_response = MagicMock()
        host_globals["_dispatch_auto_agent_alias_candidate_request"].return_value = mock_response

        fn = host_globals["_perform_codex_auto_agent_alias_candidate_request"]
        result = await fn(
            endpoint="/v1/responses",
            request=MagicMock(),
            fastapi_response=MagicMock(),
            user_api_key_dict=MagicMock(),
            candidate={"model": "test-model", "provider": "openrouter"},
            candidate_body={"model": "test-model"},
            target_url="https://api.openai.com/v1/responses",
            api_key="sk-test",
            forward_headers=False,
        )
        assert result is mock_response
        host_globals["_dispatch_auto_agent_alias_candidate_request"].assert_awaited_once()

    @pytest.mark.asyncio
    async def test_dispatch_failure_propagates(self, host_globals):
        """Exceptions from dispatch propagate to caller."""
        host_globals["_dispatch_auto_agent_alias_candidate_request"].side_effect = RuntimeError("upstream fail")

        fn = host_globals["_perform_codex_auto_agent_alias_candidate_request"]
        with pytest.raises(RuntimeError, match="upstream fail"):
            await fn(
                endpoint="/v1/responses",
                request=MagicMock(),
                fastapi_response=MagicMock(),
                user_api_key_dict=MagicMock(),
                candidate={"model": "test-model", "provider": "unknown"},
                candidate_body={},
                target_url="https://api.openai.com/v1/responses",
                api_key=None,
                forward_headers=False,
            )


# ── Callback ordering (Kimi route) ─────────────────────────────────


class TestCallbackOrdering:
    """Kimi/Alibaba handle-route must call prepare -> perform -> validate -> rollup."""

    @pytest.fixture()
    def kimi_host(self):
        host: dict[str, Any] = {"__builtins__": __builtins__}
        call_order: list[str] = []

        from fastapi.responses import Response, StreamingResponse

        host["StreamingResponse"] = StreamingResponse
        host["Response"] = Response

        async def mock_driver_run(*, prepare, perform, **kwargs):
            call_order.append("driver_run")
            plan = MagicMock()
            plan.perform_kwargs = {"litellm_metadata": {}}
            plan.prepared_request_body = {"litellm_metadata": {}}
            plan.target_url = "https://kimi.example/v1"
            await prepare(
                request=kwargs["request"],
                prepared_request_body=kwargs["prepared_request_body"],
                adapter_model=kwargs["adapter_model"],
                use_alias_candidate_probe=kwargs.get("use_alias_candidate_probe", False),
            )
            return MagicMock()

        host["_aawm_adapter_driver"] = MagicMock()
        host["_aawm_adapter_driver"].run_completion_adapter_route = mock_driver_run
        host["_build_adapted_route_rollup_kwargs"] = MagicMock(return_value={})
        host["_annotate_request_scope_for_adapted_access_log"] = MagicMock()
        host["_emit_adapted_route_access_log"] = MagicMock()
        host["_validate_codex_auto_agent_responses_payload"] = AsyncMock(return_value=MagicMock())
        host["_build_malformed_tool_call_intake_context"] = MagicMock(return_value={})
        host["_record_adapted_completed_route_rollup_after_stream"] = MagicMock()
        host["_record_adapted_completed_route_rollup_turn"] = MagicMock()
        codex_candidate_calls.install(host)
        # Override AFTER install() so mocks replace rebound functions
        mock_plan = MagicMock()
        mock_plan.perform_kwargs = {"litellm_metadata": {}}
        mock_plan.prepared_request_body = {"litellm_metadata": {}}
        mock_plan.target_url = "https://kimi.example/v1"
        host["_prepare_codex_kimi_chat_completions_adapter_route"] = AsyncMock(return_value=mock_plan)
        host["_call_order"] = call_order
        return host

    @pytest.mark.asyncio
    async def test_kimi_route_calls_validate_and_rollup(self, kimi_host):
        """handle_codex_kimi must call validate then rollup_turn for non-stream."""
        fn = kimi_host["_handle_codex_kimi_chat_completions_adapter_route"]
        mock_request = MagicMock()
        mock_request.headers = {}
        await fn(
            endpoint="/v1/responses",
            request=mock_request,
            fastapi_response=MagicMock(),
            user_api_key_dict=MagicMock(),
            prepared_request_body={"model": "kimi-test"},
            adapter_model="kimi-test",
            use_alias_candidate_probe=False,
        )
        kimi_host["_validate_codex_auto_agent_responses_payload"].assert_awaited_once()
        kimi_host["_record_adapted_completed_route_rollup_turn"].assert_called_once()


# -- Fix 1 regression: namespace tool adaptation before chat-completion --


class TestOpenRouterCompletionNamespaceToolAdaptation:
    """_perform_codex_auto_agent_openrouter_completion_request must adapt
    namespace tools to flat dispatchable function tools before the chat
    completion transform, preserving tool call/result IDs."""

    @pytest.mark.asyncio
    async def test_completion_transform_receives_flat_dispatchable_tools(self):  # noqa: PLR0915
        """The transform boundary receives flat spawn_agent/exec_command tools
        (not functions.collaboration.*), and tool call/result IDs survive."""
        host: dict[str, Any] = {"__builtins__": __builtins__}
        from fastapi.responses import Response, StreamingResponse

        host["Response"] = Response
        host["StreamingResponse"] = StreamingResponse

        # Representative dispatchable tool identities plus call/result IDs.
        flat_tools = [
            {
                "type": "function",
                "function": {
                    "name": "spawn_agent",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "exec_command",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
        ]
        original_body = {
            "model": "test-model",
            "input": [
                {
                    "type": "function_call",
                    "id": "call-abc-123",
                    "call_id": "call-abc-123",
                    "name": "functions.collaboration.spawn_agent",
                    "arguments": "{}",
                },
                {
                    "type": "function_call_output",
                    "id": "result-def-456",
                    "call_id": "call-abc-123",
                    "output": "ok",
                },
            ],
            "tools": [
                {
                    "type": "namespace",
                    "name": "functions.collaboration",
                    "tools": [
                        {
                            "type": "function",
                            "name": "spawn_agent",
                            "parameters": {"type": "object", "properties": {}},
                        }
                    ],
                }
            ],
        }

        def mock_adapt_namespace(body):
            adapted = dict(body)
            adapted["tools"] = flat_tools
            return adapted, [
                {"name": "spawn_agent", "namespace": "functions.collaboration"}
            ]

        host["_adapt_codex_namespace_tools_to_functions_from_request_body"] = (
            mock_adapt_namespace
        )
        host["_get_openrouter_api_key"] = MagicMock(return_value="sk-test")
        host["_get_openrouter_completion_adapter_upstream_model"] = MagicMock(
            return_value=None
        )
        host["_get_openrouter_target_base"] = MagicMock(
            return_value="https://openrouter.ai"
        )
        host["_build_openrouter_default_headers"] = MagicMock(return_value={})
        host["_get_proxy_shared_aiohttp_session"] = MagicMock(return_value=None)
        host["_merge_litellm_metadata"] = MagicMock(
            side_effect=lambda body, **kw: body
        )
        host["_add_route_family_logging_metadata"] = MagicMock(
            side_effect=lambda body, family: body
        )
        host["_build_langfuse_span_descriptor"] = MagicMock(return_value={})
        host["_build_adapted_route_rollup_kwargs"] = MagicMock(return_value={})
        host["_emit_adapted_route_access_log"] = MagicMock()
        host["_annotate_request_scope_for_adapted_access_log"] = MagicMock()
        host["_record_adapted_completed_route_rollup_turn"] = MagicMock()
        host["_apply_openrouter_completion_message_sanitization"] = MagicMock(
            side_effect=lambda **kw: (
                kw["request_body"],
                kw["completion_kwargs"],
                kw["litellm_metadata"],
            )
        )
        host["_perform_openrouter_completion_adapter_operation"] = AsyncMock(
            return_value=MagicMock()
        )
        host["_serialize_responses_adapter_response"] = MagicMock(
            return_value='{"id":"resp-1","output":[],"status":"completed"}'
        )
        host["_is_codex_auto_agent_malformed_tool_call_text_output"] = MagicMock(
            return_value=False
        )
        host["_is_codex_auto_agent_empty_success_responses_body"] = MagicMock(
            return_value=False
        )
        host["_build_responses_response_from_adapter_response"] = MagicMock(
            return_value=MagicMock()
        )

        import httpx as _httpx

        host["httpx"] = _httpx

        egress_calls: list[dict[str, Any]] = []

        class _MockHelpers:
            @staticmethod
            def validate_outgoing_egress(**kwargs):
                egress_calls.append(kwargs)

        host["HttpPassThroughEndpointHelpers"] = _MockHelpers

        import litellm as _litellm

        host["litellm"] = _litellm
        from typing import cast as _cast

        host["cast"] = _cast
        host["ResponsesAPIOptionalRequestParams"] = dict  # TYPE_CHECKING stub
        import json as _json

        host["json"] = _json

        codex_candidate_calls.install(host)

        mock_request = MagicMock()
        mock_request.headers = {"x-test": "1"}

        captured_transform_kwargs: list[dict[str, Any]] = []

        def capture_transform(**kwargs):
            captured_transform_kwargs.append(kwargs)
            return {"model": kwargs.get("model"), "messages": [], "tools": kwargs.get("responses_api_request", {}).get("tools")}

        mock_config_cls = MagicMock()
        mock_config_cls.transform_responses_api_request_to_chat_completion_request.side_effect = (
            capture_transform
        )
        mock_config_cls.transform_chat_completion_response_to_responses_api_response.return_value = (
            MagicMock()
        )
        with unittest.mock.patch(
            "litellm.responses.litellm_completion_transformation.transformation.LiteLLMCompletionResponsesConfig",
            mock_config_cls,
        ):
            await host["_perform_codex_auto_agent_openrouter_completion_request"](
                request=mock_request,
                adapter_model="test-model",
                request_body=dict(original_body),
                use_alias_candidate_probe=True,
            )

        # The completion transform must have run exactly once.
        assert len(captured_transform_kwargs) == 1
        transform_responses_api_request = captured_transform_kwargs[0].get(
            "responses_api_request"
        )
        assert isinstance(transform_responses_api_request, dict)

        # Transform boundary sees the adapted flat tools, not namespace tools.
        transform_tools = transform_responses_api_request.get("tools")
        assert transform_tools == flat_tools
        tool_names = {
            tool.get("function", {}).get("name")
            for tool in transform_tools
            if isinstance(tool, dict)
        }
        assert tool_names == {"spawn_agent", "exec_command"}
        assert not any(
            "functions.collaboration" in str(tool) for tool in transform_tools
        )

        # Tool call/result IDs are preserved through the transform boundary.
        transform_input = captured_transform_kwargs[0].get("input")
        input_ids = {
            item.get("id")
            for item in transform_input
            if isinstance(item, dict)
        }
        assert {"call-abc-123", "result-def-456"} <= input_ids

        # Egress validation is reached and observes the OpenRouter target.
        assert len(egress_calls) == 1
        assert egress_calls[0]["url"].endswith("/v1/chat/completions")
        assert egress_calls[0]["credential_family"] == "openrouter"
