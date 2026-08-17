"""Wave 6F module-local tests for codex_candidate_calls.py.

Pins: symbol inventory, signatures/async parity, representative dispatch
success/failure/stream behavior, callback ordering, and absence of god import.
"""

from __future__ import annotations

import ast
import copy
import inspect
import json
import pathlib
from copy import deepcopy
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
        "_bind_responses_stream_timeout_terminalizer",
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
        # CFG-004 encrypted reasoning detection
        "_is_fernet_encrypted_token",
        "_responses_output_contains_encrypted_reasoning_arguments",
        "_FERNET_TOKEN_PREFIX",
        "_FERNET_MIN_TOKEN_LENGTH",
        "_ALIBABA_ENCRYPTED_REASONING_MAX_RETRIES",
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
            == 26
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
            # Constants are not callable
            if not name.startswith(("_OPENCODE_ZEN_DIRECT_", "_FERNET_", "_ALIBABA_ENCRYPTED_")):
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

    def _completion_handle_host(self, *, prepare_name: str) -> dict[str, Any]:
        host: dict[str, Any] = {"__builtins__": __builtins__}
        from fastapi.responses import Response, StreamingResponse

        host["StreamingResponse"] = StreamingResponse
        host["Response"] = Response

        async def mock_driver_run(*, prepare, perform, **kwargs):
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
        mock_plan = MagicMock()
        mock_plan.perform_kwargs = {"litellm_metadata": {}}
        mock_plan.prepared_request_body = {"litellm_metadata": {}}
        mock_plan.target_url = "https://example.test/v1"
        host[prepare_name] = AsyncMock(return_value=mock_plan)
        return host

    @staticmethod
    def _completion_plan(*, alias: str, upstream: str, effort: str) -> tuple[MagicMock, dict[str, Any]]:
        completion_kwargs = {
            "model": upstream,
            "messages": [],
            "reasoning_effort": effort,
        }
        mock_plan = MagicMock()
        mock_plan.perform_kwargs = {
            "litellm_metadata": {"model_alias": alias},
            "completion_kwargs": completion_kwargs,
        }
        mock_plan.prepared_request_body = {
            "model": alias,
            "litellm_metadata": {"model_alias": alias},
        }
        mock_plan.target_url = "https://example.test/v1"
        return mock_plan, completion_kwargs

    @pytest.mark.asyncio
    async def test_kimi_route_calls_validate_and_rollup(self):
        """handle_codex_kimi must validate/rollup and log provider-bound body."""
        host = self._completion_handle_host(
            prepare_name="_prepare_codex_kimi_chat_completions_adapter_route"
        )
        mock_plan, completion_kwargs = self._completion_plan(
            alias="kimi-test",
            upstream="kimi-upstream",
            effort="high",
        )
        host["_prepare_codex_kimi_chat_completions_adapter_route"] = AsyncMock(
            return_value=mock_plan
        )

        await host["_handle_codex_kimi_chat_completions_adapter_route"](
            endpoint="/v1/responses",
            request=MagicMock(headers={}),
            fastapi_response=MagicMock(),
            user_api_key_dict=MagicMock(),
            prepared_request_body={"model": "kimi-test"},
            adapter_model="kimi-test",
            use_alias_candidate_probe=False,
        )

        host["_validate_codex_auto_agent_responses_payload"].assert_awaited_once()
        host["_record_adapted_completed_route_rollup_turn"].assert_called_once()
        emit_kwargs = host["_emit_adapted_route_access_log"].call_args.kwargs
        assert emit_kwargs["request_body"] is mock_plan.prepared_request_body
        assert emit_kwargs["request_body"]["model"] == "kimi-test"
        assert emit_kwargs["provider_bound_body"] is completion_kwargs
        assert emit_kwargs["provider_bound_body"]["reasoning_effort"] == "high"

    @pytest.mark.asyncio
    async def test_alibaba_route_logs_provider_bound_body(self):
        """handle_codex_alibaba must log translated completion kwargs with alias label."""
        host = self._completion_handle_host(
            prepare_name="_prepare_codex_alibaba_token_plan_adapter_route"
        )
        mock_plan, completion_kwargs = self._completion_plan(
            alias="alibaba-alias",
            upstream="qwen-upstream",
            effort="low",
        )
        host["_prepare_codex_alibaba_token_plan_adapter_route"] = AsyncMock(
            return_value=mock_plan
        )

        await host["_handle_codex_alibaba_token_plan_adapter_route"](
            endpoint="/v1/responses",
            request=MagicMock(headers={}),
            fastapi_response=MagicMock(),
            user_api_key_dict=MagicMock(),
            prepared_request_body={"model": "alibaba-alias"},
            adapter_model="alibaba-alias",
            use_alias_candidate_probe=False,
        )

        host["_validate_codex_auto_agent_responses_payload"].assert_awaited_once()
        host["_record_adapted_completed_route_rollup_turn"].assert_called_once()
        emit_kwargs = host["_emit_adapted_route_access_log"].call_args.kwargs
        assert emit_kwargs["request_body"] is mock_plan.prepared_request_body
        assert emit_kwargs["request_body"]["model"] == "alibaba-alias"
        assert emit_kwargs["provider_bound_body"] is completion_kwargs
        assert emit_kwargs["provider_bound_body"]["reasoning_effort"] == "low"


class TestSotaXaiCandidateToolAdaptation:
    _MODEL = "oa_xai/grok-4.6"
    _COLLABORATION_NAMES = [
        "followup_task",
        "interrupt_agent",
        "list_agents",
        "send_message",
        "spawn_agent",
        "wait_agent",
    ]
    _ITEM_ID = "fc_685c42deefc0819a822b6936faaa30be0c76bc1491ab6619"
    _CALL_ID = "call_sota_xai_spawn"

    @classmethod
    def _request_body(cls, *, stream: bool) -> dict[str, Any]:
        collaboration_tools = []
        for name in cls._COLLABORATION_NAMES:
            collaboration_tools.append(
                {
                    "type": "function",
                    "name": name,
                    "description": (
                        "Only use spawn_agent if and only if the user explicitly "
                        "asks for sub-agents, delegation, or parallel agent work."
                        if name == "spawn_agent"
                        else f"{name} description"
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            (
                                "message" if name == "spawn_agent" else f"{name}_value"
                            ): {"type": "string"}
                        },
                    },
                }
            )
        return {
            "model": cls._MODEL,
            "stream": stream,
            "previous_response_id": "resp_sota_xai_continuation",
            "litellm_metadata": {"model_alias": "sota-xai"},
            "tools": [
                {
                    "type": "custom",
                    "name": "apply_patch",
                    "description": "Apply a patch.",
                    "format": {
                        "type": "grammar",
                        "syntax": "lark",
                        "definition": "start: /.+/",
                    },
                },
                {
                    "type": "namespace",
                    "name": "collaboration",
                    "tools": collaboration_tools,
                },
                {"type": "tool_search", "name": "tool_search"},
            ],
            "reasoning_effort": "high",
            "input": [
                {"type": "reasoning", "id": "rs_sota_xai_drop", "summary": []},
                {
                    "type": "function_call",
                    "id": "fc_continuation_input",
                    "call_id": "call_continuation_input",
                    "namespace": "collaboration",
                    "name": "send_message",
                    "arguments": '{"send_message_value":"continue"}',
                },
                {
                    "type": "function_call_output",
                    "id": "fco_continuation_input",
                    "call_id": "call_continuation_input",
                    "output": "continued",
                },
            ],
        }

    @pytest.mark.asyncio
    @pytest.mark.parametrize("stream", [False, True])
    async def test_candidate_flattens_patches_and_restores_canonical_ids(  # noqa: PLR0915
        self,
        stream: bool,
        monkeypatch: pytest.MonkeyPatch,
    ):
        from fastapi.responses import Response, StreamingResponse
        from litellm.llms.xai import oauth
        from litellm.proxy.pass_through_endpoints import (
            llm_passthrough_endpoints as lpe,
        )
        from litellm.proxy.pass_through_endpoints.aawm_adapter_runtime.openai_passthrough_handler import (
            BaseOpenAIPassThroughHandler,
            build_runtime_from_host,
            install_runtime,
        )

        events: list[str] = []
        stage_bodies: dict[str, dict[str, Any]] = {}
        prepare_input_bodies: list[dict[str, Any]] = []
        provider_bound_bodies: list[dict[str, Any]] = []
        validator_request_bodies: list[dict[str, Any]] = []
        intake_request_bodies: list[dict[str, Any]] = []
        request_body = self._request_body(stream=stream)
        canonical_snapshot = deepcopy(request_body)
        response_body = {
            "id": "resp_sota_xai_result",
            "status": "completed",
            "model": self._MODEL,
            "output": [
                {
                    "type": "function_call",
                    "id": self._ITEM_ID,
                    "call_id": self._CALL_ID,
                    "name": "spawn_agent",
                    "arguments": '{"message":"inspect"}',
                }
            ],
        }
        candidate_fn = lpe._perform_codex_auto_agent_oa_xai_responses_request
        assert candidate_fn.__globals__ is lpe.__dict__
        assert candidate_fn.__globals__["copy"] is copy
        assert "deepcopy" not in candidate_fn.__globals__

        install_runtime(build_runtime_from_host())
        original_prepare = (
            BaseOpenAIPassThroughHandler._prepare_openai_oa_xai_context
        )
        original_custom = (
            lpe._adapt_codex_custom_tools_to_functions_from_request_body
        )
        original_namespace = (
            lpe._adapt_codex_namespace_tools_to_functions_from_request_body
        )
        original_patch = (
            lpe._apply_codex_tool_description_patches_to_request_body
        )
        original_validate = lpe._validate_codex_auto_agent_responses_payload

        class _Handler:
            @staticmethod
            async def _prepare_openai_oa_xai_context(
                *,
                endpoint: str,
                request_body: dict[str, Any],
            ):
                assert endpoint == "/v1/responses"
                events.append("prepare")
                prepare_input_bodies.append(deepcopy(request_body))
                return await original_prepare(
                    endpoint=endpoint,
                    request_body=request_body,
                )

            @staticmethod
            def _assemble_headers(*, api_key: str, request: Any):
                assert api_key == "xai-token"
                return {"authorization": "Bearer xai-token"}

        async def _pass_through_request(**kwargs: Any):
            events.append("pass")
            provider_bound_bodies.append(deepcopy(kwargs["custom_body"]))
            if stream:
                return StreamingResponse(
                    lpe._responses_sse_from_repaired_response_body(response_body),
                    media_type="text/event-stream",
                )
            return Response(
                content=json.dumps(response_body),
                media_type="application/json",
            )

        async def _validate(response: Any, **kwargs: Any):
            events.append("validate")
            validator_request_bodies.append(deepcopy(kwargs["request_body"]))
            return await original_validate(response, **kwargs)

        def _intake_context(
            request: Any,
            body: dict[str, Any],
            **kwargs: Any,
        ) -> dict[str, Any]:
            events.append("intake")
            intake_request_bodies.append(deepcopy(body))
            return {}

        def _recording_step(name: str, callback: Any):
            def _apply(body: dict[str, Any]):
                events.append(name)
                result = callback(body)
                stage_bodies[name] = deepcopy(result[0])
                return result

            return _apply

        monkeypatch.setattr(
            lpe,
            "_adapt_codex_custom_tools_to_functions_from_request_body",
            _recording_step("custom", original_custom),
        )
        monkeypatch.setattr(
            lpe,
            "_adapt_codex_namespace_tools_to_functions_from_request_body",
            _recording_step("namespace", original_namespace),
        )
        monkeypatch.setattr(
            lpe,
            "_apply_codex_tool_description_patches_to_request_body",
            _recording_step("patch", original_patch),
        )
        monkeypatch.setattr(lpe, "BaseOpenAIPassThroughHandler", _Handler)
        monkeypatch.setattr(lpe, "pass_through_request", _pass_through_request)
        monkeypatch.setattr(
            lpe,
            "_validate_codex_auto_agent_responses_payload",
            _validate,
        )
        monkeypatch.setattr(
            lpe,
            "_build_malformed_tool_call_intake_context",
            _intake_context,
        )
        monkeypatch.setattr(
            oauth,
            "get_xai_oauth_access_token",
            AsyncMock(return_value="xai-token"),
        )

        result = await candidate_fn(
            endpoint="/v1/responses",
            request=MagicMock(headers={}),
            user_api_key_dict=MagicMock(),
            request_body=request_body,
        )

        assert events == [
            "custom",
            "namespace",
            "patch",
            "prepare",
            "pass",
            "intake",
            "validate",
        ]
        assert [tool.get("name") for tool in stage_bodies["namespace"]["tools"]] == [
            "apply_patch",
            *self._COLLABORATION_NAMES,
            "tool_search",
        ]
        assert all(
            tool["type"] == "function"
            for tool in stage_bodies["namespace"]["tools"][:-1]
        )
        assert stage_bodies["namespace"]["tools"][-1]["type"] == "tool_search"
        assert prepare_input_bodies == [stage_bodies["patch"]]

        provider_body = provider_bound_bodies[0]
        provider_tools = provider_body["tools"]
        assert [tool["name"] for tool in provider_tools] == [
            "apply_patch",
            *self._COLLABORATION_NAMES,
        ]
        assert all(tool["type"] == "function" for tool in provider_tools)
        collaboration_tools = {
            tool["name"]: tool for tool in provider_tools if tool["name"] != "apply_patch"
        }
        assert set(collaboration_tools) == set(self._COLLABORATION_NAMES)
        for name, tool in collaboration_tools.items():
            assert isinstance(tool["description"], str)
            assert isinstance(tool["parameters"], dict)
            if name != "spawn_agent":
                assert tool["description"] == f"{name} description"
                assert f"{name}_value" in tool["parameters"]["properties"]
        spawn_tool = collaboration_tools["spawn_agent"]
        assert "Use subagents to parallelize independent work" in (
            spawn_tool["description"]
        )
        assert "Only use spawn_agent if and only if" not in spawn_tool["description"]
        assert {
            "agent_type",
            "model",
            "fork_turns",
            "message",
        }.issubset(spawn_tool["parameters"]["properties"])
        assert "reasoning_effort" not in provider_body
        assert provider_body["model"] == "grok-4.6"
        assert provider_body["input"][0] == {
            "type": "function_call",
            "id": "fc_continuation_input",
            "call_id": "call_continuation_input",
            "name": "send_message",
            "arguments": '{"send_message_value":"continue"}',
        }
        assert provider_body["input"][1] == canonical_snapshot["input"][2]
        assert request_body == canonical_snapshot
        assert validator_request_bodies == [canonical_snapshot]
        assert intake_request_bodies == [canonical_snapshot]
        assert validator_request_bodies[0]["previous_response_id"] == (
            "resp_sota_xai_continuation"
        )

        if isinstance(result, StreamingResponse):
            chunks = [chunk async for chunk in result.body_iterator]
            rendered = "".join(
                chunk.decode() if isinstance(chunk, bytes) else str(chunk)
                for chunk in chunks
            )
            payloads = [
                json.loads(line.removeprefix("data: "))
                for line in rendered.splitlines()
                if line.startswith("data: {")
            ]
            restored_item = next(
                payload["item"]
                for payload in payloads
                if isinstance(payload.get("item"), dict)
                and payload["item"].get("type") == "function_call"
            )
        else:
            restored_item = json.loads(result.body)["output"][0]

        assert restored_item["namespace"] == "collaboration"
        assert restored_item["id"] == self._ITEM_ID
        assert restored_item["call_id"] == self._CALL_ID



# -- Fix 1 regression: namespace tool adaptation before chat-completion --


class TestOpenRouterCompletionNamespaceToolAdaptation:
    """_perform_codex_auto_agent_openrouter_completion_request must adapt
    namespace tools to flat dispatchable function tools before the chat
    completion transform, preserving tool call/result IDs."""

    @staticmethod
    def _passthrough_request_policy(body):
        return body, []

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
        host["_adapt_codex_custom_tools_to_functions_from_request_body"] = (
            self._passthrough_request_policy
        )
        host["_apply_codex_tool_description_patches_to_request_body"] = (
            self._passthrough_request_policy
        )
        host["_drop_unsupported_codex_hosted_tools_from_request_body"] = (
            self._passthrough_request_policy
        )
        host["_drop_unsupported_codex_input_items_from_request_body"] = (
            self._passthrough_request_policy
        )
        host["_drop_tool_choice_without_tools_from_request_body"] = (
            self._passthrough_request_policy
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
        completion_kwargs = {
            "model": "openrouter-upstream",
            "messages": [],
            "reasoning_effort": "high",
            "tools": flat_tools,
        }
        host["_apply_openrouter_completion_message_sanitization"] = MagicMock(
            side_effect=lambda **kw: (
                kw["request_body"],
                completion_kwargs,
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
        host["_build_malformed_tool_call_intake_context"] = MagicMock(return_value={})
        host["_validate_codex_auto_agent_responses_payload"] = AsyncMock(return_value=MagicMock())

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

        # D1-521: access log gets final provider-bound kwargs and alias/model label.
        emit_kwargs = host["_emit_adapted_route_access_log"].call_args.kwargs
        assert emit_kwargs["provider_bound_body"] is completion_kwargs
        assert emit_kwargs["provider_bound_body"]["reasoning_effort"] == "high"
        assert emit_kwargs["request_body"]["model"] == "test-model"

    @pytest.mark.asyncio
    @staticmethod
    def _mixed_openrouter_request_body() -> dict[str, Any]:
        empty_object = {"type": "object", "properties": {}}
        return {
            "model": "openrouter/cohere/north-mini-code:free",
            "input": [
                {"type": "reasoning", "id": "rs-drop", "summary": []},
                {
                    "type": "custom_tool_call",
                    "id": "ctc-drop",
                    "call_id": "ctc-drop",
                    "name": "apply_patch",
                    "input": "---",
                },
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
                    "type": "function",
                    "function": {"name": "Grep", "parameters": empty_object},
                },
                {
                    "type": "custom",
                    "name": "apply_patch",
                    "description": "Apply a patch.",
                },
                {
                    "type": "namespace",
                    "name": "functions.collaboration",
                    "tools": [
                        {
                            "type": "function",
                            "name": "spawn_agent",
                            "parameters": empty_object,
                        }
                    ],
                },
                {
                    "type": "namespace",
                    "name": "collaboration",
                    "tools": [
                        {
                            "type": "function",
                            "name": "spawn_agent",
                            "parameters": empty_object,
                        }
                    ],
                },
                {
                    "type": "namespace",
                    "name": "functions.exec",
                    "tools": [
                        {
                            "type": "function",
                            "name": "exec_command",
                            "parameters": empty_object,
                        }
                    ],
                },
                {
                    "type": "namespace",
                    "name": "exec",
                    "tools": [
                        {
                            "type": "function",
                            "name": "exec_command",
                            "parameters": empty_object,
                        }
                    ],
                },
                {"type": "custom", "name": "unsupported_custom"},
                {
                    "type": "namespace",
                    "name": "functions.unsupported",
                    "tools": [{"type": "function", "name": "unsupported_ns"}],
                },
                {"type": "tool_search", "name": "tool_search"},
                {"type": "web_search", "name": "web_search"},
                {"type": "image_generation", "name": "image_generation"},
                {"type": "computer_use", "name": "computer_use"},
            ],
        }

    @staticmethod
    def _install_real_openrouter_helpers(host: dict[str, Any]) -> None:
        from fastapi.responses import Response, StreamingResponse
        from litellm.proxy.pass_through_endpoints import (
            llm_passthrough_endpoints as lpe,
        )
        from litellm.proxy.pass_through_endpoints.aawm_adapter_runtime.openai_passthrough_handler import (
            build_runtime_from_host,
            install_runtime,
        )
        import httpx as _httpx
        import json as _json
        import litellm as _litellm
        from typing import cast as _cast

        install_runtime(build_runtime_from_host())
        host["Response"] = Response
        host["StreamingResponse"] = StreamingResponse
        for helper_name in (
            "_adapt_codex_custom_tools_to_functions_from_request_body",
            "_adapt_codex_namespace_tools_to_functions_from_request_body",
            "_apply_codex_tool_description_patches_to_request_body",
            "_drop_unsupported_codex_hosted_tools_from_request_body",
            "_drop_unsupported_codex_input_items_from_request_body",
            "_drop_tool_choice_without_tools_from_request_body",
        ):
            host[helper_name] = getattr(lpe, helper_name)
        host.update(
            {
                "_get_openrouter_api_key": MagicMock(return_value="sk-test"),
                "_get_openrouter_completion_adapter_upstream_model": MagicMock(
                    return_value=None
                ),
                "_get_openrouter_target_base": MagicMock(
                    return_value="https://openrouter.ai"
                ),
                "_build_openrouter_default_headers": MagicMock(return_value={}),
                "_get_proxy_shared_aiohttp_session": MagicMock(return_value=None),
                "_merge_litellm_metadata": MagicMock(
                    side_effect=lambda body, **kw: body
                ),
                "_add_route_family_logging_metadata": MagicMock(
                    side_effect=lambda body, family: body
                ),
                "_build_langfuse_span_descriptor": MagicMock(return_value={}),
                "_build_adapted_route_rollup_kwargs": MagicMock(return_value={}),
                "_emit_adapted_route_access_log": MagicMock(),
                "_annotate_request_scope_for_adapted_access_log": MagicMock(),
                "_record_adapted_completed_route_rollup_turn": MagicMock(),
                "_apply_openrouter_completion_message_sanitization": MagicMock(
                    side_effect=lambda **kw: (
                        kw["request_body"],
                        {
                            "model": "openrouter-upstream",
                            "messages": [],
                            "tools": kw["request_body"].get("tools"),
                        },
                        kw["litellm_metadata"],
                    )
                ),
                "_perform_openrouter_completion_adapter_operation": AsyncMock(
                    return_value=MagicMock()
                ),
                "_serialize_responses_adapter_response": MagicMock(
                    return_value='{"id":"resp-1","output":[],"status":"completed"}'
                ),
                "_is_codex_auto_agent_malformed_tool_call_text_output": MagicMock(
                    return_value=False
                ),
                "_is_codex_auto_agent_empty_success_responses_body": MagicMock(
                    return_value=False
                ),
                "_build_responses_response_from_adapter_response": MagicMock(
                    return_value=MagicMock()
                ),
                "_build_malformed_tool_call_intake_context": MagicMock(return_value={}),
                "_validate_codex_auto_agent_responses_payload": AsyncMock(
                    return_value=MagicMock()
                ),
                "httpx": _httpx,
                "litellm": _litellm,
                "cast": _cast,
                "ResponsesAPIOptionalRequestParams": dict,
                "json": _json,
            }
        )

        class _MockHelpers:
            @staticmethod
            def validate_outgoing_egress(**kwargs):
                return None

        host["HttpPassThroughEndpointHelpers"] = _MockHelpers
        codex_candidate_calls.install(host)

    @staticmethod
    def _tool_name(tool: dict[str, Any]) -> str | None:
        nested = tool.get("function")
        if isinstance(nested, dict) and isinstance(nested.get("name"), str):
            return nested["name"]
        name = tool.get("name")
        return name if isinstance(name, str) else None

    @staticmethod
    def _assert_shaped_openrouter_request(
        *,
        result: Any,
        host: dict[str, Any],
        request_body: dict[str, Any],
        canonical_snapshot: dict[str, Any],
        captured_transform_kwargs: list[dict[str, Any]],
    ) -> None:
        assert result is host["_validate_codex_auto_agent_responses_payload"].return_value
        assert len(captured_transform_kwargs) == 1
        provider_request = captured_transform_kwargs[0]["responses_api_request"]
        provider_tools = provider_request.get("tools") or []
        assert provider_tools
        assert all(
            isinstance(tool, dict) and tool.get("type") == "function"
            for tool in provider_tools
        )
        tool_names = {
            TestOpenRouterCompletionNamespaceToolAdaptation._tool_name(tool)
            for tool in provider_tools
            if isinstance(tool, dict)
        }
        assert {
            "Grep",
            "apply_patch",
            "spawn_agent",
            "exec_command",
        } <= tool_names
        assert tool_names.isdisjoint(
            {
                "unsupported_custom",
                "unsupported_ns",
                "tool_search",
                "web_search",
                "image_generation",
                "computer_use",
            }
        )
        provider_input = captured_transform_kwargs[0].get("input") or []
        assert all(
            not (
                isinstance(item, dict)
                and item.get("type") in {"reasoning", "custom", "custom_tool_call"}
            )
            for item in provider_input
        )
        assert {"call-abc-123", "result-def-456"} <= {
            item.get("id") for item in provider_input if isinstance(item, dict)
        }
        validate_kwargs = host[
            "_validate_codex_auto_agent_responses_payload"
        ].await_args.kwargs
        assert validate_kwargs["request_body"] is request_body
        assert request_body == canonical_snapshot
        host["_perform_openrouter_completion_adapter_operation"].assert_awaited_once()

    @pytest.mark.asyncio
    async def test_completion_transform_shapes_mixed_tools_without_raising(self):
        """Real catalog/policy helpers shape mixed Codex tools for OpenRouter."""
        host: dict[str, Any] = {"__builtins__": __builtins__}
        self._install_real_openrouter_helpers(host)
        request_body = self._mixed_openrouter_request_body()
        canonical_snapshot = deepcopy(request_body)
        captured_transform_kwargs: list[dict[str, Any]] = []

        def capture_transform(**kwargs):
            captured_transform_kwargs.append(kwargs)
            return {
                "model": kwargs.get("model"),
                "messages": [],
                "tools": kwargs.get("responses_api_request", {}).get("tools"),
            }

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
            result = await host[
                "_perform_codex_auto_agent_openrouter_completion_request"
            ](
                request=MagicMock(headers={"x-test": "1"}),
                adapter_model="openrouter/cohere/north-mini-code:free",
                request_body=request_body,
                use_alias_candidate_probe=True,
            )
        self._assert_shaped_openrouter_request(
            result=result,
            host=host,
            request_body=request_body,
            canonical_snapshot=canonical_snapshot,
            captured_transform_kwargs=captured_transform_kwargs,
        )


# -- CFG-004 regression: encrypted reasoning in tool call arguments --


class TestCFG004EncryptedReasoningDetection:
    """CFG-004: Fernet token detection in function_call arguments."""

    def test_fernet_token_detected(self):
        token = "gAAAAABn" + "A" * 200
        assert codex_candidate_calls._is_fernet_encrypted_token(token) is True

    def test_plaintext_not_detected(self):
        assert codex_candidate_calls._is_fernet_encrypted_token("implement the fix") is False
        assert codex_candidate_calls._is_fernet_encrypted_token("") is False
        # Too short to be a Fernet token
        assert codex_candidate_calls._is_fernet_encrypted_token("gAAAA") is False

    def test_detection_in_spawn_agent_message(self):
        """Reproduces the exact CFG-004 shape: spawn_agent.message is gAAAA."""
        encrypted = "gAAAAABn" + "B" * 200
        tool_call = MagicMock()
        tool_call.type = "function_call"
        tool_call.name = "spawn_agent"
        tool_call.call_id = "call-1"
        tool_call.arguments = json.dumps(
            {"message": encrypted, "agent_type": "opencode", "model": "basic"}
        )

        message_item = MagicMock()
        message_item.type = "message"

        response = MagicMock()
        response.output = [message_item, tool_call]

        findings = (
            codex_candidate_calls._responses_output_contains_encrypted_reasoning_arguments(
                response
            )
        )
        assert len(findings) == 1
        assert findings[0]["name"] == "spawn_agent"
        assert findings[0]["argument_key"] == "message"
        assert findings[0]["call_id"] == "call-1"

    def test_plaintext_arguments_not_flagged(self):
        tool_call = MagicMock()
        tool_call.type = "function_call"
        tool_call.name = "spawn_agent"
        tool_call.call_id = "call-2"
        tool_call.arguments = json.dumps(
            {"message": "implement the fix", "agent_type": "opencode"}
        )

        response = MagicMock()
        response.output = [tool_call]

        findings = (
            codex_candidate_calls._responses_output_contains_encrypted_reasoning_arguments(
                response
            )
        )
        assert findings == []


class TestCFG004AlibabaRetryPath:
    """CFG-004: bounded retry preserves Alibaba provider; fail closed after exhaustion."""

    def _make_host(self):
        import httpx as _httpx

        host: dict[str, Any] = {"__builtins__": __builtins__}
        host["json"] = json
        host["httpx"] = _httpx
        host["StreamingResponse"] = MagicMock()
        host["_responses_sse_from_iterator"] = MagicMock()
        host["_annotate_request_scope_for_adapted_access_log"] = MagicMock()
        host["_get_proxy_shared_aiohttp_session"] = MagicMock(return_value=None)
        host["_build_responses_response_from_adapter_response"] = MagicMock(
            return_value="BUILT_RESPONSE"
        )
        host["_serialize_responses_adapter_response"] = MagicMock(
            return_value='{"output":[]}'
        )
        host["_build_malformed_tool_call_intake_context"] = MagicMock(return_value={})
        return host

    def _make_encrypted_response(self):
        encrypted = "gAAAAABn" + "C" * 200
        resp = MagicMock()
        tool_call = MagicMock()
        tool_call.type = "function_call"
        tool_call.name = "spawn_agent"
        tool_call.call_id = "c1"
        tool_call.arguments = json.dumps(
            {"message": encrypted, "agent_type": "opencode"}
        )
        resp.output = [tool_call]
        return resp

    def _make_plaintext_response(self):
        resp = MagicMock()
        tool_call = MagicMock()
        tool_call.type = "function_call"
        tool_call.name = "spawn_agent"
        tool_call.call_id = "c2"
        tool_call.arguments = json.dumps(
            {"message": "implement the fix", "agent_type": "opencode"}
        )
        resp.output = [tool_call]
        return resp

    def _call_kwargs(self):
        return dict(
            config=MagicMock(),
            request=MagicMock(headers={}),
            prepared_request_body={"model": "qwen3.7-max"},
            adapter_model="qwen3.7-max",
            target_url="https://token-plan.example/v1/chat/completions",
            api_key="sk-test",
            api_base="https://token-plan.example/v1",
            client_requested_stream=False,
            completion_kwargs={"model": "qwen3.7-max", "messages": []},
            request_input="test input",
            responses_api_request={},
            litellm_metadata={},
            upstream_model="qwen3.7-max",
        )

    @pytest.mark.asyncio
    async def test_retry_recovers_plaintext_on_alibaba_provider(self):
        """First call returns encrypted spawn_agent.message; retry returns
        plaintext.  Response comes from the Alibaba adapter, not another
        provider."""
        host = self._make_host()
        host["_raise_codex_auto_agent_malformed_tool_call_text_payload"] = MagicMock(
            side_effect=AssertionError("should not raise")
        )

        mock_litellm = MagicMock()
        mock_litellm.acompletion = AsyncMock(
            side_effect=[MagicMock(), MagicMock()]
        )
        mock_litellm.LlmProviders.ALIBABA_TOKEN_PLAN.value = "alibaba_token_plan"
        host["litellm"] = mock_litellm

        codex_candidate_calls.install(host)

        mock_config = MagicMock()
        mock_config.transform_chat_completion_response_to_responses_api_response.side_effect = [
            self._make_encrypted_response(),
            self._make_plaintext_response(),
        ]

        with unittest.mock.patch(
            "litellm.responses.litellm_completion_transformation.transformation.LiteLLMCompletionResponsesConfig",
            mock_config,
        ):
            result = await host["_perform_codex_alibaba_token_plan_adapter_call"](
                **self._call_kwargs()
            )

        # Two acompletion calls on the same Alibaba provider: initial + retry
        assert mock_litellm.acompletion.await_count == 2
        # Transform called twice (once per attempt)
        assert (
            mock_config.transform_chat_completion_response_to_responses_api_response.call_count
            == 2
        )
        # Plaintext response built and returned by the Alibaba adapter
        assert result == "BUILT_RESPONSE"
        host["_raise_codex_auto_agent_malformed_tool_call_text_payload"].assert_not_called()

    @pytest.mark.asyncio
    async def test_fail_closed_after_retry_exhaustion_preserves_alibaba_attribution(self):
        """All attempts return encrypted content; must raise malformed-tool-call
        with Alibaba provider attribution, never dispatch encrypted content."""
        host = self._make_host()

        raised_kwargs: list[dict[str, Any]] = []

        def capture_raise(**kwargs):
            raised_kwargs.append(kwargs)
            raise RuntimeError("malformed_tool_call")

        host["_raise_codex_auto_agent_malformed_tool_call_text_payload"] = MagicMock(
            side_effect=capture_raise
        )

        mock_litellm = MagicMock()
        mock_litellm.acompletion = AsyncMock(
            side_effect=[MagicMock(), MagicMock()]
        )
        mock_litellm.LlmProviders.ALIBABA_TOKEN_PLAN.value = "alibaba_token_plan"
        host["litellm"] = mock_litellm

        codex_candidate_calls.install(host)

        mock_config = MagicMock()
        mock_config.transform_chat_completion_response_to_responses_api_response.return_value = (
            self._make_encrypted_response()
        )

        with unittest.mock.patch(
            "litellm.responses.litellm_completion_transformation.transformation.LiteLLMCompletionResponsesConfig",
            mock_config,
        ):
            with pytest.raises(RuntimeError, match="malformed_tool_call"):
                await host["_perform_codex_alibaba_token_plan_adapter_call"](
                    **self._call_kwargs()
                )

        # Both attempts used (initial + retry), same Alibaba provider
        assert mock_litellm.acompletion.await_count == 2
        # Fail-closed with Alibaba adapter attribution
        assert len(raised_kwargs) == 1
        assert raised_kwargs[0]["adapter"] == "codex_alibaba_token_plan_chat_completions_adapter"
        assert raised_kwargs[0]["adapter_label"] == "Alibaba Token Plan"



class TestCFG004AlibabaStreamingPath:
    """CFG-004 streaming: encrypted content detected before any bytes reach
    the client; same-provider retry; plaintext SSE on success; bounded
    fail-closed when encrypted content persists."""

    def _make_host(self):
        import httpx as _httpx

        host: dict[str, Any] = {"__builtins__": __builtins__}
        host["json"] = json
        host["httpx"] = _httpx
        host["StreamingResponse"] = MagicMock()
        host["_annotate_request_scope_for_adapted_access_log"] = MagicMock()
        host["_get_proxy_shared_aiohttp_session"] = MagicMock(return_value=None)
        host["_build_malformed_tool_call_intake_context"] = MagicMock(return_value={})
        # Capture what body the SSE emitter receives
        sse_bodies: list[dict[str, Any]] = []

        async def _sse_gen(body):
            yield "data: " + json.dumps(body) + chr(10) + chr(10)

        def mock_sse_from_body(body):
            # Capture at call time: StreamingResponse is mocked and never
            # iterates the generator, so record the body argument directly.
            sse_bodies.append(body)
            return _sse_gen(body)

        host["_responses_sse_from_repaired_response_body"] = mock_sse_from_body
        host["_sse_bodies"] = sse_bodies
        return host

    def _make_encrypted_response(self):
        encrypted = "gAAAAABn" + "D" * 200
        resp = MagicMock()
        tool_call = MagicMock()
        tool_call.type = "function_call"
        tool_call.name = "spawn_agent"
        tool_call.call_id = "c-stream-1"
        tool_call.arguments = json.dumps(
            {"message": encrypted, "agent_type": "opencode"}
        )
        resp.output = [tool_call]
        return resp

    def _make_plaintext_response(self):
        resp = MagicMock()
        tool_call = MagicMock()
        tool_call.type = "function_call"
        tool_call.name = "spawn_agent"
        tool_call.call_id = "c-stream-2"
        tool_call.arguments = json.dumps(
            {"message": "implement the fix", "agent_type": "opencode"}
        )
        resp.output = [tool_call]
        return resp

    def _call_kwargs(self, *, stream: bool = True):
        return dict(
            config=MagicMock(),
            request=MagicMock(headers={}),
            prepared_request_body={"model": "qwen3.7-max"},
            adapter_model="qwen3.7-max",
            target_url="https://token-plan.example/v1/chat/completions",
            api_key="sk-test",
            api_base="https://token-plan.example/v1",
            client_requested_stream=stream,
            completion_kwargs={"model": "qwen3.7-max", "messages": []},
            request_input="test input",
            responses_api_request={},
            litellm_metadata={},
            upstream_model="qwen3.7-max",
        )

    @pytest.mark.asyncio
    async def test_streaming_plaintext_first_attempt_returns_sse(self):
        """Plaintext on first streaming attempt: SSE emitted immediately,
        no retry, no encrypted content."""
        host = self._make_host()
        host["_serialize_responses_adapter_response"] = MagicMock(
            return_value=json.dumps(
                {"output": [{"type": "function_call", "name": "spawn_agent",
                  "arguments": json.dumps({"message": "implement the fix"})}],
                 "status": "completed"}
            )
        )
        host["_raise_codex_auto_agent_malformed_tool_call_text_payload"] = MagicMock(
            side_effect=AssertionError("should not raise")
        )

        mock_litellm = MagicMock()
        mock_litellm.acompletion = AsyncMock(return_value=MagicMock())
        mock_litellm.LlmProviders.ALIBABA_TOKEN_PLAN.value = "alibaba_token_plan"
        host["litellm"] = mock_litellm

        codex_candidate_calls.install(host)

        mock_config = MagicMock()
        mock_config.transform_chat_completion_response_to_responses_api_response.return_value = (
            self._make_plaintext_response()
        )

        with unittest.mock.patch(
            "litellm.responses.litellm_completion_transformation.transformation.LiteLLMCompletionResponsesConfig",
            mock_config,
        ):
            await host["_perform_codex_alibaba_token_plan_adapter_call"](
                **self._call_kwargs(stream=True)
            )

        # Single upstream call (no retry needed)
        assert mock_litellm.acompletion.await_count == 1
        # SSE emitter received the plaintext body
        assert len(host["_sse_bodies"]) == 1
        body = host["_sse_bodies"][0]
        assert "gAAAA" not in json.dumps(body)
        # StreamingResponse was constructed
        host["StreamingResponse"].assert_called_once()

    @pytest.mark.asyncio
    async def test_streaming_encrypted_retry_recovers_plaintext(self):
        """First streaming attempt returns encrypted spawn_agent.message;
        retry returns plaintext.  SSE emitted only after plaintext confirmed.
        Same Alibaba provider used for both attempts."""
        host = self._make_host()
        host["_serialize_responses_adapter_response"] = MagicMock(
            return_value=json.dumps(
                {"output": [{"type": "function_call", "name": "spawn_agent",
                  "arguments": json.dumps({"message": "implement the fix"})}],
                 "status": "completed"}
            )
        )
        host["_raise_codex_auto_agent_malformed_tool_call_text_payload"] = MagicMock(
            side_effect=AssertionError("should not raise")
        )

        mock_litellm = MagicMock()
        mock_litellm.acompletion = AsyncMock(
            side_effect=[MagicMock(), MagicMock()]
        )
        mock_litellm.LlmProviders.ALIBABA_TOKEN_PLAN.value = "alibaba_token_plan"
        host["litellm"] = mock_litellm

        codex_candidate_calls.install(host)

        mock_config = MagicMock()
        mock_config.transform_chat_completion_response_to_responses_api_response.side_effect = [
            self._make_encrypted_response(),
            self._make_plaintext_response(),
        ]

        with unittest.mock.patch(
            "litellm.responses.litellm_completion_transformation.transformation.LiteLLMCompletionResponsesConfig",
            mock_config,
        ):
            await host["_perform_codex_alibaba_token_plan_adapter_call"](
                **self._call_kwargs(stream=True)
            )

        # Two upstream calls: initial + retry, same Alibaba provider
        assert mock_litellm.acompletion.await_count == 2
        # Transform called twice (once per attempt)
        assert (
            mock_config.transform_chat_completion_response_to_responses_api_response.call_count
            == 2
        )
        # SSE emitter received exactly one body (the plaintext one)
        assert len(host["_sse_bodies"]) == 1
        body = host["_sse_bodies"][0]
        assert "gAAAA" not in json.dumps(body)
        host["_raise_codex_auto_agent_malformed_tool_call_text_payload"].assert_not_called()

    @pytest.mark.asyncio
    async def test_streaming_encrypted_persists_fails_closed_no_bytes_dispatched(self):
        """All streaming attempts return encrypted content.  Must raise
        malformed-tool-call with Alibaba attribution.  No SSE bytes dispatched."""
        host = self._make_host()
        host["_serialize_responses_adapter_response"] = MagicMock(
            return_value=json.dumps(
                {"output": [{"type": "function_call", "name": "spawn_agent",
                  "arguments": json.dumps({"message": "gAAAAABn" + "E" * 200})}],
                 "status": "completed"}
            )
        )

        raised_kwargs: list[dict[str, Any]] = []

        def capture_raise(**kwargs):
            raised_kwargs.append(kwargs)
            raise RuntimeError("malformed_tool_call")

        host["_raise_codex_auto_agent_malformed_tool_call_text_payload"] = MagicMock(
            side_effect=capture_raise
        )

        mock_litellm = MagicMock()
        mock_litellm.acompletion = AsyncMock(
            side_effect=[MagicMock(), MagicMock()]
        )
        mock_litellm.LlmProviders.ALIBABA_TOKEN_PLAN.value = "alibaba_token_plan"
        host["litellm"] = mock_litellm

        codex_candidate_calls.install(host)

        mock_config = MagicMock()
        mock_config.transform_chat_completion_response_to_responses_api_response.return_value = (
            self._make_encrypted_response()
        )

        with unittest.mock.patch(
            "litellm.responses.litellm_completion_transformation.transformation.LiteLLMCompletionResponsesConfig",
            mock_config,
        ):
            with pytest.raises(RuntimeError, match="malformed_tool_call"):
                await host["_perform_codex_alibaba_token_plan_adapter_call"](
                    **self._call_kwargs(stream=True)
                )

        # Both attempts used (initial + retry), same Alibaba provider
        assert mock_litellm.acompletion.await_count == 2
        # No SSE bytes dispatched
        assert len(host["_sse_bodies"]) == 0
        # Fail-closed with Alibaba adapter attribution
        assert len(raised_kwargs) == 1
        assert raised_kwargs[0]["adapter"] == "codex_alibaba_token_plan_chat_completions_adapter"
        assert raised_kwargs[0]["adapter_label"] == "Alibaba Token Plan"

    @pytest.mark.asyncio
    async def test_streaming_no_encrypted_bytes_in_sse_output(self):
        """End-to-end: the actual SSE bytes yielded to the client contain
        no Fernet token prefix, proving ciphertext never reaches the wire."""
        host = self._make_host()
        plaintext_body = {
            "output": [
                {
                    "type": "function_call",
                    "name": "spawn_agent",
                    "call_id": "c-stream-3",
                    "arguments": json.dumps({"message": "implement the fix"}),
                }
            ],
            "status": "completed",
        }
        host["_serialize_responses_adapter_response"] = MagicMock(
            return_value=json.dumps(plaintext_body)
        )
        host["_raise_codex_auto_agent_malformed_tool_call_text_payload"] = MagicMock(
            side_effect=AssertionError("should not raise")
        )

        mock_litellm = MagicMock()
        mock_litellm.acompletion = AsyncMock(return_value=MagicMock())
        mock_litellm.LlmProviders.ALIBABA_TOKEN_PLAN.value = "alibaba_token_plan"
        host["litellm"] = mock_litellm

        codex_candidate_calls.install(host)

        mock_config = MagicMock()
        mock_config.transform_chat_completion_response_to_responses_api_response.return_value = (
            self._make_plaintext_response()
        )

        with unittest.mock.patch(
            "litellm.responses.litellm_completion_transformation.transformation.LiteLLMCompletionResponsesConfig",
            mock_config,
        ):
            await host["_perform_codex_alibaba_token_plan_adapter_call"](
                **self._call_kwargs(stream=True)
            )

        # Collect all SSE bytes from the mock emitter
        all_sse_text = ""
        for body in host["_sse_bodies"]:
            all_sse_text += json.dumps(body)
        assert "gAAAA" not in all_sse_text
        assert "implement the fix" in all_sse_text


# ---------------------------------------------------------------------------
# D1-521: OpenCode logs final provider_bound_body (no shared handle fixture)
# ---------------------------------------------------------------------------


class TestD1521OpenCodeProviderBoundBody:
    """OpenCode handle path logs translated completion kwargs with alias label."""

    @pytest.mark.asyncio
    async def test_opencode_route_logs_completion_kwargs_as_provider_bound_body(self):
        from fastapi.responses import Response, StreamingResponse

        completion_kwargs = {
            "model": "opencode-upstream",
            "messages": [{"role": "user", "content": "hi"}],
            "reasoning_effort": "medium",
        }
        request_body = {
            "model": "opencode-alias",
            "litellm_metadata": {"model_alias": "opencode-alias"},
        }
        normalized = MagicMock(
            request_body=request_body,
            request_input="hi",
            responses_api_request={},
            litellm_metadata=dict(request_body["litellm_metadata"]),
            completion_kwargs=completion_kwargs,
        )

        host: dict[str, Any] = {
            "__builtins__": __builtins__,
            "json": json,
            "httpx": __import__("httpx"),
            "Response": Response,
            "StreamingResponse": StreamingResponse,
            "cast": lambda typ, val: val,
            "ResponsesAPIOptionalRequestParams": dict,
            "_annotate_request_scope_for_adapted_access_log": MagicMock(),
            "_emit_adapted_route_access_log": MagicMock(),
            "_build_adapted_route_rollup_kwargs": MagicMock(return_value={}),
            "_consume_opencode_zen_tools_mode_header": MagicMock(
                side_effect=lambda request, body, probe: body
            ),
            "_prepare_opencode_zen_direct_observability_metadata": MagicMock(
                return_value=(request_body, None)
            ),
            "_get_anthropic_opencode_zen_normalization_runtime": MagicMock(return_value=object()),
            "_anthropic_opencode_zen_normalization": MagicMock(
                normalize_codex_request=AsyncMock(return_value=normalized)
            ),
            "_get_opencode_zen_target_base": MagicMock(return_value="https://opencode.example"),
            "_join_opencode_zen_passthrough_url": MagicMock(
                return_value="https://opencode.example/v1/chat/completions"
            ),
            "_load_opencode_zen_api_key_for_candidate": AsyncMock(return_value="sk-test"),
            "BaseOpenAIPassThroughHandler": MagicMock(
                _assemble_headers=MagicMock(return_value={"Authorization": "Bearer sk-test"})
            ),
            "HttpPassThroughEndpointHelpers": MagicMock(
                validate_outgoing_egress=MagicMock()
            ),
            "_build_opencode_zen_completion_call_kwargs": MagicMock(
                side_effect=lambda **kwargs: dict(kwargs["completion_kwargs"])
            ),
            "_perform_opencode_zen_completion_call": AsyncMock(return_value=MagicMock()),
            "_build_responses_response_from_adapter_response": MagicMock(return_value=MagicMock()),
            "_serialize_responses_adapter_response": MagicMock(return_value="{}"),
            "_is_codex_auto_agent_malformed_tool_call_text_output": MagicMock(return_value=False),
            "_is_codex_auto_agent_empty_success_responses_body": MagicMock(return_value=False),
            "_validate_codex_auto_agent_responses_payload": AsyncMock(return_value=MagicMock()),
            "_build_malformed_tool_call_intake_context": MagicMock(return_value={}),
            "_record_adapted_completed_route_rollup_turn": MagicMock(),
            "_record_adapted_completed_route_rollup_after_stream": MagicMock(),
        }
        import litellm as _litellm

        host["litellm"] = _litellm
        codex_candidate_calls.install(host)
        # Keep branch-specific mocks after install rebind.
        host["_emit_adapted_route_access_log"] = MagicMock()
        host["_annotate_request_scope_for_adapted_access_log"] = MagicMock()
        host["_build_adapted_route_rollup_kwargs"] = MagicMock(return_value={})
        host["_build_opencode_zen_completion_call_kwargs"] = MagicMock(
            side_effect=lambda **kwargs: dict(kwargs["completion_kwargs"])
        )
        host["_perform_opencode_zen_completion_call"] = AsyncMock(return_value=MagicMock())
        host["_consume_opencode_zen_tools_mode_header"] = MagicMock(
            side_effect=lambda request, body, probe: body
        )
        host["_prepare_opencode_zen_direct_observability_metadata"] = MagicMock(
            return_value=(request_body, None)
        )
        host["_get_anthropic_opencode_zen_normalization_runtime"] = MagicMock(return_value=object())
        host["_anthropic_opencode_zen_normalization"] = MagicMock(
            normalize_codex_request=AsyncMock(return_value=normalized)
        )
        host["_get_opencode_zen_target_base"] = MagicMock(return_value="https://opencode.example")
        host["_join_opencode_zen_passthrough_url"] = MagicMock(
            return_value="https://opencode.example/v1/chat/completions"
        )
        host["_load_opencode_zen_api_key_for_candidate"] = AsyncMock(return_value="sk-test")

        mock_config = MagicMock()
        mock_config.transform_chat_completion_response_to_responses_api_response.return_value = (
            MagicMock()
        )
        with unittest.mock.patch(
            "litellm.responses.litellm_completion_transformation.transformation.LiteLLMCompletionResponsesConfig",
            mock_config,
        ), unittest.mock.patch(
            "litellm.llms.anthropic.experimental_pass_through.providers.opencode_zen.constants._OPENCODE_ZEN_FREE_MODELS",
            set(),
        ):
            await host["_handle_codex_opencode_zen_adapter_route"](
                endpoint="/v1/responses",
                request=MagicMock(headers={}),
                fastapi_response=MagicMock(),
                user_api_key_dict=MagicMock(),
                prepared_request_body=dict(request_body),
                adapter_model="opencode-alias",
                use_alias_candidate_probe=False,
            )

        emit_kwargs = host["_emit_adapted_route_access_log"].call_args.kwargs
        assert emit_kwargs["provider_bound_body"] is completion_kwargs
        assert emit_kwargs["provider_bound_body"]["reasoning_effort"] == "medium"
        assert emit_kwargs["request_body"] is request_body
        assert emit_kwargs["request_body"]["model"] == "opencode-alias"
