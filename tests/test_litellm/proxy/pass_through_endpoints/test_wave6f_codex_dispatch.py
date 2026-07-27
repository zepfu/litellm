"""Wave 6F: codex_dispatch extraction ownership + behavior tests.

Enforces the behavior-preserving extraction contract from
``llm_passthrough_endpoints.py`` into:

- ``aawm_adapter_runtime/codex_dispatch.py``
    The ``try_dispatch_codex_request`` gate that returns ``None`` for
    non-Codex/non-alias traffic and the dispatched ``Response`` for
    supported Codex/AAWM adapter paths.

Structural ownership tests verify AST presence, dependency isolation,
signature/async parity, and no god-module import.
"""

from __future__ import annotations

import ast
import asyncio
import inspect
from pathlib import Path
from typing import Any, Optional
from unittest.mock import MagicMock

import pytest
from fastapi import Request, Response

from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.pass_through_endpoints.aawm_adapter_runtime import codex_dispatch

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
MODULE_PATH = Path(codex_dispatch.__file__).resolve()
GOD_MODULE_PATH = (
    MODULE_PATH.parents[2] / "llm_passthrough_endpoints" / "llm_passthrough_endpoints.py"
)
# Fallback: locate god module via import
try:
    from litellm.proxy.pass_through_endpoints import llm_passthrough_endpoints as lpe

    GOD_MODULE_PATH = Path(lpe.__file__).resolve()
except Exception:
    pass

TARGET_IMPORT_PATH = (
    "litellm.proxy.pass_through_endpoints.aawm_adapter_runtime.codex_dispatch"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_request() -> MagicMock:
    req = MagicMock(spec=Request)
    req.method = "POST"
    return req


def _make_response() -> MagicMock:
    return MagicMock(spec=Response)


def _make_user_api_key_dict() -> MagicMock:
    return MagicMock(spec=UserAPIKeyAuth)


def _dispatch_kwargs(
    *,
    prepared_request_body: Optional[dict[str, Any]] = None,
    request_body: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    body = prepared_request_body if prepared_request_body is not None else {"model": "gpt-4o"}
    orig = request_body if request_body is not None else body
    return {
        "endpoint": "/v1/responses",
        "request": _make_request(),
        "request_body": orig,
        "prepared_request_body": body,
        "fastapi_response": _make_response(),
        "user_api_key_dict": _make_user_api_key_dict(),
        "target_url": "https://chatgpt.com/backend-api/codex/responses",
        "api_key": "sk-test",
        "forward_headers": False,
    }


# ===================================================================
# 1. Structural / ownership tests
# ===================================================================


class TestStructuralOwnership:
    """AST-level checks on the extracted module."""

    def test_module_file_exists(self) -> None:
        assert MODULE_PATH.is_file(), f"Missing {MODULE_PATH}"

    def test_public_function_defined(self) -> None:
        tree = ast.parse(MODULE_PATH.read_text())
        func_names = {
            node.name
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        assert "try_dispatch_codex_request" in func_names
        assert "install" in func_names

    def test_try_dispatch_is_async(self) -> None:
        tree = ast.parse(MODULE_PATH.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.AsyncFunctionDef) and node.name == "try_dispatch_codex_request":
                return
        pytest.fail("try_dispatch_codex_request must be an async def")

    def test_no_god_module_import_at_module_scope(self) -> None:
        """The module must NOT import llm_passthrough_endpoints at top level."""
        tree = ast.parse(MODULE_PATH.read_text())
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert "llm_passthrough_endpoints" not in alias.name, (
                        f"Module-scope import of god module: {alias.name}"
                    )
            elif isinstance(node, ast.ImportFrom):
                assert node.module is None or "llm_passthrough_endpoints" not in node.module, (
                    f"Module-scope from-import of god module: {node.module}"
                )

    def test_host_function_names_tuple(self) -> None:
        assert hasattr(codex_dispatch, "_HOST_FUNCTION_NAMES")
        assert "try_dispatch_codex_request" in codex_dispatch._HOST_FUNCTION_NAMES

    def test_install_callable(self) -> None:
        assert callable(codex_dispatch.install)

    def test_type_checking_stubs_present(self) -> None:
        """TYPE_CHECKING block should declare host-global function stubs."""
        source = MODULE_PATH.read_text()
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.If):
                # Look for `if TYPE_CHECKING:`
                test = node.test
                if isinstance(test, ast.Name) and test.id == "TYPE_CHECKING":
                    # Should contain function stubs
                    stub_names = set()
                    for child in ast.walk(node):
                        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            stub_names.add(child.name)
                    assert "_resolve_codex_auto_agent_alias_model" in stub_names
                    assert "_handle_codex_auto_agent_alias_route" in stub_names
                    return
        pytest.fail("No TYPE_CHECKING block found")


# ===================================================================
# 2. None fall-through tests
# ===================================================================


class TestNoneFallThrough:
    """When no adapter matches, the gate must return None."""

    @pytest.mark.asyncio
    async def test_no_adapters_returns_none(self) -> None:
        kwargs = _dispatch_kwargs(prepared_request_body={"model": "gpt-4o"})
        host = {
            "_resolve_codex_auto_agent_alias_model": lambda body, *, endpoint: None,
            "_resolve_codex_opencode_zen_adapter_model": lambda body, *, endpoint: None,
            "_resolve_codex_kimi_chat_completions_adapter_model": lambda body, *, endpoint: None,
            "_resolve_codex_alibaba_token_plan_adapter_model": lambda body, *, endpoint: None,
            "_resolve_codex_antigravity_code_assist_adapter_model": lambda body, *, endpoint: None,
            "_resolve_codex_google_code_assist_adapter_model": lambda body, *, endpoint: None,
            "_normalize_codex_reasoning_effort_for_resolved_route": lambda body, *, resolved_route: (body, None),
            "_ANTIGRAVITY_CODE_ASSIST_ADAPTER_PROVIDER": "antigravity",
        }
        codex_dispatch.install(host)
        result = await host["try_dispatch_codex_request"](**kwargs)
        assert result is None

    @pytest.mark.asyncio
    async def test_empty_model_returns_none(self) -> None:
        kwargs = _dispatch_kwargs(prepared_request_body={"model": ""})
        host = {
            "_resolve_codex_auto_agent_alias_model": lambda body, *, endpoint: None,
            "_resolve_codex_opencode_zen_adapter_model": lambda body, *, endpoint: None,
            "_resolve_codex_kimi_chat_completions_adapter_model": lambda body, *, endpoint: None,
            "_resolve_codex_alibaba_token_plan_adapter_model": lambda body, *, endpoint: None,
            "_resolve_codex_antigravity_code_assist_adapter_model": lambda body, *, endpoint: None,
            "_resolve_codex_google_code_assist_adapter_model": lambda body, *, endpoint: None,
            "_normalize_codex_reasoning_effort_for_resolved_route": lambda body, *, resolved_route: (body, None),
            "_ANTIGRAVITY_CODE_ASSIST_ADAPTER_PROVIDER": "antigravity",
        }
        codex_dispatch.install(host)
        result = await host["try_dispatch_codex_request"](**kwargs)
        assert result is None

    @pytest.mark.asyncio
    async def test_non_string_model_returns_none(self) -> None:
        kwargs = _dispatch_kwargs(prepared_request_body={"model": 42})
        host = {
            "_resolve_codex_auto_agent_alias_model": lambda body, *, endpoint: None,
            "_resolve_codex_opencode_zen_adapter_model": lambda body, *, endpoint: None,
            "_resolve_codex_kimi_chat_completions_adapter_model": lambda body, *, endpoint: None,
            "_resolve_codex_alibaba_token_plan_adapter_model": lambda body, *, endpoint: None,
            "_resolve_codex_antigravity_code_assist_adapter_model": lambda body, *, endpoint: None,
            "_resolve_codex_google_code_assist_adapter_model": lambda body, *, endpoint: None,
            "_normalize_codex_reasoning_effort_for_resolved_route": lambda body, *, resolved_route: (body, None),
            "_ANTIGRAVITY_CODE_ASSIST_ADAPTER_PROVIDER": "antigravity",
        }
        codex_dispatch.install(host)
        result = await host["try_dispatch_codex_request"](**kwargs)
        assert result is None

    @pytest.mark.asyncio
    async def test_direct_model_normalization_updates_body_before_fallthrough(
        self,
    ) -> None:
        body = {"model": "gpt-4o"}
        kwargs = _dispatch_kwargs(
            prepared_request_body=body,
            request_body=body,
        )
        host = {
            "_resolve_codex_auto_agent_alias_model": lambda body, *, endpoint: None,
            "_resolve_codex_opencode_zen_adapter_model": lambda body, *, endpoint: None,
            "_resolve_codex_kimi_chat_completions_adapter_model": lambda body, *, endpoint: None,
            "_resolve_codex_alibaba_token_plan_adapter_model": lambda body, *, endpoint: None,
            "_resolve_codex_antigravity_code_assist_adapter_model": lambda body, *, endpoint: None,
            "_resolve_codex_google_code_assist_adapter_model": lambda body, *, endpoint: None,
            "_normalize_codex_reasoning_effort_for_resolved_route": (
                lambda body, *, resolved_route: (
                    {**body, "reasoning": {"effort": "high"}},
                    {"normalized": True},
                )
            ),
            "_ANTIGRAVITY_CODE_ASSIST_ADAPTER_PROVIDER": "antigravity",
        }
        codex_dispatch.install(host)

        result = await host["try_dispatch_codex_request"](**kwargs)

        assert result is None
        assert body["reasoning"] == {"effort": "high"}


# ===================================================================
# 3. Supported dispatch tests (one per adapter lane)
# ===================================================================


def _make_dispatch_host(
    *,
    matching_adapter: str,
    adapter_model: str = "test-adapter-model",
) -> dict[str, Any]:
    """Build a host_globals dict where only *matching_adapter* resolves."""
    sentinel_response = Response(content=b"dispatched", status_code=200)

    resolvers: dict[str, Any] = {
        "_resolve_codex_auto_agent_alias_model": lambda body, *, endpoint: None,
        "_resolve_codex_opencode_zen_adapter_model": lambda body, *, endpoint: None,
        "_resolve_codex_kimi_chat_completions_adapter_model": lambda body, *, endpoint: None,
        "_resolve_codex_alibaba_token_plan_adapter_model": lambda body, *, endpoint: None,
        "_resolve_codex_antigravity_code_assist_adapter_model": lambda body, *, endpoint: None,
        "_resolve_codex_google_code_assist_adapter_model": lambda body, *, endpoint: None,
    }
    if matching_adapter in resolvers:
        resolvers[matching_adapter] = lambda body, *, endpoint: adapter_model

    async def _fake_alias_route(**kwargs: Any) -> Response:
        return sentinel_response

    async def _fake_adapter_route(**kwargs: Any) -> Response:
        return sentinel_response

    host: dict[str, Any] = {
        **resolvers,
        "_apply_codex_auto_agent_prevention_guidance_to_request_body": lambda body: (body, []),
        "_apply_aawm_read_agent_guidance_to_request_body": lambda body, *, alias_model, target_field: (body, []),
        "_prepare_request_body_for_passthrough_observability": lambda *, request, request_body: request_body,
        "_safe_set_request_parsed_body": lambda request, body: None,
        "_handle_codex_auto_agent_alias_route": _fake_alias_route,
        "_handle_codex_opencode_zen_adapter_route": _fake_adapter_route,
        "_handle_codex_kimi_chat_completions_adapter_route": _fake_adapter_route,
        "_handle_codex_alibaba_token_plan_adapter_route": _fake_adapter_route,
        "_handle_codex_google_code_assist_adapter_route": _fake_adapter_route,
        "_normalize_codex_reasoning_effort_for_resolved_route": lambda body, *, resolved_route: (body, None),
        "_ANTIGRAVITY_CODE_ASSIST_ADAPTER_PROVIDER": "antigravity",
    }
    return host, sentinel_response


class TestSupportedDispatch:
    """Each adapter lane should return the dispatched Response."""

    @pytest.mark.asyncio
    async def test_auto_agent_alias_dispatch(self) -> None:
        host, sentinel = _make_dispatch_host(
            matching_adapter="_resolve_codex_auto_agent_alias_model",
            adapter_model="codex-auto-agent",
        )
        codex_dispatch.install(host)
        result = await host["try_dispatch_codex_request"](**_dispatch_kwargs())
        assert result is sentinel

    @pytest.mark.asyncio
    async def test_opencode_zen_dispatch(self) -> None:
        host, sentinel = _make_dispatch_host(
            matching_adapter="_resolve_codex_opencode_zen_adapter_model",
        )
        codex_dispatch.install(host)
        result = await host["try_dispatch_codex_request"](**_dispatch_kwargs())
        assert result is sentinel

    @pytest.mark.asyncio
    async def test_kimi_code_dispatch(self) -> None:
        host, sentinel = _make_dispatch_host(
            matching_adapter="_resolve_codex_kimi_chat_completions_adapter_model",
        )
        codex_dispatch.install(host)
        result = await host["try_dispatch_codex_request"](**_dispatch_kwargs())
        assert result is sentinel

    @pytest.mark.asyncio
    async def test_alibaba_token_plan_dispatch(self) -> None:
        host, sentinel = _make_dispatch_host(
            matching_adapter="_resolve_codex_alibaba_token_plan_adapter_model",
        )
        codex_dispatch.install(host)
        result = await host["try_dispatch_codex_request"](**_dispatch_kwargs())
        assert result is sentinel

    @pytest.mark.asyncio
    async def test_antigravity_dispatch(self) -> None:
        host, sentinel = _make_dispatch_host(
            matching_adapter="_resolve_codex_antigravity_code_assist_adapter_model",
        )
        codex_dispatch.install(host)
        result = await host["try_dispatch_codex_request"](**_dispatch_kwargs())
        assert result is sentinel

    @pytest.mark.asyncio
    async def test_google_code_assist_dispatch(self) -> None:
        host, sentinel = _make_dispatch_host(
            matching_adapter="_resolve_codex_google_code_assist_adapter_model",
        )
        codex_dispatch.install(host)
        result = await host["try_dispatch_codex_request"](**_dispatch_kwargs())
        assert result is sentinel


# ===================================================================
# 4. Dispatch ordering / priority
# ===================================================================


class TestDispatchOrdering:
    """Earlier adapters in the cascade must win over later ones."""

    @pytest.mark.asyncio
    async def test_alias_wins_over_opencode_zen(self) -> None:
        alias_resp = Response(content=b"alias", status_code=200)
        zen_resp = Response(content=b"zen", status_code=200)

        async def _alias_route(**kw: Any) -> Response:
            return alias_resp

        async def _zen_route(**kw: Any) -> Response:
            return zen_resp

        host: dict[str, Any] = {
            "_resolve_codex_auto_agent_alias_model": lambda body, *, endpoint: "alias-model",
            "_resolve_codex_opencode_zen_adapter_model": lambda body, *, endpoint: "zen-model",
            "_resolve_codex_kimi_chat_completions_adapter_model": lambda body, *, endpoint: None,
            "_resolve_codex_alibaba_token_plan_adapter_model": lambda body, *, endpoint: None,
            "_resolve_codex_antigravity_code_assist_adapter_model": lambda body, *, endpoint: None,
            "_resolve_codex_google_code_assist_adapter_model": lambda body, *, endpoint: None,
            "_apply_codex_auto_agent_prevention_guidance_to_request_body": lambda body: (body, []),
            "_apply_aawm_read_agent_guidance_to_request_body": lambda body, *, alias_model, target_field: (body, []),
            "_prepare_request_body_for_passthrough_observability": lambda *, request, request_body: request_body,
            "_safe_set_request_parsed_body": lambda request, body: None,
            "_handle_codex_auto_agent_alias_route": _alias_route,
            "_handle_codex_opencode_zen_adapter_route": _zen_route,
            "_handle_codex_kimi_chat_completions_adapter_route": _zen_route,
            "_handle_codex_alibaba_token_plan_adapter_route": _zen_route,
            "_handle_codex_google_code_assist_adapter_route": _zen_route,
            "_normalize_codex_reasoning_effort_for_resolved_route": lambda body, *, resolved_route: (body, None),
            "_ANTIGRAVITY_CODE_ASSIST_ADAPTER_PROVIDER": "antigravity",
        }
        codex_dispatch.install(host)
        result = await host["try_dispatch_codex_request"](**_dispatch_kwargs())
        assert result is alias_resp


# ===================================================================
# 5. Error propagation
# ===================================================================


class TestErrorPropagation:
    """Exceptions from adapter handlers must propagate uncaught."""

    @pytest.mark.asyncio
    async def test_handler_exception_propagates(self) -> None:
        async def _exploding_route(**kw: Any) -> Response:
            raise RuntimeError("upstream failure")

        host: dict[str, Any] = {
            "_resolve_codex_auto_agent_alias_model": lambda body, *, endpoint: "alias",
            "_resolve_codex_opencode_zen_adapter_model": lambda body, *, endpoint: None,
            "_resolve_codex_kimi_chat_completions_adapter_model": lambda body, *, endpoint: None,
            "_resolve_codex_alibaba_token_plan_adapter_model": lambda body, *, endpoint: None,
            "_resolve_codex_antigravity_code_assist_adapter_model": lambda body, *, endpoint: None,
            "_resolve_codex_google_code_assist_adapter_model": lambda body, *, endpoint: None,
            "_apply_codex_auto_agent_prevention_guidance_to_request_body": lambda body: (body, []),
            "_apply_aawm_read_agent_guidance_to_request_body": lambda body, *, alias_model, target_field: (body, []),
            "_prepare_request_body_for_passthrough_observability": lambda *, request, request_body: request_body,
            "_safe_set_request_parsed_body": lambda request, body: None,
            "_handle_codex_auto_agent_alias_route": _exploding_route,
            "_handle_codex_opencode_zen_adapter_route": _exploding_route,
            "_handle_codex_kimi_chat_completions_adapter_route": _exploding_route,
            "_handle_codex_alibaba_token_plan_adapter_route": _exploding_route,
            "_handle_codex_google_code_assist_adapter_route": _exploding_route,
            "_normalize_codex_reasoning_effort_for_resolved_route": lambda body, *, resolved_route: (body, None),
            "_ANTIGRAVITY_CODE_ASSIST_ADAPTER_PROVIDER": "antigravity",
        }
        codex_dispatch.install(host)
        with pytest.raises(RuntimeError, match="upstream failure"):
            await host["try_dispatch_codex_request"](**_dispatch_kwargs())

    @pytest.mark.asyncio
    async def test_resolver_exception_propagates(self) -> None:
        def _exploding_resolver(body: Any, *, endpoint: str) -> None:
            raise ValueError("resolver boom")

        host: dict[str, Any] = {
            "_resolve_codex_auto_agent_alias_model": _exploding_resolver,
            "_resolve_codex_opencode_zen_adapter_model": lambda body, *, endpoint: None,
            "_resolve_codex_kimi_chat_completions_adapter_model": lambda body, *, endpoint: None,
            "_resolve_codex_alibaba_token_plan_adapter_model": lambda body, *, endpoint: None,
            "_resolve_codex_antigravity_code_assist_adapter_model": lambda body, *, endpoint: None,
            "_resolve_codex_google_code_assist_adapter_model": lambda body, *, endpoint: None,
            "_normalize_codex_reasoning_effort_for_resolved_route": lambda body, *, resolved_route: (body, None),
            "_ANTIGRAVITY_CODE_ASSIST_ADAPTER_PROVIDER": "antigravity",
        }
        codex_dispatch.install(host)
        with pytest.raises(ValueError, match="resolver boom"):
            await host["try_dispatch_codex_request"](**_dispatch_kwargs())


# ===================================================================
# 6. Signature / async parity
# ===================================================================


class TestSignatureParity:
    """The extracted function must be a coroutine with the expected kwargs."""

    def test_is_coroutine_function(self) -> None:
        assert asyncio.iscoroutinefunction(codex_dispatch.try_dispatch_codex_request)

    def test_keyword_only_params(self) -> None:
        sig = inspect.signature(codex_dispatch.try_dispatch_codex_request)
        expected = {
            "endpoint",
            "request",
            "request_body",
            "prepared_request_body",
            "fastapi_response",
            "user_api_key_dict",
            "target_url",
            "api_key",
            "forward_headers",
        }
        actual = set(sig.parameters.keys())
        assert actual == expected, f"Param mismatch: {actual ^ expected}"
        for name, param in sig.parameters.items():
            assert param.kind == inspect.Parameter.KEYWORD_ONLY, (
                f"{name} must be keyword-only"
            )

    def test_return_annotation_optional_response(self) -> None:
        sig = inspect.signature(codex_dispatch.try_dispatch_codex_request)
        ann = sig.return_annotation
        # Accept Optional[Response] or the string form
        ann_str = str(ann)
        assert "Response" in ann_str
        assert "None" in ann_str or "Optional" in ann_str


# ===================================================================
# 7. No god-module import (runtime check)
# ===================================================================


class TestNoGodImport:
    """The module must not hold a reference to the god module."""

    def test_no_god_module_in_module_dict(self) -> None:
        mod_dict = vars(codex_dispatch)
        for key, val in mod_dict.items():
            if hasattr(val, "__module__") and isinstance(val.__module__, str):
                assert "llm_passthrough_endpoints" not in val.__module__, (
                    f"{key} references god module via __module__"
                )

    def test_source_has_no_god_import(self) -> None:
        source = MODULE_PATH.read_text()
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert "llm_passthrough_endpoints" not in alias.name
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    assert "llm_passthrough_endpoints" not in node.module


# ===================================================================
# 8. install() rebinding
# ===================================================================


class TestInstall:
    """install() must publish try_dispatch_codex_request into host_globals."""

    def test_install_publishes_to_host(self) -> None:
        host: dict[str, Any] = {}
        codex_dispatch.install(host)
        assert "try_dispatch_codex_request" in host
        assert callable(host["try_dispatch_codex_request"])

    def test_install_rebinds_globals(self) -> None:
        host: dict[str, Any] = {"sentinel": True}
        codex_dispatch.install(host)
        fn = host["try_dispatch_codex_request"]
        # After rebinding, the function's __globals__ should be the host dict
        if hasattr(fn, "__globals__"):
            assert fn.__globals__ is host
