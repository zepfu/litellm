"""Wave 7 owner tests: openai_passthrough_handler module.

Focused module-local tests for the extracted
``BaseOpenAIPassThroughHandler`` body now owned by
``aawm_adapter_runtime/openai_passthrough_handler.py``.

Covers:
- OpenAIPassThroughHandlerRuntime frozen-dataclass contract
- Pure utility methods (_join_url_paths, _normalize_endpoint_for_target)
- _assemble_headers / _append_openai_beta_header DI delegation
- _prepare_openai_oa_xai_context parity (success, None, fail-closed)
- _prepare_openai_grok_native_oauth_context parity (success, None, fail-closed)
- _base_openai_pass_through_handler GET/POST dispatch parity
- Unconfigured-runtime RuntimeError
- No module-scope god-module import (AST structural pin)
- build_runtime_from_host fail-closed on missing host attributes

Write-only surface: this file. No production edits.
"""

from __future__ import annotations

import ast
import asyncio
import dataclasses
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from types import ModuleType
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

import litellm
from litellm.proxy.auth.route_checks import RouteChecks
from litellm.proxy import pass_through_endpoints as pass_through_package
from litellm.proxy.pass_through_endpoints.aawm_alias_routing.request_metadata import (
    _aresolve_auto_agent_alias_route_host_attribution,
)

from litellm.proxy.pass_through_endpoints.aawm_adapter_runtime.openai_passthrough_handler import (
    BaseOpenAIPassThroughHandler,
    OpenAIPassThroughHandlerRuntime,
    build_runtime_from_host,
    install_runtime,
)

MODULE_PATH = Path(
    __import__(
        "litellm.proxy.pass_through_endpoints.aawm_adapter_runtime.openai_passthrough_handler",
        fromlist=["__file__"],
    ).__file__
).resolve()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_runtime(**overrides: Any) -> OpenAIPassThroughHandlerRuntime:
    """Build a runtime with no-op defaults; override individual seams."""
    defaults: dict[str, Any] = dict(
        prepare_oa_xai_passthrough_request_fn=AsyncMock(
            return_value=(False, None, None)
        ),
        is_openai_responses_endpoint_fn=lambda ep: "/responses" in ep,
        to_xai_native_passthrough_model_fn=lambda m: m,
        get_openai_passthrough_route_family_fn=lambda ep: "responses",
        merge_litellm_metadata_fn=lambda body, **kw: body,
        prepare_grok_native_oauth_passthrough_request_fn=AsyncMock(
            return_value=(False, None, {}, {})
        ),
        join_grok_passthrough_url_fn=lambda **kw: "https://grok.test/v1/responses",
        request_uses_codex_native_auth_fn=lambda req: False,
        resolve_codex_auto_agent_alias_model_fn=lambda body, **kw: None,
        resolve_auto_agent_alias_route_host_attribution_fn=AsyncMock(
            return_value={
                "client_ip": None,
                "client_ip_source": None,
                "host_name": None,
                "host_name_source": None,
            }
        ),
        add_route_family_logging_metadata_fn=lambda body, fam: body,
        apply_codex_tool_description_patches_fn=lambda body: (body, []),
        drop_unsupported_codex_hosted_tools_fn=lambda body: (body, []),
        drop_unsupported_codex_request_params_fn=lambda body: (body, []),
        drop_unsupported_codex_input_items_fn=lambda body: (body, []),
        is_oa_xai_request_body_fn=lambda body: False,
        is_grok_native_oauth_request_body_fn=lambda body: False,
        drop_tool_choice_without_tools_fn=lambda body: (body, []),
        add_codex_request_breakout_logging_metadata_fn=lambda body: body,
        prepare_request_body_for_passthrough_observability_fn=lambda **kw: kw.get(
            "request_body", {}
        ),
        safe_set_request_parsed_body_fn=MagicMock(),
        get_request_body_fn=AsyncMock(return_value={}),
        create_pass_through_route_fn=MagicMock(),
        try_dispatch_codex_request_fn=AsyncMock(return_value=None),
        is_assistants_api_request_fn=lambda req: False,
    )
    defaults.update(overrides)
    return OpenAIPassThroughHandlerRuntime(**defaults)


def _fake_request(method: str = "GET") -> MagicMock:
    req = MagicMock()
    req.method = method
    req.url = httpx.URL("http://localhost:4000/v1/responses")
    req.headers = {}
    return req


def _synthetic_host() -> ModuleType:
    """Return a complete host publication without importing the god module."""
    host = ModuleType(
        "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints"
    )
    callbacks: dict[str, Any] = {
        "_prepare_oa_xai_passthrough_request": AsyncMock(
            return_value=(False, None, None)
        ),
        "_is_openai_responses_endpoint": lambda ep: "/responses" in ep,
        "_to_xai_native_passthrough_model": lambda model: model,
        "_get_openai_passthrough_route_family": lambda ep: "responses",
        "_merge_litellm_metadata": lambda body, **kwargs: body,
        "_prepare_grok_native_oauth_passthrough_request": AsyncMock(
            return_value=(False, None, {}, {})
        ),
        "_join_grok_passthrough_url": (
            lambda **kwargs: "https://grok.test/v1/responses"
        ),
        "_request_uses_codex_native_auth": lambda request: False,
        "_resolve_codex_auto_agent_alias_model": (
            lambda body, **kwargs: None
        ),
        "_aresolve_auto_agent_alias_route_host_attribution": AsyncMock(
            return_value={
                "client_ip": None,
                "client_ip_source": None,
                "host_name": None,
                "host_name_source": None,
            }
        ),
        "_add_route_family_logging_metadata": lambda body, family: body,
        "_apply_codex_tool_description_patches_to_request_body": (
            lambda body: (body, [])
        ),
        "_drop_unsupported_codex_hosted_tools_from_request_body": (
            lambda body: (body, [])
        ),
        "_drop_unsupported_codex_request_params_from_request_body": (
            lambda body: (body, [])
        ),
        "_drop_unsupported_codex_input_items_from_request_body": (
            lambda body: (body, [])
        ),
        "_is_oa_xai_request_body": lambda body: False,
        "_is_grok_native_oauth_request_body": lambda body: False,
        "_drop_tool_choice_without_tools_from_request_body": (
            lambda body: (body, [])
        ),
        "_add_codex_request_breakout_logging_metadata": lambda body: body,
        "_prepare_request_body_for_passthrough_observability": (
            lambda **kwargs: kwargs["request_body"]
        ),
        "_safe_set_request_parsed_body": MagicMock(),
        "get_request_body": AsyncMock(return_value={}),
        "create_pass_through_route": MagicMock(
            return_value=AsyncMock(return_value="synthetic-response")
        ),
        "try_dispatch_codex_request": AsyncMock(return_value=None),
    }
    for name, callback in callbacks.items():
        setattr(host, name, callback)
    return host


@contextmanager
def _synthetic_host_publication(host: ModuleType):
    """Publish a synthetic god module through both import resolution paths."""
    module_name = (
        "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints"
    )
    with (
        patch.dict(sys.modules, {module_name: host}),
        patch.object(
            pass_through_package,
            "llm_passthrough_endpoints",
            host,
            create=True,
        ),
    ):
        yield


@pytest.fixture(autouse=True)
def _clean_runtime():
    """Ensure runtime is cleared before and after each test."""
    BaseOpenAIPassThroughHandler._runtime = None
    yield
    BaseOpenAIPassThroughHandler._runtime = None


# ---------------------------------------------------------------------------
# Frozen dataclass contract
# ---------------------------------------------------------------------------


class TestRuntimeDataclass:
    def test_is_frozen_dataclass(self):
        assert dataclasses.is_dataclass(OpenAIPassThroughHandlerRuntime)
        params = OpenAIPassThroughHandlerRuntime.__dataclass_params__  # type: ignore[attr-defined]
        assert params.frozen is True

    def test_expected_field_names(self):
        names = {
            f.name for f in dataclasses.fields(OpenAIPassThroughHandlerRuntime)
        }
        expected = {
            "prepare_oa_xai_passthrough_request_fn",
            "is_openai_responses_endpoint_fn",
            "to_xai_native_passthrough_model_fn",
            "get_openai_passthrough_route_family_fn",
            "merge_litellm_metadata_fn",
            "prepare_grok_native_oauth_passthrough_request_fn",
            "join_grok_passthrough_url_fn",
            "request_uses_codex_native_auth_fn",
            "resolve_codex_auto_agent_alias_model_fn",
            "resolve_auto_agent_alias_route_host_attribution_fn",
            "add_route_family_logging_metadata_fn",
            "apply_codex_tool_description_patches_fn",
            "drop_unsupported_codex_hosted_tools_fn",
            "drop_unsupported_codex_request_params_fn",
            "drop_unsupported_codex_input_items_fn",
            "is_oa_xai_request_body_fn",
            "is_grok_native_oauth_request_body_fn",
            "drop_tool_choice_without_tools_fn",
            "add_codex_request_breakout_logging_metadata_fn",
            "prepare_request_body_for_passthrough_observability_fn",
            "safe_set_request_parsed_body_fn",
            "get_request_body_fn",
            "create_pass_through_route_fn",
            "try_dispatch_codex_request_fn",
            "is_assistants_api_request_fn",
        }
        assert names == expected

    def test_immutable(self):
        rt = _make_runtime()
        with pytest.raises(dataclasses.FrozenInstanceError):
            rt.is_openai_responses_endpoint_fn = lambda ep: True  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Unconfigured runtime
# ---------------------------------------------------------------------------


class TestUnconfiguredRuntime:
    def test_get_runtime_raises_when_unconfigured(self):
        with pytest.raises(RuntimeError, match="not configured"):
            BaseOpenAIPassThroughHandler._get_runtime()

    def test_assemble_headers_raises_when_unconfigured(self):
        with pytest.raises(RuntimeError, match="not configured"):
            BaseOpenAIPassThroughHandler._assemble_headers(
                api_key="sk-test", request=_fake_request()
            )

    def test_base_handler_raises_when_unconfigured(self):
        with pytest.raises(RuntimeError, match="not configured"):
            asyncio.run(
                BaseOpenAIPassThroughHandler._base_openai_pass_through_handler(
                    endpoint="/v1/chat/completions",
                    request=_fake_request(),
                    fastapi_response=MagicMock(),
                    user_api_key_dict=MagicMock(),
                    base_target_url="https://api.openai.com",
                    api_key="sk-test",
                    custom_llm_provider=litellm.LlmProviders.OPENAI,
                )
            )


# ---------------------------------------------------------------------------
# Pure utility methods (no DI)
# ---------------------------------------------------------------------------


class TestJoinUrlPaths:
    def test_no_base_path(self):
        result = BaseOpenAIPassThroughHandler._join_url_paths(
            httpx.URL("https://api.openai.com"),
            "/v1/chat/completions",
            litellm.LlmProviders.OPENAI.value,
        )
        assert "api.openai.com/v1/chat/completions" in result

    def test_with_base_path(self):
        result = BaseOpenAIPassThroughHandler._join_url_paths(
            httpx.URL("https://api.openai.com/backend-api"),
            "/codex/responses",
            litellm.LlmProviders.OPENAI.value,
        )
        assert "/backend-api/codex/responses" in result

    def test_openai_v1_insertion(self):
        result = BaseOpenAIPassThroughHandler._join_url_paths(
            httpx.URL("https://api.openai.com"),
            "/chat/completions",
            litellm.LlmProviders.OPENAI,
        )
        assert "api.openai.com/v1/chat/completions" in result

    def test_non_openai_no_v1_insertion(self):
        result = BaseOpenAIPassThroughHandler._join_url_paths(
            httpx.URL("https://api.x.ai"),
            "/chat/completions",
            litellm.LlmProviders.XAI,
        )
        assert "/v1/" not in result


class TestNormalizeEndpointForTarget:
    def test_plain_endpoint(self):
        result = BaseOpenAIPassThroughHandler._normalize_endpoint_for_target(
            "/v1/chat/completions", "https://api.openai.com"
        )
        assert result == "/v1/chat/completions"

    def test_chatgpt_codex_strips_v1(self):
        result = BaseOpenAIPassThroughHandler._normalize_endpoint_for_target(
            "/v1/responses",
            "https://chatgpt.com/backend-api/codex",
        )
        assert result == "/responses"

    def test_base_v1_strips_v1_prefix(self):
        result = BaseOpenAIPassThroughHandler._normalize_endpoint_for_target(
            "/v1/responses",
            "https://api.example.com/v1",
        )
        assert result == "/responses"

    def test_adds_leading_slash(self):
        result = BaseOpenAIPassThroughHandler._normalize_endpoint_for_target(
            "v1/chat", "https://api.example.com"
        )
        assert result.startswith("/")


# ---------------------------------------------------------------------------
# _assemble_headers / _append_openai_beta_header
# ---------------------------------------------------------------------------


class TestAssembleHeaders:
    def test_with_api_key(self):
        install_runtime(_make_runtime())
        headers = BaseOpenAIPassThroughHandler._assemble_headers(
            api_key="sk-abc", request=_fake_request()
        )
        assert headers["authorization"] == "Bearer sk-abc"
        assert headers["api-key"] == "sk-abc"

    def test_without_api_key(self):
        install_runtime(_make_runtime())
        headers = BaseOpenAIPassThroughHandler._assemble_headers(
            api_key=None, request=_fake_request()
        )
        assert "authorization" not in headers

    def test_extra_headers_merged(self):
        install_runtime(_make_runtime())
        headers = BaseOpenAIPassThroughHandler._assemble_headers(
            api_key="sk-abc",
            request=_fake_request(),
            extra_headers={"x-custom": "val"},
        )
        assert headers["x-custom"] == "val"

    def test_assistants_beta_header_appended(self):
        install_runtime(
            _make_runtime(is_assistants_api_request_fn=lambda req: True)
        )
        headers = BaseOpenAIPassThroughHandler._assemble_headers(
            api_key=None, request=_fake_request()
        )
        assert headers["OpenAI-Beta"] == "assistants=v2"

    def test_assistants_beta_header_not_duplicated(self):
        install_runtime(
            _make_runtime(is_assistants_api_request_fn=lambda req: True)
        )
        headers = BaseOpenAIPassThroughHandler._assemble_headers(
            api_key=None,
            request=_fake_request(),
            extra_headers={"OpenAI-Beta": "assistants=v1"},
        )
        assert headers["OpenAI-Beta"] == "assistants=v1"


# ---------------------------------------------------------------------------
# _prepare_openai_oa_xai_context
# ---------------------------------------------------------------------------


class TestPrepareOaXaiContext:
    def test_returns_none_when_not_prepared(self):
        install_runtime(_make_runtime())
        result = asyncio.run(
            BaseOpenAIPassThroughHandler._prepare_openai_oa_xai_context(
                endpoint="/v1/responses",
                request_body={"model": "test"},
            )
        )
        assert result is None

    def test_fail_closed_on_missing_credential(self):
        install_runtime(
            _make_runtime(
                prepare_oa_xai_passthrough_request_fn=AsyncMock(
                    return_value=(True, None, None)
                )
            )
        )
        with pytest.raises(Exception, match="managed xAI OAuth credential"):
            asyncio.run(
                BaseOpenAIPassThroughHandler._prepare_openai_oa_xai_context(
                    endpoint="/v1/responses",
                    request_body={"model": "test"},
                )
            )

    def test_success_returns_tuple(self):
        install_runtime(
            _make_runtime(
                prepare_oa_xai_passthrough_request_fn=AsyncMock(
                    return_value=(True, "https://api.x.ai", "xai-key-123")
                ),
                to_xai_native_passthrough_model_fn=lambda m: f"xai-{m}",
            )
        )
        result = asyncio.run(
            BaseOpenAIPassThroughHandler._prepare_openai_oa_xai_context(
                endpoint="/v1/responses",
                request_body={"model": "grok-3"},
            )
        )
        assert result is not None
        base_url, api_key, body, url = result
        assert base_url == "https://api.x.ai"
        assert api_key == "xai-key-123"
        assert "api.x.ai" in url


# ---------------------------------------------------------------------------
# _prepare_openai_grok_native_oauth_context
# ---------------------------------------------------------------------------


class TestPrepareGrokNativeContext:
    def test_returns_none_when_not_prepared(self):
        install_runtime(_make_runtime())
        result = asyncio.run(
            BaseOpenAIPassThroughHandler._prepare_openai_grok_native_oauth_context(
                endpoint="/v1/responses",
                request=_fake_request(),
                request_body={"model": "test"},
                extra_headers=None,
            )
        )
        assert result is None

    def test_fail_closed_on_missing_base_url(self):
        install_runtime(
            _make_runtime(
                prepare_grok_native_oauth_passthrough_request_fn=AsyncMock(
                    return_value=(True, None, {}, {})
                )
            )
        )
        with pytest.raises(Exception, match="Grok target base URL"):
            asyncio.run(
                BaseOpenAIPassThroughHandler._prepare_openai_grok_native_oauth_context(
                    endpoint="/v1/responses",
                    request=_fake_request(),
                    request_body={"model": "test"},
                    extra_headers=None,
                )
            )

    def test_success_merges_headers(self):
        install_runtime(
            _make_runtime(
                prepare_grok_native_oauth_passthrough_request_fn=AsyncMock(
                    return_value=(
                        True,
                        "https://grok.api",
                        {"x-grok": "yes"},
                        {"model": "grok-3"},
                    )
                )
            )
        )
        result = asyncio.run(
            BaseOpenAIPassThroughHandler._prepare_openai_grok_native_oauth_context(
                endpoint="/v1/responses",
                request=_fake_request(),
                request_body={"model": "test"},
                extra_headers={"x-extra": "val"},
            )
        )
        assert result is not None
        base_url, headers, body, url = result
        assert base_url == "https://grok.api"
        assert headers["x-grok"] == "yes"
        assert headers["x-extra"] == "val"
        assert "grok.test" in url


# ---------------------------------------------------------------------------
# _base_openai_pass_through_handler dispatch parity
# ---------------------------------------------------------------------------


class TestBaseHandlerDispatch:
    def _run_handler(self, **overrides: Any) -> Any:
        endpoint_func = AsyncMock(return_value="passthrough-response")
        rt_kwargs: dict[str, Any] = dict(
            create_pass_through_route_fn=MagicMock(
                return_value=endpoint_func
            ),
        )
        rt_kwargs.update(overrides)
        install_runtime(_make_runtime(**rt_kwargs))
        return asyncio.run(
            BaseOpenAIPassThroughHandler._base_openai_pass_through_handler(
                endpoint="/v1/chat/completions",
                request=_fake_request("GET"),
                fastapi_response=MagicMock(),
                user_api_key_dict=MagicMock(),
                base_target_url="https://api.openai.com",
                api_key="sk-test",
                custom_llm_provider=litellm.LlmProviders.OPENAI,
            )
        )

    def test_get_request_creates_passthrough(self):
        result = self._run_handler()
        assert result == "passthrough-response"

    def test_codex_dispatch_short_circuits(self):
        dispatched = MagicMock(name="dispatched-response")
        install_runtime(
            _make_runtime(
                request_uses_codex_native_auth_fn=lambda req: True,
                try_dispatch_codex_request_fn=AsyncMock(
                    return_value=dispatched
                ),
                create_pass_through_route_fn=MagicMock(
                    return_value=AsyncMock()
                ),
            )
        )
        result = asyncio.run(
            BaseOpenAIPassThroughHandler._base_openai_pass_through_handler(
                endpoint="/v1/responses",
                request=_fake_request("POST"),
                fastapi_response=MagicMock(),
                user_api_key_dict=MagicMock(),
                base_target_url="https://api.openai.com",
                api_key="sk-test",
                custom_llm_provider=litellm.LlmProviders.OPENAI,
            )
        )
        assert result is dispatched

    def test_xai_context_overrides_provider(self):
        endpoint_func = AsyncMock(return_value="xai-ok")
        create_fn = MagicMock(return_value=endpoint_func)
        install_runtime(
            _make_runtime(
                prepare_oa_xai_passthrough_request_fn=AsyncMock(
                    return_value=(True, "https://api.x.ai", "xai-key")
                ),
                create_pass_through_route_fn=create_fn,
            )
        )
        asyncio.run(
            BaseOpenAIPassThroughHandler._base_openai_pass_through_handler(
                endpoint="/v1/responses",
                request=_fake_request("POST"),
                fastapi_response=MagicMock(),
                user_api_key_dict=MagicMock(),
                base_target_url="https://api.openai.com",
                api_key="sk-test",
                custom_llm_provider=litellm.LlmProviders.OPENAI,
            )
        )
        call_kwargs = create_fn.call_args
        assert call_kwargs.kwargs.get("egress_credential_family") == "xai"
        assert call_kwargs.kwargs.get("expected_target_family") == "xai"
        assert call_kwargs.kwargs["_forward_headers"] is False
        assert call_kwargs.kwargs["custom_llm_provider"] == (
            litellm.LlmProviders.XAI.value
        )
        assert call_kwargs.kwargs["custom_headers"]["authorization"] == (
            "Bearer xai-key"
        )

    def test_grok_responses_has_priority_over_codex_dispatch(self):
        dispatch = AsyncMock(return_value=MagicMock())
        endpoint_func = AsyncMock(return_value="grok-ok")
        create_fn = MagicMock(return_value=endpoint_func)
        install_runtime(
            _make_runtime(
                request_uses_codex_native_auth_fn=lambda request: True,
                prepare_grok_native_oauth_passthrough_request_fn=AsyncMock(
                    return_value=(
                        True,
                        "https://grok.api",
                        {"x-grok-auth": "oauth"},
                        {"model": "grok-native"},
                    )
                ),
                try_dispatch_codex_request_fn=dispatch,
                create_pass_through_route_fn=create_fn,
            )
        )

        result = asyncio.run(
            BaseOpenAIPassThroughHandler._base_openai_pass_through_handler(
                endpoint="/v1/responses",
                request=_fake_request("POST"),
                fastapi_response=MagicMock(),
                user_api_key_dict=MagicMock(),
                base_target_url="https://api.openai.com",
                api_key="sk-original",
                custom_llm_provider=litellm.LlmProviders.OPENAI,
                forward_headers=True,
            )
        )

        assert result == "grok-ok"
        dispatch.assert_not_awaited()
        route_kwargs = create_fn.call_args.kwargs
        assert route_kwargs["_forward_headers"] is False
        assert route_kwargs["custom_llm_provider"] == (
            litellm.LlmProviders.XAI.value
        )
        assert route_kwargs["custom_headers"] == {"x-grok-auth": "oauth"}
        assert route_kwargs["egress_credential_family"] == "xai"
        endpoint_func.assert_awaited_once()
        assert endpoint_func.await_args.kwargs["custom_body"] == {
            "model": "grok-native"
        }

    def test_observability_body_is_safe_set_and_forwarded_as_custom_body(self):
        request = _fake_request("POST")
        original_body = {"model": "gpt-4"}
        observed_body = {"model": "gpt-4", "metadata": {"observed": True}}
        safe_set = MagicMock()
        endpoint_func = AsyncMock(return_value="observed-ok")
        install_runtime(
            _make_runtime(
                get_request_body_fn=AsyncMock(return_value=original_body),
                prepare_request_body_for_passthrough_observability_fn=(
                    lambda **kwargs: observed_body
                ),
                safe_set_request_parsed_body_fn=safe_set,
                create_pass_through_route_fn=MagicMock(
                    return_value=endpoint_func
                ),
            )
        )

        result = asyncio.run(
            BaseOpenAIPassThroughHandler._base_openai_pass_through_handler(
                endpoint="/v1/chat/completions",
                request=request,
                fastapi_response=MagicMock(),
                user_api_key_dict=MagicMock(),
                base_target_url="https://api.openai.com",
                api_key="sk-test",
                custom_llm_provider=litellm.LlmProviders.OPENAI,
            )
        )

        assert result == "observed-ok"
        safe_set.assert_called_once_with(request, observed_body)
        assert endpoint_func.await_args.kwargs["custom_body"] is observed_body

    def test_streaming_url_sets_streaming_route_flag(self):
        create_fn = MagicMock(
            return_value=AsyncMock(return_value="stream-ok")
        )
        install_runtime(_make_runtime(create_pass_through_route_fn=create_fn))

        result = asyncio.run(
            BaseOpenAIPassThroughHandler._base_openai_pass_through_handler(
                endpoint="/v1/stream",
                request=_fake_request("GET"),
                fastapi_response=MagicMock(),
                user_api_key_dict=MagicMock(),
                base_target_url="https://api.openai.com",
                api_key="sk-test",
                custom_llm_provider=litellm.LlmProviders.OPENAI,
            )
        )

        assert result == "stream-ok"
        assert create_fn.call_args.kwargs["is_streaming_request"] is True

    def test_auto_agent_forces_codex_tool_policy_sequence(self):
        events: list[str] = []

        def _step(name: str):
            def _apply(body: dict[str, Any]) -> tuple[dict[str, Any], list]:
                events.append(name)
                return body, []

            return _apply

        dispatch_result = MagicMock(name="auto-agent-dispatch")
        install_runtime(
            _make_runtime(
                resolve_codex_auto_agent_alias_model_fn=(
                    lambda body, **kwargs: "basic"
                ),
                add_route_family_logging_metadata_fn=(
                    lambda body, family: events.append("route") or body
                ),
                apply_codex_tool_description_patches_fn=_step("patch"),
                drop_unsupported_codex_hosted_tools_fn=_step("hosted"),
                drop_unsupported_codex_request_params_fn=_step("params"),
                drop_unsupported_codex_input_items_fn=_step("input"),
                is_oa_xai_request_body_fn=(
                    lambda body: events.append("is_oa") or True
                ),
                drop_tool_choice_without_tools_fn=_step("tool_choice"),
                add_codex_request_breakout_logging_metadata_fn=(
                    lambda body: events.append("breakout") or body
                ),
                try_dispatch_codex_request_fn=AsyncMock(
                    return_value=dispatch_result
                ),
            )
        )

        result = asyncio.run(
            BaseOpenAIPassThroughHandler._base_openai_pass_through_handler(
                endpoint="/v1/responses",
                request=_fake_request("POST"),
                fastapi_response=MagicMock(),
                user_api_key_dict=MagicMock(),
                base_target_url="https://api.openai.com",
                api_key="sk-test",
                custom_llm_provider=litellm.LlmProviders.OPENAI,
            )
        )

        assert result is dispatch_result
        assert events == [
            "route",
            "patch",
            "hosted",
            "params",
            "input",
            "is_oa",
            "tool_choice",
            "breakout",
        ]

    def test_codex_dispatch_awaits_host_attribution_and_merges_metadata(self):
        events: list[str] = []
        expected_host_attribution = {
            "client_ip": "100.110.233.24",
            "client_ip_source": "x-forwarded-for",
            "host_name": "mahaf",
            "host_name_source": "dns",
        }
        request = _fake_request("POST")
        request.state = SimpleNamespace()

        async def fake_aresolve(req, *, allow_blocking_lookup):
            events.append("resolve")
            assert allow_blocking_lookup is True
            assert req is request
            return expected_host_attribution

        async def dispatch_fn(
            *,
            prepared_request_body: dict[str, Any],
            **kwargs: Any,
        ) -> MagicMock:
            events.append("dispatch")
            metadata = prepared_request_body.get("litellm_metadata", {})
            assert metadata["client_ip"] == "100.110.233.24"
            assert metadata["client_ip_source"] == "x-forwarded-for"
            assert metadata["host_name"] == "mahaf"
            assert metadata["host_name_source"] == "dns"
            assert getattr(request.state, "aawm_route_host_attribution") == {
                "client_ip": "100.110.233.24",
                "client_ip_source": "x-forwarded-for",
                "host_name": "mahaf",
                "host_name_source": "dns",
            }
            return MagicMock(name="dispatched-response")

        with patch.dict(
            _aresolve_auto_agent_alias_route_host_attribution.__globals__,
            {
                "aresolve_aawm_route_host_attribution": fake_aresolve,
            },
        ):
            install_runtime(
                _make_runtime(
                    resolve_codex_auto_agent_alias_model_fn=(
                        lambda body, **kwargs: "basic"
                    ),
                    resolve_auto_agent_alias_route_host_attribution_fn=(
                        _aresolve_auto_agent_alias_route_host_attribution
                    ),
                    add_route_family_logging_metadata_fn=lambda body, family: body,
                    try_dispatch_codex_request_fn=dispatch_fn,
                )
            )
            result = asyncio.run(
                BaseOpenAIPassThroughHandler._base_openai_pass_through_handler(
                    endpoint="/v1/responses",
                    request=request,
                    fastapi_response=MagicMock(),
                    user_api_key_dict=MagicMock(),
                    base_target_url="https://api.openai.com",
                    api_key="sk-test",
                    custom_llm_provider=litellm.LlmProviders.OPENAI,
                )
            )

        assert result is not None
        assert events == ["resolve", "dispatch"]


# ---------------------------------------------------------------------------
# AST structural pin: no module-scope god import
# ---------------------------------------------------------------------------


class TestNoGodModuleImport:
    def test_god_import_occurs_only_inside_runtime_builder(self):
        source = MODULE_PATH.read_text()
        tree = ast.parse(source)

        class GodImportVisitor(ast.NodeVisitor):
            def __init__(self) -> None:
                self.function_stack: list[str] = []
                self.import_scopes: list[tuple[str, ...]] = []

            def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
                self.function_stack.append(node.name)
                self.generic_visit(node)
                self.function_stack.pop()

            def visit_Import(self, node: ast.Import) -> None:
                if any(
                    "llm_passthrough_endpoints" in alias.name
                    for alias in node.names
                ):
                    self.import_scopes.append(tuple(self.function_stack))

            def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
                if (
                    node.module
                    and "llm_passthrough_endpoints" in node.module
                ) or any(
                    alias.name == "llm_passthrough_endpoints"
                    for alias in node.names
                ):
                    self.import_scopes.append(tuple(self.function_stack))

        visitor = GodImportVisitor()
        visitor.visit(tree)
        assert visitor.import_scopes == [("build_runtime_from_host",)]


# ---------------------------------------------------------------------------
# build_runtime_from_host fail-closed
# ---------------------------------------------------------------------------


class TestBuildRuntimeFromHost:
    def test_host_callback_monkeypatch_remains_live_after_build_and_install(self):
        host = _synthetic_host()
        with _synthetic_host_publication(host):
            install_runtime(build_runtime_from_host())
            host._merge_litellm_metadata = (
                lambda body, **kwargs: {**body, "late": True}
            )

            result = asyncio.run(
                BaseOpenAIPassThroughHandler._prepare_openai_oa_xai_context(
                    endpoint="/v1/responses",
                    request_body={"model": "grok"},
                )
            )
            assert result is None
            assert BaseOpenAIPassThroughHandler._get_runtime().merge_litellm_metadata_fn(
                {"model": "grok"}
            ) == {"model": "grok", "late": True}

    def test_try_dispatch_monkeypatch_remains_live_after_build_and_install(self):
        host = _synthetic_host()
        dispatched = MagicMock(name="late-dispatched")
        late_dispatch = AsyncMock(return_value=dispatched)
        with _synthetic_host_publication(host):
            install_runtime(build_runtime_from_host())
            host._request_uses_codex_native_auth = lambda request: True
            host.try_dispatch_codex_request = late_dispatch

            result = asyncio.run(
                BaseOpenAIPassThroughHandler._base_openai_pass_through_handler(
                    endpoint="/v1/responses",
                    request=_fake_request("POST"),
                    fastapi_response=MagicMock(),
                    user_api_key_dict=MagicMock(),
                    base_target_url="https://api.openai.com",
                    api_key="sk-test",
                    custom_llm_provider=litellm.LlmProviders.OPENAI,
                )
            )

        assert result is dispatched
        late_dispatch.assert_awaited_once()

    def test_route_checks_monkeypatch_remains_live_after_build_and_install(self):
        host = _synthetic_host()
        with (
            _synthetic_host_publication(host),
            patch.object(
                RouteChecks,
                "_is_assistants_api_request",
                return_value=False,
            ) as route_check,
        ):
            install_runtime(build_runtime_from_host())
            route_check.return_value = True
            headers = BaseOpenAIPassThroughHandler._assemble_headers(
                api_key=None,
                request=_fake_request(),
            )

        assert headers["OpenAI-Beta"] == "assistants=v2"
        route_check.assert_called_once()

    def test_missing_required_host_publication_fails_closed(self):
        host = _synthetic_host()
        with _synthetic_host_publication(host):
            install_runtime(build_runtime_from_host())
            del host._merge_litellm_metadata

            with pytest.raises(
                RuntimeError,
                match=r"llm_passthrough_endpoints callback "
                r"'_merge_litellm_metadata' is not published",
            ):
                BaseOpenAIPassThroughHandler._get_runtime().merge_litellm_metadata_fn(
                    {}
                )

    def test_missing_route_checks_publication_fails_closed(self):
        host = _synthetic_host()
        with (
            _synthetic_host_publication(host),
            patch.object(
                RouteChecks,
                "_is_assistants_api_request",
                new=None,
            ),
        ):
            install_runtime(build_runtime_from_host())
            with pytest.raises(
                RuntimeError,
                match=r"RouteChecks callback "
                r"'_is_assistants_api_request' is not published",
            ):
                BaseOpenAIPassThroughHandler._assemble_headers(
                    api_key=None,
                    request=_fake_request(),
                )


# ---------------------------------------------------------------------------
# install_runtime contract
# ---------------------------------------------------------------------------


class TestInstallRuntime:
    def test_install_sets_class_runtime(self):
        rt = _make_runtime()
        install_runtime(rt)
        assert BaseOpenAIPassThroughHandler._runtime is rt

    def test_install_replaces_previous(self):
        rt1 = _make_runtime()
        rt2 = _make_runtime()
        install_runtime(rt1)
        install_runtime(rt2)
        assert BaseOpenAIPassThroughHandler._runtime is rt2
