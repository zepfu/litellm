"""Wave 7 module-local tests for anthropic_native owner functions.

Write scope: this file only.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from litellm.proxy.pass_through_endpoints.aawm_adapter_runtime.anthropic_native import (
    _ANTHROPIC_BETA_HEADER_NAME,
    _ANTHROPIC_CONTEXT_1M_BETA_HEADER,
    _ANTHROPIC_DANGEROUS_DIRECT_BROWSER_ACCESS_HEADER_NAME,
    _ANTHROPIC_NATIVE_PASSTHROUGH_MODEL_ALIASES,
    _perform_anthropic_native_passthrough_request,
    _append_anthropic_beta_header_value,
    _get_header_value_case_insensitive,
    _normalize_anthropic_native_passthrough_model_alias,
    _prepare_anthropic_context_1m_native_passthrough,
    _prepare_anthropic_oauth_native_passthrough_headers,
    AnthropicNativeRuntime,
    install,
)
import litellm.proxy.pass_through_endpoints.aawm_adapter_runtime.anthropic_native as _mod


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_request(headers: dict[str, str]) -> MagicMock:
    """Build a minimal Request-like mock with a .headers mapping."""
    req = MagicMock()
    req.headers = headers
    return req


# ---------------------------------------------------------------------------
# _get_header_value_case_insensitive
# ---------------------------------------------------------------------------


class TestGetHeaderValueCaseInsensitive:
    def test_exact_match(self):
        headers = {"Authorization": "Bearer tok"}
        assert _get_header_value_case_insensitive(headers, "Authorization") == "Bearer tok"

    def test_case_insensitive_fallback(self):
        headers = {"authorization": "Bearer tok"}
        assert _get_header_value_case_insensitive(headers, "Authorization") == "Bearer tok"

    def test_missing_returns_none(self):
        assert _get_header_value_case_insensitive({}, "x-missing") is None

    def test_non_string_value_coerced(self):
        headers = {"x-count": 42}
        assert _get_header_value_case_insensitive(headers, "x-count") == "42"


# ---------------------------------------------------------------------------
# _append_anthropic_beta_header_value
# ---------------------------------------------------------------------------


class TestAppendAnthropicBetaHeaderValue:
    def test_appends_to_empty(self):
        headers: dict[str, Any] = {}
        result = _append_anthropic_beta_header_value(headers, "beta-a")
        assert result[_ANTHROPIC_BETA_HEADER_NAME] == "beta-a"

    def test_appends_new_value(self):
        headers: dict[str, Any] = {_ANTHROPIC_BETA_HEADER_NAME: "beta-a"}
        result = _append_anthropic_beta_header_value(headers, "beta-b")
        assert result[_ANTHROPIC_BETA_HEADER_NAME] == "beta-a, beta-b"

    def test_no_duplicate(self):
        headers: dict[str, Any] = {_ANTHROPIC_BETA_HEADER_NAME: "beta-a, beta-b"}
        result = _append_anthropic_beta_header_value(headers, "beta-a")
        assert result[_ANTHROPIC_BETA_HEADER_NAME] == "beta-a, beta-b"

    def test_case_insensitive_existing_key_replaced(self):
        headers: dict[str, Any] = {"Anthropic-Beta": "old"}
        result = _append_anthropic_beta_header_value(headers, "new")
        assert "Anthropic-Beta" not in result
        assert result[_ANTHROPIC_BETA_HEADER_NAME] == "old, new"


# ---------------------------------------------------------------------------
# _prepare_anthropic_oauth_native_passthrough_headers
# ---------------------------------------------------------------------------


class TestPrepareAnthropicOauthNativePassthroughHeaders:
    def test_non_oauth_returns_unchanged(self):
        req = _make_request({"authorization": "Bearer sk-ant-api03-regular"})
        custom = {"x-api-key": "key"}
        result_headers, is_oauth = _prepare_anthropic_oauth_native_passthrough_headers(
            request=req, custom_headers=custom
        )
        assert is_oauth is False
        assert result_headers is custom

    def test_oauth_key_adds_beta_and_browser_access(self):
        req = _make_request({"authorization": "Bearer sk-ant-oat01-oauth-token"})
        custom: dict[str, Any] = {"x-api-key": "key"}
        result_headers, is_oauth = _prepare_anthropic_oauth_native_passthrough_headers(
            request=req, custom_headers=custom
        )
        assert is_oauth is True
        assert "oauth-2025-04-20" in result_headers[_ANTHROPIC_BETA_HEADER_NAME]
        assert result_headers[_ANTHROPIC_DANGEROUS_DIRECT_BROWSER_ACCESS_HEADER_NAME] == "true"

    def test_oauth_merges_request_beta_values(self):
        req = _make_request({
            "authorization": "Bearer sk-ant-oat01-token",
            "anthropic-beta": "custom-beta-1, custom-beta-2",
        })
        custom: dict[str, Any] = {}
        result_headers, is_oauth = _prepare_anthropic_oauth_native_passthrough_headers(
            request=req, custom_headers=custom
        )
        assert is_oauth is True
        beta = result_headers[_ANTHROPIC_BETA_HEADER_NAME]
        assert "custom-beta-1" in beta
        assert "custom-beta-2" in beta
        assert "oauth-2025-04-20" in beta

    def test_no_auth_header_returns_unchanged(self):
        req = _make_request({})
        custom = {"x-api-key": "key"}
        result_headers, is_oauth = _prepare_anthropic_oauth_native_passthrough_headers(
            request=req, custom_headers=custom
        )
        assert is_oauth is False
        assert result_headers is custom


# ---------------------------------------------------------------------------
# _normalize_anthropic_native_passthrough_model_alias
# ---------------------------------------------------------------------------


class TestNormalizeAnthropicNativePassthroughModelAlias:
    def test_known_alias_normalized(self):
        body = {"model": "opus", "messages": []}
        result, changed = _normalize_anthropic_native_passthrough_model_alias(body)
        assert changed is True
        assert result["model"] == "claude-opus-4-6"
        assert result["litellm_metadata"]["anthropic_native_passthrough_model_alias"] == "opus"

    def test_unknown_model_unchanged(self):
        body = {"model": "gpt-4o", "messages": []}
        result, changed = _normalize_anthropic_native_passthrough_model_alias(body)
        assert changed is False
        assert result is body

    def test_already_canonical_unchanged(self):
        body = {"model": "claude-opus-4-6", "messages": []}
        result, changed = _normalize_anthropic_native_passthrough_model_alias(body)
        assert changed is False

    def test_non_string_model_unchanged(self):
        body = {"model": 123, "messages": []}
        result, changed = _normalize_anthropic_native_passthrough_model_alias(body)
        assert changed is False
        assert result is body

    def test_empty_string_model_unchanged(self):
        body = {"model": "  ", "messages": []}
        result, changed = _normalize_anthropic_native_passthrough_model_alias(body)
        assert changed is False

    def test_alias_with_1m_suffix(self):
        body = {"model": "sonnet[1m]", "messages": []}
        result, changed = _normalize_anthropic_native_passthrough_model_alias(body)
        assert changed is True
        assert result["model"] == "claude-sonnet-4-20250514[1m]"

    def test_metadata_preserves_existing_keys(self):
        body = {"model": "haiku", "litellm_metadata": {"existing": "val"}}
        result, changed = _normalize_anthropic_native_passthrough_model_alias(body)
        assert changed is True
        assert result["litellm_metadata"]["existing"] == "val"
        assert result["litellm_metadata"]["inbound_model_alias"] == "haiku"

    def test_original_body_not_mutated(self):
        body = {"model": "opus", "messages": []}
        _normalize_anthropic_native_passthrough_model_alias(body)
        assert body["model"] == "opus"
        assert "litellm_metadata" not in body

    def test_all_aliases_resolve(self):
        for alias, canonical in _ANTHROPIC_NATIVE_PASSTHROUGH_MODEL_ALIASES.items():
            body = {"model": alias}
            result, changed = _normalize_anthropic_native_passthrough_model_alias(body)
            if alias == canonical:
                assert changed is False, f"{alias} should be identity"
            else:
                assert changed is True, f"{alias} should normalize"
                assert result["model"] == canonical


# ---------------------------------------------------------------------------
# _prepare_anthropic_context_1m_native_passthrough
# ---------------------------------------------------------------------------


class TestPrepareAnthropicContext1mNativePassthrough:
    def test_1m_suffix_stripped_and_beta_added(self):
        req = _make_request({})
        body = {"model": "claude-sonnet-4-6[1m]", "messages": []}
        custom: dict[str, Any] = {"x-api-key": "key"}
        result_body, result_headers, changed = _prepare_anthropic_context_1m_native_passthrough(
            request=req, request_body=body, custom_headers=custom
        )
        assert changed is True
        assert result_body["model"] == "claude-sonnet-4-6"
        assert _ANTHROPIC_CONTEXT_1M_BETA_HEADER in result_headers[_ANTHROPIC_BETA_HEADER_NAME]

    def test_no_suffix_unchanged(self):
        req = _make_request({})
        body = {"model": "claude-sonnet-4-6", "messages": []}
        custom: dict[str, Any] = {}
        result_body, result_headers, changed = _prepare_anthropic_context_1m_native_passthrough(
            request=req, request_body=body, custom_headers=custom
        )
        assert changed is False
        assert result_body is body
        assert result_headers is custom

    def test_non_string_model_unchanged(self):
        req = _make_request({})
        body = {"model": None}
        custom: dict[str, Any] = {}
        _, _, changed = _prepare_anthropic_context_1m_native_passthrough(
            request=req, request_body=body, custom_headers=custom
        )
        assert changed is False

    def test_bare_suffix_unchanged(self):
        req = _make_request({})
        body = {"model": "[1m]"}
        custom: dict[str, Any] = {}
        _, _, changed = _prepare_anthropic_context_1m_native_passthrough(
            request=req, request_body=body, custom_headers=custom
        )
        assert changed is False

    def test_request_beta_forwarded(self):
        req = _make_request({"anthropic-beta": "existing-beta"})
        body = {"model": "claude-opus-4-6[1m]"}
        custom: dict[str, Any] = {}
        _, result_headers, changed = _prepare_anthropic_context_1m_native_passthrough(
            request=req, request_body=body, custom_headers=custom
        )
        assert changed is True
        beta = result_headers[_ANTHROPIC_BETA_HEADER_NAME]
        assert "existing-beta" in beta
        assert _ANTHROPIC_CONTEXT_1M_BETA_HEADER in beta

    def test_metadata_set(self):
        req = _make_request({})
        body = {"model": "claude-haiku-4-5[1m]"}
        custom: dict[str, Any] = {}
        result_body, _, changed = _prepare_anthropic_context_1m_native_passthrough(
            request=req, request_body=body, custom_headers=custom
        )
        assert changed is True
        meta = result_body["litellm_metadata"]
        assert meta["inbound_model_alias"] == "claude-haiku-4-5[1m]"
        assert meta["anthropic_native_passthrough_normalized_model"] == "claude-haiku-4-5"

    def test_original_body_not_mutated(self):
        req = _make_request({})
        body = {"model": "claude-sonnet-4-6[1m]"}
        custom: dict[str, Any] = {}
        _prepare_anthropic_context_1m_native_passthrough(
            request=req, request_body=body, custom_headers=custom
        )
        assert body["model"] == "claude-sonnet-4-6[1m]"


# ---------------------------------------------------------------------------
# install() seam
# ---------------------------------------------------------------------------


class TestInstallSeam:
    def test_install_publishes_all_owned_symbols(self):
        host: dict[str, Any] = {}
        install(host)
        expected = {
            "_ANTHROPIC_CONTEXT_1M_MODEL_SUFFIX",
            "_ANTHROPIC_CONTEXT_1M_BETA_HEADER",
            "_ANTHROPIC_BETA_HEADER_NAME",
            "_ANTHROPIC_BETA_XPASS_HEADER_NAME",
            "_ANTHROPIC_DANGEROUS_DIRECT_BROWSER_ACCESS_HEADER_NAME",
            "_ANTHROPIC_NATIVE_PASSTHROUGH_MODEL_ALIASES",
            "_get_header_value_case_insensitive",
            "_append_anthropic_beta_header_value",
            "_prepare_anthropic_oauth_native_passthrough_headers",
            "_normalize_anthropic_native_passthrough_model_alias",
            "_prepare_anthropic_context_1m_native_passthrough",
            "_perform_anthropic_native_passthrough_request",
            "AnthropicNativeRuntime",
        }
        assert expected.issubset(host.keys())

    def test_installed_functions_are_callable(self):
        host: dict[str, Any] = {}
        install(host)
        assert callable(host["_get_header_value_case_insensitive"])
        assert callable(host["_append_anthropic_beta_header_value"])
        assert callable(host["_prepare_anthropic_oauth_native_passthrough_headers"])
        assert callable(host["_normalize_anthropic_native_passthrough_model_alias"])
        assert callable(host["_prepare_anthropic_context_1m_native_passthrough"])
        assert callable(host["_perform_anthropic_native_passthrough_request"])

    def test_install_with_runtime_sets_module_runtime(self):
        async def fake_streaming(request: Any) -> bool:
            return False

        def fake_create(**kwargs: Any) -> Any:
            return None

        rt = AnthropicNativeRuntime(
            is_streaming_request_fn=fake_streaming,
            create_pass_through_route=fake_create,
        )
        host: dict[str, Any] = {}
        install(host, runtime=rt)
        assert _mod._runtime is rt
        # Cleanup: reset module runtime to avoid cross-test pollution
        _mod._runtime = None


# ---------------------------------------------------------------------------
# _perform_anthropic_native_passthrough_request
# ---------------------------------------------------------------------------


class TestPerformAnthropicNativePassthroughRequest:
    @pytest.fixture(autouse=True)
    def _reset_runtime(self):
        """Ensure module runtime is clean before/after each test."""
        _mod._runtime = None
        yield
        _mod._runtime = None

    @pytest.mark.asyncio
    async def test_fails_closed_without_runtime(self):
        req = _make_request({})
        with pytest.raises(RuntimeError, match="runtime not installed"):
            await _perform_anthropic_native_passthrough_request(
                endpoint="/v1/messages",
                request=req,
                fastapi_response=MagicMock(),
                user_api_key_dict=MagicMock(),
                target_url="https://api.anthropic.com/v1/messages",
                custom_headers={},
            )

    @pytest.mark.asyncio
    async def test_streaming_detection_forwarded(self):
        """is_streaming_request_fn result is passed to create_pass_through_route."""
        captured: dict[str, Any] = {}

        async def fake_streaming(request: Any) -> bool:
            return True

        async def fake_endpoint_func(*args: Any) -> MagicMock:
            return MagicMock(status_code=200)

        def fake_create(**kwargs: Any) -> Any:
            captured.update(kwargs)
            return fake_endpoint_func

        rt = AnthropicNativeRuntime(
            is_streaming_request_fn=fake_streaming,
            create_pass_through_route=fake_create,
        )
        _mod._runtime = rt

        req = _make_request({"content-type": "application/json"})
        resp = await _perform_anthropic_native_passthrough_request(
            endpoint="/v1/messages",
            request=req,
            fastapi_response=MagicMock(),
            user_api_key_dict=MagicMock(),
            target_url="https://api.anthropic.com/v1/messages",
            custom_headers={"x-api-key": "test-key"},
        )
        assert captured["is_streaming_request"] is True
        assert captured["target"] == "https://api.anthropic.com/v1/messages"
        assert captured["custom_headers"] == {"x-api-key": "test-key"}
        assert captured["_forward_headers"] is True
        assert captured["endpoint"] == "/v1/messages"
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_non_streaming_detection(self):
        captured: dict[str, Any] = {}

        async def fake_streaming(request: Any) -> bool:
            return False

        async def fake_endpoint_func(*args: Any) -> MagicMock:
            return MagicMock(status_code=200)

        def fake_create(**kwargs: Any) -> Any:
            captured.update(kwargs)
            return fake_endpoint_func

        _mod._runtime = AnthropicNativeRuntime(
            is_streaming_request_fn=fake_streaming,
            create_pass_through_route=fake_create,
        )

        await _perform_anthropic_native_passthrough_request(
            endpoint="/v1/messages",
            request=_make_request({}),
            fastapi_response=MagicMock(),
            user_api_key_dict=MagicMock(),
            target_url="https://api.anthropic.com/v1/messages",
            custom_headers={},
        )
        assert captured["is_streaming_request"] is False

    @pytest.mark.asyncio
    async def test_blocked_headers_forwarded(self):
        captured: dict[str, Any] = {}

        async def fake_streaming(request: Any) -> bool:
            return False

        async def fake_endpoint_func(*args: Any) -> MagicMock:
            return MagicMock()

        def fake_create(**kwargs: Any) -> Any:
            captured.update(kwargs)
            return fake_endpoint_func

        _mod._runtime = AnthropicNativeRuntime(
            is_streaming_request_fn=fake_streaming,
            create_pass_through_route=fake_create,
        )

        blocked = ["x-internal-", "x-debug-"]
        await _perform_anthropic_native_passthrough_request(
            endpoint="/v1/messages",
            request=_make_request({}),
            fastapi_response=MagicMock(),
            user_api_key_dict=MagicMock(),
            target_url="https://api.anthropic.com/v1/messages",
            custom_headers={},
            blocked_pass_through_prefixed_headers=blocked,
        )
        assert captured["blocked_pass_through_prefixed_headers"] == blocked

    @pytest.mark.asyncio
    async def test_endpoint_func_receives_request_response_key(self):
        """endpoint_func is called with (request, fastapi_response, user_api_key_dict)."""
        call_args: list[Any] = []

        async def fake_streaming(request: Any) -> bool:
            return False

        async def fake_endpoint_func(*args: Any) -> MagicMock:
            call_args.extend(args)
            return MagicMock()

        def fake_create(**kwargs: Any) -> Any:
            return fake_endpoint_func

        _mod._runtime = AnthropicNativeRuntime(
            is_streaming_request_fn=fake_streaming,
            create_pass_through_route=fake_create,
        )

        req = _make_request({})
        fastapi_resp = MagicMock()
        key_dict = MagicMock()
        await _perform_anthropic_native_passthrough_request(
            endpoint="/v1/messages",
            request=req,
            fastapi_response=fastapi_resp,
            user_api_key_dict=key_dict,
            target_url="https://api.anthropic.com/v1/messages",
            custom_headers={},
        )
        assert call_args == [req, fastapi_resp, key_dict]

    @pytest.mark.asyncio
    async def test_target_url_passed_verbatim(self):
        """Egress target is exactly what the caller provides (Anthropic-native only)."""
        captured: dict[str, Any] = {}

        async def fake_streaming(request: Any) -> bool:
            return False

        async def fake_endpoint_func(*args: Any) -> MagicMock:
            return MagicMock()

        def fake_create(**kwargs: Any) -> Any:
            captured.update(kwargs)
            return fake_endpoint_func

        _mod._runtime = AnthropicNativeRuntime(
            is_streaming_request_fn=fake_streaming,
            create_pass_through_route=fake_create,
        )

        target = "https://api.anthropic.com/v1/messages"
        await _perform_anthropic_native_passthrough_request(
            endpoint="/anthropic/v1/messages",
            request=_make_request({}),
            fastapi_response=MagicMock(),
            user_api_key_dict=MagicMock(),
            target_url=target,
            custom_headers={},
        )
        assert captured["target"] == target
# Structural: no god-module import at module scope
# ---------------------------------------------------------------------------


class TestStructuralConstraints:
    def test_no_god_module_import(self):
        import ast
        from pathlib import Path

        mod_path = Path(
            __file__
        ).parents[3] / "litellm" / "proxy" / "pass_through_endpoints" / "aawm_adapter_runtime" / "anthropic_native.py"
        # Resolve relative to repo root
        import litellm.proxy.pass_through_endpoints.aawm_adapter_runtime.anthropic_native as mod

        mod_path = Path(mod.__file__).resolve()
        tree = ast.parse(mod_path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert "llm_passthrough_endpoints" not in alias.name
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    assert "llm_passthrough_endpoints" not in node.module
