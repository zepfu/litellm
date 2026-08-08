"""Wave 7 owner tests: codex_auto_agent_route module.

Focused module-local tests for the extracted
``_handle_codex_auto_agent_alias_route`` body now owned by
``aawm_adapter_runtime/codex_auto_agent_route.py``.

Covers:
- CodexAutoAgentRouteRuntime frozen-dataclass contract
- handle_codex_auto_agent_alias_route delegation to candidate_loop.handle_alias_route
- perform_candidate_request closure kwarg forwarding
- max_candidate_attempts derivation from enumeration
- no module-scope god-module import (AST structural pin)
- build_runtime_from_host fail-closed on missing host attributes
- client_product_label extraction and forwarding to enumeration resolver
- get_active_cooldown_state_fn passthrough (unwrapped) to candidate loop
- perform closure exact-response identity

Write-only surface: this file. No production edits.
"""

from __future__ import annotations

import ast
import asyncio
import dataclasses
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from litellm.proxy.pass_through_endpoints.aawm_adapter_runtime.codex_auto_agent_route import (
    CodexAutoAgentRouteRuntime,
    build_runtime_from_host,
    handle_codex_auto_agent_alias_route,
)

MODULE_PATH = Path(
    __import__(
        "litellm.proxy.pass_through_endpoints.aawm_adapter_runtime.codex_auto_agent_route",
        fromlist=["__file__"],
    ).__file__
).resolve()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_runtime(**overrides: Any) -> CodexAutoAgentRouteRuntime:
    """Build a runtime with no-op defaults; override individual seams."""
    defaults: dict[str, Any] = dict(
        extract_client_product_label_fn=lambda req, body: None,
        perform_candidate_request_fn=AsyncMock(),
        select_candidate_fn=MagicMock(),
        resolve_cooldown_publication_fn=MagicMock(),
        publish_cooldown_memory_fn=MagicMock(),
        persist_cooldown_fn=AsyncMock(),
        set_session_affinity_fn=AsyncMock(),
        add_alias_metadata_fn=MagicMock(),
        raise_redispatch_fn=MagicMock(),
        get_active_cooldown_state_fn=AsyncMock(return_value=(0.0, "memory")),
        resolve_selection_enumeration_fn=MagicMock(),
    )
    defaults.update(overrides)
    return CodexAutoAgentRouteRuntime(**defaults)


def _fake_enumeration(n_candidates: int = 3) -> MagicMock:
    enum = MagicMock()
    enum.candidates = tuple({"provider": "openai", "model": f"m{i}"} for i in range(n_candidates))
    return enum


def _fake_request() -> MagicMock:
    req = MagicMock()
    req.state = MagicMock()
    return req


# ---------------------------------------------------------------------------
# Frozen dataclass contract
# ---------------------------------------------------------------------------


class TestRuntimeDataclass:
    def test_is_frozen_dataclass(self):
        assert dataclasses.is_dataclass(CodexAutoAgentRouteRuntime)
        params = CodexAutoAgentRouteRuntime.__dataclass_params__  # type: ignore[attr-defined]
        assert params.frozen is True

    def test_expected_field_names(self):
        names = {f.name for f in dataclasses.fields(CodexAutoAgentRouteRuntime)}
        expected = {
            "extract_client_product_label_fn",
            "perform_candidate_request_fn",
            "select_candidate_fn",
            "resolve_cooldown_publication_fn",
            "publish_cooldown_memory_fn",
            "persist_cooldown_fn",
            "set_session_affinity_fn",
            "add_alias_metadata_fn",
            "raise_redispatch_fn",
            "get_active_cooldown_state_fn",
            "resolve_selection_enumeration_fn",
        }
        assert names == expected

    def test_immutable(self):
        rt = _make_runtime()
        with pytest.raises(dataclasses.FrozenInstanceError):
            rt.extract_client_product_label_fn = lambda req, body: "other"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Handler delegation
# ---------------------------------------------------------------------------


class TestHandleDelegation:
    def test_delegates_to_handle_alias_route(self):
        """The handler must call candidate_loop.handle_alias_route with exact kwargs."""
        sentinel_response = MagicMock(name="response")
        enum = _fake_enumeration(2)
        rt = _make_runtime(
            extract_client_product_label_fn=lambda req, body: "Codex/1.0",
            resolve_selection_enumeration_fn=MagicMock(return_value=enum),
        )
        req = _fake_request()
        body: dict[str, Any] = {"model": "some-model"}
        resp = MagicMock()
        uak = MagicMock()

        with patch(
            "litellm.proxy.pass_through_endpoints.aawm_adapter_runtime.codex_auto_agent_route.handle_alias_route",
            new_callable=AsyncMock,
            return_value=sentinel_response,
        ) as mock_loop:
            result = asyncio.run(
                handle_codex_auto_agent_alias_route(
                    rt,
                    endpoint="/v1/responses",
                    request=req,
                    fastapi_response=resp,
                    user_api_key_dict=uak,
                    prepared_request_body=body,
                    target_url="https://example.com/v1/responses",
                    api_key="sk-test",
                    forward_headers=True,
                    canonical_alias="resolved-model",
                )
            )

        assert result is sentinel_response
        mock_loop.assert_awaited_once()
        call_args = mock_loop.call_args
        services = call_args[0][0]
        assert call_args[1]["alias_family"] == "codex_auto_agent"
        assert call_args[1]["alias_model"] == "resolved-model"
        assert call_args[1]["request"] is req
        assert call_args[1]["prepared_request_body"] is body
        assert call_args[1]["max_candidate_attempts"] == 2
        assert call_args[1]["attempts_metadata_key"] == "codex_auto_agent_attempts"
        assert call_args[1]["skipped_candidates_metadata_key"] == "codex_auto_agent_skipped_candidates"
        assert call_args[1]["no_candidate_detail"] == "No Codex auto-agent alias candidates were available."
        assert call_args[1]["log_label"] == "Codex"
        # services bundle wired from runtime
        assert services.select_candidate_fn is rt.select_candidate_fn
        assert services.resolve_cooldown_publication_fn is rt.resolve_cooldown_publication_fn
        assert services.publish_cooldown_memory_fn is rt.publish_cooldown_memory_fn
        assert services.persist_cooldown_fn is rt.persist_cooldown_fn
        assert services.set_session_affinity_fn is rt.set_session_affinity_fn
        assert services.add_alias_metadata_fn is rt.add_alias_metadata_fn
        assert services.raise_redispatch_fn is rt.raise_redispatch_fn

    def test_max_candidate_attempts_from_enumeration(self):
        """max_candidate_attempts equals len(enumeration.candidates)."""
        for n in (0, 1, 5):
            enum = _fake_enumeration(n)
            rt = _make_runtime(
                resolve_selection_enumeration_fn=MagicMock(return_value=enum),
            )
            with patch(
                "litellm.proxy.pass_through_endpoints.aawm_adapter_runtime.codex_auto_agent_route.handle_alias_route",
                new_callable=AsyncMock,
                return_value=MagicMock(),
            ) as mock_loop:
                asyncio.run(
                    handle_codex_auto_agent_alias_route(
                        rt,
                        endpoint="/v1/responses",
                        request=_fake_request(),
                        fastapi_response=MagicMock(),
                        user_api_key_dict=MagicMock(),
                        prepared_request_body={},
                        target_url="https://example.com",
                        api_key=None,
                        forward_headers=False,
                        canonical_alias="basic",
                    )
                )
            assert mock_loop.call_args[1]["max_candidate_attempts"] == n


# ---------------------------------------------------------------------------
# Perform-candidate-request closure
# ---------------------------------------------------------------------------


class TestPerformClosure:
    def test_closure_forwards_all_kwargs(self):
        """The wrapped perform_candidate_request_fn receives endpoint/request/etc."""
        captured: dict[str, Any] = {}

        async def _capture_perform(**kwargs: Any) -> MagicMock:
            captured.update(kwargs)
            return MagicMock(name="upstream_response")

        enum = _fake_enumeration(1)
        rt = _make_runtime(
            perform_candidate_request_fn=_capture_perform,
            resolve_selection_enumeration_fn=MagicMock(return_value=enum),
        )
        req = _fake_request()
        resp = MagicMock(name="fastapi_resp")
        uak = MagicMock(name="uak")

        # We need to capture the closure that gets passed as
        # services.perform_candidate_request_fn. Patch handle_alias_route to
        # grab the services bundle and invoke the closure directly.
        captured_services: list[Any] = []

        async def _grab_services(services: Any, **kwargs: Any) -> MagicMock:
            captured_services.append(services)
            return MagicMock()

        with patch(
            "litellm.proxy.pass_through_endpoints.aawm_adapter_runtime.codex_auto_agent_route.handle_alias_route",
            side_effect=_grab_services,
        ):
            asyncio.run(
                handle_codex_auto_agent_alias_route(
                    rt,
                    endpoint="/v1/responses",
                    request=req,
                    fastapi_response=resp,
                    user_api_key_dict=uak,
                    prepared_request_body={"model": "m"},
                    target_url="https://target.example.com/v1/responses",
                    api_key="sk-abc",
                    forward_headers=True,
                    canonical_alias="basic",
                )
            )

        assert len(captured_services) == 1
        perform_fn = captured_services[0].perform_candidate_request_fn

        # Invoke the closure as the candidate loop would
        candidate = {"provider": "openai", "model": "gpt-5"}
        candidate_body = {"model": "gpt-5", "input": "hello"}
        asyncio.run(
            perform_fn(candidate=candidate, candidate_body=candidate_body)
        )

        assert captured["endpoint"] == "/v1/responses"
        assert captured["request"] is req
        assert captured["fastapi_response"] is resp
        assert captured["user_api_key_dict"] is uak
        assert captured["candidate"] is candidate
        assert captured["candidate_body"] is candidate_body
        assert captured["target_url"] == "https://target.example.com/v1/responses"
        assert captured["api_key"] == "sk-abc"
        assert captured["forward_headers"] is True


# ---------------------------------------------------------------------------
# AST structural pin: no module-scope god-module import
# ---------------------------------------------------------------------------


class TestNoModuleScopeGodImport:
    def test_no_toplevel_god_module_import(self):
        """The module must NOT import llm_passthrough_endpoints at module scope."""
        source = MODULE_PATH.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(MODULE_PATH))
        assert isinstance(tree, ast.Module)
        for node in tree.body:
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert "llm_passthrough_endpoints" not in alias.name, (
                        f"Module-scope import of god module found: {alias.name}"
                    )
            elif isinstance(node, ast.ImportFrom):
                mod = node.module or ""
                assert "llm_passthrough_endpoints" not in mod, (
                    f"Module-scope from-import of god module found: {mod}"
                )

    def test_god_import_only_inside_build_runtime(self):
        """The lazy god-module import must live inside build_runtime_from_host."""
        source = MODULE_PATH.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(MODULE_PATH))
        assert isinstance(tree, ast.Module)
        parents = {
            child: parent
            for parent in ast.walk(tree)
            for child in ast.iter_child_nodes(parent)
        }

        def _is_god_import(node: ast.AST) -> bool:
            if isinstance(node, ast.Import):
                references = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                references = [node.module or "", *(alias.name for alias in node.names)]
            else:
                return False
            return any("llm_passthrough_endpoints" in value for value in references)

        def _enclosing_function(node: ast.AST) -> str | None:
            parent = parents.get(node)
            while parent is not None:
                if isinstance(parent, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    return parent.name
                parent = parents.get(parent)
            return None

        god_imports = [node for node in ast.walk(tree) if _is_god_import(node)]
        assert len(god_imports) == 1, (
            f"Expected exactly one lazy god-module import, found {len(god_imports)}"
        )
        assert _enclosing_function(god_imports[0]) == "build_runtime_from_host"


# ---------------------------------------------------------------------------
# Behavioral: build_runtime_from_host fail-closed on missing host attributes
# ---------------------------------------------------------------------------


class TestBuildRuntimeFailClosed:
    """build_runtime_from_host must fail closed (AttributeError) when the
    god module is missing required attributes, preventing partial runtime
    construction."""

    def test_missing_host_attr_raises_attribute_error(self):
        """If the god module lacks a required attribute, build_runtime_from_host
        raises AttributeError rather than silently constructing a partial
        runtime."""
        fake_host = MagicMock(spec=[])  # empty spec: no attributes at all
        with patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints",
            fake_host,
            create=True,
        ):
            with pytest.raises(AttributeError):
                build_runtime_from_host()

    def test_partial_host_attrs_still_fails_closed(self):
        """Even if most attrs exist, one missing attr must prevent
        construction."""
        fake_host = MagicMock()
        # Provide all but one required attribute
        fake_host._extract_auto_agent_alias_client_product_label = lambda r, b: None
        fake_host._perform_codex_auto_agent_alias_candidate_request = AsyncMock()
        fake_host._select_codex_auto_agent_candidate = AsyncMock()
        fake_host._resolve_auto_agent_cooldown_publication_plan = MagicMock()
        fake_host._publish_codex_cooldown_memory = MagicMock()
        fake_host._persist_codex_cooldown_durable = AsyncMock()
        fake_host._set_codex_auto_agent_session_affinity = AsyncMock()
        fake_host._add_codex_auto_agent_alias_metadata = MagicMock()
        fake_host._raise_codex_auto_agent_redispatch_required = MagicMock()
        fake_host._get_codex_auto_agent_active_cooldown_state = AsyncMock()
        # Deliberately omit _resolve_aawm_alias_selection_enumeration
        del fake_host._resolve_aawm_alias_selection_enumeration

        with patch(
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints",
            fake_host,
            create=True,
        ):
            with pytest.raises(AttributeError):
                build_runtime_from_host()


# ---------------------------------------------------------------------------
# Behavioral: client_product_label extraction and forwarding
# ---------------------------------------------------------------------------


class TestClientProductLabelForwarding:
    """The handler must extract client_product_label and forward it to
    resolve_selection_enumeration_fn."""

    def test_client_product_label_passed_to_enumeration(self):
        """extract_client_product_label_fn result is forwarded as
        client_product_label kwarg to resolve_selection_enumeration_fn."""
        enum = _fake_enumeration(2)
        resolve_enum_mock = MagicMock(return_value=enum)
        rt = _make_runtime(
            extract_client_product_label_fn=lambda req, body: "Codex-CLI/2.0",
            resolve_selection_enumeration_fn=resolve_enum_mock,
        )
        req = _fake_request()
        body: dict[str, Any] = {"model": "m"}

        with patch(
            "litellm.proxy.pass_through_endpoints.aawm_adapter_runtime.codex_auto_agent_route.handle_alias_route",
            new_callable=AsyncMock,
            return_value=MagicMock(),
        ):
            asyncio.run(
                handle_codex_auto_agent_alias_route(
                    rt,
                    endpoint="/v1/responses",
                    request=req,
                    fastapi_response=MagicMock(),
                    user_api_key_dict=MagicMock(),
                    prepared_request_body=body,
                    target_url="https://example.com",
                    api_key=None,
                    forward_headers=False,
                    canonical_alias="basic",
                )
            )

        resolve_enum_mock.assert_called_once_with(
            req,
            "basic",
            ingress="codex",
            client_product_label="Codex-CLI/2.0",
        )

    def test_none_client_product_label_forwarded(self):
        """When extract returns None, None is forwarded (not omitted)."""
        enum = _fake_enumeration(1)
        resolve_enum_mock = MagicMock(return_value=enum)
        rt = _make_runtime(
            extract_client_product_label_fn=lambda req, body: None,
            resolve_selection_enumeration_fn=resolve_enum_mock,
        )

        with patch(
            "litellm.proxy.pass_through_endpoints.aawm_adapter_runtime.codex_auto_agent_route.handle_alias_route",
            new_callable=AsyncMock,
            return_value=MagicMock(),
        ):
            asyncio.run(
                handle_codex_auto_agent_alias_route(
                    rt,
                    endpoint="/v1/responses",
                    request=_fake_request(),
                    fastapi_response=MagicMock(),
                    user_api_key_dict=MagicMock(),
                    prepared_request_body={},
                    target_url="https://example.com",
                    api_key=None,
                    forward_headers=False,
                    canonical_alias="basic",
                )
            )

        call_kwargs = resolve_enum_mock.call_args
        assert call_kwargs[1]["client_product_label"] is None


# ---------------------------------------------------------------------------
# Behavioral: get_active_cooldown_state_fn passthrough
# ---------------------------------------------------------------------------


class TestCooldownStatePassthrough:
    """get_active_cooldown_state_fn must be passed directly (unwrapped) to
    handle_alias_route, unlike the legacy facade which wraps it."""

    def test_cooldown_fn_identity_preserved(self):
        """The exact runtime.get_active_cooldown_state_fn object is passed
        as get_active_cooldown_state_fn to handle_alias_route."""
        sentinel_fn = AsyncMock(return_value=(0.0, "none"))
        enum = _fake_enumeration(1)
        rt = _make_runtime(
            get_active_cooldown_state_fn=sentinel_fn,
            resolve_selection_enumeration_fn=MagicMock(return_value=enum),
        )

        with patch(
            "litellm.proxy.pass_through_endpoints.aawm_adapter_runtime.codex_auto_agent_route.handle_alias_route",
            new_callable=AsyncMock,
            return_value=MagicMock(),
        ) as mock_loop:
            asyncio.run(
                handle_codex_auto_agent_alias_route(
                    rt,
                    endpoint="/v1/responses",
                    request=_fake_request(),
                    fastapi_response=MagicMock(),
                    user_api_key_dict=MagicMock(),
                    prepared_request_body={},
                    target_url="https://example.com",
                    api_key=None,
                    forward_headers=False,
                    canonical_alias="basic",
                )
            )

        passed_fn = mock_loop.call_args[1]["get_active_cooldown_state_fn"]
        assert passed_fn is sentinel_fn


# ---------------------------------------------------------------------------
# Behavioral: perform closure exact-response identity
# ---------------------------------------------------------------------------


class TestPerformClosureIdentity:
    """The perform closure must return the exact response object from the
    underlying runtime.perform_candidate_request_fn."""

    def test_returns_exact_sentinel(self):
        sentinel = MagicMock(name="exact_upstream_response")

        async def _return_sentinel(**kwargs: Any) -> MagicMock:
            return sentinel

        enum = _fake_enumeration(1)
        rt = _make_runtime(
            perform_candidate_request_fn=_return_sentinel,
            resolve_selection_enumeration_fn=MagicMock(return_value=enum),
        )

        captured_services: list[Any] = []

        async def _grab_services(services: Any, **kwargs: Any) -> MagicMock:
            captured_services.append(services)
            return MagicMock()

        with patch(
            "litellm.proxy.pass_through_endpoints.aawm_adapter_runtime.codex_auto_agent_route.handle_alias_route",
            side_effect=_grab_services,
        ):
            asyncio.run(
                handle_codex_auto_agent_alias_route(
                    rt,
                    endpoint="/v1/responses",
                    request=_fake_request(),
                    fastapi_response=MagicMock(),
                    user_api_key_dict=MagicMock(),
                    prepared_request_body={},
                    target_url="https://example.com",
                    api_key="sk-x",
                    forward_headers=True,
                    canonical_alias="basic",
                )
            )

        perform_fn = captured_services[0].perform_candidate_request_fn
        result = asyncio.run(
            perform_fn(candidate={"m": 1}, candidate_body={"b": 2})
        )
        assert result is sentinel
