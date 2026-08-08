from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import HTTPException

import litellm
from litellm.proxy import proxy_server
from litellm.proxy.health_endpoints import _health_endpoints
from litellm.proxy.pass_through_endpoints import aawm_context_query
from litellm.proxy.pass_through_endpoints import llm_passthrough_endpoints as host
from litellm.proxy.pass_through_endpoints.aawm_alias_routing import (
    config_startup,
    openrouter_quota,
)


@pytest.fixture(autouse=True)
def restore_control_plane_host_state():
    saved_status = host.get_aawm_claude_control_plane_initialization_status()
    saved_callbacks = (
        host._aawm_add_claude_post_rewrite_context_file_logging_metadata,
        host._aawm_apply_claude_control_plane_rewrites_to_anthropic_request_body,
        host._aawm_expand_aawm_dynamic_directives_in_anthropic_request_body,
    )
    yield
    host._aawm_claude_control_plane_initialization_status = saved_status
    (
        host._aawm_add_claude_post_rewrite_context_file_logging_metadata,
        host._aawm_apply_claude_control_plane_rewrites_to_anthropic_request_body,
        host._aawm_expand_aawm_dynamic_directives_in_anthropic_request_body,
    ) = saved_callbacks


def _missing_module(name: str) -> ModuleNotFoundError:
    error = ModuleNotFoundError(f"No module named {name!r}")
    error.name = name
    return error


@pytest.mark.asyncio
async def test_initialization_composes_active_callbacks_and_neutral_pool_owner() -> None:
    captured: dict[str, Any] = {}

    async def get_agent_memories(**_kwargs: Any) -> str:
        return "memory"

    async def get_context(**_kwargs: Any) -> str:
        return "context"

    async def get_reference_identifiers(**_kwargs: Any) -> str:
        return "identifier"

    class _Rewriter:
        def add_post_rewrite_context_file_metadata(
            self,
            request_body: dict[str, Any],
        ) -> dict[str, Any]:
            return {**request_body, "metadata": True}

        async def apply_rewrites(
            self,
            request_body: dict[str, Any],
            _billing_header_fields: dict[str, str],
        ) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
            return {**request_body, "rewritten": True}, [], []

        async def expand_dynamic_context(
            self,
            request_body: dict[str, Any],
        ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
            return {**request_body, "expanded": True}, []

    def build_services(**kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs)
        return kwargs

    prompt_services = object()
    prompt_replacement_module = SimpleNamespace(
        build_claude_prompt_replacement_services=lambda: prompt_services,
    )
    control_plane_module = SimpleNamespace(
        _call_aawm_get_agent_memories=get_agent_memories,
        _call_aawm_context_grab=get_context,
        _call_aawm_reference_identifier_list=get_reference_identifiers,
        build_claude_control_plane_services=build_services,
        compose_claude_control_plane=lambda _services: _Rewriter(),
    )

    host._initialize_aawm_claude_control_plane(
        import_module=lambda name: (
            control_plane_module
            if name == host._AAWM_CLAUDE_CONTROL_PLANE_MODULE
            else (
                prompt_replacement_module
                if name == host._AAWM_CLAUDE_PROMPT_REPLACEMENT_MODULE
                else pytest.fail(f"unexpected import: {name}")
            )
        )
    )

    assert host.get_aawm_claude_control_plane_initialization_status() == {
        "state": "active",
        "mode": "enabled",
        "ready": True,
        "reason": None,
        "error_type": None,
    }
    assert captured["prompt"] is prompt_services
    assert (
        await captured["context_query"].get_agent_memories(
            agent_name="writer",
            tenant_id="litellm",
        )
        == "memory"
    )
    rewritten, _, _ = await host._aawm_apply_claude_control_plane_rewrites_to_anthropic_request_body(
        {},
        {},
    )
    assert rewritten == {"rewritten": True}
    assert host._get_aawm_dynamic_injection_pool is (aawm_context_query._get_aawm_dynamic_injection_pool)
    assert openrouter_quota._get_dynamic_injection_pool is (aawm_context_query._get_aawm_dynamic_injection_pool)


@pytest.mark.asyncio
async def test_exact_optional_module_absence_is_visible_degraded_mode() -> None:
    host._initialize_aawm_claude_control_plane(
        import_module=lambda _name: (_ for _ in ()).throw(_missing_module(host._AAWM_CLAUDE_CONTROL_PLANE_MODULE))
    )

    assert host.get_aawm_claude_control_plane_initialization_status() == {
        "state": "degraded",
        "mode": "optional",
        "ready": True,
        "reason": "optional_module_absent",
        "error_type": None,
    }
    request_body = {"system": "unchanged"}
    rewritten, overrides, patches = await host._aawm_apply_claude_control_plane_rewrites_to_anthropic_request_body(
        request_body,
        {},
    )
    assert rewritten is request_body
    assert overrides == []
    assert patches == []


@pytest.mark.asyncio
@pytest.mark.parametrize("failure_kind", ("transitive_import", "composition"))
async def test_required_or_composition_failure_is_sanitized_and_fail_closed(
    failure_kind: str,
) -> None:
    sensitive_detail = "/runtime/secret/provider-owner"

    if failure_kind == "transitive_import":
        control_plane_module = SimpleNamespace()

        def importer(name: str) -> Any:
            if name == host._AAWM_CLAUDE_CONTROL_PLANE_MODULE:
                return control_plane_module
            raise _missing_module(host._AAWM_CLAUDE_PROMPT_REPLACEMENT_MODULE)

        expected_error_type = "ModuleNotFoundError"
    else:
        prompt_replacement_module = SimpleNamespace(
            build_claude_prompt_replacement_services=lambda: object(),
        )
        control_plane_module = SimpleNamespace(
            _call_aawm_get_agent_memories=lambda **_kwargs: None,
            _call_aawm_context_grab=lambda **_kwargs: None,
            _call_aawm_reference_identifier_list=lambda **_kwargs: None,
            build_claude_control_plane_services=lambda **_kwargs: (_ for _ in ()).throw(RuntimeError(sensitive_detail)),
        )

        def importer(name: str) -> Any:
            if name == host._AAWM_CLAUDE_CONTROL_PLANE_MODULE:
                return control_plane_module
            return prompt_replacement_module

        expected_error_type = "RuntimeError"

    host._initialize_aawm_claude_control_plane(import_module=importer)
    status = host.get_aawm_claude_control_plane_initialization_status()

    assert status == {
        "state": "failed",
        "mode": "unavailable",
        "ready": False,
        "reason": "initialization_failed",
        "error_type": expected_error_type,
    }
    assert sensitive_detail not in str(status)
    with pytest.raises(
        RuntimeError,
        match="^AAWM Claude control plane is unavailable$",
    ):
        await host._aawm_apply_claude_control_plane_rewrites_to_anthropic_request_body(
            {},
            {},
        )


def _configure_readiness_dependencies(
    monkeypatch: pytest.MonkeyPatch,
    *,
    status: dict[str, Any],
) -> None:
    monkeypatch.setattr(config_startup, "is_startup_healthy", lambda: True)
    monkeypatch.setattr(
        config_startup,
        "get_startup_status",
        lambda: {"state": "active"},
    )
    monkeypatch.setattr(
        host,
        "get_aawm_claude_control_plane_initialization_status",
        lambda: dict(status),
    )
    monkeypatch.setattr(
        host,
        "is_aawm_claude_control_plane_ready",
        lambda: bool(status["ready"]),
    )
    monkeypatch.setattr(proxy_server, "prisma_client", None)
    monkeypatch.setattr(proxy_server, "version", "test")
    monkeypatch.setattr(litellm, "cache", None)
    monkeypatch.setattr(litellm, "success_callback", [])
    monkeypatch.setattr(
        _health_endpoints,
        "_get_aawm_alias_routing_cache_status",
        lambda: {},
    )
    monkeypatch.setattr(
        _health_endpoints.AsyncHTTPHandler,
        "_should_use_aiohttp_transport",
        lambda: False,
    )


@pytest.mark.asyncio
async def test_readiness_reports_explicit_degraded_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    status = {
        "state": "degraded",
        "mode": "optional",
        "ready": True,
        "reason": "optional_module_absent",
        "error_type": None,
    }
    _configure_readiness_dependencies(monkeypatch, status=status)

    response = await _health_endpoints.health_readiness()

    assert response["status"] == "healthy"
    assert response["aawm_claude_control_plane"] == status


@pytest.mark.asyncio
async def test_readiness_fails_for_required_control_plane_defect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    status = {
        "state": "failed",
        "mode": "unavailable",
        "ready": False,
        "reason": "initialization_failed",
        "error_type": "ModuleNotFoundError",
    }
    _configure_readiness_dependencies(monkeypatch, status=status)

    with pytest.raises(HTTPException) as exc_info:
        await _health_endpoints.health_readiness()

    assert exc_info.value.status_code == 503
    assert exc_info.value.detail == {
        "status": "not_ready",
        "reason": "aawm_claude_control_plane_unavailable",
        "aawm_claude_control_plane": status,
    }
