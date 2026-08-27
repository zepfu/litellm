"""GREEN Wave 6 smoke tests: module imports, YAML compile stability, and
refresh-endpoint route registration for the AAWM alias config framework."""

from __future__ import annotations

import asyncio
import os

from fastapi import FastAPI
from fastapi.routing import APIRoute
import httpx

REFRESH_PATH = "/aawm/alias-config/refresh"

_REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
)
_ALIAS_CONFIG_DIR = os.path.join(
    _REPO_ROOT, "litellm", "proxy", "aawm_alias_config"
)
_BASIC_YAML_PATH = os.path.join(_ALIAS_CONFIG_DIR, "basic.yaml")


def test_module_imports() -> None:
    """The alias-routing config framework modules import cleanly."""
    from litellm.proxy.pass_through_endpoints.aawm_alias_routing import (
        classification,
        config_compiler,
        failure_vocabulary,
    )

    assert hasattr(config_compiler, "compile_yaml")
    assert hasattr(classification, "classify_failure")
    assert hasattr(classification, "classify_exception")
    assert hasattr(failure_vocabulary, "FailureEvent")
    assert hasattr(failure_vocabulary, "FailureClassRegistry")
    assert hasattr(failure_vocabulary, "is_coolable")


def test_basic_yaml_compiles() -> None:
    """``basic.yaml`` compiles into a valid snapshot with a stable content-derived hash."""
    from litellm.proxy.pass_through_endpoints.aawm_alias_routing import (
        config_compiler as compiler,
    )

    with open(_BASIC_YAML_PATH, "r", encoding="utf-8") as handle:
        raw_yaml = handle.read()

    first = compiler.compile_yaml(raw_yaml)
    second = compiler.compile_yaml(raw_yaml)

    assert "basic" in first.aliases
    assert len(first.aliases["basic"].candidates) > 0
    # config_hash is a pure content hash of the source YAML -- identical
    # input yields an identical hash across independent compiles, even
    # though config_epoch increments each time.
    assert first.config_hash == second.config_hash
    assert first.config_version == second.config_version
    assert first.config_epoch != second.config_epoch

    basic_candidate = first.aliases["basic"].candidates[0]
    assert basic_candidate.provider == "zai_coding_plan"
    assert basic_candidate.model == "zai_coding_plan/glm-5.3-flash"
    assert basic_candidate.route_family == (
        "codex_zai_coding_plan_chat_completions_adapter"
    )
    assert basic_candidate.priority == 100
    assert basic_candidate.reasoning_effort == "low"


def test_alpha_stabilization_alias_mappings_are_single_direct_candidates() -> None:
    """Temporary alpha aliases expose only their explicitly assigned route."""
    from litellm.proxy.pass_through_endpoints.aawm_alias_routing import (
        config_compiler as compiler,
    )

    expected = {
        "basic.yaml": (
            "basic",
            "zai_coding_plan",
            "zai_coding_plan/glm-5.3-flash",
            "codex_zai_coding_plan_chat_completions_adapter",
            "low",
        ),
        "read.yaml": (
            "read",
            "zai_coding_plan",
            "zai_coding_plan/glm-5.3-flash",
            "codex_zai_coding_plan_chat_completions_adapter",
            "low",
        ),
        "work.yaml": (
            "work",
            "cursor_agent",
            "cursor_agent/cursor-grok-4.6-high",
            "codex_cursor_agent_aiserver_adapter",
            None,
        ),
        "expert.yaml": (
            "expert",
            "openai",
            "gpt-5.6-terra",
            "codex_responses",
            "max",
        ),
        "sota-openai.yaml": (
            "sota-openai",
            "openai",
            "gpt-5.6-sol",
            "codex_responses",
            "medium",
        ),
    }

    for filename, (
        alias_name,
        provider,
        model,
        route_family,
        reasoning_effort,
    ) in expected.items():
        with open(
            os.path.join(_ALIAS_CONFIG_DIR, filename), "r", encoding="utf-8"
        ) as handle:
            snapshot = compiler.compile_yaml(handle.read())

        candidates = snapshot.aliases[alias_name].candidates
        assert len(candidates) == 1, alias_name
        candidate = candidates[0]
        assert getattr(candidate, "alias_name", None) is None
        assert candidate.provider == provider
        assert candidate.model == model
        assert candidate.route_family == route_family
        assert candidate.priority == 100
        assert candidate.reasoning_effort == reasoning_effort


def test_refresh_endpoint_registered() -> None:
    """``POST /aawm/alias-config/refresh`` is registered on the pass-through router."""
    from litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints import router

    matching_routes = [route for route in router.routes if isinstance(route, APIRoute) and route.path == REFRESH_PATH]
    assert matching_routes, f"expected a registered route at {REFRESH_PATH}"
    assert any("POST" in route.methods for route in matching_routes)

    app = FastAPI()
    app.include_router(router)

    async def _post() -> httpx.Response:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            return await client.post(
                REFRESH_PATH,
                json={
                    "yaml": (
                        "defaults: {}\n"
                        "aliases:\n"
                        "  - name: basic\n"
                        "    candidates:\n"
                        "      - provider: openai\n"
                        "        model: gpt-5.4-mini\n"
                        "        route_family: codex_responses\n"
                        "        priority: 0\n"
                    )
                },
            )

    response = asyncio.run(_post())
    assert response.status_code == 200
    payload = response.json()
    assert "active_config_hash" in payload
    assert "config_version" in payload
