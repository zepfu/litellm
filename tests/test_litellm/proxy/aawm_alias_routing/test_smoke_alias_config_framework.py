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
_BASIC_YAML_PATH = os.path.join(_REPO_ROOT, "litellm", "proxy", "aawm_alias_config", "basic.yaml")


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
