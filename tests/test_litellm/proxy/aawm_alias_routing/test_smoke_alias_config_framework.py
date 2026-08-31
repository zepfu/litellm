"""GREEN Wave 6 smoke tests: module imports, YAML compile stability, and
refresh-endpoint route registration for the AAWM alias config framework."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

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
    """CFG-041: basic.yaml compiles via directory since it references basic-other."""
    from litellm.proxy.pass_through_endpoints.aawm_alias_routing.config_startup import (
        compile_directory,
    )

    snapshot = compile_directory(Path(_ALIAS_CONFIG_DIR))
    first = snapshot
    second = compile_directory(Path(_ALIAS_CONFIG_DIR))

    assert "basic" in first.aliases
    assert len(first.aliases["basic"].candidates) > 0
    # config_hash is a pure content hash of the source YAML -- identical
    # input yields an identical hash across independent compiles, even
    # though config_epoch increments each time.
    assert first.config_hash == second.config_hash
    assert first.config_version == second.config_version
    assert first.config_epoch != second.config_epoch

    basic_candidate = first.aliases["basic"].candidates[0]
    assert basic_candidate.provider == "openrouter"
    assert basic_candidate.model == "openrouter/cohere/north-mini-code:free"
    assert basic_candidate.route_family == (
        "codex_openrouter_completion_adapter"
    )
    assert basic_candidate.priority == 80


def test_standalone_alias_mappings() -> None:
    """Standalone aliases expose only their explicitly assigned route."""
    from litellm.proxy.pass_through_endpoints.aawm_alias_routing import (
        config_compiler as compiler,
    )

    expected = {
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


def test_work_yaml_compiles_current_graph() -> None:
    from litellm.proxy.pass_through_endpoints.aawm_alias_routing.config_snapshot import (
        AliasReference,
        RoutingCandidate,
    )
    from litellm.proxy.pass_through_endpoints.aawm_alias_routing.config_startup import (
        compile_directory,
    )

    snapshot = compile_directory(Path(_ALIAS_CONFIG_DIR))
    entries = snapshot.aliases["work"].candidates

    assert len(entries) == 4
    assert isinstance(entries[0], AliasReference)
    assert all(
        isinstance(entry, RoutingCandidate)
        for index, entry in enumerate(entries)
        if index != 0
    )
    assert [
        (
            ("REF", entry.alias_name, None, entry.priority)
            if isinstance(entry, AliasReference)
            else (
                entry.provider,
                entry.model,
                entry.route_family,
                entry.priority,
            )
        )
        for entry in entries
    ] == [
        (
            "REF",
            "work-other",
            None,
            110,
        ),
        ("anthropic", "claude-sonnet-5[1m]", "anthropic_messages", 80),
        ("anthropic", "claude-sonnet-5", "anthropic_messages", 70),
        ("openai", "gpt-5.6-luna", "codex_responses", 0),
    ]
    for candidate in entries[1:3]:
        assert isinstance(candidate, RoutingCandidate)
        assert candidate.anthropic_route_family == "anthropic_messages"
        assert candidate.reasoning_effort == "max"
        assert candidate.tui_attached == "Claude"
    luna = entries[-1]
    assert isinstance(luna, RoutingCandidate)
    assert luna.reasoning_effort == "max"


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
