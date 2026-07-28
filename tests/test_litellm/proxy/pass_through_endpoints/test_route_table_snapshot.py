"""Wave 0 guardrail: route-table snapshot for ``llm_passthrough_endpoints.router``.

Guards Waves 4-7 (god-module decomposition) against accidental changes to the
registered route paths/methods/endpoint-owning-module, per
``.analysis/plan-godmodule-decomposition-r3-remediation-2026-07-23.md``
Wave 0. The router is declared at ``llm_passthrough_endpoints.py:1218`` and
included into the proxy app at ``proxy_server.py:13525``.
"""

from __future__ import annotations

from litellm.proxy.pass_through_endpoints import llm_passthrough_endpoints as lpe

# Snapshot generated on develop @ e3dc89f634a61e89aeaab98c7fbf91b7bdae896c via:
#   {(route.path, tuple(sorted(route.methods)), route.endpoint.__module__)
#    for route in lpe.router.routes}
_EXPECTED_ROUTES: frozenset[tuple[str, tuple[str, ...], str]] = frozenset(
    {
        (
            "/aawm/alias-config/refresh",
            ("POST",),
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints",
        ),
        (
            "/anthropic",
            ("HEAD",),
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints",
        ),
        (
            "/anthropic/",
            ("HEAD",),
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints",
        ),
        (
            "/anthropic/{endpoint:path}",
            ("DELETE", "GET", "PATCH", "POST", "PUT"),
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints",
        ),
        (
            "/assemblyai/{endpoint:path}",
            ("DELETE", "GET", "PATCH", "POST", "PUT"),
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints",
        ),
        (
            "/azure/{endpoint:path}",
            ("DELETE", "GET", "PATCH", "POST", "PUT"),
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints",
        ),
        (
            "/azure_ai/{endpoint:path}",
            ("DELETE", "GET", "PATCH", "POST", "PUT"),
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints",
        ),
        (
            "/bedrock/{endpoint:path}",
            ("DELETE", "GET", "PATCH", "POST", "PUT"),
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints",
        ),
        (
            "/cohere/{endpoint:path}",
            ("DELETE", "GET", "PATCH", "POST", "PUT"),
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints",
        ),
        (
            "/cursor/{endpoint:path}",
            ("DELETE", "GET", "PATCH", "POST", "PUT"),
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints",
        ),
        (
            "/eu.assemblyai/{endpoint:path}",
            ("DELETE", "GET", "PATCH", "POST", "PUT"),
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints",
        ),
        (
            "/gemini/{endpoint:path}",
            ("DELETE", "GET", "PATCH", "POST", "PUT"),
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints",
        ),
        (
            "/grok/{endpoint:path}",
            ("DELETE", "GET", "PATCH", "POST", "PUT"),
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints",
        ),
        (
            "/milvus/{endpoint:path}",
            ("DELETE", "GET", "PATCH", "POST", "PUT"),
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints",
        ),
        (
            "/mistral/{endpoint:path}",
            ("DELETE", "GET", "PATCH", "POST", "PUT"),
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints",
        ),
        (
            "/openai/{endpoint:path}",
            ("DELETE", "GET", "PATCH", "POST", "PUT"),
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints",
        ),
        (
            "/openai_passthrough/{endpoint:path}",
            ("DELETE", "GET", "PATCH", "POST", "PUT"),
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints",
        ),
        (
            "/opencode/{endpoint:path}",
            ("DELETE", "GET", "PATCH", "POST", "PUT"),
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints",
        ),
        (
            "/vertex-ai/{endpoint:path}",
            ("DELETE", "GET", "PATCH", "POST", "PUT"),
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints",
        ),
        (
            "/vertex_ai/discovery/{endpoint:path}",
            ("DELETE", "GET", "PATCH", "POST", "PUT"),
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints",
        ),
        (
            "/vertex_ai/{endpoint:path}",
            ("DELETE", "GET", "PATCH", "POST", "PUT"),
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints",
        ),
        (
            "/vllm/{endpoint:path}",
            ("DELETE", "GET", "PATCH", "POST", "PUT"),
            "litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints",
        ),
    }
)


def test_passthrough_router_paths_and_methods_stable() -> None:
    """The router's ``(path, methods, endpoint module)`` set must not drift silently."""
    observed = frozenset(
        (route.path, tuple(sorted(route.methods)), route.endpoint.__module__) for route in lpe.router.routes
    )
    assert observed == _EXPECTED_ROUTES, (
        "Passthrough router route table changed. If this is an intentional "
        "route registration change, update _EXPECTED_ROUTES deliberately; "
        f"missing={_EXPECTED_ROUTES - observed} extra={observed - _EXPECTED_ROUTES}"
    )
