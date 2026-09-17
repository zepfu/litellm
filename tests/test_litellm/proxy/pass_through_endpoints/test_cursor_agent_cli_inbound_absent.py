"""Inbound Cursor Agent CLI Connect surface is registered separately from Cloud Agents.

The host ``cursoral`` / ``cursoralt`` / ``cursorala`` launchers aim
``cursor-agent --agent-endpoint`` at LiteLLM so the default HTTP/2 turn
(``POST /agent.v1.AgentService/Run``) lands on the named container.

Cloud Agents ``/cursor/{endpoint:path}`` stays the ``api.cursor.com`` ``/v0``
product. HTTP/1.1 ``RunSSE`` + ``BidiAppend`` is the CLI compatibility lane
and is registered without claiming ``RunPoll`` as the ``--print`` path.
"""

from __future__ import annotations

from pathlib import Path

from litellm.proxy.pass_through_endpoints import llm_passthrough_endpoints as lpe
from litellm.proxy.pass_through_endpoints.cursor_agent_cli_inbound import (
    CURSOR_AGENT_BIDI_APPEND_PATH,
    CURSOR_AGENT_RUNSSE_PATH,
)

REPO_ROOT = Path(__file__).resolve().parents[4]


def _route_paths() -> set[str]:
    return {route.path for route in lpe.router.routes}


def test_cloud_agents_cursor_passthrough_is_registered() -> None:
    """``/cursor/{endpoint:path}`` is Cloud Agents (api.cursor.com), not Agent CLI."""
    assert "/cursor/{endpoint:path}" in _route_paths()


def test_inbound_agent_cli_connect_run_is_registered() -> None:
    """Default CLI turn path is registered on the inbound FastAPI router."""
    paths = _route_paths()
    assert "/agent.v1.AgentService/Run" in paths
    assert CURSOR_AGENT_RUNSSE_PATH in paths
    assert CURSOR_AGENT_BIDI_APPEND_PATH in paths
    assert "/agent.v1.AgentService/RunPoll" not in paths


def test_inbound_http1_handlers_do_not_claim_runpoll() -> None:
    runsse = lpe.cursor_agent_cli_runsse_route
    bidi = lpe.cursor_agent_cli_bidi_append_route
    for handler in (runsse, bidi):
        doc = handler.__doc__ or ""
        assert "HTTP/1.1" in doc
        assert "RunPoll" in doc
        assert "/v0/agents" not in doc


def test_cursor_passthrough_handler_is_cloud_agents_not_agent_cli() -> None:
    """The registered /cursor handler documents Cloud Agents /v0, not Connect Run."""
    handler = lpe.cursor_proxy_route
    doc = handler.__doc__ or ""
    assert "Cloud Agents" in doc
    assert "/v0/agents" in doc
    assert "AgentService" not in doc
    assert "agentn" not in doc


def test_inbound_run_handler_is_not_cloud_agents() -> None:
    handler = lpe.cursor_agent_cli_run_route
    doc = handler.__doc__ or ""
    assert "Connect" in doc
    assert "cursoral" in doc
    assert "--agent-endpoint" in doc
    assert "/v0/agents" not in doc


def test_inbound_run_does_not_collide_with_cloud_agents_cursor() -> None:
    """Inbound Agent CLI Run is a distinct FastAPI route from Cloud Agents /cursor."""
    run = lpe.cursor_agent_cli_run_route
    cursor = lpe.cursor_proxy_route
    assert run is not cursor
    assert run.__name__ == "cursor_agent_cli_run_route"
    assert cursor.__name__ == "cursor_proxy_route"
    methods_by_path = {
        route.path: set(route.methods or ()) for route in lpe.router.routes
    }
    assert "POST" in methods_by_path["/agent.v1.AgentService/Run"]
    assert "POST" in methods_by_path["/cursor/{endpoint:path}"]
    assert "/cursor/{endpoint:path}" != "/agent.v1.AgentService/Run"


def test_provider_docs_name_agent_endpoint_not_dashboard_override() -> None:
    docs = (REPO_ROOT / "docs/my-website/docs/providers/cursor_agent.md").read_text(
        encoding="utf-8"
    )
    assert "--agent-endpoint" in docs
    assert "CURSOR_API_ENDPOINT" in docs
    assert "do not invent" in docs.lower() or "no environment twin" in docs.lower()
    assert "http2_required" in docs
    assert "cursor_agent_cli_inbound" in docs
    assert "CURSOR_AGENT_ENDPOINT" in docs
