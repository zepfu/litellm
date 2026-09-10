"""Inbound Cursor Agent CLI Connect surface is registered separately from Cloud Agents.

The host ``cursoral`` / ``cursoralt`` / ``cursorala`` launchers aim
``cursor-agent --agent-endpoint`` at LiteLLM so the default HTTP/2 turn
(``POST /agent.v1.AgentService/Run``) lands on the named container.

Cloud Agents ``/cursor/{endpoint:path}`` stays the ``api.cursor.com`` ``/v0``
product. HTTP/1.1 ``RunSSE`` + ``BidiAppend`` is a later compatibility lane
and is not registered with this HTTP/2 ``Run`` landing. ``RunPoll`` is not
the ``--print`` path.
"""

from __future__ import annotations

from litellm.proxy.pass_through_endpoints import llm_passthrough_endpoints as lpe
from litellm.proxy.pass_through_endpoints.cursor_agent_cli_inbound import (
    CURSOR_AGENT_BIDI_APPEND_PATH,
    CURSOR_AGENT_RUNSSE_PATH,
)


def _route_paths() -> set[str]:
    return {route.path for route in lpe.router.routes}


def test_cloud_agents_cursor_passthrough_is_registered() -> None:
    """``/cursor/{endpoint:path}`` is Cloud Agents (api.cursor.com), not Agent CLI."""
    assert "/cursor/{endpoint:path}" in _route_paths()


def test_inbound_agent_cli_connect_run_is_registered() -> None:
    """Default CLI turn path is registered on the inbound FastAPI router."""
    paths = _route_paths()
    assert "/agent.v1.AgentService/Run" in paths
    assert CURSOR_AGENT_RUNSSE_PATH not in paths
    assert CURSOR_AGENT_BIDI_APPEND_PATH not in paths
    assert "/agent.v1.AgentService/RunPoll" not in paths


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
