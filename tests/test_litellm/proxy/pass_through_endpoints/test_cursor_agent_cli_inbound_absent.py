"""Inbound Cursor Agent CLI Connect surface is not registered.

The host ``cursoral`` / ``cursoralt`` / ``cursorala`` launchers aim
``cursor-agent --agent-endpoint`` at LiteLLM so the default HTTP/2 turn
(``POST /agent.v1.AgentService/Run``) would land on the named container.

That path is not a Cloud Agents ``/cursor/{endpoint:path}`` pass-through
and is not the outbound ``cursor_agent`` provider. This test drives the
shipped passthrough router: Cloud Agents ``/cursor`` stays registered,
and no inbound Agent CLI Connect route exists. A new inbound endpoint
is a later CURSOR queue item, not this launcher goal.
"""

from __future__ import annotations

from litellm.proxy.pass_through_endpoints import llm_passthrough_endpoints as lpe


def _route_paths() -> set[str]:
    return {route.path for route in lpe.router.routes}


def test_cloud_agents_cursor_passthrough_is_registered() -> None:
    """``/cursor/{endpoint:path}`` is Cloud Agents (api.cursor.com), not Agent CLI."""
    assert "/cursor/{endpoint:path}" in _route_paths()


def test_inbound_agent_cli_connect_run_is_not_registered() -> None:
    """Default CLI turn path is absent on the inbound FastAPI router."""
    paths = _route_paths()
    assert "/agent.v1.AgentService/Run" not in paths
    assert "/agent.v1.AgentService/RunSSE" not in paths
    assert "/aiserver.v1.BidiService/BidiAppend" not in paths
    assert not any("AgentService" in path for path in paths)
    assert not any(path.startswith("/agent.v1.") for path in paths)


def test_cursor_passthrough_handler_is_cloud_agents_not_agent_cli() -> None:
    """The registered /cursor handler documents Cloud Agents /v0, not Connect Run."""
    handler = lpe.cursor_proxy_route
    doc = handler.__doc__ or ""
    assert "Cloud Agents" in doc
    assert "/v0/agents" in doc
    assert "AgentService" not in doc
    assert "agentn" not in doc
