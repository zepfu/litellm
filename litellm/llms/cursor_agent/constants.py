"""Stdlib Cursor Agent constants shared by sidecar and provider modules."""

from __future__ import annotations

CURSOR_AGENT_PROVIDER = "cursor_agent"
CLOUD_AGENTS_PROVIDER = "cursor"
CURSOR_AGENT_TURN_HOST = "https://agentn.global.api5.cursor.sh"
CURSOR_AGENT_DASHBOARD_HOST = "https://api2.cursor.sh"
CURSOR_AGENT_RUN_PATH = "/agent.v1.AgentService/Run"
CURSOR_AGENT_USAGE_PATH = "/aiserver.v1.DashboardService/GetCurrentPeriodUsage"
CURSOR_AGENT_AUTH_EXCHANGE_PATH = "/auth/exchange_user_api_key"
CURSOR_AGENT_CLIENT_VERSION = "2026.08.11-e8db854"
CURSOR_AGENT_CONNECT_CONTENT_TYPE = "application/connect+proto"
CURSOR_AGENT_CONNECT_MAX_FRAME_BYTES = 16 * 1024 * 1024
CURSOR_AGENT_AUTH_REFRESH_SECONDS = 300
CURSOR_API_KEY_ENV = "CURSOR_API_KEY"
CURSOR_AUTH_TOKEN_ENV = "CURSOR_AUTH_TOKEN"
CURSOR_CLI_KEY_ENV = "CURSOR_CLI_KEY"
CURSOR_AGENT_AUTH_FILE_ENV = "LITELLM_CURSOR_AGENT_AUTH_FILE"

# Cursor CLI catalog selectors can resolve to a backend model plus parameters.
# Keep this provider-local so aliases can continue exposing the native selector.
CURSOR_AGENT_REQUESTED_MODEL_OVERRIDES = {
    "cursor-grok-4.6-high": {
        "modelId": "grok-4.6",
        "parameters": [
            {"id": "effort", "value": "high"},
            {"id": "fast", "value": "false"},
        ],
    },
}
