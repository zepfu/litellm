"""AAWM Agent Identity Callback for Langfuse attribution.

Extracts agent identity from the SubagentStart hook context injected into
request prompts, then enriches the Langfuse trace_name so each agent's
API calls can be distinguished.

The hook injects: "You are '<agent-name>' and you are working..."
When no agent designation is found, defaults to "orchestrator".

Enriches the langfuse_trace_name request header from "claude-code" to
"claude-code.<agent>" (e.g. "claude-code.ops") so the value survives
Langfuse's add_metadata_from_header() which overwrites metadata from headers.

Registration in litellm-config.yaml:
    litellm_settings:
      callbacks: ["litellm.integrations.aawm_agent_identity.AawmAgentIdentity"]
      success_callback: ["langfuse"]
"""

import re
from typing import Any, Dict, List

from litellm._logging import verbose_logger
from litellm.integrations.custom_logger import CustomLogger

_AGENT_RE = re.compile(r"You are '([^']+)'")
_DEFAULT_AGENT = "orchestrator"


def _content_to_text(content: Any) -> str:
    """Convert message content (string or Anthropic content blocks) to plain text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for block in content:
            if isinstance(block, dict):
                parts.append(str(block.get("text", "")))
            else:
                parts.append(str(block))
        return "\n".join(parts)
    return str(content) if content else ""


def _extract_agent_name(kwargs: Dict[str, Any]) -> str:
    """Extract agent name from request content.

    Checks two sources:
    1. kwargs["messages"] — standard LLM call path (scans all messages).
    2. kwargs["passthrough_logging_payload"]["request_body"] — pass-through
       endpoint path. Checks both "system" (top-level Anthropic field) and
       "messages" within the request body.

    Returns:
        The agent name if found, otherwise _DEFAULT_AGENT ("orchestrator").
    """
    # --- Standard messages path ---
    messages = kwargs.get("messages")
    if messages and isinstance(messages, list):
        for message in messages:
            if not isinstance(message, dict):
                continue
            text = _content_to_text(message.get("content", ""))
            match = _AGENT_RE.search(text)
            if match:
                return match.group(1)

    # --- Pass-through path ---
    payload = kwargs.get("passthrough_logging_payload")
    if isinstance(payload, dict):
        request_body = payload.get("request_body")
        if isinstance(request_body, dict):
            # Check top-level "system" field (Anthropic format)
            system = request_body.get("system")
            if system:
                text = _content_to_text(system)
                match = _AGENT_RE.search(text)
                if match:
                    return match.group(1)
            # Check messages within request body
            msgs = request_body.get("messages")
            if msgs and isinstance(msgs, list):
                for msg in msgs:
                    if not isinstance(msg, dict):
                        continue
                    text = _content_to_text(msg.get("content", ""))
                    match = _AGENT_RE.search(text)
                    if match:
                        return match.group(1)

    return _DEFAULT_AGENT


def _process_kwargs(kwargs: Dict[str, Any]) -> None:
    """Extract agent name and enrich trace_name in request headers.

    Modifies the langfuse_trace_name header directly so the enriched value
    survives Langfuse's add_metadata_from_header() which overwrites metadata
    from headers.
    """
    agent_name = _extract_agent_name(kwargs)

    # Find the headers dict — try litellm_params.proxy_server_request.headers first,
    # then passthrough_logging_payload.request_headers
    headers = None
    litellm_params = kwargs.get("litellm_params") or {}
    proxy_server_request = litellm_params.get("proxy_server_request") or {}
    headers = proxy_server_request.get("headers")

    if not isinstance(headers, dict):
        # Try passthrough_logging_payload
        payload = kwargs.get("passthrough_logging_payload")
        if isinstance(payload, dict):
            headers = payload.get("request_headers")

    if isinstance(headers, dict):
        current = headers.get("langfuse_trace_name")
        if current == "claude-code":
            headers["langfuse_trace_name"] = f"claude-code.{agent_name}"
            verbose_logger.debug(
                "AawmAgentIdentity: enriched header trace_name to claude-code.%s",
                agent_name,
            )
            return

    # Fallback: set metadata directly (for non-header cases)
    metadata: Dict[str, Any] = litellm_params.get("metadata") or {}
    current_trace_name = metadata.get("trace_name")
    if current_trace_name == "claude-code":
        metadata["trace_name"] = f"claude-code.{agent_name}"
    elif not current_trace_name:
        metadata["trace_name"] = agent_name
    litellm_params["metadata"] = metadata
    kwargs["litellm_params"] = litellm_params

    verbose_logger.debug(
        "AawmAgentIdentity: agent=%s, trace_name=%s",
        agent_name,
        metadata.get("trace_name"),
    )


class AawmAgentIdentity(CustomLogger):
    """CustomLogger that enriches Langfuse trace_name with agent identity.

    Parses agent name from SubagentStart hook context in the request prompt.
    Defaults to "orchestrator" when no agent designation is found.

    Registration:
        litellm_settings:
          callbacks: ["litellm.integrations.aawm_agent_identity.AawmAgentIdentity"]
          success_callback: ["langfuse"]
    """

    def log_success_event(
        self,
        kwargs: Dict[str, Any],
        response_obj: Any,
        start_time: Any,
        end_time: Any,
    ) -> None:
        try:
            _process_kwargs(kwargs)
        except Exception as exc:
            verbose_logger.warning(
                "AawmAgentIdentity.log_success_event failed: %s", exc
            )

    async def async_log_success_event(
        self,
        kwargs: Dict[str, Any],
        response_obj: Any,
        start_time: Any,
        end_time: Any,
    ) -> None:
        try:
            _process_kwargs(kwargs)
        except Exception as exc:
            verbose_logger.warning(
                "AawmAgentIdentity.async_log_success_event failed: %s", exc
            )
