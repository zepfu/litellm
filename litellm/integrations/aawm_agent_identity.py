"""AAWM Agent Identity Callback for Langfuse attribution.

Extracts agent identity from the SubagentStart hook context injected into
request prompts, then enriches the langfuse_trace_name request header so
each agent's API calls can be distinguished in Langfuse.

The hook injects: "You are '<agent-name>' and you are working..."
When no agent designation is found, defaults to "orchestrator".

Enriches langfuse_trace_name from "claude-code" to "claude-code.<agent>"
(e.g. "claude-code.ops").

Uses BOTH logging_hook() (sync) and async_logging_hook() (async) to modify
headers BEFORE Langfuse's add_metadata_from_header() reads them.  The sync
hook is critical for pass-through endpoints because Langfuse runs as a string
callback ("langfuse") in the sync success_handler — the async hook alone
would race with the thread-pool-submitted sync handler.

Registration in litellm-config.yaml:
    litellm_settings:
      callbacks: ["aawm_litellm_callbacks.agent_identity.AawmAgentIdentity"]
      success_callback: ["langfuse"]
"""

import re
from typing import Any, Dict, List, Tuple

from litellm._logging import verbose_logger
from litellm.integrations.custom_logger import CustomLogger

_AGENT_RE = re.compile(r"You are '([^']+)' and you are working")
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

    Checks four sources:
    1. kwargs["messages"] — system messages only (standard LLM call path).
    2. kwargs["system"] — Anthropic pass-through transforms system to top-level kwarg.
    3. kwargs["passthrough_logging_payload"]["request_body"]["system"] — pass-through system field.
    4. kwargs["passthrough_logging_payload"]["request_body"]["messages"] — first user message.
       Claude Code SubagentStart hook injects identity into the task prompt (first user message),
       not the system prompt. The tightened regex avoids false positives.

    Returns:
        The agent name if found, otherwise _DEFAULT_AGENT ("orchestrator").
    """
    # --- Standard messages path (system messages only) ---
    messages = kwargs.get("messages")
    if messages and isinstance(messages, list):
        for message in messages:
            if not isinstance(message, dict):
                continue
            if message.get("role") != "system":
                continue
            text = _content_to_text(message.get("content", ""))
            match = _AGENT_RE.search(text)
            if match:
                return match.group(1)

    # --- Direct system field (Anthropic pass-through transforms system to top-level kwarg) ---
    system_direct = kwargs.get("system")
    if system_direct:
        text = _content_to_text(system_direct)
        match = _AGENT_RE.search(text)
        if match:
            return match.group(1)

    # --- Pass-through path (system field) ---
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

            # Check first user message — Claude Code injects agent identity
            # into the task prompt (first user message), not the system prompt.
            pt_messages = request_body.get("messages")
            if pt_messages and isinstance(pt_messages, list):
                for msg in pt_messages[:3]:
                    if not isinstance(msg, dict):
                        continue
                    if msg.get("role") != "user":
                        continue
                    text = _content_to_text(msg.get("content", ""))
                    match = _AGENT_RE.search(text)
                    if match:
                        return match.group(1)
                    break  # Only check first user message

    return _DEFAULT_AGENT


def _ensure_mutable_headers(kwargs: Dict[str, Any]) -> dict:
    """Ensure proxy_server_request.headers is a mutable dict.

    Starlette Headers objects are immutable. This converts them to a plain
    dict in-place on kwargs so subsequent modifications persist.

    Returns the mutable headers dict, or empty dict if no headers found.
    """
    litellm_params = kwargs.get("litellm_params") or {}
    psr = litellm_params.get("proxy_server_request") or {}
    headers = psr.get("headers")

    if headers is None:
        return {}

    # Convert immutable Starlette Headers (or any non-dict mapping) to dict
    if not isinstance(headers, dict):
        headers = dict(headers)
        psr["headers"] = headers

    return headers


def _enrich_trace_name(kwargs: Dict[str, Any], result: Any) -> Tuple[dict, Any]:
    """Shared enrichment logic for both sync and async hooks.

    Extracts agent name, converts headers to mutable dict, and enriches
    langfuse_trace_name header from "claude-code" to "claude-code.<agent>".
    Falls back to setting metadata directly if headers are unavailable.

    Returns (kwargs, result) tuple as required by logging hook contract.
    """
    agent_name = _extract_agent_name(kwargs)
    headers = _ensure_mutable_headers(kwargs)

    if headers:
        current = headers.get("langfuse_trace_name")
        if current == "claude-code":
            headers["langfuse_trace_name"] = f"claude-code.{agent_name}"
            verbose_logger.debug(
                "AawmAgentIdentity: enriched header trace_name to claude-code.%s",
                agent_name,
            )
            return kwargs, result

    # Fallback: set metadata directly
    litellm_params = kwargs.get("litellm_params") or {}
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
    return kwargs, result


class AawmAgentIdentity(CustomLogger):
    """CustomLogger that enriches Langfuse trace_name with agent identity.

    Implements both sync logging_hook() and async async_logging_hook() to
    cover all code paths:
    - Sync: pass-through endpoints run Langfuse in sync success_handler (thread pool)
    - Async: standard LLM calls run Langfuse in async_success_handler
    """

    def logging_hook(
        self, kwargs: Dict[str, Any], result: Any, call_type: str
    ) -> Tuple[dict, Any]:
        """Sync hook — runs in success_handler Loop 1, BEFORE Langfuse Loop 2.

        Critical for pass-through endpoints where Langfuse is a string callback
        processed in the sync success_handler thread pool.
        """
        try:
            return _enrich_trace_name(kwargs, result)
        except Exception as exc:
            verbose_logger.warning(
                "AawmAgentIdentity.logging_hook failed: %s", exc
            )
            return kwargs, result

    async def async_logging_hook(
        self, kwargs: Dict[str, Any], result: Any, call_type: str
    ) -> Tuple[dict, Any]:
        """Async hook — runs in async_success_handler Loop 1, BEFORE async Loop 2."""
        try:
            return _enrich_trace_name(kwargs, result)
        except Exception as exc:
            verbose_logger.warning(
                "AawmAgentIdentity.async_logging_hook failed: %s", exc
            )
            return kwargs, result


# Module-level instance for config registration via get_instance_fn().
# Config must reference this instance name, not the class name:
#   callbacks: ["litellm.integrations.aawm_agent_identity.aawm_agent_identity_instance"]
aawm_agent_identity_instance = AawmAgentIdentity()
