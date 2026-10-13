"""AAWM Payload Capture Callback.

Dumps full API request bodies to /tmp/captures/ for offline analysis of
system prompts, user messages, and tool definitions flowing through the
LiteLLM proxy.

One directory is written per request:
    /tmp/captures/{timestamp}_{counter:04d}_{agent}/
        system.json   — system prompt blocks
        messages.json — user/assistant message array
        tools.json    — tool definitions array
        meta.json     — model, stream, timestamp, agent name, other fields

Enable via AAWM_CAPTURE=1 environment variable, but only when LiteLLM
debug logging is also enabled. Disabled by default.
Truthy values: "1", "true", "True", "yes", "on" (case-insensitive).
Falsy values:  "", "0", "false", "False", "no", "off" — all disabled.

Pass-through path: request body is in
    kwargs["passthrough_logging_payload"]["request_body"]

Provider request path: when LiteLLM retained a provider-transformed logical
request in Logging.model_call_details, write that object to provider_request.json.
It is not the final byte-for-byte httpx request, and its stream flag can change
after Logging.pre_call stores it.

Standard path: reconstructed from kwargs["messages"], kwargs["system"], etc.

Registration in litellm-config.yaml:
    litellm_settings:
      success_callback:
        - "langfuse"
        - "litellm.integrations.aawm_agent_identity.aawm_agent_identity_instance"
        - "litellm.integrations.aawm_payload_capture.aawm_payload_capture_instance"
"""

import asyncio
import json
import os
import re
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from litellm._logging import _is_debugging_on, verbose_logger
from litellm.integrations.custom_logger import CustomLogger


def _is_truthy(value: str) -> bool:
    """Return True only for recognised truthy strings.

    Treats "", "0", "false", "False", "no", "off" as disabled so that
    AAWM_CAPTURE=0 correctly disables capture (unlike a bare truthiness
    check which would treat any non-empty string as enabled).
    """
    return value.strip().lower() in {"1", "true", "yes", "on"}


_CAPTURE_DIR = Path("/tmp/captures")
_AGENT_RE = re.compile(r"You are '([^']+)' and you are working")
_DEFAULT_AGENT = "orchestrator"

# Thread-safe monotonic counter for directory uniqueness within a second.
_counter_lock = threading.Lock()
_counter = 0


def _next_counter() -> int:
    global _counter
    with _counter_lock:
        _counter += 1
        return _counter


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

    Checks four sources in order:
    1. kwargs["messages"] — system-role messages (standard LLM call path).
    2. kwargs["system"] — direct system kwarg.
    3. passthrough_logging_payload request_body["system"].
    4. First user message in request_body["messages"] — Claude Code injects
       agent identity into the task prompt, not the system prompt.

    Returns the agent name if found, otherwise _DEFAULT_AGENT.
    """
    # Standard messages path — system role only
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

    # Direct system kwarg
    system_direct = kwargs.get("system")
    if system_direct:
        text = _content_to_text(system_direct)
        match = _AGENT_RE.search(text)
        if match:
            return match.group(1)

    # Pass-through payload
    payload = kwargs.get("passthrough_logging_payload")
    if isinstance(payload, dict):
        request_body = payload.get("request_body")
        if isinstance(request_body, dict):
            system = request_body.get("system")
            if system:
                text = _content_to_text(system)
                match = _AGENT_RE.search(text)
                if match:
                    return match.group(1)

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


def _get_request_parts(
    kwargs: Dict[str, Any],
) -> Tuple[Any, Optional[List[Any]], Optional[List[Any]]]:
    """Extract (system, messages, tools) from kwargs.

    For pass-through traffic the full body lives in
    kwargs["passthrough_logging_payload"]["request_body"].

    For the standard path the fields are top-level kwargs keys.

    Returns a 3-tuple of (system, messages, tools) where each element
    may be None if absent.
    """
    payload = kwargs.get("passthrough_logging_payload")
    if isinstance(payload, dict):
        request_body = payload.get("request_body")
        if isinstance(request_body, dict):
            return (
                request_body.get("system"),
                request_body.get("messages"),
                request_body.get("tools"),
            )

    # Standard path
    return (
        kwargs.get("system"),
        kwargs.get("messages"),
        kwargs.get("tools"),
    )


def _get_provider_request_body(kwargs: Dict[str, Any]) -> Any:
    """Return the retained provider request body when LiteLLM preserved it.

    The custom HTTP handler stores the provider-transformed logical request in
    ``Logging.model_call_details.additional_args.complete_input_dict`` before
    the success logging hook runs. Some providers and call paths do not supply
    that field; in those cases return ``None`` rather than fabricating a body.
    """
    model_call_details = kwargs.get("model_call_details")
    if not isinstance(model_call_details, dict):
        return None

    additional_args = model_call_details.get("additional_args")
    if not isinstance(additional_args, dict):
        return None

    return additional_args.get("complete_input_dict")


def _build_meta(kwargs: Dict[str, Any], agent: str, ts: str) -> Dict[str, Any]:
    """Build the meta.json payload from kwargs top-level fields."""
    meta: Dict[str, Any] = {
        "timestamp": ts,
        "agent": agent,
    }

    # Pull common top-level fields
    for field in ("model", "stream", "call_type", "litellm_call_id"):
        val = kwargs.get(field)
        if val is not None:
            meta[field] = val

    # From pass-through request body: model, max_tokens, temperature, etc.
    payload = kwargs.get("passthrough_logging_payload")
    if isinstance(payload, dict):
        request_body = payload.get("request_body")
        if isinstance(request_body, dict):
            for field in (
                "model",
                "max_tokens",
                "temperature",
                "top_p",
                "top_k",
                "stream",
                "anthropic_version",
            ):
                val = request_body.get(field)
                if val is not None:
                    meta[field] = val

    if _get_provider_request_body(kwargs) is not None:
        meta["provider_request"] = {
            "artifact": "provider_request.json",
            "source": (
                "Logging.model_call_details.additional_args.complete_input_dict"
            ),
            "captured_at": "Logging.pre_call",
            "request_stage": (
                "provider-transformed logical request after extra_body and signing"
            ),
            "final_httpx_serialization": False,
            "stream_finalized": False,
            "litellm_call_id": kwargs.get("litellm_call_id"),
            "litellm_trace_id": kwargs.get("litellm_trace_id"),
        }

    return meta


def _dump_capture(kwargs: Dict[str, Any]) -> None:
    """Write one capture directory with system/messages/tools/meta JSON files."""
    if not _is_truthy(os.environ.get("AAWM_CAPTURE", "")) or not _is_debugging_on():
        return

    try:
        ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        counter = _next_counter()
        agent = _extract_agent_name(kwargs)
        # Sanitise agent name for filesystem use
        safe_agent = re.sub(r"[^\w\-]", "_", agent)[:40]
        dir_name = f"{ts}_{counter:04d}_{safe_agent}"
        capture_dir = _CAPTURE_DIR / dir_name
        capture_dir.mkdir(parents=True, exist_ok=True)

        system, messages, tools = _get_request_parts(kwargs)
        meta = _build_meta(kwargs, agent, ts)

        def _write(filename: str, data: Any) -> None:
            (capture_dir / filename).write_text(
                json.dumps(data, indent=2, ensure_ascii=False, default=str),
                encoding="utf-8",
            )

        _write("system.json", system)
        _write("messages.json", messages)
        _write("tools.json", tools)
        _write("meta.json", meta)

        provider_request_body = _get_provider_request_body(kwargs)
        if provider_request_body is not None:
            _write("provider_request.json", provider_request_body)

        tool_count = len(tools) if isinstance(tools, list) else 0
        msg_count = len(messages) if isinstance(messages, list) else 0
        verbose_logger.info(
            "AawmPayloadCapture: wrote %s (agent=%s, msgs=%d, tools=%d)",
            capture_dir,
            agent,
            msg_count,
            tool_count,
        )
    except Exception as exc:
        verbose_logger.warning("AawmPayloadCapture: capture failed: %s", exc)


class AawmPayloadCapture(CustomLogger):
    """CustomLogger that dumps full request payloads to /tmp/captures/.

    Implements both sync logging_hook() and async async_logging_hook() to
    cover all code paths — same pattern as AawmAgentIdentity.
    """

    def logging_hook(
        self, kwargs: Dict[str, Any], result: Any, call_type: str
    ) -> Tuple[dict, Any]:
        """Sync hook — captures payload, passes kwargs/result through unchanged."""
        _dump_capture(kwargs)
        return kwargs, result

    async def async_logging_hook(
        self, kwargs: Dict[str, Any], result: Any, call_type: str
    ) -> Tuple[dict, Any]:
        """Async hook — offload filesystem capture so the event loop stays free.

        Gating remains inside ``_dump_capture`` (AAWM_CAPTURE + debug logging).
        """
        await asyncio.to_thread(_dump_capture, kwargs)
        return kwargs, result


# Module-level instance for config registration.
# Config must reference this instance name:
#   success_callback:
#     - "litellm.integrations.aawm_payload_capture.aawm_payload_capture_instance"
aawm_payload_capture_instance = AawmPayloadCapture()
