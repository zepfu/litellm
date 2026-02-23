"""AAWM Agent Identity Callback for Langfuse attribution.

This CustomLogger extracts agent/tenant/task identity markers injected into
LLM request system prompts by AAWM's context hook, then maps them to
Langfuse metadata fields so cost and usage can be attributed per agent,
tenant, and task.

Markers injected into system prompts by AAWM:
    LF_AGENT: <agent-name>        → metadata["trace_name"]  (Langfuse trace name)
    LF_TENANT: <project/repo>     → metadata["trace_user_id"] (Langfuse user_id)
    LF_TASK_IDS: <id1/id2/...>   → metadata["session_id"]  (Langfuse session ID)

Registration in litellm-config.yaml:
    litellm_settings:
      callbacks: ["litellm.integrations.aawm_agent_identity.AawmAgentIdentity"]
      success_callback: ["langfuse"]
"""

import re
from typing import Any, Dict, List, Optional

from litellm._logging import verbose_logger
from litellm.integrations.custom_logger import CustomLogger

# Matches LF_AGENT, LF_TENANT, or LF_TASK_IDS followed by optional whitespace and a value
_MARKER_RE = re.compile(r"LF_(AGENT|TENANT|TASK_IDS):\s*(\S+)")


def _extract_markers(messages: List[Any]) -> Dict[str, str]:
    """Extract AAWM identity markers from system messages.

    Scans all messages with role == "system". The content field may be a
    plain string or a list of content blocks (each a dict with a "text" key).

    Args:
        messages: The messages list from kwargs["messages"].

    Returns:
        Dict mapping marker names ("AGENT", "TENANT", "TASK_IDS") to their
        extracted values. Only keys that were found are included.
    """
    found: Dict[str, str] = {}
    for message in messages:
        if not isinstance(message, dict):
            continue
        if message.get("role") != "system":
            continue

        content = message.get("content", "")
        if isinstance(content, list):
            # content blocks: [{"type": "text", "text": "..."}, ...]
            text_parts: List[str] = []
            for block in content:
                if isinstance(block, dict):
                    text_parts.append(str(block.get("text", "")))
                else:
                    text_parts.append(str(block))
            content = "\n".join(text_parts)
        elif not isinstance(content, str):
            content = str(content)

        for match in _MARKER_RE.finditer(content):
            marker_key = match.group(1)   # "AGENT", "TENANT", or "TASK_IDS"
            marker_val = match.group(2)
            if marker_key not in found:
                # Take the first occurrence of each marker
                found[marker_key] = marker_val

        if len(found) == 3:
            # All three markers found; no need to scan further
            break

    return found


def _apply_markers_to_metadata(
    markers: Dict[str, str],
    metadata: Dict[str, Any],
) -> None:
    """Write extracted markers into the metadata dict.

    Only sets a key when:
    - The corresponding marker was found in the system prompt.
    - The key does not already have a value in metadata (user-specified
      values take precedence).

    Mapping:
        LF_AGENT    → metadata["trace_name"]
        LF_TENANT   → metadata["trace_user_id"]
        LF_TASK_IDS → metadata["session_id"]

    Args:
        markers: Dict returned by _extract_markers().
        metadata: The metadata dict from kwargs["litellm_params"]["metadata"].
            Modified in place.
    """
    mapping = {
        "AGENT": "trace_name",
        "TENANT": "trace_user_id",
        "TASK_IDS": "session_id",
    }
    for marker_key, meta_key in mapping.items():
        if marker_key in markers and not metadata.get(meta_key):
            metadata[meta_key] = markers[marker_key]


def _process_kwargs(kwargs: Dict[str, Any]) -> None:
    """Core logic: extract markers from kwargs and inject into metadata.

    Args:
        kwargs: The kwargs dict passed to log_success_event /
            async_log_success_event.
    """
    messages = kwargs.get("messages")
    if not messages or not isinstance(messages, list):
        return

    markers = _extract_markers(messages)
    if not markers:
        verbose_logger.debug(
            "AawmAgentIdentity: no LF_ markers found in system prompt"
        )
        return

    litellm_params: Dict[str, Any] = kwargs.get("litellm_params") or {}
    metadata: Dict[str, Any] = litellm_params.get("metadata") or {}

    _apply_markers_to_metadata(markers, metadata)

    # Write back in case metadata was None / not yet in litellm_params
    litellm_params["metadata"] = metadata
    kwargs["litellm_params"] = litellm_params

    verbose_logger.debug(
        "AawmAgentIdentity: applied markers %s to metadata", markers
    )


class AawmAgentIdentity(CustomLogger):
    """CustomLogger that injects AAWM agent identity into Langfuse metadata.

    Reads LF_AGENT / LF_TENANT / LF_TASK_IDS markers from system prompts
    and sets the corresponding Langfuse metadata keys before the Langfuse
    success callback runs.

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
        """Synchronous success hook — extract markers and set metadata."""
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
        """Asynchronous success hook — extract markers and set metadata."""
        try:
            _process_kwargs(kwargs)
        except Exception as exc:
            verbose_logger.warning(
                "AawmAgentIdentity.async_log_success_event failed: %s", exc
            )
