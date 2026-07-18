"""Structured/configurable task-state preservation contract (RR-054 #35).

Avoids single-source English-literal coupling by:
1. Preferring an explicit structured metadata flag on messages when present
2. Allowing env-configured marker phrases (comma/newline separated)
3. Falling back to a versioned default marker set
"""

from __future__ import annotations

import os
import re
from typing import Callable, Optional, Sequence

from .types import Payload

DEFAULT_TASK_STATE_MARKERS: tuple[str, ...] = (
    "Run this numbered script",
    "numbered script",
    "next and only valid tool call",
    "A final response immediately after Bash is invalid",
    "After WebFetch",
    "Your team's answer",
    "continue the task",
    "do not stop until",
    "acceptance criteria",
    "Current user question",
    "IMPORTANT:",
    "You must",
)

TASK_STATE_MARKERS_ENV = "AAWM_GOOGLE_ADAPTER_TASK_STATE_MARKERS"
# Structured message metadata keys that force selection without English matching.
STRUCTURED_TASK_STATE_FLAGS = (
    "aawm_preserve_task_state",
    "preserve_task_state",
    "aawm_task_state",
    "task_state",
)


def resolve_task_state_markers(
    env_value: Optional[str] = None,
    *,
    defaults: Sequence[str] = DEFAULT_TASK_STATE_MARKERS,
) -> tuple[str, ...]:
    raw = env_value
    if raw is None:
        raw = os.getenv(TASK_STATE_MARKERS_ENV)
    if raw is None or not str(raw).strip():
        return tuple(defaults)
    parts = re.split(r"[\n,|]+", str(raw))
    markers = tuple(part.strip() for part in parts if part and part.strip())
    return markers or tuple(defaults)


def message_has_structured_task_state_flag(message: Payload) -> bool:
    metadata = message.get("metadata")
    if isinstance(metadata, dict):
        for key in STRUCTURED_TASK_STATE_FLAGS:
            if metadata.get(key) in (True, 1, "1", "true", "True", "yes"):
                return True
    for key in STRUCTURED_TASK_STATE_FLAGS:
        if message.get(key) in (True, 1, "1", "true", "True", "yes"):
            return True
    return False


def select_task_state_source(
    messages: list[Payload],
    *,
    extract_text: Callable[[Payload], str],
    is_skippable: Callable[[Payload], bool],
    markers: Optional[Sequence[str]] = None,
) -> Optional[tuple[int, str, str]]:
    """Return (index, text, source_kind) or None.

    source_kind is ``structured``, ``marker``, or ``fallback``.
    """
    active_markers = tuple(markers) if markers is not None else resolve_task_state_markers()
    fallback: Optional[tuple[int, str]] = None
    for index, message in enumerate(messages):
        if not isinstance(message, dict) or is_skippable(message):
            continue
        text = extract_text(message).strip()
        if not text:
            continue
        if fallback is None:
            fallback = (index, text)
        if message_has_structured_task_state_flag(message):
            return index, text, "structured"
        if any(marker in text for marker in active_markers):
            return index, text, "marker"
    if fallback is None:
        return None
    return fallback[0], fallback[1], "fallback"
