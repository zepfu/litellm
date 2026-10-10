"""CFG-047 Codex spawn_agent message contract for harness prompts.

The native outer ``message`` argument must be a string. That string is
serialized exact-three-key JSON: ``cfg047``, ``encoding``, and ``text``.
This module does not decrypt or normalize assignments.
"""

from __future__ import annotations

import json
from typing import Any, Mapping

_FRAME_KEYS = frozenset({"cfg047", "encoding", "text"})


class CodexAssignmentError(ValueError):
    """A spawn_agent message is not a strict CFG-047 text string."""


def codex_assignment_message(text: str) -> str:
    """Return the native spawn_agent message string for *text*."""

    frame = {"cfg047": 1, "encoding": "text", "text": text}
    return json.dumps(frame, separators=(",", ":"), ensure_ascii=False)


def assert_codex_assignment_message(message: Any) -> Mapping[str, Any]:
    """Accept only a string that is exact-three-key CFG-047 text JSON.

    A JSON object, bare prose, a different key set, or a non-text encoding
    is rejected. The returned mapping is the parsed frame.
    """

    if isinstance(message, Mapping):
        raise CodexAssignmentError(
            "spawn_agent message must be a string, not a JSON object"
        )
    if not isinstance(message, str) or not message.strip():
        raise CodexAssignmentError(
            "spawn_agent message must be a non-empty CFG-047 string"
        )
    try:
        decoded = json.loads(message)
    except json.JSONDecodeError as exc:
        raise CodexAssignmentError(
            "spawn_agent message must be CFG-047 JSON, not bare prose"
        ) from exc
    if not isinstance(decoded, dict) or set(decoded) != _FRAME_KEYS:
        raise CodexAssignmentError(
            "spawn_agent message must be exact-three-key CFG-047 JSON"
        )
    if decoded.get("cfg047") != 1 or decoded.get("encoding") != "text":
        raise CodexAssignmentError(
            "spawn_agent message must use cfg047=1 and encoding=text"
        )
    assignment = decoded.get("text")
    if not isinstance(assignment, str) or not assignment:
        raise CodexAssignmentError("spawn_agent text must be a non-empty string")
    return decoded
