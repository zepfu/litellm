"""Data classes for the prompt fragment analyzer pipeline."""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class FragmentType(str, Enum):
    SYSTEM_PROMPT = "system-prompt"
    SYSTEM_DATA = "system-data"
    SYSTEM_REMINDER = "system-reminder"
    TOOL_DESCRIPTION = "tool-description"
    AGENT_PROMPT = "agent-prompt"
    SKILL = "skill"
    UNKNOWN = "unknown"


class TriggerType(str, Enum):
    SESSION_START = "session_start"
    AGENT_DISPATCH = "agent_dispatch"
    CONTEXT_INJECTION = "context_injection"
    TOOL_LOAD = "tool_load"
    TOOL_SEARCH_RESULT = "tool_search_result"
    MODE_CHANGE = "mode_change"
    TOOL_RESULT = "tool_result"
    USER_MESSAGE = "user_message"
    UNKNOWN = "unknown"


class ContentSource(str, Enum):
    SYSTEM = "system"
    MESSAGES = "messages"
    TOOLS = "tools"


# Patterns used for fingerprint normalization
_TEMPLATE_VAR_PATTERN = re.compile(r"\$\{[^}]+\}")
_UUID_PATTERN = re.compile(
    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", re.IGNORECASE
)
_TIMESTAMP_PATTERN = re.compile(
    r"\d{4}[-/]\d{2}[-/]\d{2}[T ]\d{2}:\d{2}:\d{2}(?:[Z.]\S*)?"
)
_DATE_PATTERN = re.compile(r"\b\d{4}[-/]\d{2}[-/]\d{2}\b")
_LONG_NUMERIC_PATTERN = re.compile(r"\b\d{8,}\b")
_FILE_PATH_PATTERN = re.compile(r"(?:/[\w.\-]+){2,}|(?:[A-Za-z]:\\[\w.\\\-]+)")
_WHITESPACE_PATTERN = re.compile(r"\s+")


def _normalize_text(text: str) -> str:
    """Normalize text for fingerprinting by stripping variable content."""
    text = _TEMPLATE_VAR_PATTERN.sub("<VAR>", text)
    text = _UUID_PATTERN.sub("<UUID>", text)
    text = _TIMESTAMP_PATTERN.sub("<TS>", text)
    text = _DATE_PATTERN.sub("<DATE>", text)
    text = _LONG_NUMERIC_PATTERN.sub("<NUM>", text)
    text = _FILE_PATH_PATTERN.sub("<PATH>", text)
    text = _WHITESPACE_PATTERN.sub(" ", text)
    return text.strip().lower()


def fingerprint(text: str) -> str:
    """SHA-256 of whitespace-normalized, template-stripped text."""
    normalized = _normalize_text(text)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def content_hash(text: str) -> str:
    """SHA-256 of exact text (no normalization) for dedup."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass
class RequestSnapshot:
    """One captured API request."""

    request_id: str
    timestamp: str
    agent: str
    model: str
    stream: bool
    call_type: str
    litellm_call_id: str
    max_tokens: int
    system_blocks: list[dict[str, Any]]
    messages: list[dict[str, Any]]
    tools: list[dict[str, Any]]


@dataclass
class Fragment:
    """An identified prompt fragment."""

    text: str
    fp: str  # fingerprint
    chash: str  # content_hash
    source: ContentSource
    fragment_type: FragmentType = FragmentType.UNKNOWN
    confidence: float = 0.0
    catalog_match: Optional[str] = None
    trigger: TriggerType = TriggerType.UNKNOWN
    trigger_detail: str = ""

    @classmethod
    def from_text(
        cls,
        text: str,
        source: ContentSource,
        **kwargs: Any,
    ) -> "Fragment":
        return cls(
            text=text,
            fp=fingerprint(text),
            chash=content_hash(text),
            source=source,
            **kwargs,
        )


@dataclass
class Observation:
    """A fragment seen in a request context."""

    fragment: Fragment
    request_id: str
    turn_index: int
    is_new: bool
    previous_tool_calls: list[str] = field(default_factory=list)


@dataclass
class ReplacementEntry:
    """Replacement table row."""

    fingerprint: str
    fragment_type: FragmentType
    catalog_match: Optional[str]
    original_preview: str  # first 200 chars
    replacement_text: Optional[str]  # None = not authored yet
    enabled: bool
    observation_count: int
    first_seen: str
    last_seen: str
