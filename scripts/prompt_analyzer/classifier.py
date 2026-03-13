"""Fragment classifier.

Type classification cascade:
  1. Tag-based (confidence 0.95)
  2. Pattern-based (confidence 0.8)
  3. Catalog match (catalog confidence)
  4. Fallback: UNKNOWN (confidence 0.3)

Trigger identification based on previous assistant tool calls.
"""
from __future__ import annotations

import copy
import re
from typing import Optional

from scripts.prompt_analyzer.catalog import CatalogIndex
from scripts.prompt_analyzer.models import (
    ContentSource,
    Fragment,
    FragmentType,
    TriggerType,
)

_TOOL_RESULT_JSON_PATTERN = re.compile(r'"type"\s*:\s*"tool_result"')
_SKILL_KEYWORDS = re.compile(r"\bslash\s+command\b|\bskill\b", re.IGNORECASE)
_MODE_KEYWORDS = re.compile(r"\bplan\s+mode\b|\bfast\s+mode\b|\bauto\s+mode\b", re.IGNORECASE)
_AGENT_DISPATCH_TOOLS = re.compile(r"\bAgent\b")
_TOOL_SEARCH_TOOLS = re.compile(r"search|find|lookup|query", re.IGNORECASE)


def _classify_type(
    fragment: Fragment,
    catalog: Optional[CatalogIndex],
) -> tuple[FragmentType, float, Optional[str]]:
    """Return (fragment_type, confidence, catalog_match)."""
    text = fragment.text
    source = fragment.source

    # --- Step 1: Tag-based (confidence 0.95) ---
    if "<system-reminder>" in text:
        return FragmentType.SYSTEM_REMINDER, 0.95, None

    if source == ContentSource.TOOLS:
        return FragmentType.TOOL_DESCRIPTION, 0.95, None

    # System blocks with cache_control are typically the main system prompt
    # (We check the text for the phrase rather than inspecting the raw block here
    # since Fragment only holds text.)
    if source == ContentSource.SYSTEM and (
        "You are Claude Code" in text or "You are an interactive agent" in text
    ):
        return FragmentType.SYSTEM_PROMPT, 0.95, None

    # --- Step 2: Pattern-based (confidence 0.8) ---
    if text.startswith("You are '") or text.startswith("You are "):
        if len(text) < 2000:
            return FragmentType.AGENT_PROMPT, 0.8, None
        return FragmentType.SYSTEM_PROMPT, 0.8, None

    if _TOOL_RESULT_JSON_PATTERN.search(text):
        return FragmentType.SYSTEM_DATA, 0.8, None

    if _SKILL_KEYWORDS.search(text):
        return FragmentType.SKILL, 0.8, None

    # --- Step 3: Catalog match ---
    if catalog is not None:
        catalog_name, confidence = catalog.match(text)
        if catalog_name is not None and confidence >= 0.5:
            # Try to infer type from catalog entry name
            ftype = _infer_type_from_catalog_name(catalog_name)
            return ftype, confidence, catalog_name

    # --- Step 4: Fallback ---
    return FragmentType.UNKNOWN, 0.3, None


def _infer_type_from_catalog_name(name: str) -> FragmentType:
    name_lower = name.lower()
    if "system" in name_lower and "prompt" in name_lower:
        return FragmentType.SYSTEM_PROMPT
    if "tool" in name_lower:
        return FragmentType.TOOL_DESCRIPTION
    if "reminder" in name_lower:
        return FragmentType.SYSTEM_REMINDER
    if "agent" in name_lower:
        return FragmentType.AGENT_PROMPT
    if "skill" in name_lower:
        return FragmentType.SKILL
    return FragmentType.UNKNOWN


def _classify_trigger(
    fragment: Fragment,
    fragment_type: FragmentType,
    prev_assistant_tool_calls: list[str],
    is_first_request: bool,
) -> tuple[TriggerType, str]:
    """Return (trigger_type, trigger_detail)."""
    text = fragment.text

    if is_first_request:
        return TriggerType.SESSION_START, "first request in session"

    if any(_AGENT_DISPATCH_TOOLS.search(tc) for tc in prev_assistant_tool_calls):
        return TriggerType.AGENT_DISPATCH, "previous turn used Agent tool"

    if fragment_type == FragmentType.SYSTEM_REMINDER and (
        "CLAUDE.md" in text or "memory" in text.lower() or "claudeMd" in text
    ):
        return TriggerType.CONTEXT_INJECTION, "CLAUDE.md / memory context"

    if fragment.source == ContentSource.TOOLS:
        return TriggerType.TOOL_LOAD, "new tool loaded"

    if prev_assistant_tool_calls and any(
        _TOOL_SEARCH_TOOLS.search(tc) for tc in prev_assistant_tool_calls
    ):
        return TriggerType.TOOL_SEARCH_RESULT, "previous turn used search-like tool"

    if _MODE_KEYWORDS.search(text):
        return TriggerType.MODE_CHANGE, "mode change keywords detected"

    if prev_assistant_tool_calls:
        return TriggerType.TOOL_RESULT, f"prev tools: {', '.join(prev_assistant_tool_calls[:3])}"

    return TriggerType.UNKNOWN, ""


def classify(
    fragment: Fragment,
    prev_assistant_tool_calls: list[str],
    catalog: Optional[CatalogIndex] = None,
    is_first_request: bool = False,
) -> Fragment:
    """Return a copy of fragment with type/trigger fields populated."""
    result = copy.copy(fragment)

    ftype, confidence, catalog_match = _classify_type(fragment, catalog)
    result.fragment_type = ftype
    result.confidence = confidence
    result.catalog_match = catalog_match

    trigger, trigger_detail = _classify_trigger(
        fragment, ftype, prev_assistant_tool_calls, is_first_request
    )
    result.trigger = trigger
    result.trigger_detail = trigger_detail

    return result
