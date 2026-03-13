"""Array diffing for consecutive RequestSnapshot objects.

Finds NEW content between consecutive API requests using content_hash dedup.
"""
from __future__ import annotations

from typing import Optional

from scripts.prompt_analyzer.models import (
    ContentSource,
    Fragment,
    RequestSnapshot,
    content_hash,
)


def _extract_text_from_block(block: object) -> str:
    """Safely extract text from a content block dict."""
    if isinstance(block, dict):
        return block.get("text", "") or ""
    return str(block)


def _extract_message_text_blocks(messages: list[dict]) -> list[str]:
    """Extract all text strings from messages array in order."""
    texts: list[str] = []
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, str):
            if content.strip():
                texts.append(content)
        elif isinstance(content, list):
            for block in content:
                text = _extract_text_from_block(block)
                if text.strip():
                    texts.append(text)
    return texts


class Differ:
    """Stateful differ that accumulates seen content hashes across calls."""

    def __init__(self) -> None:
        self._seen_hashes: set[str] = set()

    def diff_requests(
        self, prev: Optional[RequestSnapshot], curr: RequestSnapshot
    ) -> list[Fragment]:
        """Return fragments that are new in curr relative to prev.

        When prev is None (first request), all content is considered new.
        Uses content_hash for exact dedup — identical text in any source
        is reported only once across all calls.
        """
        fragments: list[Fragment] = []

        # --- System blocks ---
        prev_system_hashes: set[str] = set()
        if prev is not None:
            for block in prev.system_blocks:
                text = _extract_text_from_block(block)
                if text.strip():
                    prev_system_hashes.add(content_hash(text))

        for block in curr.system_blocks:
            text = _extract_text_from_block(block)
            if not text.strip():
                continue
            ch = content_hash(text)
            if ch not in prev_system_hashes and ch not in self._seen_hashes:
                self._seen_hashes.add(ch)
                fragments.append(Fragment.from_text(text, ContentSource.SYSTEM))

        # --- Messages ---
        prev_message_hashes: set[str] = set()
        if prev is not None:
            for text in _extract_message_text_blocks(prev.messages):
                prev_message_hashes.add(content_hash(text))

        for text in _extract_message_text_blocks(curr.messages):
            ch = content_hash(text)
            if ch not in prev_message_hashes and ch not in self._seen_hashes:
                self._seen_hashes.add(ch)
                fragments.append(Fragment.from_text(text, ContentSource.MESSAGES))

        # --- Tools ---
        prev_tool_names: set[str] = set()
        if prev is not None:
            for tool in prev.tools:
                name = tool.get("name", "")
                if name:
                    prev_tool_names.add(name)

        for tool in curr.tools:
            name = tool.get("name", "")
            if not name:
                continue
            description = tool.get("description", "") or ""
            if not description.strip():
                description = f"Tool: {name}"
            ch = content_hash(description)
            if name not in prev_tool_names and ch not in self._seen_hashes:
                self._seen_hashes.add(ch)
                fragments.append(Fragment.from_text(description, ContentSource.TOOLS))

        return fragments


def diff_requests(
    prev: Optional[RequestSnapshot],
    curr: RequestSnapshot,
    seen_hashes: Optional[set[str]] = None,
) -> list[Fragment]:
    """Stateless wrapper — pass seen_hashes to accumulate across calls.

    If seen_hashes is provided it is mutated in place (caller owns it).
    """
    if seen_hashes is None:
        seen_hashes = set()

    differ = Differ()
    differ._seen_hashes = seen_hashes
    return differ.diff_requests(prev, curr)
