"""Pane needle matching. Needles that only appear in the sent prompt are ignored."""

from __future__ import annotations


def _pane_has_any(
    pane: str,
    needles: list[str],
    *,
    prompt: str | None = None,
) -> bool:
    """True if a needle is in the pane and is not just the sent prompt."""
    sent = prompt or ""
    return any(token and token in pane and token not in sent for token in needles)
