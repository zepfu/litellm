"""RR-054 #54: retained bare DOTALL system-reminder helper path coverage.

This file targets retained helpers that still apply a bare non-greedy DOTALL
``<system-reminder>.*?</system-reminder>`` scan over client-controlled text:

- factory ``_sanitize_gemini_tool_response_text``
- control-plane ``_extract_aawm_dispatch_context_references``

Each path is exercised with:

1. thousands of unmatched openers (no closers) — must stay near-linear / cheap
2. legitimate closed blocks — must keep functional behavior
"""

from __future__ import annotations

import time
from typing import Any, Callable

from litellm.litellm_core_utils.prompt_templates import factory as prompt_factory
from litellm.proxy.pass_through_endpoints import aawm_claude_control_plane as cp

# Mirror RR-054 #54 adversarial scale used by parser/operational residual tests.
_OPENER_COUNT = 6000
_FILLER_TOKEN = "payload "
_FILLER_COUNT = 20000
_BOUND_SECONDS = 0.5


def _unmatched_openers_payload(
    *,
    opener_count: int = _OPENER_COUNT,
    filler_count: int = _FILLER_COUNT,
) -> str:
    adversarial = ("<system-reminder>\n" * opener_count) + (
        _FILLER_TOKEN * filler_count
    )
    assert opener_count >= 1000
    assert adversarial.count("<system-reminder>") == opener_count
    assert "</system-reminder>" not in adversarial
    assert len(adversarial) > 200_000
    return adversarial


def _closed_system_reminder_blocks(
    *,
    count: int = 5,
    body_pad: int = 200,
    trailing_prompt: str = "Continue the task with tool use.",
) -> str:
    closed = "".join(
        (
            "<system-reminder>\n"
            f"SubagentStart hook additional context: CLAUDE.md body {i} "
            + ("x" * body_pad)
            + "\n</system-reminder>\n"
        )
        for i in range(count)
    )
    if trailing_prompt:
        return closed + trailing_prompt
    return closed


def _assert_bounded(
    label: str,
    fn: Callable[[], Any],
    *,
    opener_count: int = _OPENER_COUNT,
    limit_seconds: float = _BOUND_SECONDS,
) -> Any:
    t0 = time.perf_counter()
    result = fn()
    elapsed = time.perf_counter() - t0
    assert elapsed < limit_seconds, (
        f"RR-054 #54 {label} still expensive on {opener_count} unmatched "
        f"system-reminder openers: {elapsed:.3f}s"
    )
    return result


# ---------------------------------------------------------------------------
# factory._sanitize_gemini_tool_response_text
# ---------------------------------------------------------------------------


def test_rr054_sanitize_gemini_unmatched_openers_bounded() -> None:
    adversarial = _unmatched_openers_payload()

    out = _assert_bounded(
        "sanitize_gemini_tool_response_text",
        lambda: prompt_factory._sanitize_gemini_tool_response_text(adversarial),
    )

    assert isinstance(out, str)
    # No closers => regex does not strip; sanitized path still returns text.
    assert out
    assert "<system-reminder>" in out


def test_rr054_sanitize_gemini_closed_blocks_strip_reminders() -> None:
    closed = _closed_system_reminder_blocks()

    out = prompt_factory._sanitize_gemini_tool_response_text(closed)

    assert out == "Continue the task with tool use."
    assert "<system-reminder>" not in out
    assert "</system-reminder>" not in out


def test_rr054_sanitize_gemini_closed_only_falls_back_to_stripped_original() -> None:
    pure_closed = _closed_system_reminder_blocks(trailing_prompt="")

    out = prompt_factory._sanitize_gemini_tool_response_text(pure_closed)

    # After stripping only-reminder content the cleaned string is empty, so the
    # helper falls back to the original stripped text.
    assert out == pure_closed.strip()
    assert "<system-reminder>" in out


# ---------------------------------------------------------------------------
# aawm_claude_control_plane._extract_aawm_dispatch_context_references
# ---------------------------------------------------------------------------


def test_rr054_dispatch_context_refs_unmatched_openers_bounded() -> None:
    adversarial = _unmatched_openers_payload() + " see `KEEPME` and SQL"

    out = _assert_bounded(
        "extract_aawm_dispatch_context_references",
        lambda: cp._extract_aawm_dispatch_context_references(adversarial),
    )

    assert isinstance(out, list)
    # The bare DOTALL sub may be expensive, but when it finds no closed blocks
    # the payload remains searchable for outside references.
    names = {name for name, _kind in out}
    assert "KEEPME" in names


def test_rr054_dispatch_context_refs_closed_blocks_strip_inside_refs() -> None:
    inside_only = (
        "<system-reminder>\n"
        "SubagentStart hook additional context: `INSIDE` SQL\n"
        "</system-reminder>\n"
        " outside `OUTSIDE` and API"
    )

    out = cp._extract_aawm_dispatch_context_references(inside_only)

    names = {name for name, _kind in out}
    assert "OUTSIDE" in names
    assert "INSIDE" not in names


def test_rr054_dispatch_context_refs_closed_blocks_still_find_outside() -> None:
    closed = _closed_system_reminder_blocks(
        trailing_prompt=" inspect `KEEPME` and SQL now"
    )

    out = cp._extract_aawm_dispatch_context_references(closed)

    assert ("KEEPME", "dispatch_backtick") in out
