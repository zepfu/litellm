"""RR-054 #54: remaining bare DOTALL system-reminder helper path coverage.

Existing adversarial coverage already bounds the primary OpenAI / Google
compaction entrypoints. This file targets the remaining helpers that still
apply a bare non-greedy DOTALL ``<system-reminder>.*?</system-reminder>``
scan over client-controlled text:

- ``_extract_google_adapter_latest_user_prompt_text``
- ``_split_google_adapter_inline_context_and_prompt``
- ``_compact_google_adapter_oversized_text_part``
- ``_extract_google_adapter_preserved_task_excerpt``
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
from litellm.proxy.pass_through_endpoints import llm_passthrough_endpoints as lpe

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
# _extract_google_adapter_latest_user_prompt_text
# ---------------------------------------------------------------------------


def test_rr054_latest_user_prompt_unmatched_openers_bounded() -> None:
    adversarial = _unmatched_openers_payload()
    messages = [{"role": "user", "content": adversarial}]

    out = _assert_bounded(
        "latest_user_prompt",
        lambda: lpe._extract_google_adapter_latest_user_prompt_text(messages),
    )

    # No closers => the DOTALL scan finds nothing and the full text is returned.
    assert out == adversarial.strip()


def test_rr054_latest_user_prompt_closed_blocks_return_trailing_prompt() -> None:
    closed = _closed_system_reminder_blocks()
    messages = [
        {"role": "assistant", "content": "prior"},
        {"role": "user", "content": closed},
    ]

    out = lpe._extract_google_adapter_latest_user_prompt_text(messages)

    assert out == "Continue the task with tool use."


def test_rr054_latest_user_prompt_closed_only_skips_to_prior_user() -> None:
    pure_closed = _closed_system_reminder_blocks(trailing_prompt="")
    messages = [
        {"role": "user", "content": "real prior prompt"},
        {"role": "user", "content": pure_closed},
    ]

    out = lpe._extract_google_adapter_latest_user_prompt_text(messages)

    assert out == "real prior prompt"


# ---------------------------------------------------------------------------
# _split_google_adapter_inline_context_and_prompt
# ---------------------------------------------------------------------------


def test_rr054_split_inline_context_unmatched_openers_bounded() -> None:
    adversarial = _unmatched_openers_payload()
    request_block: dict[str, Any] = {
        "contents": [{"role": "user", "parts": [{"text": adversarial}]}]
    }

    changes = _assert_bounded(
        "split_inline_context",
        lambda: lpe._split_google_adapter_inline_context_and_prompt(request_block),
    )

    assert changes == {}
    assert len(request_block["contents"]) == 1
    assert request_block["contents"][0]["parts"][0]["text"] == adversarial


def test_rr054_split_inline_context_closed_blocks_split_trailing_prompt() -> None:
    closed = _closed_system_reminder_blocks()
    request_block: dict[str, Any] = {
        "contents": [{"role": "user", "parts": [{"text": closed}]}]
    }

    changes = lpe._split_google_adapter_inline_context_and_prompt(request_block)

    assert changes == {
        "split_inline_context_prompt_count": 1,
        "split_inline_context_prompt_chars": len("Continue the task with tool use."),
    }
    assert len(request_block["contents"]) == 2
    context_text = request_block["contents"][0]["parts"][0]["text"]
    prompt_text = request_block["contents"][1]["parts"][0]["text"]
    assert context_text.startswith("<system-reminder>")
    assert context_text.count("<system-reminder>") == 5
    assert context_text.count("</system-reminder>") == 5
    assert prompt_text == "Continue the task with tool use."


# ---------------------------------------------------------------------------
# _compact_google_adapter_oversized_text_part
# ---------------------------------------------------------------------------


def test_rr054_oversized_text_part_unmatched_openers_bounded() -> None:
    adversarial = _unmatched_openers_payload()
    part = {"text": adversarial}

    updated, changed, stats = _assert_bounded(
        "oversized_text_part",
        lambda: lpe._compact_google_adapter_oversized_text_part(
            part,
            cap=128,
            pure_context_cap=64,
            head_keep=40,
            tail_keep=40,
            is_followup_request=False,
        ),
    )

    assert isinstance(updated, dict)
    assert changed is True
    assert stats["compacted_count"] == 1
    # Unmatched openers => not reminder-only pure context; falls through to
    # head/tail oversized compaction rather than pure-context truncate.
    assert stats["pure_context_compacted_count"] == 0
    assert "Gemini adapter compacted oversized user text" in updated["text"]
    assert len(updated["text"]) < len(adversarial)


def test_rr054_oversized_text_part_closed_subagent_context_still_compacts() -> None:
    # Single closed SubagentStart reminder large enough to trip the subagent cap.
    closed = (
        "<system-reminder>\n"
        "SubagentStart hook additional context: "
        + ("x" * 5000)
        + "\n</system-reminder>\n"
    )
    part = {"text": closed}

    updated, changed, stats = lpe._compact_google_adapter_oversized_text_part(
        part,
        cap=10_000,
        pure_context_cap=80,
        head_keep=40,
        tail_keep=40,
        is_followup_request=False,
    )

    assert changed is True
    assert stats["compacted_count"] == 1
    assert stats["pure_context_compacted_count"] == 1
    assert stats["subagent_context_compacted_count"] == 1
    assert isinstance(updated, dict)
    assert updated["text"].startswith("<system-reminder>")
    assert len(updated["text"]) == stats["compacted_text_chars"]
    assert len(updated["text"]) < len(closed)


def test_rr054_oversized_text_part_closed_blocks_with_trailing_prompt_still_work() -> None:
    closed = _closed_system_reminder_blocks(body_pad=400)
    part = {"text": closed}

    updated, changed, stats = lpe._compact_google_adapter_oversized_text_part(
        part,
        cap=128,
        pure_context_cap=64,
        head_keep=40,
        tail_keep=40,
        is_followup_request=False,
    )

    assert changed is True
    assert stats["compacted_count"] == 1
    assert isinstance(updated, dict)
    # Trailing prompt means this is not pure reminder-only context.
    assert stats["pure_context_compacted_count"] == 0
    assert "Gemini adapter compacted oversized user text" in updated["text"]


# ---------------------------------------------------------------------------
# _extract_google_adapter_preserved_task_excerpt
# ---------------------------------------------------------------------------


def test_rr054_preserved_task_excerpt_unmatched_openers_bounded() -> None:
    adversarial = _unmatched_openers_payload()

    out = _assert_bounded(
        "preserved_task_excerpt",
        lambda: lpe._extract_google_adapter_preserved_task_excerpt(adversarial),
    )

    assert isinstance(out, str)
    assert out  # unmatched => no strip; capped tail of original text
    cap = lpe._get_google_adapter_preserved_task_state_char_cap()
    assert len(out) <= cap


def test_rr054_preserved_task_excerpt_closed_blocks_prefer_trailing_prompt() -> None:
    closed = _closed_system_reminder_blocks()

    out = lpe._extract_google_adapter_preserved_task_excerpt(closed)

    assert out == "Continue the task with tool use."


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
