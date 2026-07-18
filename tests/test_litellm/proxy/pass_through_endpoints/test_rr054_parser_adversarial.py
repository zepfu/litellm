"""RR-054 focused adversarial parser tests.

Covers:
- trailing valid JSON after a Grok composer tool payload (#44)
- bounded handling of thousands of unmatched system-reminder openers (#54)
"""

from __future__ import annotations

import time
from typing import Any

from litellm.proxy.pass_through_endpoints import llm_passthrough_endpoints as lpe


def _exec_command_advertised_tools() -> dict[str, dict[str, Any]]:
    return {
        "exec_command": {
            "type": "object",
            "properties": {"cmd": {"type": "string"}},
            "required": ["cmd"],
            "additionalProperties": False,
        }
    }


def test_rr054_parser_trailing_valid_json_after_grok_tool_payload() -> None:
    """Trailing free-form JSON after a Grok tool payload must stay outside the payload.

    RR-054 #44: once the tool payload object is decoded, a following newline plus
    valid JSON is free-form text, not part of the tool payload or strip span.
    """
    trailing_json = '{"note": "free form json text that must remain visible"}'
    text = (
        "Intro keeps.\n"
        "Tool label: exec_command\n"
        "Correlation ref: call-1\n"
        'Input payload: {"cmd": "ls"}\n'
        f"{trailing_json}\n"
        "Tool label: exec_command\n"
        "Correlation ref: call-2\n"
        'Input payload: {"cmd": "pwd"}'
    )

    blocks = lpe._parse_grok_composer_literal_tool_label_blocks(text)
    assert len(blocks) == 2

    first = blocks[0]
    assert first["payload"].strip() == '{"cmd": "ls"}'
    first_span = text[int(first["start"]) : int(first["end"])]
    assert trailing_json not in first_span
    assert lpe._parse_grok_composer_literal_tool_payload_json(first["payload"]) == {
        "cmd": "ls"
    }

    leftover, items = lpe._repair_grok_composer_literal_tool_calls_in_text(
        text,
        advertised_tools=_exec_command_advertised_tools(),
    )
    assert items is not None and len(items) == 2
    assert all(item.get("name") == "exec_command" for item in items)
    assert leftover is not None
    assert "free form json text that must remain visible" in leftover
    assert "Intro keeps." in leftover
    assert "Tool label:" not in leftover


def test_rr054_parser_thousands_of_unmatched_system_reminder_openers_bounded() -> None:
    """Thousands of unmatched <system-reminder> openers must stay near-linear.

    RR-054 #54: adversarial client text with many openers and no closers must not
    quadratic-burn reminder compaction. Guarded paths should finish quickly.
    """
    # Thousands of unmatched openers plus filler (no closers). Unbounded non-greedy
    # DOTALL scans over this shape commonly take multi-second / 10s+; a bounded path
    # should return well under half a second.
    opener_count = 6000
    adversarial = ("<system-reminder>\n" * opener_count) + ("payload " * 20000)
    assert opener_count >= 1000
    assert adversarial.count("<system-reminder>") == opener_count
    assert "</system-reminder>" not in adversarial
    assert len(adversarial) > 200_000

    t0 = time.perf_counter()
    openai_out, openai_compacted, _markers, _meta = (
        lpe._compact_openai_adapter_claude_context_text(
            adversarial,
            cap=128,
        )
    )
    openai_elapsed = time.perf_counter() - t0

    assert isinstance(openai_out, str)
    assert openai_compacted == 0  # no closed blocks to compact
    assert openai_elapsed < 0.5, (
        "RR-054 #54 OpenAI system-reminder compact still expensive on "
        f"{opener_count} unmatched openers: {openai_elapsed:.3f}s "
        f"(len={len(adversarial)})"
    )

    t0 = time.perf_counter()
    (
        google_out,
        google_compacted,
        _hooks,
        _meta,
    ) = lpe._compact_expanded_claude_persisted_output_text_for_google_adapter(
        adversarial,
        auxiliary_context_char_cap=128,
    )
    google_elapsed = time.perf_counter() - t0

    assert isinstance(google_out, str)
    assert google_compacted == 0
    assert google_elapsed < 0.5, (
        "RR-054 #54 Google auxiliary/system-reminder compact still expensive on "
        f"{opener_count} unmatched openers: {google_elapsed:.3f}s "
        f"(len={len(adversarial)})"
    )
