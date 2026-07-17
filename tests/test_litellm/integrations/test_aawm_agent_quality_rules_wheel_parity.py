"""Parity guard: wheel-build quality rules must match the canonical module.

Stop-gap for RR-002/003/008: the packaged callback wheel under
``.wheel-build/aawm_litellm_callbacks/`` ships a copy of
``aawm_agent_quality_rules.py``. That copy must stay byte-identical to
``litellm/integrations/aawm_agent_quality_rules.py`` so malformed / literal
tool-call detection and ``tool_call_names`` threading do not drift.

Note: ``agent_identity.py`` in the wheel package may still be a full byte
copy of the integration module; this stop-gap only enforces parity for
``aawm_agent_quality_rules.py``. Full package extraction is Wave I, not here.

Re-sync with::

    python scripts/sync_aawm_agent_quality_rules_to_wheel.py
"""

from __future__ import annotations

import hashlib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
CANONICAL = REPO_ROOT / "litellm" / "integrations" / "aawm_agent_quality_rules.py"
WHEEL_COPY = (
    REPO_ROOT
    / ".wheel-build"
    / "aawm_litellm_callbacks"
    / "aawm_agent_quality_rules.py"
)

# Critical symbols that previously went missing from the wheel copy (~176 lines).
_CRITICAL_MARKERS = (
    "def is_malformed_function_tag_literal_text",
    "def is_malformed_claude_xml_literal_invocation_text",
    "def is_malformed_composer_call_literal_text",
    "def is_malformed_grok_literal_tool_label_transcript_text",
    "_COMPOSER_CALL_TEXT_MARKERS",
    "_GROK_LITERAL_TOOL_LABEL_LINE_RE",
    "tool_call_names: Sequence[str] = ()",
    "literal_tool_call_text",
    "malformed_tool_call_text",
    "clipped_tool_call_names",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_wheel_quality_rules_file_exists() -> None:
    assert CANONICAL.is_file(), f"canonical missing: {CANONICAL}"
    assert WHEEL_COPY.is_file(), f"wheel copy missing: {WHEEL_COPY}"


def test_wheel_quality_rules_byte_identical_to_canonical() -> None:
    """Wheel package copy must match canonical content exactly.

    The quality-rules module has no package-specific imports today, so a
    full-file hash compare is the right stop-gap. Prefer
    ``scripts/sync_aawm_agent_quality_rules_to_wheel.py`` over hand edits.
    """
    canonical_hash = _sha256(CANONICAL)
    wheel_hash = _sha256(WHEEL_COPY)
    if canonical_hash != wheel_hash:
        canonical_text = CANONICAL.read_text(encoding="utf-8")
        wheel_text = WHEEL_COPY.read_text(encoding="utf-8")
        raise AssertionError(
            "`.wheel-build/aawm_litellm_callbacks/aawm_agent_quality_rules.py` "
            "diverged from `litellm/integrations/aawm_agent_quality_rules.py`.\n"
            f"canonical sha256={canonical_hash} ({len(canonical_text)} chars)\n"
            f"wheel      sha256={wheel_hash} ({len(wheel_text)} chars)\n"
            "Re-sync with: python scripts/sync_aawm_agent_quality_rules_to_wheel.py"
        )


def test_wheel_quality_rules_contains_critical_malformed_tool_markers() -> None:
    """Marker check so failures name the missing detection surface, not only a hash."""
    wheel_text = WHEEL_COPY.read_text(encoding="utf-8")
    missing = [marker for marker in _CRITICAL_MARKERS if marker not in wheel_text]
    assert not missing, (
        "wheel quality rules missing critical malformed/literal tool-call markers: "
        f"{missing}"
    )
