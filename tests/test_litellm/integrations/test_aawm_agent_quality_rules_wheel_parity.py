"""Parity guard: wheel-build quality rules must match canonical sources.

Stop-gap for RR-002/003/008: the packaged callback wheel under
``.wheel-build/aawm_litellm_callbacks/`` ships copies of
``aawm_agent_quality_rules.py`` and ``aawm_agent_quality_rules.json``. Those
copies must stay byte-identical to the canonical files under
``litellm/integrations/`` so malformed / literal tool-call detection,
``tool_call_names`` threading, and rule configuration do not drift.

Note: ``agent_identity.py`` in the wheel package may still be a full byte
copy of the integration module; this stop-gap enforces parity for the
canonical Python and JSON rule files, ``aawm_agent_quality_rules.py`` and
``aawm_agent_quality_rules.json``. Full package extraction is Wave I, not here.

Re-sync with::

    python scripts/sync_aawm_agent_quality_rules_to_wheel.py
"""

from __future__ import annotations

import hashlib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
CANONICAL_DIR = REPO_ROOT / "litellm" / "integrations"
WHEEL_DIR = REPO_ROOT / ".wheel-build" / "aawm_litellm_callbacks"
QUALITY_RULE_FILES = (
    "aawm_agent_quality_rules.py",
    "aawm_agent_quality_rules.json",
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


def _quality_rule_paths(file_name: str) -> tuple[Path, Path]:
    return CANONICAL_DIR / file_name, WHEEL_DIR / file_name


def test_wheel_quality_rules_file_exists() -> None:
    for file_name in QUALITY_RULE_FILES:
        canonical, wheel_copy = _quality_rule_paths(file_name)
        assert canonical.is_file(), f"canonical missing: {canonical}"
        assert wheel_copy.is_file(), f"wheel copy missing: {wheel_copy}"


def test_wheel_quality_rules_byte_identical_to_canonical() -> None:
    """Wheel package copies must match canonical content exactly.

    The quality-rules files have no package-specific content, so a full-file
    hash compare is the right stop-gap. Prefer
    ``scripts/sync_aawm_agent_quality_rules_to_wheel.py`` over hand edits.
    """
    for file_name in QUALITY_RULE_FILES:
        canonical, wheel_copy = _quality_rule_paths(file_name)
        canonical_hash = _sha256(canonical)
        wheel_hash = _sha256(wheel_copy)
        if canonical_hash != wheel_hash:
            canonical_text = canonical.read_text(encoding="utf-8")
            wheel_text = wheel_copy.read_text(encoding="utf-8")
            raise AssertionError(
                f"`.wheel-build/aawm_litellm_callbacks/{file_name}` diverged "
                f"from `litellm/integrations/{file_name}`.\n"
                f"canonical sha256={canonical_hash} ({len(canonical_text)} chars)\n"
                f"wheel      sha256={wheel_hash} ({len(wheel_text)} chars)\n"
                "Re-sync with: python scripts/sync_aawm_agent_quality_rules_to_wheel.py"
            )


def test_wheel_quality_rules_contains_critical_malformed_tool_markers() -> None:
    """Marker check so failures name the missing detection surface, not only a hash."""
    wheel_text = (WHEEL_DIR / "aawm_agent_quality_rules.py").read_text(
        encoding="utf-8"
    )
    missing = [marker for marker in _CRITICAL_MARKERS if marker not in wheel_text]
    assert not missing, (
        "wheel quality rules missing critical malformed/literal tool-call markers: "
        f"{missing}"
    )
