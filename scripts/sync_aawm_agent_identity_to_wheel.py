#!/usr/bin/env python3
"""Guard single-source packaging for callback-wheel agent_identity (RR-003).

Canonical implementation lives only in::

    litellm/integrations/aawm_agent_identity/

``.wheel-build/aawm_litellm_callbacks/agent_identity.py`` must remain a thin
checkout loader that re-exports that package. The published
``aawm-litellm-callbacks`` wheel force-includes every canonical package module
via hatch (see ``.wheel-build/pyproject.toml``); do **not** reintroduce a full
maintained source copy under ``.wheel-build/``.

Usage::

    python scripts/sync_aawm_agent_identity_to_wheel.py
    python scripts/sync_aawm_agent_identity_to_wheel.py --check

Both modes are read-only checks. There is nothing to copy: packaging pulls the
canonical package modules at wheel build time.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CANONICAL_PACKAGE = REPO_ROOT / "litellm" / "integrations" / "aawm_agent_identity"
CANONICAL_INIT = CANONICAL_PACKAGE / "__init__.py"
LOADER = REPO_ROOT / ".wheel-build" / "aawm_litellm_callbacks" / "agent_identity.py"
PYPROJECT = REPO_ROOT / ".wheel-build" / "pyproject.toml"

_MAX_LOADER_LINES = 80
_FORCE_INCLUDE_SECTIONS = (
    "tool.hatch.build.targets.wheel.force-include",
    "tool.hatch.build.targets.sdist.force-include",
)
_LEGACY_FORCE_INCLUDE_SNIPPET = (
    '"../litellm/integrations/aawm_agent_identity.py" = '
    '"aawm_litellm_callbacks/agent_identity.py"'
)
# RR-006: wheel must also ship package-owned session_history modules so the
# force-included agent_identity package imports resolve inside the wheel.
_SESSION_HISTORY_FORCE_INCLUDE_SNIPPETS = (
    '"../litellm/integrations/aawm_session_history/__init__.py" = '
    '"litellm/integrations/aawm_session_history/__init__.py"',
    '"../litellm/integrations/aawm_session_history/runtime.py" = '
    '"litellm/integrations/aawm_session_history/runtime.py"',
    '"../litellm/integrations/aawm_session_history/writer.py" = '
    '"litellm/integrations/aawm_session_history/writer.py"',
    '"../litellm/integrations/aawm_session_history/spool.py" = '
    '"litellm/integrations/aawm_session_history/spool.py"',
    '"../litellm/integrations/aawm_session_history/retry.py" = '
    '"litellm/integrations/aawm_session_history/retry.py"',
    '"../litellm/integrations/aawm_session_history/record.py" = '
    '"litellm/integrations/aawm_session_history/record.py"',
    '"../litellm/integrations/aawm_session_history/sql.py" = '
    '"litellm/integrations/aawm_session_history/sql.py"',
    '"../litellm/integrations/aawm_session_history/identity_selection.py" = '
    '"litellm/integrations/aawm_session_history/identity_selection.py"',
    '"../litellm/integrations/aawm_session_history_sql.py" = '
    '"litellm/integrations/aawm_session_history_sql.py"',
)
_REQUIRED_LOADER_MARKERS = (
    "Checkout loader for aawm_litellm_callbacks",
    "litellm.integrations.aawm_agent_identity",
    "force-includes the canonical package",
)
_FORBIDDEN_LOADER_MARKERS = (
    "class AawmAgentIdentity",
    "def _enqueue_session_history_record",
    "def _spool_session_history_records",
)


def _line_count(path: Path) -> int:
    return path.read_text(encoding="utf-8").count("\n") + 1


def _canonical_package_force_include_snippets() -> tuple[str, ...]:
    snippets = []
    for source_path in sorted(CANONICAL_PACKAGE.rglob("*.py")):
        repo_relative = source_path.relative_to(REPO_ROOT).as_posix()
        snippets.append(f'"../{repo_relative}" = "{repo_relative}"')
    return tuple(snippets)


def _section_body(text: str, section_name: str) -> str | None:
    header_match = re.search(
        rf"(?m)^\[{re.escape(section_name)}\]\s*$",
        text,
    )
    if header_match is None:
        return None
    body_start = header_match.end()
    next_section = re.search(r"(?m)^\[", text[body_start:])
    if next_section is None:
        return text[body_start:]
    return text[body_start : body_start + next_section.start()]


def _validate() -> list[str]:
    errors: list[str] = []
    if not CANONICAL_PACKAGE.is_dir():
        errors.append(f"missing canonical package {CANONICAL_PACKAGE}")
        return errors
    if not CANONICAL_INIT.is_file():
        errors.append(f"missing canonical package initializer {CANONICAL_INIT}")
        return errors
    if not LOADER.is_file():
        errors.append(f"missing checkout loader {LOADER}")
        return errors
    if not PYPROJECT.is_file():
        errors.append(f"missing {PYPROJECT}")
        return errors

    loader_text = LOADER.read_text(encoding="utf-8")
    loader_lines = _line_count(LOADER)
    if loader_lines > _MAX_LOADER_LINES:
        errors.append(
            f"checkout loader too large ({loader_lines} lines > "
            f"{_MAX_LOADER_LINES}); full source copy is not allowed"
        )
    for marker in _REQUIRED_LOADER_MARKERS:
        if marker not in loader_text:
            errors.append(f"checkout loader missing required marker: {marker!r}")
    for marker in _FORBIDDEN_LOADER_MARKERS:
        if marker in loader_text:
            errors.append(
                f"checkout loader looks like a full implementation "
                f"(found {marker!r}); use thin re-export only"
            )
    if loader_text == CANONICAL_INIT.read_text(encoding="utf-8"):
        errors.append(
            "checkout loader is byte-identical to canonical __init__; dual-maintained "
            "full source copy is forbidden"
        )

    pyproject_text = PYPROJECT.read_text(encoding="utf-8")
    if 'build-backend = "hatchling.build"' not in pyproject_text:
        errors.append(
            ".wheel-build/pyproject.toml must use hatchling.build for "
            "force-include packaging"
        )
    required_snippets = (
        *_canonical_package_force_include_snippets(),
        *_SESSION_HISTORY_FORCE_INCLUDE_SNIPPETS,
    )
    for section_name in _FORCE_INCLUDE_SECTIONS:
        section_body = _section_body(pyproject_text, section_name)
        if section_body is None:
            errors.append(
                f".wheel-build/pyproject.toml missing [{section_name}]"
            )
            continue
        for snippet in required_snippets:
            if snippet not in section_body:
                errors.append(
                    f".wheel-build/pyproject.toml [{section_name}] missing "
                    f"force-include mapping: {snippet}"
                )
    if _LEGACY_FORCE_INCLUDE_SNIPPET in pyproject_text:
        errors.append(
            ".wheel-build/pyproject.toml still contains the obsolete "
            "single-file agent_identity force-include mapping"
        )
    # Reject accidental setuptools dual-copy package layouts that drop force-include.
    if (
        re.search(
            r"(?m)^\[tool\.setuptools",
            pyproject_text,
        )
        and "force-include" not in pyproject_text
    ):
        errors.append(
            ".wheel-build/pyproject.toml appears to use setuptools without "
            "force-include; single-source packaging requires hatch force-include"
        )
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit 1 when single-source packaging guards fail (default behavior).",
    )
    # --check is accepted for historical callers; both modes validate only.
    parser.parse_args(argv)

    errors = _validate()
    if errors:
        for error in errors:
            print(f"error: {error}", file=sys.stderr)  # noqa: T201
        print(  # noqa: T201
            "RR-003 single-source packaging check failed.\n"
            "Canonical source: litellm/integrations/aawm_agent_identity/\n"
            "Checkout path must stay a thin loader; hatch must force-include "
            "every canonical package module into the published wheel and sdist.",
            file=sys.stderr,
        )
        return 1

    print(  # noqa: T201
        "ok: thin checkout loader + hatch force-include single-source packaging "
        "(aawm_agent_identity package + aawm_session_history)"
    )
    print(f"  canonical: {CANONICAL_PACKAGE}")  # noqa: T201
    print(f"  loader:    {LOADER} ({_line_count(LOADER)} lines)")  # noqa: T201
    print(f"  packaging: {PYPROJECT}")  # noqa: T201
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
