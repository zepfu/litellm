#!/usr/bin/env python3
"""Guard single-source packaging for callback-wheel agent_identity (RR-003).

Canonical implementation lives only at::

    litellm/integrations/aawm_agent_identity.py

``.wheel-build/aawm_litellm_callbacks/agent_identity.py`` must remain a thin
checkout loader that re-exports that module. The published
``aawm-litellm-callbacks`` wheel force-includes the canonical file via hatch
(see ``.wheel-build/pyproject.toml``); do **not** reintroduce a full maintained
source copy under ``.wheel-build/``.

Usage::

    python scripts/sync_aawm_agent_identity_to_wheel.py
    python scripts/sync_aawm_agent_identity_to_wheel.py --check

Both modes are read-only checks. There is nothing to copy: packaging pulls the
canonical module at wheel build time.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CANONICAL = REPO_ROOT / "litellm" / "integrations" / "aawm_agent_identity.py"
LOADER = REPO_ROOT / ".wheel-build" / "aawm_litellm_callbacks" / "agent_identity.py"
PYPROJECT = REPO_ROOT / ".wheel-build" / "pyproject.toml"

_MAX_LOADER_LINES = 80
_FORCE_INCLUDE_SNIPPET = (
    '"../litellm/integrations/aawm_agent_identity.py" = '
    '"aawm_litellm_callbacks/agent_identity.py"'
)
# RR-006: wheel must also ship package-owned session_history modules so the
# force-included agent_identity import path resolves inside the wheel.
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
    "force-includes the canonical file",
)
_FORBIDDEN_LOADER_MARKERS = (
    "class AawmAgentIdentity",
    "def _enqueue_session_history_record",
    "def _spool_session_history_records",
)


def _line_count(path: Path) -> int:
    return path.read_text(encoding="utf-8").count("\n") + 1


def _validate() -> list[str]:
    errors: list[str] = []
    if not CANONICAL.is_file():
        errors.append(f"missing canonical {CANONICAL}")
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
    if loader_text == CANONICAL.read_text(encoding="utf-8"):
        errors.append(
            "checkout loader is byte-identical to canonical; dual-maintained "
            "full source copy is forbidden"
        )

    pyproject_text = PYPROJECT.read_text(encoding="utf-8")
    if 'build-backend = "hatchling.build"' not in pyproject_text:
        errors.append(
            ".wheel-build/pyproject.toml must use hatchling.build for "
            "force-include packaging"
        )
    if "[tool.hatch.build.targets.wheel.force-include]" not in pyproject_text:
        errors.append(
            ".wheel-build/pyproject.toml missing "
            "[tool.hatch.build.targets.wheel.force-include]"
        )
    if _FORCE_INCLUDE_SNIPPET not in pyproject_text:
        errors.append(
            ".wheel-build/pyproject.toml missing force-include mapping of "
            "canonical agent_identity into the callback package"
        )
    for snippet in _SESSION_HISTORY_FORCE_INCLUDE_SNIPPETS:
        if snippet not in pyproject_text:
            errors.append(
                ".wheel-build/pyproject.toml missing session_history "
                f"force-include mapping: {snippet}"
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
            "Canonical source: litellm/integrations/aawm_agent_identity.py\n"
            "Checkout path must stay a thin loader; hatch force-includes "
            "the canonical module into the published wheel.",
            file=sys.stderr,
        )
        return 1

    print(  # noqa: T201
        "ok: thin checkout loader + hatch force-include single-source packaging (agent_identity + aawm_session_history)"
    )
    print(f"  canonical: {CANONICAL}")  # noqa: T201
    print(f"  loader:    {LOADER} ({_line_count(LOADER)} lines)")  # noqa: T201
    print(f"  packaging: {PYPROJECT}")  # noqa: T201
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
