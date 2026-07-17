#!/usr/bin/env python3
"""Sync canonical aawm_agent_quality_rules.py into the callback wheel package.

Stop-gap (RR-002/003/008): keep
``.wheel-build/aawm_litellm_callbacks/aawm_agent_quality_rules.py`` byte-identical
to ``litellm/integrations/aawm_agent_quality_rules.py``.

``agent_identity.py`` may still be a separate full-byte copy of the integration
module; this helper only covers quality rules. Full package extraction is Wave I.

Usage::

    python scripts/sync_aawm_agent_quality_rules_to_wheel.py
    python scripts/sync_aawm_agent_quality_rules_to_wheel.py --check
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CANONICAL = REPO_ROOT / "litellm" / "integrations" / "aawm_agent_quality_rules.py"
WHEEL_COPY = (
    REPO_ROOT
    / ".wheel-build"
    / "aawm_litellm_callbacks"
    / "aawm_agent_quality_rules.py"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit 1 if wheel copy differs; do not write.",
    )
    args = parser.parse_args(argv)

    if not CANONICAL.is_file():
        print(f"error: canonical missing: {CANONICAL}", file=sys.stderr)
        return 2
    if not WHEEL_COPY.parent.is_dir():
        print(f"error: wheel package dir missing: {WHEEL_COPY.parent}", file=sys.stderr)
        return 2

    canonical_hash = _sha256(CANONICAL)
    if WHEEL_COPY.is_file() and _sha256(WHEEL_COPY) == canonical_hash:
        print(f"already in sync (sha256={canonical_hash})")
        return 0

    if args.check:
        wheel_hash = _sha256(WHEEL_COPY) if WHEEL_COPY.is_file() else "<missing>"
        print(
            "out of sync:\n"
            f"  canonical sha256={canonical_hash}\n"
            f"  wheel      sha256={wheel_hash}\n"
            "run without --check to sync",
            file=sys.stderr,
        )
        return 1

    shutil.copyfile(CANONICAL, WHEEL_COPY)
    print(f"synced {CANONICAL} -> {WHEEL_COPY} (sha256={canonical_hash})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
