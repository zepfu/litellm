#!/usr/bin/env python3
"""Sync canonical aawm_agent_identity.py into the callback wheel package.

Stop-gap (RR-003): keep
``.wheel-build/aawm_litellm_callbacks/agent_identity.py`` byte-identical
to ``litellm/integrations/aawm_agent_identity.py`` until Wave I extraction.

Usage::

    python scripts/sync_aawm_agent_identity_to_wheel.py
    python scripts/sync_aawm_agent_identity_to_wheel.py --check
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CANONICAL = REPO_ROOT / "litellm" / "integrations" / "aawm_agent_identity.py"
WHEEL_COPY = (
    REPO_ROOT
    / ".wheel-build"
    / "aawm_litellm_callbacks"
    / "agent_identity.py"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit 1 if wheel copy differs from canonical (no write).",
    )
    args = parser.parse_args(argv)
    if not CANONICAL.is_file():
        print(f"missing canonical {CANONICAL}", file=sys.stderr)
        return 2
    WHEEL_COPY.parent.mkdir(parents=True, exist_ok=True)
    if args.check:
        if not WHEEL_COPY.is_file():
            print(f"missing wheel copy {WHEEL_COPY}", file=sys.stderr)
            return 1
        if _sha256(CANONICAL) != _sha256(WHEEL_COPY):
            print(
                f"drift: {WHEEL_COPY} != {CANONICAL}",
                file=sys.stderr,
            )
            return 1
        print(f"ok {_sha256(CANONICAL)}")
        return 0
    shutil.copy2(CANONICAL, WHEEL_COPY)
    print(f"synced {CANONICAL} -> {WHEEL_COPY} (sha256={_sha256(WHEEL_COPY)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
