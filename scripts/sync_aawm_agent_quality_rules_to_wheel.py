#!/usr/bin/env python3
"""Sync canonical aawm_agent_quality_rules files into the callback wheel package.

Stop-gap (RR-002/003/008): keep
``.wheel-build/aawm_litellm_callbacks/aawm_agent_quality_rules.py`` and
``.wheel-build/aawm_litellm_callbacks/aawm_agent_quality_rules.json``
byte-identical to their canonical files under ``litellm/integrations/``.

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
CANONICAL_DIR = REPO_ROOT / "litellm" / "integrations"
WHEEL_DIR = REPO_ROOT / ".wheel-build" / "aawm_litellm_callbacks"
QUALITY_RULE_FILES = (
    "aawm_agent_quality_rules.py",
    "aawm_agent_quality_rules.json",
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

    if not WHEEL_DIR.is_dir():
        print(  # noqa: T201
            f"error: wheel package dir missing: {WHEEL_DIR}", file=sys.stderr
        )
        return 2

    file_pairs = tuple(
        (
            CANONICAL_DIR / file_name,
            WHEEL_DIR / file_name,
        )
        for file_name in QUALITY_RULE_FILES
    )
    for canonical, _wheel_copy in file_pairs:
        if not canonical.is_file():
            print(  # noqa: T201
                f"error: canonical missing: {canonical}", file=sys.stderr
            )
            return 2

    mismatches = [
        (canonical, wheel_copy)
        for canonical, wheel_copy in file_pairs
        if not wheel_copy.is_file()
        or _sha256(wheel_copy) != _sha256(canonical)
    ]
    if not mismatches:
        print("already in sync")  # noqa: T201
        for _canonical, wheel_copy in file_pairs:
            print(  # noqa: T201
                f"  {wheel_copy.name} sha256={_sha256(wheel_copy)}"
            )
        return 0

    if args.check:
        print("out of sync:", file=sys.stderr)  # noqa: T201
        for canonical, wheel_copy in mismatches:
            wheel_hash = _sha256(wheel_copy) if wheel_copy.is_file() else "<missing>"
            print(  # noqa: T201
                f"  {wheel_copy.name}: canonical sha256={_sha256(canonical)}; "
                f"wheel sha256={wheel_hash}",
                file=sys.stderr,
            )
        print("run without --check to sync", file=sys.stderr)  # noqa: T201
        return 1

    for canonical, wheel_copy in mismatches:
        shutil.copyfile(canonical, wheel_copy)
        print(  # noqa: T201
            f"synced {canonical} -> {wheel_copy} (sha256={_sha256(canonical)})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
