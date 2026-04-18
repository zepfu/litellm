#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ArtifactGroup:
    name: str
    paths: tuple[str, ...]
    version_file: Path
    pattern: str
    tag_prefix: str


ROOT = Path(__file__).resolve().parents[1]

GROUPS = (
    ArtifactGroup(
        name="callback",
        paths=(
            ".wheel-build/",
            "litellm/integrations/aawm_agent_identity.py",
            "litellm/integrations/aawm_payload_capture.py",
        ),
        version_file=ROOT / ".wheel-build" / "pyproject.toml",
        pattern=r'version = "([0-9]+)\.([0-9]+)\.([0-9]+)"',
        tag_prefix="cb-v",
    ),
    ArtifactGroup(
        name="control_plane",
        paths=(
            ".control-plane-wheel-build/",
            "litellm/proxy/pass_through_endpoints/aawm_claude_control_plane.py",
            "context-replacement/",
        ),
        version_file=ROOT / ".control-plane-wheel-build" / "pyproject.toml",
        pattern=r'version = "([0-9]+)\.([0-9]+)\.([0-9]+)"',
        tag_prefix="cp-v",
    ),
    ArtifactGroup(
        name="harness",
        paths=("scripts/local-ci/",),
        version_file=ROOT / "scripts" / "local-ci" / "harness-version.txt",
        pattern=r"^([0-9]+)\.([0-9]+)\.([0-9]+)$",
        tag_prefix="h-v",
    ),
    ArtifactGroup(
        name="config",
        paths=(
            "model_prices_and_context_window.json",
            "litellm/model_prices_and_context_window_backup.json",
            "scripts/build_model_config_bundle.py",
        ),
        version_file=ROOT / "model-config-version.txt",
        pattern=r"^([0-9]+)\.([0-9]+)\.([0-9]+)$",
        tag_prefix="cfg-v",
    ),
)


def _get_changed_files(before: str, after: str) -> list[str]:
    if not before or set(before) == {"0"}:
        cmd = ["git", "diff-tree", "--no-commit-id", "--name-only", "-r", after]
    else:
        cmd = ["git", "diff", "--name-only", before, after]
    output = subprocess.check_output(cmd, cwd=ROOT, text=True)
    return [line for line in output.splitlines() if line]


def _path_matches(changed_file: str, prefixes: tuple[str, ...]) -> bool:
    return any(changed_file == prefix or changed_file.startswith(prefix) for prefix in prefixes)


def _bump_version_text(text: str, pattern: str) -> tuple[str, str, str]:
    match = re.search(pattern, text, re.MULTILINE)
    if match is None:
        raise ValueError(f"Unable to parse version using pattern: {pattern}")
    old_version = ".".join(match.groups())
    major, minor, patch = map(int, match.groups())
    new_version = f"{major}.{minor}.{patch + 1}"
    start, end = match.span()
    updated = text[:start] + match.group(0).replace(old_version, new_version, 1) + text[end:]
    return updated, old_version, new_version


def main() -> int:
    parser = argparse.ArgumentParser(description="Bump AAWM artifact versions based on changed files.")
    parser.add_argument("--before", required=True, help="Previous git SHA from the push event")
    parser.add_argument("--after", required=True, help="Current git SHA from the push event")
    parser.add_argument(
        "--write",
        action="store_true",
        help="Write bumped versions back to disk",
    )
    args = parser.parse_args()

    changed_files = _get_changed_files(args.before, args.after)
    bumped: list[dict[str, str]] = []

    for group in GROUPS:
        if not any(_path_matches(path, group.paths) for path in changed_files):
            continue
        original = group.version_file.read_text(encoding="utf-8")
        updated, old_version, new_version = _bump_version_text(original, group.pattern)
        if args.write and updated != original:
            group.version_file.write_text(updated, encoding="utf-8")
        bumped.append(
            {
                "name": group.name,
                "old_version": old_version,
                "new_version": new_version,
                "tag": f"{group.tag_prefix}{new_version}",
                "version_file": str(group.version_file.relative_to(ROOT)),
            }
        )

    payload = {
        "changed_files": changed_files,
        "bumped": bumped,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
