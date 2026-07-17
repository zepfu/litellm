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
            "litellm/bundled_model_prices_and_context_window_fallback.json",
            "scripts/build_model_config_bundle.py",
        ),
        version_file=ROOT / "model-config-version.txt",
        pattern=r"^([0-9]+)\.([0-9]+)\.([0-9]+)$",
        tag_prefix="cfg-v",
    ),
)

# Sentinel: must never appear in GROUPS paths or derived workflow watch paths.
_FORBIDDEN_WORKFLOW_PATHS = frozenset(
    {
        "litellm/model_prices_and_context_window_backup.json",
    }
)


def _normalize_group_path(path: str) -> str:
    """Normalize a group path for matching and workflow emission."""
    return path.replace("\\", "/").strip()


def path_to_workflow_glob(path: str) -> str:
    """Map a GROUPS path entry to a GitHub Actions ``on.push.paths`` glob.

    Directory prefixes (trailing slash) become recursive ``/**`` globs so the
    workflow trigger stays aligned with ``_path_matches`` prefix matching.
    """
    normalized = _normalize_group_path(path)
    if not normalized:
        raise ValueError("artifact group path must be non-empty")
    if normalized.endswith("/"):
        return f"{normalized.rstrip('/')}/**"
    return normalized


def workflow_watch_paths(groups: tuple[ArtifactGroup, ...] = GROUPS) -> tuple[str, ...]:
    """Derive the ordered, de-duplicated autobump workflow path list from GROUPS.

    This is the script-side single source of truth for watch paths so the
    workflow trigger cannot silently drift from the bump groups (RR-073).
    """
    seen: set[str] = set()
    ordered: list[str] = []
    for group in groups:
        for path in group.paths:
            glob = path_to_workflow_glob(path)
            if glob in _FORBIDDEN_WORKFLOW_PATHS:
                raise ValueError(
                    f"artifact group path maps to forbidden workflow path: {glob}"
                )
            if glob in seen:
                continue
            seen.add(glob)
            ordered.append(glob)
    return tuple(ordered)


def _get_changed_files(before: str, after: str) -> list[str]:
    if not before or set(before) == {"0"}:
        cmd = ["git", "diff-tree", "--no-commit-id", "--name-only", "-r", after]
    else:
        cmd = ["git", "diff", "--name-only", before, after]
    output = subprocess.check_output(cmd, cwd=ROOT, text=True)
    return [line for line in output.splitlines() if line]


def _path_matches(changed_file: str, prefixes: tuple[str, ...]) -> bool:
    changed = _normalize_group_path(changed_file)
    for prefix in prefixes:
        normalized = _normalize_group_path(prefix)
        if changed == normalized or changed.startswith(normalized):
            return True
    return False


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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Bump AAWM artifact versions based on changed files."
    )
    parser.add_argument(
        "--before",
        help="Previous git SHA from the push event",
    )
    parser.add_argument(
        "--after",
        help="Current git SHA from the push event",
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="Write bumped versions back to disk",
    )
    parser.add_argument(
        "--print-workflow-paths",
        action="store_true",
        help=(
            "Print the GitHub Actions on.push.paths list derived from GROUPS "
            "(one path per line) and exit. Does not require --before/--after."
        ),
    )
    args = parser.parse_args(argv)

    if args.print_workflow_paths:
        for path in workflow_watch_paths():
            print(path)  # noqa: T201
        return 0

    if not args.before or not args.after:
        parser.error("--before and --after are required unless --print-workflow-paths is set")

    changed_files = _get_changed_files(args.before, args.after)
    bumped: list[dict[str, str]] = []

    for group in GROUPS:
        if not any(_path_matches(path, group.paths) for path in changed_files):
            continue
        original = group.version_file.read_text(encoding="utf-8")
        updated, old_version, new_version = _bump_version_text(original, group.pattern)
        if args.write and updated != original:
            group.version_file.write_text(updated, encoding="utf-8")
        try:
            version_file_display = str(group.version_file.relative_to(ROOT))
        except ValueError:
            # Tests (and unusual out-of-tree version files) may use absolute paths.
            version_file_display = str(group.version_file)
        bumped.append(
            {
                "name": group.name,
                "old_version": old_version,
                "new_version": new_version,
                "tag": f"{group.tag_prefix}{new_version}",
                "version_file": version_file_display,
            }
        )

    payload = {
        "changed_files": changed_files,
        "bumped": bumped,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))  # noqa: T201
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
