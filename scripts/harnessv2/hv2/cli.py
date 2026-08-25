"""Argparse for harness v2. Values come from YAML, not a Python enum.

The runnable entry is ``scripts/harnessv2/run.py``. This module only
parses flags; it does not inspect Docker or send HTTP.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="harnessv2",
        description=(
            "YAML/JSON-first LiteLLM acceptance harness. "
            "Ohmypi and Codex are implemented TUIs. Claude is out of scope. "
            "--instance is a Docker container name; host port comes from "
            "docker inspect. Never targets aawm-litellm (:4000) or "
            "litellm-dev (:4001)."
        ),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Root YAML/JSON (default scripts/harnessv2/config/harness.yaml)",
    )
    parser.add_argument(
        "--overlay",
        type=Path,
        default=None,
        help="YAML/JSON deep-merged on top of the loaded config",
    )
    instance = parser.add_mutually_exclusive_group()
    instance.add_argument(
        "--instance",
        dest="instance",
        default=None,
        help="Docker container name or YAML alias (default from targets.yaml)",
    )
    instance.add_argument(
        "--container",
        dest="instance",
        default=None,
        help="Same as --instance",
    )
    instance.add_argument(
        "--target",
        dest="instance",
        default=None,
        help="Same as --instance",
    )
    parser.add_argument(
        "--test",
        dest="test",
        required=True,
        help="Kind name from kinds.yaml (platform, catalog, model, orchestration)",
    )
    parser.add_argument(
        "--tui",
        default=None,
        help="TUI name from tuis.yaml (required when the kind requires a TUI)",
    )
    parser.add_argument(
        "--model",
        action="append",
        default=None,
        help="Model id or group (repeatable / comma-separated). Default from YAML.",
    )
    parser.add_argument(
        "--orchestration-parent",
        default=None,
        help="Parent alias or group (default from kinds.yaml / models.yaml)",
    )
    parser.add_argument(
        "--orchestration-children",
        default=None,
        help="Child aliases or group (default orchestration_children)",
    )
    parser.add_argument(
        "--write-artifact",
        type=Path,
        default=None,
        help="Write a JSON artifact to this path",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved plan and exit 0. No TUI, HTTP, or docker logs.",
    )
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def split_csv(values: Sequence[str] | None) -> list[str]:
    if not values:
        return []
    out: list[str] = []
    for item in values:
        for part in str(item).split(","):
            token = part.strip()
            if token:
                out.append(token)
    return out
