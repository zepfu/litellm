#!/usr/bin/env python3
"""Write a testing overlay that enables/disables Codex OAuth accounts.

The overlay is JSON of ``{label: bool}``. It contains no secrets and does
not touch auth.json files. ``load_codex_oauth_inventory()`` honors the
overlay on every parse so a process that reloads inventory can switch
account1/account2 without rewriting Compose JSON.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Mapping, Optional

from litellm.secret_managers.codex_oauth_inventory import (
    CODEX_OAUTH_ACCOUNT_ENABLE_FILE_ENV,
    load_codex_oauth_account_enable_overlay,
)

_KNOWN_LABELS = ("account1", "account2")
_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})
_FALSE_VALUES = frozenset({"0", "false", "no", "off"})
_DEFAULT_RELATIVE_PATH = Path(".analysis/runtime/codex_oauth_account_enable.json")


def default_overlay_path(repo_root: Optional[Path] = None) -> Path:
    env_path = os.getenv(CODEX_OAUTH_ACCOUNT_ENABLE_FILE_ENV)
    if isinstance(env_path, str) and env_path.strip():
        return Path(env_path.strip())
    root = repo_root or Path(__file__).resolve().parents[1]
    return root / _DEFAULT_RELATIVE_PATH


def parse_enabled_flag(value: str, *, source: str) -> bool:
    normalized = value.strip().lower()
    if normalized in _TRUE_VALUES:
        return True
    if normalized in _FALSE_VALUES:
        return False
    raise ValueError(f"{source} must be one of on/off/true/false/1/0")


def build_overlay_payload(
    *,
    only: Optional[str] = None,
    account1: Optional[str] = None,
    account2: Optional[str] = None,
) -> dict[str, bool]:
    if only is not None:
        if only not in _KNOWN_LABELS:
            raise ValueError(f"--only must be one of {', '.join(_KNOWN_LABELS)}")
        if account1 is not None or account2 is not None:
            raise ValueError("--only cannot be combined with --account1/--account2")
        return {
            "account1": only == "account1",
            "account2": only == "account2",
        }
    if account1 is None and account2 is None:
        raise ValueError("provide --only, --clear, --show, or --account1/--account2")
    payload: dict[str, bool] = {}
    if account1 is not None:
        payload["account1"] = parse_enabled_flag(account1, source="--account1")
    if account2 is not None:
        payload["account2"] = parse_enabled_flag(account2, source="--account2")
    return payload


def write_overlay(path: Path, payload: Mapping[str, bool]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2) + "\n", encoding="utf-8")
    return path


def clear_overlay(path: Path) -> bool:
    if not path.is_file():
        return False
    path.unlink()
    return True


def show_overlay(path: Path) -> str:
    exists = path.is_file()
    overlay = load_codex_oauth_account_enable_overlay(str(path))
    lines = [
        f"overlay_path={path}",
        f"overlay_exists={'true' if exists else 'false'}",
        f"account1={overlay['account1'] if 'account1' in overlay else 'unset'}",
        f"account2={overlay['account2'] if 'account2' in overlay else 'unset'}",
        f"env_file={CODEX_OAUTH_ACCOUNT_ENABLE_FILE_ENV}",
    ]
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Switch Codex OAuth account1/account2 for testing. "
            "Writes a JSON overlay; never prints tokens or hashes."
        )
    )
    parser.add_argument(
        "--file",
        default=None,
        help=(
            "Overlay path. Default is "
            f"${CODEX_OAUTH_ACCOUNT_ENABLE_FILE_ENV} or "
            f"{_DEFAULT_RELATIVE_PATH}."
        ),
    )
    parser.add_argument(
        "--only",
        choices=_KNOWN_LABELS,
        default=None,
        help="Enable exactly one account and disable the other",
    )
    parser.add_argument("--account1", default=None, help="on/off for account1")
    parser.add_argument("--account2", default=None, help="on/off for account2")
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Delete the overlay so JSON/env defaults apply",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Print overlay path and effective account1/account2 flags",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    path = Path(args.file) if args.file else default_overlay_path()
    try:
        if args.show and not (args.clear or args.only or args.account1 or args.account2):
            sys.stdout.write(show_overlay(path) + "\n")
            return 0
        if args.clear:
            if args.only or args.account1 or args.account2:
                raise ValueError("--clear cannot be combined with enable flags")
            existed = clear_overlay(path)
            sys.stdout.write(
                f"cleared={str(existed).lower()} overlay_path={path}\n"
            )
            return 0
        payload = build_overlay_payload(
            only=args.only,
            account1=args.account1,
            account2=args.account2,
        )
        write_overlay(path, payload)
        sys.stdout.write(show_overlay(path) + "\n")
        return 0
    except ValueError as exc:
        sys.stderr.write(f"{exc}\n")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
