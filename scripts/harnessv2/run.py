#!/usr/bin/env python3
"""Harness v2 CLI entry. Thin interpreter over YAML/JSON."""

from __future__ import annotations

import json
import sys
from pathlib import Path

HARNESS_DIR = Path(__file__).resolve().parent
if str(HARNESS_DIR) not in sys.path:
    sys.path.insert(0, str(HARNESS_DIR))

from hv2.cli import parse_args  # noqa: E402
from hv2.errors import HarnessError  # noqa: E402
from hv2.kinds.runner import run_plan  # noqa: E402
from hv2.load_config import load_config  # noqa: E402
from hv2.plan import build_plan  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        config = load_config(args.config, overlay=args.overlay)
        plan = build_plan(
            config=config,
            kind=str(args.test),
            instance_token=args.instance,
            tui=args.tui,
            models=args.model,
            orchestration_parent=args.orchestration_parent,
            orchestration_children=args.orchestration_children,
            dry_run=bool(args.dry_run),
            write_artifact=args.write_artifact,
        )
        if plan.dry_run:
            artifact = run_plan(plan)
            json.dump(artifact["plan"], sys.stdout, indent=2)
            sys.stdout.write("\n")
            return 0
        artifact = run_plan(plan)
        if artifact.get("ok"):
            return 0
        for item in artifact.get("failures") or []:
            sys.stderr.write(f"FAIL: {item}\n")
        return 1
    except HarnessError as exc:
        sys.stderr.write(f"harnessv2: {exc}\n")
        return int(getattr(exc, "exit_code", 2) or 2)


if __name__ == "__main__":
    raise SystemExit(main())
