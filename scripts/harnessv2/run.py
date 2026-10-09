#!/usr/bin/env python3
"""Harness v2 CLI entry. Thin interpreter over YAML/JSON."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

HARNESS_DIR = Path(__file__).resolve().parent
if str(HARNESS_DIR) not in sys.path:
    sys.path.insert(0, str(HARNESS_DIR))

from hv2.cli import parse_args, split_csv  # noqa: E402
from hv2.errors import HarnessError  # noqa: E402
from hv2.kinds.runner import run_plan  # noqa: E402
from hv2.load_config import load_config  # noqa: E402
from hv2.plan import build_plan  # noqa: E402
from hv2.suite.execute import execute_suite  # noqa: E402
from hv2.suite.matrix import selection_from_args  # noqa: E402
from hv2.suite.report import dumps  # noqa: E402


def _run_suite(args: Any, config: dict) -> int:
    selection = selection_from_args(
        tuis=split_csv(args.suite_tui),
        kinds=split_csv(args.suite_kind),
        models=split_csv(args.suite_model),
        parents=split_csv(args.suite_parent),
        children=split_csv([args.suite_children]) if args.suite_children else None,
        include_shared=bool(args.suite_shared),
    )
    result = execute_suite(
        config,
        selection,
        instance_token=args.instance,
        dry_run=True,
    )
    sys.stdout.write(dumps(result))
    if result.get("dry_run"):
        planned = int((result.get("counts") or {}).get("planned") or 0)
        launches = result.get("launches") if isinstance(result.get("launches"), dict) else {}
        if planned > 0 and int(launches.get("attempts") or 0) == 0 and not result.get("runner_error"):
            return 0
    return int(result.get("exit_code") or 0)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        config = load_config(args.config, overlay=args.overlay)
        if args.suite:
            return _run_suite(args, config)
        if not args.test:
            sys.stderr.write("harnessv2: --test is required unless --suite is set\n")
            return 2
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
