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
from hv2.suite.live import interactive_runner  # noqa: E402
from hv2.suite.matrix import selection_from_args  # noqa: E402
from hv2.suite.report import dumps  # noqa: E402


def _progress(event: Any) -> None:
    if not isinstance(event, dict):
        return
    name = event.get("event")
    case_id = event.get("case_id")
    status = event.get("status")
    if name is None or case_id is None or status is None:
        return
    sys.stderr.write(f"progress {name} {case_id} {status}\n")


def _load_suite_evidence(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise HarnessError(f"suite evidence must be a JSON object: {path}")
    evidence: dict[str, Any] = {}
    for key, value in payload.items():
        if not isinstance(value, dict):
            raise HarnessError(f"suite evidence for {key!r} must be a JSON object")
        evidence[str(key)] = value
    return evidence


def _run_suite(args: Any, config: dict) -> int:
    selection = selection_from_args(
        tuis=split_csv(args.suite_tui),
        kinds=split_csv(args.suite_kind),
        models=split_csv(args.suite_model),
        parents=split_csv(args.suite_parent),
        children=split_csv([args.suite_children]) if args.suite_children else None,
        include_shared=bool(args.suite_shared),
    )
    evidence_path = getattr(args, "suite_evidence", None)
    if args.dry_run:
        kwargs: dict[str, Any] = {"dry_run": True}
    elif evidence_path is not None:
        kwargs = {
            "dry_run": False,
            "live": False,
            "evidence_by_case": _load_suite_evidence(evidence_path),
        }
    else:
        kwargs = {
            "dry_run": False,
            "live": True,
            "runner": interactive_runner(config),
        }
    result = execute_suite(
        config,
        selection,
        instance_token=args.instance,
        write_path=args.write_artifact,
        report_path=args.suite_report,
        state_dir=args.suite_state_dir,
        resume=bool(args.suite_resume),
        progress=_progress,
        **kwargs,
    )
    sys.stdout.write(dumps(result))
    return int(result["exit_code"])


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
