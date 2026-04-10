#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Any


def _load(path: pathlib.Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _compare_family(name: str, baseline: dict[str, Any], candidate: dict[str, Any]) -> tuple[list[str], list[str]]:
    hard: list[str] = []
    soft: list[str] = []

    if baseline.get("passed") and not candidate.get("passed"):
        hard.append(f"{name}: candidate failed but baseline passed")

    baseline_langfuse = baseline.get("langfuse", {})
    candidate_langfuse = candidate.get("langfuse", {})

    required_names = baseline_langfuse.get("required_trace_names") or baseline_langfuse.get(
        "expected_trace_names", []
    )
    candidate_names = set(candidate_langfuse.get("actual_trace_names", []))
    for trace_name in required_names:
        if trace_name not in candidate_names:
            hard.append(f"{name}: missing required trace name {trace_name}")

    expected_user_ids = baseline_langfuse.get("expected_user_ids", [])
    candidate_user_ids = set(candidate_langfuse.get("actual_user_ids", []))
    for user_id in expected_user_ids:
        if user_id not in candidate_user_ids:
            hard.append(f"{name}: missing expected user id {user_id}")

    if name == "claude":
        non_orchestrator = [trace for trace in candidate_names if trace != "claude-code.orchestrator"]
        if not non_orchestrator:
            hard.append("claude: missing persona/subagent traces")

    baseline_count = int(baseline_langfuse.get("trace_count", 0))
    candidate_count = int(candidate_langfuse.get("trace_count", 0))
    if baseline_count and candidate_count < baseline_count:
        soft.append(
            f"{name}: trace_count dropped from {baseline_count} to {candidate_count}"
        )

    baseline_excerpt = baseline.get("response_excerpt", "")
    candidate_excerpt = candidate.get("response_excerpt", "")
    if baseline_excerpt and candidate_excerpt and baseline_excerpt != candidate_excerpt:
        soft.append(f"{name}: response excerpt changed")

    return hard, soft


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare two local acceptance artifacts.")
    parser.add_argument("baseline", help="Baseline JSON artifact")
    parser.add_argument("candidate", help="Candidate JSON artifact")
    args = parser.parse_args()

    baseline = _load(pathlib.Path(args.baseline))
    candidate = _load(pathlib.Path(args.candidate))

    hard_failures: list[str] = []
    soft_drift: list[str] = []

    for family in ("codex", "gemini", "claude"):
        base_family = baseline.get("results", {}).get(family, {})
        cand_family = candidate.get("results", {}).get(family, {})
        hard, soft = _compare_family(family, base_family, cand_family)
        hard_failures.extend(hard)
        soft_drift.extend(soft)

    report = {
        "passed": not hard_failures,
        "hard_failures": hard_failures,
        "soft_drift": soft_drift,
    }
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
