"""Frozen suite contracts: counts, assertions, timing, and exit classes.

Count rules (headline cases only):

- One case is one declared TUI plus one scenario plus one model or one
  orchestration parent. Shared platform/catalog/log checks are ``shared_checks``.
  Child and tool assertions are ``assertions`` on the parent case. Neither
  increments ``counts.cases.*``.
- ``planned`` is the selected matrix size and never changes.
- ``finished`` is ``passed + failed`` only. ``errored``, ``blocked``,
  ``skipped``, ``running``, and ``incomplete`` stay out of ``finished``.
- Retries add attempts. They do not add cases and do not increment ``passed``
  or ``finished`` more than once per case. The latest terminal attempt is the
  case outcome.
- ``passed`` requires every required assertion on that case to be ``pass``.
  An empty failure list is not a pass.
- A suite exits 0 only when every selected case is ``passed`` and every
  required suite assertion is ``pass``. Validation failures exit 1. Runner or
  setup failures exit 2. A halt, cancel, deadline, or still-running case exits
  3. Skipped required cases and missing terminal evidence cannot exit 0.
"""

from __future__ import annotations

from typing import Any, Mapping

from hv2.suite import PHASES, SCHEMA_VERSION

COUNT_DEFINITIONS: dict[str, str] = {
    "planned": "Selected cases in the resolved matrix. Stable for the run.",
    "launched": "Cases whose TUI launch was attempted.",
    "ready": "Cases whose dedicated session reported ready.",
    "scenario_started": "Cases whose prompt was delivered.",
    "started": "Cases that left planned (launch attempted or later).",
    "executed": "Alias of finished: passed + failed. Retries are not cases.",
    "finished": "passed + failed. Excludes errored, blocked, skipped, running, incomplete.",
    "passed": "Latest attempt passed every required assertion.",
    "failed": "Latest attempt finished and a required assertion failed.",
    "errored": "Launch, readiness, or controller error before a scenario verdict.",
    "blocked": "Declared unsupported client or missing required contract.",
    "skipped": "Explicitly not executed. Required skips forbid exit 0.",
    "running": "Attempt persisted and still in progress.",
    "incomplete": "Selected but not terminal: halt, cancel, deadline, or omitted.",
}


def empty_counts() -> dict[str, int]:
    return {key: 0 for key in COUNT_DEFINITIONS}


def case_counts(cases: list[Mapping[str, Any]]) -> dict[str, int]:
    """Headline counts from current case status. Attempts stay out.

    ``planned`` is the matrix size. Status buckets in ``by_status`` sum to
    that size. ``finished`` and ``executed`` are ``passed + failed`` only.
    """

    by_status = {key: 0 for key in COUNT_DEFINITIONS if key not in {"executed"}}
    by_status["planned"] = 0
    for case in cases:
        status = str(case.get("status") or "planned")
        if status not in by_status or status in {"executed", "finished", "started"}:
            status = "incomplete"
        by_status[status] = by_status.get(status, 0) + 1
    passed = by_status.get("passed", 0)
    failed = by_status.get("failed", 0)
    finished = passed + failed
    started = len(cases) - by_status.get("planned", 0)
    return {
        "planned": len(cases),
        "started": started,
        "executed": finished,
        "finished": finished,
        "passed": passed,
        "failed": failed,
        "errored": by_status.get("errored", 0),
        "blocked": by_status.get("blocked", 0),
        "skipped": by_status.get("skipped", 0),
        "running": by_status.get("running", 0),
        "incomplete": by_status.get("incomplete", 0),
        "launched": by_status.get("launched", 0),
        "ready": by_status.get("ready", 0),
        "scenario_started": by_status.get("scenario_started", 0),
        "by_status": by_status,
    }


def reconcile_manifest(
    matrix_cases: list[Mapping[str, Any]], result_cases: list[Mapping[str, Any]]
) -> dict[str, Any]:
    """Every selected case id must appear exactly once in the result."""

    selected = [str(case.get("case_id")) for case in matrix_cases]
    present = [str(case.get("case_id")) for case in result_cases]
    missing = [item for item in selected if item not in present]
    extra = [item for item in present if item not in selected]
    duplicated = sorted({item for item in present if present.count(item) > 1})
    return {
        "ok": not missing and not extra and not duplicated and len(present) == len(selected),
        "planned": len(selected),
        "represented": len(present),
        "missing": missing,
        "extra": extra,
        "duplicated": duplicated,
    }


def assertion(
    *,
    code: str,
    expected: Any,
    observed: Any,
    status: str,
    evidence_refs: list[Mapping[str, Any]] | None = None,
    required: bool = True,
    case_id: str | None = None,
    detail: str | None = None,
    correlation: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if status not in {"pass", "fail", "inconclusive"}:
        raise ValueError(f"assertion status {status!r} is not pass/fail/inconclusive")
    row: dict[str, Any] = {
        "code": code,
        "expected": expected,
        "observed": observed,
        "status": status,
        "required": required,
        "evidence_refs": list(evidence_refs or []),
        "correlation": dict(correlation) if isinstance(correlation, Mapping) else None,
    }
    if case_id:
        row["case_id"] = case_id
    if detail:
        row["detail"] = detail
    return row


def empty_timing() -> dict[str, Any]:
    return {
        "units": {
            "timestamps": "utc_iso8601",
            "durations": "seconds",
            "monotonic": "seconds",
        },
        "suite": {
            "started_at": None,
            "finished_at": None,
            "wall_seconds": None,
            "monotonic_seconds": None,
        },
        "shared_overhead_seconds": None,
        "exclusive_phase_seconds": {phase: None for phase in PHASES},
        "overlap_seconds": None,
        "unattributed_seconds": None,
        "unavailable": [],
        "tui_harness": [],
        "cases": [],
        "provider_spans": [],
        "tool_spans": [],
        "child_spans": [],
        "notes": [
            "wall_seconds is elapsed suite time.",
            "Nested provider, tool, and child spans are not added into wall_seconds.",
            "Unavailable fields stay null and are listed in unavailable.",
            "Phases are measured, not inferred from tokens or polling budgets.",
        ],
    }


def suite_skeleton() -> dict[str, Any]:
    return {
        "schema": SCHEMA_VERSION,
        "exit_class": None,
        "exit_code": None,
        "ok": False,
        "dry_run": False,
        "identity": {},
        "matrix": {},
        "counts": empty_counts(),
        "count_definitions": dict(COUNT_DEFINITIONS),
        "launches": {"attempts": 0, "ready": 0},
        "cases": [],
        "shared_checks": [],
        "infrastructure_findings": [],
        "assertions": [],
        "successes": [],
        "failures": [],
        "timing": empty_timing(),
        "progress": [],
        "warnings": [],
        "reconciliation": {},
    }
