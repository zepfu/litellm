"""Suite selection, matrix, verdicts, timing, and one-result reports.

The single-kind ``--test`` path stays in ``hv2.plan`` / ``hv2.kinds.runner``.
This package adds the explicit multi-TUI suite beside it. Pure resolution,
assertion evaluation, and timing reconciliation do not launch a TUI.
"""

from __future__ import annotations

SCHEMA_VERSION = "harnessv2.suite.v1"

EXIT_SUCCESS = 0
EXIT_VALIDATION = 1
EXIT_RUNNER = 2
EXIT_INCOMPLETE = 3

CASE_STATUSES = (
    "planned",
    "launched",
    "ready",
    "scenario_started",
    "running",
    "passed",
    "failed",
    "errored",
    "blocked",
    "skipped",
    "incomplete",
)

TERMINAL_CASE_STATUSES = frozenset(
    {"passed", "failed", "errored", "blocked", "skipped", "incomplete"}
)
FINISHED_CASE_STATUSES = frozenset({"passed", "failed"})
NONTERMINAL_CASE_STATUSES = frozenset(
    {"planned", "launched", "ready", "scenario_started", "running"}
)

PHASES = (
    "preparation",
    "launch_readiness",
    "prompt_delivery",
    "response_execution",
    "validation",
    "evidence_collection",
    "cleanup_retention",
)

ASSERTION_STATUSES = ("pass", "fail", "inconclusive")

__all__ = [
    "ASSERTION_STATUSES",
    "CASE_STATUSES",
    "EXIT_INCOMPLETE",
    "EXIT_RUNNER",
    "EXIT_SUCCESS",
    "EXIT_VALIDATION",
    "FINISHED_CASE_STATUSES",
    "NONTERMINAL_CASE_STATUSES",
    "PHASES",
    "SCHEMA_VERSION",
    "TERMINAL_CASE_STATUSES",
]
