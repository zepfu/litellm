"""Project one suite result into stdout JSON and a readable report."""

from __future__ import annotations

import json
from typing import Any, Mapping

from hv2.suite import EXIT_INCOMPLETE, EXIT_RUNNER, EXIT_SUCCESS, EXIT_VALIDATION
from hv2.suite.schema import COUNT_DEFINITIONS, case_counts, reconcile_manifest

_FAILURE_STATUSES = frozenset({"failed", "errored", "blocked", "incomplete"})
_INCOMPLETE_FLAGS = ("halted", "cancelled", "deadline")


def exit_from_result(result: Mapping[str, Any]) -> tuple[str, int]:
    """Stable classes: success 0, validation 1, runner/setup 2, incomplete 3.

    Exit 0 is impossible when a selected case is missing, still running,
    skipped, or otherwise not passed. Required suite assertions must pass.
    """

    if result.get("runner_error") or result.get("setup_error"):
        return "runner", EXIT_RUNNER
    counts = result.get("counts") if isinstance(result.get("counts"), Mapping) else {}
    reconciliation = (
        result.get("reconciliation") if isinstance(result.get("reconciliation"), Mapping) else {}
    )
    if any(result.get(flag) for flag in _INCOMPLETE_FLAGS):
        return "incomplete", EXIT_INCOMPLETE
    if int(counts.get("running") or 0) or int(counts.get("incomplete") or 0):
        return "incomplete", EXIT_INCOMPLETE
    if reconciliation and reconciliation.get("ok") is not True:
        return "incomplete", EXIT_INCOMPLETE
    if reconciliation.get("missing"):
        return "incomplete", EXIT_INCOMPLETE
    if int(counts.get("skipped") or 0):
        return "incomplete", EXIT_INCOMPLETE
    assertions = result.get("assertions") if isinstance(result.get("assertions"), list) else []
    required_failed = any(_required_suite_failed(row) for row in assertions)
    failures = result.get("failures") if isinstance(result.get("failures"), list) else []
    checks_failed = result.get("suite_checks_failed") is True or _failed_shared_checks(result)
    if (
        int(counts.get("failed") or 0)
        or int(counts.get("errored") or 0)
        or int(counts.get("blocked") or 0)
        or failures
        or required_failed
        or checks_failed
    ):
        return "validation", EXIT_VALIDATION
    planned = int(counts.get("planned") or 0)
    passed = int(counts.get("passed") or 0)
    # A dry-run prints the resolved matrix and does not execute cases.
    # An executed run still cannot exit 0 while any selected case is unpassed.
    if (
        result.get("dry_run")
        and reconciliation.get("ok") is True
        and not reconciliation.get("missing")
    ):
        return "success", EXIT_SUCCESS
    if planned > 0 and passed == planned and not _non_pass_residue(counts):
        return "success", EXIT_SUCCESS
    return "incomplete", EXIT_INCOMPLETE


def _failed_shared_checks(result: Mapping[str, Any]) -> bool:
    """A failed required shared check is not a case assertion and not exit 0.

    Live and fixture rows stay skipped or recorded. Docker findings are not
    invented here; only a classified finding whose ok is false counts.
    """

    findings = result.get("infrastructure_findings")
    if isinstance(findings, list) and any(
        isinstance(item, Mapping) and item.get("ok") is False for item in findings
    ):
        return True
    shared = result.get("shared_checks")
    if isinstance(shared, list) and any(
        isinstance(row, Mapping) and row.get("status") == "failed" for row in shared
    ):
        return True
    return False


def _required_suite_failed(row: Any) -> bool:
    if not isinstance(row, Mapping):
        return False
    if row.get("scope") != "suite":
        return False
    if row.get("required", True) is False:
        return False
    return row.get("status") != "pass"


def _non_pass_residue(counts: Mapping[str, Any]) -> bool:
    keys = (
        "failed",
        "errored",
        "blocked",
        "skipped",
        "running",
        "incomplete",
        "launched",
        "ready",
        "scenario_started",
    )
    return any(int(counts.get(key) or 0) for key in keys)


def project(result: Mapping[str, Any]) -> dict[str, Any]:
    """Return the canonical result. Callers print this object, not a second verdict.

    Counts are recomputed from case status. A matrix with cases is reconciled
    against those cases. Retries and child assertions do not add cases.
    Infrastructure findings stay suite-level; a missing machine correlation
    is not rewritten into a provider incident.
    """

    body = dict(result)
    cases = [dict(item) for item in body.get("cases") or [] if isinstance(item, Mapping)]
    body["cases"] = cases
    matrix = body.get("matrix") if isinstance(body.get("matrix"), Mapping) else {}
    matrix_cases = []
    if isinstance(matrix.get("cases"), list):
        matrix_cases = [item for item in matrix["cases"] if isinstance(item, Mapping)]
    counts = case_counts(cases)
    body["counts"] = counts
    body["count_definitions"] = dict(body.get("count_definitions") or COUNT_DEFINITIONS)
    if matrix_cases:
        body["reconciliation"] = reconcile_manifest(matrix_cases, cases)
    body["counts_by_tui"] = _counts_by_tui(cases)
    body["successes"] = [_success_row(case) for case in cases if case.get("status") == "passed"]
    body["failures"] = [
        _failure_row(case) for case in cases if case.get("status") in _FAILURE_STATUSES
    ]
    body["infrastructure_findings"] = _infrastructure(body.get("infrastructure_findings"))
    body["suite_checks_failed"] = body.get("suite_checks_failed") is True or _failed_shared_checks(
        body
    )
    exit_class, exit_code = exit_from_result(body)
    body["exit_class"] = exit_class
    body["exit_code"] = exit_code
    body["ok"] = exit_code == EXIT_SUCCESS
    return body


def _counts_by_tui(cases: list[Mapping[str, Any]]) -> dict[str, dict[str, int]]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for case in cases:
        grouped.setdefault(str(case.get("tui") or "-"), []).append(case)
    return {tui: case_counts(rows) for tui, rows in grouped.items()}


def _success_row(case: Mapping[str, Any]) -> dict[str, Any]:
    passing = [
        row.get("code")
        for row in (case.get("assertions") or [])
        if isinstance(row, Mapping) and row.get("status") == "pass" and row.get("code")
    ]
    return {
        "case_id": case.get("case_id"),
        "tui": case.get("tui"),
        "alias": case.get("alias"),
        "attempt_id": case.get("attempt_id"),
        "assertions": passing,
        "evidence_refs": _passing_evidence(case),
    }


def _passing_evidence(case: Mapping[str, Any]) -> list[Any]:
    refs: list[Any] = []
    for row in case.get("assertions") or []:
        if not isinstance(row, Mapping) or row.get("status") != "pass":
            continue
        for ref in row.get("evidence_refs") or []:
            if ref not in refs:
                refs.append(ref)
    return refs


def _failure_row(case: Mapping[str, Any]) -> dict[str, Any]:
    failing = [
        {
            "code": row.get("code"),
            "expected": row.get("expected"),
            "observed": row.get("observed"),
            "status": row.get("status"),
            "evidence_refs": list(row.get("evidence_refs") or []),
            "detail": row.get("detail"),
        }
        for row in (case.get("assertions") or [])
        if isinstance(row, Mapping) and row.get("status") != "pass"
    ]
    error = case.get("error") if isinstance(case.get("error"), Mapping) else None
    return {
        "case_id": case.get("case_id"),
        "tui": case.get("tui"),
        "alias": case.get("alias"),
        "attempt_id": case.get("attempt_id"),
        "status": case.get("status"),
        "assertions": failing,
        "error": dict(error) if error else None,
        "correlation": case.get("correlation"),
    }


def _infrastructure(raw: Any) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    if not isinstance(raw, list):
        return findings
    for item in raw:
        if not isinstance(item, Mapping):
            continue
        causal = item.get("causal_case_id")
        findings.append(
            {
                "class": item.get("class") or "infrastructure",
                "source": item.get("source"),
                "ok": item.get("ok"),
                "causal_case_id": causal if causal else None,
                "failures": list(item.get("failures") or []),
                "detail": item.get("detail"),
            }
        )
    return findings


def render_text(result: Mapping[str, Any]) -> str:
    counts = result.get("counts") if isinstance(result.get("counts"), Mapping) else {}
    lines = [
        f"harnessv2 suite {result.get('exit_class')} (exit {result.get('exit_code')})",
        (
            "cases planned={planned} started={started} executed={executed} "
            "finished={finished} passed={passed} failed={failed} "
            "errored={errored} blocked={blocked} skipped={skipped} "
            "running={running} incomplete={incomplete}"
        ).format(
            planned=counts.get("planned"),
            started=counts.get("started"),
            executed=counts.get("executed"),
            finished=counts.get("finished"),
            passed=counts.get("passed"),
            failed=counts.get("failed"),
            errored=counts.get("errored"),
            blocked=counts.get("blocked"),
            skipped=counts.get("skipped"),
            running=counts.get("running"),
            incomplete=counts.get("incomplete"),
        ),
        "finished counts passed+failed only; retries and child/shared checks are not cases.",
    ]
    identity = result.get("identity") if isinstance(result.get("identity"), Mapping) else {}
    if identity:
        lines.append(
            "identity commit={commit} config_hash={config} container={container} runtime={runtime}".format(
                commit=identity.get("source_commit"),
                config=identity.get("config_hash"),
                container=identity.get("container"),
                runtime=identity.get("runtime"),
            )
        )
    launches = result.get("launches") if isinstance(result.get("launches"), Mapping) else {}
    lines.append(f"launches attempts={launches.get('attempts')} ready={launches.get('ready')}")
    by_tui = result.get("counts_by_tui") if isinstance(result.get("counts_by_tui"), Mapping) else {}
    for tui, row in by_tui.items():
        if not isinstance(row, Mapping):
            continue
        lines.append(
            f"tui {tui} planned={row.get('planned')} finished={row.get('finished')} "
            f"passed={row.get('passed')} failed={row.get('failed')}"
        )
    for case in result.get("cases") or []:
        if not isinstance(case, Mapping):
            continue
        lines.append(
            f"case {case.get('case_id')} {case.get('tui')} {case.get('kind')} "
            f"{case.get('alias') or case.get('parent') or '-'} {case.get('status')}"
        )
    for success in result.get("successes") or []:
        if not isinstance(success, Mapping):
            continue
        codes = ",".join(str(code) for code in (success.get("assertions") or []) if code)
        lines.append(
            f"success {success.get('case_id')} {success.get('tui')} "
            f"{success.get('alias') or '-'} assertions={codes or '-'}"
        )
    for failure in result.get("failures") or []:
        if not isinstance(failure, Mapping):
            continue
        lines.append(
            f"failure {failure.get('case_id')} {failure.get('tui')} "
            f"{failure.get('alias') or '-'} attempt={failure.get('attempt_id')} "
            f"status={failure.get('status')}"
        )
        error = failure.get("error") if isinstance(failure.get("error"), Mapping) else None
        if error:
            lines.append(
                "  error reason={reason} detail={detail}".format(
                    reason=error.get("reason"),
                    detail=error.get("detail"),
                )
            )
        if failure.get("correlation") is not None:
            lines.append(f"  correlation {failure.get('correlation')}")
        for row in failure.get("assertions") or []:
            if not isinstance(row, Mapping):
                continue
            lines.append(
                f"  assertion {row.get('code')} expected={row.get('expected')} "
                f"observed={row.get('observed')}"
            )
            if row.get("evidence_refs"):
                lines.append(f"  evidence {row.get('evidence_refs')}")
    for finding in result.get("infrastructure_findings") or []:
        if isinstance(finding, Mapping):
            lines.append(
                f"infrastructure {finding.get('source')} ok={finding.get('ok')} "
                f"causal_case_id={finding.get('causal_case_id')}"
            )
    reconciliation = (
        result.get("reconciliation") if isinstance(result.get("reconciliation"), Mapping) else {}
    )
    if reconciliation:
        lines.append(
            "reconciliation ok={ok} missing={missing} extra={extra} duplicated={duplicated}".format(
                ok=reconciliation.get("ok"),
                missing=reconciliation.get("missing"),
                extra=reconciliation.get("extra"),
                duplicated=reconciliation.get("duplicated"),
            )
        )
    lines.extend(_timing_lines(result.get("timing")))
    return "\n".join(lines) + "\n"


def _timing_lines(timing: Any) -> list[str]:
    body = timing if isinstance(timing, Mapping) else {}
    suite = body.get("suite") if isinstance(body.get("suite"), Mapping) else {}
    lines = [
        "timing wall_seconds={wall} nested_span_sum_seconds={nested} "
        "unattributed_seconds={unattributed} overlap_seconds={overlap} "
        "shared_overhead_seconds={shared}".format(
            wall=suite.get("wall_seconds"),
            nested=body.get("nested_span_sum_seconds"),
            unattributed=body.get("unattributed_seconds"),
            overlap=body.get("overlap_seconds"),
            shared=body.get("shared_overhead_seconds"),
        )
    ]
    for row in body.get("tui_harness") or []:
        if not isinstance(row, Mapping):
            continue
        lines.append(
            f"timing tui {row.get('tui')} duration_seconds={row.get('duration_seconds')}"
        )
    unavailable = body.get("unavailable") if isinstance(body.get("unavailable"), list) else []
    if unavailable:
        lines.append("timing unavailable: " + ", ".join(str(item) for item in unavailable))
    elif body:
        lines.append("timing unavailable: none")
    return lines


def dumps(result: Mapping[str, Any]) -> str:
    return json.dumps(result, indent=2, sort_keys=True, default=str) + "\n"
