"""Suite controller. Fixture evidence executes without launching a TUI.

Live interactive execution calls the injected runner once per selected case
that is not resumed. State is written before each case is evaluated. A launch
failure, halt, or controller exception records every remaining selected case
as incomplete instead of dropping it.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable, Mapping

from hv2.artifact import utc_now_iso, write_artifact
from hv2.errors import HarnessError
from hv2.suite.matrix import resolve_matrix
from hv2.suite.report import dumps, project, render_text
from hv2.suite.schema import suite_skeleton
from hv2.suite.timing import reconcile, utc_now
from hv2.suite.verdict import (
    case_status_from_assertions,
    classify_infrastructure,
    evaluate_case,
)

ProgressFn = Callable[[Mapping[str, Any]], None]
RunnerFn = Callable[[Mapping[str, Any]], Mapping[str, Any]]


class _SuiteSetupError(HarnessError):
    """Live mode was asked to run without an injected runner."""


def _state_path(root: Path, run_id: str) -> Path:
    return root / f"{run_id}.state.json"


def _contract_fingerprint(matrix: Mapping[str, Any]) -> str:
    identity = (
        matrix.get("identity") if isinstance(matrix.get("identity"), Mapping) else {}
    )
    return (
        str(identity.get("config_hash") or "")
        + ":"
        + str(identity.get("source_commit") or "")
    )


def load_resume(path: Path, matrix: Mapping[str, Any]) -> dict[str, Any] | None:
    """Load prior state. A changed config_hash:source_commit invalidates passes."""

    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return None
    if payload.get("contract") != _contract_fingerprint(matrix):
        payload["resume_invalidated"] = True
        payload["cases"] = []
        return payload
    return payload


def _persist(path: Path, state: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(state, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def _phase(
    case_id: str, attempt_id: str, name: str, start: float, end: float
) -> dict[str, Any]:
    return {
        "case_id": case_id,
        "attempt_id": attempt_id,
        "phase": name,
        "start": start,
        "end": end,
    }


def _state_snapshot(
    contract: str,
    cases_out: list[dict[str, Any]],
    pending: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    rows = list(cases_out)
    if pending is not None:
        rows.append(dict(pending))
    return {"contract": contract, "cases": rows}


def _record_retry(
    case: dict[str, Any], policy: Mapping[str, Any], *, reason: str
) -> None:
    """A retry is another attempt on the same case. It does not add a case."""

    budget = int(policy.get("retry_budget") or 0)
    if budget <= 0 or case.get("status") == "passed":
        return
    case_id = str(case.get("case_id") or "")
    attempts = list(case.get("attempts") or [])
    attempts.append(
        {
            "attempt_id": f"{case_id}:{len(attempts) + 1}",
            "status": "recorded",
            "reason": reason,
        }
    )
    case["attempts"] = attempts


def _mark_incomplete(
    case: dict[str, Any], reason: str, detail: str | None = None
) -> dict[str, Any]:
    case["status"] = "incomplete"
    error: dict[str, Any] = {"reason": reason}
    if detail:
        error["detail"] = detail
    case["error"] = error
    return case


def _publish(
    result: dict[str, Any], progress: ProgressFn | None, event: Mapping[str, Any]
) -> None:
    result["progress"].append(dict(event))
    if progress is not None:
        progress(event)


def _evaluate_evidence(
    case: dict[str, Any],
    evidence: Mapping[str, Any] | None,
    phases: list[dict[str, Any]],
    *,
    phase_origin: float,
) -> None:
    case_id = str(case["case_id"])
    attempt_id = str(case["attempt_id"])
    phase_ready = time.monotonic()
    phases.append(
        _phase(case_id, attempt_id, "preparation", phase_origin, phase_origin)
    )
    phases.append(
        _phase(case_id, attempt_id, "launch_readiness", phase_origin, phase_ready)
    )
    phases.append(
        _phase(case_id, attempt_id, "prompt_delivery", phase_ready, phase_ready)
    )
    response_end = time.monotonic()
    phases.append(
        _phase(case_id, attempt_id, "response_execution", phase_ready, response_end)
    )
    assertions = evaluate_case(
        case, evidence if isinstance(evidence, Mapping) else None
    )
    validate_end = time.monotonic()
    phases.append(_phase(case_id, attempt_id, "validation", response_end, validate_end))
    phases.append(
        _phase(case_id, attempt_id, "evidence_collection", validate_end, validate_end)
    )
    phases.append(
        _phase(case_id, attempt_id, "cleanup_retention", validate_end, time.monotonic())
    )
    case["assertions"] = assertions
    case["evidence"] = evidence
    case["status"] = case_status_from_assertions(assertions, executed=True)
    case["attempts"] = [
        {
            "attempt_id": attempt_id,
            "status": case["status"],
            "finished_at": utc_now_iso(),
        }
    ]


def _reuse_passed(
    case_id: str, prior_cases: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any] | None:
    prior = prior_cases.get(case_id)
    if prior is None:
        return None
    reused = dict(prior)
    reused["resumed"] = True
    reused["status"] = "passed"
    return reused


def _positive_seconds(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    seconds = float(value)
    if seconds <= 0:
        return None
    return seconds


def _suite_deadline(policy: Mapping[str, Any]) -> float | None:
    seconds = _positive_seconds(policy.get("deadline_seconds"))
    if seconds is None:
        return None
    return time.monotonic() + seconds


def _guard_case(
    case: dict[str, Any],
    *,
    live: bool,
    runner: RunnerFn | None,
    evidence_map: Mapping[str, Mapping[str, Any]],
    phases: list[dict[str, Any]],
    launches: dict[str, int],
    policy: Mapping[str, Any],
) -> tuple[str | None, str | None, bool]:
    """Run one case. Returns (halt_token, controller_error, case_deadline)."""

    started = time.monotonic()
    try:
        stop = _run_case(
            case,
            live=live,
            runner=runner,
            evidence_map=evidence_map,
            phases=phases,
            launches=launches,
            policy=policy,
        )
    except _SuiteSetupError:
        raise
    except HarnessError as exc:
        detail = str(exc)
        _mark_incomplete(case, "controller", detail)
        return "controller", detail, False
    except Exception as exc:
        detail = str(exc)
        _mark_incomplete(case, "controller", detail)
        return "controller", detail, False
    limit = _positive_seconds(policy.get("case_timeout_seconds"))
    if limit is not None and (time.monotonic() - started) > limit:
        _mark_incomplete(case, "deadline", "case_timeout_seconds exceeded")
        return "deadline", None, True
    return stop, None, False


def _finish_suite(
    result: dict[str, Any],
    *,
    matrix: Mapping[str, Any],
    selected: list[dict[str, Any]],
    cases_out: list[dict[str, Any]],
    phases: list[dict[str, Any]],
    launches: Mapping[str, int],
    live: bool,
    halted: bool,
    runner_error: str | None,
    infrastructure: list[Mapping[str, Any]] | None,
    started_mono: float,
    started_at: str,
    progress: ProgressFn | None,
) -> None:
    represented = {str(row.get("case_id")) for row in cases_out}
    for planned in selected:
        case_id = str(planned["case_id"])
        if case_id in represented:
            continue
        _mark_incomplete(planned, "omitted", "selected case was not recorded")
        cases_out.append(planned)
        _publish(
            result,
            progress,
            {"event": "case", "case_id": case_id, "status": "incomplete"},
        )
    shared_rows = []
    for check in matrix["shared_checks"]:
        row = dict(check)
        row["status"] = "skipped" if live else "recorded"
        row["counts_as_case"] = False
        shared_rows.append(row)
    result["cases"] = cases_out
    result["shared_checks"] = shared_rows
    result["infrastructure_findings"] = [
        classify_infrastructure(str(item.get("source") or "shared"), item)
        for item in (infrastructure or [])
    ]
    result["assertions"] = [
        row
        for case in cases_out
        for row in (case.get("assertions") or [])
        if isinstance(row, Mapping)
    ]
    result["launches"] = {"attempts": launches["attempts"], "ready": launches["ready"]}
    result["halted"] = halted
    result["runner_error"] = runner_error
    result["suite_checks_failed"] = _suite_checks_failed(
        result["shared_checks"], result["infrastructure_findings"]
    )
    result["timing"] = reconcile(
        suite_start_mono=started_mono,
        suite_end_mono=time.monotonic(),
        suite_started_at=started_at,
        suite_finished_at=utc_now(),
        phases=phases,
        nested_spans=[],
        shared_overhead_seconds=0.0,
    )
    result["identity"]["runtime"] = "fixture" if not live else "interactive"


def _run_case(
    case: dict[str, Any],
    *,
    live: bool,
    runner: RunnerFn | None,
    evidence_map: Mapping[str, Mapping[str, Any]],
    phases: list[dict[str, Any]],
    launches: dict[str, int],
    policy: Mapping[str, Any],
) -> str | None:
    """Evaluate one case. Returns a halt reason, or None to continue."""

    case_id = str(case["case_id"])
    attempt_id = str(case["attempt_id"])
    case["status"] = "running"
    case["attempts"] = [
        {"attempt_id": attempt_id, "status": "running", "started_at": utc_now_iso()}
    ]
    phase_origin = time.monotonic()
    evidence: Mapping[str, Any] | None = None
    if live:
        if runner is None:
            raise _SuiteSetupError("live suite requires a runner")
        launches["attempts"] += 1
        case["status"] = "launched"
        try:
            payload = runner(case)
        except HarnessError:
            raise
        except Exception as exc:
            payload = {"launch_error": str(exc)}
        if not isinstance(payload, Mapping):
            payload = {"launch_error": "runner returned a non-mapping"}
        if payload.get("ready"):
            launches["ready"] += 1
            case["status"] = "ready"
        raw_evidence = payload.get("evidence")
        if isinstance(raw_evidence, Mapping):
            evidence = raw_evidence
        launch_error = payload.get("launch_error")
        if launch_error:
            case["status"] = "errored"
            case["error"] = {"reason": "launch", "detail": launch_error}
            case["attempts"] = [
                {
                    "attempt_id": attempt_id,
                    "status": "errored",
                    "finished_at": utc_now_iso(),
                }
            ]
            _record_retry(case, policy, reason="retry_budget_available_not_relaunched")
            return "fail_fast" if policy.get("fail_fast") else None
    else:
        evidence = evidence_map.get(case_id)
        if evidence is None:
            alias = case.get("alias")
            if alias:
                evidence = evidence_map.get(str(alias))
    _evaluate_evidence(case, evidence, phases, phase_origin=phase_origin)
    _record_retry(
        case, policy, reason="retry_budget_available_not_auto_launched_in_fixture"
    )
    return None


def _suite_checks_failed(
    shared_rows: list[Mapping[str, Any]],
    findings: list[Mapping[str, Any]],
) -> bool:
    """Required shared checks are suite-level. They do not invent a case cause."""

    if any(item.get("ok") is False for item in findings):
        return True
    return any(row.get("status") == "failed" for row in shared_rows)


def _prepare_running(case: dict[str, Any]) -> None:
    case["status"] = "running"
    case["attempts"] = [
        {
            "attempt_id": str(case["attempt_id"]),
            "status": "running",
            "started_at": utc_now_iso(),
        }
    ]


def _walk_cases(
    *,
    result: dict[str, Any],
    selected: list[dict[str, Any]],
    prior_cases: Mapping[str, Mapping[str, Any]],
    contract: str,
    state_file: Path,
    live: bool,
    runner: RunnerFn | None,
    evidence_map: Mapping[str, Mapping[str, Any]],
    phases: list[dict[str, Any]],
    launches: dict[str, int],
    policy: Mapping[str, Any],
    progress: ProgressFn | None,
    cancelled: bool = False,
) -> tuple[list[dict[str, Any]], bool, str | None]:
    cases_out: list[dict[str, Any]] = []
    halted = False
    halt_reason = "halted_before_start"
    halt_detail: str | None = None
    runner_error: str | None = None
    deadline = _suite_deadline(policy)
    if cancelled:
        result["cancelled"] = True
        halted = True
        halt_reason = "cancelled"
    for case in selected:
        case_id = str(case["case_id"])
        reused = _reuse_passed(case_id, prior_cases)
        if reused is not None:
            cases_out.append(reused)
            _publish(
                result,
                progress,
                {
                    "event": "case",
                    "case_id": case_id,
                    "status": "passed",
                    "resumed": True,
                },
            )
            continue
        if halted:
            _mark_incomplete(case, halt_reason, halt_detail)
            cases_out.append(case)
            _publish(
                result,
                progress,
                {"event": "case", "case_id": case_id, "status": "incomplete"},
            )
            if halt_reason in {"deadline", "cancelled"}:
                _persist(state_file, _state_snapshot(contract, cases_out))
            continue
        if deadline is not None and time.monotonic() >= deadline:
            result["deadline"] = True
            halted = True
            halt_reason = "deadline"
            halt_detail = "deadline_seconds exceeded"
            _mark_incomplete(case, halt_reason, halt_detail)
            cases_out.append(case)
            _publish(
                result,
                progress,
                {"event": "case", "case_id": case_id, "status": "incomplete"},
            )
            _persist(state_file, _state_snapshot(contract, cases_out))
            continue
        _prepare_running(case)
        _persist(state_file, _state_snapshot(contract, cases_out, case))
        _publish(
            result,
            progress,
            {
                "event": "case_started",
                "case_id": case_id,
                "attempt_id": case["attempt_id"],
            },
        )
        stop, controller_error, case_deadline = _guard_case(
            case,
            live=live,
            runner=runner,
            evidence_map=evidence_map,
            phases=phases,
            launches=launches,
            policy=policy,
        )
        if controller_error is not None:
            runner_error = controller_error
            halted = True
            halt_reason = "halted_before_start"
            halt_detail = controller_error
        elif case_deadline:
            result["deadline"] = True
            halted = True
            halt_reason = "deadline"
            halt_detail = "case_timeout_seconds exceeded"
        cases_out.append(case)
        _publish(
            result,
            progress,
            {"event": "case", "case_id": case_id, "status": case["status"]},
        )
        _persist(state_file, _state_snapshot(contract, cases_out))
        if stop == "fail_fast":
            halted = True
            halt_reason = "halted_before_start"
            halt_detail = "fail_fast"
    return cases_out, halted, runner_error


def _dry_run_result(
    result: dict[str, Any],
    matrix: Mapping[str, Any],
    *,
    started_mono: float,
    started_at: str,
    progress: ProgressFn | None,
    write_path: Path | None,
    report_path: Path | None,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    result["cases"] = [dict(case) for case in matrix["cases"]]
    result["shared_checks"] = [
        {**dict(check), "counts_as_case": False} for check in matrix["shared_checks"]
    ]
    result["identity"]["runtime"] = "dry_run"
    result["launches"] = {"attempts": 0, "ready": 0}
    result["timing"] = reconcile(
        suite_start_mono=started_mono,
        suite_end_mono=time.monotonic(),
        suite_started_at=started_at,
        suite_finished_at=utc_now(),
        phases=[],
        shared_overhead_seconds=0.0,
    )
    projected = project(result)
    _emit(projected, progress, write_path, report_path, config)
    return projected


def execute_suite(
    config: Mapping[str, Any],
    selection: Mapping[str, Any],
    *,
    instance_token: str | None,
    dry_run: bool = False,
    evidence_by_case: Mapping[str, Mapping[str, Any]] | None = None,
    live: bool = False,
    state_dir: Path | None = None,
    resume: bool = False,
    progress: ProgressFn | None = None,
    write_path: Path | None = None,
    report_path: Path | None = None,
    infrastructure: list[Mapping[str, Any]] | None = None,
    runner: RunnerFn | None = None,
    cancelled: bool = False,
) -> dict[str, Any]:
    """Run or dry-run the resolved matrix and return one projected result."""

    started_at = utc_now()
    started_mono = time.monotonic()
    matrix = resolve_matrix(
        config, selection, instance_token=instance_token, dry_run=dry_run
    )
    result = suite_skeleton()
    result["dry_run"] = dry_run
    result["matrix"] = matrix
    result["identity"] = dict(matrix["identity"])
    result["run_id"] = matrix["run_id"]
    policy = matrix["policy"]
    meta = config.get("_meta") if isinstance(config.get("_meta"), Mapping) else {}
    state_root = (
        state_dir or Path(meta.get("repo_root") or ".") / ".analysis" / "harnessv2"
    )
    state_file = _state_path(state_root, str(matrix["run_id"]))
    contract = _contract_fingerprint(matrix)
    prior = load_resume(state_file, matrix) if resume else None
    prior_invalid = bool((prior or {}).get("resume_invalidated"))
    prior_cases = {
        str(row.get("case_id")): row
        for row in (prior or {}).get("cases") or []
        if isinstance(row, Mapping)
        and row.get("status") == "passed"
        and not prior_invalid
    }
    if dry_run:
        return _dry_run_result(
            result,
            matrix,
            started_mono=started_mono,
            started_at=started_at,
            progress=progress,
            write_path=write_path,
            report_path=report_path,
            config=config,
        )

    phases: list[dict[str, Any]] = []
    launches = {"attempts": 0, "ready": 0}
    selected = [dict(planned) for planned in matrix["cases"]]
    cases_out, halted, runner_error = _walk_cases(
        result=result,
        selected=selected,
        prior_cases=prior_cases,
        contract=contract,
        state_file=state_file,
        live=live,
        runner=runner,
        evidence_map=evidence_by_case or {},
        phases=phases,
        launches=launches,
        policy=policy,
        progress=progress,
        cancelled=cancelled,
    )
    _finish_suite(
        result,
        matrix=matrix,
        selected=selected,
        cases_out=cases_out,
        phases=phases,
        launches=launches,
        live=live,
        halted=halted,
        runner_error=runner_error,
        infrastructure=infrastructure,
        started_mono=started_mono,
        started_at=started_at,
        progress=progress,
    )
    _persist(state_file, _state_snapshot(contract, cases_out))
    projected = project(result)
    _emit(projected, progress, write_path, report_path, config)
    return projected


def _emit(
    result: Mapping[str, Any],
    progress: ProgressFn | None,
    write_path: Path | None,
    report_path: Path | None,
    config: Mapping[str, Any],
) -> None:
    if progress is not None:
        progress(
            {
                "event": "summary",
                "exit_class": result.get("exit_class"),
                "exit_code": result.get("exit_code"),
            }
        )
    if write_path is not None:
        write_artifact(write_path, result, config)
    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(render_text(result), encoding="utf-8")


def format_dry_run(result: Mapping[str, Any]) -> str:
    return dumps(result)
