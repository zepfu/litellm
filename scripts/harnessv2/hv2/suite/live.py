"""Stock interactive suite runner. One case uses the existing TUI plan path.

This module does not spawn ``codex exec``, ``-p``, or ``--print``. It builds
one ``RunPlan`` and calls ``run_plan``. Evidence is copied only from fields
the step recorded. A prompt echo is never a response value.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping

from hv2.errors import HarnessError
from hv2.kinds.runner import run_plan
from hv2.plan import build_plan

RunnerFn = Callable[[Mapping[str, Any]], Mapping[str, Any]]

_DETAIL_LIMIT = 500


def interactive_runner(config: Mapping[str, Any]) -> RunnerFn:
    """Return ``case ->`` launch result using the stock interactive drivers."""

    def _run(case: Mapping[str, Any]) -> Mapping[str, Any]:
        try:
            return _run_case(config, case)
        except HarnessError:
            raise
        except Exception as exc:
            return {"ready": False, "launch_error": str(exc)}

    return _run


def _run_case(config: Mapping[str, Any], case: Mapping[str, Any]) -> Mapping[str, Any]:
    plan = build_plan(
        config=config,
        kind=str(case.get("kind") or ""),
        instance_token=case.get("instance_token") if isinstance(case.get("instance_token"), str) else None,
        tui=str(case.get("tui") or "") or None,
        models=_model_args(case),
        orchestration_parent=_parent_arg(case),
        orchestration_children=_children_arg(case),
        dry_run=False,
        write_artifact=None,
    )
    artifact = run_plan(plan)
    step = _scenario_step(artifact, str(case.get("kind") or ""))
    if step is None:
        return {
            "ready": False,
            "launch_error": _bounded(_launch_detail(artifact, "interactive step did not run")),
            "evidence": None,
        }
    if _launch_failed(step) or not _session_ready(step):
        return {
            "ready": False,
            "launch_error": _bounded(_launch_detail(step, "session did not become ready")),
            "evidence": None,
        }
    return {"ready": True, "evidence": _evidence_from_step(case, step)}


def _model_args(case: Mapping[str, Any]) -> list[str] | None:
    if str(case.get("kind") or "") != "model":
        return None
    alias = case.get("alias") or case.get("model")
    if not alias:
        return None
    return [str(alias)]


def _parent_arg(case: Mapping[str, Any]) -> str | None:
    if str(case.get("kind") or "") != "orchestration":
        return None
    parent = case.get("parent") or case.get("alias")
    return str(parent) if parent else None


def _children_arg(case: Mapping[str, Any]) -> str | None:
    if str(case.get("kind") or "") != "orchestration":
        return None
    children = case.get("children")
    if not isinstance(children, list) or not children:
        return None
    return ",".join(str(item) for item in children if str(item).strip())


def _scenario_step(artifact: Mapping[str, Any], kind: str) -> dict[str, Any] | None:
    wanted = {
        "model": "tui_model",
        "orchestration": "tui_orchestration",
        "catalog": "tui_catalog",
    }.get(kind)
    if wanted is None:
        return None
    results = artifact.get("results")
    if not isinstance(results, list):
        return None
    for row in results:
        if isinstance(row, Mapping) and str(row.get("name") or "") == wanted:
            return dict(row)
    return None


def _turn(step: Mapping[str, Any]) -> Mapping[str, Any] | None:
    for key in ("models", "parents"):
        rows = step.get(key)
        if isinstance(rows, list):
            for row in rows:
                if isinstance(row, Mapping):
                    return row
    return None


def _session_value(step: Mapping[str, Any]) -> Any:
    turn = _turn(step)
    if turn is not None and "session" in turn:
        return turn.get("session")
    if "session" in step:
        return step.get("session")
    return None


def _launch_failed(step: Mapping[str, Any]) -> bool:
    turn = _turn(step)
    if turn is not None and "launch_ok" in turn:
        return turn.get("launch_ok") is False
    if "launch_ok" in step:
        return step.get("launch_ok") is False
    return False


def _session_ready(step: Mapping[str, Any]) -> bool:
    """True only when the step recorded a ready session, not merely a launch."""

    if str(step.get("name") or "") == "tui_catalog":
        return not bool(step.get("skipped")) and step.get("ok") is not False
    turn = _turn(step)
    if turn is None:
        return False
    if "launch_ok" in turn and turn.get("launch_ok") is not True:
        return False
    session = turn.get("session")
    if isinstance(session, str):
        return bool(session.strip())
    return False


def _launch_detail(source: Mapping[str, Any], fallback: str) -> str:
    failures = source.get("failures")
    if isinstance(failures, list):
        text = "; ".join(str(item) for item in failures if str(item).strip())
        if text:
            return text
    for key in ("reason", "error", "detail"):
        value = source.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    turn = _turn(source)
    if turn is not None:
        preview = turn.get("pane_preview")
        if isinstance(preview, str) and preview.strip():
            return preview.strip()
    return fallback


def _bounded(text: str) -> str:
    cleaned = " ".join(text.split())
    if len(cleaned) <= _DETAIL_LIMIT:
        return cleaned
    return cleaned[: _DETAIL_LIMIT - 3] + "..."


def _evidence_from_step(case: Mapping[str, Any], step: Mapping[str, Any]) -> dict[str, Any]:
    kind = str(case.get("kind") or "")
    turn = _turn(step)
    session = _session_value(step)
    evidence: dict[str, Any] = {
        "bound": _bound(case, session),
        "response_value": None,
        "tool": None,
        "children": None,
    }
    if kind == "model" and turn is not None:
        _apply_model_fields(evidence, turn)
    elif kind == "orchestration" and turn is not None:
        _apply_orchestration_fields(evidence, turn)
    elif kind == "catalog":
        _apply_catalog_fields(evidence, step)
    return evidence


def _bound(case: Mapping[str, Any], session: Any) -> dict[str, Any]:
    session_id = session.strip() if isinstance(session, str) else None
    if session_id == "":
        session_id = None
    return {
        "case_id": case.get("case_id"),
        "requested_alias": case.get("alias"),
        "session_id": session_id,
    }


def _apply_model_fields(evidence: dict[str, Any], turn: Mapping[str, Any]) -> None:
    tool_contract = str(turn.get("pass_mode") or "") == "tool_command" or turn.get("tool_pass") is True
    if tool_contract:
        evidence["contract"] = "tool_command"
        command = turn.get("tool_command") if isinstance(turn.get("tool_command"), str) else None
        stdout = turn.get("tool_stdout") if isinstance(turn.get("tool_stdout"), str) else None
        exit_status = turn.get("tool_exit_status")
        if command or stdout or exit_status is not None:
            evidence["tool"] = {
                "command": command,
                "stdout": stdout,
                "exit_status": exit_status,
            }
            evidence["expected_tool"] = {"command": command, "exit_status": 0}
    if "exact_pong" in turn and turn.get("exact_pong") is True and not tool_contract:
        evidence["response_value"] = "PONG"
    if "provider_404" in turn and turn.get("provider_404") is True:
        evidence["provider_error"] = {"status": 404}
    if "completed" in turn:
        evidence["completed"] = bool(turn.get("completed"))


def _apply_orchestration_fields(evidence: dict[str, Any], turn: Mapping[str, Any]) -> None:
    if "child_evidence" in turn and isinstance(turn.get("child_evidence"), Mapping):
        evidence["children"] = _child_rows(turn["child_evidence"])
    if "tool_pass" in turn:
        evidence["tool"] = {"recorded": True, "ok": bool(turn.get("tool_pass"))}
    if "completed" in turn:
        evidence["parent_completed"] = bool(turn.get("completed"))


def _child_rows(recorded: Mapping[str, Any]) -> list[dict[str, Any]] | None:
    """Copy child rows only when the step recorded per-child evidence."""

    raw = recorded.get("child_rows")
    if isinstance(raw, list):
        rows = [dict(item) for item in raw if isinstance(item, Mapping)]
        return rows or None
    children = recorded.get("children")
    if (
        isinstance(children, list)
        and children
        and all(isinstance(item, Mapping) for item in children)
    ):
        return [dict(item) for item in children]
    return None


def _apply_catalog_fields(evidence: dict[str, Any], step: Mapping[str, Any]) -> None:
    if "catalog_ids" in step:
        evidence["catalog_ids"] = step.get("catalog_ids")
    if "expected_catalog_ids" in step:
        evidence["expected_catalog_ids"] = step.get("expected_catalog_ids")
