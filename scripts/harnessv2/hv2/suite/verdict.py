"""Mechanical case verdicts. Pane recap and acknowledgements are not success."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from hv2.suite.schema import assertion

_ACK_MARKERS = (
    "※ recap:",
    "recap:",
    "spawned",
    "spawn acknowledgement",
    "idle glyph",
)


def _refs(evidence: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw = evidence.get("evidence_refs")
    if isinstance(raw, list):
        return [dict(item) for item in raw if isinstance(item, Mapping)]
    return []


def _bound_ok(case: Mapping[str, Any], evidence: Mapping[str, Any]) -> bool:
    bound = evidence.get("bound")
    if not isinstance(bound, Mapping):
        return False
    if str(bound.get("case_id") or "") != str(case.get("case_id") or ""):
        return False
    if evidence.get("stale") or evidence.get("cross_session") or evidence.get("missing"):
        return False
    requested = str(case.get("alias") or "")
    if requested and str(bound.get("requested_alias") or "") not in {"", requested}:
        return False
    return bool(bound.get("session_id") or bound.get("workspace"))


def classify_infrastructure(source: str, payload: Mapping[str, Any]) -> dict[str, Any]:
    """Docker and error-JSONL findings are suite-level, not a case cause."""

    failures = [str(item) for item in (payload.get("failures") or [])]
    return {
        "class": "infrastructure",
        "source": source,
        "causal_case_id": None,
        "ok": not failures and bool(payload.get("ok", not failures)),
        "failures": failures,
        "detail": str(payload.get("detail") or "")[:500],
    }


def attribute_error(evidence: Mapping[str, Any]) -> dict[str, Any] | None:
    """Causal case attribution requires a declared machine correlation."""

    correlation = evidence.get("correlation")
    if not isinstance(correlation, Mapping):
        return None
    if correlation.get("matched") is True and correlation.get("method") == "machine":
        return {
            "case_id": correlation.get("case_id"),
            "method": "machine",
            "causal": True,
        }
    return None


def _assertion_correlation(
    case: Mapping[str, Any], evidence: Mapping[str, Any] | None
) -> dict[str, Any] | None:
    """Copy a recorded correlation, else identity from the case and bound session."""

    if isinstance(evidence, Mapping):
        recorded = evidence.get("correlation")
        if isinstance(recorded, Mapping):
            return dict(recorded)
    case_id = case.get("case_id")
    alias = case.get("alias")
    if not evidence:
        if case_id is None and alias is None:
            return None
        return {"case_id": case_id, "alias": alias}
    bound = evidence.get("bound") if isinstance(evidence, Mapping) else None
    row: dict[str, Any] = {"case_id": case_id, "alias": alias}
    if isinstance(bound, Mapping) and bound.get("session_id"):
        row["session_id"] = bound.get("session_id")
    return row


def evaluate_case(
    case: Mapping[str, Any],
    evidence: Mapping[str, Any] | None,
) -> list[dict[str, Any]]:
    """Return assertions for one executed case. Never promotes missing evidence."""

    case_id = str(case.get("case_id") or "")
    correlation = _assertion_correlation(case, evidence)
    if not evidence:
        return [
            assertion(
                code="evidence.missing",
                expected="bound terminal evidence",
                observed=None,
                status="inconclusive",
                case_id=case_id,
                required=True,
                correlation=correlation,
            )
        ]
    refs = _refs(evidence)
    if evidence.get("missing") or evidence.get("stale") or evidence.get("cross_session"):
        reason = (
            "missing"
            if evidence.get("missing")
            else "stale"
            if evidence.get("stale")
            else "cross_session"
        )
        return [
            assertion(
                code=f"evidence.{reason}",
                expected="fresh evidence for the selected session",
                observed=reason,
                status="inconclusive",
                evidence_refs=refs,
                case_id=case_id,
                correlation=correlation,
            )
        ]
    if evidence.get("acknowledgement_only"):
        return [
            assertion(
                code="evidence.acknowledgement_only",
                expected="response, tool, or child completion",
                observed="acknowledgement",
                status="fail",
                evidence_refs=refs,
                case_id=case_id,
                detail="recap, echo, selector, idle glyph, or spawn acknowledgement",
                correlation=correlation,
            )
        ]
    if not _bound_ok(case, evidence):
        return [
            assertion(
                code="evidence.unbound",
                expected={"case_id": case_id, "alias": case.get("alias")},
                observed=evidence.get("bound"),
                status="inconclusive",
                evidence_refs=refs,
                case_id=case_id,
                correlation=correlation,
            )
        ]

    expected_error = evidence.get("expect_error")
    provider_error = evidence.get("provider_error")
    assertions: list[dict[str, Any]] = []
    if isinstance(expected_error, Mapping):
        want = expected_error.get("status")
        got = provider_error.get("status") if isinstance(provider_error, Mapping) else None
        attributed = bool(
            isinstance(provider_error, Mapping) and provider_error.get("attributed") is True
        )
        ok = got == want and attributed
        assertions.append(
            assertion(
                code="expected_provider_error",
                expected={"status": want, "attributed": True},
                observed={"status": got, "attributed": attributed},
                status="pass" if ok else "fail",
                evidence_refs=refs,
                case_id=case_id,
                correlation=correlation,
            )
        )
        return assertions

    if isinstance(provider_error, Mapping) and provider_error.get("status"):
        assertions.append(
            assertion(
                code="unexpected_provider_error",
                expected="success evidence",
                observed=dict(provider_error),
                status="fail",
                evidence_refs=refs,
                case_id=case_id,
                correlation=correlation,
            )
        )
        return assertions

    kind = str(case.get("kind") or "")
    if kind == "model":
        assertions.extend(_model_assertions(case, evidence, refs, correlation))
    elif kind == "orchestration":
        assertions.extend(_child_assertions(case, evidence, refs, correlation))
    elif kind == "catalog":
        observed = evidence.get("catalog_ids")
        expected = evidence.get("expected_catalog_ids")
        ok = isinstance(observed, list) and isinstance(expected, list) and set(expected) <= set(observed)
        assertions.append(
            assertion(
                code="catalog.ids",
                expected=expected,
                observed=observed,
                status="pass" if ok else "inconclusive" if observed is None else "fail",
                evidence_refs=refs,
                case_id=case_id,
                correlation=correlation,
            )
        )
    else:
        assertions.append(
            assertion(
                code="scenario.unsupported",
                expected="model, orchestration, or catalog",
                observed=kind,
                status="inconclusive",
                case_id=case_id,
                correlation=correlation,
            )
        )
    return assertions


def _model_assertions(
    case: Mapping[str, Any],
    evidence: Mapping[str, Any],
    refs: Sequence[Mapping[str, Any]],
    correlation: Mapping[str, Any] | None,
) -> list[dict[str, Any]]:
    case_id = str(case.get("case_id") or "")
    contract = str(evidence.get("contract") or "response")
    if contract == "tool_command":
        tool = evidence.get("tool") if isinstance(evidence.get("tool"), Mapping) else {}
        has_fields = any(
            tool.get(name) is not None for name in ("command", "stdout", "exit_status")
        )
        if has_fields:
            expected = evidence.get("expected_tool") if isinstance(evidence.get("expected_tool"), Mapping) else {
                "command": tool.get("command"),
                "exit_status": 0,
            }
            command_ok = bool(tool.get("command")) and tool.get("command") == expected.get("command")
            exit_ok = tool.get("exit_status") == expected.get("exit_status", 0)
            stdout_ok = bool(str(tool.get("stdout") or "").strip())
            status = "pass" if command_ok and exit_ok and stdout_ok else "fail"
            observed: Any = {
                "command": tool.get("command"),
                "exit_status": tool.get("exit_status"),
                "stdout": tool.get("stdout"),
            }
            expected_row: Any = dict(expected)
        elif evidence.get("tool_pass_recorded") is True and evidence.get("completed") is True:
            status = "pass"
            expected_row = {"tool_pass": True, "completed": True}
            observed = {"tool_pass": True, "completed": True}
        elif evidence.get("tool_pass_recorded") is False or evidence.get("completed") is False:
            status = "fail"
            expected_row = {"tool_pass": True, "completed": True}
            observed = {
                "tool_pass": evidence.get("tool_pass_recorded"),
                "completed": evidence.get("completed"),
            }
        else:
            status = "inconclusive"
            expected_row = {"tool_pass": True, "completed": True}
            observed = {
                "command": None,
                "exit_status": None,
                "stdout": None,
            }
        return [
            assertion(
                code="tool.command",
                expected=expected_row,
                observed=observed,
                status=status,
                evidence_refs=list(refs),
                case_id=case_id,
                correlation=correlation,
            )
        ]
    expected = evidence.get("expected_response", "PONG")
    observed = evidence.get("response_value")
    if observed is None:
        status = "inconclusive"
    elif observed == expected:
        status = "pass"
    else:
        status = "fail"
    return [
        assertion(
            code="response.value",
            expected=expected,
            observed=observed,
            status=status,
            evidence_refs=list(refs),
            case_id=case_id,
            correlation=correlation,
        )
    ]


def _parent_tool_ok(tool: Mapping[str, Any] | None) -> bool:
    if not isinstance(tool, Mapping):
        return False
    command = tool.get("command")
    stdout = str(tool.get("stdout") or "").strip()
    return bool(command) and tool.get("exit_status") == 0 and bool(stdout)


def _spawn_contract_ok(evidence: Mapping[str, Any]) -> bool:
    if evidence.get("parent_completed") is not True or not _spawn_ok(evidence):
        return False
    detail = evidence.get("spawn_detail")
    if not isinstance(detail, Mapping):
        return False
    contract = evidence.get("spawn_contract")
    if contract == "grok_spawn_tool":
        return (
            detail.get("spawn_chrome") is True
            and detail.get("pwd_row") is True
            and detail.get("uname_row") is True
        )
    if contract == "muse_spawn_tool":
        return (
            detail.get("spawn_chrome") is True
            and detail.get("child_completed") is True
        )
    return False


def _spawn_contract_assertion(
    case_id: str,
    evidence: Mapping[str, Any],
    refs: Sequence[Mapping[str, Any]],
    correlation: Mapping[str, Any] | None,
) -> dict[str, Any]:
    detail = evidence.get("spawn_detail") if isinstance(evidence.get("spawn_detail"), Mapping) else {}
    recorded = evidence.get("spawn_contract") in {"grok_spawn_tool", "muse_spawn_tool"}
    if _spawn_contract_ok(evidence):
        status = "pass"
    elif recorded and (
        evidence.get("spawn_ok") is False or evidence.get("parent_completed") is False
    ):
        status = "fail"
    else:
        status = "inconclusive"
    return assertion(
        code="orchestration.spawn",
        expected={
            "contract": evidence.get("spawn_contract"),
            "parent_completed": True,
            "spawn_ok": True,
        },
        observed={
            "contract": evidence.get("spawn_contract"),
            "parent_completed": evidence.get("parent_completed"),
            "spawn_ok": evidence.get("spawn_ok"),
            "detail": dict(detail),
        },
        status=status,
        evidence_refs=list(refs),
        case_id=case_id,
        correlation=correlation,
    )


def _spawn_ok(evidence: Mapping[str, Any]) -> bool:
    if evidence.get("spawn_ok") is not True:
        return False
    failures = evidence.get("spawn_failures")
    return not isinstance(failures, list) or not failures


def _child_assertions(
    case: Mapping[str, Any],
    evidence: Mapping[str, Any],
    refs: Sequence[Mapping[str, Any]],
    correlation: Mapping[str, Any] | None,
) -> list[dict[str, Any]]:
    case_id = str(case.get("case_id") or "")
    wanted = [str(item) for item in (case.get("children") or [])]
    rows = evidence.get("children") if isinstance(evidence.get("children"), list) else None
    if rows is None and evidence.get("spawn_contract") in {
        "grok_spawn_tool",
        "muse_spawn_tool",
    }:
        return [
            _spawn_contract_assertion(case_id, evidence, refs, correlation)
        ]
    if rows is None:
        return [
            assertion(
                code="child.completion",
                expected=wanted,
                observed=None,
                status="inconclusive",
                evidence_refs=list(refs),
                case_id=case_id,
                correlation=correlation,
            )
        ]
    by_alias = {
        str(row.get("alias")): row
        for row in rows
        if isinstance(row, Mapping) and row.get("alias")
    }
    assertions: list[dict[str, Any]] = []
    if not wanted:
        tool = evidence.get("tool") if isinstance(evidence.get("tool"), Mapping) else None
        parent_ok = evidence.get("parent_completed") is True and (
            _spawn_ok(evidence) or _parent_tool_ok(tool)
        )
        assertions.append(
            assertion(
                code="orchestration.parent",
                expected="parent completed with declared evidence",
                observed={"parent_completed": evidence.get("parent_completed"), "tool": tool},
                status="pass" if parent_ok else "inconclusive",
                evidence_refs=list(refs),
                case_id=case_id,
                correlation=correlation,
            )
        )
        return assertions
    for alias in wanted:
        row = by_alias.get(alias)
        if not isinstance(row, Mapping):
            assertions.append(
                assertion(
                    code="child.completion",
                    expected={"alias": alias, "completed": True},
                    observed=None,
                    status="fail",
                    evidence_refs=list(refs),
                    case_id=case_id,
                    correlation=correlation,
                )
            )
            continue
        producer = str(row.get("producer") or "")
        completed = bool(row.get("completed")) and bool(row.get("created"))
        identity_ok = bool(producer) and producer != "acknowledgement"
        status = "pass" if completed and identity_ok else "fail"
        assertions.append(
            assertion(
                code="child.completion",
                expected={"alias": alias, "created": True, "completed": True, "producer": "observed"},
                observed={
                    "alias": alias,
                    "created": row.get("created"),
                    "completed": row.get("completed"),
                    "producer": producer or None,
                },
                status=status,
                evidence_refs=list(refs),
                case_id=case_id,
                correlation=correlation,
            )
        )
    return assertions


def case_status_from_assertions(
    assertions: Sequence[Mapping[str, Any]], *, executed: bool
) -> str:
    if not executed:
        return "incomplete"
    required = [row for row in assertions if row.get("required", True)]
    if any(row.get("status") == "fail" for row in required):
        return "failed"
    if any(row.get("status") != "pass" for row in required):
        return "failed"
    if required and all(row.get("status") == "pass" for row in required):
        return "passed"
    return "failed"


def text_is_acknowledgement_only(text: str) -> bool:
    lowered = text.lower()
    if any(marker in lowered for marker in _ACK_MARKERS):
        has_pong = "\npong\n" in f"\n{lowered}\n"
        return not has_pong
    return False
