"""Harness v2 suite matrix, verdicts, timing, and exits.

Calls the shipped entry points. Does not reimplement them.
"""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

_REPO = Path(__file__).resolve().parents[3]
_HV2 = _REPO / "scripts" / "harnessv2"

if str(_HV2) not in sys.path:
    sys.path.insert(0, str(_HV2))


def _load() -> Any:
    importlib.invalidate_caches()
    import run as hv2_run
    from hv2.checks.session_history import session_history_result
    from hv2.load_config import load_config
    from hv2.suite.report import exit_from_result, project, render_text
    from hv2.suite.schema import case_counts
    from hv2.suite.timing import consume_wait_checkpoints, reconcile
    from hv2.suite.verdict import classify_infrastructure, evaluate_case
    from hv2.suite.execute import execute_suite
    from hv2.suite.live import _evidence_from_step
    from hv2.suite.matrix import selection_from_args

    return SimpleNamespace(
        main=hv2_run.main,
        session_history_result=session_history_result,
        load_config=load_config,
        exit_from_result=exit_from_result,
        project=project,
        render_text=render_text,
        case_counts=case_counts,
        consume_wait_checkpoints=consume_wait_checkpoints,
        reconcile=reconcile,
        classify_infrastructure=classify_infrastructure,
        evaluate_case=evaluate_case,
        evidence_from_step=_evidence_from_step,
        execute_suite=execute_suite,
        selection_from_args=selection_from_args,
    )


@pytest.fixture(scope="module")
def hv2() -> Any:
    return _load()


def _bound(case_id: str, alias: str, **extra: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "bound": {
            "case_id": case_id,
            "requested_alias": alias,
            "session_id": "sess-1",
        }
    }
    payload.update(extra)
    return payload


def _pong_evidence(case_id: str, alias: str) -> dict[str, Any]:
    return _bound(
        case_id,
        alias,
        contract="response",
        expected_response="PONG",
        response_value="PONG",
    )


def test_suite_dry_run_matrix_does_not_launch(hv2: Any, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    def _forbid_docker(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("run_docker must not be called")

    def _forbid_urlopen(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("urlopen must not be called")

    import hv2.docker_guard as docker_guard
    import urllib.request

    monkeypatch.setattr(docker_guard, "run_docker", _forbid_docker)
    monkeypatch.setattr(urllib.request, "urlopen", _forbid_urlopen)

    code = hv2.main(
        [
            "--suite",
            "--suite-tui",
            "ohmypi",
            "--suite-tui",
            "codex",
            "--suite-kind",
            "orchestration",
            "--suite-parent",
            "sota-openai,basic",
            "--dry-run",
        ]
    )
    captured = capsys.readouterr()
    assert code == 0
    body = json.loads(captured.out)
    matrix = body["matrix"]
    cases = matrix["cases"]
    assert len(cases) == 4
    pairs = {(case["tui"], case["parent"]) for case in cases}
    assert pairs == {
        ("ohmypi", "sota-openai"),
        ("ohmypi", "basic"),
        ("codex", "sota-openai"),
        ("codex", "basic"),
    }
    assert body["counts"]["planned"] == 4
    assert matrix["planned_cases"] == 4
    shared = matrix["shared_checks"]
    assert shared
    assert all(check.get("counts_as_case") is False for check in shared)
    assert body["counts"]["planned"] == len(cases)
    assert body["counts"]["planned"] != len(cases) + len(shared)
    identity = body["identity"]
    assert identity["config_hash"]
    assert identity["source_commit"]
    assert body["launches"]["attempts"] == 0
    assert identity["runtime"] == "dry_run"


def test_case_counts_retries_and_child_assertions_are_not_cases(hv2: Any) -> None:
    cases = [
        {
            "case_id": "c1",
            "status": "passed",
            "attempts": [
                {"attempt_id": "c1:1", "status": "failed"},
                {"attempt_id": "c1:2", "status": "passed"},
            ],
            "assertions": [
                {"code": "child.completion", "status": "pass"},
                {"code": "child.completion", "status": "pass"},
                {"code": "child.completion", "status": "pass"},
            ],
        }
    ]
    counts = hv2.case_counts(cases)
    assert counts["planned"] == 1
    assert counts["passed"] == 1
    assert counts["finished"] == counts["passed"] + counts["failed"]
    assert counts["incomplete"] == 0
    assert counts["finished"] == 1


def test_evaluate_case_evidence_and_expected_error(hv2: Any) -> None:
    case = {"case_id": "case-1", "alias": "basic", "kind": "model"}
    bad = {
        "missing": _bound("case-1", "basic", missing=True),
        "stale": _bound("case-1", "basic", stale=True),
        "cross_session": _bound("case-1", "basic", cross_session=True),
        "acknowledgement_only": _bound("case-1", "basic", acknowledgement_only=True),
    }
    for label, evidence in bad.items():
        rows = hv2.evaluate_case(case, evidence)
        assert rows
        assert all(row["status"] != "pass" for row in rows), label

    expected = _bound(
        "case-1",
        "basic",
        expect_error={"status": 404},
        provider_error={"status": 404, "attributed": True},
    )
    passed = hv2.evaluate_case(case, expected)
    assert passed[0]["status"] == "pass"
    assert passed[0]["code"] == "expected_provider_error"

    unexpected = _bound(
        "case-1",
        "basic",
        provider_error={"status": 404, "attributed": True},
    )
    failed = hv2.evaluate_case(case, unexpected)
    assert failed[0]["status"] == "fail"
    assert failed[0]["code"] == "unexpected_provider_error"

    finding = hv2.classify_infrastructure(
        "error_jsonl",
        {"ok": False, "failures": ["boom"], "detail": "shared"},
    )
    assert finding["causal_case_id"] is None
    assert finding["class"] == "infrastructure"


def test_reconcile_wall_nested_and_wait_checkpoints(hv2: Any) -> None:
    timing = hv2.reconcile(
        suite_start_mono=10.0,
        suite_end_mono=16.5,
        suite_started_at="2026-10-09T00:00:00+00:00",
        suite_finished_at="2026-10-09T00:00:06+00:00",
        phases=[
            {"case_id": "c1", "phase": "preparation", "start": 10.0, "end": 11.0},
            {"case_id": "c1", "phase": "validation", "start": 12.0, "end": 14.0},
        ],
        nested_spans=[
            {"kind": "provider", "duration_seconds": 9.0},
            {"kind": "tool", "duration_seconds": 4.0},
        ],
        shared_overhead_seconds=0.5,
    )
    wall = timing["suite"]["wall_seconds"]
    nested = timing["nested_span_sum_seconds"]
    assert wall == pytest.approx(6.5)
    assert nested == pytest.approx(13.0)
    assert nested != wall
    assert timing["wall_excludes_nested_spans"] is True
    empty = hv2.reconcile(
        suite_start_mono=0.0,
        suite_end_mono=1.0,
        suite_started_at="2026-10-09T00:00:00+00:00",
        suite_finished_at="2026-10-09T00:00:01+00:00",
        phases=[],
        nested_spans=[],
        shared_overhead_seconds=0.0,
    )
    assert empty["nested_span_sum_seconds"] is None
    assert "nested_span_sum_seconds" in empty["unavailable"]
    assert "phase.launch_readiness" in timing["unavailable"]
    assert "phase.prompt_delivery" in timing["unavailable"]
    assert "phase.response_execution" in timing["unavailable"]
    assert "phase.evidence_collection" in timing["unavailable"]
    assert "phase.cleanup_retention" in timing["unavailable"]
    exclusive = timing["exclusive_phase_seconds"]
    assert exclusive["launch_readiness"] is None
    assert exclusive["preparation"] == pytest.approx(1.0)

    rows = hv2.consume_wait_checkpoints(
        [
            {
                "litellm_call_id": "call-a",
                "sequence": 1,
                "durations_ms": {"queue": 10},
                "status": "running",
            },
            {
                "litellm_call_id": "call-a",
                "sequence": 3,
                "durations_ms": {"queue": 40},
                "status": "done",
            },
            {
                "litellm_call_id": "call-a",
                "sequence": 2,
                "durations_ms": {"queue": 20},
                "status": "running",
            },
            {
                "litellm_call_id": "call-b",
                "sequence": 1,
                "durations_ms": {"queue": 5},
                "status": "done",
            },
        ]
    )
    by_call = {row["litellm_call_id"]: row for row in rows}
    assert by_call["call-a"]["sequence"] == 3
    assert by_call["call-a"]["durations_ms"]["queue"] == 40
    assert by_call["call-b"]["sequence"] == 1
    assert len(rows) == 2


def test_execute_suite_fixture_projects_same_result(hv2: Any, tmp_path: Path) -> None:
    config = hv2.load_config()
    selection = hv2.selection_from_args(
        tuis=["ohmypi"],
        kinds=["model"],
        models=["basic", "work"],
        parents=[],
        children=None,
        include_shared=False,
    )
    state_dir = tmp_path / "suite-state"
    write_path = tmp_path / "artifact.json"
    report_path = tmp_path / "report.txt"
    assert ".analysis" not in state_dir.parts
    preview = hv2.execute_suite(
        config,
        selection,
        instance_token=None,
        dry_run=True,
    )
    cases = preview["matrix"]["cases"]
    assert len(cases) == 2
    basic = next(case for case in cases if case["alias"] == "basic")
    work = next(case for case in cases if case["alias"] == "work")
    evidence = {basic["case_id"]: _pong_evidence(basic["case_id"], "basic")}
    result = hv2.execute_suite(
        config,
        selection,
        instance_token=None,
        dry_run=False,
        live=False,
        evidence_by_case=evidence,
        state_dir=state_dir,
        write_path=write_path,
        report_path=report_path,
    )
    by_id = {case["case_id"]: case for case in result["cases"]}
    assert basic["case_id"] in by_id
    assert work["case_id"] in by_id
    assert by_id[basic["case_id"]]["status"] == "passed"
    assert by_id[work["case_id"]]["status"] != "passed"
    assert result["counts"]["planned"] == 2
    assert result["exit_code"] != 0

    artifact = json.loads(write_path.read_text(encoding="utf-8"))
    report = report_path.read_text(encoding="utf-8")
    assert artifact["exit_code"] == result["exit_code"]
    assert artifact["counts"]["planned"] == result["counts"]["planned"]
    rendered = hv2.render_text(result)
    assert report == rendered
    assert f"exit {result['exit_code']}" in report
    assert "planned=2" in report
    assert state_dir.is_dir()
    assert list(state_dir.glob("*.state.json"))


def test_exit_classes_from_projected_results(hv2: Any, tmp_path: Path) -> None:
    config = hv2.load_config()
    selection = hv2.selection_from_args(
        tuis=["ohmypi"],
        kinds=["model"],
        models=["basic"],
        parents=[],
        children=None,
        include_shared=False,
    )

    def _boom(_case: dict[str, Any]) -> dict[str, Any]:
        raise RuntimeError("runner exploded")

    runner_result = hv2.execute_suite(
        config,
        selection,
        instance_token=None,
        dry_run=False,
        live=True,
        runner=_boom,
        state_dir=tmp_path / "runner-state",
    )
    # live runner failure is recorded on the case; a controller exception is
    # runner_error. Force the shipped classifier through project().
    assert runner_result["exit_code"] != 0

    halted = {
        "runner_error": None,
        "halted": True,
        "dry_run": False,
        "cases": [
            {"case_id": "a", "status": "passed", "tui": "ohmypi"},
            {"case_id": "b", "status": "incomplete", "tui": "ohmypi"},
        ],
        "matrix": {
            "cases": [
                {"case_id": "a"},
                {"case_id": "b"},
            ]
        },
        "assertions": [],
        "infrastructure_findings": [],
    }
    halted_projected = hv2.project(halted)
    assert halted_projected["exit_code"] == 3
    assert hv2.exit_from_result(halted_projected) == ("incomplete", 3)

    passed = {
        "runner_error": None,
        "halted": False,
        "dry_run": False,
        "cases": [
            {
                "case_id": "a",
                "status": "passed",
                "tui": "ohmypi",
                "alias": "basic",
                "attempt_id": "a:1",
                "assertions": [{"code": "response.value", "status": "pass", "required": True}],
            }
        ],
        "matrix": {"cases": [{"case_id": "a"}]},
        "assertions": [{"code": "response.value", "status": "pass", "required": True, "case_id": "a"}],
        "infrastructure_findings": [],
    }
    passed_projected = hv2.project(passed)
    assert passed_projected["exit_code"] == 0
    assert hv2.exit_from_result(passed_projected) == ("success", 0)

    runner_error = {
        "runner_error": "controller blew up",
        "halted": True,
        "dry_run": False,
        "cases": [{"case_id": "a", "status": "incomplete", "tui": "ohmypi"}],
        "matrix": {"cases": [{"case_id": "a"}]},
        "assertions": [],
        "infrastructure_findings": [],
    }
    runner_projected = hv2.project(runner_error)
    assert runner_projected["exit_code"] == 2
    assert hv2.exit_from_result(runner_projected) == ("runner", 2)

    missing = {
        "runner_error": None,
        "halted": False,
        "dry_run": False,
        "cases": [{"case_id": "a", "status": "passed", "tui": "ohmypi"}],
        "matrix": {"cases": [{"case_id": "a"}, {"case_id": "omitted"}]},
        "assertions": [],
        "infrastructure_findings": [],
    }
    missing_projected = hv2.project(missing)
    assert missing_projected["exit_code"] != 0
    assert missing_projected["reconciliation"]["missing"] == ["omitted"]


def _stock_case(kind: str, case_id: str, alias: str) -> dict[str, Any]:
    return {
        "case_id": case_id,
        "kind": kind,
        "alias": alias,
        "children": [],
    }


def _assert_one(hv2: Any, case: dict[str, Any], step: dict[str, Any]) -> dict[str, Any]:
    evidence = hv2.evidence_from_step(case, step)
    rows = hv2.evaluate_case(case, evidence)
    assert len(rows) == 1
    return rows[0]


def test_evidence_from_step_maps_stock_model_and_spawn_records(hv2: Any) -> None:
    """Stock tui_model and tui_orchestration rows, not hand-built verdicts."""

    codex = _stock_case("model", "codex-case", "basic")
    passed = _assert_one(
        hv2,
        codex,
        {
            "models": [
                {
                    "session": "hv2-codex-basic",
                    "pass_mode": "tool_command",
                    "tool_pass": True,
                    "completed": True,
                    "exact_pong": False,
                }
            ]
        },
    )
    assert passed["code"] == "tool.command"
    assert passed["status"] == "pass"

    missing = _assert_one(
        hv2,
        codex,
        {
            "models": [
                {
                    "session": "hv2-codex-basic",
                    "pass_mode": "tool_command",
                    "completed": True,
                    "exact_pong": False,
                }
            ]
        },
    )
    assert missing["status"] == "inconclusive"
    assert missing["observed"]["command"] is None

    failed = _assert_one(
        hv2,
        codex,
        {
            "models": [
                {
                    "session": "hv2-codex-basic",
                    "pass_mode": "tool_command",
                    "tool_pass": False,
                    "completed": False,
                }
            ]
        },
    )
    assert failed["status"] == "fail"

    pong = _assert_one(
        hv2,
        codex,
        {
            "models": [
                {
                    "session": "hv2-codex-basic",
                    "pass_mode": "exact_pong",
                    "exact_pong": True,
                    "tool_pass": True,
                    "completed": True,
                }
            ]
        },
    )
    assert pong["code"] == "response.value"
    assert pong["status"] == "pass"
    assert pong["observed"] == "PONG"

    grok = _stock_case("orchestration", "grok-case", "grok-4.7")
    grok_pass = _assert_one(
        hv2,
        grok,
        {
            "parents": [
                {
                    "session": "hv2-grok-parent",
                    "completed": True,
                    "child_evidence": {
                        "kind": "grok_spawn_tool",
                        "ok": True,
                        "failures": [],
                        "spawn_chrome": True,
                        "pwd_row": True,
                        "uname_row": True,
                    },
                }
            ]
        },
    )
    assert grok_pass["code"] == "orchestration.spawn"
    assert grok_pass["status"] == "pass"

    grok_ack = _assert_one(
        hv2,
        grok,
        {
            "parents": [
                {
                    "session": "hv2-grok-parent",
                    "completed": True,
                    "tool_pass": True,
                    "child_evidence": {
                        "kind": "grok_spawn_tool",
                        "ok": False,
                        "failures": ["prompt-echo-only spawn"],
                        "spawn_chrome": False,
                        "pwd_row": False,
                        "uname_row": False,
                    },
                }
            ]
        },
    )
    assert grok_ack["status"] == "fail"

    muse = _stock_case("orchestration", "muse-case", "muse-spark-1.3-contributor")
    muse_pass = _assert_one(
        hv2,
        muse,
        {
            "parents": [
                {
                    "session": "hv2-muse-parent",
                    "completed": True,
                    "child_evidence": {
                        "kind": "muse_spawn_tool",
                        "ok": True,
                        "failures": [],
                        "spawn_chrome": True,
                        "child_completed": True,
                    },
                }
            ]
        },
    )
    assert muse_pass["code"] == "orchestration.spawn"
    assert muse_pass["status"] == "pass"

    muse_open = _assert_one(
        hv2,
        muse,
        {
            "parents": [
                {
                    "session": "hv2-muse-parent",
                    "completed": True,
                    "child_evidence": {
                        "kind": "muse_spawn_tool",
                        "ok": True,
                        "failures": [],
                        "spawn_chrome": True,
                        "child_completed": False,
                    },
                }
            ]
        },
    )
    assert muse_open["status"] == "inconclusive"


def test_session_history_shipped_config_stays_skipped(hv2: Any) -> None:
    result = hv2.session_history_result(hv2.load_config())
    assert result["enabled"] is False
    assert result["skipped"] is True
