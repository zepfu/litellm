"""Focused tests for RR-078: compare_artifacts Claude persona/subagent check."""

from __future__ import annotations

import importlib.util
import json
import pathlib
import sys
from typing import Any

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[3]
MODULE_PATH = ROOT / "scripts" / "local-ci" / "compare_artifacts.py"


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "compare_artifacts_rr078", MODULE_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def compare_artifacts():
    return _load_module()


def _family(
    *,
    passed: bool = True,
    actual_trace_names: list[str] | None = None,
    actual_user_ids: list[str] | None = None,
    required_trace_names: list[str] | None = None,
    expected_user_ids: list[str] | None = None,
    trace_count: int = 0,
) -> dict[str, Any]:
    return {
        "passed": passed,
        "langfuse": {
            "required_trace_names": required_trace_names or [],
            "expected_user_ids": expected_user_ids or [],
            "actual_trace_names": actual_trace_names or [],
            "actual_user_ids": actual_user_ids or [],
            "trace_count": trace_count,
        },
        "response_excerpt": "",
    }


def test_claude_unenriched_default_trace_name_is_not_persona_evidence(
    compare_artifacts,
):
    """Bare 'claude-code' means enrichment never fired — hard fail."""
    baseline = _family(
        actual_trace_names=["claude-code.orchestrator", "claude-code.explorer"],
        required_trace_names=["claude-code.orchestrator"],
    )
    candidate = _family(
        actual_trace_names=["claude-code"],
        required_trace_names=["claude-code.orchestrator"],
    )
    hard, soft = compare_artifacts._compare_family("claude", baseline, candidate)
    assert "claude: missing persona/subagent traces" in hard
    # Also missing required orchestrator name when only bare default is present.
    assert any("missing required trace name" in item for item in hard)
    assert soft == []


def test_claude_only_orchestrator_still_missing_persona(compare_artifacts):
    baseline = _family(
        actual_trace_names=["claude-code.orchestrator", "claude-code.worker"],
    )
    candidate = _family(actual_trace_names=["claude-code.orchestrator"])
    hard, _ = compare_artifacts._compare_family("claude", baseline, candidate)
    assert "claude: missing persona/subagent traces" in hard


def test_claude_only_default_and_orchestrator_missing_persona(compare_artifacts):
    candidate = _family(
        actual_trace_names=["claude-code", "claude-code.orchestrator"],
    )
    hard, _ = compare_artifacts._compare_family(
        "claude", _family(), candidate
    )
    assert "claude: missing persona/subagent traces" in hard


def test_claude_enriched_subagent_trace_passes_persona_check(compare_artifacts):
    candidate = _family(
        actual_trace_names=[
            "claude-code.orchestrator",
            "claude-code.explorer",
            "claude-code.worker",
        ],
        required_trace_names=["claude-code.orchestrator"],
    )
    hard, soft = compare_artifacts._compare_family(
        "claude",
        _family(required_trace_names=["claude-code.orchestrator"]),
        candidate,
    )
    assert "claude: missing persona/subagent traces" not in hard
    assert hard == []
    assert soft == []


def test_non_claude_families_skip_persona_check(compare_artifacts):
    candidate = _family(actual_trace_names=["codex-only"])
    hard, _ = compare_artifacts._compare_family("codex", _family(), candidate)
    assert hard == []
    hard, _ = compare_artifacts._compare_family("gemini", _family(), candidate)
    assert hard == []


def test_main_reports_hard_failure_for_unenriched_claude(
    compare_artifacts, tmp_path, capsys
):
    baseline = {
        "results": {
            "codex": _family(passed=True),
            "gemini": _family(passed=True),
            "claude": _family(
                passed=True,
                actual_trace_names=[
                    "claude-code.orchestrator",
                    "claude-code.explorer",
                ],
                required_trace_names=["claude-code.orchestrator"],
            ),
        }
    }
    candidate = {
        "results": {
            "codex": _family(passed=True),
            "gemini": _family(passed=True),
            "claude": _family(
                passed=True,
                # Enrichment broken: every trace is the unenriched default.
                actual_trace_names=["claude-code"],
                required_trace_names=["claude-code.orchestrator"],
            ),
        }
    }
    baseline_path = tmp_path / "baseline.json"
    candidate_path = tmp_path / "candidate.json"
    baseline_path.write_text(json.dumps(baseline), encoding="utf-8")
    candidate_path.write_text(json.dumps(candidate), encoding="utf-8")

    # Invoke via argv path used by the script.
    old_argv = sys.argv[:]
    try:
        sys.argv = [
            "compare_artifacts.py",
            str(baseline_path),
            str(candidate_path),
        ]
        code = compare_artifacts.main()
    finally:
        sys.argv = old_argv

    assert code == 1
    report = json.loads(capsys.readouterr().out)
    assert report["passed"] is False
    assert "claude: missing persona/subagent traces" in report["hard_failures"]
