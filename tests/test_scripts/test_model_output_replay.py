from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURES_DIR = REPO_ROOT / "tests/fixtures/model_output_replay"
CLI = REPO_ROOT / "scripts/dev-smoke/model_output_replay.py"
SUPPORT_PATH = REPO_ROOT / "tests/support/model_output_replay.py"


def _load_replay_module():
    spec = importlib.util.spec_from_file_location("model_output_replay", SUPPORT_PATH)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


replay = _load_replay_module()


def test_manifest_requires_all_fields(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps({"fixture_id": "only_id"}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(replay.ManifestValidationError, match="missing required fields"):
        replay.load_manifest(manifest)


def test_resolve_default_exec_fixture_detect_and_repair() -> None:
    manifest = FIXTURES_DIR / "manifest.jsonl"
    loaded = replay.resolve_fixture(
        manifest_path=manifest,
        fixture_id="grok_literal_exec_command_repairable",
        fixtures_dir=FIXTURES_DIR,
    )
    detect = replay.run_replay_lane(loaded, "detect")
    repair = replay.run_replay_lane(loaded, "repair")
    assert detect["malformed_detected"] is True
    assert repair["repair_succeeded"] is True
    assert repair["disposition"] == "repaired"
    replay_result = replay.run_replay(loaded, ("detect", "repair"))
    assert replay_result["expected_disposition"] == "repaired"
    assert replay_result["disposition"] == replay_result["expected_disposition"]
    assert (
        replay_result["source_kind"]
        == "synthetic_sanitized_model_authored_malformed_output"
    )
    assert (
        replay_result["transcript_surface"]
        == "adapter_visible_openai_responses_message_output_text"
    )


def test_apply_patch_fixture_fail_closed_without_advertised_tool() -> None:
    loaded = replay.resolve_fixture(
        manifest_path=FIXTURES_DIR / "manifest.jsonl",
        fixture_id="grok_literal_apply_patch_fail_closed",
        fixtures_dir=FIXTURES_DIR,
    )
    repair = replay.run_replay_lane(loaded, "repair")
    assert repair["malformed_detected"] is True
    assert repair["repair_succeeded"] is False
    assert repair["disposition"] == "fail_closed"


def test_production_helpers_invoked_via_monkeypatch(monkeypatch: pytest.MonkeyPatch) -> None:
    loaded = replay.resolve_fixture(
        manifest_path=FIXTURES_DIR / "manifest.jsonl",
        fixture_id="grok_literal_exec_command_repairable",
        fixtures_dir=FIXTURES_DIR,
    )

    calls: dict[str, int] = {
        "detect": 0,
        "repair": 0,
        "evidence": 0,
        "intake": 0,
        "scorer": 0,
    }

    import litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints as pte
    import litellm.proxy.aawm_runtime_error_logging as rel
    import litellm.integrations.aawm_agent_quality_rules as aqr

    real_detect = pte._is_codex_auto_agent_malformed_tool_call_text_output
    real_repair = (
        pte._try_repair_codex_auto_agent_grok_native_composer_literal_tool_call_response_body
    )
    real_evidence = rel.extract_malformed_tool_call_evidence
    real_intake = rel.build_malformed_tool_call_intake_record
    real_scorer = aqr.score_agent_quality_context

    def detect_wrapped(body):
        calls["detect"] += 1
        return real_detect(body)

    def repair_wrapped(body, *, request_body):
        calls["repair"] += 1
        return real_repair(body, request_body=request_body)

    def evidence_wrapped(body, *, max_items=None):
        calls["evidence"] += 1
        return real_evidence(body, max_items=max_items)

    def intake_wrapped(**kwargs):
        calls["intake"] += 1
        return real_intake(**kwargs)

    def scorer_wrapped(**kwargs):
        calls["scorer"] += 1
        return real_scorer(**kwargs)

    monkeypatch.setattr(
        pte,
        "_is_codex_auto_agent_malformed_tool_call_text_output",
        detect_wrapped,
    )
    monkeypatch.setattr(
        pte,
        "_try_repair_codex_auto_agent_grok_native_composer_literal_tool_call_response_body",
        repair_wrapped,
    )
    monkeypatch.setattr(rel, "extract_malformed_tool_call_evidence", evidence_wrapped)
    monkeypatch.setattr(rel, "build_malformed_tool_call_intake_record", intake_wrapped)
    monkeypatch.setattr(aqr, "score_agent_quality_context", scorer_wrapped)

    replay.run_replay_lane(loaded, "detect")
    replay.run_replay_lane(loaded, "repair")
    replay.run_replay_lane(loaded, "intake")
    replay.run_replay_lane(loaded, "scorer")

    assert calls["detect"] >= 2
    assert calls["repair"] >= 1
    assert calls["evidence"] >= 1
    assert calls["intake"] == 1
    assert calls["scorer"] == 1


def test_cli_json_does_not_emit_raw_transcript_markers() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            str(CLI),
            "--fixture",
            "grok_literal_exec_command_repairable",
            "--lane",
            "repair",
            "--json",
        ],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    combined = proc.stdout + proc.stderr
    assert "Tool label:" not in combined
    assert "Input payload:" not in combined
    payload = json.loads(proc.stdout)
    assert payload["fixture_id"] == "grok_literal_exec_command_repairable"
    assert payload["lanes"]["repair"]["repair_succeeded"] is True
