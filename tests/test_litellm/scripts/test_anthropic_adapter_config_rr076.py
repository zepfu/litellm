"""RR-076: anthropic_adapter_config.json prompts must not hardcode checkout abs paths."""

from __future__ import annotations

import json
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]
_CONFIG = _REPO / "scripts" / "local-ci" / "anthropic_adapter_config.json"
_ABS_ROOT = "/home/zepfu/projects/litellm"
_FIXTURE_REL = "scripts/local-ci/sequential_core_tools_fixture.txt"


def _load_config() -> dict:
    return json.loads(_CONFIG.read_text(encoding="utf-8"))


def test_config_has_no_operator_absolute_checkout_paths() -> None:
    text = _CONFIG.read_text(encoding="utf-8")
    assert _ABS_ROOT not in text
    assert "/home/zepfu/" not in text
    # still a valid JSON document
    cfg = json.loads(text)
    assert isinstance(cfg.get("cases"), dict)
    assert cfg["cases"]


def test_tool_case_prompts_use_repo_relative_paths() -> None:
    cfg = _load_config()
    fixture_cases = [
        name
        for name, case in cfg["cases"].items()
        if isinstance(case, dict)
        and isinstance(case.get("command"), list)
        and len(case["command"]) >= 3
        and _FIXTURE_REL in str(case["command"][2])
    ]
    assert fixture_cases, "expected sequential/parallel tool cases referencing the fixture"

    for name in fixture_cases:
        prompt = cfg["cases"][name]["command"][2]
        assert isinstance(prompt, str)
        assert _ABS_ROOT not in prompt
        assert "/home/zepfu/" not in prompt
        assert _FIXTURE_REL in prompt
        # Glob base is the checkout cwd (repo root), not a host-absolute path
        assert "from ." in prompt
        assert f"path exactly `{_FIXTURE_REL}`" in prompt or f"path exactly {_FIXTURE_REL}" in prompt


def test_read_pages_sanitizer_uses_relative_todo_path() -> None:
    cfg = _load_config()
    case = cfg["cases"]["claude_adapter_gpt55_read_pages_sanitizer"]
    prompt = case["command"][2]
    assert _ABS_ROOT not in prompt
    assert "TODO.md" in prompt
    assert "read TODO.md" in prompt.lower() or "read TODO.md" in prompt
    assert "/home/zepfu" not in prompt
    # fixture file still present for tool cases; TODO.md is a repo-root relative target
    assert (_REPO / "TODO.md").is_file()
    assert (_REPO / _FIXTURE_REL).is_file()


def test_harness_run_id_template_preserved_for_sequential_probe() -> None:
    cfg = _load_config()
    prompt = cfg["cases"]["claude_adapter_gpt55_child_sequential_core_tools"]["command"][2]
    assert "{harness_run_id}" in prompt
    assert "gpt55-sequential-tool-probe-{harness_run_id}.txt" in prompt
