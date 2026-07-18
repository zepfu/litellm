"""RR-079: local-ci config.json portable Claude prompts and live trace floor.

Findings for config.json:
1. Claude -p @paths must not embed machine-local absolute home paths. Source
   config uses @{config_dir}/... which run_acceptance._load_suite_config expands
   against the config file directory, and which build_harness_bundle preserves
   for standalone release artifacts.
2. minimum_trace_count remains declared for Claude (top-level + fanout modes).
   Concurrent run_acceptance._enforce_minimum_trace_count consumes the key, so
   deletion is not the correct config-side fix; values stay as the intended
   floors (3 minimal / 14 full / 3 top-level default).
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
_CONFIG_PATH = _REPO_ROOT / "scripts" / "local-ci" / "config.json"
_BUNDLE_SCRIPT = _REPO_ROOT / "scripts" / "local-ci" / "build_harness_bundle.py"
_RUN_ACCEPTANCE = _REPO_ROOT / "scripts" / "local-ci" / "run_acceptance.py"
_LOCAL_CI = _REPO_ROOT / "scripts" / "local-ci"

PORTABLE_MINIMAL = "@{config_dir}/claude_acceptance_prompt.txt"
PORTABLE_FULL = "@{config_dir}/claude_acceptance_prompt_full_fanout.txt"


@pytest.fixture(scope="module")
def config() -> dict:
    return json.loads(_CONFIG_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def bhb():
    name = "build_harness_bundle_rr079"
    spec = importlib.util.spec_from_file_location(name, _BUNDLE_SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def ra():
    name = "run_acceptance_rr079"
    spec = importlib.util.spec_from_file_location(name, _RUN_ACCEPTANCE)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_should_use_portable_claude_prompt_paths(config) -> None:
    assert config["claude"]["command"][2] == PORTABLE_MINIMAL
    assert config["claude"]["fanout_modes"]["minimal"]["command"][2] == PORTABLE_MINIMAL
    assert config["claude"]["fanout_modes"]["full"]["command"][2] == PORTABLE_FULL


def test_should_not_embed_machine_home_absolute_paths() -> None:
    text = _CONFIG_PATH.read_text(encoding="utf-8")
    assert "/home/zepfu" not in text
    assert "@/home/" not in text
    assert text.count("@{config_dir}/claude_acceptance_prompt") == 3


def test_referenced_prompt_files_exist_beside_config() -> None:
    assert (_LOCAL_CI / "claude_acceptance_prompt.txt").is_file()
    assert (_LOCAL_CI / "claude_acceptance_prompt_full_fanout.txt").is_file()


def test_should_declare_minimum_trace_count_floors(config) -> None:
    assert config["claude"]["minimum_trace_count"] == 3
    assert config["claude"]["fanout_modes"]["minimal"]["minimum_trace_count"] == 3
    assert config["claude"]["fanout_modes"]["full"]["minimum_trace_count"] == 14


def test_should_keep_required_trace_names_aligned_with_floors(config) -> None:
    minimal = config["claude"]["fanout_modes"]["minimal"]["required_trace_names"]
    full = config["claude"]["fanout_modes"]["full"]["required_trace_names"]
    assert len(minimal) == config["claude"]["fanout_modes"]["minimal"]["minimum_trace_count"]
    assert len(full) == config["claude"]["fanout_modes"]["full"]["minimum_trace_count"]
    assert "claude-code.orchestrator" in minimal
    assert "claude-code.orchestrator" in full


def test_should_preserve_portable_paths_through_bundle_render(bhb, config) -> None:
    portable = bhb.rewrite_config_paths_for_bundle(
        config, source_config_dir=bhb.LOCAL_CI_DIR
    )
    assert portable["claude"]["command"][2] == PORTABLE_MINIMAL
    assert portable["claude"]["fanout_modes"]["minimal"]["command"][2] == PORTABLE_MINIMAL
    assert portable["claude"]["fanout_modes"]["full"]["command"][2] == PORTABLE_FULL

    rendered = bhb.render_portable_config_bytes().decode("utf-8")
    assert "/home/zepfu" not in rendered
    rendered_cfg = json.loads(rendered)
    assert rendered_cfg["claude"]["command"][2] == PORTABLE_MINIMAL
    assert rendered_cfg["claude"]["fanout_modes"]["full"]["command"][2] == PORTABLE_FULL
    assert rendered_cfg["claude"]["fanout_modes"]["full"]["minimum_trace_count"] == 14


def test_should_expand_config_dir_placeholders_via_suite_loader(ra) -> None:
    loaded = ra._load_suite_config(_CONFIG_PATH)
    expanded = loaded["claude"]["command"][2]
    assert expanded.startswith("@")
    assert "{config_dir}" not in expanded
    assert "/home/zepfu" not in expanded or expanded.startswith("@")
    # Resolved path must point at the real prompt file.
    resolved = Path(expanded[1:])
    assert resolved.is_file()
    assert resolved.name == "claude_acceptance_prompt.txt"
    assert resolved.parent.resolve() == _LOCAL_CI.resolve()

    full = loaded["claude"]["fanout_modes"]["full"]["command"][2]
    assert Path(full[1:]).name == "claude_acceptance_prompt_full_fanout.txt"
    assert Path(full[1:]).is_file()


def test_should_enforce_minimum_trace_count_from_config_values(ra, config) -> None:
    # Offline proof: config floors are readable by the live enforcer.
    failures = ra._enforce_minimum_trace_count(
        family="claude",
        traces=[{}, {}],  # 2 < 3
        config=config["claude"],
    )
    assert failures
    assert "minimum_trace_count" in failures[0]

    ok = ra._enforce_minimum_trace_count(
        family="claude",
        traces=[{}] * 3,
        config=config["claude"]["fanout_modes"]["minimal"],
    )
    assert ok == []

    full_fail = ra._enforce_minimum_trace_count(
        family="claude",
        traces=[{}] * 13,
        config=config["claude"]["fanout_modes"]["full"],
    )
    assert full_fail
    assert "14" in full_fail[0]
