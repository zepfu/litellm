"""RR-073: bump script GROUPS is the source of truth for autobump watch paths."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[3]
_SCRIPT = _REPO / "scripts" / "bump_aawm_artifact_versions.py"
_WORKFLOW = _REPO / ".github" / "workflows" / "aawm-artifact-autobump.yml"
_BUNDLED_REL = "litellm/bundled_model_prices_and_context_window_fallback.json"
_BACKUP_REL = "litellm/model_prices_and_context_window_backup.json"
_CANONICAL_REL = "model_prices_and_context_window.json"


def _load_bump_module():
    name = "bump_aawm_artifact_versions_rr073"
    spec = importlib.util.spec_from_file_location(name, _SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def bump_mod():
    return _load_bump_module()


def _config_group(bump_mod):
    groups = {g.name: g for g in bump_mod.GROUPS}
    assert "config" in groups
    return groups["config"]


def test_config_group_watches_bundled_fallback_not_nonexistent_backup(bump_mod) -> None:
    config = _config_group(bump_mod)
    assert _BUNDLED_REL in config.paths
    assert _CANONICAL_REL in config.paths
    assert "scripts/build_model_config_bundle.py" in config.paths
    assert _BACKUP_REL not in config.paths
    assert (_REPO / _BUNDLED_REL).is_file()
    assert not (_REPO / _BACKUP_REL).exists()


def test_workflow_watch_paths_derived_from_groups_include_bundled(bump_mod) -> None:
    paths = bump_mod.workflow_watch_paths()
    assert _BUNDLED_REL in paths
    assert _BACKUP_REL not in paths
    assert _CANONICAL_REL in paths
    # directory prefixes become recursive globs used by Actions path filters
    assert ".wheel-build/**" in paths
    assert "scripts/local-ci/**" in paths
    assert "context-replacement/**" in paths
    assert paths == tuple(dict.fromkeys(paths))  # de-duplicated, order-stable


def test_path_to_workflow_glob_directory_and_file(bump_mod) -> None:
    assert bump_mod.path_to_workflow_glob(".wheel-build/") == ".wheel-build/**"
    assert bump_mod.path_to_workflow_glob(_BUNDLED_REL) == _BUNDLED_REL
    with pytest.raises(ValueError):
        bump_mod.path_to_workflow_glob("   ")


def test_print_workflow_paths_cli_matches_helper(bump_mod, capsys) -> None:
    rc = bump_mod.main(["--print-workflow-paths"])
    assert rc == 0
    printed = [line for line in capsys.readouterr().out.splitlines() if line]
    assert printed == list(bump_mod.workflow_watch_paths())
    assert _BUNDLED_REL in printed
    assert _BACKUP_REL not in printed


def test_config_path_match_accepts_bundled_only_change(bump_mod) -> None:
    config = _config_group(bump_mod)
    assert bump_mod._path_matches(_BUNDLED_REL, config.paths)
    assert bump_mod._path_matches(_CANONICAL_REL, config.paths)
    assert not bump_mod._path_matches(_BACKUP_REL, config.paths)
    assert not bump_mod._path_matches("unrelated/file.py", config.paths)


def test_config_group_bumps_when_only_bundled_fallback_changes(bump_mod, monkeypatch, tmp_path, capsys) -> None:
    """A push that touches only the packaged fallback JSON must select config."""
    version_file = tmp_path / "model-config-version.txt"
    version_file.write_text("1.2.3\n", encoding="utf-8")

    # Build a temporary config group pointing at the temp version file.
    config = bump_mod.ArtifactGroup(
        name="config",
        paths=(
            _CANONICAL_REL,
            _BUNDLED_REL,
            "scripts/build_model_config_bundle.py",
        ),
        version_file=version_file,
        pattern=r"^([0-9]+)\.([0-9]+)\.([0-9]+)$",
        tag_prefix="cfg-v",
    )

    monkeypatch.setattr(bump_mod, "GROUPS", (config,))
    monkeypatch.setattr(
        bump_mod,
        "_get_changed_files",
        lambda before, after: [_BUNDLED_REL],
    )

    rc = bump_mod.main(["--before", "abc", "--after", "def", "--write"])
    assert rc == 0
    assert version_file.read_text(encoding="utf-8") == "1.2.4\n"
    # selection correctness is the RR-073 acceptance signal for bundled-only pushes
    payload = json.loads(capsys.readouterr().out)
    assert len(payload["bumped"]) == 1
    assert payload["bumped"][0]["name"] == "config"
    assert payload["bumped"][0]["tag"] == "cfg-v1.2.4"


def test_workflow_yaml_paths_are_superset_of_derived_paths(bump_mod) -> None:
    """Regression guard: live workflow must cover every GROUPS-derived path.

    Workflow edits are outside RR-073 write scope; this asserts current checkout
    alignment so a future dead backup path cannot reappear unnoticed.
    """
    text = _WORKFLOW.read_text(encoding="utf-8")
    derived = bump_mod.workflow_watch_paths()
    for path in derived:
        assert f'"{path}"' in text or f"'{path}'" in text, path
    assert _BUNDLED_REL in text
    assert _BACKUP_REL not in text


def test_forbidden_backup_path_rejected_from_groups(bump_mod) -> None:
    bad = bump_mod.ArtifactGroup(
        name="bad_config",
        paths=(_BACKUP_REL,),
        version_file=_REPO / "model-config-version.txt",
        pattern=r"^([0-9]+)\.([0-9]+)\.([0-9]+)$",
        tag_prefix="cfg-v",
    )
    with pytest.raises(ValueError, match="forbidden"):
        bump_mod.workflow_watch_paths((bad,))


def test_bump_version_text_preserves_semver_patch_increment(bump_mod) -> None:
    updated, old, new = bump_mod._bump_version_text("0.0.9\n", r"^([0-9]+)\.([0-9]+)\.([0-9]+)$")
    assert (old, new, updated) == ("0.0.9", "0.0.10", "0.0.10\n")


def test_main_requires_before_after_without_print_flag(bump_mod) -> None:
    with pytest.raises(SystemExit) as exc:
        bump_mod.main([])
    assert exc.value.code == 2
