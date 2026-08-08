"""Focused tests for CFG-002: immutable-inventory directory loading, startup
activation, fail-closed readiness, inventory drift detection, and compose mount.
"""

from __future__ import annotations

import os
import threading
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from litellm.proxy.pass_through_endpoints.aawm_alias_routing.config_startup import (
    ConfigDirectoryError,
    DEFAULT_CONFIG_DIR,
    _compile_inventory,
    _ConfigInventory,
    _InventoryFile,
    _scan_inventory,
    activate_alias_config_directory,
    compile_directory,
    get_startup_status,
    is_startup_failed,
    is_startup_healthy,
    is_startup_not_loaded,
    reset_startup_state,
)
import litellm.proxy.pass_through_endpoints.aawm_alias_routing.config_startup as _config_startup_module
from litellm.proxy.pass_through_endpoints.aawm_alias_routing.config_snapshot import (
    RoutingSnapshot,
)
from litellm.proxy.pass_through_endpoints.aawm_alias_routing.snapshot_select import (
    _lookup_active_snapshot_canonical_alias,
    _select_snapshot_candidates,
    get_active_routing_snapshot,
    set_active_routing_snapshot,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clean_startup_state():
    """Reset startup state and snapshot holder around every test."""
    reset_startup_state()
    set_active_routing_snapshot(None)
    yield
    reset_startup_state()
    set_active_routing_snapshot(None)


def _write_alias_yaml(directory: Path, filename: str, alias_name: str, model: str = "gpt-5.4-mini") -> Path:
    """Write a minimal valid alias config file and return its path."""
    doc = {
        "aliases": [
            {
                "name": alias_name,
                "candidates": [
                    {
                        "provider": "openai",
                        "model": model,
                        "route_family": "codex_responses",
                        "priority": 10,
                    },
                ],
            }
        ]
    }
    path = directory / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.dump(doc, default_flow_style=False), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Scan inventory (replaces TestDiscoverConfigFiles)
# ---------------------------------------------------------------------------


class TestScanInventory:
    def test_discovers_yaml_and_yml(self, tmp_path: Path) -> None:
        _write_alias_yaml(tmp_path, "a.yaml", "alpha")
        _write_alias_yaml(tmp_path, "b.yml", "beta")
        inv = _scan_inventory(tmp_path)
        assert inv.file_names == ("a.yaml", "b.yml")

    def test_rejects_txt_regular_file(self, tmp_path: Path) -> None:
        _write_alias_yaml(tmp_path, "a.yaml", "alpha")
        (tmp_path / "notes.txt").write_text("not yaml", encoding="utf-8")
        with pytest.raises(ConfigDirectoryError, match="unsupported file type"):
            _scan_inventory(tmp_path)

    def test_rejects_json_regular_file(self, tmp_path: Path) -> None:
        _write_alias_yaml(tmp_path, "a.yaml", "alpha")
        (tmp_path / "config.json").write_text("{}", encoding="utf-8")
        with pytest.raises(ConfigDirectoryError, match="unsupported file type"):
            _scan_inventory(tmp_path)

    def test_deterministic_relative_path_ordering(self, tmp_path: Path) -> None:
        _write_alias_yaml(tmp_path, "z.yaml", "zulu")
        _write_alias_yaml(tmp_path, "a.yaml", "alpha")
        _write_alias_yaml(tmp_path, "sub/m.yaml", "mike")
        inv = _scan_inventory(tmp_path)
        assert inv.file_names == ("a.yaml", "sub/m.yaml", "z.yaml")

    def test_nonexistent_directory_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigDirectoryError, match="does not exist"):
            _scan_inventory(tmp_path / "nope")

    def test_rejects_directory_with_yaml_suffix(self, tmp_path: Path) -> None:
        (tmp_path / "foo.yaml").mkdir()
        with pytest.raises(ConfigDirectoryError, match="directory with YAML suffix"):
            _scan_inventory(tmp_path)

    def test_rejects_symlink_with_yaml_suffix(self, tmp_path: Path) -> None:
        target = tmp_path / "real.yaml"
        _write_alias_yaml(tmp_path, "real.yaml", "alpha")
        link = tmp_path / "link.yaml"
        link.symlink_to(target)
        with pytest.raises(ConfigDirectoryError, match="symlink not allowed"):
            _scan_inventory(tmp_path)

    def test_rejects_fifo_with_yaml_suffix(self, tmp_path: Path) -> None:
        fifo_path = tmp_path / "pipe.yaml"
        os.mkfifo(str(fifo_path))
        with pytest.raises(ConfigDirectoryError, match="unsupported non-regular file"):
            _scan_inventory(tmp_path)

    def test_non_yaml_symlink_rejected(self, tmp_path: Path) -> None:
        target = tmp_path / "real.txt"
        target.write_text("hello", encoding="utf-8")
        link = tmp_path / "link.txt"
        link.symlink_to(target)
        _write_alias_yaml(tmp_path, "a.yaml", "alpha")
        with pytest.raises(ConfigDirectoryError, match="symlink not allowed"):
            _scan_inventory(tmp_path)

    def test_rejects_fifo_without_yaml_suffix(self, tmp_path: Path) -> None:
        fifo_path = tmp_path / "pipe.txt"
        os.mkfifo(str(fifo_path))
        _write_alias_yaml(tmp_path, "a.yaml", "alpha")
        with pytest.raises(ConfigDirectoryError, match="unsupported non-regular file"):
            _scan_inventory(tmp_path)

    def test_rejects_symlink_root(self, tmp_path: Path) -> None:
        real_dir = tmp_path / "real"
        real_dir.mkdir()
        _write_alias_yaml(real_dir, "a.yaml", "alpha")
        link_dir = tmp_path / "link"
        link_dir.symlink_to(real_dir)
        with pytest.raises(ConfigDirectoryError, match="must not be a symlink"):
            _scan_inventory(link_dir)

    def test_rejects_case_variant_yaml_suffix(self, tmp_path: Path) -> None:
        _write_alias_yaml(tmp_path, "a.yaml", "alpha")
        (tmp_path / "b.YAML").write_text("aliases: []", encoding="utf-8")
        with pytest.raises(ConfigDirectoryError, match="case-variant YAML suffix"):
            _scan_inventory(tmp_path)

    def test_rejects_yml_case_variant(self, tmp_path: Path) -> None:
        _write_alias_yaml(tmp_path, "a.yaml", "alpha")
        (tmp_path / "b.Yml").write_text("aliases: []", encoding="utf-8")
        with pytest.raises(ConfigDirectoryError, match="case-variant YAML suffix"):
            _scan_inventory(tmp_path)

    def test_rejects_unreadable_nested_directory(self, tmp_path: Path) -> None:
        nested = tmp_path / "sub"
        nested.mkdir()
        _write_alias_yaml(tmp_path, "a.yaml", "alpha")
        nested.chmod(0o000)
        try:
            with pytest.raises(ConfigDirectoryError, match="unreadable directory"):
                _scan_inventory(tmp_path)
        finally:
            nested.chmod(0o755)

    def test_inventory_captures_raw_bytes(self, tmp_path: Path) -> None:
        _write_alias_yaml(tmp_path, "a.yaml", "alpha")
        inv = _scan_inventory(tmp_path)
        assert len(inv.files) == 1
        assert inv.files[0].raw_bytes == (tmp_path / "a.yaml").read_bytes()
        assert inv.files[0].content_digest
        assert inv.files[0].size == len(inv.files[0].raw_bytes)

    def test_inventory_has_identity_metadata(self, tmp_path: Path) -> None:
        _write_alias_yaml(tmp_path, "a.yaml", "alpha")
        inv = _scan_inventory(tmp_path)
        f = inv.files[0]
        assert f.st_dev > 0
        assert f.st_ino > 0
        assert f.st_mtime_ns > 0
        assert f.st_ctime_ns > 0

    def test_inventory_directories_have_sorted_children(self, tmp_path: Path) -> None:
        _write_alias_yaml(tmp_path, "b.yaml", "beta")
        _write_alias_yaml(tmp_path, "a.yaml", "alpha")
        inv = _scan_inventory(tmp_path)
        root_dir = inv.directories[0]
        assert root_dir.relative_name == "."
        assert root_dir.child_names == ("a.yaml", "b.yaml")


# ---------------------------------------------------------------------------
# Compile directory
# ---------------------------------------------------------------------------


class TestCompileDirectory:
    def test_single_file_compiles(self, tmp_path: Path) -> None:
        _write_alias_yaml(tmp_path, "basic.yaml", "basic")
        snapshot = compile_directory(tmp_path)
        assert isinstance(snapshot, RoutingSnapshot)
        assert "basic" in snapshot.aliases
        assert snapshot.config_hash
        assert snapshot.config_version

    def test_multi_file_merge(self, tmp_path: Path) -> None:
        _write_alias_yaml(tmp_path, "a.yaml", "alpha")
        _write_alias_yaml(tmp_path, "b.yaml", "beta")
        snapshot = compile_directory(tmp_path)
        assert set(snapshot.aliases.keys()) == {"alpha", "beta"}

    def test_duplicate_alias_across_files_raises(self, tmp_path: Path) -> None:
        _write_alias_yaml(tmp_path, "a.yaml", "dup")
        _write_alias_yaml(tmp_path, "b.yaml", "dup", model="gpt-5.6-luna")
        with pytest.raises(ConfigDirectoryError, match="duplicate alias"):
            compile_directory(tmp_path)

    def test_case_insensitive_duplicate_across_files_raises(self, tmp_path: Path) -> None:
        _write_alias_yaml(tmp_path, "a.yaml", "Example")
        _write_alias_yaml(tmp_path, "b.yaml", "example", model="gpt-5.6-luna")
        with pytest.raises(ConfigDirectoryError, match="case-insensitive"):
            compile_directory(tmp_path)

    def test_case_insensitive_duplicate_within_file_raises(self, tmp_path: Path) -> None:
        doc = {
            "aliases": [
                {
                    "name": "Example",
                    "candidates": [
                        {"provider": "openai", "model": "gpt-5.4-mini", "route_family": "codex_responses", "priority": 10}
                    ],
                },
                {
                    "name": "example",
                    "candidates": [
                        {"provider": "openai", "model": "gpt-5.6-luna", "route_family": "codex_responses", "priority": 10}
                    ],
                },
            ]
        }
        (tmp_path / "a.yaml").write_text(yaml.dump(doc), encoding="utf-8")
        with pytest.raises(ValidationError, match="duplicate alias name"):
            compile_directory(tmp_path)

    def test_empty_directory_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigDirectoryError, match="no YAML config files"):
            compile_directory(tmp_path)

    def test_invalid_yaml_raises(self, tmp_path: Path) -> None:
        (tmp_path / "bad.yaml").write_text("{ invalid: [unclosed", encoding="utf-8")
        with pytest.raises(ConfigDirectoryError, match="invalid YAML"):
            compile_directory(tmp_path)

    def test_schema_validation_failure_raises(self, tmp_path: Path) -> None:
        (tmp_path / "bad.yaml").write_text(
            yaml.dump({"aliases": [{"name": "x", "candidates": []}]}),
            encoding="utf-8",
        )
        with pytest.raises(Exception):  # pydantic.ValidationError
            compile_directory(tmp_path)

    def test_conflicting_defaults_raises(self, tmp_path: Path) -> None:
        doc_a = {
            "defaults": {"route_family": "codex_responses"},
            "aliases": [
                {"name": "a", "candidates": [
                    {"provider": "openai", "model": "gpt-5.4-mini", "priority": 1}
                ]}
            ],
        }
        doc_b = {
            "defaults": {"route_family": "anthropic_messages"},
            "aliases": [
                {"name": "b", "candidates": [
                    {"provider": "openai", "model": "gpt-5.6-luna", "priority": 1}
                ]}
            ],
        }
        (tmp_path / "a.yaml").write_text(yaml.dump(doc_a), encoding="utf-8")
        (tmp_path / "b.yaml").write_text(yaml.dump(doc_b), encoding="utf-8")
        with pytest.raises(ConfigDirectoryError, match="conflicting defaults"):
            compile_directory(tmp_path)

    def test_identical_defaults_ok(self, tmp_path: Path) -> None:
        defaults = {"route_family": "codex_responses"}
        for name in ("a", "b"):
            doc = {
                "defaults": dict(defaults),
                "aliases": [
                    {"name": name, "candidates": [
                        {"provider": "openai", "model": f"gpt-5.4-mini-{name}", "priority": 1}
                    ]}
                ],
            }
            (tmp_path / f"{name}.yaml").write_text(yaml.dump(doc), encoding="utf-8")
        snapshot = compile_directory(tmp_path)
        assert set(snapshot.aliases.keys()) == {"a", "b"}

    def test_deterministic_hash_across_runs(self, tmp_path: Path) -> None:
        _write_alias_yaml(tmp_path, "a.yaml", "alpha")
        _write_alias_yaml(tmp_path, "b.yaml", "beta")
        s1 = compile_directory(tmp_path)
        s2 = compile_directory(tmp_path)
        assert s1.config_hash == s2.config_hash
        assert s1.config_version == s2.config_version

    def test_invalid_utf8_raises(self, tmp_path: Path) -> None:
        (tmp_path / "bad.yaml").write_bytes(b"\xff\xfe\x00\x01aliases: []")
        with pytest.raises(ConfigDirectoryError, match="cannot read config file"):
            compile_directory(tmp_path)

    def test_symlink_rejected_at_compile(self, tmp_path: Path) -> None:
        _write_alias_yaml(tmp_path, "real.yaml", "alpha")
        (tmp_path / "link.yaml").symlink_to(tmp_path / "real.yaml")
        with pytest.raises(ConfigDirectoryError, match="symlink not allowed"):
            compile_directory(tmp_path)

    def test_file_addition_changes_hash(self, tmp_path: Path) -> None:
        _write_alias_yaml(tmp_path, "a.yaml", "alpha")
        s1 = compile_directory(tmp_path)
        _write_alias_yaml(tmp_path, "b.yaml", "beta")
        s2 = compile_directory(tmp_path)
        assert s1.config_hash != s2.config_hash

    def test_file_removal_changes_hash(self, tmp_path: Path) -> None:
        _write_alias_yaml(tmp_path, "a.yaml", "alpha")
        _write_alias_yaml(tmp_path, "b.yaml", "beta")
        s1 = compile_directory(tmp_path)
        (tmp_path / "b.yaml").unlink()
        s2 = compile_directory(tmp_path)
        assert s1.config_hash != s2.config_hash

    def test_no_files_parameter_override(self, tmp_path: Path) -> None:
        """compile_directory has no files= parameter; only full scan."""
        import inspect
        sig = inspect.signature(compile_directory)
        assert "files" not in sig.parameters


# ---------------------------------------------------------------------------
# Compile API: no external/subset inventory injection
# ---------------------------------------------------------------------------


class TestCompileAPINoInjection:
    def test_compile_inventory_rejects_absolute_relative_name(self) -> None:
        """_compile_inventory rejects inventory entries with absolute names."""
        inv = _ConfigInventory(
            root_relative_name=".",
            root_st_dev=1,
            root_st_ino=1,
            root_st_mtime_ns=1,
            root_st_ctime_ns=1,
            files=(
                _InventoryFile(
                    relative_name="/etc/passwd",
                    raw_bytes=b"aliases: []",
                    content_digest="abc",
                    size=11,
                    st_dev=1,
                    st_ino=1,
                    st_mtime_ns=1,
                    st_ctime_ns=1,
                ),
            ),
            directories=(),
        )
        with pytest.raises(ConfigDirectoryError, match="invalid relative name"):
            _compile_inventory(inv)

    def test_compile_inventory_rejects_dotdot_relative_name(self) -> None:
        """_compile_inventory rejects inventory entries with '..' components."""
        inv = _ConfigInventory(
            root_relative_name=".",
            root_st_dev=1,
            root_st_ino=1,
            root_st_mtime_ns=1,
            root_st_ctime_ns=1,
            files=(
                _InventoryFile(
                    relative_name="../evil.yaml",
                    raw_bytes=b"aliases: []",
                    content_digest="abc",
                    size=11,
                    st_dev=1,
                    st_ino=1,
                    st_mtime_ns=1,
                    st_ctime_ns=1,
                ),
            ),
            directories=(),
        )
        with pytest.raises(ConfigDirectoryError, match="invalid relative name"):
            _compile_inventory(inv)

    def test_compile_inventory_rejects_empty_inventory(self) -> None:
        """_compile_inventory rejects an empty file set."""
        inv = _ConfigInventory(
            root_relative_name=".",
            root_st_dev=1,
            root_st_ino=1,
            root_st_mtime_ns=1,
            root_st_ctime_ns=1,
            files=(),
            directories=(),
        )
        with pytest.raises(ConfigDirectoryError, match="no YAML config files"):
            _compile_inventory(inv)


# ---------------------------------------------------------------------------
# Empty aliases
# ---------------------------------------------------------------------------


class TestEmptyAliases:
    def test_empty_aliases_list_is_unhealthy(self, tmp_path: Path) -> None:
        (tmp_path / "empty.yaml").write_text(
            yaml.dump({"aliases": []}), encoding="utf-8"
        )
        with pytest.raises(ConfigDirectoryError, match="no aliases"):
            compile_directory(tmp_path)

    def test_empty_aliases_activation_fails_closed(self, tmp_path: Path) -> None:
        (tmp_path / "empty.yaml").write_text(
            yaml.dump({"aliases": []}), encoding="utf-8"
        )
        activate_alias_config_directory(tmp_path)
        assert not is_startup_healthy()
        assert get_active_routing_snapshot() is None


# ---------------------------------------------------------------------------
# Startup activation + readiness gate
# ---------------------------------------------------------------------------


class TestStartupActivation:
    def test_successful_activation(self, tmp_path: Path) -> None:
        _write_alias_yaml(tmp_path, "basic.yaml", "basic")
        activate_alias_config_directory(tmp_path)
        assert is_startup_healthy()
        status = get_startup_status()
        assert status["state"] == "active"
        assert status["alias_count"] == 1
        assert status["aliases"] == ["basic"]
        assert status["files"] == ["basic.yaml"]
        assert status["config_hash"]
        assert status["config_version"]
        assert "config_epoch" in status
        assert status["activation_result"] == "success"
        snap = get_active_routing_snapshot()
        assert snap is not None
        assert snap.config_hash == status["config_hash"]

    def test_failed_activation_blocks_readiness(self, tmp_path: Path) -> None:
        (tmp_path / "bad.yaml").write_text(":::invalid:::", encoding="utf-8")
        activate_alias_config_directory(tmp_path)
        assert not is_startup_healthy()
        status = get_startup_status()
        assert status["state"] == "failed"
        assert "error_class" in status
        assert get_active_routing_snapshot() is None

    def test_empty_directory_fails_closed(self, tmp_path: Path) -> None:
        activate_alias_config_directory(tmp_path)
        assert not is_startup_healthy()

    def test_not_loaded_initially(self) -> None:
        assert not is_startup_healthy()
        assert is_startup_not_loaded()
        status = get_startup_status()
        assert status["state"] == "not_loaded"

    def test_sanitized_status_no_raw_yaml(self, tmp_path: Path) -> None:
        _write_alias_yaml(tmp_path, "basic.yaml", "basic")
        activate_alias_config_directory(tmp_path)
        status = get_startup_status()
        status_str = str(status)
        assert "candidates" not in status_str
        assert "gpt-5.4-mini" not in status_str

    def test_sanitized_failed_status_no_raw_error(self, tmp_path: Path) -> None:
        (tmp_path / "bad.yaml").write_text(":::invalid:::", encoding="utf-8")
        activate_alias_config_directory(tmp_path)
        status = get_startup_status()
        status_str = str(status)
        assert str(tmp_path) not in status_str
        assert "invalid YAML" not in status_str
        assert set(status.keys()) == {"state", "error_class"}

    def test_default_config_dir_loads(self) -> None:
        if not DEFAULT_CONFIG_DIR.is_dir():
            pytest.skip("aawm_alias_config directory absent")
        activate_alias_config_directory()
        assert is_startup_healthy()
        status = get_startup_status()
        assert status["state"] == "active"
        assert "basic" in status["aliases"]


# ---------------------------------------------------------------------------
# Not-loaded readiness 503
# ---------------------------------------------------------------------------


class TestNotLoadedReadiness503:
    def test_not_loaded_is_not_healthy(self) -> None:
        """not_loaded state must not pass is_startup_healthy."""
        assert is_startup_not_loaded()
        assert not is_startup_healthy()

    def test_not_loaded_status_state(self) -> None:
        status = get_startup_status()
        assert status["state"] == "not_loaded"
        assert "config_hash" not in status


# ---------------------------------------------------------------------------
# Success evidence contents and secret exclusion
# ---------------------------------------------------------------------------


class TestSuccessEvidence:
    def test_success_status_contains_required_fields(self, tmp_path: Path) -> None:
        _write_alias_yaml(tmp_path, "basic.yaml", "basic")
        activate_alias_config_directory(tmp_path)
        status = get_startup_status()
        assert status["state"] == "active"
        assert "config_hash" in status
        assert "config_version" in status
        assert "config_epoch" in status
        assert "files" in status
        assert "aliases" in status
        assert "alias_count" in status
        assert status["activation_result"] == "success"

    def test_success_status_excludes_secrets(self, tmp_path: Path) -> None:
        _write_alias_yaml(tmp_path, "basic.yaml", "basic")
        activate_alias_config_directory(tmp_path)
        status = get_startup_status()
        status_str = str(status)
        # No absolute paths.
        assert str(tmp_path) not in status_str
        # No raw YAML content.
        assert "candidates" not in status_str
        assert "provider" not in status_str
        # No model strings.
        assert "gpt-5.4-mini" not in status_str

    def test_success_log_contains_relative_names(self, tmp_path: Path, caplog) -> None:
        import logging
        _write_alias_yaml(tmp_path, "basic.yaml", "basic")
        with caplog.at_level(logging.INFO, logger="litellm.proxy.pass_through_endpoints.aawm_alias_routing.config_startup"):
            activate_alias_config_directory(tmp_path)
        log_text = caplog.text
        assert "basic.yaml" in log_text
        assert "config_version=" in log_text
        assert "config_hash=" in log_text
        assert "result=success" in log_text
        # No absolute paths in log.
        assert str(tmp_path) not in log_text


# ---------------------------------------------------------------------------
# Deterministic names/status
# ---------------------------------------------------------------------------


class TestDeterministicNames:
    def test_file_names_are_relative_posix(self, tmp_path: Path) -> None:
        _write_alias_yaml(tmp_path, "sub/b.yaml", "beta")
        _write_alias_yaml(tmp_path, "a.yaml", "alpha")
        inv = _scan_inventory(tmp_path)
        for name in inv.file_names:
            assert not name.startswith("/")
            assert "\\" not in name
            assert ".." not in name.split("/")

    def test_status_files_are_relative(self, tmp_path: Path) -> None:
        _write_alias_yaml(tmp_path, "sub/basic.yaml", "basic")
        activate_alias_config_directory(tmp_path)
        status = get_startup_status()
        for f in status["files"]:
            assert not f.startswith("/")
            assert ".." not in f


# ---------------------------------------------------------------------------
# Inventory drift detection (parameterized)
# ---------------------------------------------------------------------------


class TestInventoryDrift:
    """Parameterized drift cases: activate_alias_config_directory must fail
    closed when the directory changes between the first and second scan."""

    def _setup_good_dir(self, tmp_path: Path) -> Path:
        good = tmp_path / "cfg"
        good.mkdir()
        _write_alias_yaml(good, "basic.yaml", "basic")
        return good

    @pytest.mark.parametrize("drift_type", [
        "add",
        "remove",
        "edit",
        "replace_file",
        "replace_root",
        "replace_nested_dir",
    ])
    def test_drift_fails_closed(self, tmp_path: Path, drift_type: str, monkeypatch: pytest.MonkeyPatch) -> None:
        """Each drift type between scan and revalidation fails closed."""
        good = self._setup_good_dir(tmp_path)
        nested = good / "sub"
        nested.mkdir()
        _write_alias_yaml(nested, "extra.yaml", "extra")

        # Monkeypatch _scan_inventory to inject drift after the first call.
        real_scan = _config_startup_module._scan_inventory
        call_count = [0]

        def _drifting_scan(config_dir: Path) -> _ConfigInventory:
            call_count[0] += 1
            if call_count[0] == 2:
                # Inject drift before the second scan.
                if drift_type == "add":
                    _write_alias_yaml(good, "injected.yaml", "injected")
                elif drift_type == "remove":
                    (good / "basic.yaml").unlink()
                elif drift_type == "edit":
                    _write_alias_yaml(good, "basic.yaml", "basic", model="gpt-5.9-edited")
                elif drift_type == "replace_file":
                    (good / "basic.yaml").unlink()
                    _write_alias_yaml(good, "basic.yaml", "basic", model="gpt-5.9-replaced")
                elif drift_type == "replace_root":
                    # Replace root with symlink to external dir.
                    external = tmp_path / "external"
                    external.mkdir()
                    _write_alias_yaml(external, "evil.yaml", "evil")
                    good.rename(tmp_path / "cfg_moved")
                    good.symlink_to(external)
                elif drift_type == "replace_nested_dir":
                    external = tmp_path / "external_nested"
                    external.mkdir()
                    _write_alias_yaml(external, "evil.yaml", "evil")
                    nested.rename(tmp_path / "sub_moved")
                    nested.symlink_to(external)
            return real_scan(config_dir)

        monkeypatch.setattr(_config_startup_module, "_scan_inventory", _drifting_scan)
        activate_alias_config_directory(good)
        assert not is_startup_healthy()
        assert is_startup_failed()
        assert get_active_routing_snapshot() is None

    def test_no_drift_succeeds(self, tmp_path: Path) -> None:
        """Without drift, activation succeeds."""
        good = self._setup_good_dir(tmp_path)
        activate_alias_config_directory(good)
        assert is_startup_healthy()
        assert get_active_routing_snapshot() is not None


# ---------------------------------------------------------------------------
# Stale snapshot prevention
# ---------------------------------------------------------------------------


class TestStaleSnapshotPrevention:
    def test_failure_clears_prior_snapshot(self, tmp_path: Path) -> None:
        good_dir = tmp_path / "good"
        good_dir.mkdir()
        _write_alias_yaml(good_dir, "basic.yaml", "basic")
        activate_alias_config_directory(good_dir)
        assert is_startup_healthy()
        assert get_active_routing_snapshot() is not None

        bad_dir = tmp_path / "bad"
        bad_dir.mkdir()
        (bad_dir / "broken.yaml").write_text(":::bad:::", encoding="utf-8")
        activate_alias_config_directory(bad_dir)
        assert not is_startup_healthy()
        assert get_active_routing_snapshot() is None

    def test_failure_clears_alias_lookup_and_selection(self, tmp_path: Path) -> None:
        good_dir = tmp_path / "good"
        good_dir.mkdir()
        _write_alias_yaml(good_dir, "basic.yaml", "basic")
        activate_alias_config_directory(good_dir)

        bad_dir = tmp_path / "bad"
        bad_dir.mkdir()
        (bad_dir / "broken.yaml").write_text(":::bad:::", encoding="utf-8")
        activate_alias_config_directory(bad_dir)

        assert _lookup_active_snapshot_canonical_alias("basic") is None
        assert _select_snapshot_candidates("basic", ingress="codex") == ()
        assert _select_snapshot_candidates("basic", ingress="anthropic") == ()


# ---------------------------------------------------------------------------
# Selector integration with snapshot
# ---------------------------------------------------------------------------


class TestSelectorIntegration:
    def test_codex_selector_uses_snapshot(self, tmp_path: Path) -> None:
        good_dir = tmp_path / "good"
        good_dir.mkdir()
        _write_alias_yaml(good_dir, "basic.yaml", "basic")
        activate_alias_config_directory(good_dir)
        assert is_startup_healthy()

        result = _select_snapshot_candidates("basic", ingress="codex")
        assert len(result) > 0
        assert result[0]["model"] == "gpt-5.4-mini"

    def test_anthropic_selector_uses_snapshot(self, tmp_path: Path) -> None:
        good_dir = tmp_path / "good"
        good_dir.mkdir()
        _write_alias_yaml(good_dir, "basic.yaml", "basic")
        activate_alias_config_directory(good_dir)
        assert is_startup_healthy()

        selector_result = _select_snapshot_candidates(
            "basic",
            ingress="anthropic",
        )
        assert len(selector_result) > 0
        assert selector_result[0]["route_family"] == (
            "anthropic_openai_responses_adapter"
        )
        assert "config_epoch_tag" in selector_result[0]

    def test_basic_anthropic_alias_resolves_from_snapshot(
        self, tmp_path: Path
    ) -> None:
        good_dir = tmp_path / "good"
        good_dir.mkdir()
        _write_alias_yaml(good_dir, "basic.yaml", "basic")
        activate_alias_config_directory(good_dir)
        assert is_startup_healthy()

        candidates = _select_snapshot_candidates(
            "basic",
            ingress="anthropic",
        )
        assert [candidate["model"] for candidate in candidates] == ["gpt-5.4-mini"]
        assert candidates[0]["route_family"] == "anthropic_openai_responses_adapter"
        assert "config_epoch_tag" in candidates[0]


# ---------------------------------------------------------------------------
# Failure-window: all getters zero
# ---------------------------------------------------------------------------


class TestFailureWindowAliasesUnavailable:
    def test_alias_lookup_and_selection_fail_closed_after_failure(
        self, tmp_path: Path
    ) -> None:
        good_dir = tmp_path / "good"
        good_dir.mkdir()
        _write_alias_yaml(good_dir, "basic.yaml", "basic")
        activate_alias_config_directory(good_dir)

        bad_dir = tmp_path / "bad"
        bad_dir.mkdir()
        (bad_dir / "broken.yaml").write_text(":::bad:::", encoding="utf-8")
        activate_alias_config_directory(bad_dir)
        assert is_startup_failed()

        assert _lookup_active_snapshot_canonical_alias("basic") is None
        assert _select_snapshot_candidates("basic", ingress="codex") == ()
        assert _select_snapshot_candidates("basic", ingress="anthropic") == ()

    def test_failure_window_pause_all_getters_zero(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        good_dir = tmp_path / "good"
        good_dir.mkdir()
        _write_alias_yaml(good_dir, "basic.yaml", "basic")
        activate_alias_config_directory(good_dir)

        bad_dir = tmp_path / "bad"
        bad_dir.mkdir()
        (bad_dir / "broken.yaml").write_text(":::bad:::", encoding="utf-8")

        real_set = set_active_routing_snapshot
        observations: dict[str, object] = {}
        pause_entered = threading.Event()
        pause_release = threading.Event()

        def _pausing_set(snapshot):
            if snapshot is None:
                observations["lookup"] = _lookup_active_snapshot_canonical_alias(
                    "basic"
                )
                observations["codex"] = _select_snapshot_candidates(
                    "basic",
                    ingress="codex",
                )
                observations["anthropic"] = _select_snapshot_candidates(
                    "basic",
                    ingress="anthropic",
                )
                pause_entered.set()
                pause_release.wait(timeout=10)
            return real_set(snapshot)

        monkeypatch.setattr(
            _config_startup_module, "set_active_routing_snapshot", _pausing_set
        )

        t = threading.Thread(
            target=lambda: activate_alias_config_directory(bad_dir), daemon=True
        )
        t.start()
        assert pause_entered.wait(timeout=10)

        assert get_active_routing_snapshot() is not None
        assert observations == {
            "lookup": None,
            "codex": (),
            "anthropic": (),
        }

        pause_release.set()
        t.join(timeout=10)


# ---------------------------------------------------------------------------
# Production hook removed
# ---------------------------------------------------------------------------


class TestProductionHookRemoved:
    def test_no_failure_state_published_hook_in_production(self) -> None:
        assert not hasattr(_config_startup_module, "_failure_state_published_hook")

    def test_activate_does_not_reference_hook(self) -> None:
        import inspect
        source = inspect.getsource(_config_startup_module.activate_alias_config_directory)
        assert "_failure_state_published_hook" not in source


# ---------------------------------------------------------------------------
# Short-read oversize detection
# ---------------------------------------------------------------------------


class TestShortReadOversize:
    def test_oversize_detected_with_simulated_short_reads(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import litellm.proxy.pass_through_endpoints.aawm_alias_routing.config_startup as cs

        large = tmp_path / "large.yaml"
        large.write_bytes(b"a" * (cs._MAX_CONFIG_FILE_BYTES + 1))

        real_os_read = os.read
        call_count = [0]

        def _short_read(fd, n):
            call_count[0] += 1
            return real_os_read(fd, min(n, 4096))

        monkeypatch.setattr(os, "read", _short_read)
        with pytest.raises(ConfigDirectoryError, match="exceeds maximum size"):
            compile_directory(tmp_path)
        assert call_count[0] > 1

    def test_exact_limit_file_accepted_via_short_reads(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import litellm.proxy.pass_through_endpoints.aawm_alias_routing.config_startup as cs

        header = b"aliases:\n- name: basic\n  candidates:\n  - provider: openai\n    model: gpt-5.4-mini\n    route_family: codex_responses\n    priority: 10\n"
        pad_len = cs._MAX_CONFIG_FILE_BYTES - len(header)
        exact = tmp_path / "exact.yaml"
        exact.write_bytes(header + b"#" * pad_len)
        assert exact.stat().st_size == cs._MAX_CONFIG_FILE_BYTES

        real_os_read = os.read

        def _short_read(fd, n):
            return real_os_read(fd, min(n, 8192))

        monkeypatch.setattr(os, "read", _short_read)
        try:
            compile_directory(tmp_path)
        except ConfigDirectoryError as exc:
            assert "exceeds maximum size" not in str(exc)


# ---------------------------------------------------------------------------
# Coherent snapshot reference (single capture per selection)
# ---------------------------------------------------------------------------


class TestCoherentSnapshotReference:
    def test_clear_between_steps_preserves_coherent_selection(
        self, tmp_path: Path
    ) -> None:
        dir_a = tmp_path / "a"
        dir_a.mkdir()
        _write_alias_yaml(dir_a, "basic.yaml", "basic", model="gpt-5.4-mini")
        activate_alias_config_directory(dir_a)
        snap_a = get_active_routing_snapshot()
        assert snap_a is not None

        result_a = _select_snapshot_candidates("basic", ingress="codex")
        assert len(result_a) > 0
        epoch_a = snap_a.config_hash
        assert all(c["config_epoch_tag"] == epoch_a for c in result_a)

        dir_b = tmp_path / "b"
        dir_b.mkdir()
        _write_alias_yaml(dir_b, "basic.yaml", "basic", model="gpt-5.6-luna")
        activate_alias_config_directory(dir_b)
        snap_b = get_active_routing_snapshot()
        assert snap_b is not None
        assert snap_b.config_hash != epoch_a

        result_b = _select_snapshot_candidates("basic", ingress="codex")
        assert len(result_b) > 0
        epoch_b = snap_b.config_hash
        assert all(c["config_epoch_tag"] == epoch_b for c in result_b)
        assert result_a[0]["model"] != result_b[0]["model"]

    def test_anthropic_clear_swap_coherent(self, tmp_path: Path) -> None:
        dir_a = tmp_path / "a"
        dir_a.mkdir()
        _write_alias_yaml(dir_a, "basic.yaml", "basic", model="gpt-5.4-mini")
        activate_alias_config_directory(dir_a)
        snap_a = get_active_routing_snapshot()
        assert snap_a is not None

        result_a = _select_snapshot_candidates("basic", ingress="anthropic")
        assert len(result_a) > 0
        epoch_a = snap_a.config_hash
        assert all(c["config_epoch_tag"] == epoch_a for c in result_a)

        set_active_routing_snapshot(None)
        result_cleared = _select_snapshot_candidates("basic", ingress="anthropic")
        assert result_cleared == ()

    def test_single_snapshot_capture_no_second_global_fetch(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import litellm.proxy.pass_through_endpoints.aawm_alias_routing.snapshot_select as ss

        good_dir = tmp_path / "good"
        good_dir.mkdir()
        _write_alias_yaml(good_dir, "basic.yaml", "basic")
        activate_alias_config_directory(good_dir)

        call_count = [0]
        real_get = ss.get_active_routing_snapshot

        def _counting_get():
            call_count[0] += 1
            return real_get()

        monkeypatch.setattr(ss, "get_active_routing_snapshot", _counting_get)
        result = ss._select_snapshot_candidates("basic", ingress="codex")
        assert len(result) > 0
        assert call_count[0] == 1


# ---------------------------------------------------------------------------
# Non-regular descriptor coverage
# ---------------------------------------------------------------------------


class TestNonRegularDescriptorCoverage:
    def test_fifo_rejected_by_scan(self, tmp_path: Path) -> None:
        """A FIFO in the config directory is rejected during scan."""
        _write_alias_yaml(tmp_path, "a.yaml", "alpha")
        os.mkfifo(str(tmp_path / "pipe.yaml"))
        with pytest.raises(ConfigDirectoryError, match="unsupported non-regular file"):
            compile_directory(tmp_path)

    def test_dev_null_symlink_rejected(self, tmp_path: Path) -> None:
        """/dev/null via symlink is rejected."""
        _write_alias_yaml(tmp_path, "a.yaml", "alpha")
        (tmp_path / "evil.yaml").symlink_to("/dev/null")
        with pytest.raises(ConfigDirectoryError, match="symlink not allowed"):
            compile_directory(tmp_path)


# ---------------------------------------------------------------------------
# TOCTOU: root/ancestor symlink replacement
# ---------------------------------------------------------------------------


class TestTOCTOUAncestorSymlink:
    def test_root_replaced_with_symlink_fails_closed(self, tmp_path: Path) -> None:
        """Config root replaced by a symlink: compile must fail."""
        real_dir = tmp_path / "real_config"
        real_dir.mkdir()
        _write_alias_yaml(real_dir, "basic.yaml", "basic")

        external_dir = tmp_path / "external"
        external_dir.mkdir()
        _write_alias_yaml(external_dir, "evil.yaml", "evil")

        real_dir.rename(tmp_path / "real_config_moved")
        real_dir.symlink_to(external_dir)

        with pytest.raises(ConfigDirectoryError):
            compile_directory(real_dir)

    def test_nested_ancestor_replaced_with_symlink_fails_closed(
        self, tmp_path: Path
    ) -> None:
        """A nested ancestor directory replaced by a symlink: compile must fail."""
        config_root = tmp_path / "config"
        config_root.mkdir()
        nested = config_root / "sub"
        nested.mkdir()
        _write_alias_yaml(nested, "basic.yaml", "basic")

        external_dir = tmp_path / "external"
        external_dir.mkdir()
        _write_alias_yaml(external_dir, "evil.yaml", "evil")

        nested.rename(tmp_path / "sub_moved")
        nested.symlink_to(external_dir)

        with pytest.raises(ConfigDirectoryError):
            compile_directory(config_root)

    def test_root_symlink_replacement_activation_fails_closed(
        self, tmp_path: Path
    ) -> None:
        real_dir = tmp_path / "cfg"
        real_dir.mkdir()
        _write_alias_yaml(real_dir, "basic.yaml", "basic")

        external_dir = tmp_path / "external"
        external_dir.mkdir()
        _write_alias_yaml(external_dir, "evil.yaml", "evil")

        activate_alias_config_directory(real_dir)
        assert is_startup_healthy()

        reset_startup_state()
        set_active_routing_snapshot(None)
        real_dir.rename(tmp_path / "cfg_moved")
        real_dir.symlink_to(external_dir)

        activate_alias_config_directory(real_dir)
        assert not is_startup_healthy()
        assert is_startup_failed()
        assert get_active_routing_snapshot() is None

    def test_valid_nested_file_still_compiles(
        self, tmp_path: Path
    ) -> None:
        """Regression: legitimate nested files still compile correctly."""
        nested = tmp_path / "sub" / "deep"
        nested.mkdir(parents=True)
        _write_alias_yaml(nested, "basic.yaml", "basic")
        _write_alias_yaml(tmp_path, "top.yaml", "top")

        snapshot = compile_directory(tmp_path)
        assert set(snapshot.aliases.keys()) == {"basic", "top"}


# ---------------------------------------------------------------------------
# Structural dev read-only mount
# ---------------------------------------------------------------------------


class TestDevReadOnlyMount:
    def test_compose_has_read_only_alias_config_mount(self) -> None:
        """docker-compose.dev.yml mounts aawm_alias_config as :ro."""
        compose_path = Path(__file__).resolve().parents[4] / "docker-compose.dev.yml"
        if not compose_path.exists():
            pytest.skip("docker-compose.dev.yml not found")
        content = compose_path.read_text(encoding="utf-8")
        assert "./litellm/proxy/aawm_alias_config:/app/litellm/proxy/aawm_alias_config:ro" in content

    def test_compose_has_read_only_alias_routing_mount(self) -> None:
        """docker-compose.dev.yml mounts alias routing dependencies as :ro."""
        compose_path = Path(__file__).resolve().parents[4] / "docker-compose.dev.yml"
        if not compose_path.exists():
            pytest.skip("docker-compose.dev.yml not found")
        content = compose_path.read_text(encoding="utf-8")
        assert "aawm_alias_routing:/app/litellm/proxy/pass_through_endpoints/aawm_alias_routing:ro" in content
        assert (
            "./litellm/secret_managers/codex_oauth_inventory.py:"
            "/app/litellm/secret_managers/codex_oauth_inventory.py:ro"
        ) in content
        assert (
            "./litellm/llms/xai/route_descriptors.py:"
            "/app/litellm/llms/xai/route_descriptors.py:ro"
        ) in content
        assert (
            "./litellm/secret_managers/grok_oidc_auth_path.py:"
            "/app/litellm/secret_managers/grok_oidc_auth_path.py:ro"
        ) in content
        assert (
            "./litellm/proxy/pass_through_endpoints/aawm_context_query.py:"
            "/app/litellm/proxy/pass_through_endpoints/aawm_context_query.py:ro"
        ) in content
