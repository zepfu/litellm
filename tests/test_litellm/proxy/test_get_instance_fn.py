"""Tests for get_instance_fn package-aware module resolution.

Covers:
- File-module loading (existing behavior preserved)
- Package-module loading with relative imports and instance retrieval
- Missing module raises bounded ImportError
- Real litellm.integrations.aawm_agent_identity.aawm_agent_identity_instance
  resolves via config_file_path without remote calls
"""

import os
import sys
import textwrap

import pytest

sys.path.insert(0, os.path.abspath("../.."))

from litellm.proxy.types_utils.utils import get_instance_fn


@pytest.fixture
def file_module_dir(tmp_path):
    """Create a simple file-based module for testing."""
    mod_file = tmp_path / "my_callbacks.py"
    mod_file.write_text(
        textwrap.dedent("""\
            class MyHandler:
                def __init__(self):
                    self.loaded = True

            my_handler_instance = MyHandler()
        """)
    )
    config_file = tmp_path / "config.yaml"
    config_file.write_text("litellm_settings: {}\n")
    return tmp_path


@pytest.fixture
def package_module_dir(tmp_path):
    """Create a package-style module with relative imports."""
    pkg_dir = tmp_path / "my_package"
    pkg_dir.mkdir()
    # Submodule that will be imported relatively
    helper = pkg_dir / "helper.py"
    helper.write_text(
        textwrap.dedent("""\
            HELPER_VALUE = 42
        """)
    )
    # Package __init__.py with a relative import
    init = pkg_dir / "__init__.py"
    init.write_text(
        textwrap.dedent("""\
            from .helper import HELPER_VALUE

            class PackageHandler:
                def __init__(self):
                    self.value = HELPER_VALUE

            package_handler_instance = PackageHandler()
        """)
    )
    config_file = tmp_path / "config.yaml"
    config_file.write_text("litellm_settings: {}\n")
    return tmp_path


class TestFileModuleResolution:
    """Existing file-module behavior is preserved."""

    def test_loads_file_module_and_retrieves_instance(self, file_module_dir):
        config_path = str(file_module_dir / "config.yaml")
        instance = get_instance_fn(
            value="my_callbacks.my_handler_instance",
            config_file_path=config_path,
        )
        assert instance.loaded is True

    def test_file_module_takes_precedence_over_package(self, tmp_path):
        """When both <mod>.py and <mod>/__init__.py exist, file wins."""
        # Create file module
        mod_file = tmp_path / "dual.py"
        mod_file.write_text("source = 'file'\n")
        # Create package with same name
        pkg_dir = tmp_path / "dual"
        pkg_dir.mkdir()
        (pkg_dir / "__init__.py").write_text("source = 'package'\n")
        config_file = tmp_path / "config.yaml"
        config_file.write_text("")

        instance = get_instance_fn(
            value="dual.source",
            config_file_path=str(config_file),
        )
        assert instance == "file"


class TestPackageModuleResolution:
    """Package-style modules resolve with correct submodule_search_locations."""

    def test_loads_package_with_relative_import(self, package_module_dir):
        config_path = str(package_module_dir / "config.yaml")
        instance = get_instance_fn(
            value="my_package.package_handler_instance",
            config_file_path=config_path,
        )
        assert instance.value == 42

    def test_package_module_has_correct_name(self, package_module_dir):
        config_path = str(package_module_dir / "config.yaml")
        instance = get_instance_fn(
            value="my_package.package_handler_instance",
            config_file_path=config_path,
        )
        # The module should be named correctly for relative imports to work
        mod = type(instance).__module__
        assert mod == "my_package"


class TestMissingModule:
    """Missing modules raise bounded ImportError."""

    def test_missing_module_raises_import_error(self, tmp_path):
        config_file = tmp_path / "config.yaml"
        config_file.write_text("")
        with pytest.raises(ImportError, match="Could not import some_instance from nonexistent_module"):
            get_instance_fn(
                value="nonexistent_module.some_instance",
                config_file_path=str(config_file),
            )

    def test_missing_instance_in_valid_module_raises(self, file_module_dir):
        config_path = str(file_module_dir / "config.yaml")
        with pytest.raises((ImportError, AttributeError)):
            get_instance_fn(
                value="my_callbacks.does_not_exist",
                config_file_path=config_path,
            )


class TestRealAawmAgentIdentity:
    """The real callback config path resolves without remote calls."""

    def test_resolves_aawm_agent_identity_instance(self):
        """litellm.integrations.aawm_agent_identity.aawm_agent_identity_instance
        resolves using the repo root as config_file_path directory."""
        repo_root = os.path.dirname(
            os.path.dirname(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            )
        )
        # Use a config path whose dirname is the repo root so the module
        # path resolves to litellm/integrations/aawm_agent_identity/__init__.py
        config_path = os.path.join(repo_root, "litellm-config.yaml")

        instance = get_instance_fn(
            value="litellm.integrations.aawm_agent_identity.aawm_agent_identity_instance",
            config_file_path=config_path,
        )
        assert instance is not None
        # Verify it's the expected class
        assert type(instance).__name__ == "AawmAgentIdentity"


class TestSysModulesRestoreOnFailure:
    """On package exec failure, sys.modules is restored exactly."""

    def test_prior_module_restored_and_partial_children_removed(self, tmp_path):
        """(a) Prior module restored and partial/new children removed after
        package exec failure."""
        pkg_dir = tmp_path / "fail_pkg"
        pkg_dir.mkdir()
        # __init__.py that raises during exec
        (pkg_dir / "__init__.py").write_text("raise RuntimeError('boom')\n")
        (pkg_dir / "child.py").write_text("X = 1\n")
        config_file = tmp_path / "config.yaml"
        config_file.write_text("")

        import types

        # Seed a prior module entry for the package name
        sentinel = types.ModuleType("fail_pkg")
        sentinel.__marker__ = "original"  # type: ignore[attr-defined]
        sys.modules["fail_pkg"] = sentinel

        try:
            with pytest.raises(RuntimeError, match="boom"):
                get_instance_fn(
                    value="fail_pkg.some_instance",
                    config_file_path=str(config_file),
                )
            # The original sentinel must be restored
            assert sys.modules.get("fail_pkg") is sentinel
            assert sys.modules["fail_pkg"].__marker__ == "original"  # type: ignore[attr-defined]
            # No partial children should remain
            assert "fail_pkg.child" not in sys.modules
        finally:
            sys.modules.pop("fail_pkg", None)


class TestChildReloadOnFileChange:
    """On successful reload, child helpers reflect current file content."""

    def test_child_helper_change_observed_on_second_load(self, tmp_path):
        """(b) Child helper file change is observed on second load."""
        pkg_dir = tmp_path / "reload_pkg"
        pkg_dir.mkdir()
        (pkg_dir / "helper.py").write_text("VALUE = 'first'\n")
        (pkg_dir / "__init__.py").write_text(
            textwrap.dedent("""\
                from .helper import VALUE
                result = VALUE
            """)
        )
        config_file = tmp_path / "config.yaml"
        config_file.write_text("")
        config_path = str(config_file)

        # First load
        val1 = get_instance_fn(
            value="reload_pkg.result", config_file_path=config_path
        )
        assert val1 == "first"

        # Modify child helper
        (pkg_dir / "helper.py").write_text("VALUE = 'second'\n")

        # Second load must see the updated file
        val2 = get_instance_fn(
            value="reload_pkg.result", config_file_path=config_path
        )
        assert val2 == "second"

        # Cleanup
        for key in list(sys.modules):
            if key == "reload_pkg" or key.startswith("reload_pkg."):
                del sys.modules[key]


class TestUnrelatedModuleUntouched:
    """Unrelated modules with similar prefixes are not disturbed."""

    def test_similarly_prefixed_module_untouched(self, tmp_path):
        """(c) Unrelated similarly prefixed module is untouched."""
        import types

        pkg_dir = tmp_path / "mypkg"
        pkg_dir.mkdir()
        (pkg_dir / "__init__.py").write_text("val = 1\n")
        config_file = tmp_path / "config.yaml"
        config_file.write_text("")

        # Seed an unrelated module whose name starts with 'mypkg' but is NOT
        # a child (no dot separator)
        unrelated = types.ModuleType("mypkg_extra")
        unrelated.__tag__ = "unrelated"  # type: ignore[attr-defined]
        sys.modules["mypkg_extra"] = unrelated

        try:
            get_instance_fn(
                value="mypkg.val", config_file_path=str(config_file)
            )
            # Unrelated module must be untouched
            assert sys.modules.get("mypkg_extra") is unrelated
            assert sys.modules["mypkg_extra"].__tag__ == "unrelated"  # type: ignore[attr-defined]
        finally:
            sys.modules.pop("mypkg_extra", None)
            for key in list(sys.modules):
                if key == "mypkg" or key.startswith("mypkg."):
                    del sys.modules[key]


class TestSuccessfulRealCallbackPath:
    """Successful real callback path remains functional after changes."""

    def test_real_callback_resolves_and_sys_modules_consistent(self):
        """(d) Successful real callback path remains - verify sys.modules
        is left in a consistent state after loading the real package."""
        repo_root = os.path.dirname(
            os.path.dirname(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            )
        )
        config_path = os.path.join(repo_root, "litellm-config.yaml")
        mod_name = "litellm.integrations.aawm_agent_identity"

        instance = get_instance_fn(
            value=f"{mod_name}.aawm_agent_identity_instance",
            config_file_path=config_path,
        )
        assert instance is not None
        assert type(instance).__name__ == "AawmAgentIdentity"
        # The package must be registered in sys.modules after success
        assert mod_name in sys.modules
