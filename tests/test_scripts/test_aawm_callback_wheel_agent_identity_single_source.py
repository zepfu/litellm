"""RR-003: callback wheel must not maintain a full agent_identity source copy.

Canonical implementation lives in
``litellm/integrations/aawm_agent_identity/``.

``.wheel-build/aawm_litellm_callbacks/agent_identity.py`` is a thin checkout
loader only. Hatch force-includes the canonical package into the published
``aawm-litellm-callbacks`` wheel so standalone installs get the full package
without dual-maintained source trees.
"""

from __future__ import annotations

import ast
import hashlib
import importlib
import subprocess
import sys
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
WHEEL_BUILD = REPO_ROOT / ".wheel-build"
CANONICAL_PACKAGE = (
    REPO_ROOT / "litellm" / "integrations" / "aawm_agent_identity"
)
CANONICAL_INIT = CANONICAL_PACKAGE / "__init__.py"
CHECKOUT_LOADER = WHEEL_BUILD / "aawm_litellm_callbacks" / "agent_identity.py"
PYPROJECT = WHEEL_BUILD / "pyproject.toml"
REQUIRED_PACKAGE_MODULES = (
    "__init__.py",
    "interfaces.py",
    "provider_cache.py",
)

# Loader must stay far smaller than the canonical god-file (~21k lines).
_MAX_LOADER_LINES = 80
_REQUIRED_HELPERS = (
    "AawmAgentIdentity",
    "aawm_agent_identity_instance",
    "_build_alias_routing_audit_only_record",
    "_enqueue_session_history_record",
    "_spool_session_history_records",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _line_count(path: Path) -> int:
    return path.read_text(encoding="utf-8").count("\n") + 1


def test_checkout_agent_identity_is_thin_loader_not_full_copy() -> None:
    assert CANONICAL_PACKAGE.is_dir()
    assert CANONICAL_INIT.is_file()
    assert CHECKOUT_LOADER.is_file()

    loader_text = CHECKOUT_LOADER.read_text(encoding="utf-8")
    canonical_text = CANONICAL_INIT.read_text(encoding="utf-8")

    assert _line_count(CHECKOUT_LOADER) <= _MAX_LOADER_LINES
    assert len(loader_text) < len(canonical_text) // 10
    assert _sha256(CHECKOUT_LOADER) != _sha256(CANONICAL_INIT)

    # Must not reintroduce a second maintained implementation body.
    assert "class AawmAgentIdentity" not in loader_text
    assert "Checkout loader for aawm_litellm_callbacks" in loader_text
    assert "litellm.integrations.aawm_agent_identity" in loader_text


def test_wheel_build_force_includes_canonical_agent_identity() -> None:
    text = PYPROJECT.read_text(encoding="utf-8")
    assert 'build-backend = "hatchling.build"' in text
    assert "[tool.hatch.build.targets.wheel.force-include]" in text
    assert "[tool.hatch.build.targets.sdist.force-include]" in text
    for module_name in REQUIRED_PACKAGE_MODULES:
        mapping = (
            f'"../litellm/integrations/aawm_agent_identity/{module_name}" = '
            f'"litellm/integrations/aawm_agent_identity/{module_name}"'
        )
        assert text.count(mapping) == 2


def test_built_callback_wheel_ships_canonical_agent_identity_not_loader(
    tmp_path: Path,
) -> None:
    wheel_dir = tmp_path / "dist"
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "build",
            "--wheel",
            "--outdir",
            str(wheel_dir),
            "--no-isolation",
            str(WHEEL_BUILD),
        ],
        check=False,
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )
    assert completed.returncode == 0, (
        "callback wheel build failed:\n"
        f"stdout:\n{completed.stdout}\n"
        f"stderr:\n{completed.stderr}"
    )

    wheels = list(wheel_dir.glob("aawm_litellm_callbacks-*-py3-none-any.whl"))
    assert len(wheels) == 1, wheels
    wheel_path = wheels[0]

    with zipfile.ZipFile(wheel_path) as archive:
        member_names = set(archive.namelist())
        packaged_loader = archive.read(
            "aawm_litellm_callbacks/agent_identity.py"
        )

    assert "aawm_litellm_callbacks/agent_identity.py" in member_names
    assert "aawm_litellm_callbacks/aawm_agent_quality_rules.py" in member_names
    assert "aawm_litellm_callbacks/aawm_agent_quality_rules.json" in member_names
    assert "aawm_litellm_callbacks/__init__.py" in member_names

    with zipfile.ZipFile(wheel_path) as archive:
        assert packaged_loader == CHECKOUT_LOADER.read_bytes()
        for module_name in REQUIRED_PACKAGE_MODULES:
            member = (
                "litellm/integrations/aawm_agent_identity/"
                f"{module_name}"
            )
            assert member in member_names
            assert archive.read(member) == (
                CANONICAL_PACKAGE / module_name
            ).read_bytes()

        package_init = archive.read(
            "litellm/integrations/aawm_agent_identity/__init__.py"
        )
    tree = ast.parse(package_init.decode("utf-8"))
    class_names = {
        node.name for node in tree.body if isinstance(node, ast.ClassDef)
    }
    assert "AawmAgentIdentity" in class_names


def test_checkout_loader_reexports_canonical_public_symbols() -> None:
    # Source-tree path: thin loader must re-export the canonical symbols used
    # by config registration and sibling dual-import probes.
    if str(WHEEL_BUILD) not in sys.path:
        sys.path.insert(0, str(WHEEL_BUILD))

    for key in list(sys.modules):
        if key == "aawm_litellm_callbacks" or key.startswith("aawm_litellm_callbacks."):
            del sys.modules[key]

    module = importlib.import_module("aawm_litellm_callbacks.agent_identity")
    canonical = importlib.import_module("litellm.integrations.aawm_agent_identity")

    assert module.AawmAgentIdentity is canonical.AawmAgentIdentity
    assert module.aawm_agent_identity_instance is canonical.aawm_agent_identity_instance
    assert (
        module._enqueue_session_history_record
        is canonical._enqueue_session_history_record
    )
    assert (
        module._spool_session_history_records
        is canonical._spool_session_history_records
    )
    assert callable(module._enqueue_session_history_record)
    assert callable(module._spool_session_history_records)
