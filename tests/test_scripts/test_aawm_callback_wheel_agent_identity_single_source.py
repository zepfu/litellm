"""RR-003: callback wheel must not maintain a full agent_identity source copy.

Canonical implementation lives at
``litellm/integrations/aawm_agent_identity.py``.

``.wheel-build/aawm_litellm_callbacks/agent_identity.py`` is a thin checkout
loader only. Hatch force-includes the canonical module into the published
``aawm-litellm-callbacks`` wheel so standalone installs get the full module
without dual-maintained source trees.
"""

from __future__ import annotations

import ast
import hashlib
import importlib
import os
import subprocess
import sys
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
WHEEL_BUILD = REPO_ROOT / ".wheel-build"
CANONICAL = REPO_ROOT / "litellm" / "integrations" / "aawm_agent_identity.py"
CHECKOUT_LOADER = WHEEL_BUILD / "aawm_litellm_callbacks" / "agent_identity.py"
PYPROJECT = WHEEL_BUILD / "pyproject.toml"

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
    assert CANONICAL.is_file()
    assert CHECKOUT_LOADER.is_file()

    loader_text = CHECKOUT_LOADER.read_text(encoding="utf-8")
    canonical_text = CANONICAL.read_text(encoding="utf-8")

    assert _line_count(CHECKOUT_LOADER) <= _MAX_LOADER_LINES
    assert len(loader_text) < len(canonical_text) // 10
    assert _sha256(CHECKOUT_LOADER) != _sha256(CANONICAL)

    # Must not reintroduce a second maintained implementation body.
    assert "class AawmAgentIdentity" not in loader_text
    assert "Checkout loader for aawm_litellm_callbacks" in loader_text
    assert "litellm.integrations.aawm_agent_identity" in loader_text
    assert "force-includes the canonical file" in loader_text


def test_wheel_build_force_includes_canonical_agent_identity() -> None:
    text = PYPROJECT.read_text(encoding="utf-8")
    assert 'build-backend = "hatchling.build"' in text
    assert (
        '"../litellm/integrations/aawm_agent_identity.py" = '
        '"aawm_litellm_callbacks/agent_identity.py"'
    ) in text
    assert "[tool.hatch.build.targets.wheel.force-include]" in text


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
        packaged = archive.read("aawm_litellm_callbacks/agent_identity.py")

    assert "aawm_litellm_callbacks/agent_identity.py" in member_names
    assert "aawm_litellm_callbacks/aawm_agent_quality_rules.py" in member_names
    assert "aawm_litellm_callbacks/aawm_agent_quality_rules.json" in member_names
    assert "aawm_litellm_callbacks/__init__.py" in member_names

    canonical_bytes = CANONICAL.read_bytes()
    assert (
        hashlib.sha256(packaged).hexdigest()
        == hashlib.sha256(canonical_bytes).hexdigest()
    )
    assert b"Checkout loader for aawm_litellm_callbacks" not in packaged
    assert b"class AawmAgentIdentity" in packaged

    # Import the packaged module from an isolated extract. Prefer the extract
    # for aawm_litellm_callbacks while still resolving the declared litellm
    # dependency from the repo checkout (editable install / PYTHONPATH layout).
    install_dir = tmp_path / "site"
    install_dir.mkdir()
    with zipfile.ZipFile(wheel_path) as archive:
        for member in archive.namelist():
            if member.startswith("aawm_litellm_callbacks/"):
                archive.extract(member, path=install_dir)

    packaged_module_path = install_dir / "aawm_litellm_callbacks" / "agent_identity.py"
    assert packaged_module_path.is_file()
    assert packaged_module_path.read_bytes() == canonical_bytes

    # Parse without requiring a full runtime import for structural proof.
    tree = ast.parse(packaged_module_path.read_text(encoding="utf-8"))
    class_names = {node.name for node in tree.body if isinstance(node, ast.ClassDef)}
    assert "AawmAgentIdentity" in class_names

    probe = tmp_path / "import_packaged_helpers.py"
    probe.write_text(
        "\n".join(
            [
                "from __future__ import annotations",
                "",
                "import importlib",
                "import sys",
                f"repo_root = {str(REPO_ROOT)!r}",
                f"install_dir = {str(install_dir)!r}",
                # Drop empty/cwd path entries, then prefer wheel extract over",
                # checkout paths so we import the packaged module body.",
                "cleaned = [p for p in sys.path if p not in ('', repo_root, '.')]",
                "sys.path = [install_dir, repo_root] + cleaned",
                "module = importlib.import_module(",
                "    'aawm_litellm_callbacks.agent_identity'",
                ")",
                "assert module.__file__ is not None",
                "assert module.__file__.startswith(install_dir), module.__file__",
                f"helpers = {list(_REQUIRED_HELPERS)!r}",
                "for name in helpers:",
                "    assert hasattr(module, name), name",
                "assert callable(module.AawmAgentIdentity)",
                "assert module.aawm_agent_identity_instance is not None",
                "print('ok')",
                "",
            ]
        ),
        encoding="utf-8",
    )
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    run = subprocess.run(
        [sys.executable, str(probe)],
        check=False,
        capture_output=True,
        text=True,
        cwd=str(tmp_path),
        env=env,
    )
    assert (
        run.returncode == 0
    ), f"packaged import failed:\nstdout:\n{run.stdout}\nstderr:\n{run.stderr}"
    assert run.stdout.strip() == "ok"


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
