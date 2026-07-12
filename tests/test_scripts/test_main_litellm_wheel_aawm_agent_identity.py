"""Regression: main LiteLLM wheel must ship aawm_agent_identity helpers."""

from __future__ import annotations

import os
import subprocess
import sys
import zipfile
from pathlib import Path

from poetry.core.factory import Factory
from poetry.core.masonry.builders.wheel import WheelBuilder

REPO_ROOT = Path(__file__).resolve().parents[2]
AGENT_IDENTITY_MEMBER = "litellm/integrations/aawm_agent_identity.py"
EXCLUDED_MEMBERS = (
    "litellm/integrations/aawm_payload_capture.py",
    "litellm/proxy/pass_through_endpoints/aawm_claude_control_plane.py",
)
REQUIRED_HELPERS = (
    "_build_alias_routing_audit_only_record",
    "_enqueue_session_history_record",
    "_spool_session_history_records",
)


def _install_wheel_to_target(*, wheel_path: Path, install_dir: Path) -> None:
    """Install a pure wheel into ``install_dir`` without network or pip.

    Prefer ``python -m pip`` when available. Otherwise extract the wheel zip
    into the target directory so offline/venv-without-pip environments still
    prove packaging contents and importability.
    """
    pip_probe = subprocess.run(
        [sys.executable, "-m", "pip", "--version"],
        check=False,
        capture_output=True,
        text=True,
    )
    if pip_probe.returncode == 0:
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--no-index",
                "--no-deps",
                "--target",
                str(install_dir),
                str(wheel_path),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode == 0:
            return

    with zipfile.ZipFile(wheel_path) as archive:
        for member in archive.namelist():
            # Skip dist-info / data directories for a minimal importable layout.
            if member.startswith("litellm/") or member == "litellm":
                archive.extract(member, path=install_dir)


def test_main_litellm_wheel_includes_aawm_agent_identity_helpers(tmp_path: Path) -> None:
    poetry = Factory().create_poetry(REPO_ROOT)
    wheel_dir = tmp_path / "dist"
    wheel_path = WheelBuilder(poetry).build(target_dir=wheel_dir)

    assert wheel_path.is_file()
    assert wheel_path.suffix == ".whl"

    with zipfile.ZipFile(wheel_path) as archive:
        member_names = set(archive.namelist())

    assert AGENT_IDENTITY_MEMBER in member_names
    for excluded_member in EXCLUDED_MEMBERS:
        assert excluded_member not in member_names

    install_dir = tmp_path / "site"
    install_dir.mkdir()
    _install_wheel_to_target(wheel_path=wheel_path, install_dir=install_dir)

    installed_module = (
        install_dir / "litellm" / "integrations" / "aawm_agent_identity.py"
    )
    assert installed_module.is_file()

    # Import from the isolated install target, not the repo checkout.
    probe = tmp_path / "import_helpers.py"
    probe.write_text(
        "\n".join(
            [
                "from __future__ import annotations",
                "",
                "import importlib",
                "import sys",
                f"repo_root = {str(REPO_ROOT)!r}",
                "sys.path = [p for p in sys.path if p not in ('', repo_root)]",
                f"sys.path.insert(0, {str(install_dir)!r})",
                "module = importlib.import_module(",
                "    'litellm.integrations.aawm_agent_identity'",
                ")",
                f"helpers = {REQUIRED_HELPERS!r}",
                "for name in helpers:",
                "    value = getattr(module, name)",
                "    assert callable(value), name",
                "print('ok')",
                "",
            ]
        ),
        encoding="utf-8",
    )
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    completed = subprocess.run(
        [sys.executable, str(probe)],
        check=True,
        capture_output=True,
        text=True,
        cwd=str(tmp_path),
        env=env,
    )
    assert completed.stdout.strip() == "ok"
