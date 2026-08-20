"""Packaging contract for stdlib Cursor Agent usage helpers in the sidecar image."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DOCKERFILE_PATH = (
    _REPO_ROOT / "docker" / "Dockerfile.provider_status_observations"
)
_LOOP_SCRIPT = "run_provider_status_observations_loop.py"

_SCRIPT_FILES = (
    "record_provider_status_observations.py",
    "grok_oidc_refresh.py",
    "codex_oauth_refresh.py",
    "xai_oauth_refresh.py",
    "kimi_oauth_refresh.py",
    _LOOP_SCRIPT,
)

_SECRET_MANAGER_FILES = (
    "credential_error_sanitizer.py",
    "credential_file_lock.py",
    "credential_file_metadata.py",
    "credential_file_write.py",
    "grok_oidc_auth_path.py",
    "codex_oauth_inventory.py",
    "kimi_native_contract.py",
    "grok_native_version_contract.py",
)

_CURSOR_FILES = (
    "constants.py",
    "dashboard.py",
    "usage.py",
)


def _dockerfile() -> str:
    return _DOCKERFILE_PATH.read_text(encoding="utf-8")


def _build_sidecar_layout(tmp_path: Path) -> Path:
    image_root = tmp_path / "app"
    scripts_dir = image_root / "scripts"
    secret_dir = image_root / "litellm" / "secret_managers"
    cursor_dir = image_root / "litellm" / "llms" / "cursor_agent"
    scripts_dir.mkdir(parents=True)
    secret_dir.mkdir(parents=True)
    cursor_dir.mkdir(parents=True)

    for name in _SCRIPT_FILES:
        shutil.copy2(_REPO_ROOT / "scripts" / name, scripts_dir / name)
    for name in _SECRET_MANAGER_FILES:
        shutil.copy2(
            _REPO_ROOT / "litellm" / "secret_managers" / name,
            secret_dir / name,
        )
    for name in _CURSOR_FILES:
        shutil.copy2(
            _REPO_ROOT / "litellm" / "llms" / "cursor_agent" / name,
            cursor_dir / name,
        )

    for init_path in (
        scripts_dir / "__init__.py",
        image_root / "litellm" / "__init__.py",
        secret_dir / "__init__.py",
        image_root / "litellm" / "llms" / "__init__.py",
        cursor_dir / "__init__.py",
    ):
        init_path.write_text("", encoding="utf-8")
    return image_root


def test_dockerfile_copies_stdlib_cursor_usage_helpers() -> None:
    dockerfile = _dockerfile()

    for name in _CURSOR_FILES:
        assert (
            f"COPY litellm/llms/cursor_agent/{name} "
            f"/app/litellm/llms/cursor_agent/{name}"
        ) in dockerfile


def test_dockerfile_touches_llms_package_inits() -> None:
    dockerfile = _dockerfile()

    assert "/app/litellm/llms/__init__.py" in dockerfile
    assert "/app/litellm/llms/cursor_agent/__init__.py" in dockerfile


def test_dockerfile_does_not_ship_full_cursor_agent_provider() -> None:
    dockerfile = _dockerfile()

    assert "common_utils.py" not in dockerfile
    assert "chat/transformation.py" not in dockerfile
    assert "COPY litellm /app/litellm" not in dockerfile
    assert "httpx" not in dockerfile
    assert not any(
        "litellm" in line for line in dockerfile.splitlines() if "pip install" in line
    )


def test_loop_does_not_import_cursor_agent_common_utils() -> None:
    loop_source = (
        _REPO_ROOT / "scripts" / _LOOP_SCRIPT
    ).read_text(encoding="utf-8")

    assert "litellm.llms.cursor_agent.common_utils" not in loop_source
    assert "from litellm.llms.cursor_agent.constants import" in loop_source
    assert "from litellm.llms.cursor_agent.dashboard import" in loop_source
    assert "from litellm.llms.cursor_agent.usage import" in loop_source


def test_gold_lock_imports_loop_from_copied_sidecar_layout(tmp_path: Path) -> None:
    image_root = _build_sidecar_layout(tmp_path)
    loop_path = image_root / "scripts" / _LOOP_SCRIPT
    helper = """
import importlib.util
import sys
from pathlib import Path

loop_path = Path(sys.argv[1])
spec = importlib.util.spec_from_file_location(
    "run_provider_status_observations_loop",
    loop_path,
)
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
assert "litellm.llms.cursor_agent.common_utils" not in sys.modules
assert "httpx" not in sys.modules
from litellm.llms.cursor_agent.constants import CURSOR_AGENT_DASHBOARD_HOST
from litellm.llms.cursor_agent.dashboard import build_dashboard_headers
assert CURSOR_AGENT_DASHBOARD_HOST == "https://api2.cursor.sh"
assert build_dashboard_headers("token", request_id="req-1")["authorization"] == (
    "Bearer token"
)
"""
    env = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}
    env["PYTHONPATH"] = str(image_root)
    result = subprocess.run(
        [sys.executable, "-c", helper, str(loop_path)],
        cwd=str(image_root),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr + result.stdout


def test_gold_lock_imports_loop_with_xai_reset_poll_enabled_and_missing_auth(
    tmp_path: Path,
) -> None:
    image_root = _build_sidecar_layout(tmp_path)
    loop_path = image_root / "scripts" / _LOOP_SCRIPT
    helper = """
import importlib.util
import os
import sys
from pathlib import Path

loop_path = Path(sys.argv[1])
os.environ["AAWM_XAI_RESET_POLL_ENABLED"] = "1"
os.environ.pop("AAWM_GROK_OIDC_AUTH_FILE", None)
spec = importlib.util.spec_from_file_location(
    "run_provider_status_observations_loop",
    loop_path,
)
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
assert module.xai_reset_poll_enabled() is True
assert "RedeemReset" not in Path(loop_path).read_text(encoding="utf-8")
assert "httpx" not in sys.modules
"""
    env = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}
    env["PYTHONPATH"] = str(image_root)
    env["AAWM_XAI_RESET_POLL_ENABLED"] = "1"
    env.pop("AAWM_GROK_OIDC_AUTH_FILE", None)
    result = subprocess.run(
        [sys.executable, "-c", helper, str(loop_path)],
        cwd=str(image_root),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr + result.stdout
