"""Packaging contract for the stdlib conversation_init module in the sidecar image."""

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
    "nous_oauth_refresh.py",
    "cursor_agent_auth_refresh.py",
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
    "connect.py",
    "usage.py",
)

_CHATGPT_FILES = ("conversation_init.py",)


def _dockerfile() -> str:
    return _DOCKERFILE_PATH.read_text(encoding="utf-8")


def _build_sidecar_layout(tmp_path: Path) -> Path:
    image_root = tmp_path / "app"
    scripts_dir = image_root / "scripts"
    secret_dir = image_root / "litellm" / "secret_managers"
    cursor_dir = image_root / "litellm" / "llms" / "cursor_agent"
    chatgpt_dir = image_root / "litellm" / "llms" / "chatgpt"
    scripts_dir.mkdir(parents=True)
    secret_dir.mkdir(parents=True)
    cursor_dir.mkdir(parents=True)
    chatgpt_dir.mkdir(parents=True)

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
    for name in _CHATGPT_FILES:
        shutil.copy2(
            _REPO_ROOT / "litellm" / "llms" / "chatgpt" / name,
            chatgpt_dir / name,
        )

    for init_path in (
        scripts_dir / "__init__.py",
        image_root / "litellm" / "__init__.py",
        secret_dir / "__init__.py",
        image_root / "litellm" / "llms" / "__init__.py",
        cursor_dir / "__init__.py",
        chatgpt_dir / "__init__.py",
    ):
        init_path.write_text("", encoding="utf-8")
    return image_root


def test_dockerfile_copies_conversation_init() -> None:
    dockerfile = _dockerfile()

    assert (
        "COPY litellm/llms/chatgpt/conversation_init.py "
        "/app/litellm/llms/chatgpt/conversation_init.py"
    ) in dockerfile


def test_dockerfile_touches_chatgpt_package_init() -> None:
    dockerfile = _dockerfile()

    assert "/app/litellm/llms/chatgpt/__init__.py" in dockerfile


def test_dockerfile_does_not_ship_chatgpt_authenticator() -> None:
    dockerfile = _dockerfile()

    assert "authenticator.py" not in dockerfile
    assert "common_utils.py" not in dockerfile
    assert "httpx" not in dockerfile


def test_loop_imports_conversation_init_from_stdlib() -> None:
    loop_source = (
        _REPO_ROOT / "scripts" / _LOOP_SCRIPT
    ).read_text(encoding="utf-8")

    assert "from litellm.llms.chatgpt.conversation_init import" in loop_source


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
assert "litellm.llms.chatgpt.common_utils" not in sys.modules
assert "httpx" not in sys.modules
from litellm.llms.chatgpt.conversation_init import (
    CHATGPT_CONVERSATION_INIT_DEFAULT_URL,
    conversation_init_request_contract,
)
contract = conversation_init_request_contract()
assert contract["method"] == "POST"
assert contract["body_omitted"] is True
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
