"""Focused compose ownership coverage for the shared Kimi Code CLI credential."""

from __future__ import annotations

import re
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[3]
_COMPOSE_PATH = _REPO_ROOT / "docker-compose.dev.yml"
_PROVIDER_STATUS_DOCKERFILE_PATH = _REPO_ROOT / "docker" / "Dockerfile.provider_status_observations"
_KIMI_AUTH_FILE = "/home/zepfu/.kimi-code/credentials/kimi-code.json"
_KIMI_CREDENTIAL_DIR = "/home/zepfu/.kimi-code/credentials"
_KIMI_OAUTH_DIR = "/home/zepfu/.kimi-code/oauth"
_KIMI_LOCK_SENTINEL = "/home/zepfu/.kimi-code/oauth/kimi-code"
_DOC_PATH = _REPO_ROOT / "docs" / "aawm-oauth-credential-maintenance.md"


def _service_block(compose: str, service_name: str) -> str:
    marker = f"  {service_name}:\n"
    start = compose.index(marker)
    remainder = compose[start + len(marker) :]
    next_service = re.search(r"(?m)^  [a-zA-Z0-9_-]+:\n", remainder)
    return remainder[: next_service.start()] if next_service else remainder


def test_litellm_dev_kimi_cli_credential_directory_mount_is_read_only() -> None:
    compose = _COMPOSE_PATH.read_text(encoding="utf-8")
    litellm_dev = _service_block(compose, "litellm-dev")

    assert f"- {_KIMI_CREDENTIAL_DIR}:{_KIMI_CREDENTIAL_DIR}:ro" in litellm_dev
    assert f"- {_KIMI_AUTH_FILE}:{_KIMI_AUTH_FILE}:ro" not in litellm_dev
    assert (
        "- LITELLM_KIMI_OAUTH_AUTH_FILE="
        "${LITELLM_KIMI_OAUTH_AUTH_FILE:-"
        "/home/zepfu/.kimi-code/credentials/kimi-code.json}"
    ) in litellm_dev


def test_litellm_dev_kimi_gateway_port_is_published_only_on_loopback() -> None:
    compose = _COMPOSE_PATH.read_text(encoding="utf-8")
    litellm_dev = _service_block(compose, "litellm-dev")

    assert '- "127.0.0.1:4001:4001"' in litellm_dev
    assert '- "4001:4001"' not in litellm_dev


def test_provider_status_has_only_required_shared_kimi_cli_write_mounts() -> None:
    compose = _COMPOSE_PATH.read_text(encoding="utf-8")
    sidecar = _service_block(compose, "provider-status-observations")

    assert f"- {_KIMI_CREDENTIAL_DIR}:{_KIMI_CREDENTIAL_DIR}\n" in sidecar
    assert f"- {_KIMI_OAUTH_DIR}:{_KIMI_OAUTH_DIR}\n" in sidecar
    assert f"- {_KIMI_CREDENTIAL_DIR}:{_KIMI_CREDENTIAL_DIR}:ro" not in sidecar
    assert f"- {_KIMI_OAUTH_DIR}:{_KIMI_OAUTH_DIR}:ro" not in sidecar
    assert "/home/zepfu/.kimi-code:/home/zepfu/.kimi-code" not in sidecar
    assert ("- AAWM_KIMI_OAUTH_REFRESH_ENABLED=" "${AAWM_KIMI_OAUTH_REFRESH_ENABLED:-1}") in sidecar
    assert (
        "- AAWM_KIMI_OAUTH_AUTH_FILE="
        "${AAWM_KIMI_OAUTH_AUTH_FILE:-"
        "/home/zepfu/.kimi-code/credentials/kimi-code.json}"
    ) in sidecar
    assert (
        "- AAWM_KIMI_OAUTH_LOCK_FILE=" "${AAWM_KIMI_OAUTH_LOCK_FILE:-" "/home/zepfu/.kimi-code/oauth/kimi-code}"
    ) in sidecar
    assert _KIMI_LOCK_SENTINEL in sidecar
    assert ("- AAWM_KIMI_OAUTH_FORCE_REFRESH=" "${AAWM_KIMI_OAUTH_FORCE_REFRESH:-0}") in sidecar
    for name, default in (
        ("AAWM_KIMI_OAUTH_AUTH_FILE_UID", "1000"),
        ("AAWM_KIMI_OAUTH_AUTH_FILE_GID", "1000"),
        ("AAWM_KIMI_OAUTH_AUTH_FILE_MODE", "0o600"),
        ("AAWM_KIMI_OAUTH_REFRESH_INTERVAL_SECONDS", "300"),
        ("AAWM_KIMI_OAUTH_HTTP_TIMEOUT_SECONDS", "30"),
    ):
        assert f"- {name}=${{{name}:-{default}}}" in sidecar


def test_provider_status_image_packages_kimi_oauth_refresh_dependencies() -> None:
    dockerfile = _PROVIDER_STATUS_DOCKERFILE_PATH.read_text(encoding="utf-8")

    assert "    PYTHONPATH=/app \\\n" in dockerfile
    assert ("COPY scripts/kimi_oauth_refresh.py " "/app/scripts/kimi_oauth_refresh.py") in dockerfile
    for helper_name in (
        "credential_error_sanitizer.py",
        "credential_file_lock.py",
        "credential_file_metadata.py",
        "credential_file_write.py",
    ):
        assert (
            f"COPY litellm/secret_managers/{helper_name} " f"/app/litellm/secret_managers/{helper_name}"
        ) in dockerfile

    assert "/app/litellm/__init__.py" in dockerfile
    assert "/app/litellm/secret_managers/__init__.py" in dockerfile
    assert "COPY litellm /app/litellm" not in dockerfile
    assert "python-dotenv" not in dockerfile
    assert 'pip install --no-cache-dir "litellm' not in dockerfile


def test_kimi_credential_docs_use_only_the_existing_cli_grant() -> None:
    docs = _DOC_PATH.read_text(encoding="utf-8")

    assert "~/.kimi-code/credentials/kimi-code.json" in docs
    assert "~/.kimi-code/oauth/kimi-code" in docs
    assert "proper-lockfile" in docs
    assert "kimi-code.lock" in docs
    assert "~/.litellm/kimi/kimi-code.json" not in docs
    assert "scripts/kimi_oauth_refresh.py --login" not in docs
    assert "device enrollment" not in docs
