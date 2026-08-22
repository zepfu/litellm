"""Focused compose ownership coverage for the Hermes Nous Portal OAuth credential."""

from __future__ import annotations

import re
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
_COMPOSE_PATH = _REPO_ROOT / "docker-compose.dev.yml"
_PROD_COMPOSE_PATH = _REPO_ROOT / "docker-compose.yml"
_PROVIDER_STATUS_DOCKERFILE_PATH = (
    _REPO_ROOT / "docker" / "Dockerfile.provider_status_observations"
)
_HERMES_DIR = "/home/zepfu/.hermes"
_HERMES_AUTH_FILE = "/home/zepfu/.hermes/auth.json"
_HERMES_LOCK_FILE = "/home/zepfu/.hermes/auth.lock"


def _service_block(compose: str, service_name: str) -> str:
    marker = f"  {service_name}:\n"
    start = compose.index(marker)
    remainder = compose[start + len(marker) :]
    next_service = re.search(r"(?m)^  [a-zA-Z0-9_-]+:\n", remainder)
    return remainder[: next_service.start()] if next_service else remainder


def _sibling_oauth_families_wired(compose: str) -> bool:
    return (
        "AAWM_XAI_OAUTH_REFRESH_ENABLED" in compose
        or "AAWM_KIMI_OAUTH_REFRESH_ENABLED" in compose
        or "AAWM_GROK_OIDC_REFRESH_ENABLED" in compose
    )


def test_litellm_dev_hermes_directory_mount_is_read_only() -> None:
    compose = _COMPOSE_PATH.read_text(encoding="utf-8")
    litellm_dev = _service_block(compose, "litellm-dev")

    assert f"- {_HERMES_DIR}:{_HERMES_DIR}:ro" in litellm_dev
    assert f"- {_HERMES_AUTH_FILE}:{_HERMES_AUTH_FILE}" not in litellm_dev
    assert f"- {_HERMES_AUTH_FILE}:{_HERMES_AUTH_FILE}:ro" not in litellm_dev
    assert f"- {_HERMES_DIR}:{_HERMES_DIR}\n" not in litellm_dev
    assert (
        "- LITELLM_NOUS_OAUTH_AUTH_FILE="
        "${LITELLM_NOUS_OAUTH_AUTH_FILE:-"
        "/home/zepfu/.hermes/auth.json}"
    ) in litellm_dev


def test_provider_status_has_writable_hermes_directory_and_nous_env() -> None:
    compose = _COMPOSE_PATH.read_text(encoding="utf-8")
    sidecar = _service_block(compose, "provider-status-observations")

    assert f"- {_HERMES_DIR}:{_HERMES_DIR}\n" in sidecar
    assert f"- {_HERMES_DIR}:{_HERMES_DIR}:ro" not in sidecar
    assert f"- {_HERMES_AUTH_FILE}:{_HERMES_AUTH_FILE}" not in sidecar
    assert ("- AAWM_NOUS_OAUTH_REFRESH_ENABLED=" "${AAWM_NOUS_OAUTH_REFRESH_ENABLED:-1}") in sidecar
    # Compose-source default is enabled; argparse default remains 0.
    assert (
        "- AAWM_NOUS_OAUTH_AUTH_FILE="
        "${AAWM_NOUS_OAUTH_AUTH_FILE:-"
        "/home/zepfu/.hermes/auth.json}"
    ) in sidecar
    assert (
        "- AAWM_NOUS_OAUTH_LOCK_FILE="
        "${AAWM_NOUS_OAUTH_LOCK_FILE:-"
        "/home/zepfu/.hermes/auth.lock}"
    ) in sidecar
    assert _HERMES_LOCK_FILE in sidecar
    assert ("- AAWM_NOUS_OAUTH_FORCE_REFRESH=" "${AAWM_NOUS_OAUTH_FORCE_REFRESH:-0}") in sidecar
    for name, default in (
        ("AAWM_NOUS_OAUTH_AUTH_FILE_UID", "1000"),
        ("AAWM_NOUS_OAUTH_AUTH_FILE_GID", "1000"),
        ("AAWM_NOUS_OAUTH_AUTH_FILE_MODE", "0o600"),
        ("AAWM_NOUS_OAUTH_REFRESH_INTERVAL_SECONDS", "300"),
        ("AAWM_NOUS_OAUTH_REFRESH_BUFFER_SECONDS", "900"),
        ("AAWM_NOUS_OAUTH_HTTP_TIMEOUT_SECONDS", "30"),
    ):
        assert f"- {name}=${{{name}:-{default}}}" in sidecar


def test_provider_status_image_packages_nous_oauth_refresh_script() -> None:
    dockerfile = _PROVIDER_STATUS_DOCKERFILE_PATH.read_text(encoding="utf-8")

    assert ("COPY scripts/nous_oauth_refresh.py " "/app/scripts/nous_oauth_refresh.py") in dockerfile
    assert "COPY litellm /app/litellm" not in dockerfile
    for helper_name in (
        "credential_error_sanitizer.py",
        "credential_file_lock.py",
        "credential_file_metadata.py",
        "credential_file_write.py",
    ):
        assert (
            f"COPY litellm/secret_managers/{helper_name} "
            f"/app/litellm/secret_managers/{helper_name}"
        ) in dockerfile


def test_prod_compose_does_not_invent_nous_hermes_wiring() -> None:
    prod = _PROD_COMPOSE_PATH.read_text(encoding="utf-8")
    if _sibling_oauth_families_wired(prod):
        assert "/home/zepfu/.hermes:/home/zepfu/.hermes" in prod
        assert "AAWM_NOUS_OAUTH_REFRESH_ENABLED" in prod
        return
    assert "AAWM_NOUS_OAUTH" not in prod
    assert "/home/zepfu/.hermes" not in prod


def test_dev_compose_has_no_litellm_alpha_service() -> None:
    compose = _COMPOSE_PATH.read_text(encoding="utf-8")
    assert "\n  litellm-alpha:\n" not in compose
    assert "litellm-alpha:" not in _service_block(compose, "litellm-dev")
    assert "litellm-alpha:" not in _service_block(
        compose, "provider-status-observations"
    )
