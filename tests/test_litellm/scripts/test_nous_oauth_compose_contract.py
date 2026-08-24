"""Focused compose ownership coverage for the Hermes Nous Portal OAuth credential."""

from __future__ import annotations

import re
from pathlib import Path

from litellm.secret_managers.hermes_nous_auth import resolve_hermes_nous_auth_path

_REPO_ROOT = Path(__file__).resolve().parents[3]
_COMPOSE_PATH = _REPO_ROOT / "docker-compose.dev.yml"
_ALPHA_COMPOSE_PATH = _REPO_ROOT / "docker-compose.alpha.yml"
_PROD_COMPOSE_PATH = _REPO_ROOT / "docker-compose.yml"
_PROVIDER_STATUS_DOCKERFILE_PATH = (
    _REPO_ROOT / "docker" / "Dockerfile.provider_status_observations"
)
_HERMES_DIR = "/home/zepfu/.hermes"
_HERMES_AUTH_FILE = "/home/zepfu/.hermes/auth.json"
_HERMES_LOCK_FILE = "/home/zepfu/.hermes/auth.lock"
_CANONICAL_AUTH_FILE_EXPORT = (
    "- LITELLM_NOUS_OAUTH_AUTH_FILE="
    "${LITELLM_NOUS_OAUTH_AUTH_FILE:-"
    "/home/zepfu/.hermes/auth.json}"
)
_CANONICAL_AUTH_FILE_DEFAULT_RE = re.compile(
    r"LITELLM_NOUS_OAUTH_AUTH_FILE="
    r"\$\{LITELLM_NOUS_OAUTH_AUTH_FILE:-([^}]+)\}"
)


def _service_block(compose: str, service_name: str) -> str:
    marker = f"  {service_name}:\n"
    start = compose.index(marker)
    remainder = compose[start + len(marker) :]
    next_service = re.search(r"(?m)^  [a-zA-Z0-9_-]+:\n", remainder)
    return remainder[: next_service.start()] if next_service else remainder


def _exported_nous_oauth_auth_file_default(service_block: str) -> str:
    match = _CANONICAL_AUTH_FILE_DEFAULT_RE.search(service_block)
    assert match is not None
    return match.group(1)


def _assert_read_only_hermes_consumer(service_block: str) -> None:
    assert f"- {_HERMES_DIR}:{_HERMES_DIR}:ro" in service_block
    assert f"- {_HERMES_AUTH_FILE}:{_HERMES_AUTH_FILE}" not in service_block
    assert f"- {_HERMES_AUTH_FILE}:{_HERMES_AUTH_FILE}:ro" not in service_block
    assert f"- {_HERMES_DIR}:{_HERMES_DIR}\n" not in service_block
    assert _CANONICAL_AUTH_FILE_EXPORT in service_block


def _sibling_oauth_families_wired(compose: str) -> bool:
    return (
        "AAWM_XAI_OAUTH_REFRESH_ENABLED" in compose
        or "AAWM_KIMI_OAUTH_REFRESH_ENABLED" in compose
        or "AAWM_GROK_OIDC_REFRESH_ENABLED" in compose
    )


def test_litellm_dev_hermes_directory_mount_is_read_only() -> None:
    compose = _COMPOSE_PATH.read_text(encoding="utf-8")
    _assert_read_only_hermes_consumer(_service_block(compose, "litellm-dev"))


def test_litellm_alpha_hermes_directory_mount_is_read_only() -> None:
    compose = _ALPHA_COMPOSE_PATH.read_text(encoding="utf-8")
    _assert_read_only_hermes_consumer(_service_block(compose, "litellm-alpha"))


def test_compose_canonical_export_is_accepted_by_shipped_reader(monkeypatch) -> None:
    defaults = []
    for compose_path, service_name in (
        (_COMPOSE_PATH, "litellm-dev"),
        (_ALPHA_COMPOSE_PATH, "litellm-alpha"),
    ):
        service_block = _service_block(
            compose_path.read_text(encoding="utf-8"), service_name
        )
        defaults.append(_exported_nous_oauth_auth_file_default(service_block))

    assert defaults[0] == defaults[1]
    exported_default = defaults[0]
    assert exported_default == _HERMES_AUTH_FILE

    monkeypatch.setenv("LITELLM_NOUS_OAUTH_AUTH_FILE", exported_default)
    monkeypatch.delenv("LITELLM_HERMES_AUTH_FILE", raising=False)
    monkeypatch.delenv("AAWM_HERMES_AUTH_FILE", raising=False)
    assert resolve_hermes_nous_auth_path() == exported_default


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
