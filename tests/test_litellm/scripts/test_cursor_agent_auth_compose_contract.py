"""Focused Compose ownership contract for the Cursor Agent auth credential."""

from __future__ import annotations

import re
from pathlib import Path

import yaml


_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEV_COMPOSE_PATH = _REPO_ROOT / "docker-compose.dev.yml"
_ALPHA_COMPOSE_PATH = _REPO_ROOT / "docker-compose.alpha.yml"
_PROVIDER_STATUS_DOCKERFILE_PATH = _REPO_ROOT / "docker" / "Dockerfile.provider_status_observations"
_CURSOR_AUTH_REFRESH_SCRIPT_PATH = _REPO_ROOT / "scripts" / "cursor_agent_auth_refresh.py"
_STATUS_DOC_PATH = _REPO_ROOT / "docs" / "aawm-provider-status-observations.md"
_CURSOR_DOC_PATH = _REPO_ROOT / "docs" / "my-website" / "docs" / "providers" / "cursor_agent.md"
_CURSOR_DIR = "/home/zepfu/.config/cursor"
_CURSOR_AUTH_FILE = f"{_CURSOR_DIR}/auth.json"
_CURSOR_LOCK_FILE = f"{_CURSOR_AUTH_FILE}.lock"
_LITELLM_PROXY_SERVICES = (
    ("docker-compose.dev.yml", "litellm-dev"),
    ("docker-compose.alpha.yml", "litellm-alpha"),
)
_AUTH_ENV_DEFAULTS = {
    "AAWM_CURSOR_AGENT_AUTH_REFRESH_ENABLED": "1",
    "AAWM_CURSOR_AGENT_AUTH_FILE": _CURSOR_AUTH_FILE,
    "AAWM_CURSOR_AGENT_AUTH_LOCK_FILE": _CURSOR_LOCK_FILE,
    "AAWM_CURSOR_AGENT_AUTH_FILE_UID": "1000",
    "AAWM_CURSOR_AGENT_AUTH_FILE_GID": "1000",
    "AAWM_CURSOR_AGENT_AUTH_FILE_MODE": "0o600",
    "AAWM_CURSOR_AGENT_AUTH_REFRESH_INTERVAL_SECONDS": "300",
    "AAWM_CURSOR_AGENT_AUTH_REFRESH_BUFFER_SECONDS": "900",
    "AAWM_CURSOR_AGENT_AUTH_FORCE_REFRESH": "0",
    "AAWM_CURSOR_AGENT_AUTH_HTTP_TIMEOUT_SECONDS": "30",
}


def _service_block(compose: str, service_name: str) -> str:
    marker = f"  {service_name}:\n"
    start = compose.index(marker)
    remainder = compose[start + len(marker) :]
    next_service = re.search(r"(?m)^  [a-zA-Z0-9_-]+:\n", remainder)
    return remainder[: next_service.start()] if next_service else remainder


def _assert_read_only_cursor_proxy_consumer(service_block: str) -> None:
    assert f"- {_CURSOR_DIR}:{_CURSOR_DIR}:ro\n" in service_block
    assert f"- {_CURSOR_AUTH_FILE}:{_CURSOR_AUTH_FILE}" not in service_block
    assert f"- {_CURSOR_DIR}:{_CURSOR_DIR}\n" not in service_block
    assert (
        f"- LITELLM_CURSOR_AGENT_AUTH_FILE=${{LITELLM_CURSOR_AGENT_AUTH_FILE:-{_CURSOR_AUTH_FILE}}}"
    ) in service_block


def test_every_litellm_proxy_consumes_cursor_directory_read_only() -> None:
    for compose_name, service_name in _LITELLM_PROXY_SERVICES:
        compose = (_REPO_ROOT / compose_name).read_text(encoding="utf-8")
        _assert_read_only_cursor_proxy_consumer(_service_block(compose, service_name))


def test_compose_cursor_consumer_exports_the_expected_auth_path() -> None:
    for compose_name, service_name in _LITELLM_PROXY_SERVICES:
        compose = (_REPO_ROOT / compose_name).read_text(encoding="utf-8")
        service = _service_block(compose, service_name)
        match = re.search(
            r"LITELLM_CURSOR_AGENT_AUTH_FILE="
            r"\$\{LITELLM_CURSOR_AGENT_AUTH_FILE:-([^}]+)\}",
            service,
        )
        assert match is not None
        assert match.group(1) == _CURSOR_AUTH_FILE


def test_provider_status_has_writable_cursor_directory_and_exact_auth_env() -> None:
    compose = _DEV_COMPOSE_PATH.read_text(encoding="utf-8")
    sidecar = _service_block(compose, "provider-status-observations")

    assert f"- {_CURSOR_DIR}:{_CURSOR_DIR}\n" in sidecar
    assert f"- {_CURSOR_DIR}:{_CURSOR_DIR}:ro" not in sidecar
    assert f"- {_CURSOR_AUTH_FILE}:{_CURSOR_AUTH_FILE}" not in sidecar
    for name, default in _AUTH_ENV_DEFAULTS.items():
        assert f"- {name}=${{{name}:-{default}}}" in sidecar

    assert "AAWM_CURSOR_AGENT_USAGE_POLL_ENABLED=${AAWM_CURSOR_AGENT_USAGE_POLL_ENABLED:-0}" in sidecar
    assert (
        "AAWM_CURSOR_AGENT_USAGE_POLL_INTERVAL_SECONDS="
        "${AAWM_CURSOR_AGENT_USAGE_POLL_INTERVAL_SECONDS:-600}" in sidecar
    )


def test_provider_status_image_packages_cursor_auth_refresh_dependencies() -> None:
    dockerfile = _PROVIDER_STATUS_DOCKERFILE_PATH.read_text(encoding="utf-8")

    assert _CURSOR_AUTH_REFRESH_SCRIPT_PATH.is_file()
    assert "COPY scripts/cursor_agent_auth_refresh.py /app/scripts/cursor_agent_auth_refresh.py" in dockerfile
    assert "COPY litellm/llms/cursor_agent/connect.py /app/litellm/llms/cursor_agent/connect.py" in dockerfile
    for helper_name in (
        "credential_error_sanitizer.py",
        "credential_file_lock.py",
        "credential_file_metadata.py",
        "credential_file_write.py",
    ):
        assert (f"COPY litellm/secret_managers/{helper_name} /app/litellm/secret_managers/{helper_name}") in dockerfile
    assert "COPY litellm /app/litellm" not in dockerfile


def test_cursor_docs_describe_sidecar_exchange_boundary() -> None:
    status_docs = _STATUS_DOC_PATH.read_text(encoding="utf-8")
    provider_docs = _CURSOR_DOC_PATH.read_text(encoding="utf-8")

    assert _CURSOR_AUTH_FILE in status_docs
    assert _CURSOR_LOCK_FILE in status_docs
    assert "AAWM_CURSOR_AGENT_AUTH_REFRESH_INTERVAL_SECONDS=300" in status_docs
    assert "AAWM_CURSOR_AGENT_AUTH_REFRESH_BUFFER_SECONDS=900" in status_docs
    assert "AAWM_CURSOR_AGENT_AUTH_FORCE_REFRESH=0" in status_docs
    assert "/auth/exchange_user_api_key" in status_docs
    assert "accessToken" in status_docs
    assert "refreshToken" in status_docs
    assert "fails closed" in status_docs

    assert "verified API-key exchange" in provider_docs
    assert "does not execute or depend on the Cursor CLI" in provider_docs
    assert "fails closed" in provider_docs
    assert "refreshToken-only grant" not in provider_docs
    assert "refreshToken-only grant" not in status_docs


def test_compose_files_parse_and_declare_expected_services() -> None:
    dev = yaml.safe_load(_DEV_COMPOSE_PATH.read_text(encoding="utf-8"))
    alpha = yaml.safe_load(_ALPHA_COMPOSE_PATH.read_text(encoding="utf-8"))

    assert set(dev["services"]) >= {"litellm-dev", "provider-status-observations"}
    assert set(alpha["services"]) == {"litellm-alpha"}

    assert dev["services"]["provider-status-observations"]["volumes"] is not None
    assert alpha["services"]["litellm-alpha"]["volumes"] is not None


def test_alias_routing_redis_uses_exact_aof_only_command() -> None:
    dev = yaml.safe_load(_DEV_COMPOSE_PATH.read_text(encoding="utf-8"))

    assert dev["services"]["aawm-alias-routing-redis"]["command"] == [
        "redis-server",
        "--save",
        "",
        "--appendonly",
        "yes",
    ]
