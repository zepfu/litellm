"""Packaging contract for the Grok native client-version cache (xai-002).

Asserts that the provider-status Dockerfile and docker-compose.dev.yml
correctly package ``grok_native_version_contract.py``, expose the common
cache env defaults, mount the host cache directory read-only, and do not
couple the version cache to managed OAuth or a hardcoded billing version.
"""

from __future__ import annotations

import re
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_COMPOSE_PATH = _REPO_ROOT / "docker-compose.dev.yml"
_DOCKERFILE_PATH = (
    _REPO_ROOT / "docker" / "Dockerfile.provider_status_observations"
)

_CACHE_DIR_MOUNT = (
    "${AAWM_GROK_CLIENT_VERSION_CACHE_DIR"
    ":-/home/zepfu/.cache/aawm/grok}:/run/aawm/grok:ro"
)
_CACHE_PATH_ENV = (
    "AAWM_GROK_CLIENT_VERSION_CACHE_PATH="
    "${AAWM_GROK_CLIENT_VERSION_CACHE_PATH"
    ":-/run/aawm/grok/native-client-version.json}"
)
_CACHE_MAX_AGE_ENV = (
    "AAWM_GROK_CLIENT_VERSION_CACHE_MAX_AGE_SECONDS="
    "${AAWM_GROK_CLIENT_VERSION_CACHE_MAX_AGE_SECONDS:-172800}"
)


def _service_block(compose: str, service_name: str) -> str:
    marker = f"  {service_name}:\n"
    start = compose.index(marker)
    remainder = compose[start + len(marker) :]
    next_service = re.search(r"(?m)^  [a-zA-Z0-9_-]+:\n", remainder)
    return remainder[: next_service.start()] if next_service else remainder


# ---------------------------------------------------------------------------
# Dockerfile scaffolding
# ---------------------------------------------------------------------------


def test_dockerfile_copies_grok_native_version_contract() -> None:
    dockerfile = _DOCKERFILE_PATH.read_text(encoding="utf-8")

    assert (
        "COPY litellm/secret_managers/grok_native_version_contract.py "
        "/app/litellm/secret_managers/grok_native_version_contract.py"
    ) in dockerfile


def test_dockerfile_copies_secret_manager_sidecar_dependencies() -> None:
    """The loop imports these pure-stdlib secret-manager modules at startup;
    the minimal image must ship them."""
    dockerfile = _DOCKERFILE_PATH.read_text(encoding="utf-8")

    for module in (
        "grok_oidc_auth_path",
        "codex_oauth_inventory",
    ):
        assert (
            f"COPY litellm/secret_managers/{module}.py "
            f"/app/litellm/secret_managers/{module}.py"
        ) in dockerfile


def test_dockerfile_does_not_ship_full_aawm_integrations() -> None:
    """The psycopg-only sidecar image must not package the full
    ``litellm.integrations.aawm_agent_identity`` dependency chain."""
    dockerfile = _DOCKERFILE_PATH.read_text(encoding="utf-8")

    assert "litellm/integrations" not in dockerfile


def test_dockerfile_creates_secret_managers_init_for_import() -> None:
    dockerfile = _DOCKERFILE_PATH.read_text(encoding="utf-8")

    assert "/app/litellm/__init__.py" in dockerfile
    assert "/app/litellm/secret_managers/__init__.py" in dockerfile


def test_dockerfile_packages_stdlib_cursor_usage_helpers() -> None:
    """The slim sidecar copies stdlib Cursor usage helpers, not common_utils."""
    dockerfile = _DOCKERFILE_PATH.read_text(encoding="utf-8")

    assert (
        "COPY litellm/llms/cursor_agent/usage.py "
        "/app/litellm/llms/cursor_agent/usage.py"
    ) in dockerfile
    assert (
        "COPY litellm/llms/cursor_agent/usage_client.py "
        "/app/litellm/llms/cursor_agent/usage_client.py"
    ) in dockerfile
    assert "/app/litellm/llms/__init__.py" in dockerfile
    assert "/app/litellm/llms/cursor_agent/__init__.py" in dockerfile
    assert "COPY litellm/llms/cursor_agent/common_utils.py" not in dockerfile
    assert "COPY litellm/llms/__init__.py" not in dockerfile
    assert "COPY litellm/llms/cursor_agent/__init__.py" not in dockerfile
    assert "COPY litellm/__init__.py" not in dockerfile
    assert "COPY litellm /app/litellm" not in dockerfile
    assert 'pip install "litellm' not in dockerfile
    assert "pip install litellm" not in dockerfile
    assert "httpx" not in dockerfile
    assert 'pip install --no-cache-dir "psycopg[binary]==3.3.4"' in dockerfile


def test_dockerfile_has_no_hardcoded_billing_version() -> None:
    dockerfile = _DOCKERFILE_PATH.read_text(encoding="utf-8")

    assert "AAWM_GROK_BILLING_CLIENT_VERSION" not in dockerfile
    assert "0.2.55" not in dockerfile


def test_dockerfile_sets_cache_env_defaults() -> None:
    dockerfile = _DOCKERFILE_PATH.read_text(encoding="utf-8")

    assert (
        "AAWM_GROK_CLIENT_VERSION_CACHE_PATH="
        "/run/aawm/grok/native-client-version.json"
    ) in dockerfile
    assert (
        "AAWM_GROK_CLIENT_VERSION_CACHE_MAX_AGE_SECONDS=172800"
    ) in dockerfile


def test_dockerfile_creates_cache_directory() -> None:
    dockerfile = _DOCKERFILE_PATH.read_text(encoding="utf-8")

    assert "RUN mkdir -p /run/aawm/grok" in dockerfile


# ---------------------------------------------------------------------------
# Compose: litellm-dev service
# ---------------------------------------------------------------------------


def test_litellm_dev_mounts_cache_dir_read_only() -> None:
    compose = _COMPOSE_PATH.read_text(encoding="utf-8")
    litellm_dev = _service_block(compose, "litellm-dev")

    assert f"- {_CACHE_DIR_MOUNT}" in litellm_dev


def test_litellm_dev_sets_cache_envs() -> None:
    compose = _COMPOSE_PATH.read_text(encoding="utf-8")
    litellm_dev = _service_block(compose, "litellm-dev")

    assert f"- {_CACHE_PATH_ENV}" in litellm_dev
    assert f"- {_CACHE_MAX_AGE_ENV}" in litellm_dev


def test_litellm_dev_has_no_writable_cache_mount() -> None:
    compose = _COMPOSE_PATH.read_text(encoding="utf-8")
    litellm_dev = _service_block(compose, "litellm-dev")

    # A writable mount would lack the :ro suffix.
    assert "/run/aawm/grok\n" not in litellm_dev
    assert "/run/aawm/grok:" not in litellm_dev.replace(
        "/run/aawm/grok:ro", ""
    )


def test_litellm_dev_has_no_hardcoded_billing_version() -> None:
    compose = _COMPOSE_PATH.read_text(encoding="utf-8")
    litellm_dev = _service_block(compose, "litellm-dev")

    assert "AAWM_GROK_BILLING_CLIENT_VERSION" not in litellm_dev


def test_litellm_dev_wires_fail_closed_acceptance_controls() -> None:
    compose = _COMPOSE_PATH.read_text(encoding="utf-8")
    litellm_dev = _service_block(compose, "litellm-dev")

    assert (
        "- AAWM_OPENAI_FAULT_PLAN_ENABLED=${AAWM_OPENAI_FAULT_PLAN_ENABLED:-0}"
        in litellm_dev
    )
    assert (
        "- AAWM_CFG004_ACCEPTANCE_ENABLED=${AAWM_CFG004_ACCEPTANCE_ENABLED:-0}"
        in litellm_dev
    )
    assert (
        "- AAWM_CFG004_ACCEPTANCE_RUN_ID=${AAWM_CFG004_ACCEPTANCE_RUN_ID:-}"
        in litellm_dev
    )

    for service_name in (
        "aawm-alias-routing-redis",
        "provider-status-observations",
    ):
        other = _service_block(compose, service_name)
        assert "AAWM_OPENAI_FAULT_PLAN_ENABLED" not in other
        assert "AAWM_CFG004_ACCEPTANCE_ENABLED" not in other
        assert "AAWM_CFG004_ACCEPTANCE_RUN_ID" not in other


# ---------------------------------------------------------------------------
# Compose: provider-status-observations service
# ---------------------------------------------------------------------------


def test_provider_status_mounts_cache_dir_read_only() -> None:
    compose = _COMPOSE_PATH.read_text(encoding="utf-8")
    sidecar = _service_block(compose, "provider-status-observations")

    assert f"- {_CACHE_DIR_MOUNT}" in sidecar


def test_provider_status_sets_cache_envs() -> None:
    compose = _COMPOSE_PATH.read_text(encoding="utf-8")
    sidecar = _service_block(compose, "provider-status-observations")

    assert f"- {_CACHE_PATH_ENV}" in sidecar
    assert f"- {_CACHE_MAX_AGE_ENV}" in sidecar


def test_provider_status_has_no_writable_cache_mount() -> None:
    compose = _COMPOSE_PATH.read_text(encoding="utf-8")
    sidecar = _service_block(compose, "provider-status-observations")

    assert "/run/aawm/grok\n" not in sidecar
    assert "/run/aawm/grok:" not in sidecar.replace(
        "/run/aawm/grok:ro", ""
    )


def test_provider_status_has_no_hardcoded_billing_version() -> None:
    compose = _COMPOSE_PATH.read_text(encoding="utf-8")
    sidecar = _service_block(compose, "provider-status-observations")

    assert "AAWM_GROK_BILLING_CLIENT_VERSION" not in sidecar
    assert "0.2.55" not in sidecar


# ---------------------------------------------------------------------------
# No managed OAuth coupling
# ---------------------------------------------------------------------------


def test_cache_mount_is_independent_of_managed_xai_oauth() -> None:
    """The version-cache mount must not reference the managed xAI OAuth
    credential path."""
    compose = _COMPOSE_PATH.read_text(encoding="utf-8")

    for service in ("litellm-dev", "provider-status-observations"):
        block = _service_block(compose, service)
        cache_lines = [
            ln for ln in block.splitlines() if "/run/aawm/grok" in ln
        ]
        for ln in cache_lines:
            assert ".litellm/xai" not in ln
            assert "oauth" not in ln.lower()


def test_no_file_bind_for_cache_json() -> None:
    """The cache JSON must never be file-bound; only the parent directory
    is mounted."""
    compose = _COMPOSE_PATH.read_text(encoding="utf-8")

    assert "native-client-version.json:" not in compose
