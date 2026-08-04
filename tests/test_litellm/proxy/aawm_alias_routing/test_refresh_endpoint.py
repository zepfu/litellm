"""RED-phase tests for Wave 5: unauthenticated, fail-closed alias-config refresh endpoint.

New route: ``POST /aawm/alias-config/refresh`` (no auth dependency), registered
on ``router`` in ``llm_passthrough_endpoints.py``. Validates -> compiles ->
atomically swaps the active snapshot; on failure preserves last-known-good.

The route does not exist yet, so all requests will 404 in red phase — that is
the correct red signal (assertions on status_code != 404 will fail).
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from pathlib import Path

import httpx
import pytest
from fastapi import FastAPI

from litellm.proxy.pass_through_endpoints import llm_passthrough_endpoints as lpe
from litellm.proxy.pass_through_endpoints.aawm_alias_routing import (  # type: ignore[import-not-found]
    config_refresh,
    config_snapshot,
    config_startup,
)

REFRESH_PATH = "/aawm/alias-config/refresh"

_VALID_YAML = """
defaults: {}
aliases:
  - name: basic
    candidates:
      - provider: openrouter
        model: openrouter/refresh-test-model
        route_family: codex_openrouter_completion_adapter
        priority: 100
      - provider: openai
        model: gpt-5.4-mini
        route_family: codex_responses
        priority: 0
"""

_INVALID_YAML = """
defaults: {}
aliases:
  - name: basic
    candidates:
      - provider: totally_unregistered_provider_xyz
        model: whatever
        route_family: codex_responses
        priority: 0
"""


class _ASGIClient:
    def __init__(self, app: FastAPI) -> None:
        self._app = app

    def post(self, path: str, *, json: object) -> httpx.Response:
        async def _post() -> httpx.Response:
            transport = httpx.ASGITransport(app=self._app)
            async with httpx.AsyncClient(
                transport=transport,
                base_url="http://testserver",
            ) as client:
                return await client.post(path, json=json)

        return asyncio.run(_post())


def _client() -> _ASGIClient:
    app = FastAPI()
    app.include_router(lpe.router)
    return _ASGIClient(app)


@pytest.fixture(autouse=True)
def _reset_alias_routing_state() -> Iterator[None]:
    previous_snapshot = config_refresh.get_active_routing_snapshot()
    lpe.reset_alias_routing_state_for_tests()
    try:
        yield
    finally:
        lpe.reset_alias_routing_state_for_tests()
        config_refresh.set_active_routing_snapshot(previous_snapshot)


def test_valid_refresh_compiles_and_activates() -> None:
    """Valid YAML compiles + atomically activates; response has hashes + changed=true."""
    client = _client()
    response = client.post(REFRESH_PATH, json={"yaml": _VALID_YAML})
    assert response.status_code == 200
    payload = response.json()
    assert payload["changed"] is True
    assert "attempted_config_hash" in payload
    assert "active_config_hash" in payload
    assert payload["attempted_config_hash"] == payload["active_config_hash"]
    assert "config_version" in payload


def test_noop_refresh_reports_no_change() -> None:
    """Identical re-post reports changed=false with the same active hash."""
    client = _client()
    first = client.post(REFRESH_PATH, json={"yaml": _VALID_YAML})
    assert first.status_code == 200
    first_hash = first.json()["active_config_hash"]

    second = client.post(REFRESH_PATH, json={"yaml": _VALID_YAML})
    assert second.status_code == 200
    second_payload = second.json()
    assert second_payload["changed"] is False
    assert second_payload["active_config_hash"] == first_hash


def test_invalid_refresh_fails_closed() -> None:
    """Malformed config is rejected; previously active snapshot remains active."""
    client = _client()
    good = client.post(REFRESH_PATH, json={"yaml": _VALID_YAML})
    assert good.status_code == 200
    good_hash = good.json()["active_config_hash"]

    bad = client.post(REFRESH_PATH, json={"yaml": _INVALID_YAML})
    assert bad.status_code in (400, 422)
    bad_payload = bad.json()
    # Secret-safe: no raw config content echoed back.
    assert _VALID_YAML not in str(bad_payload)
    assert _INVALID_YAML not in str(bad_payload)

    # Last-known-good remains active after a failed refresh.
    active_snapshot = config_snapshot.get_active_snapshot()
    assert active_snapshot is not None
    assert active_snapshot.config_hash == good_hash


def test_in_flight_uses_prior_snapshot() -> None:
    """A selection begun before the swap uses the prior immutable snapshot."""
    client = _client()
    first = client.post(REFRESH_PATH, json={"yaml": _VALID_YAML})
    assert first.status_code == 200
    prior_snapshot = config_snapshot.get_active_snapshot()
    assert prior_snapshot is not None

    updated_yaml = _VALID_YAML.replace("openrouter/refresh-test-model", "openrouter/refresh-test-model-v2")
    second = client.post(REFRESH_PATH, json={"yaml": updated_yaml})
    assert second.status_code == 200

    # The reference captured before the swap must remain unmutated (immutability
    # + atomic swap — not a mutation of the same object in place).
    prior_models = [c.model for c in prior_snapshot.aliases["basic"].candidates]
    assert "openrouter/refresh-test-model" in prior_models
    assert "openrouter/refresh-test-model-v2" not in prior_models


def test_no_auth_required() -> None:
    """Accepts an unauthenticated LAN request (no Authorization header, no api key)."""
    client = _client()
    response = client.post(REFRESH_PATH, json={"yaml": _VALID_YAML})
    assert response.status_code == 200


def test_response_omits_secrets() -> None:
    """Response never includes credentials or raw config secrets."""
    client = _client()
    secret_bearing_yaml = _VALID_YAML + "\n# api_key: sk-super-secret-value-should-not-leak\n"
    response = client.post(REFRESH_PATH, json={"yaml": secret_bearing_yaml})
    body_text = response.text
    assert "sk-super-secret-value-should-not-leak" not in body_text


# ---------------------------------------------------------------------------
# Pass 3: readiness/status reflects live active snapshot after refresh
# ---------------------------------------------------------------------------

_STARTUP_YAML = """
defaults: {}
aliases:
  - name: basic
    candidates:
      - provider: openai
        model: gpt-5.4-mini
        route_family: codex_responses
        priority: 100
"""

_REFRESH_YAML = """
defaults: {}
aliases:
  - name: basic
    candidates:
      - provider: openrouter
        model: openrouter/pass3-model
        route_family: codex_openrouter_completion_adapter
        priority: 100
      - provider: openai
        model: gpt-5.4-mini
        route_family: codex_responses
        priority: 0
"""


@pytest.fixture()
def _startup_dir(tmp_path: Path) -> Iterator[Path]:
    """Create a temp config directory with a startup YAML and activate it."""
    config_dir = tmp_path / "alias_config"
    config_dir.mkdir()
    (config_dir / "basic.yaml").write_text(_STARTUP_YAML, encoding="utf-8")
    config_startup.reset_startup_state()
    config_startup.activate_alias_config_directory(config_dir)
    assert config_startup.is_startup_healthy()
    yield config_dir
    config_startup.reset_startup_state()


def test_readiness_reflects_refreshed_identity(_startup_dir: Path) -> None:
    """After live refresh, get_startup_status hash/version/epoch/aliases match active snapshot."""
    original_status = config_startup.get_startup_status()
    assert original_status["state"] == "active"
    original_hash = original_status["config_hash"]

    client = _client()
    response = client.post(REFRESH_PATH, json={"yaml": _REFRESH_YAML})
    assert response.status_code == 200
    payload = response.json()
    assert payload["changed"] is True
    refreshed_hash = payload["active_config_hash"]
    assert refreshed_hash != original_hash

    # Readiness/status must now reflect the refreshed snapshot.
    status = config_startup.get_startup_status()
    assert status["state"] == "active"
    assert status["config_hash"] == refreshed_hash
    assert status["config_version"] == payload["config_version"]

    # Active snapshot holder agrees.
    active = config_snapshot.get_active_snapshot()
    assert active is not None
    assert active.config_hash == status["config_hash"]
    assert active.config_version == status["config_version"]
    assert active.config_epoch == status["config_epoch"]

    # Alias identity from status matches the active snapshot.
    assert sorted(active.aliases.keys()) == status["aliases"]
    assert len(active.aliases) == status["alias_count"]


def test_readiness_restores_original_identity_after_restore(
    _startup_dir: Path,
) -> None:
    """After refresh then restore, readiness returns original hash/version."""
    original_status = config_startup.get_startup_status()
    original_hash = original_status["config_hash"]
    original_version = original_status["config_version"]

    client = _client()

    # Refresh to new config.
    r1 = client.post(REFRESH_PATH, json={"yaml": _REFRESH_YAML})
    assert r1.status_code == 200
    assert r1.json()["changed"] is True

    # Restore original config.
    r2 = client.post(REFRESH_PATH, json={"yaml": _STARTUP_YAML})
    assert r2.status_code == 200
    assert r2.json()["changed"] is True

    status = config_startup.get_startup_status()
    assert status["state"] == "active"
    assert status["config_hash"] == original_hash
    assert status["config_version"] == original_version


def test_readiness_coherent_after_unchanged_refresh(
    _startup_dir: Path,
) -> None:
    """Unchanged refresh (changed=false) keeps readiness identity stable."""
    client = _client()
    r1 = client.post(REFRESH_PATH, json={"yaml": _REFRESH_YAML})
    assert r1.status_code == 200
    hash_after_first = r1.json()["active_config_hash"]

    r2 = client.post(REFRESH_PATH, json={"yaml": _REFRESH_YAML})
    assert r2.status_code == 200
    assert r2.json()["changed"] is False

    status = config_startup.get_startup_status()
    assert status["config_hash"] == hash_after_first
    active = config_snapshot.get_active_snapshot()
    assert active is not None
    assert active.config_hash == status["config_hash"]


def test_readiness_preserves_lkg_after_invalid_refresh(
    _startup_dir: Path,
) -> None:
    """Invalid refresh preserves LKG; readiness still reflects last good snapshot."""
    client = _client()
    good = client.post(REFRESH_PATH, json={"yaml": _REFRESH_YAML})
    assert good.status_code == 200
    good_hash = good.json()["active_config_hash"]

    bad = client.post(REFRESH_PATH, json={"yaml": _INVALID_YAML})
    assert bad.status_code in (400, 422)

    status = config_startup.get_startup_status()
    assert status["state"] == "active"
    assert status["config_hash"] == good_hash


def test_startup_failed_status_not_affected_by_refresh_holder() -> None:
    """Failed startup state still reports failed regardless of holder contents."""
    config_startup.reset_startup_state()
    # Simulate a failed startup by activating a nonexistent directory.
    config_startup.activate_alias_config_directory(Path("/nonexistent_dir_xyz"))
    assert config_startup.is_startup_failed()

    status = config_startup.get_startup_status()
    assert status["state"] == "failed"
    assert "error_class" in status


def test_not_loaded_status() -> None:
    """Not-loaded startup state reports not_loaded."""
    config_startup.reset_startup_state()
    status = config_startup.get_startup_status()
    assert status["state"] == "not_loaded"
