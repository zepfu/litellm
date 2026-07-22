"""RED-phase tests for Wave 5: unauthenticated, fail-closed alias-config refresh endpoint.

New route: ``POST /aawm/alias-config/refresh`` (no auth dependency), registered
on ``router`` in ``llm_passthrough_endpoints.py``. Validates -> compiles ->
atomically swaps the active snapshot; on failure preserves last-known-good.

The route does not exist yet, so all requests will 404 in red phase — that is
the correct red signal (assertions on status_code != 404 will fail).
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints import router
from litellm.proxy.pass_through_endpoints.aawm_alias_routing import (  # type: ignore[import-not-found]
    config_snapshot,
)

REFRESH_PATH = "/aawm/alias-config/refresh"

_VALID_YAML = """
defaults: {}
aliases:
  - name: read
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
  - name: read
    candidates:
      - provider: totally_unregistered_provider_xyz
        model: whatever
        route_family: codex_responses
        priority: 0
"""


def _client() -> TestClient:
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


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
    assert active_snapshot.config_hash == good_hash


def test_in_flight_uses_prior_snapshot() -> None:
    """A selection begun before the swap uses the prior immutable snapshot."""
    client = _client()
    first = client.post(REFRESH_PATH, json={"yaml": _VALID_YAML})
    assert first.status_code == 200
    prior_snapshot = config_snapshot.get_active_snapshot()

    updated_yaml = _VALID_YAML.replace("openrouter/refresh-test-model", "openrouter/refresh-test-model-v2")
    second = client.post(REFRESH_PATH, json={"yaml": updated_yaml})
    assert second.status_code == 200

    # The reference captured before the swap must remain unmutated (immutability
    # + atomic swap — not a mutation of the same object in place).
    prior_models = [c.model for c in prior_snapshot.aliases["read"].candidates]
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
