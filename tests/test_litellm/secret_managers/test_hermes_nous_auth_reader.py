"""NOUS-002: read-only Hermes Nous auth loader.

Never writes Hermes state. Never treats xAI/Copilot slots as Nous.
Tests use tmp fixtures only — they must not read ``~/.hermes/auth.json``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest

from litellm.secret_managers import hermes_nous_auth

_PROVIDERS_TOKEN = "fixture-nous-providers-access-token"
_PROVIDERS_AGENT_KEY = "fixture-nous-providers-agent-key"
_POOL_TOKEN = "fixture-nous-pool-access-token"
_POOL_AGENT_KEY = "fixture-nous-pool-agent-key"
_XAI_TOKEN = "fixture-xai-oauth-access-token"
_COPILOT_TOKEN = "fixture-copilot-access-token"
_SECRET_FIXTURES = (
    _PROVIDERS_TOKEN,
    _PROVIDERS_AGENT_KEY,
    _POOL_TOKEN,
    _POOL_AGENT_KEY,
    _XAI_TOKEN,
    _COPILOT_TOKEN,
)


def _write_auth(path: Path, payload: dict[str, Any]) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _providers_slot() -> dict[str, Any]:
    return {
        "access_token": _PROVIDERS_TOKEN,
        "agent_key": _PROVIDERS_AGENT_KEY,
        "refresh_token": "fixture-nous-providers-refresh-token",
    }


def _pool_slot() -> dict[str, Any]:
    return {
        "access_token": _POOL_TOKEN,
        "agent_key": _POOL_AGENT_KEY,
        "refresh_token": "fixture-nous-pool-refresh-token",
    }


def _full_document() -> dict[str, Any]:
    return {
        "providers": {"nous": _providers_slot()},
        "credential_pool": {"nous": _pool_slot()},
        "xai-oauth": {"access_token": _XAI_TOKEN},
        "copilot": {"access_token": _COPILOT_TOKEN},
    }


def test_prefers_providers_nous_over_credential_pool(tmp_path, monkeypatch):
    auth_path = _write_auth(tmp_path / "auth.json", _full_document())
    monkeypatch.setenv("LITELLM_HERMES_AUTH_FILE", str(auth_path))
    monkeypatch.delenv("AAWM_HERMES_AUTH_FILE", raising=False)

    token = hermes_nous_auth.load_nous_invoke_jwt()
    assert token == _PROVIDERS_TOKEN
    assert token != _POOL_TOKEN


def test_falls_back_to_credential_pool_nous_when_providers_missing(
    tmp_path, monkeypatch
):
    payload = {
        "providers": {},
        "credential_pool": {"nous": _pool_slot()},
        "xai-oauth": {"access_token": _XAI_TOKEN},
        "copilot": {"access_token": _COPILOT_TOKEN},
    }
    auth_path = _write_auth(tmp_path / "auth.json", payload)
    monkeypatch.setenv("LITELLM_HERMES_AUTH_FILE", str(auth_path))
    monkeypatch.delenv("AAWM_HERMES_AUTH_FILE", raising=False)

    token = hermes_nous_auth.load_nous_invoke_jwt()
    assert token == _POOL_TOKEN


def test_fail_closed_when_neither_slot_usable(tmp_path, monkeypatch):
    payload = {
        "providers": {},
        "credential_pool": {},
        "xai-oauth": {"access_token": _XAI_TOKEN},
        "copilot": {"access_token": _COPILOT_TOKEN},
    }
    auth_path = _write_auth(tmp_path / "auth.json", payload)
    monkeypatch.setenv("LITELLM_HERMES_AUTH_FILE", str(auth_path))
    monkeypatch.delenv("AAWM_HERMES_AUTH_FILE", raising=False)

    with pytest.raises(Exception):
        hermes_nous_auth.load_nous_invoke_jwt()


def test_does_not_read_xai_oauth_or_copilot_as_nous(tmp_path, monkeypatch):
    payload = {
        "providers": {"xai": {"access_token": _XAI_TOKEN}},
        "credential_pool": {"copilot": {"access_token": _COPILOT_TOKEN}},
        "xai-oauth": {"access_token": _XAI_TOKEN},
        "copilot": {"access_token": _COPILOT_TOKEN},
    }
    auth_path = _write_auth(tmp_path / "auth.json", payload)
    monkeypatch.setenv("LITELLM_HERMES_AUTH_FILE", str(auth_path))
    monkeypatch.delenv("AAWM_HERMES_AUTH_FILE", raising=False)

    with pytest.raises(Exception) as exc_info:
        hermes_nous_auth.load_nous_invoke_jwt()
    message = str(exc_info.value)
    assert _XAI_TOKEN not in message
    assert _COPILOT_TOKEN not in message


def test_env_LITELLM_HERMES_AUTH_FILE_and_AAWM_HERMES_AUTH_FILE(
    tmp_path, monkeypatch
):
    litellm_path = _write_auth(
        tmp_path / "litellm-auth.json",
        {"providers": {"nous": _providers_slot()}},
    )
    aawm_path = _write_auth(
        tmp_path / "aawm-auth.json",
        {"credential_pool": {"nous": _pool_slot()}},
    )

    monkeypatch.setenv("LITELLM_HERMES_AUTH_FILE", str(litellm_path))
    monkeypatch.setenv("AAWM_HERMES_AUTH_FILE", str(aawm_path))
    assert hermes_nous_auth.resolve_hermes_nous_auth_path() == str(litellm_path)
    assert hermes_nous_auth.load_nous_invoke_jwt() == _PROVIDERS_TOKEN

    monkeypatch.delenv("LITELLM_HERMES_AUTH_FILE", raising=False)
    monkeypatch.setenv("AAWM_HERMES_AUTH_FILE", str(aawm_path))
    assert hermes_nous_auth.resolve_hermes_nous_auth_path() == str(aawm_path)
    assert hermes_nous_auth.load_nous_invoke_jwt() == _POOL_TOKEN

    monkeypatch.delenv("AAWM_HERMES_AUTH_FILE", raising=False)
    resolved = hermes_nous_auth.resolve_hermes_nous_auth_path()
    assert resolved == os.path.expanduser("~/.hermes/auth.json") or resolved.endswith(
        os.path.join(".hermes", "auth.json")
    )


def test_reader_never_calls_write_helpers(tmp_path, monkeypatch):
    auth_path = _write_auth(tmp_path / "auth.json", _full_document())
    monkeypatch.setenv("LITELLM_HERMES_AUTH_FILE", str(auth_path))
    monkeypatch.delenv("AAWM_HERMES_AUTH_FILE", raising=False)

    def _fail_write(*_args: Any, **_kwargs: Any) -> None:
        pytest.fail("write_and_publish_private_text must not be called")

    monkeypatch.setattr(
        hermes_nous_auth,
        "write_and_publish_private_text",
        _fail_write,
        raising=False,
    )
    try:
        import litellm.proxy.auth.aawm_oauth_refresh as oauth_refresh

        monkeypatch.setattr(
            oauth_refresh,
            "write_and_publish_private_text",
            _fail_write,
            raising=False,
        )
    except Exception:
        pass

    token = hermes_nous_auth.load_nous_invoke_jwt()
    assert token == _PROVIDERS_TOKEN
    assert auth_path.read_text(encoding="utf-8") == json.dumps(_full_document())


def test_errors_are_sanitized(tmp_path, monkeypatch):
    payload = {
        "providers": {
            "nous": {
                "access_token": _PROVIDERS_TOKEN,
                "agent_key": _PROVIDERS_AGENT_KEY,
            }
        }
    }
    auth_path = _write_auth(tmp_path / "auth.json", payload)
    monkeypatch.setenv("LITELLM_HERMES_AUTH_FILE", str(auth_path))
    monkeypatch.delenv("AAWM_HERMES_AUTH_FILE", raising=False)

    original_json_load = json.load

    def _explode(handle: Any) -> Any:
        _ = original_json_load(handle)
        raise RuntimeError(
            f"failed while holding access_token={_PROVIDERS_TOKEN} "
            f"agent_key={_PROVIDERS_AGENT_KEY}"
        )

    monkeypatch.setattr(hermes_nous_auth.json, "load", _explode)

    with pytest.raises(Exception) as exc_info:
        hermes_nous_auth.load_nous_invoke_jwt()
    message = str(exc_info.value)
    for secret in _SECRET_FIXTURES:
        assert secret not in message
