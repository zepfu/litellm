"""D1-592: sidecar resolvers must not silently use $HOME defaults when flagged.

With ``AAWM_REQUIRE_EXPLICIT_AUTH_PATHS=1`` and no AAWM_* auth file env, grok /
kimi / xai / codex sidecar resolvers must not return ~/.grok, ~/.kimi-code,
~/.codex, or ~/.litellm/xai defaults. They must raise a sanitized auth error.
Flag off keeps today's fallback.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts import run_provider_status_observations_loop as loop


AAWM_AUTH_ENVS = (
    "AAWM_GROK_OIDC_AUTH_FILE",
    "AAWM_CODEX_AUTH_FILE",
    "AAWM_XAI_OAUTH_AUTH_FILE",
    "AAWM_KIMI_OAUTH_AUTH_FILE",
    "LITELLM_XAI_GROK_AUTH_FILE",
    "LITELLM_XAI_OAUTH_GROK_AUTH_FILE",
    "GROK_AUTH_FILE",
    "GROK_HOME",
    "LITELLM_CODEX_AUTH_FILE",
    "CHATGPT_AUTH_FILE",
    "LITELLM_CODEX_TOKEN_DIR",
    "CHATGPT_TOKEN_DIR",
    "LITELLM_XAI_OAUTH_AUTH_FILE",
    "LITELLM_XAI_OAUTH_MIGRATED_AUTH_FILE",
)


def _clear_auth_env(monkeypatch) -> None:
    monkeypatch.delenv("AAWM_REQUIRE_EXPLICIT_AUTH_PATHS", raising=False)
    for name in AAWM_AUTH_ENVS:
        monkeypatch.delenv(name, raising=False)


def _assert_sanitized_auth_error(exc: BaseException, *, home: Path) -> None:
    message = str(exc)
    assert "telemetry_class=auth" in message or getattr(exc, "telemetry_class", None) == "auth"
    lowered = message.lower()
    assert str(home) not in message
    assert "/.grok/" not in lowered
    assert "/.kimi-code/" not in lowered
    assert "/.codex/" not in lowered
    assert "/.litellm/xai/" not in lowered
    secretish = ("token", "password", "secret", "refresh_token")
    for needle in secretish:
        assert needle not in lowered or "auth" in lowered


@pytest.mark.parametrize(
    ("resolver_name", "default_fragment"),
    (
        ("_resolve_grok_sidecar_auth_file", ".grok/auth.json"),
        ("_resolve_kimi_oauth_sidecar_auth_file", ".kimi-code/credentials/kimi-code.json"),
        ("_resolve_xai_oauth_sidecar_auth_file", ".litellm/xai/oauth-auth.json"),
        ("_resolve_codex_sidecar_auth_file", ".codex/auth.json"),
    ),
)
def test_require_explicit_auth_paths_rejects_home_defaults(
    monkeypatch, tmp_path, resolver_name, default_fragment
) -> None:
    _clear_auth_env(monkeypatch)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("AAWM_REQUIRE_EXPLICIT_AUTH_PATHS", "1")
    resolver = getattr(loop, resolver_name)

    with pytest.raises(Exception) as captured:
        resolver(None)

    _assert_sanitized_auth_error(captured.value, home=tmp_path)
    # Must not return a default path even if a caller catches and continues.
    assert default_fragment not in str(captured.value)


@pytest.mark.parametrize(
    ("resolver_name", "default_fragment"),
    (
        ("_resolve_grok_sidecar_auth_file", ".grok/auth.json"),
        ("_resolve_kimi_oauth_sidecar_auth_file", ".kimi-code/credentials/kimi-code.json"),
        ("_resolve_xai_oauth_sidecar_auth_file", ".litellm/xai/oauth-auth.json"),
        ("_resolve_codex_sidecar_auth_file", ".codex/auth.json"),
    ),
)
def test_flag_off_keeps_today_home_fallback(
    monkeypatch, tmp_path, resolver_name, default_fragment
) -> None:
    _clear_auth_env(monkeypatch)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("AAWM_REQUIRE_EXPLICIT_AUTH_PATHS", "0")
    resolver = getattr(loop, resolver_name)

    resolved_path, source = resolver(None)

    assert source == "default"
    assert default_fragment in resolved_path.replace("\\", "/")
    assert str(tmp_path) in resolved_path or resolved_path.startswith("~")


def test_explicit_aawm_env_still_wins_when_flag_on(monkeypatch, tmp_path) -> None:
    _clear_auth_env(monkeypatch)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("AAWM_REQUIRE_EXPLICIT_AUTH_PATHS", "1")
    grok = tmp_path / "managed" / "grok.json"
    grok.parent.mkdir()
    grok.write_text("{}", encoding="utf-8")
    monkeypatch.setenv("AAWM_GROK_OIDC_AUTH_FILE", str(grok))

    resolved_path, source = loop._resolve_grok_sidecar_auth_file(None)
    assert resolved_path == str(grok)
    assert source == "AAWM_GROK_OIDC_AUTH_FILE"
