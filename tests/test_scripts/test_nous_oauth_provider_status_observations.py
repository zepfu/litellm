"""Focused Nous OAuth compose + observation mapping for the provider-status sidecar."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from scripts import run_provider_status_observations_loop as loop


def _nous_oauth_auth_persist_config(**overrides):
    from dataclasses import replace

    config = loop.ProviderStatusLoopConfig(
        apply=True,
        dsn="postgresql://aawm:aawm_dev@pgbouncer:6432/aawm_tristore",
        environment="dev",
        interval_seconds=300.0,
        timeout=2.0,
        ping_count=1,
        ping_timeout=2,
        skip_icmp=True,
        once=True,
        setup_schema=False,
        db_lock_timeout_ms=1000,
        db_statement_timeout_ms=5000,
        grok_oidc_refresh_enabled=False,
        nous_oauth_refresh_enabled=True,
        nous_oauth_auth_file="~/.hermes/auth.json",
        nous_oauth_auth_file_source="default",
        nous_oauth_lock_file="~/.hermes/auth.lock",
        nous_oauth_refresh_interval_seconds=300.0,
        nous_oauth_refresh_buffer_seconds=900,
        nous_oauth_force_refresh=False,
        nous_oauth_http_timeout_seconds=30.0,
    )
    if overrides:
        config = replace(config, **overrides)
    return config


def _nous_oauth_refresh_sidecar_event(**overrides) -> dict:
    event = {
        "event": "nous_oauth_refresh",
        "observed_at": "2026-06-19T12:00:00Z",
        "environment": "dev",
        "attempted": True,
        "refreshed": True,
        "skipped": False,
        "auth_file": "~/.hermes/auth.json",
        "scope": "inference:invoke",
        "expires_at": "2026-06-19T13:00:00Z",
        "error_class": None,
        "error_message": None,
    }
    event.update(overrides)
    return event


def test_provider_status_compose_wires_nous_oauth_sidecar_refresh() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    compose_text = (repo_root / "docker-compose.dev.yml").read_text()
    dockerfile_text = (
        repo_root / "docker/Dockerfile.provider_status_observations"
    ).read_text()

    assert "- /home/zepfu/.hermes:/home/zepfu/.hermes:ro" in compose_text
    assert "- /home/zepfu/.hermes:/home/zepfu/.hermes" in compose_text
    assert "- /home/zepfu/.hermes/auth.json:/home/zepfu/.hermes/auth.json" not in compose_text
    assert (
        "LITELLM_NOUS_OAUTH_AUTH_FILE=${LITELLM_NOUS_OAUTH_AUTH_FILE:-"
        "/home/zepfu/.hermes/auth.json}"
    ) in compose_text
    assert (
        "AAWM_NOUS_OAUTH_REFRESH_ENABLED=${AAWM_NOUS_OAUTH_REFRESH_ENABLED:-1}"
        in compose_text
    )
    assert (
        "AAWM_NOUS_OAUTH_AUTH_FILE=${AAWM_NOUS_OAUTH_AUTH_FILE:-"
        "/home/zepfu/.hermes/auth.json}"
    ) in compose_text
    assert (
        "AAWM_NOUS_OAUTH_LOCK_FILE=${AAWM_NOUS_OAUTH_LOCK_FILE:-"
        "/home/zepfu/.hermes/auth.lock}"
    ) in compose_text
    assert (
        "AAWM_NOUS_OAUTH_REFRESH_INTERVAL_SECONDS=${AAWM_NOUS_OAUTH_REFRESH_INTERVAL_SECONDS:-300}"
        in compose_text
    )
    assert (
        "AAWM_NOUS_OAUTH_REFRESH_BUFFER_SECONDS=${AAWM_NOUS_OAUTH_REFRESH_BUFFER_SECONDS:-900}"
        in compose_text
    )
    assert (
        "AAWM_NOUS_OAUTH_FORCE_REFRESH=${AAWM_NOUS_OAUTH_FORCE_REFRESH:-0}"
        in compose_text
    )
    assert (
        "COPY scripts/nous_oauth_refresh.py "
        "/app/scripts/nous_oauth_refresh.py"
    ) in dockerfile_text
    wsl_compose = (repo_root / "docker-compose.wsl-grok-oidc.yml").read_text()
    assert "/home/zepfu/.hermes" not in wsl_compose
    assert "AAWM_NOUS_OAUTH" not in wsl_compose


def test_build_nous_oauth_auth_observation_maps_successful_refresh() -> None:
    config = _nous_oauth_auth_persist_config()
    event = _nous_oauth_refresh_sidecar_event()

    observation = loop._build_nous_oauth_auth_observation(config, event)

    assert observation["observed_at"] == datetime(2026, 6, 19, 12, 0, tzinfo=timezone.utc)
    assert observation["environment"] == "dev"
    assert observation["provider"] == "nous"
    assert observation["auth_family"] == "nous_oauth"
    assert observation["credential_scope"] == "inference:invoke"
    assert observation["auth_file_hash"] == hashlib.sha256(
        event["auth_file"].encode("utf-8")
    ).hexdigest()
    assert observation["status"] == "refreshed"
    assert observation["attempted"] is True
    assert observation["refreshed"] is True
    assert observation["skipped"] is False
    assert observation["expires_at"] == datetime(2026, 6, 19, 13, 0, tzinfo=timezone.utc)
    assert observation["last_success_at"] == observation["observed_at"]
    assert observation["source_task"] == "nous_oauth_refresh"
    observation_json = json.dumps(observation, default=str)
    assert "refresh_token" not in observation_json
    assert "access_token" not in observation_json
    assert "agent_key" not in observation_json
    assert "/home/zepfu/.hermes/auth.json" not in observation_json
    assert "~/.hermes/auth.json" not in observation_json


def test_build_nous_oauth_auth_observation_sanitizes_refresh_failure() -> None:
    config = _nous_oauth_auth_persist_config()
    event = _nous_oauth_refresh_sidecar_event(
        refreshed=False,
        skipped=False,
        expires_at=None,
        error_class="ValueError",
        error_message=(
            "token refresh failed with refresh_token=old-refresh "
            "agent_key=old-agent-key and Authorization=Bearer leaked-token"
        ),
    )

    observation = loop._build_nous_oauth_auth_observation(config, event)

    assert observation["status"] == "failed"
    assert observation["last_success_at"] is None
    assert observation["error_class"] == "ValueError"
    assert "REDACTED" in observation["error_message"]
    rendered = json.dumps(observation, default=str)
    assert "old-refresh" not in rendered
    assert "old-agent-key" not in rendered
    assert "leaked-token" not in rendered


def test_run_due_sidecar_tasks_persists_nous_oauth_auth_observation_when_apply_enabled(
    monkeypatch,
) -> None:
    config = _nous_oauth_auth_persist_config(
        nous_oauth_force_refresh=True,
        nous_oauth_refresh_interval_seconds=3600.0,
    )
    captured = {}

    def fake_nous_refresh(*_args, **kwargs):
        kwargs["on_token_endpoint_attempt"]()
        return {
            "attempted": True,
            "refreshed": True,
            "skipped": False,
            "auth_file": config.nous_oauth_auth_file,
            "scope": "inference:invoke",
            "expires_at": "2026-06-19T13:00:00Z",
            "error_class": None,
            "error_message": None,
        }

    monkeypatch.setattr(
        loop.nous_oauth_refresh,
        "refresh_nous_oauth_auth_file",
        fake_nous_refresh,
    )

    def fake_persist(persist_config, event):
        captured["config"] = persist_config
        captured["event"] = dict(event)
        return True, 1, None, None

    monkeypatch.setattr(loop, "_persist_nous_oauth_auth_observation", fake_persist)

    events = loop.run_due_sidecar_tasks(
        config,
        loop.SidecarTaskState(),
        now_monotonic=100.0,
        now_wall=datetime(2026, 6, 19, 12, 0, tzinfo=timezone.utc),
    )

    refresh_events = [
        event for event in events if event.get("event") == "nous_oauth_refresh"
    ]
    assert len(refresh_events) == 1
    assert refresh_events[0]["auth_observation_status"] == "refreshed"
    assert refresh_events[0]["auth_observation_persisted"] is True
    assert refresh_events[0]["auth_observation_inserted_count"] == 1
    assert captured["config"] is config
    assert captured["event"]["scope"] == "inference:invoke"


def test_passive_auth_health_poll_maps_sanitized_nous_oauth_row(monkeypatch) -> None:
    config = _nous_oauth_auth_persist_config(
        apply=True,
        provider_auth_health_poll_enabled=True,
        provider_auth_health_poll_interval_seconds=3600.0,
        nous_oauth_refresh_enabled=False,
    )
    persisted = []

    monkeypatch.setattr(
        loop.nous_oauth_refresh,
        "inspect_nous_oauth_credential_health",
        lambda *_args, **_kwargs: {
            "attempted": True,
            "refreshed": False,
            "skipped": False,
            "auth_file": config.nous_oauth_auth_file,
            "scope": "inference:invoke",
            "health_status": "fresh",
            "expires_at": "2026-07-22T18:00:00Z",
            "error_class": None,
            "error_message": None,
        },
    )
    for inspector_name, owner in (
        ("inspect_grok_oidc_credential_health", loop.grok_oidc_refresh),
        ("inspect_codex_oauth_credential_health", loop.codex_oauth_refresh),
        ("inspect_xai_oauth_credential_health", loop.xai_oauth_refresh),
        ("inspect_kimi_oauth_credential_health", loop.kimi_oauth_refresh),
    ):
        monkeypatch.setattr(
            owner,
            inspector_name,
            lambda *_args, **_kwargs: {
                "attempted": True,
                "refreshed": False,
                "skipped": False,
                "health_status": "fresh",
                "expires_at": "2026-07-22T18:00:00Z",
                "error_class": None,
                "error_message": None,
            },
        )

    def fake_persist(_config, observation):
        persisted.append(dict(observation))
        return True, 1, None, None

    monkeypatch.setattr(
        loop, "_persist_passive_provider_auth_observation", fake_persist
    )

    events = loop._run_provider_auth_health_poll_task(
        config,
        loop.SidecarTaskState(),
        now_monotonic=100.0,
    )
    nous_events = [
        event
        for event in events
        if event.get("event") == "nous_oauth_passive_health_inspection"
    ]
    nous_rows = [row for row in persisted if row.get("auth_family") == "nous_oauth"]

    assert len(nous_events) == 1
    assert nous_events[0]["source_task"] == "provider_auth_health_poll"
    assert "auth_file" not in nous_events[0]
    assert len(nous_rows) == 1
    assert nous_rows[0]["provider"] == "nous"
    assert nous_rows[0]["auth_family"] == "nous_oauth"
    assert nous_rows[0]["source_task"] == "provider_auth_health_poll"
    rendered = json.dumps(nous_rows[0], default=str)
    assert "old-access" not in rendered
    assert "old-refresh" not in rendered
    assert "old-agent-key" not in rendered
    assert "~/.hermes/auth.json" not in rendered
