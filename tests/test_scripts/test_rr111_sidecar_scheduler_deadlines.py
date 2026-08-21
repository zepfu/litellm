"""RR-111: slow optional sidecar polls must not block credential refresh or shutdown."""

from __future__ import annotations

import threading
import time
from datetime import datetime, timezone

from scripts import run_provider_status_observations_loop as loop


def _scheduler_config(**overrides):
    kwargs = {
        "apply": False,
        "dsn": None,
        "environment": "dev",
        "interval_seconds": 300.0,
        "timeout": 2.0,
        "ping_count": 1,
        "ping_timeout": 2,
        "skip_icmp": False,
        "once": True,
        "setup_schema": False,
        "db_lock_timeout_ms": 1000,
        "db_statement_timeout_ms": 5000,
        "grok_oidc_refresh_enabled": True,
        "grok_oidc_auth_file": "/tmp/rr111-grok-oidc-auth.json",
        "alibaba_quota_poll_enabled": True,
        "alibaba_quota_poll_interval_seconds": 300.0,
    }
    kwargs.update(overrides)
    return loop.ProviderStatusLoopConfig(**kwargs)


def test_slow_optional_poll_does_not_delay_oauth_refresh_or_unbounded_shutdown(monkeypatch) -> None:
    poll_started = threading.Event()
    poll_release = threading.Event()
    oauth_completed = threading.Event()
    finished: list[list] = []
    errors: list[BaseException] = []

    def fake_metadata_repair(*_args, **_kwargs):
        return None

    def fake_oauth(config, state, *, now_monotonic, now_wall=None):
        del config, state, now_monotonic
        oauth_completed.set()
        wall = now_wall or datetime.now(timezone.utc)
        return {
            "event": "grok_oidc_refresh",
            "observed_at": wall.isoformat().replace("+00:00", "Z"),
            "environment": "dev",
            "attempted": True,
            "refreshed": True,
            "skipped": False,
        }

    def slow_optional_poll(config, state, *, now_monotonic):
        del config, state, now_monotonic
        poll_started.set()
        if not poll_release.wait(timeout=2.0):
            raise TimeoutError("optional poll was not released")
        return {
            "event": "alibaba_quota_poll",
            "attempted": True,
            "skipped": False,
        }

    monkeypatch.setattr(loop, "_run_grok_oidc_metadata_repair_task", fake_metadata_repair)
    monkeypatch.setattr(loop, "_run_grok_oidc_refresh_task", fake_oauth)
    monkeypatch.setattr(loop, "_run_alibaba_quota_poll_task", slow_optional_poll)

    def _run() -> None:
        try:
            events = loop.run_due_sidecar_tasks(
                _scheduler_config(),
                loop.SidecarTaskState(),
                now_monotonic=100.0,
            )
            finished.append(events)
        except BaseException as exc:  # pragma: no cover - surfaced below
            errors.append(exc)

    worker = threading.Thread(target=_run, name="rr111-sidecar-scheduler")
    started = time.monotonic()
    worker.start()
    assert oauth_completed.wait(timeout=0.2)
    oauth_elapsed = time.monotonic() - started
    assert oauth_elapsed < 0.15
    worker.join(timeout=0.2)
    still_running_after_deadline = worker.is_alive()
    poll_release.set()
    worker.join(timeout=1.0)

    assert errors == []
    assert still_running_after_deadline is False
    assert finished
    events = finished[0]
    assert any(event.get("event") == "grok_oidc_refresh" for event in events)
