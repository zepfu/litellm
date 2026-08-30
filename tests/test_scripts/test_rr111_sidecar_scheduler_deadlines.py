"""RR-111: slow optional sidecar polls must not block credential refresh or shutdown."""

from __future__ import annotations

import threading
import time
from datetime import datetime, timedelta, timezone

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
        "alibaba_quota_poll_interval_seconds": 600.0,
    }
    kwargs.update(overrides)
    return loop.ProviderStatusLoopConfig(**kwargs)


def _iso(value: datetime) -> str:
    return value.isoformat().replace("+00:00", "Z")


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


def test_next_sidecar_wake_uses_the_earliest_managed_provider_deadline() -> None:
    wall_now = datetime(2026, 8, 30, 12, 0, tzinfo=timezone.utc)
    state = loop.SidecarTaskState(next_generic_cycle_due_monotonic=500.0)
    state.grok_oidc_refresh_schedule.next_refresh_check_at = _iso(
        wall_now + timedelta(seconds=250)
    )
    state.xai_oauth_refresh_schedule.next_refresh_check_at = _iso(
        wall_now + timedelta(seconds=200)
    )
    state.kimi_oauth_refresh_schedule.next_refresh_check_at = _iso(
        wall_now + timedelta(seconds=150)
    )
    state.nous_oauth_refresh_schedule.next_refresh_check_at = _iso(
        wall_now + timedelta(seconds=100)
    )
    state.cursor_agent_auth_refresh_schedule.next_refresh_check_at = _iso(
        wall_now + timedelta(seconds=80)
    )
    state.codex_oauth_refresh_schedule_by_label["account1"] = (
        loop.OAuthRefreshScheduleState(
            next_refresh_check_at=_iso(wall_now + timedelta(seconds=60))
        )
    )
    state.codex_oauth_refresh_schedule_by_label["account2"] = (
        loop.OAuthRefreshScheduleState(
            next_refresh_check_at=_iso(wall_now + timedelta(seconds=40))
        )
    )
    config = _scheduler_config(
        interval_seconds=400.0,
        codex_oauth_refresh_enabled=True,
        xai_oauth_refresh_enabled=True,
        kimi_oauth_refresh_enabled=True,
        nous_oauth_refresh_enabled=True,
        cursor_agent_auth_refresh_enabled=True,
    )

    assert (
        loop._next_sidecar_wake_delay(
            config,
            state,
            now=100.0,
            wall_now=wall_now,
        )
        == 40.0
    )


def test_managed_refresh_runs_at_deadline_before_generic_cycle(monkeypatch) -> None:
    wall_now = datetime(2026, 8, 30, 12, 0, tzinfo=timezone.utc)
    state = loop.SidecarTaskState()
    state.grok_oidc_refresh_schedule.next_refresh_check_at = _iso(wall_now)
    state.xai_oauth_refresh_schedule.next_refresh_check_at = _iso(wall_now)
    state.kimi_oauth_refresh_schedule.next_refresh_check_at = _iso(
        wall_now + timedelta(seconds=1)
    )
    state.nous_oauth_refresh_schedule.next_refresh_check_at = _iso(
        wall_now + timedelta(seconds=1)
    )
    state.cursor_agent_auth_refresh_schedule.next_refresh_check_at = _iso(
        wall_now + timedelta(seconds=1)
    )
    state.codex_oauth_refresh_schedule_by_label["account1"] = (
        loop.OAuthRefreshScheduleState(next_refresh_check_at=_iso(wall_now))
    )
    state.codex_oauth_refresh_schedule_by_label["account2"] = (
        loop.OAuthRefreshScheduleState(
            next_refresh_check_at=_iso(wall_now + timedelta(seconds=1))
        )
    )
    config = _scheduler_config(
        codex_oauth_refresh_enabled=True,
        xai_oauth_refresh_enabled=True,
        kimi_oauth_refresh_enabled=True,
        nous_oauth_refresh_enabled=True,
        cursor_agent_auth_refresh_enabled=True,
    )
    calls: list[str] = []

    def fake_runner(name):
        def run(_config, _state, *, now_monotonic, now_wall=None):
            calls.append(name)
            assert now_monotonic == 101.0
            assert now_wall == wall_now
            return {"event": name}

        return run

    for name in (
        "grok_oidc_refresh",
        "codex_oauth_refresh",
        "xai_oauth_refresh",
        "kimi_oauth_refresh",
        "nous_oauth_refresh",
        "cursor_agent_auth_refresh",
    ):
        monkeypatch.setattr(loop, f"_run_{name}_task", fake_runner(name))

    assert (
        loop._run_due_managed_refresh_tasks(
            config,
            state,
            now=100.0,
            wall_now=wall_now - timedelta(seconds=1),
        )
        == []
    )
    events = loop._run_due_managed_refresh_tasks(
        config,
        state,
        now=101.0,
        wall_now=wall_now,
    )

    assert calls == [
        "grok_oidc_refresh",
        "codex_oauth_refresh",
        "xai_oauth_refresh",
    ]
    assert [event["event"] for event in events] == calls
