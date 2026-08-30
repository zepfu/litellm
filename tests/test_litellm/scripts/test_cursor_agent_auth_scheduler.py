"""Focused CURSOR-012 coverage for the Cursor auth scheduler integration."""

from __future__ import annotations

from argparse import Namespace
import base64
import importlib.util
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

import pytest


_REPO_ROOT = Path(__file__).resolve().parents[3]
_SCRIPT = _REPO_ROOT / "scripts" / "run_provider_status_observations_loop.py"


def _load() -> ModuleType:
    name = "run_provider_status_observations_loop_cursor_012"
    if name in sys.modules:
        del sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def loop() -> ModuleType:
    return _load()


def _jwt(expires_at: int) -> str:
    def encode(value: dict[str, object]) -> str:
        raw = json.dumps(value, separators=(",", ":")).encode("utf-8")
        return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")

    return f"{encode({'alg': 'none'})}.{encode({'exp': expires_at})}.signature"


def _config(loop: ModuleType, tmp_path: Path, **overrides):
    from dataclasses import replace

    config = loop.ProviderStatusLoopConfig(
        apply=False,
        dsn=None,
        environment="test",
        interval_seconds=600.0,
        timeout=2.0,
        ping_count=1,
        ping_timeout=2,
        skip_icmp=True,
        once=True,
        setup_schema=False,
        db_lock_timeout_ms=1000,
        db_statement_timeout_ms=5000,
        cursor_agent_auth_refresh_enabled=True,
        cursor_agent_auth_file=str(tmp_path / "auth.json"),
        cursor_agent_auth_lock_file=str(tmp_path / "auth.json.lock"),
        cursor_agent_auth_refresh_interval_seconds=300.0,
        cursor_agent_auth_refresh_buffer_seconds=300,
        cursor_agent_auth_force_refresh=False,
        cursor_agent_auth_http_timeout_seconds=17.5,
        cursor_agent_usage_poll_enabled=False,
    )
    if overrides:
        config = replace(config, **overrides)
    return config


def _write_auth(path: Path, *, access_expires_at: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "accessToken": _jwt(access_expires_at),
                "refreshToken": "refresh-secret-value",
                "apiKey": "api-secret-value",
            }
        ),
        encoding="utf-8",
    )


def _iso(value: datetime) -> str:
    return value.isoformat().replace("+00:00", "Z")


class _Response:
    def __init__(self, payload: dict[str, object], *, status: int = 200) -> None:
        self._body = json.dumps(payload).encode("utf-8")
        self.status = status

    def __enter__(self):
        return self

    def __exit__(self, *_args) -> None:
        return None

    def getcode(self) -> int:
        return self.status

    def read(self) -> bytes:
        return self._body


def test_cursor_agent_auth_scheduler_defaults_are_exact(loop, monkeypatch) -> None:
    for name in (
        "AAWM_CURSOR_AGENT_AUTH_REFRESH_ENABLED",
        "AAWM_CURSOR_AGENT_AUTH_FILE",
        "AAWM_CURSOR_AGENT_AUTH_LOCK_FILE",
        "AAWM_CURSOR_AGENT_AUTH_FILE_MODE",
        "AAWM_CURSOR_AGENT_AUTH_FILE_UID",
        "AAWM_CURSOR_AGENT_AUTH_FILE_GID",
        "AAWM_CURSOR_AGENT_AUTH_HTTP_TIMEOUT_SECONDS",
        "AAWM_CURSOR_AGENT_AUTH_REFRESH_INTERVAL_SECONDS",
        "AAWM_CURSOR_AGENT_AUTH_REFRESH_BUFFER_SECONDS",
        "AAWM_CURSOR_AGENT_AUTH_FORCE_REFRESH",
    ):
        monkeypatch.delenv(name, raising=False)

    config = loop.parse_config([])

    assert config.cursor_agent_auth_refresh_enabled is False
    assert config.cursor_agent_auth_file == (
        "/home/zepfu/.config/cursor/auth.json"
    )
    assert config.cursor_agent_auth_file_source == "default"
    assert config.cursor_agent_auth_lock_file == (
        "/home/zepfu/.config/cursor/auth.json.lock"
    )
    assert config.cursor_agent_auth_refresh_interval_seconds == 300.0
    assert config.cursor_agent_auth_refresh_buffer_seconds == 300
    assert config.cursor_agent_auth_force_refresh is False
    assert config.cursor_agent_auth_http_timeout_seconds == 30.0
    assert loop.run_due_sidecar_tasks(
        config,
        loop.SidecarTaskState(),
        now_monotonic=1.0,
    ) == []
    assert loop._sidecar_one_shot_policy(
        {"event": "cursor_agent_auth_refresh"}
    ) == "required"


def test_cursor_agent_auth_validation_uses_dedicated_refresh_interval(
    loop,
) -> None:
    args = Namespace(
        cursor_agent_auth_refresh_enabled=True,
        cursor_agent_auth_force_refresh=False,
        cursor_agent_auth_refresh_interval_seconds=300.0,
        cursor_agent_auth_refresh_buffer_seconds=300,
        cursor_agent_auth_http_timeout_seconds=30.0,
        interval_seconds=1_200.0,
    )

    loop._validate_cursor_agent_auth_config_args(args)

    args.cursor_agent_auth_refresh_interval_seconds = 901.0
    with pytest.raises(
        SystemExit,
        match=(
            r"--cursor-agent-auth-refresh-interval-seconds=901 exceeds "
            r"--cursor-agent-auth-refresh-buffer-seconds=300; "
            r"outer eligibility cadence must not exceed the minimum refresh threshold"
        ),
    ):
        loop._validate_cursor_agent_auth_config_args(args)


def test_cursor_agent_auth_scheduler_disabled_gate_has_no_side_effects(
    loop,
    tmp_path: Path,
) -> None:
    config = _config(
        loop,
        tmp_path,
        cursor_agent_auth_refresh_enabled=False,
    )

    with patch.object(
        loop.cursor_agent_auth_refresh,
        "inspect_cursor_agent_auth_refresh_eligibility",
        side_effect=AssertionError("disabled task must not inspect"),
    ), patch.object(
        loop.cursor_agent_auth_refresh,
        "refresh_cursor_agent_auth_file",
        side_effect=AssertionError("disabled task must not refresh"),
    ):
        assert loop._run_cursor_agent_auth_refresh_task(
            config,
            loop.SidecarTaskState(),
            now_monotonic=100.0,
        ) is None


def test_cursor_agent_auth_refreshes_at_boundary_and_wakes_independently(
    loop,
    tmp_path: Path,
    monkeypatch,
) -> None:
    wall_now = datetime(2026, 8, 28, 12, 0, tzinfo=timezone.utc)
    auth_path = tmp_path / "auth.json"
    _write_auth(auth_path, access_expires_at=int(wall_now.timestamp()) - 10)
    config = _config(
        loop,
        tmp_path,
        interval_seconds=400.0,
    )
    schedule = loop.OAuthRefreshScheduleState(
        next_refresh_check_at=_iso(wall_now),
    )
    state = loop.SidecarTaskState(
        cursor_agent_auth_refresh_schedule=schedule,
    )
    inspected: list[datetime] = []
    refresh_calls: list[dict[str, object]] = []

    def inspect(*_args, now, **_kwargs):
        observed = now()
        inspected.append(observed)
        due = bool(refresh_calls) or observed >= wall_now
        return {
            "eligibility_checked_at": _iso(observed),
            "expires_at": _iso(
                wall_now + timedelta(seconds=300)
                if refresh_calls
                else wall_now - timedelta(seconds=10)
            ),
            "refresh_due_at": _iso(
                wall_now
                if refresh_calls
                else wall_now - timedelta(seconds=10)
            ),
            "next_refresh_check_at": _iso(
                wall_now + timedelta(seconds=300)
                if refresh_calls
                else wall_now
            ),
            "eligible": due,
            "credential_health": "fresh" if refresh_calls else "expired",
            "usable": True,
            "error_class": None,
            "error_message": None,
        }

    def refresh(auth_file, **kwargs):
        refresh_calls.append({"auth_file": auth_file, **kwargs})
        callback = kwargs["on_exchange_attempt"]
        callback()
        return {
            "attempted": True,
            "refreshed": True,
            "skipped": False,
            "credential_shape": "accessToken+refreshToken+apiKey",
            "refresh_capability": "apiKey_exchange",
            "refresh_method": "apiKey_exchange",
            "health_status": "fresh",
            "credential_health": "fresh",
            "usable": True,
            "expires_at": _iso(wall_now + timedelta(hours=2)),
            "error_class": None,
            "error_message": None,
        }

    class _FixedDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return wall_now if tz is not None else wall_now.replace(tzinfo=None)

    monkeypatch.setattr(
        loop.cursor_agent_auth_refresh,
        "inspect_cursor_agent_auth_refresh_eligibility",
        inspect,
    )
    monkeypatch.setattr(
        loop.cursor_agent_auth_refresh,
        "refresh_cursor_agent_auth_file",
        refresh,
    )
    monkeypatch.setattr(loop, "datetime", _FixedDatetime)

    event = loop._run_cursor_agent_auth_refresh_task(
        config,
        state,
        now_monotonic=100.0,
        now_wall=wall_now,
    )

    assert event is not None
    assert event["event"] == "cursor_agent_auth_refresh"
    assert event["actual_attempted"] is True
    assert event["refreshed"] is True
    assert event["health_status"] == "fresh"
    assert event["credential_health"] == "fresh"
    assert event["usable"] is True
    assert event["refresh_method"] == "apiKey_exchange"
    assert event["eligibility_cadence_seconds"] == 300.0
    assert event["refresh_attempt_interval_seconds"] == 300.0
    assert len(inspected) == 2
    assert inspected[0] == wall_now
    assert refresh_calls == [
        {
            "auth_file": str(auth_path),
            "buffer_seconds": 300,
            "force": False,
            "lock_file": str(tmp_path / "auth.json.lock"),
            "http_timeout_seconds": 17.5,
            "on_exchange_attempt": refresh_calls[0]["on_exchange_attempt"],
        }
    ]

    # The next sidecar wake is the Cursor deadline, not the generic 400-second
    # provider cycle; the task's own refresh interval remains independently 300.
    wake_delay = loop._next_sidecar_wake_delay(
        config,
        state,
        now=100.0,
    )
    assert config.interval_seconds == 400.0
    assert state.cursor_agent_auth_last_attempt_monotonic == 100.0
    assert wake_delay == 300.0


def test_cursor_agent_auth_real_eligibility_refreshes_at_expiry_minus_buffer(
    loop,
    tmp_path: Path,
    monkeypatch,
) -> None:
    from dataclasses import replace

    wall_start = datetime(2026, 8, 28, 12, 0, tzinfo=timezone.utc)
    refresh_at = wall_start + timedelta(seconds=300)
    expires_at = refresh_at + timedelta(seconds=300)
    auth_path = tmp_path / "auth.json"
    _write_auth(auth_path, access_expires_at=int(expires_at.timestamp()))
    config = replace(
        _config(loop, tmp_path),
        once=False,
    )
    request_details: dict[str, object] = {}

    def fake_urlopen(request, timeout=None):
        request_details["url"] = request.full_url
        request_details["method"] = request.get_method()
        request_details["timeout"] = timeout
        return _Response(
            {
                "accessToken": _jwt(
                    int((refresh_at + timedelta(hours=1)).timestamp())
                ),
                "refreshToken": "rotated-refresh-secret",
            }
        )

    monkeypatch.setattr(
        loop.cursor_agent_auth_refresh.time,
        "time",
        lambda: refresh_at.timestamp(),
    )
    monkeypatch.setattr(
        loop.cursor_agent_auth_refresh.urllib_request,
        "urlopen",
        fake_urlopen,
    )

    event = loop._run_cursor_agent_auth_refresh_task(
        config,
        loop.SidecarTaskState(),
        now_monotonic=100.0,
        now_wall=refresh_at,
    )

    assert request_details == {
        "url": "https://api2.cursor.sh/auth/exchange_user_api_key",
        "method": "POST",
        "timeout": 17.5,
    }
    assert event is not None
    assert event["pre_expires_at"] == _iso(expires_at)
    assert event["pre_refresh_due_at"] == _iso(refresh_at)
    assert event["actual_attempted"] is True
    assert event["refreshed"] is True
    assert event["health_status"] == "fresh"
    assert event["credential_health"] == "fresh"
    assert event["usable"] is True
    assert event["auth_observation_status"] == "fresh"
    assert event["credential_fingerprint"]
    assert event["previous_credential_fingerprint"]
    assert (
        event["credential_fingerprint"]
        != event["previous_credential_fingerprint"]
    )
    observation = loop._build_cursor_agent_auth_observation(config, event)
    assert (
        observation["metadata"]["credential_fingerprint"]
        == event["credential_fingerprint"]
    )
    assert (
        observation["metadata"]["previous_credential_fingerprint"]
        == event["previous_credential_fingerprint"]
    )
    serialized = json.dumps(
        {"event": event, "observation": observation},
        sort_keys=True,
        default=str,
    )
    for value in (
        "refresh-secret-value",
        "api-secret-value",
        "rotated-refresh-secret",
    ):
        assert value not in serialized


def test_main_retains_generic_deadline_across_cursor_wake(
    loop,
    tmp_path: Path,
    monkeypatch,
) -> None:
    from dataclasses import replace

    config = replace(
        _config(loop, tmp_path),
        interval_seconds=600.0,
        once=False,
    )
    wall_start = datetime(2026, 8, 28, 12, 0, tzinfo=timezone.utc)
    clock = {"monotonic": 0.0}
    generic_calls: list[float] = []
    cursor_calls: list[float] = []
    sidecar_calls: list[float] = []
    emitted: list[dict[str, object]] = []
    sleep_calls: list[tuple[float, float]] = []
    handlers: dict[int, object] = {}

    class _ClockDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            value = wall_start + timedelta(seconds=clock["monotonic"])
            return value if tz is not None else value.replace(tzinfo=None)

    def fake_cycle(_config):
        generic_calls.append(clock["monotonic"])
        return {"event": "provider_status_observations_cycle"}

    def fake_run_due_sidecar_tasks(_config, state):
        now = clock["monotonic"]
        sidecar_calls.append(now)
        if now == 0.0:
            state.cursor_agent_auth_refresh_schedule.next_refresh_check_at = _iso(
                wall_start + timedelta(seconds=300)
            )
        elif now == 600.0:
            assert state.next_generic_cycle_due_monotonic == 1_200.0
            handlers[loop.signal.SIGTERM](loop.signal.SIGTERM, None)
        return []

    def fake_cursor_refresh(_config, state, *, now_monotonic, now_wall=None):
        cursor_calls.append(now_monotonic)
        assert now_wall == wall_start + timedelta(seconds=300)
        assert state.next_generic_cycle_due_monotonic == 600.0
        state.cursor_agent_auth_refresh_schedule.next_refresh_check_at = _iso(
            wall_start + timedelta(seconds=900)
        )
        return {
            "event": "cursor_agent_auth_refresh",
            "refreshed": True,
            "skipped": False,
        }

    def fake_sleep(seconds: float) -> None:
        sleep_calls.append((clock["monotonic"], seconds))
        if clock["monotonic"] == 0.0:
            clock["monotonic"] = 300.0
        elif clock["monotonic"] == 300.0:
            clock["monotonic"] = 600.0
        else:
            pytest.fail(f"unexpected sleep at {clock['monotonic']}")

    monkeypatch.setattr(loop, "parse_config", lambda _argv: config)
    monkeypatch.setattr(loop, "validate_runtime_guardrails", lambda _config: None)
    monkeypatch.setattr(loop, "datetime", _ClockDatetime)
    monkeypatch.setattr(loop.time, "monotonic", lambda: clock["monotonic"])
    monkeypatch.setattr(loop.time, "sleep", fake_sleep)
    monkeypatch.setattr(loop.signal, "signal", handlers.__setitem__)
    monkeypatch.setattr(loop, "run_cycle", fake_cycle)
    monkeypatch.setattr(
        loop,
        "run_due_sidecar_tasks",
        fake_run_due_sidecar_tasks,
    )
    monkeypatch.setattr(
        loop,
        "_run_cursor_agent_auth_refresh_task",
        fake_cursor_refresh,
    )
    monkeypatch.setattr(loop, "_emit", emitted.append)

    assert loop.main([]) == 0
    assert generic_calls == [0.0, 600.0]
    assert cursor_calls == [300.0]
    assert sidecar_calls == [0.0, 600.0]
    assert sleep_calls == [(0.0, 1.0), (300.0, 1.0)]
    assert emitted[-1]["event"] == "provider_status_observations_stopped"


def test_cursor_agent_auth_boundary_adjacent_check_skips_refresh(
    loop,
    tmp_path: Path,
    monkeypatch,
) -> None:
    wall_now = datetime(2026, 8, 28, 12, 0, tzinfo=timezone.utc)
    due_at = wall_now + timedelta(seconds=1)
    auth_path = tmp_path / "auth.json"
    _write_auth(auth_path, access_expires_at=int(due_at.timestamp()) + 1)
    config = _config(loop, tmp_path)
    state = loop.SidecarTaskState(
        cursor_agent_auth_refresh_schedule=loop.OAuthRefreshScheduleState(
            next_refresh_check_at=_iso(due_at),
        ),
    )
    inspected: list[datetime] = []

    def inspect(*_args, now, **_kwargs):
        observed = now()
        inspected.append(observed)
        return {
            "eligibility_checked_at": _iso(observed),
            "expires_at": _iso(due_at + timedelta(seconds=1)),
            "refresh_due_at": _iso(due_at),
            "next_refresh_check_at": _iso(due_at),
            "eligible": False,
            "credential_health": "fresh",
            "usable": True,
            "error_class": None,
            "error_message": None,
        }

    monkeypatch.setattr(
        loop.cursor_agent_auth_refresh,
        "inspect_cursor_agent_auth_refresh_eligibility",
        inspect,
    )
    with patch.object(
        loop.cursor_agent_auth_refresh,
        "refresh_cursor_agent_auth_file",
        side_effect=AssertionError("one second before the boundary must not refresh"),
    ):
        event = loop._run_cursor_agent_auth_refresh_task(
            config,
            state,
            now_monotonic=100.0,
            now_wall=due_at - timedelta(seconds=1),
        )

    assert event is not None
    assert event["refresh_result_class"] == "refresh_not_due"
    assert event["actual_attempted"] is False
    assert event["skipped"] is True
    assert inspected == [due_at - timedelta(seconds=1)]
    # The deadline is translated against the host clock; the persisted wake
    # contract is "now", not a stale outer-cycle interval.
    assert loop._next_sidecar_wake_delay(config, state, now=100.0) == 0.0


def test_cursor_agent_auth_event_is_sanitized_and_usage_poll_stays_disabled(
    loop,
    tmp_path: Path,
) -> None:
    wall_now = datetime(2026, 8, 28, 12, 0, tzinfo=timezone.utc)
    auth_path = tmp_path / "auth.json"
    _write_auth(auth_path, access_expires_at=int(wall_now.timestamp()) + 3_600)
    config = _config(loop, tmp_path)

    event = loop._run_cursor_agent_auth_refresh_task(
        config,
        loop.SidecarTaskState(),
        now_monotonic=100.0,
        now_wall=wall_now,
    )

    assert event is not None
    assert event["credential_fingerprint"]
    assert event["previous_credential_fingerprint"] is None
    observation = loop._build_cursor_agent_auth_observation(config, event)
    assert (
        observation["metadata"]["credential_fingerprint"]
        == event["credential_fingerprint"]
    )
    assert observation["metadata"]["previous_credential_fingerprint"] is None
    serialized = json.dumps(
        {"event": event, "observation": observation},
        default=str,
    )
    for value in ("refresh-secret-value", "api-secret-value", _jwt(0)):
        assert value not in serialized
    assert event["attempted"] is False
    assert event["skipped"] is True
    assert event["health_status"] == "fresh"
    assert event["credential_health"] == "fresh"
    assert event["usable"] is True
    assert config.cursor_agent_usage_poll_enabled is False
    assert state_usage_is_untouched(loop, config)


def state_usage_is_untouched(loop: ModuleType, config) -> bool:
    state = loop.SidecarTaskState()
    loop._run_cursor_agent_usage_poll_task(
        config,
        state,
        now_monotonic=100.0,
    )
    return state.cursor_agent_usage_last_attempt_monotonic is None
