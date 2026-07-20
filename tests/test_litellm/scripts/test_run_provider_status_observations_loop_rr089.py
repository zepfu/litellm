"""RR-089: append-safe, deduplicated observability anomaly error JSONL intake."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType

import pytest

_REPO = Path(__file__).resolve().parents[3]
_SCRIPT = _REPO / "scripts" / "run_provider_status_observations_loop.py"


def _load() -> ModuleType:
    name = "run_provider_status_observations_loop_rr089"
    # Always reload so tests see the current script body after review fixes.
    if name in sys.modules:
        del sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, _SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def loop() -> ModuleType:
    return _load()


def _config(loop: ModuleType, tmp_path: Path, **overrides):
    from dataclasses import replace

    base = loop.ProviderStatusLoopConfig(
        apply=False,
        dsn=None,
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
        observability_anomaly_scan_enabled=True,
        observability_anomaly_scan_error_log_dir=str(tmp_path),
    )
    if overrides:
        base = replace(base, **overrides)
    return base


def _anomaly(**overrides):
    payload = {
        "anomaly_class": "missing_repository_for_agent_context",
        "expected": "repository should be derivable",
        "row_count": 2,
        "examples": [
            {
                "row_id": 123,
                "session_id": "session-123",
                "model": "grok-composer-2.5-fast",
                "client_name": "Grok",
            }
        ],
    }
    payload.update(overrides)
    return payload


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


class _KimiUsageResponse:
    def __init__(self, body: bytes, *, status: int = 200) -> None:
        self.body = body
        self.status = status

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def getcode(self):
        return self.status

    def read(self):
        return self.body


def _kimi_usage_http_error(loop, status_code: int):
    return loop.urllib_error.HTTPError(
        loop.DEFAULT_KIMI_USAGE_URL,
        status_code,
        "sanitized test error",
        {},
        None,
    )


def test_rr089_standing_anomaly_is_not_reappended(loop, tmp_path) -> None:
    config = _config(loop, tmp_path)
    observed = datetime(2026, 7, 17, 12, 0, tzinfo=timezone.utc)

    first_written, path = loop._write_observability_anomaly_error_records(
        config,
        observed_at=observed,
        anomalies=[_anomaly()],
    )
    second_written, path2 = loop._write_observability_anomaly_error_records(
        config,
        observed_at=datetime(2026, 7, 17, 13, 0, tzinfo=timezone.utc),
        anomalies=[_anomaly()],
    )

    assert path == path2 == tmp_path / "dev-error.jsonl"
    assert first_written == 1
    assert second_written == 0
    records = _read_jsonl(path)
    assert len(records) == 1
    assert records[0]["anomaly_class"] == "missing_repository_for_agent_context"
    assert records[0]["row_count"] == 2
    # Standing scan keeps the original observed_at rather than hourly churn.
    assert records[0]["observed_at"].startswith("2026-07-17T12:00:00")


def test_rr089_changed_standing_anomaly_appends_fresh_row(loop, tmp_path) -> None:
    """Material changes append; prior unresolved intake stays readable."""
    config = _config(loop, tmp_path)
    path = tmp_path / "dev-error.jsonl"

    loop._write_observability_anomaly_error_records(
        config,
        observed_at=datetime(2026, 7, 17, 12, 0, tzinfo=timezone.utc),
        anomalies=[_anomaly(row_count=2)],
    )
    written, _ = loop._write_observability_anomaly_error_records(
        config,
        observed_at=datetime(2026, 7, 17, 14, 0, tzinfo=timezone.utc),
        anomalies=[
            _anomaly(
                row_count=5,
                examples=[{"row_id": 999, "session_id": "session-999"}],
            )
        ],
    )

    assert written == 1
    records = _read_jsonl(path)
    assert len(records) == 2
    assert records[0]["row_count"] == 2
    assert records[0]["observed_at"].startswith("2026-07-17T12:00:00")
    assert records[1]["row_count"] == 5
    assert records[1]["examples"][0]["row_id"] == 999
    assert records[1]["observed_at"].startswith("2026-07-17T14:00:00")
    # Latest identity still suppresses identical re-appends.
    written_again, _ = loop._write_observability_anomaly_error_records(
        config,
        observed_at=datetime(2026, 7, 17, 15, 0, tzinfo=timezone.utc),
        anomalies=[
            _anomaly(
                row_count=5,
                examples=[{"row_id": 999, "session_id": "session-999"}],
            )
        ],
    )
    assert written_again == 0
    assert len(_read_jsonl(path)) == 2


def test_rr089_preserves_unrelated_intake_records(loop, tmp_path) -> None:
    config = _config(loop, tmp_path)
    path = tmp_path / "dev-error.jsonl"
    unrelated = {
        "event": "SomeOtherError",
        "environment": "dev",
        "message": "unrelated durable intake",
        "fingerprint": "abc123",
    }
    path.write_text(json.dumps(unrelated) + "\n", encoding="utf-8")

    written, _ = loop._write_observability_anomaly_error_records(
        config,
        observed_at=datetime(2026, 7, 17, 12, 0, tzinfo=timezone.utc),
        anomalies=[_anomaly()],
    )
    written_again, _ = loop._write_observability_anomaly_error_records(
        config,
        observed_at=datetime(2026, 7, 17, 13, 0, tzinfo=timezone.utc),
        anomalies=[_anomaly()],
    )

    assert written == 1
    assert written_again == 0
    records = _read_jsonl(path)
    assert len(records) == 2
    assert records[0] == unrelated
    assert records[1]["event"] == "aawm_observability_anomaly"
    assert records[1]["anomaly_class"] == "missing_repository_for_agent_context"


def test_rr089_does_not_delete_unresolved_intake_on_noop(loop, tmp_path) -> None:
    config = _config(loop, tmp_path)
    path = tmp_path / "dev-error.jsonl"
    seed = {
        "event": "aawm_observability_anomaly",
        "environment": "dev",
        "anomaly_class": "missing_repository_for_agent_context",
        "anomaly_source": "provider_status_observations_sidecar",
        "row_count": 2,
        "expected": "repository should be derivable",
        "lookback_hours": 4.0,
        "examples": [
            {
                "row_id": 123,
                "session_id": "session-123",
                "model": "grok-composer-2.5-fast",
                "client_name": "Grok",
            }
        ],
        "observed_at": "2026-07-17T11:00:00Z",
    }
    path.write_text(json.dumps(seed, sort_keys=True) + "\n", encoding="utf-8")
    before = path.read_text(encoding="utf-8")

    written, _ = loop._write_observability_anomaly_error_records(
        config,
        observed_at=datetime(2026, 7, 17, 15, 0, tzinfo=timezone.utc),
        anomalies=[_anomaly()],
    )

    assert written == 0
    assert path.exists()
    assert path.read_text(encoding="utf-8") == before


def test_rr089_standing_dedupe_uses_latest_historical_identity(loop, tmp_path) -> None:
    """Pre-RR-089 append growth is not re-appended when content is still standing."""
    config = _config(loop, tmp_path)
    path = tmp_path / "dev-error.jsonl"
    unrelated = {
        "event": "SomeOtherError",
        "environment": "dev",
        "message": "keep unrelated intake",
        "fingerprint": "preserve-me",
    }
    older = {
        "event": "aawm_observability_anomaly",
        "environment": "dev",
        "anomaly_class": "missing_repository_for_agent_context",
        "anomaly_source": "provider_status_observations_sidecar",
        "row_count": 2,
        "expected": "repository should be derivable",
        "lookback_hours": 4.0,
        "examples": [
            {
                "row_id": 123,
                "session_id": "session-123",
                "model": "grok-composer-2.5-fast",
                "client_name": "Grok",
            }
        ],
        "observed_at": "2026-07-17T10:00:00Z",
    }
    newer_duplicate = {
        **older,
        "observed_at": "2026-07-17T11:00:00Z",
    }
    path.write_text(
        "\n".join(json.dumps(row) for row in (unrelated, older, newer_duplicate))
        + "\n",
        encoding="utf-8",
    )
    before = path.read_text(encoding="utf-8")

    # Standing identical content must not rewrite or collapse the shared file.
    written, _ = loop._write_observability_anomaly_error_records(
        config,
        observed_at=datetime(2026, 7, 17, 15, 0, tzinfo=timezone.utc),
        anomalies=[_anomaly()],
    )

    assert written == 0
    assert path.read_text(encoding="utf-8") == before
    records = _read_jsonl(path)
    assert len(records) == 3
    assert records[0] == unrelated


def test_rr089_projected_size_refusal_preserves_unresolved_intake(
    loop, tmp_path, monkeypatch
) -> None:
    config = _config(loop, tmp_path)
    path = tmp_path / "dev-error.jsonl"
    monkeypatch.setenv("LITELLM_AAWM_ERROR_LOG_MAX_BYTES", "200")
    # backup_count=0 must not delete active unresolved intake on this writer.
    monkeypatch.setenv("LITELLM_AAWM_ERROR_LOG_BACKUP_COUNT", "0")

    bloated = {
        "event": "SomeOtherError",
        "environment": "dev",
        "message": "x" * 400,
        "fingerprint": "keep-me",
    }
    path.write_text(json.dumps(bloated) + "\n", encoding="utf-8")
    before = path.read_text(encoding="utf-8")
    before_size = path.stat().st_size

    written, _ = loop._write_observability_anomaly_error_records(
        config,
        observed_at=datetime(2026, 7, 17, 12, 0, tzinfo=timezone.utc),
        anomalies=[_anomaly()],
    )

    assert written == 0
    assert path.exists()
    assert path.read_text(encoding="utf-8") == before
    assert path.stat().st_size == before_size
    assert not Path(f"{path}.1").exists()
    assert not any(tmp_path.glob(".*tmp*"))


def test_rr089_distinct_anomaly_classes_are_not_collapsed(loop, tmp_path) -> None:
    config = _config(loop, tmp_path)
    path = tmp_path / "dev-error.jsonl"

    written, _ = loop._write_observability_anomaly_error_records(
        config,
        observed_at=datetime(2026, 7, 17, 12, 0, tzinfo=timezone.utc),
        anomalies=[
            _anomaly(anomaly_class="missing_repository_for_agent_context"),
            _anomaly(
                anomaly_class="rate_limit_observation_gap",
                row_count=3,
                examples=[{"provider": "openai", "quota_key": "requests"}],
            ),
        ],
    )
    written_again, _ = loop._write_observability_anomaly_error_records(
        config,
        observed_at=datetime(2026, 7, 17, 13, 0, tzinfo=timezone.utc),
        anomalies=[
            _anomaly(anomaly_class="missing_repository_for_agent_context"),
            _anomaly(
                anomaly_class="rate_limit_observation_gap",
                row_count=3,
                examples=[{"provider": "openai", "quota_key": "requests"}],
            ),
        ],
    )

    assert written == 2
    assert written_again == 0
    records = _read_jsonl(path)
    assert len(records) == 2
    classes = {rec["anomaly_class"] for rec in records}
    assert classes == {
        "missing_repository_for_agent_context",
        "rate_limit_observation_gap",
    }


def test_rr089_concurrent_generic_append_is_not_lost(loop, tmp_path) -> None:
    """Shared active JSONL is never rewritten; concurrent appends stay durable."""
    config = _config(loop, tmp_path)
    path = tmp_path / "dev-error.jsonl"
    generic = {
        "event": "generic_error_from_other_writer",
        "environment": "dev",
        "fingerprint": "concurrent-generic",
        "message": "must survive sidecar anomaly write",
    }

    # Seed one anomaly first so the second write is a material change append.
    loop._write_observability_anomaly_error_records(
        config,
        observed_at=datetime(2026, 7, 17, 12, 0, tzinfo=timezone.utc),
        anomalies=[_anomaly(row_count=1)],
    )

    barrier = threading.Barrier(2)
    errors: list[BaseException] = []

    def generic_writer() -> None:
        try:
            barrier.wait(timeout=5)
            flags = os.O_APPEND | os.O_CREAT | os.O_WRONLY
            fd = os.open(path, flags, 0o644)
            try:
                os.write(fd, (json.dumps(generic) + "\n").encode("utf-8"))
            finally:
                os.close(fd)
        except BaseException as exc:  # pragma: no cover - surfaced below
            errors.append(exc)

    def anomaly_writer() -> None:
        try:
            barrier.wait(timeout=5)
            loop._write_observability_anomaly_error_records(
                config,
                observed_at=datetime(2026, 7, 17, 12, 5, tzinfo=timezone.utc),
                anomalies=[_anomaly(row_count=9)],
            )
        except BaseException as exc:  # pragma: no cover - surfaced below
            errors.append(exc)

    threads = [
        threading.Thread(target=generic_writer),
        threading.Thread(target=anomaly_writer),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)
        assert not thread.is_alive()

    assert not errors
    records = _read_jsonl(path)
    fingerprints = {rec.get("fingerprint") for rec in records}
    assert "concurrent-generic" in fingerprints
    anomaly_rows = [
        rec for rec in records if rec.get("event") == "aawm_observability_anomaly"
    ]
    assert any(rec.get("row_count") == 1 for rec in anomaly_rows)
    assert any(rec.get("row_count") == 9 for rec in anomaly_rows)
    # No rewrite temps / backups from this writer path.
    assert not list(tmp_path.glob(".*tmp*"))
    assert not list(tmp_path.glob("*.1"))


def test_rr089_writer_never_uses_replace_or_unlink_on_active_jsonl(
    loop, tmp_path, monkeypatch
) -> None:
    config = _config(loop, tmp_path)
    path = tmp_path / "dev-error.jsonl"
    path.write_text(
        json.dumps({"event": "preexisting", "fingerprint": "stay"}) + "\n",
        encoding="utf-8",
    )
    before_inode = path.stat().st_ino
    replace_calls: list[tuple] = []

    real_replace = loop.os.replace

    def tracking_replace(src, dst, *args, **kwargs):
        replace_calls.append((src, dst))
        # Fail if the shared active intake is the replace destination.
        if Path(dst).resolve() == path.resolve():
            raise AssertionError("os.replace must not rewrite shared intake")
        return real_replace(src, dst, *args, **kwargs)

    monkeypatch.setattr(loop.os, "replace", tracking_replace)

    written, _ = loop._write_observability_anomaly_error_records(
        config,
        observed_at=datetime(2026, 7, 17, 12, 0, tzinfo=timezone.utc),
        anomalies=[_anomaly()],
    )
    assert written == 1
    assert path.exists()
    assert path.stat().st_ino == before_inode
    assert replace_calls == []
    records = _read_jsonl(path)
    assert records[0]["fingerprint"] == "stay"
    assert records[1]["event"] == "aawm_observability_anomaly"


def test_rr089_sidecar_scan_reports_zero_writes_for_standing_anomaly(
    loop, tmp_path, monkeypatch
) -> None:
    config = _config(loop, tmp_path)
    anomalies = [_anomaly()]
    monkeypatch.setattr(
        loop,
        "_collect_observability_anomalies",
        lambda _config: anomalies,
    )

    state = loop.SidecarTaskState()
    first = loop.run_due_sidecar_tasks(config, state, now_monotonic=100.0)
    second = loop.run_due_sidecar_tasks(
        config,
        state,
        now_monotonic=100.0 + config.observability_anomaly_scan_interval_seconds,
    )

    assert first[0]["event"] == "observability_anomaly_scan"
    assert first[0]["error_log_record_count"] == 1
    assert second[0]["event"] == "observability_anomaly_scan"
    assert second[0]["error_log_record_count"] == 0
    assert len(_read_jsonl(tmp_path / "dev-error.jsonl")) == 1


def test_rr089_kimi_oauth_refresh_is_disabled_by_default(loop, monkeypatch) -> None:
    for name in (
        "AAWM_KIMI_OAUTH_REFRESH_ENABLED",
        "AAWM_KIMI_OAUTH_AUTH_FILE",
        "AAWM_KIMI_OAUTH_LOCK_FILE",
        "AAWM_KIMI_OAUTH_REFRESH_INTERVAL_SECONDS",
        "AAWM_KIMI_OAUTH_FORCE_REFRESH",
        "AAWM_KIMI_OAUTH_HTTP_TIMEOUT_SECONDS",
    ):
        monkeypatch.delenv(name, raising=False)

    config = loop.parse_config([])

    assert config.kimi_oauth_refresh_enabled is False
    assert config.kimi_oauth_auth_file == "~/.kimi-code/credentials/kimi-code.json"
    assert config.kimi_oauth_lock_file == "~/.kimi-code/oauth/kimi-code"
    assert config.kimi_oauth_refresh_interval_seconds == 3600.0
    assert config.kimi_oauth_force_refresh is False
    assert config.kimi_oauth_http_timeout_seconds == 30.0
    assert loop.run_due_sidecar_tasks(config, loop.SidecarTaskState(), now_monotonic=1.0) == []


def test_rr089_kimi_oauth_auth_file_resolves_env_then_cli(loop, monkeypatch) -> None:
    env_auth_file = "~/k-credentials/from-env.json"
    cli_auth_file = "~/k-credentials/from-cli.json"
    monkeypatch.setenv("AAWM_KIMI_OAUTH_AUTH_FILE", env_auth_file)

    env_config = loop.parse_config([])

    assert env_config.kimi_oauth_auth_file == str(Path(env_auth_file).expanduser())
    assert env_config.kimi_oauth_auth_file_source == "AAWM_KIMI_OAUTH_AUTH_FILE"

    monkeypatch.delenv("AAWM_KIMI_OAUTH_AUTH_FILE")
    cli_config = loop.parse_config(["--kimi-oauth-auth-file", cli_auth_file])

    assert cli_config.kimi_oauth_auth_file == str(Path(cli_auth_file).expanduser())
    assert cli_config.kimi_oauth_auth_file_source == "explicit"


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (
            ["--kimi-oauth-refresh-interval-seconds", "0"],
            "--kimi-oauth-refresh-interval-seconds must be greater than 0",
        ),
        (
            ["--kimi-oauth-http-timeout-seconds", "0"],
            "--kimi-oauth-http-timeout-seconds must be greater than 0",
        ),
    ],
)
def test_rr089_kimi_oauth_config_validation(loop, argv, message) -> None:
    with pytest.raises(SystemExit, match=message):
        loop.parse_config(argv)


def test_rr089_kimi_oauth_refresh_invocation_and_event_shape(loop, tmp_path, monkeypatch) -> None:
    config = _config(
        loop,
        tmp_path,
        kimi_oauth_refresh_enabled=True,
        kimi_oauth_auth_file=str(tmp_path / "kimi.json"),
        kimi_oauth_lock_file=str(tmp_path / "kimi.lock"),
        kimi_oauth_refresh_interval_seconds=60.0,
        kimi_oauth_force_refresh=True,
        kimi_oauth_http_timeout_seconds=17.0,
    )
    calls: list[tuple] = []

    def refresh(auth_file, **kwargs):
        calls.append((auth_file, kwargs))
        return {
            "attempted": True,
            "refreshed": True,
            "skipped": False,
            "auth_file": auth_file,
            "scope": "kimi-code",
            "expires_at": "2026-07-19T16:00:00Z",
            "auth_degraded": False,
            "error_class": None,
            "error_message": None,
        }

    monkeypatch.setattr(loop.kimi_oauth_refresh, "refresh_kimi_oauth_auth_file", refresh)

    event = loop._run_kimi_oauth_refresh_task(
        config,
        loop.SidecarTaskState(),
        now_monotonic=100.0,
    )

    assert event is not None
    assert calls == [
        (
            config.kimi_oauth_auth_file,
            {
                "force": True,
                "lock_file": config.kimi_oauth_lock_file,
                "http_timeout_seconds": 17.0,
            },
        )
    ]
    assert event["event"] == "kimi_oauth_refresh"
    assert event["environment"] == "dev"
    assert event["attempted"] is True
    assert event["refreshed"] is True
    assert event["skipped"] is False
    assert "logged_in" not in event
    assert event["auth_observation_status"] == "refreshed"
    assert event["auth_observation_persisted"] is False
    assert event["auth_observation_skip_reason"] == "apply_disabled"
    observation = loop._build_kimi_oauth_auth_observation(config, event)
    assert observation["provider"] == "kimi_code"


def test_rr089_kimi_oauth_refresh_obeys_interval(loop, tmp_path, monkeypatch) -> None:
    config = _config(
        loop,
        tmp_path,
        kimi_oauth_refresh_enabled=True,
        kimi_oauth_refresh_interval_seconds=60.0,
    )
    calls: list[float] = []

    def refresh(*args, **kwargs):
        calls.append(1.0)
        return {
            "attempted": False,
            "refreshed": False,
            "skipped": True,
            "auth_file": config.kimi_oauth_auth_file,
            "scope": "kimi-code",
            "expires_at": None,
            "auth_degraded": False,
            "error_class": None,
            "error_message": None,
        }

    monkeypatch.setattr(loop.kimi_oauth_refresh, "refresh_kimi_oauth_auth_file", refresh)
    state = loop.SidecarTaskState()

    assert loop._run_kimi_oauth_refresh_task(config, state, now_monotonic=100.0)
    assert loop._run_kimi_oauth_refresh_task(config, state, now_monotonic=159.0) is None
    assert loop._run_kimi_oauth_refresh_task(config, state, now_monotonic=160.0)
    assert len(calls) == 2


def test_rr089_kimi_oauth_refresh_event_redacts_tokens(loop, tmp_path, monkeypatch) -> None:
    config = _config(loop, tmp_path, kimi_oauth_refresh_enabled=True)
    secret = "refresh-token-should-never-leak-123456"

    def refresh(*args, **kwargs):
        return {
            "attempted": True,
            "refreshed": False,
            "skipped": False,
            "auth_file": config.kimi_oauth_auth_file,
            "scope": "kimi-code",
            "expires_at": None,
            "auth_degraded": False,
            "error_class": "KimiOAuthError",
            "error_message": f"refresh_token={secret} rejected",
            "access_token": "access-token-should-never-leak",
            "refresh_token": secret,
        }

    monkeypatch.setattr(loop.kimi_oauth_refresh, "refresh_kimi_oauth_auth_file", refresh)

    event = loop._run_kimi_oauth_refresh_task(
        config,
        loop.SidecarTaskState(),
        now_monotonic=100.0,
    )

    assert event is not None
    serialized = json.dumps(event, sort_keys=True)
    assert secret not in serialized
    assert "access-token-should-never-leak" not in serialized
    assert "access_token" not in event
    assert "refresh_token" not in event
    assert "REDACTED" in event["error_message"]


@pytest.fixture
def current_kimi_usage_payload():
    return {
        "authentication": {
            "method": "oauth",
            "scope": ["kimi-code", "usage:read"],
            "accessToken": "authentication-access-token-must-not-persist",
            "xMsh": "authentication-x-msh-must-not-persist",
        },
        "domain": "api.kimi.com",
        "limits": [
            {
                "detail": {
                    "limit": 100,
                    "remaining": 100,
                    "resetTime": "2026-07-19T17:00:00Z",
                },
                "window": {
                    "duration": 300,
                    "timeUnit": "TIME_UNIT_MINUTE",
                },
            },
            {
                "detail": {
                    "limit": 20,
                    "remaining": 10,
                    "resetTime": "2026-07-19T18:00:00Z",
                },
                "window": {
                    "duration": 60,
                    "timeUnit": "TIME_UNIT_MINUTE",
                },
            },
        ],
        "parallel": {"limit": 4},
        "subType": "INDIVIDUAL",
        "totalQuota": {},
        "usage": {
            "limit": 700,
            "remaining": 525,
            "resetTime": "2026-07-26T12:00:00Z",
        },
        "user": {
            "userId": "user-current-123",
            "businessId": "business-current-456",
            "membership": {"level": "PRO"},
            "region": "US",
            "email": "current@example.invalid",
            "displayName": "Current User",
            "deviceId": "device-current-123",
        },
    }


@pytest.fixture
def populated_monthly_kimi_usage_payloads(current_kimi_usage_payload):
    variants = {}
    for name, total_quota in (
        (
            "current",
            {
                "limit": 3000,
                "remaining": 2400,
                "resetTime": "2026-08-01T00:00:00Z",
            },
        ),
        (
            "drifted",
            {
                "quota": "3000",
                "available": "2400",
                "reset_at": "2026-08-01T00:00:00Z",
            },
        ),
    ):
        payload = json.loads(json.dumps(current_kimi_usage_payload))
        payload["totalQuota"] = total_quota
        variants[name] = payload
    return variants


@pytest.fixture
def drifted_kimi_usage_payload():
    return {
        "result": {
            "accountId": "account-drifted-456",
            "quotas": {
                "five_hours": {
                    "total": "120",
                    "consumed": "20",
                    "endsAt": "2026-07-19T18:00:00Z",
                },
                "seven_days": {
                    "capacity": 700,
                    "usage": 42,
                    "nextReset": "2026-07-26T12:00:00Z",
                },
                "monthly_usage": {
                    "quota": 3000,
                    "available": 2800,
                    "periodStart": "2026-07-01T00:00:00Z",
                },
            },
            "parallel": {"max": "8"},
        }
    }


@pytest.fixture
def malformed_kimi_usage_payload():
    return {
        "data": {
            "account": {"id": "account-malformed-789"},
            "usages": {
                "5h": {"limit": "not-a-number", "used": 1},
                "7d": ["not", "an", "object"],
                "monthly": {"reset_at": "not-a-timestamp"},
            },
            "parallel": {"limit": "not-a-number"},
        }
    }


def test_rr089_kimi_usage_poll_is_disabled_by_default(loop, monkeypatch) -> None:
    for name in (
        "AAWM_KIMI_USAGE_POLL_ENABLED",
        "AAWM_KIMI_USAGE_POLL_INTERVAL_SECONDS",
        "AAWM_KIMI_USAGE_POLL_HTTP_TIMEOUT_SECONDS",
    ):
        monkeypatch.delenv(name, raising=False)

    config = loop.parse_config([])

    assert config.kimi_usage_poll_enabled is False
    assert config.kimi_usage_poll_interval_seconds == 3600.0
    assert config.kimi_usage_poll_http_timeout_seconds == 30.0
    assert loop.DEFAULT_KIMI_USAGE_URL == "https://api.kimi.com/coding/v1/usages"
    assert loop.run_due_sidecar_tasks(config, loop.SidecarTaskState(), now_monotonic=1.0) == []


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (
            ["--kimi-usage-poll-interval-seconds", "0"],
            "--kimi-usage-poll-interval-seconds must be greater than 0",
        ),
        (
            ["--kimi-usage-poll-http-timeout-seconds", "0"],
            "--kimi-usage-poll-http-timeout-seconds must be greater than 0",
        ),
    ],
)
def test_rr089_kimi_usage_poll_config_validation(loop, argv, message) -> None:
    with pytest.raises(SystemExit, match=message):
        loop.parse_config(argv)


def test_rr089_kimi_usage_current_fixture_preserves_quota_units_and_redacts(
    loop, tmp_path, current_kimi_usage_payload
) -> None:
    observed_at = datetime(2026, 7, 19, 16, 0, tzinfo=timezone.utc)
    config = _config(loop, tmp_path, kimi_usage_poll_enabled=True)

    payloads, summary = loop._build_kimi_usage_rate_limit_payloads(
        config,
        observed_at=observed_at,
        response_body=current_kimi_usage_payload,
    )

    assert summary == {
        "source_version": "kimi_code_usage_v2",
        "native_contract_version": "kimi_code_0.27.0_managed_usage_v1",
        "window_states": {
            "5h": "valid_zero",
            "7d": "valid_nonzero",
            "monthly": "absent",
        },
        "window_sources": {
            "5h": "native",
            "7d": "native",
            "monthly": "native",
        },
        "malformed_windows": [],
        "parallel_state": "valid_nonzero",
        "parallel_limit_present": True,
        "parallel_row_emitted": False,
        "account_identity_hashed": True,
        "account_identity_fields": ["userId", "businessId"],
        "provider_metadata": {
            "authentication_method": "oauth",
            "authentication_scope": ["kimi-code", "usage:read"],
            "domain": "api.kimi.com",
            "membership_level": "PRO",
            "parallel_limit": 4.0,
            "region": "US",
            "sub_type": "INDIVIDUAL",
        },
        "valid_window_count": 2,
        "parser_path": "native",
        "telemetry_status": "valid",
    }
    assert set(current_kimi_usage_payload) == {
        "authentication",
        "domain",
        "limits",
        "parallel",
        "subType",
        "totalQuota",
        "usage",
        "user",
    }
    assert [payload[6] for payload in payloads] == [
        "kimi_code_5h:quota_units",
        "kimi_code_7d:quota_units",
    ]
    assert [payload[8] for payload in payloads] == [
        "quota_units",
        "quota_units",
    ]
    assert payloads[0][0:14] == (
        observed_at,
        "kimi-code",
        None,
        hashlib.sha256((b"kimi-code-account|userId=user-current-123|" b"businessId=business-current-456")).hexdigest(),
        "kimi_code",
        "kimi-code",
        "kimi_code_5h:quota_units",
        "5h",
        "quota_units",
        datetime(2026, 7, 19, 17, 0, tzinfo=timezone.utc),
        100.0,
        100.0,
        0.0,
        100.0,
    )
    raw_fields = json.loads(payloads[0][16])
    assert raw_fields["window_state"] == "valid_zero"
    assert raw_fields["parser_path"] == "native"
    assert raw_fields["quota_unit"] == "quota_units"
    assert raw_fields["parallel_limit"] == 4.0
    assert raw_fields["provider_metadata"] == summary["provider_metadata"]
    serialized = json.dumps(
        {"payloads": payloads, "summary": summary},
        sort_keys=True,
        default=str,
    )
    for expected in (
        "api.kimi.com",
        "INDIVIDUAL",
        "oauth",
        "kimi-code",
        "usage:read",
        "PRO",
        "US",
    ):
        assert expected in serialized
    for forbidden in (
        "user-current-123",
        "business-current-456",
        "current@example.invalid",
        "Current User",
        "device-current-123",
        "authentication-access-token-must-not-persist",
        "authentication-x-msh-must-not-persist",
    ):
        assert forbidden not in serialized


def test_rr089_kimi_usage_empty_total_quota_is_absent(loop, tmp_path, current_kimi_usage_payload) -> None:
    config = _config(loop, tmp_path, kimi_usage_poll_enabled=True)
    payloads, summary = loop._build_kimi_usage_rate_limit_payloads(
        config,
        observed_at=datetime(2026, 7, 19, 16, 0, tzinfo=timezone.utc),
        response_body=current_kimi_usage_payload,
    )

    assert summary["window_states"]["monthly"] == "absent"
    assert "monthly" not in summary["malformed_windows"]
    assert all(payload[6] != "kimi_code_monthly:quota_units" for payload in payloads)


def test_rr089_kimi_usage_populated_total_quota_supports_current_and_drifted_fields(
    loop, tmp_path, populated_monthly_kimi_usage_payloads
) -> None:
    config = _config(loop, tmp_path, kimi_usage_poll_enabled=True)

    for payload in populated_monthly_kimi_usage_payloads.values():
        rows, summary = loop._build_kimi_usage_rate_limit_payloads(
            config,
            observed_at=datetime(2026, 7, 19, 16, 0, tzinfo=timezone.utc),
            response_body=payload,
        )
        by_key = {row[6]: row for row in rows}
        monthly = by_key["kimi_code_monthly:quota_units"]

        assert summary["window_states"]["monthly"] == "valid_nonzero"
        assert summary["window_sources"]["monthly"] == "native"
        assert monthly[9] == datetime(2026, 8, 1, 0, 0, tzinfo=timezone.utc)
        assert monthly[10:14] == (80.0, 3000.0, 600.0, 2400.0)


def test_rr089_kimi_usage_drifted_fixture_parses_native_window_aliases(
    loop, tmp_path, drifted_kimi_usage_payload
) -> None:
    config = _config(loop, tmp_path, kimi_usage_poll_enabled=True)
    payloads, summary = loop._build_kimi_usage_rate_limit_payloads(
        config,
        observed_at=datetime(2026, 7, 19, 16, 0, tzinfo=timezone.utc),
        response_body=drifted_kimi_usage_payload,
    )

    assert summary["window_states"] == {
        "5h": "valid_nonzero",
        "7d": "valid_nonzero",
        "monthly": "valid_nonzero",
    }
    assert summary["parallel_state"] == "valid_nonzero"
    assert summary["parallel_row_emitted"] is False
    assert summary["parser_path"] == "drift_fallback"
    assert len(payloads) == 3
    by_key = {payload[6]: payload for payload in payloads}
    assert by_key["kimi_code_5h:quota_units"][11:14] == (120.0, 20.0, 100.0)
    assert by_key["kimi_code_7d:quota_units"][11:14] == (700.0, 42.0, 658.0)
    assert by_key["kimi_code_monthly:quota_units"][11:14] == (3000.0, 200.0, 2800.0)
    assert all("parallel" not in payload[6] for payload in payloads)


def test_rr089_kimi_usage_malformed_fixture_emits_no_quota_rows(loop, tmp_path, malformed_kimi_usage_payload) -> None:
    config = _config(loop, tmp_path, kimi_usage_poll_enabled=True)
    payloads, summary = loop._build_kimi_usage_rate_limit_payloads(
        config,
        observed_at=datetime(2026, 7, 19, 16, 0, tzinfo=timezone.utc),
        response_body=malformed_kimi_usage_payload,
    )

    assert payloads == []
    assert summary["window_states"] == {
        "5h": "malformed",
        "7d": "malformed",
        "monthly": "malformed",
    }
    assert summary["parallel_state"] == "malformed"
    assert summary["telemetry_status"] == "malformed"


def test_rr089_kimi_usage_request_reads_existing_credential_without_copying(loop, tmp_path, monkeypatch) -> None:
    credential = tmp_path / "kimi-code.json"
    credential.write_text(
        json.dumps(
            {
                "access_token": "credential-access-token",
                "refresh_token": "credential-refresh-token",
            }
        ),
        encoding="utf-8",
    )
    config = _config(
        loop,
        tmp_path,
        kimi_usage_poll_enabled=True,
        kimi_oauth_auth_file=str(credential),
    )
    requests = []

    class Response:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def getcode(self):
            return self.status

        def read(self):
            return b'{"data":{"account":{"id":"account-1"},"usages":{"5h":{"limit":1}}}}'

    def urlopen(request, timeout):
        requests.append((request, timeout))
        return Response()

    monkeypatch.setattr(loop.urllib_request, "urlopen", urlopen)

    fetched = loop._fetch_kimi_usage_payload(config)

    assert fetched["status_code"] == 200
    assert fetched["attempt_count"] == 1
    assert fetched["retry_count"] == 0
    assert fetched["refresh_attempted"] is False
    assert fetched["refresh_succeeded"] is False
    assert json.dumps(fetched, sort_keys=True).find("credential-access-token") == -1
    request, timeout = requests[0]
    assert request.full_url == "https://api.kimi.com/coding/v1/usages"
    assert request.get_method() == "GET"
    assert request.get_header("Authorization") == "Bearer credential-access-token"
    assert timeout == 30.0
    assert "x-msh" not in {key.lower() for key, _value in request.header_items()}
    assert credential.read_text(encoding="utf-8").count("credential-refresh-token") == 1


def test_rr089_kimi_usage_401_refreshes_same_credential_and_retries_once(loop, tmp_path, monkeypatch) -> None:
    credential = tmp_path / "kimi-code.json"
    credential.write_text(
        json.dumps(
            {
                "access_token": "stale-access-token",
                "refresh_token": "shared-refresh-token",
            }
        ),
        encoding="utf-8",
    )
    lock_file = tmp_path / "kimi-code"
    config = _config(
        loop,
        tmp_path,
        kimi_usage_poll_enabled=True,
        kimi_oauth_auth_file=str(credential),
        kimi_oauth_lock_file=str(lock_file),
        kimi_oauth_http_timeout_seconds=17.0,
    )
    requests = []
    responses = iter(
        (
            _kimi_usage_http_error(loop, 401),
            _KimiUsageResponse(b'{"usage":{"limit":10,"remaining":5}}'),
        )
    )
    refresh_calls = []

    def urlopen(request, timeout):
        requests.append((request.get_header("Authorization"), timeout))
        result = next(responses)
        if isinstance(result, Exception):
            raise result
        return result

    def refresh(auth_file, **kwargs):
        refresh_calls.append((auth_file, kwargs))
        credential.write_text(
            json.dumps(
                {
                    "access_token": "fresh-access-token",
                    "refresh_token": "rotated-refresh-token",
                }
            ),
            encoding="utf-8",
        )
        return {
            "attempted": True,
            "refreshed": True,
            "skipped": False,
            "auth_file": auth_file,
            "error_class": None,
            "error_message": "authorization=must-not-persist",
            "access_token": "refresh-summary-token-must-not-persist",
        }

    monkeypatch.setattr(loop.urllib_request, "urlopen", urlopen)
    monkeypatch.setattr(
        loop.kimi_oauth_refresh,
        "refresh_kimi_oauth_auth_file",
        refresh,
    )

    fetched = loop._fetch_kimi_usage_payload(config)

    assert requests == [
        ("Bearer stale-access-token", 30.0),
        ("Bearer fresh-access-token", 30.0),
    ]
    assert refresh_calls == [
        (
            str(credential),
            {
                "force": True,
                "lock_file": str(lock_file),
                "http_timeout_seconds": 17.0,
            },
        )
    ]
    assert fetched["attempt_count"] == 2
    assert fetched["retry_count"] == 1
    assert fetched["refresh_attempted"] is True
    assert fetched["refresh_succeeded"] is True
    serialized = json.dumps(fetched, sort_keys=True)
    assert "stale-access-token" not in serialized
    assert "fresh-access-token" not in serialized
    assert "refresh-summary-token-must-not-persist" not in serialized
    assert "authorization=must-not-persist" not in serialized


def test_rr089_kimi_usage_refresh_failure_is_sanitized(loop, tmp_path, monkeypatch) -> None:
    credential = tmp_path / "kimi-code.json"
    credential.write_text(
        json.dumps(
            {
                "access_token": "stale-access-token",
                "refresh_token": "shared-refresh-token",
            }
        ),
        encoding="utf-8",
    )
    config = _config(
        loop,
        tmp_path,
        kimi_usage_poll_enabled=True,
        kimi_oauth_auth_file=str(credential),
    )
    monkeypatch.setattr(
        loop.urllib_request,
        "urlopen",
        lambda *args, **kwargs: (_ for _ in ()).throw(_kimi_usage_http_error(loop, 403)),
    )
    monkeypatch.setattr(
        loop.kimi_oauth_refresh,
        "refresh_kimi_oauth_auth_file",
        lambda *args, **kwargs: {
            "attempted": True,
            "refreshed": False,
            "skipped": False,
            "auth_degraded": True,
            "error_class": "KimiOAuthAuthorizationError",
            "error_message": ("refresh_token=raw-refresh-secret " "authorization=Bearer raw-access-secret"),
            "access_token": "raw-access-secret",
            "refresh_token": "raw-refresh-secret",
        },
    )

    event = loop._run_kimi_usage_poll_task(
        config,
        loop.SidecarTaskState(),
        now_monotonic=100.0,
    )

    assert event is not None
    assert event["status_code"] == 403
    assert event["telemetry_class"] == "auth"
    assert event["attempt_count"] == 1
    assert event["retry_count"] == 0
    assert event["refresh_attempted"] is True
    assert event["refresh_succeeded"] is False
    assert event["error_message"] == ("Kimi Code usage credential refresh failed after an authentication " "rejection.")
    serialized = json.dumps(event, sort_keys=True)
    assert "raw-access-secret" not in serialized
    assert "raw-refresh-secret" not in serialized


def test_rr089_kimi_usage_second_401_stops_without_second_refresh(loop, tmp_path, monkeypatch) -> None:
    credential = tmp_path / "kimi-code.json"
    credential.write_text(
        '{"access_token":"stale-token","refresh_token":"refresh-token"}',
        encoding="utf-8",
    )
    config = _config(
        loop,
        tmp_path,
        kimi_usage_poll_enabled=True,
        kimi_oauth_auth_file=str(credential),
    )
    requests = []
    refresh_calls = []

    def urlopen(request, timeout):
        requests.append((request.get_header("Authorization"), timeout))
        raise _kimi_usage_http_error(loop, 401)

    def refresh(*args, **kwargs):
        refresh_calls.append((args, kwargs))
        credential.write_text(
            '{"access_token":"fresh-token","refresh_token":"rotated-token"}',
            encoding="utf-8",
        )
        return {
            "attempted": True,
            "refreshed": True,
            "skipped": False,
            "error_class": None,
        }

    monkeypatch.setattr(loop.urllib_request, "urlopen", urlopen)
    monkeypatch.setattr(
        loop.kimi_oauth_refresh,
        "refresh_kimi_oauth_auth_file",
        refresh,
    )

    with pytest.raises(loop.KimiUsagePollError) as exc_info:
        loop._fetch_kimi_usage_payload(config)

    error = exc_info.value
    assert error.status_code == 401
    assert error.telemetry_class == "auth"
    assert error.attempt_count == 2
    assert error.retry_count == 1
    assert error.refresh_attempted is True
    assert error.refresh_succeeded is True
    assert requests == [
        ("Bearer stale-token", 30.0),
        ("Bearer fresh-token", 30.0),
    ]
    assert len(refresh_calls) == 1


@pytest.mark.parametrize(
    ("status_code", "telemetry_class"),
    [
        (404, "not_found"),
        (429, "rate_limit"),
        (500, "upstream"),
        (503, "upstream"),
    ],
)
def test_rr089_kimi_usage_non_auth_http_errors_do_not_refresh(
    loop, tmp_path, monkeypatch, status_code, telemetry_class
) -> None:
    credential = tmp_path / "kimi-code.json"
    credential.write_text(
        '{"access_token":"current-token","refresh_token":"refresh-token"}',
        encoding="utf-8",
    )
    config = _config(
        loop,
        tmp_path,
        kimi_usage_poll_enabled=True,
        kimi_oauth_auth_file=str(credential),
    )
    monkeypatch.setattr(
        loop.urllib_request,
        "urlopen",
        lambda *args, **kwargs: (_ for _ in ()).throw(_kimi_usage_http_error(loop, status_code)),
    )
    monkeypatch.setattr(
        loop.kimi_oauth_refresh,
        "refresh_kimi_oauth_auth_file",
        lambda *args, **kwargs: pytest.fail("non-auth errors must not refresh"),
    )

    with pytest.raises(loop.KimiUsagePollError) as exc_info:
        loop._fetch_kimi_usage_payload(config)

    error = exc_info.value
    assert error.status_code == status_code
    assert error.telemetry_class == telemetry_class
    assert error.attempt_count == 1
    assert error.retry_count == 0
    assert error.refresh_attempted is False
    assert error.refresh_succeeded is False


def test_rr089_kimi_usage_malformed_response_does_not_refresh(loop, tmp_path, monkeypatch) -> None:
    credential = tmp_path / "kimi-code.json"
    credential.write_text(
        '{"access_token":"current-token","refresh_token":"refresh-token"}',
        encoding="utf-8",
    )
    config = _config(
        loop,
        tmp_path,
        kimi_usage_poll_enabled=True,
        kimi_oauth_auth_file=str(credential),
    )
    monkeypatch.setattr(
        loop.urllib_request,
        "urlopen",
        lambda *args, **kwargs: _KimiUsageResponse(b"not-json"),
    )
    monkeypatch.setattr(
        loop.kimi_oauth_refresh,
        "refresh_kimi_oauth_auth_file",
        lambda *args, **kwargs: pytest.fail("malformed telemetry must not refresh"),
    )

    with pytest.raises(loop.KimiUsagePollError) as exc_info:
        loop._fetch_kimi_usage_payload(config)

    error = exc_info.value
    assert error.telemetry_class == "malformed_telemetry"
    assert error.attempt_count == 1
    assert error.refresh_attempted is False


def test_rr089_kimi_usage_refresh_triggered_poll_does_not_recurse(
    loop, tmp_path, monkeypatch, current_kimi_usage_payload
) -> None:
    credential = tmp_path / "kimi-code.json"
    credential.write_text(
        '{"access_token":"stale-token","refresh_token":"refresh-token"}',
        encoding="utf-8",
    )
    config = _config(
        loop,
        tmp_path,
        kimi_usage_poll_enabled=True,
        kimi_oauth_auth_file=str(credential),
        kimi_usage_poll_interval_seconds=3600.0,
    )
    responses = iter(
        (
            _kimi_usage_http_error(loop, 401),
            _KimiUsageResponse(json.dumps(current_kimi_usage_payload).encode("utf-8")),
        )
    )
    refresh_calls = []

    def urlopen(*args, **kwargs):
        result = next(responses)
        if isinstance(result, Exception):
            raise result
        return result

    def refresh(*args, **kwargs):
        refresh_calls.append((args, kwargs))
        credential.write_text(
            '{"access_token":"fresh-token","refresh_token":"rotated-token"}',
            encoding="utf-8",
        )
        return {
            "attempted": True,
            "refreshed": True,
            "skipped": False,
            "error_class": None,
        }

    monkeypatch.setattr(loop.urllib_request, "urlopen", urlopen)
    monkeypatch.setattr(
        loop.kimi_oauth_refresh,
        "refresh_kimi_oauth_auth_file",
        refresh,
    )

    state = loop.SidecarTaskState(
        kimi_usage_last_attempt_monotonic=99.0,
        kimi_usage_refresh_pending=True,
    )
    event = loop._run_kimi_usage_poll_task(
        config,
        state,
        now_monotonic=100.0,
    )

    assert event is not None
    assert event["trigger"] == "oauth_refresh"
    assert event["attempt_count"] == 2
    assert event["retry_count"] == 1
    assert event["refresh_attempted"] is True
    assert event["refresh_succeeded"] is True
    assert state.kimi_usage_refresh_pending is False
    assert len(refresh_calls) == 1
    assert loop._run_kimi_usage_poll_task(config, state, now_monotonic=101.0) is None


def test_rr089_kimi_usage_poll_scheduling_and_refresh_trigger(
    loop, tmp_path, monkeypatch, current_kimi_usage_payload
) -> None:
    config = _config(
        loop,
        tmp_path,
        kimi_oauth_refresh_enabled=True,
        kimi_usage_poll_enabled=True,
        kimi_oauth_refresh_interval_seconds=3600.0,
        kimi_usage_poll_interval_seconds=3600.0,
        observability_anomaly_scan_enabled=False,
    )
    monkeypatch.setattr(
        loop.kimi_oauth_refresh,
        "refresh_kimi_oauth_auth_file",
        lambda *args, **kwargs: {
            "attempted": True,
            "refreshed": True,
            "skipped": False,
            "auth_file": config.kimi_oauth_auth_file,
            "scope": "kimi-code",
            "expires_at": None,
            "auth_degraded": False,
            "error_class": None,
            "error_message": None,
        },
    )
    monkeypatch.setattr(
        loop,
        "_fetch_kimi_usage_payload",
        lambda _config: {"status_code": 200, "payload": current_kimi_usage_payload},
    )

    state = loop.SidecarTaskState(kimi_usage_last_attempt_monotonic=99.0)
    events = loop.run_due_sidecar_tasks(config, state, now_monotonic=100.0)

    assert [event["event"] for event in events] == [
        "kimi_oauth_refresh",
        "kimi_usage_poll",
    ]
    usage_event = events[1]
    assert usage_event["trigger"] == "oauth_refresh"
    assert usage_event["observation_count"] == 2
    assert usage_event["persisted"] is False
    assert loop._run_kimi_usage_poll_task(config, state, now_monotonic=101.0) is None


def test_rr089_kimi_usage_persistence_reuses_exact_dedupe_sql_payload(
    loop, tmp_path, populated_monthly_kimi_usage_payloads, monkeypatch
) -> None:
    config = _config(loop, tmp_path, apply=True, kimi_usage_poll_enabled=True)
    payloads, _summary = loop._build_kimi_usage_rate_limit_payloads(
        config,
        observed_at=datetime(2026, 7, 19, 16, 0, tzinfo=timezone.utc),
        response_body=populated_monthly_kimi_usage_payloads["current"],
    )
    calls = []
    rowcounts = iter((1, 1, 1, 0, 0, 0))

    class Cursor:
        rowcount = 0

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def execute(self, sql, params=None):
            calls.append((sql, params))
            if sql == loop.GROK_BILLING_RATE_LIMIT_INSERT_SQL:
                self.rowcount = next(rowcounts)

    class Connection:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def cursor(self):
            return Cursor()

        def rollback(self):
            raise AssertionError("dedupe-path test must not roll back")

    monkeypatch.setattr(loop, "_resolve_dsn", lambda _config: "postgresql://test")
    monkeypatch.setattr(loop.probes.psycopg, "connect", lambda _dsn: Connection())

    inserted = loop._persist_kimi_usage_observations(config, payloads)
    deduped = loop._persist_kimi_usage_observations(config, payloads)

    data_calls = [(sql, params) for sql, params in calls if sql == loop.GROK_BILLING_RATE_LIMIT_INSERT_SQL]
    assert inserted == 3
    assert deduped == 0
    assert [params for _sql, params in data_calls[:3]] == payloads
    assert len(data_calls) == 6
    assert "pg_advisory_xact_lock" in loop.GROK_BILLING_RATE_LIMIT_INSERT_SQL
    assert "latest.provider = candidate.provider" in loop.GROK_BILLING_RATE_LIMIT_INSERT_SQL
    assert "latest.quota_key = candidate.quota_key" in loop.GROK_BILLING_RATE_LIMIT_INSERT_SQL
    assert "latest.source IS NOT DISTINCT FROM candidate.source" in loop.GROK_BILLING_RATE_LIMIT_INSERT_SQL
