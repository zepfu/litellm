"""RR-089: append-safe, deduplicated observability anomaly error JSONL intake."""

from __future__ import annotations

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
