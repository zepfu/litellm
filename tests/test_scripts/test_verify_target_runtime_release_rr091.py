"""Focused RR-091 coverage for verify_target_runtime_release remaining findings."""

from __future__ import annotations

import inspect
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import scripts.verify_target_runtime_release as verifier


def test_should_require_existing_release_runbook_file_for_prod_checkpoint(
    tmp_path, capsys
) -> None:
    """RR-091 Medium #2: non-empty fabricated runbook path must fail closed."""
    missing = tmp_path / "does-not-exist-runbook.md"
    code = verifier.main(
        [
            "--target",
            "prod",
            "--quiet",
            "--release-runbook",
            str(missing),
            "--image-tag",
            "doesnotexist",
            "--callback-wheel",
            "nope",
            "--db-name",
            "whatever",
        ]
    )
    assert code == 1

    # Non-quiet path for evidence fields.
    code = verifier.main(
        [
            "--target",
            "prod",
            "--release-runbook",
            str(missing),
            "--image-tag",
            "doesnotexist",
            "--callback-wheel",
            "nope",
            "--db-name",
            "whatever",
        ]
    )
    assert code == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is False
    assert "release_runbook_file" in payload["missing_checkpoint_fields"]
    assert "release_runbook" not in payload["missing_checkpoint_fields"]
    assert "existing" in payload["reason"].lower() or "release-runbook" in payload["reason"]


def test_should_accept_existing_release_runbook_file_for_prod_checkpoint(
    tmp_path, capsys
) -> None:
    """RR-091 Medium #2: real on-disk runbook satisfies the file gate."""
    runbook = tmp_path / "release-runbook.md"
    runbook.write_text("# release\n", encoding="utf-8")
    code = verifier.main(
        [
            "--target",
            "prod",
            "--lane",
            "target-manifest",
            "--release-runbook",
            str(runbook),
            "--image-tag",
            "v1.2.3",
            "--callback-wheel",
            "wheel-abc",
            "--db-name",
            "aawm_tristore",
            "--workspace-root",
            str(tmp_path),
        ]
    )
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["manifest"]["release_runbook_exists"] is True
    assert Path(payload["manifest"]["release_runbook"]).is_file()


def test_should_treat_directory_runbook_as_missing_file(tmp_path, capsys) -> None:
    directory = tmp_path / "runbook-dir"
    directory.mkdir()
    missing = verifier._prod_checkpoint_missing(
        SimpleNamespace(
            release_runbook=str(directory),
            image_tag="tag",
            callback_wheel="wheel",
            db_name="db",
        )
    )
    assert "release_runbook_file" in missing
    assert "release_runbook" not in missing


def test_should_redact_session_history_db_exceptions(monkeypatch, tmp_path, capsys) -> None:
    """RR-091 High #1 already landed: DB failures stay on the redaction path."""
    workspace = tmp_path / "litellm"
    workspace.mkdir()

    def boom(**kwargs):
        raise RuntimeError(
            "connection failed postgresql://user:supersecret@10.1.2.3:5432/db "
            "password=leaked-pass Bearer secret-token-xyz"
        )

    monkeypatch.setattr(verifier, "query_session_history_row", boom)
    monkeypatch.setenv(
        "AAWM_DATABASE_URL", "postgresql://user:supersecret@10.1.2.3:5432/db"
    )

    code = verifier.main(
        [
            "--target",
            "dev",
            "--lane",
            "session-history",
            "--workspace-root",
            str(workspace),
            "--marker-id",
            "m-redact",
            "--session-id",
            "sess-redact",
        ]
    )
    assert code == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is False
    reason = payload["reason"]
    assert "session-history query failed" in reason
    assert "supersecret" not in reason
    assert "leaked-pass" not in reason
    assert "secret-token-xyz" not in reason
    assert "10.1.2.3" not in reason
    assert "Bearer" in reason  # header retained, value redacted
    assert "[REDACTED]" in reason or "<redacted-dsn>" in reason


def test_should_build_bounded_session_history_query_with_time_window() -> None:
    """RR-091 Low #3: replace fragile LIMIT 10 with configurable/time-bounded SQL."""
    query_source = inspect.getsource(verifier.query_session_history_row)
    fetch_source = inspect.getsource(verifier._fetch_session_history_candidates)
    module_source = Path(verifier.__file__).read_text(encoding="utf-8")
    assert "LIMIT 10" not in module_source
    assert "created_at >=" in fetch_source
    assert "row_limit" in query_source
    assert "ORDER BY created_at DESC LIMIT" in fetch_source

    assert verifier._session_history_row_limit(None) == verifier.DEFAULT_SESSION_HISTORY_ROW_LIMIT
    assert verifier._session_history_row_limit(0) == 1
    assert (
        verifier._session_history_row_limit(verifier.MAX_SESSION_HISTORY_ROW_LIMIT + 99)
        == verifier.MAX_SESSION_HISTORY_ROW_LIMIT
    )
    assert verifier._session_history_row_limit(42) == 42

    absolute = verifier._session_history_created_after(
        created_after="2026-07-01T00:00:00Z",
        lookback_hours=1.0,
    )
    assert absolute == datetime(2026, 7, 1, tzinfo=timezone.utc)

    before = datetime.now(timezone.utc)
    windowed = verifier._session_history_created_after(
        created_after=None,
        lookback_hours=24.0,
    )
    after = datetime.now(timezone.utc)
    assert windowed is not None
    assert before - timedelta(hours=24) <= windowed <= after - timedelta(hours=24) + timedelta(
        seconds=2
    )

    assert (
        verifier._session_history_created_after(created_after=None, lookback_hours=0)
        is None
    )
    assert (
        verifier._session_history_created_after(created_after=None, lookback_hours=-1)
        is None
    )


def test_should_pass_row_limit_and_created_after_into_query(
    monkeypatch, tmp_path
) -> None:
    workspace = tmp_path / "litellm"
    workspace.mkdir()
    captured: dict = {}

    def fake_query(**kwargs):
        captured.update(kwargs)
        return (
            {
                "id": 7,
                "session_id": "sess-bound",
                "trace_id": "trace-bound",
                "litellm_call_id": "call-bound",
                "inbound_model_alias": "aawm-read",
                "provider": "openai",
                "model": "gpt-5",
                "repository": "litellm",
                "tenant_id": "litellm",
                "created_at": datetime(2026, 7, 1, 12, 0, tzinfo=timezone.utc),
                "route_family": "openai_passthrough",
                "marker_match_sources": ["metadata.d1_453_marker_id"],
            },
            "aawm_tristore",
        )

    monkeypatch.setattr(verifier, "query_session_history_row", fake_query)
    monkeypatch.setenv(
        "AAWM_DATABASE_URL", "postgresql://u:p@127.0.0.1:5432/aawm_tristore"
    )

    code = verifier.main(
        [
            "--target",
            "dev",
            "--quiet",
            "--lane",
            "session-history",
            "--workspace-root",
            str(workspace),
            "--marker-id",
            "marker-bound",
            "--session-id",
            "sess-bound",
            "--session-history-row-limit",
            "50",
            "--session-history-created-after",
            "2026-06-30T00:00:00+00:00",
        ]
    )
    assert code == 0
    assert captured["row_limit"] == 50
    assert captured["created_after"] == datetime(2026, 6, 30, tzinfo=timezone.utc)


def test_should_default_lookback_window_when_absolute_not_set(
    monkeypatch, tmp_path
) -> None:
    workspace = tmp_path / "litellm"
    workspace.mkdir()
    captured: dict = {}

    def fake_query(**kwargs):
        captured.update(kwargs)
        return (None, "aawm_tristore")

    monkeypatch.setattr(verifier, "query_session_history_row", fake_query)
    monkeypatch.setenv(
        "AAWM_DATABASE_URL", "postgresql://u:p@127.0.0.1:5432/aawm_tristore"
    )

    before = datetime.now(timezone.utc) - timedelta(
        hours=verifier.DEFAULT_SESSION_HISTORY_LOOKBACK_HOURS
    )
    code = verifier.main(
        [
            "--target",
            "dev",
            "--quiet",
            "--lane",
            "session-history",
            "--workspace-root",
            str(workspace),
            "--marker-id",
            "marker-default-window",
            "--trace-id",
            "trace-default-window",
        ]
    )
    after = datetime.now(timezone.utc) - timedelta(
        hours=verifier.DEFAULT_SESSION_HISTORY_LOOKBACK_HOURS
    )
    assert code == 3  # fail-closed: no match
    assert captured["row_limit"] == verifier.DEFAULT_SESSION_HISTORY_ROW_LIMIT
    assert captured["created_after"] is not None
    assert before <= captured["created_after"] <= after + timedelta(seconds=2)


def test_should_reject_invalid_created_after(monkeypatch, tmp_path, capsys) -> None:
    workspace = tmp_path / "litellm"
    workspace.mkdir()
    monkeypatch.setenv(
        "AAWM_DATABASE_URL", "postgresql://u:p@127.0.0.1:5432/aawm_tristore"
    )
    monkeypatch.setattr(
        verifier,
        "query_session_history_row",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("must not query")),
    )

    code = verifier.main(
        [
            "--target",
            "dev",
            "--lane",
            "session-history",
            "--workspace-root",
            str(workspace),
            "--marker-id",
            "m",
            "--session-id",
            "s",
            "--session-history-created-after",
            "not-a-timestamp",
        ]
    )
    assert code == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is False
    assert "invalid --session-history-created-after" in payload["reason"]


def test_should_not_probe_live_http_or_docker() -> None:
    """Guardrail: verifier remains offline; no live target probes."""
    source = Path(verifier.__file__).read_text(encoding="utf-8")
    assert "import requests" not in source
    assert "urllib.request" not in source
    assert "httpx" not in source
    # Profile may list docker_inspect as an allowed action name only.
    assert "docker_inspect" in source
    # No live docker CLI invocation patterns (git is the only subprocess).
    assert '["docker"' not in source
    assert "['docker'" not in source
    assert '"docker",' not in source
    assert "'docker'," not in source
