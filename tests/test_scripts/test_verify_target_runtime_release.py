from __future__ import annotations

import inspect
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import scripts.verify_target_runtime_release as verifier

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_should_redact_secret_like_values() -> None:
    raw = (
        "Bearer secret-token-abc "
        "sk-live1234567890 "
        "api_key=supersecret "
        "postgresql://user:pass@host/db "
        "10.0.0.5"
    )
    redacted = verifier.redact_text(raw)
    assert "secret-token-abc" not in redacted
    assert "sk-live1234567890" not in redacted
    assert "supersecret" not in redacted
    assert "pass@" not in redacted
    assert "10.0.0.5" not in redacted
    assert "[REDACTED]" in redacted or "sk-[REDACTED]" in redacted


def test_should_separate_referenced_artifact_owner_from_active_workspace(tmp_path) -> None:
    workspace = tmp_path / "litellm"
    workspace.mkdir()
    ref = "/home/zepfu/projects/dashboard-shell/.analysis/suggestion.md"
    owners = verifier._referenced_artifact_owners([ref])
    assert owners == [
        {
            "path": ref,
            "referenced_repository": "dashboard-shell",
        }
    ]
    assert verifier._active_repository(workspace) == "litellm"


def test_should_match_marker_from_correlation_or_bounded_metadata() -> None:
    assert verifier._marker_sources_for_row(
        {"session_id": "marker-1", "metadata": {"prompt": "other"}},
        "marker-1",
    ) == ["session_id"]
    assert verifier._marker_sources_for_row(
        {"session_id": "session-1", "metadata": {"d1_453_marker_id": "marker-2"}},
        "marker-2",
    ) == ["metadata.d1_453_marker_id"]
    assert verifier._marker_sources_for_row(
        {"session_id": "session-1", "metadata": {"d1_453_marker_id": "marker-2"}},
        "marker-3",
    ) == []


def test_session_history_query_should_not_scan_metadata_text() -> None:
    source = inspect.getsource(verifier.query_session_history_row)
    assert "metadata::text" not in source
    assert "ILIKE" not in source


def test_should_build_dev_target_manifest(tmp_path) -> None:
    workspace = tmp_path / "litellm"
    workspace.mkdir()
    args = verifier._parse_args(
        ["--target", "dev", "--workspace-root", str(workspace)]
    )
    manifest = verifier._build_manifest(
        target="dev",
        lanes_requested=["target-manifest"],
        lanes_executed=["target-manifest"],
        lanes_skipped=["session-history"],
        workspace_root=workspace,
        referenced_artifacts=[],
        args=args,
    )
    assert manifest["schema_version"] == verifier.SCHEMA_VERSION
    assert manifest["target"] == "dev"
    assert manifest["compose_project"] == "litellm"
    assert manifest["container_name"] == "litellm-dev"
    assert manifest["base_url"] == "http://127.0.0.1:4001"
    assert manifest["db_name"] == "aawm_tristore"
    assert manifest["source_mode"] == "bind_mount"
    assert "session_history_probe" in manifest["allowed_actions"]
    assert manifest["skipped_lanes"] == ["session-history"]
    assert manifest["provenance"]["active_repository"] == "litellm"


def test_should_refuse_prod_before_probes_without_checkpoints(monkeypatch) -> None:
    monkeypatch.setattr(
        verifier,
        "query_session_history_row",
        lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("prod refusal must not query session_history")
        ),
    )

    code = verifier.main(
        ["--target", "prod", "--quiet", "--lane", "session-history", "--marker-id", "m1"]
    )
    assert code == 1


def test_should_list_missing_prod_checkpoint_fields_in_refusal(capsys) -> None:
    verifier.main(["--target", "prod"])
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["ok"] is False
    assert "release_runbook" in payload["missing_checkpoint_fields"]
    assert "image_tag" in payload["missing_checkpoint_fields"]
    assert "callback_wheel" in payload["missing_checkpoint_fields"]
    assert "db_name" in payload["missing_checkpoint_fields"]
    assert payload["prod_read_only"] is True
    assert "read-only" in payload["reason"].lower()


def test_should_reject_unsupported_lane() -> None:
    code = verifier.main(
        ["--target", "dev", "--quiet", "--lane", "langfuse-sidecar"]
    )
    assert code == 2


def test_should_succeed_session_history_lane_with_persisted_row(
    monkeypatch, tmp_path
) -> None:
    workspace = tmp_path / "litellm"
    workspace.mkdir()
    created = datetime(2026, 6, 1, 12, 0, tzinfo=timezone.utc)

    def fake_query(**kwargs):
        return (
            {
                "id": 99,
                "session_id": "sess-1",
                "trace_id": "trace-1",
                "litellm_call_id": "call-1",
                "inbound_model_alias": "aawm-read",
                "provider": "openai",
                "model": "gpt-5",
                "repository": "litellm",
                "tenant_id": "litellm",
                "created_at": created,
                "route_family": "openai_passthrough",
            },
            "aawm_tristore",
        )

    monkeypatch.setattr(verifier, "query_session_history_row", fake_query)
    monkeypatch.setenv("AAWM_DATABASE_URL", "postgresql://u:p@127.0.0.1:5432/aawm_tristore")

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
            "marker-abc",
            "--session-id",
            "sess-1",
            "--db-name",
            "aawm_tristore",
        ]
    )
    assert code == 0


def test_should_fail_session_history_lane_when_row_missing(monkeypatch, tmp_path) -> None:
    workspace = tmp_path / "litellm"
    workspace.mkdir()
    monkeypatch.setattr(
        verifier,
        "query_session_history_row",
        lambda **kwargs: (None, "aawm_tristore"),
    )
    monkeypatch.setenv("AAWM_DATABASE_URL", "postgresql://u:p@127.0.0.1:5432/aawm_tristore")

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
            "marker-missing",
            "--trace-id",
            "trace-missing",
        ]
    )
    assert code == 3


def test_should_fail_expected_repository_mismatch(monkeypatch, tmp_path) -> None:
    workspace = tmp_path / "litellm"
    workspace.mkdir()

    def fake_query(**kwargs):
        return (
            {
                "id": 1,
                "session_id": "s",
                "trace_id": "t",
                "litellm_call_id": "c",
                "inbound_model_alias": "aawm-read",
                "provider": "openai",
                "model": "gpt-5",
                "repository": "other-repo",
                "tenant_id": "other-repo",
                "created_at": datetime.now(timezone.utc),
                "route_family": None,
            },
            "aawm_tristore",
        )

    monkeypatch.setattr(verifier, "query_session_history_row", fake_query)
    monkeypatch.setenv("AAWM_DATABASE_URL", "postgresql://u:p@127.0.0.1:5432/aawm_tristore")

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
            "m",
            "--session-id",
            "s",
            "--expected-repository",
            "litellm",
        ]
    )
    assert code == 4


def test_should_fail_expected_tenant_mismatch(monkeypatch, tmp_path) -> None:
    workspace = tmp_path / "litellm"
    workspace.mkdir()

    def fake_query(**kwargs):
        return (
            {
                "id": 2,
                "session_id": "s2",
                "trace_id": "t2",
                "litellm_call_id": "c2",
                "inbound_model_alias": None,
                "provider": "anthropic",
                "model": "claude",
                "repository": "litellm",
                "tenant_id": "wrong-tenant",
                "created_at": datetime.now(timezone.utc),
                "route_family": "anthropic",
            },
            "aawm_tristore",
        )

    monkeypatch.setattr(verifier, "query_session_history_row", fake_query)
    monkeypatch.setenv("AAWM_DATABASE_URL", "postgresql://u:p@127.0.0.1:5432/aawm_tristore")

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
            "m2",
            "--litellm-call-id",
            "c2",
            "--expected-tenant-id",
            "litellm",
        ]
    )
    assert code == 4


def test_prod_refusal_does_not_import_psycopg_on_module_path() -> None:
    script = REPO_ROOT / "scripts" / "verify_target_runtime_release.py"
    proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "--target",
            "prod",
        ],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        timeout=30,
    )
    assert proc.returncode == 1
    combined = proc.stdout + proc.stderr
    assert "missing_checkpoint_fields" in combined
