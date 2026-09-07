"""CLI, loopback API, and dashboard contract tests (D1-752 semantic lane)."""

from __future__ import annotations

import copy
import json
import threading
from http.client import HTTPConnection
from http.server import HTTPServer
from pathlib import Path
from typing import Any, cast

import yaml

from scripts.chatgpt_chat_usage_capture.api import DASHBOARD_HTML, UsageAPI, make_handler
from scripts.chatgpt_chat_usage_capture.cli import main
from scripts.chatgpt_chat_usage_capture.config import parse_config
from scripts.chatgpt_chat_usage_capture.ledger import Ledger
from scripts.chatgpt_chat_usage_capture.tests.test_collector import MINIMAL_CONFIG

FIXTURE_ROOT = str(Path(__file__).parent / "fixtures")


def _write_config(tmp_path: Path, *, overrides: dict[str, Any] | None = None) -> Path:
    payload = copy.deepcopy(MINIMAL_CONFIG)
    application = dict(payload.get("application") or {})
    application["database_path"] = str(tmp_path / "usage.sqlite")
    application["bind_host"] = "127.0.0.1"
    application["bind_port"] = 0
    payload["application"] = application
    if overrides:
        for key, value in overrides.items():
            if key == "application" and isinstance(value, dict):
                payload["application"].update(value)
            else:
                payload[key] = value
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def _cli(capsys, config_path: Path, *argv: str) -> tuple[int, Any]:
    code = main(["--config", str(config_path), "--fixture-root", FIXTURE_ROOT, *argv])
    captured = capsys.readouterr()
    stdout = captured.out.strip()
    payload: Any = stdout
    if stdout:
        try:
            payload = json.loads(stdout)
        except json.JSONDecodeError:
            payload = stdout
    return code, payload


def _headers(mapping: dict[str, str]) -> dict[str, str]:
    return mapping


def test_cli_commands_cover_spec_surface(tmp_path: Path, capsys) -> None:
    config_path = _write_config(tmp_path)

    code, payload = _cli(capsys, config_path, "init")
    assert code == 0
    assert payload["status"] == "initialized"
    assert payload["accounts"] == ["test-account"]

    code, payload = _cli(capsys, config_path, "inspect-capabilities")
    assert code == 0
    assert payload["account_id"] == "test-account"
    assert payload["surface"] == "chat"
    assert "capabilities" in payload

    code, payload = _cli(capsys, config_path, "models", "review")
    assert code == 0
    assert payload["mapping_version"] == "test"
    assert "astra_pro" in payload["canonical_families"]

    code, payload = _cli(capsys, config_path, "backfill", "--since", "14d")
    assert code == 0
    assert payload["mode"] in {"backfill", "catch_up"}
    assert payload["result"] == "complete"

    code, payload = _cli(capsys, config_path, "refresh")
    assert code == 0
    assert payload["result"] == "complete"

    code, payload = _cli(capsys, config_path, "run", "--once")
    assert code == 0
    assert payload["result"] == "complete" or payload.get("status") == "not_due"

    code, payload = _cli(capsys, config_path, "status")
    assert code == 0
    assert payload["account_id"] == "test-account"
    assert payload["quota_buckets"]
    assert payload["label"].startswith("Working estimate")

    code, payload = _cli(capsys, config_path, "report", "--since", "7d")
    assert code == 0
    assert payload["range"]["kind"] in {"elapsed", "elapsed_default"}
    assert payload["range"]["label"] == "last_seven_days_elapsed_or_default"
    astra = next(item for item in payload["quota_buckets"] if item["bucket_id"] == "pro200-astra-chat")
    assert astra["working_remaining_estimate"] is None
    assert astra["working_usage_estimate"] is None

    code, payload = _cli(capsys, config_path, "schedule", "set", "--every", "3h")
    assert code == 0
    assert payload["refresh_interval"] == "PT3H"
    assert payload["next_due_at"]

    code, payload = _cli(capsys, config_path, "windows", "list")
    assert code == 0
    assert {item["bucket_id"] for item in payload} == {"pro200-astra-chat", "pro200-sol-chat"}

    code, payload = _cli(
        capsys,
        config_path,
        "windows",
        "set-explicit",
        "--bucket",
        "pro200-astra-chat",
        "--start",
        "2026-09-06T00:00:00Z",
        "--end",
        "2026-09-13T00:00:00Z",
        "--evidence",
        "operator-assumption",
        "--reason",
        "cli-test",
    )
    assert code == 0
    assert payload["status"] == "set"
    assert payload["window"]["window_type"] == "operator_explicit"

    code, payload = _cli(
        capsys,
        config_path,
        "quota",
        "record-observation",
        "--bucket",
        "pro200-astra-chat",
        "--remaining",
        "120",
        "--capacity",
        "200",
        "--as-of",
        "2026-09-06T12:00:00Z",
    )
    assert code == 0
    assert payload["window_unchanged"] is True
    assert payload["observation"]["remaining"] == 120
    assert payload["observation"]["window_start"] is None

    listed = _cli(capsys, config_path, "windows", "list")[1]
    astra_window = next(item for item in listed if item["bucket_id"] == "pro200-astra-chat")
    assert astra_window["window_type"] == "operator_explicit"
    assert str(astra_window["start_at"]).startswith("2026-09-06")

    code, payload = _cli(capsys, config_path, "rebuild", "--dry-run")
    assert code == 0
    assert payload["dry_run"] is True
    assert payload["revision_id"] is None

    code, payload = _cli(capsys, config_path, "rebuild", "--apply")
    assert code == 0
    assert payload["published"] is True
    assert payload["revision_id"]

    code, payload = _cli(capsys, config_path, "retention", "prune", "--as-of", "2026-09-06T12:00:00Z")
    assert code == 0
    assert payload["aliases_preserved"] is True
    assert "tombstoned_attempts" in payload

    code, payload = _cli(capsys, config_path, "export", "--since", "7d", "--format", "json")
    assert code == 0
    assert payload["account_id"] == "test-account"
    assert "quota_buckets" in payload


def test_api_v1_routes_and_health(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path, overrides={"application": {"local_api_auth": "disabled"}})
    config = parse_config(yaml.safe_load(config_path.read_text()), source_path=config_path)
    ledger = Ledger(config.application.database_path)
    try:
        ledger.upsert_account(
            {
                "collector_account_id": "test-account",
                "quota_owner_id": "test-account",
                "surface": "chat",
                "auth_state": "ready",
            }
        )
        api = UsageAPI(config, ledger, fixture_root=FIXTURE_ROOT)
        headers = _headers({})

        status, payload = api.dispatch("GET", "/health/live", headers, {})
        assert status == 200
        assert payload["status"] == "live"
        assert payload["surface"] == "chat"

        status, payload = api.dispatch("GET", "/health/ready", headers, {})
        assert status == 200
        assert payload["account_id"] == "test-account"

        status, payload = api.dispatch("GET", "/api/v1/accounts", headers, {})
        assert status == 200
        assert payload["accounts"][0]["id"] == "test-account"

        status, payload = api.dispatch("GET", "/api/v1/status", headers, {})
        assert status == 200
        assert payload["quota_buckets"]

        status, payload = api.dispatch("GET", "/api/v1/usage?lookback=P7D", headers, {})
        assert status == 200
        assert payload["range"]["kind"] in {"elapsed", "elapsed_default"}

        status, payload = api.dispatch("GET", "/api/v1/attempts", headers, {})
        assert status == 200
        assert "attempts" in payload

        status, payload = api.dispatch("GET", "/api/v1/quota-windows", headers, {})
        assert status == 200
        assert payload["windows"]

        status, payload = api.dispatch("GET", "/api/v1/quota-observations", headers, {})
        assert status == 200
        assert payload["observations"] == []

        status, payload = api.dispatch("GET", "/api/v1/coverage", headers, {})
        assert status == 200
        assert payload["account_id"] == "test-account"

        status, payload = api.dispatch("GET", "/api/v1/runs", headers, {})
        assert status == 200
        assert "runs" in payload

        status, payload = api.dispatch("POST", "/api/v1/refresh", headers, {"force": True, "mode": "refresh"})
        assert status == 200
        assert payload["status"] in {"complete", "not_due", "coalesced"}

        status, payload = api.dispatch("POST", "/api/v1/schedule", headers, {"every": "3h"})
        assert status == 200
        assert payload["refresh_interval"] == "PT3H"

        status, payload = api.dispatch(
            "POST",
            "/api/v1/window-definitions",
            headers,
            {
                "bucket": "pro200-sol-chat",
                "start": "2026-09-06T00:00:00Z",
                "end": "2026-09-07T00:00:00Z",
                "evidence": "operator-assumption",
                "reason": "api-test",
            },
        )
        assert status == 200
        assert payload["window"]["window_type"] == "operator_explicit"

        status, payload = api.dispatch(
            "POST",
            "/api/v1/manual-observations",
            headers,
            {"bucket": "pro200-sol-chat", "remaining": 80, "capacity": 200},
        )
        assert status == 200
        assert payload["window_unchanged"] is True
        assert payload["observation"]["window_start"] is None

        status, payload = api.dispatch("POST", "/api/v1/rebuild-preview", headers, {})
        assert status == 200
        assert payload["dry_run"] is True

        status, payload = api.dispatch("POST", "/api/v1/rebuild-apply", headers, {})
        assert status == 200
        assert payload["published"] is True
    finally:
        ledger.close()


def test_api_auth_csrf_and_dashboard_unknown_remaining(tmp_path: Path) -> None:
    config_path = _write_config(
        tmp_path,
        overrides={"application": {"local_api_auth": "required", "local_api_token": "secret-token"}},
    )
    config = parse_config(yaml.safe_load(config_path.read_text()), source_path=config_path)
    ledger = Ledger(config.application.database_path)
    try:
        api = UsageAPI(config, ledger, fixture_root=FIXTURE_ROOT)
        assert api.authorized(_headers({})) is False
        assert api.authorized(_headers({"Authorization": "Bearer secret-token"})) is True
        assert api.csrf_ok("POST", _headers({"Origin": "http://evil.example"})) is False
        assert api.csrf_ok("POST", _headers({"Origin": "http://127.0.0.1:8765"})) is True
        assert "function fmt(value){return (value===null||value===undefined)?\"Unknown\":value;}" in DASHBOARD_HTML
        assert "window.__USAGE_TOKEN" in DASHBOARD_HTML

        handler = make_handler(api)
        server = HTTPServer(("127.0.0.1", 0), handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        host, port = cast(tuple[str, int], server.server_address)
        try:
            unauth = HTTPConnection(host, port, timeout=5)
            unauth.request("GET", "/api/v1/status")
            response = unauth.getresponse()
            body = json.loads(response.read().decode("utf-8"))
            unauth.close()
            assert response.status == 401
            assert body["error"] == "unauthorized"

            csrf = HTTPConnection(host, port, timeout=5)
            csrf.request(
                "POST",
                "/api/v1/refresh",
                body=b"{}",
                headers={
                    "Authorization": "Bearer secret-token",
                    "Origin": "https://evil.example",
                    "Content-Type": "application/json",
                    "Content-Length": "2",
                },
            )
            response = csrf.getresponse()
            body = json.loads(response.read().decode("utf-8"))
            csrf.close()
            assert response.status == 403
            assert body["error"] == "csrf_rejected"

            dash = HTTPConnection(host, port, timeout=5)
            dash.request("GET", "/")
            response = dash.getresponse()
            html = response.read().decode("utf-8")
            dash.close()
            assert response.status == 200
            assert 'window.__USAGE_TOKEN="secret-token";' in html
            assert "Unknown" in html
            assert "function fmt(value)" in html
        finally:
            server.shutdown()
            server.server_close()
    finally:
        ledger.close()


def test_cli_report_lookback_is_not_a_quota_window(tmp_path: Path, capsys) -> None:
    config_path = _write_config(tmp_path)
    assert _cli(capsys, config_path, "init")[0] == 0
    code, payload = _cli(capsys, config_path, "report", "--since", "24h")
    assert code == 0
    assert payload["range"]["kind"] in {"elapsed", "elapsed_default"}
    assert payload["range"]["label"] == "last_24_hours_elapsed"
    assert payload["activity_only"] is False
