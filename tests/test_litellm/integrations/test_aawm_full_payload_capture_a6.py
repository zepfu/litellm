"""A6: full-payload capture control-file trust, header drop, atomic 0600 writes."""

from __future__ import annotations

import json
import os
import stat
from pathlib import Path
from unittest.mock import patch

import litellm.integrations.aawm_passthrough_shape_capture as capture


def test_full_payload_headers_drop_authorization() -> None:
    headers = {
        "Authorization": "Bearer secret-token",
        "x-api-key": "sk-test",
        "content-type": "application/json",
        "x-request-id": "abc",
    }
    sanitized = capture._full_payload_headers(headers)
    lower_keys = {k.lower() for k in sanitized}
    assert "authorization" not in lower_keys
    assert "x-api-key" not in lower_keys
    assert "content-type" in lower_keys
    assert sanitized["content-type"] == "application/json"

    items = capture._full_payload_header_items(headers)
    names = {item["name"].lower() for item in items}
    assert "authorization" not in names
    assert "x-api-key" not in names


def test_untrusted_world_writable_control_file_is_ignored(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.delenv(capture._FULL_PAYLOAD_ENV_FLAG, raising=False)
    # Simulate default /tmp control: parent world-writable
    parent = tmp_path / "world"
    parent.mkdir()
    os.chmod(parent, 0o777)
    control = parent / "pass_through_full_payloads.enabled"
    control.write_text("1", encoding="utf-8")
    os.chmod(control, 0o644)

    monkeypatch.setenv(
        capture._FULL_PAYLOAD_CONTROL_FILE_ENV, str(control)
    )
    assert capture.passthrough_full_payload_capture_enabled() is False


def test_trusted_control_file_enables_capture(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv(capture._FULL_PAYLOAD_ENV_FLAG, raising=False)
    parent = tmp_path / "owned"
    parent.mkdir()
    os.chmod(parent, 0o700)
    control = parent / "enabled"
    control.write_text("1", encoding="utf-8")
    os.chmod(control, 0o600)

    monkeypatch.setenv(capture._FULL_PAYLOAD_CONTROL_FILE_ENV, str(control))
    assert capture.passthrough_full_payload_capture_enabled() is True


def test_env_flag_enables_without_control_file(monkeypatch) -> None:
    monkeypatch.setenv(capture._FULL_PAYLOAD_ENV_FLAG, "1")
    monkeypatch.delenv(capture._FULL_PAYLOAD_CONTROL_FILE_ENV, raising=False)
    assert capture.passthrough_full_payload_capture_enabled() is True


def test_write_full_payload_artifact_is_mode_0600(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv(capture._FULL_PAYLOAD_ENV_FLAG, "1")
    monkeypatch.setenv(capture._FULL_PAYLOAD_DIR_ENV, str(tmp_path / "caps"))

    path_str = capture._write_full_payload_artifact(
        {
            "provider": "anthropic",
            "mode": "test",
            "litellm_call_id": "call-a6-1",
            "body": {"ok": True},
        }
    )
    assert path_str is not None
    path = Path(path_str)
    assert path.is_file()
    mode = path.stat().st_mode & 0o777
    assert mode == 0o600
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["body"]["ok"] is True

    cap_dir = tmp_path / "caps"
    dir_mode = cap_dir.stat().st_mode & 0o777
    # owner rwx only preferred
    assert dir_mode & stat.S_IWOTH == 0
    assert dir_mode & stat.S_IXUSR
