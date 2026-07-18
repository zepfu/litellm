"""RR-065 residuals for scripts/antigravity_oauth_refresh.py (#5-#11, #7 discovery)."""

from __future__ import annotations

import importlib.util
import logging
import os
import sys
from pathlib import Path
from typing import Any

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SCRIPT = _REPO_ROOT / "scripts" / "antigravity_oauth_refresh.py"


def _load_antigravity_module():
    name = "antigravity_oauth_refresh_rr065"
    spec = importlib.util.spec_from_file_location(name, _SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def antigravity():
    return _load_antigravity_module()


def test_defaults_are_portable_tilde_paths(antigravity) -> None:
    assert antigravity.DEFAULT_ANTIGRAVITY_AUTH_FILE.startswith("~/")
    assert antigravity.DEFAULT_ANTIGRAVITY_LOCK_FILE.startswith("~/")
    assert "/home/zepfu" not in antigravity.DEFAULT_ANTIGRAVITY_AUTH_FILE
    assert "/home/zepfu" not in antigravity.DEFAULT_ANTIGRAVITY_LOCK_FILE
    for path in antigravity._DEFAULT_CLI_BINARY_PATHS:
        assert path.startswith("~/") or path.startswith("$")
        assert "/home/zepfu" not in path


def test_default_paths_expanduser(antigravity) -> None:
    auth = Path(antigravity.DEFAULT_ANTIGRAVITY_AUTH_FILE).expanduser()
    lock = Path(antigravity.DEFAULT_ANTIGRAVITY_LOCK_FILE).expanduser()
    assert str(auth).startswith(str(Path.home()))
    assert str(lock).startswith(str(Path.home()))
    assert "~" not in str(auth)
    cli = Path(antigravity._DEFAULT_CLI_BINARY_PATHS[0]).expanduser()
    assert str(cli).startswith(str(Path.home()))


def test_metadata_helpers_use_shared_owner(antigravity, tmp_path: Path) -> None:
    from litellm.secret_managers.credential_file_metadata import (
        CredentialFileMetadata,
    )

    target = tmp_path / "token.json"
    target.write_text("{}", encoding="utf-8")
    os.chmod(target, 0o600)
    meta = antigravity._snapshot_credential_file_metadata(target)
    assert isinstance(meta, CredentialFileMetadata)
    assert meta.mode == 0o600

    # write path preserves metadata via shared apply
    antigravity._write_token_data(
        target,
        {
            "token": {
                "access_token": "a",
                "refresh_token": "r",
                "expiry": "2099-01-01T00:00:00Z",
            }
        },
    )
    assert target.stat().st_mode & 0o777 == 0o600


def test_resolve_metadata_env_override_and_clamp(
    antigravity, tmp_path: Path, monkeypatch
) -> None:
    target = tmp_path / "token.json"
    target.write_text("{}", encoding="utf-8")
    os.chmod(target, 0o644)
    monkeypatch.setenv("AAWM_ANTIGRAVITY_AUTH_FILE_MODE", "0o640")
    meta = antigravity._resolve_credential_file_metadata(target)
    # group/other bits from env or existing mode are clamped private
    assert meta.mode == 0o600

    monkeypatch.setenv("AAWM_ANTIGRAVITY_AUTH_FILE_MODE", "0o600")
    meta = antigravity._resolve_credential_file_metadata(target)
    assert meta.mode == 0o600


def test_write_private_text_clamps_group_other_bits(
    antigravity, tmp_path: Path
) -> None:
    path = tmp_path / "secret.tmp"
    antigravity._write_private_text(path, "secret\n", mode=0o644)
    assert path.stat().st_mode & 0o777 == 0o600


def test_lock_wrapper_delegates_only(antigravity) -> None:
    src = Path(antigravity.__file__).read_text(encoding="utf-8")
    assert "from litellm.secret_managers.credential_file_lock import" in src
    assert "import fcntl" not in src
    assert src.count("with credential_file_lock(lock_path)") == 1


def test_client_pair_diagnostic_id_is_stable_and_non_secret(antigravity) -> None:
    client_id = "123456-" + "abc.apps.googleusercontent.com"
    client_secret = "GOC" + "SPX-synthetic-test-value"
    pair_id = antigravity._client_pair_diagnostic_id(
        client_id,
        client_secret,
    )
    assert pair_id == antigravity._client_pair_diagnostic_id(
        client_id,
        client_secret,
    )
    assert len(pair_id) == 12
    assert "GOCSPX" not in pair_id
    assert "apps.googleusercontent.com" not in pair_id
    assert "secretvalue" not in pair_id
    other = antigravity._client_pair_diagnostic_id(
        "999999-" + "xyz.apps.googleusercontent.com",
        "GOC" + "SPX-other-synthetic-value",
    )
    assert other != pair_id


def test_cli_binary_scan_logs_counts_without_secrets(antigravity, caplog) -> None:
    client_id = b"123456-" + b"abcdefgh.apps.googleusercontent.com"
    client_secret = b"GOC" + b"SPX-SyntheticValue01"
    blob = b"xx" + client_id + b"yy" + client_secret + b"zz"
    with caplog.at_level(logging.INFO, logger=antigravity.logger.name):
        pairs = antigravity._extract_cli_client_value_candidates(blob)
    assert pairs
    assert all(p.source == "cli_binary_scan" for p in pairs)
    assert all(p.proximity_bytes is not None for p in pairs)
    joined = " ".join(rec.getMessage() for rec in caplog.records)
    assert "pairs=" in joined
    assert "secrets=" in joined
    assert "client_ids=" in joined
    assert "nearest_proximity_bytes=" in joined
    assert client_secret.decode() not in joined
    assert client_id.decode() not in joined


def test_cli_binary_scan_splits_adjacent_secrets_and_ranks_proximity(
    antigravity, caplog
) -> None:
    """RR-065 #7: do not concatenate adjacent GOCSPX blobs; rank by distance."""
    secret_a = b"GOC" + b"SPX-SyntheticAlphaValue000000000"
    secret_b = b"GOC" + b"SPX-SyntheticBetaValue0000000000"
    client_near = b"884354919052-" + b"syntheticnear.apps.googleusercontent.com"
    client_far = b"1071006060591-" + b"syntheticfar.apps.googleusercontent.com"
    blob = (
        b"PREFIX"
        + secret_a
        + secret_b
        + b"https://cloudcode-pa.googleapis.com"
        + (b"X" * 1000)
        + client_near
        + (b"Y" * 50_000)
        + client_far
    )
    with caplog.at_level(logging.INFO, logger=antigravity.logger.name):
        pairs = antigravity._extract_cli_client_value_candidates(blob)
    secrets = {p.client_secret for p in pairs}
    assert secret_a.decode() in secrets
    assert secret_b.decode() in secrets
    # Concatenated scrape must not survive validation.
    assert all(p.client_secret.count("GOCSPX-") == 1 for p in pairs)
    assert all("http" not in p.client_secret.lower() for p in pairs)
    # Nearest pairs first.
    assert pairs[0].proximity_bytes is not None
    assert pairs[0].proximity_bytes <= pairs[-1].proximity_bytes
    assert pairs[0].client_id == client_near.decode()
    assert len(pairs) <= antigravity._CLI_BINARY_MAX_CANDIDATE_PAIRS
    joined = " ".join(rec.getMessage() for rec in caplog.records)
    assert secret_a.decode() not in joined
    assert secret_b.decode() not in joined
    assert client_near.decode() not in joined


def test_client_discovery_skips_binary_when_token_file_has_pair(
    antigravity, monkeypatch, caplog
) -> None:
    """Safer token-file source must short-circuit binary scrape (RR-065 #7)."""
    client_id = "123456-" + "tokenfile.apps.googleusercontent.com"
    client_secret = "GOC" + "SPX-token-file-synthetic"
    token_data = {
        "token": {
            "refresh_token": "rt",
            "client_id": client_id,
            "client_secret": client_secret,
        }
    }
    called = {"binary": 0}

    def boom(_cli_path=None):  # noqa: ANN001
        called["binary"] += 1
        raise AssertionError("binary scan should be skipped")

    monkeypatch.setattr(antigravity, "_load_cli_client_value_candidates", boom)
    with caplog.at_level(logging.INFO, logger=antigravity.logger.name):
        candidates = antigravity._get_client_value_candidates(
            token_data,
            client_id=None,
            client_secret=None,
            cli_path="/nonexistent/agy",
        )
    assert called["binary"] == 0
    assert len(candidates) == 1
    assert candidates[0].source == "token_file"
    joined = " ".join(rec.getMessage() for rec in caplog.records)
    assert "skipping cli binary scan" in joined
    assert client_secret not in joined


def test_client_discovery_falls_back_to_binary_only_when_needed(
    antigravity, monkeypatch, caplog
) -> None:
    client_id = "123456-" + "binaryxx.apps.googleusercontent.com"
    client_secret = "GOC" + "SPX-binary-synthetic"
    fake = [
        antigravity._ClientCredentialCandidate(
            client_id=client_id,
            client_secret=client_secret,
            source="cli_binary_scan",
            proximity_bytes=1234,
        )
    ]
    monkeypatch.setattr(
        antigravity, "_load_cli_client_value_candidates", lambda *_a, **_k: fake
    )
    with caplog.at_level(logging.INFO, logger=antigravity.logger.name):
        candidates = antigravity._get_client_value_candidates(
            {"token": {"refresh_token": "rt"}},
            client_id=None,
            client_secret=None,
            cli_path=None,
        )
    assert len(candidates) == 1
    assert candidates[0].source == "cli_binary_scan"
    assert candidates[0].proximity_bytes == 1234
    joined = " ".join(rec.getMessage() for rec in caplog.records)
    assert "falling back to cli binary scan" in joined


def test_direct_refresh_logs_selected_pair_without_secrets(
    antigravity, monkeypatch, caplog
) -> None:
    token_data = {
        "token": {
            "access_token": "old",
            "refresh_token": "refresh-token-value",
            "expiry": "2000-01-01T00:00:00Z",
        }
    }
    client_id = "123456-" + "diagxxxx.apps.googleusercontent.com"
    client_secret = "GOC" + "SPX-diagnostic-synthetic"
    pair_id = antigravity._client_pair_diagnostic_id(client_id, client_secret)

    def fake_post(**kwargs: Any) -> dict[str, Any]:
        assert kwargs["client_id"] == client_id
        assert kwargs["client_secret"] == client_secret
        return {"access_token": "new-access", "expires_in": 3600}

    monkeypatch.setattr(antigravity, "_post_refresh_request", fake_post)
    monkeypatch.setattr(
        antigravity,
        "_get_client_value_candidates",
        lambda *a, **k: [
            antigravity._ClientCredentialCandidate(
                client_id=client_id,
                client_secret=client_secret,
                source="env",
            )
        ],
    )
    with caplog.at_level(logging.INFO, logger=antigravity.logger.name):
        refreshed = antigravity._refresh_token_data_direct(
            token_data,
            token_endpoint="https://example.invalid/token",
            client_id=None,
            client_secret=None,
            cli_path=None,
            http_timeout_seconds=1.0,
        )
    assert refreshed["token"]["access_token"] == "new-access"
    # Successful pair is persisted so later refreshes skip binary discovery.
    assert refreshed["token"]["client_id"] == client_id
    assert refreshed["token"]["client_secret"] == client_secret
    assert refreshed["client_id"] == client_id
    joined = " ".join(rec.getMessage() for rec in caplog.records)
    assert f"id={pair_id}" in joined
    assert "selected pair" in joined
    assert "source=env" in joined
    assert client_id not in joined
    assert client_secret not in joined
    assert "refresh-token-value" not in joined
    assert "new-access" not in joined


def test_direct_refresh_rejects_wrong_binary_pair_then_selects_next(
    antigravity, monkeypatch, caplog
) -> None:
    """Wrong nearest-id pairing must be diagnosable and skippable."""
    token_data = {
        "token": {
            "access_token": "old",
            "refresh_token": "refresh-token-value",
            "expiry": "2000-01-01T00:00:00Z",
        }
    }
    wrong = antigravity._ClientCredentialCandidate(
        client_id="111111-" + "wrongpair.apps.googleusercontent.com",
        client_secret="GOC" + "SPX-wrong-synthetic",
        source="cli_binary_scan",
        proximity_bytes=10,
    )
    right = antigravity._ClientCredentialCandidate(
        client_id="222222-" + "rightpair.apps.googleusercontent.com",
        client_secret="GOC" + "SPX-right-synthetic",
        source="cli_binary_scan",
        proximity_bytes=50,
    )
    calls: list[str] = []

    def fake_post(**kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs["client_id"])
        if kwargs["client_id"] == wrong.client_id:
            raise antigravity._RefreshHttpError(
                "status=401, error=invalid_client",
                oauth_error="invalid_client",
            )
        return {"access_token": "new-access", "expires_in": 3600}

    monkeypatch.setattr(antigravity, "_post_refresh_request", fake_post)
    monkeypatch.setattr(
        antigravity,
        "_get_client_value_candidates",
        lambda *a, **k: [wrong, right],
    )
    with caplog.at_level(logging.INFO, logger=antigravity.logger.name):
        refreshed = antigravity._refresh_token_data_direct(
            token_data,
            token_endpoint="https://example.invalid/token",
            client_id=None,
            client_secret=None,
            cli_path=None,
            http_timeout_seconds=1.0,
        )
    assert calls == [wrong.client_id, right.client_id]
    assert refreshed["token"]["client_id"] == right.client_id
    joined = " ".join(rec.getMessage() for rec in caplog.records)
    wrong_id = antigravity._client_pair_diagnostic_id(
        wrong.client_id, wrong.client_secret
    )
    right_id = antigravity._client_pair_diagnostic_id(
        right.client_id, right.client_secret
    )
    assert f"id={wrong_id}" in joined
    assert "invalid_client" in joined
    assert f"id={right_id}" in joined
    assert "source=cli_binary_scan" in joined
    assert wrong.client_secret not in joined
    assert right.client_secret not in joined


def test_cli_log_path_under_private_dir_and_cleaned_on_failure(
    antigravity, tmp_path: Path, monkeypatch
) -> None:
    # Seed must sit under the CLI-recognized path layout so refresh_home resolves
    # before staging replaces HOME with the private temp tree.
    seed = (
        tmp_path
        / "seed-home"
        / ".gemini"
        / "antigravity-cli"
        / "antigravity-oauth-token"
    )
    seed.parent.mkdir(parents=True)
    seed.write_text(
        '{"token":{"access_token":"a","refresh_token":"r","expiry":"2099-01-01T00:00:00Z"}}',
        encoding="utf-8",
    )
    auth = (
        tmp_path / "managed" / ".gemini" / "antigravity-cli" / "antigravity-oauth-token"
    )
    auth.parent.mkdir(parents=True)
    auth.write_text(seed.read_text(encoding="utf-8"), encoding="utf-8")

    fake_cli = tmp_path / "agy"
    fake_cli.write_text("#!/bin/sh\nexit 1\n", encoding="utf-8")
    fake_cli.chmod(0o755)

    captured: dict[str, Any] = {}

    def fake_run(cmd, **kwargs):  # noqa: ANN001
        captured["cmd"] = cmd
        captured["env"] = kwargs.get("env")
        log_file = Path(cmd[cmd.index("--log-file") + 1])
        captured["log_file"] = log_file
        # CLI would create the log; simulate that.
        log_file.parent.mkdir(parents=True, exist_ok=True)
        log_file.write_text("cli diagnostics\n", encoding="utf-8")
        assert log_file.parent.stat().st_mode & 0o777 == 0o700
        # parent should not be world-writable /tmp root itself
        assert log_file.parent != Path(os.getenv("TMPDIR") or "/tmp")

        class Result:
            returncode = 1

        return Result()

    monkeypatch.setattr(antigravity.subprocess, "run", fake_run)
    monkeypatch.setenv("TMPDIR", str(tmp_path / "tmp"))
    (tmp_path / "tmp").mkdir(exist_ok=True)

    with pytest.raises(ValueError, match="AGY CLI silent auth refresh failed"):
        antigravity._refresh_token_data_via_cli(
            auth,
            seed_auth_file=seed,
            original_token_data={
                "token": {
                    "access_token": "a",
                    "refresh_token": "r",
                    "expiry": "2099-01-01T00:00:00Z",
                }
            },
            cli_path=fake_cli,
            timeout_seconds=5.0,
        )

    log_file = captured["log_file"]
    # unconditional cleanup removes staged home (and log inside it)
    assert not log_file.exists()
    assert not log_file.parent.exists() or not any(log_file.parent.rglob("*"))
    # staged tree removed
    staged_candidates = list((tmp_path / "tmp").glob("litellm-antigravity-cli-home-*"))
    assert staged_candidates == []


def test_cli_log_private_dir_when_no_seed_staging(
    antigravity, tmp_path: Path, monkeypatch
) -> None:
    auth = tmp_path / "home" / ".gemini" / "antigravity-cli" / "antigravity-oauth-token"
    auth.parent.mkdir(parents=True)
    auth.write_text(
        '{"token":{"access_token":"a","refresh_token":"r","expiry":"2099-01-01T00:00:00Z"}}',
        encoding="utf-8",
    )
    fake_cli = tmp_path / "agy"
    fake_cli.write_text("#!/bin/sh\nexit 1\n", encoding="utf-8")
    fake_cli.chmod(0o755)

    captured: dict[str, Any] = {}

    def fake_run(cmd, **kwargs):  # noqa: ANN001
        log_file = Path(cmd[cmd.index("--log-file") + 1])
        captured["log_file"] = log_file
        log_file.parent.mkdir(parents=True, exist_ok=True)
        log_file.write_text("diag\n", encoding="utf-8")
        assert log_file.parent.stat().st_mode & 0o777 == 0o700

        class Result:
            returncode = 1

        return Result()

    monkeypatch.setattr(antigravity.subprocess, "run", fake_run)
    monkeypatch.setenv("TMPDIR", str(tmp_path / "tmp"))
    (tmp_path / "tmp").mkdir(exist_ok=True)

    with pytest.raises(ValueError, match="AGY CLI silent auth refresh failed"):
        antigravity._refresh_token_data_via_cli(
            auth,
            seed_auth_file=None,
            original_token_data={
                "token": {
                    "access_token": "a",
                    "refresh_token": "r",
                    "expiry": "2099-01-01T00:00:00Z",
                }
            },
            cli_path=fake_cli,
            timeout_seconds=5.0,
        )

    log_file = captured["log_file"]
    assert not log_file.exists()
    assert list((tmp_path / "tmp").glob("litellm-antigravity-cli-log-*")) == []


def test_private_temp_dir_is_unpredictable_and_private(
    antigravity, tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    first = antigravity._make_private_temp_dir("litellm-antigravity-cli-home-")
    second = antigravity._make_private_temp_dir("litellm-antigravity-cli-home-")
    try:
        assert first != second
        assert first.parent == tmp_path
        assert second.parent == tmp_path
        assert first.name.startswith("litellm-antigravity-cli-home-")
        assert first.stat().st_mode & 0o777 == 0o700
        assert second.stat().st_mode & 0o777 == 0o700
        # names must not be a pure pid+monotonic pattern that omits randomness
        assert not first.name.endswith(f"-{os.getpid()}")
    finally:
        antigravity._cleanup_staged_home(first)
        antigravity._cleanup_staged_home(second)
        assert not first.exists()
        assert not second.exists()


def test_write_token_data_creates_tmp_at_private_mode(
    antigravity, tmp_path: Path
) -> None:
    target = tmp_path / "token.json"
    antigravity._write_token_data(
        target,
        {
            "token": {
                "access_token": "a",
                "refresh_token": "r",
                "expiry": "2099-01-01T00:00:00Z",
            }
        },
    )
    assert target.exists()
    assert target.stat().st_mode & 0o777 == 0o600
    leftovers = list(tmp_path.glob(".token.json.*.tmp"))
    assert leftovers == []


def test_write_token_data_uses_shared_publish_api(antigravity) -> None:
    src = Path(antigravity.__file__).read_text(encoding="utf-8")
    assert "from litellm.secret_managers.credential_file_write import" in src
    assert "write_and_publish_private_text" in src
    # Predictable pid-only temp publication must be gone.
    assert 'f".{auth_path.name}.{os.getpid()}.tmp"' not in src
    assert "os.replace(tmp_path, auth_path)" not in src


def test_write_token_data_refuses_symlink_target(
    antigravity, tmp_path: Path
) -> None:
    from litellm.secret_managers.credential_file_metadata import (
        CredentialPathIsSymlinkError,
    )

    real = tmp_path / "real-token.json"
    real.write_text('{"token":{"access_token":"keep"}}', encoding="utf-8")
    os.chmod(real, 0o600)
    link = tmp_path / "token.json"
    link.symlink_to(real)
    with pytest.raises(CredentialPathIsSymlinkError):
        antigravity._write_token_data(
            link,
            {
                "token": {
                    "access_token": "attacker",
                    "refresh_token": "r",
                    "expiry": "2099-01-01T00:00:00Z",
                }
            },
        )
    assert real.read_text(encoding="utf-8") == '{"token":{"access_token":"keep"}}'
    leftovers = list(tmp_path.glob(".token.json.*.tmp"))
    assert leftovers == []


def test_write_token_data_temp_is_unpredictable_not_pid_only(
    antigravity, tmp_path: Path, monkeypatch
) -> None:
    """Exclusive temps must include a random token, not only pid."""
    from litellm.secret_managers import credential_file_write as cfw

    seen: list[Path] = []
    real_write_temp = cfw.write_private_temp_file_text

    def spy_write_temp(final_path, content, **kwargs):  # noqa: ANN001
        tmp = real_write_temp(final_path, content, **kwargs)
        seen.append(tmp)
        return tmp

    monkeypatch.setattr(cfw, "write_private_temp_file_text", spy_write_temp)
    # re-bind consumer import path
    monkeypatch.setattr(
        antigravity,
        "write_and_publish_private_text",
        lambda final_path, content, **kwargs: cfw.write_and_publish_private_text(
            final_path, content, **kwargs
        ),
    )
    target = tmp_path / "token.json"
    antigravity._write_token_data(
        target,
        {
            "token": {
                "access_token": "a",
                "refresh_token": "r",
                "expiry": "2099-01-01T00:00:00Z",
            }
        },
    )
    assert target.exists()
    assert seen, "temp publish path should run"
    name = seen[0].name
    assert str(os.getpid()) in name
    parts = name.split(".")
    assert any(len(p) >= 16 and all(c in "0123456789abcdef" for c in p) for p in parts)


def test_apply_metadata_refuses_symlink(antigravity, tmp_path: Path) -> None:
    from litellm.secret_managers.credential_file_metadata import (
        CredentialFileMetadata,
        CredentialPathIsSymlinkError,
    )

    real = tmp_path / "real.json"
    real.write_text("{}", encoding="utf-8")
    os.chmod(real, 0o600)
    link = tmp_path / "link.json"
    link.symlink_to(real)
    meta = CredentialFileMetadata(uid=None, gid=None, mode=0o600)
    with pytest.raises(CredentialPathIsSymlinkError):
        antigravity._apply_credential_file_metadata(link, meta)
    assert real.stat().st_mode & 0o777 == 0o600


def test_write_private_text_refuses_symlink_target(
    antigravity, tmp_path: Path
) -> None:
    from litellm.secret_managers.credential_file_metadata import (
        CredentialPathIsSymlinkError,
    )

    real = tmp_path / "elsewhere.secret"
    real.write_text("victim", encoding="utf-8")
    os.chmod(real, 0o600)
    link = tmp_path / "secret.tmp"
    link.symlink_to(real)
    with pytest.raises((CredentialPathIsSymlinkError, OSError)):
        antigravity._write_private_text(link, "attacker\n", mode=0o600)
    assert real.read_text(encoding="utf-8") == "victim"


# ---------------------------------------------------------------------------
# Shared error sanitizer residual (RR-065)
# ---------------------------------------------------------------------------


def test_sanitize_redacts_secret_values_not_only_labels(antigravity) -> None:
    raw = (
        "invalid_grant: access_token=eyJhbGciOi.live.token.value "
        "refresh_token: rt-super-secret "
        "client_secret=cs-xyz id_token= id.tok"
    )
    sanitized = antigravity._sanitize_error_message(raw)
    assert "eyJhbGciOi.live.token.value" not in sanitized
    assert "rt-super-secret" not in sanitized
    assert "cs-xyz" not in sanitized
    assert "id.tok" not in sanitized
    assert "access_token=[REDACTED]" in sanitized
    assert "refresh_token=[REDACTED]" in sanitized
    assert "client_secret=[REDACTED]" in sanitized
    assert "super-secret" not in sanitized


def test_sanitize_delegates_to_shared_helper(antigravity) -> None:
    from litellm.secret_managers.credential_error_sanitizer import (
        DEFAULT_SECRET_FIELD_NAMES,
        sanitize_credential_error_message,
    )

    assert antigravity._SECRET_FIELD_NAMES is DEFAULT_SECRET_FIELD_NAMES
    raw = "refresh_token=secret-value"
    assert antigravity._sanitize_error_message(raw) == sanitize_credential_error_message(
        raw, limit=500
    )
    long = "x" * 2000
    assert antigravity._sanitize_error_message(long, limit=100) == (
        sanitize_credential_error_message(long, limit=100)
    )


def test_sanitize_still_truncates_long_messages(antigravity) -> None:
    long = "x" * 2000
    out = antigravity._sanitize_error_message(long, limit=100)
    assert len(out) <= 100
    assert out.endswith("...")
    defaulted = antigravity._sanitize_error_message(long)
    assert len(defaulted) <= 500


def test_extract_oauth_error_hint_sanitizes_embedded_tokens(antigravity) -> None:
    import json

    body = json.dumps(
        {
            "error": "invalid_grant",
            "error_description": "access_token=eyJ.leaked.value was rejected",
        }
    )
    hint = antigravity._extract_oauth_error_hint(body)
    assert hint is not None
    assert "eyJ.leaked.value" not in hint
    assert "access_token=[REDACTED]" in hint or "invalid_grant" in hint
