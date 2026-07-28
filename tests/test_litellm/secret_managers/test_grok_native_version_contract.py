"""Focused tests for the Grok native client-version contract module."""

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import pytest

from litellm.secret_managers import (
    grok_native_version_contract as version_contract,
)
from litellm.secret_managers.grok_native_version_contract import (
    GROK_VERSION_CACHE_MAX_AGE_ENV,
    GROK_VERSION_CACHE_PATH_ENV,
    GROK_VERSION_DEFAULT_CACHE_PATH,
    GROK_VERSION_DEFAULT_MAX_AGE_SECONDS,
    GROK_VERSION_MAX_BYTES,
    GROK_VERSION_SCHEMA_VERSION,
    GrokNativeVersionError,
    GrokNativeVersionMetadata,
    GrokNativeVersionRecord,
    resolve_grok_native_version,
    try_resolve_grok_native_version,
)


def _utc_z(epoch: float) -> str:
    """Format epoch seconds as RFC 3339 UTC ending in Z."""
    return datetime.fromtimestamp(epoch, tz=timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )


def _make_payload(
    *,
    version: str = "0.1.211",
    build: str = "a1b2c3d4",
    channel: str = "stable",
    observed_at: str | None = None,
    now: float | None = None,
    **overrides,
) -> dict:
    ref = now if now is not None else time.time()
    payload = {
        "schema_version": GROK_VERSION_SCHEMA_VERSION,
        "client": "grok-cli",
        "version": version,
        "build": build,
        "channel": channel,
        "source": "installed-grok-cli",
        "observed_at": observed_at if observed_at is not None else _utc_z(ref - 60),
    }
    payload.update(overrides)
    return payload


def _write(tmp_path: Path, payload: dict, name: str = "cache.json") -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Valid resolution
# ---------------------------------------------------------------------------


class TestValidResolution:
    def test_valid_record_resolves(self, tmp_path: Path):
        now = time.time()
        payload = _make_payload(now=now)
        path = _write(tmp_path, payload)

        record, meta = resolve_grok_native_version(
            cache_path=str(path), now=now
        )

        assert isinstance(record, GrokNativeVersionRecord)
        assert isinstance(meta, GrokNativeVersionMetadata)
        assert record.version == "0.1.211"
        assert record.build == "a1b2c3d4"
        assert record.channel == "stable"
        assert record.client == "grok-cli"
        assert record.source == "installed-grok-cli"
        assert record.schema_version == 1
        assert meta.version == "0.1.211"
        assert meta.age_seconds >= 0

    def test_multi_segment_version_accepted(self, tmp_path: Path):
        now = time.time()
        payload = _make_payload(version="1.2.3.4", now=now)
        path = _write(tmp_path, payload)
        record, _ = resolve_grok_native_version(
            cache_path=str(path), now=now
        )
        assert record.version == "1.2.3.4"

# ---------------------------------------------------------------------------
# Missing / symlink / non-regular / unreadable
# ---------------------------------------------------------------------------


class TestFileLevelRejection:
    def test_missing_file_raises(self, tmp_path: Path):
        with pytest.raises(GrokNativeVersionError, match="missing"):
            resolve_grok_native_version(
                cache_path=str(tmp_path / "nonexistent.json")
            )

    def test_symlink_rejected(self, tmp_path: Path):
        now = time.time()
        payload = _make_payload(now=now)
        real = _write(tmp_path, payload, name="real.json")
        link = tmp_path / "link.json"
        os.symlink(str(real), str(link))

        with pytest.raises(GrokNativeVersionError, match="symlink"):
            resolve_grok_native_version(cache_path=str(link), now=now)

    def test_directory_rejected(self, tmp_path: Path):
        subdir = tmp_path / "subdir"
        subdir.mkdir()

        with pytest.raises(GrokNativeVersionError, match="not a regular"):
            resolve_grok_native_version(cache_path=str(subdir))

    def test_unreadable_file_rejected(self, tmp_path: Path):
        now = time.time()
        payload = _make_payload(now=now)
        path = _write(tmp_path, payload)
        os.chmod(str(path), 0o000)

        with pytest.raises(GrokNativeVersionError, match="unreadable"):
            resolve_grok_native_version(cache_path=str(path), now=now)

    def test_open_descriptor_is_used_after_path_is_replaced_by_symlink(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        now = time.time()
        path = _write(
            tmp_path,
            _make_payload(version="1.2.3", now=now),
        )
        replacement_target = _write(
            tmp_path,
            _make_payload(version="9.9.9", now=now),
            name="replacement-target.json",
        )
        real_open = os.open
        replaced = False

        def open_then_replace(configured_path: str, flags: int) -> int:
            nonlocal replaced
            file_descriptor = real_open(configured_path, flags)
            if not replaced:
                replacement_link = tmp_path / "replacement-link"
                os.symlink(replacement_target, replacement_link)
                os.replace(replacement_link, path)
                replaced = True
            return file_descriptor

        monkeypatch.setattr(version_contract.os, "open", open_then_replace)

        record, _ = resolve_grok_native_version(
            cache_path=str(path),
            now=now,
        )
        assert record.version == "1.2.3"

        with pytest.raises(GrokNativeVersionError, match="symlink"):
            resolve_grok_native_version(cache_path=str(path), now=now)

    def test_missing_o_nofollow_fails_closed(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        path = _write(tmp_path, _make_payload())
        monkeypatch.delattr(version_contract.os, "O_NOFOLLOW")

        with pytest.raises(GrokNativeVersionError, match="unsupported"):
            resolve_grok_native_version(cache_path=str(path))

    def test_bounded_read_rejects_growth_after_fstat(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        path = tmp_path / "cache.json"
        path.write_bytes(b"x" * (GROK_VERSION_MAX_BYTES + 1))
        real_fstat = os.fstat

        def underreport_size(file_descriptor: int) -> os.stat_result:
            values = list(real_fstat(file_descriptor))
            values[6] = 0
            return os.stat_result(values)

        monkeypatch.setattr(
            version_contract.os,
            "fstat",
            underreport_size,
        )

        with pytest.raises(GrokNativeVersionError, match="exceeds"):
            resolve_grok_native_version(cache_path=str(path))


# ---------------------------------------------------------------------------
# Malformed JSON / structure
# ---------------------------------------------------------------------------


class TestMalformedRejection:
    def test_invalid_utf8_rejected_without_leaking_contents_or_path(
        self,
        tmp_path: Path,
    ):
        secret = "secret-cache-target"
        path = tmp_path / secret
        path.write_bytes(b"\xffsuper-secret-file-content")

        with pytest.raises(GrokNativeVersionError, match="UTF-8") as exc_info:
            resolve_grok_native_version(cache_path=str(path))

        error = str(exc_info.value)
        assert secret not in error
        assert "super-secret-file-content" not in error

    def test_read_error_is_sanitized(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        path = _write(tmp_path, _make_payload())

        def fail_read(_file_descriptor: int, _size: int) -> bytes:
            raise OSError("sensitive-kernel-detail")

        monkeypatch.setattr(version_contract.os, "read", fail_read)

        with pytest.raises(GrokNativeVersionError, match="could not be read") as exc_info:
            resolve_grok_native_version(cache_path=str(path))

        error = str(exc_info.value)
        assert str(path) not in error
        assert "sensitive-kernel-detail" not in error

    def test_invalid_json_rejected(self, tmp_path: Path):
        path = tmp_path / "bad.json"
        path.write_text("{not json", encoding="utf-8")

        with pytest.raises(GrokNativeVersionError, match="not valid JSON"):
            resolve_grok_native_version(cache_path=str(path))

    def test_array_root_rejected(self, tmp_path: Path):
        path = tmp_path / "array.json"
        path.write_text("[1, 2]", encoding="utf-8")

        with pytest.raises(GrokNativeVersionError, match="JSON object"):
            resolve_grok_native_version(cache_path=str(path))

    @pytest.mark.parametrize(
        "field",
        [
            "schema_version",
            "client",
            "version",
            "build",
            "channel",
            "source",
            "observed_at",
        ],
    )
    def test_missing_required_field_rejected(
        self, tmp_path: Path, field: str
    ):
        payload = _make_payload()
        del payload[field]
        path = _write(tmp_path, payload)

        with pytest.raises(GrokNativeVersionError, match="missing required"):
            resolve_grok_native_version(cache_path=str(path))

    def test_unknown_field_rejected(self, tmp_path: Path):
        payload = _make_payload()
        payload["evil"] = "injected"
        path = _write(tmp_path, payload)

        with pytest.raises(GrokNativeVersionError, match="unknown fields"):
            resolve_grok_native_version(cache_path=str(path))


# ---------------------------------------------------------------------------
# Field validation
# ---------------------------------------------------------------------------


class TestFieldValidation:
    def test_wrong_schema_version_rejected(self, tmp_path: Path):
        payload = _make_payload(schema_version=99)
        path = _write(tmp_path, payload)

        with pytest.raises(GrokNativeVersionError, match="schema_version"):
            resolve_grok_native_version(cache_path=str(path))

    def test_bool_schema_version_rejected(self, tmp_path: Path):
        payload = _make_payload(schema_version=True)
        path = _write(tmp_path, payload)

        with pytest.raises(GrokNativeVersionError, match="schema_version"):
            resolve_grok_native_version(cache_path=str(path))

    def test_wrong_client_rejected(self, tmp_path: Path):
        payload = _make_payload(client="other-cli")
        path = _write(tmp_path, payload)

        with pytest.raises(GrokNativeVersionError, match="client"):
            resolve_grok_native_version(cache_path=str(path))

    def test_wrong_source_rejected(self, tmp_path: Path):
        payload = _make_payload(source="manual")
        path = _write(tmp_path, payload)

        with pytest.raises(GrokNativeVersionError, match="source"):
            resolve_grok_native_version(cache_path=str(path))

    @pytest.mark.parametrize(
        "bad_version",
        [
            "",
            "42",
            "abc",
            "1.2.3-beta",
            "1..2",
            ".1.2",
            "1.2.",
            "v1.2.3",
            "1 2",
            " 1.2",
            "1.2 ",
            "+1.2",
            "1_0.2",
            "١.٢",
            "１.２",
        ],
    )
    def test_invalid_version_rejected(
        self, tmp_path: Path, bad_version: str
    ):
        payload = _make_payload(version=bad_version)
        path = _write(tmp_path, payload)

        with pytest.raises(GrokNativeVersionError, match="version"):
            resolve_grok_native_version(cache_path=str(path))

    @pytest.mark.parametrize(
        "bad_build",
        ["", "ABCDEF", "0x1a2b", "a1b2g3", "a1 b2", "a1\nb2"],
    )
    def test_invalid_build_rejected(self, tmp_path: Path, bad_build: str):
        payload = _make_payload(build=bad_build)
        path = _write(tmp_path, payload)

        with pytest.raises(GrokNativeVersionError, match="build"):
            resolve_grok_native_version(cache_path=str(path))

    @pytest.mark.parametrize(
        "bad_channel",
        ["", "stable channel", "ch@nnel", "a/b", "chan\nnel"],
    )
    def test_invalid_channel_rejected(
        self, tmp_path: Path, bad_channel: str
    ):
        payload = _make_payload(channel=bad_channel)
        path = _write(tmp_path, payload)

        with pytest.raises(GrokNativeVersionError, match="channel"):
            resolve_grok_native_version(cache_path=str(path))

    @pytest.mark.parametrize(
        "bad_time",
        [
            "",
            "2026-07-28T12:00:00+00:00",  # offset, not Z
            "2026-07-28 12:00:00Z",  # space, not T
            "not-a-time",
            "2026-13-40T99:99:99Z",  # invalid values
        ],
    )
    def test_invalid_observed_at_rejected(
        self, tmp_path: Path, bad_time: str
    ):
        payload = _make_payload(observed_at=bad_time)
        path = _write(tmp_path, payload)

        with pytest.raises(GrokNativeVersionError, match="observed_at"):
            resolve_grok_native_version(cache_path=str(path))


# ---------------------------------------------------------------------------
# Time validation: stale / future
# ---------------------------------------------------------------------------


class TestTimeValidation:
    def test_stale_record_rejected(self, tmp_path: Path):
        now = time.time()
        old = now - GROK_VERSION_DEFAULT_MAX_AGE_SECONDS - 10
        payload = _make_payload(observed_at=_utc_z(old), now=now)
        path = _write(tmp_path, payload)

        with pytest.raises(GrokNativeVersionError, match="stale"):
            resolve_grok_native_version(cache_path=str(path), now=now)

    def test_any_future_record_rejected(self, tmp_path: Path):
        now = 1_800_000_000.0
        future = now + 1
        payload = _make_payload(observed_at=_utc_z(future), now=now)
        path = _write(tmp_path, payload)

        with pytest.raises(GrokNativeVersionError, match="future"):
            resolve_grok_native_version(cache_path=str(path), now=now)

    def test_record_observed_exactly_now_is_accepted(self, tmp_path: Path):
        now = 1_800_000_000.0
        payload = _make_payload(observed_at=_utc_z(now), now=now)
        path = _write(tmp_path, payload)

        record, _ = resolve_grok_native_version(
            cache_path=str(path), now=now
        )
        assert record.version == "0.1.211"

    def test_max_age_env_override(self, tmp_path: Path, monkeypatch):
        now = time.time()
        old = now - 100
        payload = _make_payload(observed_at=_utc_z(old), now=now)
        path = _write(tmp_path, payload)

        # Default max age accepts this.
        record, _ = resolve_grok_native_version(
            cache_path=str(path), now=now
        )
        assert record.version == "0.1.211"

        # Tighter max age rejects it.
        monkeypatch.setenv(GROK_VERSION_CACHE_MAX_AGE_ENV, "50")
        with pytest.raises(GrokNativeVersionError, match="stale"):
            resolve_grok_native_version(cache_path=str(path), now=now)

    @pytest.mark.parametrize(
        "bad_val",
        [
            "",
            "0",
            "00",
            "01",
            "+1",
            "-1",
            " 1",
            "1 ",
            "1_0",
            "١",
            "１",
            "abc",
        ],
    )
    def test_invalid_max_age_env_rejected(
        self, tmp_path: Path, monkeypatch, bad_val: str
    ):
        now = time.time()
        payload = _make_payload(now=now)
        path = _write(tmp_path, payload)
        monkeypatch.setenv(GROK_VERSION_CACHE_MAX_AGE_ENV, bad_val)

        with pytest.raises(GrokNativeVersionError, match="ASCII decimal"):
            resolve_grok_native_version(cache_path=str(path), now=now)


# ---------------------------------------------------------------------------
# Environment path configuration
# ---------------------------------------------------------------------------


class TestEnvPathConfig:
    @pytest.mark.parametrize("configured_path", ["cache.json", "./cache.json", ""])
    def test_relative_or_empty_explicit_path_rejected(
        self,
        configured_path: str,
    ):
        with pytest.raises(GrokNativeVersionError, match="absolute"):
            resolve_grok_native_version(cache_path=configured_path)

    @pytest.mark.parametrize("configured_path", ["cache.json", ""])
    def test_relative_or_empty_env_path_rejected(
        self,
        monkeypatch: pytest.MonkeyPatch,
        configured_path: str,
    ):
        monkeypatch.setenv(GROK_VERSION_CACHE_PATH_ENV, configured_path)

        with pytest.raises(GrokNativeVersionError, match="absolute"):
            resolve_grok_native_version()

    def test_env_path_used(self, tmp_path: Path, monkeypatch):
        now = time.time()
        payload = _make_payload(now=now)
        path = _write(tmp_path, payload)
        monkeypatch.setenv(GROK_VERSION_CACHE_PATH_ENV, str(path))

        record, meta = resolve_grok_native_version(now=now)
        assert record.version == "0.1.211"
        assert meta.cache_path == str(path)

    def test_default_path_constant(self):
        assert (
            GROK_VERSION_DEFAULT_CACHE_PATH
            == "/run/aawm/grok/native-client-version.json"
        )


# ---------------------------------------------------------------------------
# Atomic replacement (no caching across calls)
# ---------------------------------------------------------------------------


class TestAtomicReplacement:
    def test_replacement_observed_on_next_call(self, tmp_path: Path):
        now = time.time()
        v1 = _make_payload(version="1.0.0", now=now)
        path = _write(tmp_path, v1)

        r1, _ = resolve_grok_native_version(cache_path=str(path), now=now)
        assert r1.version == "1.0.0"

        # Atomic replacement.
        v2 = _make_payload(version="2.0.0", now=now)
        tmp_file = tmp_path / "cache.json.tmp"
        tmp_file.write_text(json.dumps(v2), encoding="utf-8")
        os.replace(str(tmp_file), str(path))

        r2, _ = resolve_grok_native_version(cache_path=str(path), now=now)
        assert r2.version == "2.0.0"

    def test_no_caching_across_calls(self, tmp_path: Path):
        """Each call re-reads; changing the file changes the result."""
        now = time.time()
        path = _write(tmp_path, _make_payload(version="1.0.0", now=now))

        for expected in ("1.0.0", "3.0.0", "7.7.7"):
            payload = _make_payload(version=expected, now=now)
            tmp_file = tmp_path / "swap.json"
            tmp_file.write_text(json.dumps(payload), encoding="utf-8")
            os.replace(str(tmp_file), str(path))

            record, _ = resolve_grok_native_version(
                cache_path=str(path), now=now
            )
            assert record.version == expected


# ---------------------------------------------------------------------------
# try_resolve non-raising variant
# ---------------------------------------------------------------------------


class TestTryResolve:
    def test_returns_none_on_missing(self, tmp_path: Path):
        result = try_resolve_grok_native_version(
            cache_path=str(tmp_path / "nope.json")
        )
        assert result is None

    def test_returns_record_on_valid(self, tmp_path: Path):
        now = time.time()
        path = _write(tmp_path, _make_payload(now=now))
        result = try_resolve_grok_native_version(
            cache_path=str(path), now=now
        )
        assert result is not None
        assert result[0].version == "0.1.211"


# ---------------------------------------------------------------------------
# Error message safety
# ---------------------------------------------------------------------------


class TestErrorSafety:
    def test_error_does_not_leak_file_contents(self, tmp_path: Path):
        secret = "super-secret-api-key-12345"
        path = tmp_path / "cache.json"
        path.write_text(
            json.dumps({"secret": secret, "schema_version": 1}),
            encoding="utf-8",
        )

        with pytest.raises(GrokNativeVersionError) as exc_info:
            resolve_grok_native_version(cache_path=str(path))

        assert secret not in str(exc_info.value)
