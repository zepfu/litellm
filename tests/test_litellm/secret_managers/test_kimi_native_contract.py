"""Focused tests for the Kimi native contract descriptor module."""

import json
import multiprocessing
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pytest

from litellm.secret_managers.kimi_native_contract import (
    KIMI_NATIVE_BASE_URL,
    KIMI_NATIVE_BUILTIN_CLIENT_VERSION,
    KIMI_NATIVE_BUILTIN_DEVICE_ID,
    KIMI_NATIVE_CONTRACT_MAX_BYTES,
    KIMI_NATIVE_CONTRACT_PATH_ENV,
    KIMI_NATIVE_CONTRACT_REQUIRED_ENV,
    KIMI_NATIVE_CONTRACT_SOURCE_BUILTIN,
    KIMI_NATIVE_CONTRACT_SOURCE_DESCRIPTOR,
    KIMI_NATIVE_CONTRACT_SOURCE_STALE,
    KIMI_NATIVE_SCHEMA_VERSION,
    KimiNativeContractError,
    _reset_source_telemetry_state,
    build_outbound_headers,
    compute_canonical_digest,
    resolve_contract,
    resolve_endpoint_url,
)


@pytest.fixture(autouse=True)
def _fresh_source_telemetry_state():
    _reset_source_telemetry_state()
    yield
    _reset_source_telemetry_state()


def _make_payload(
    *,
    client_name: str = "kimi-code",
    client_version: str = "0.29.1",
    user_agent: str = "kimi-code-cli/0.29.1",
    base_url: str = KIMI_NATIVE_BASE_URL,
    issued_at: float | None = None,
    expires_at: float | None = None,
    x_msh_platform: str = "kimi_code_cli",
    x_msh_version: str = "0.29.1",
    x_msh_device_name: str = "aawm-service-node",
    x_msh_device_model: str = "aawm-managed",
    x_msh_os_version: str = "linux-6.x",
    x_msh_device_id: str = "0d3f8a2e-7b14-4c6a-9e5f-a1b2c3d4e5f6",
    **overrides,
) -> dict:
    now = time.time()
    payload = {
        "schema_version": KIMI_NATIVE_SCHEMA_VERSION,
        "client_name": client_name,
        "client_version": client_version,
        "base_url": base_url,
        "user_agent": user_agent,
        "issued_at": issued_at if issued_at is not None else now - 60,
        "expires_at": expires_at if expires_at is not None else now + 3600,
        "x_msh_platform": x_msh_platform,
        "x_msh_version": x_msh_version,
        "x_msh_device_name": x_msh_device_name,
        "x_msh_device_model": x_msh_device_model,
        "x_msh_os_version": x_msh_os_version,
        "x_msh_device_id": x_msh_device_id,
    }
    payload.update(overrides)
    payload["digest"] = compute_canonical_digest(payload)
    return payload


def _write_contract(tmp_path: Path, payload: dict) -> Path:
    path = tmp_path / "contract.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Schema and digest
# ---------------------------------------------------------------------------


class TestSchemaAndDigest:
    def test_valid_contract_resolves(self, tmp_path: Path):
        payload = _make_payload()
        path = _write_contract(tmp_path, payload)

        contract = resolve_contract(str(path), required=True)

        assert contract is not None
        assert contract.schema_version == KIMI_NATIVE_SCHEMA_VERSION
        assert contract.client_name == "kimi-code"
        assert contract.client_version == "0.29.1"
        assert contract.base_url == KIMI_NATIVE_BASE_URL
        assert contract.user_agent == "kimi-code-cli/0.29.1"
        assert contract.digest.startswith("sha256:")

    def test_digest_is_deterministic(self):
        payload = _make_payload()
        d1 = compute_canonical_digest(payload)
        d2 = compute_canonical_digest(payload)
        assert d1 == d2

    def test_digest_excludes_digest_field(self):
        payload = _make_payload()
        digest_without = compute_canonical_digest(
            {k: v for k, v in payload.items() if k != "digest"}
        )
        assert payload["digest"] == digest_without

    def test_tampered_payload_falls_back_to_builtin(self, tmp_path: Path):
        """MS-035: digest-invalid descriptors fall back to the conservative
        builtin identity even when required=true."""
        payload = _make_payload()
        payload["x_msh_device_name"] = "tampered-node"  # tamper after digest
        path = _write_contract(tmp_path, payload)

        contract = resolve_contract(str(path), required=True)
        assert contract is not None
        assert contract.source == KIMI_NATIVE_CONTRACT_SOURCE_BUILTIN

    @pytest.mark.parametrize(
        "field",
        [
            "schema_version",
            "client_name",
            "client_version",
            "base_url",
            "user_agent",
            "issued_at",
            "expires_at",
            "digest",
        ],
    )
    def test_missing_required_field_falls_back_to_builtin(
        self, tmp_path: Path, field: str
    ):
        payload = _make_payload()
        del payload[field]
        # Recompute digest without the field so digest itself is valid
        # (skip when the deleted field IS digest)
        if field != "digest":
            payload["digest"] = compute_canonical_digest(payload)
        path = _write_contract(tmp_path, payload)

        contract = resolve_contract(str(path), required=True)
        assert contract is not None
        assert contract.source == KIMI_NATIVE_CONTRACT_SOURCE_BUILTIN

    def test_unknown_field_falls_back_to_builtin(self, tmp_path: Path):
        payload = _make_payload()
        payload["evil_field"] = "injected"
        payload["digest"] = compute_canonical_digest(payload)
        path = _write_contract(tmp_path, payload)

        contract = resolve_contract(str(path), required=True)
        assert contract is not None
        assert contract.source == KIMI_NATIVE_CONTRACT_SOURCE_BUILTIN

    def test_wrong_schema_version_falls_back_to_builtin(self, tmp_path: Path):
        payload = _make_payload()
        payload["schema_version"] = 99
        payload["digest"] = compute_canonical_digest(payload)
        path = _write_contract(tmp_path, payload)

        contract = resolve_contract(str(path), required=True)
        assert contract is not None
        assert contract.source == KIMI_NATIVE_CONTRACT_SOURCE_BUILTIN


# ---------------------------------------------------------------------------
# Stale / missing / hostile rejection
# ---------------------------------------------------------------------------


class TestStaleMissingHostile:
    def test_expired_contract_resolves_with_stale_source(self, tmp_path: Path):
        """MS-035: an expired but structurally valid descriptor stays usable.

        The resolver returns the descriptor's older claimed identity with a
        ``stale`` source classification instead of failing the route.
        """
        payload = _make_payload(expires_at=time.time() - 10)
        path = _write_contract(tmp_path, payload)

        contract = resolve_contract(str(path), required=True)

        assert contract is not None
        assert contract.source == KIMI_NATIVE_CONTRACT_SOURCE_STALE
        assert contract.client_version == "0.29.1"
        assert contract.user_agent == "kimi-code-cli/0.29.1"
        assert contract.x_msh_platform == "kimi_code_cli"

    def test_fresh_contract_resolves_with_descriptor_source(self, tmp_path: Path):
        payload = _make_payload()
        path = _write_contract(tmp_path, payload)

        contract = resolve_contract(str(path), required=True)

        assert contract is not None
        assert contract.source == KIMI_NATIVE_CONTRACT_SOURCE_DESCRIPTOR

    def test_expired_contract_not_required_keeps_stale_source(self, tmp_path: Path):
        payload = _make_payload(expires_at=time.time() - 10)
        path = _write_contract(tmp_path, payload)

        contract = resolve_contract(str(path), required=False)

        assert contract is not None
        assert contract.source == KIMI_NATIVE_CONTRACT_SOURCE_STALE

    def test_stale_source_telemetry_is_sanitized_and_deduplicated(
        self, tmp_path: Path, caplog
    ):
        """Stale telemetry names only the classification and path, is emitted
        once per transition, and never leaks descriptor body contents."""
        import logging

        payload = _make_payload(expires_at=time.time() - 10)
        path = _write_contract(tmp_path, payload)

        logger = logging.getLogger("litellm.secret_managers.kimi_native_contract")
        with caplog.at_level(logging.WARNING, logger=logger.name):
            resolve_contract(str(path), required=True)
            resolve_contract(str(path), required=True)

        warnings = [
            record
            for record in caplog.records
            if record.name == logger.name and record.levelno == logging.WARNING
        ]
        assert len(warnings) == 1
        message = warnings[0].getMessage()
        assert "source=stale" in message
        assert str(path) in message
        assert "0.29.1" not in message
        assert "aawm-service-node" not in message
        assert "0d3f8a2e" not in message

    def test_builtin_source_telemetry_is_sanitized_and_deduplicated(
        self, tmp_path: Path, caplog
    ):
        import logging

        logger = logging.getLogger("litellm.secret_managers.kimi_native_contract")
        with caplog.at_level(logging.WARNING, logger=logger.name):
            resolve_contract(str(tmp_path / "absent.json"), required=True)
            resolve_contract(str(tmp_path / "absent.json"), required=True)

        warnings = [
            record
            for record in caplog.records
            if record.name == logger.name and record.levelno == logging.WARNING
        ]
        assert len(warnings) == 1
        message = warnings[0].getMessage()
        assert "source=builtin" in message
        assert "aawm-service-node" not in message

    def test_builtin_identity_is_coherent_and_conservative(self, tmp_path: Path):
        """The built-in fallback identity keeps UA/X-Msh coherence and never
        fabricates a digest."""
        contract = resolve_contract(str(tmp_path / "absent.json"), required=True)

        assert contract is not None
        assert contract.source == KIMI_NATIVE_CONTRACT_SOURCE_BUILTIN
        assert contract.client_name == "kimi-code"
        assert contract.user_agent == f"kimi-code-cli/{contract.client_version}"
        assert contract.x_msh_platform == "kimi_code_cli"
        assert contract.x_msh_version == contract.client_version
        assert contract.base_url == KIMI_NATIVE_BASE_URL
        assert contract.digest == ""
        assert contract.client_version  # non-empty conservative identity

    def test_builtin_identity_uses_pinned_defaults_when_lookup_unavailable(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """With no discoverable installed client, the conservative pinned
        floor identity is used (no network, no writes)."""
        monkeypatch.setenv("KIMI_CODE_HOME", str(tmp_path / "no-kimi-home"))

        contract = resolve_contract(str(tmp_path / "absent.json"), required=True)

        assert contract is not None
        assert contract.source == KIMI_NATIVE_CONTRACT_SOURCE_BUILTIN
        assert contract.client_version == KIMI_NATIVE_BUILTIN_CLIENT_VERSION
        assert contract.x_msh_device_id == KIMI_NATIVE_BUILTIN_DEVICE_ID

    def test_builtin_identity_prefers_installed_client_version_and_device(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """When the installed client exposes a version/device ID, the built-in
        identity derives from it (installed-client contract, read-only)."""
        import subprocess

        kimi_home = tmp_path / "kimi-home"
        kimi_home.mkdir()
        (kimi_home / "device_id").write_text(
            "11111111-2222-3333-4444-555555555555", encoding="utf-8"
        )
        monkeypatch.setenv("KIMI_CODE_HOME", str(kimi_home))

        real_run = subprocess.run

        def fake_run(*args, **kwargs):
            class Result:
                stdout = "0.31.0\n"

            return Result()

        monkeypatch.setattr(subprocess, "run", fake_run)
        try:
            contract = resolve_contract(str(tmp_path / "absent.json"), required=True)
        finally:
            monkeypatch.setattr(subprocess, "run", real_run)

        assert contract is not None
        assert contract.source == KIMI_NATIVE_CONTRACT_SOURCE_BUILTIN
        assert contract.client_version == "0.31.0"
        assert contract.user_agent == "kimi-code-cli/0.31.0"
        assert contract.x_msh_device_id == "11111111-2222-3333-4444-555555555555"

    def test_future_issued_at_falls_back_to_builtin(self, tmp_path: Path):
        payload = _make_payload(issued_at=time.time() + 600)
        path = _write_contract(tmp_path, payload)

        contract = resolve_contract(str(path), required=True)
        assert contract is not None
        assert contract.source == KIMI_NATIVE_CONTRACT_SOURCE_BUILTIN

    def test_missing_file_returns_none_when_not_required(self, tmp_path: Path):
        result = resolve_contract(str(tmp_path / "nonexistent.json"))
        assert result is None

    def test_missing_file_resolves_builtin_when_required(self, tmp_path: Path):
        """MS-035: a missing descriptor never terminates Kimi solely for
        contract unavailability; the conservative built-in identity is used."""
        contract = resolve_contract(str(tmp_path / "nonexistent.json"), required=True)

        assert contract is not None
        assert contract.source == KIMI_NATIVE_CONTRACT_SOURCE_BUILTIN

    def test_non_regular_file_rejected_when_required(self, tmp_path: Path):
        dir_path = tmp_path / "subdir"
        dir_path.mkdir()

        contract = resolve_contract(str(dir_path), required=True)
        assert contract is not None
        assert contract.source == KIMI_NATIVE_CONTRACT_SOURCE_BUILTIN

    def test_oversized_file_rejected_when_required(self, tmp_path: Path):
        path = tmp_path / "big.json"
        path.write_text("x" * (KIMI_NATIVE_CONTRACT_MAX_BYTES + 1), encoding="utf-8")

        contract = resolve_contract(str(path), required=True)
        assert contract is not None
        assert contract.source == KIMI_NATIVE_CONTRACT_SOURCE_BUILTIN

    def test_invalid_json_rejected_when_required(self, tmp_path: Path):
        path = tmp_path / "bad.json"
        path.write_text("{not json", encoding="utf-8")

        contract = resolve_contract(str(path), required=True)
        assert contract is not None
        assert contract.source == KIMI_NATIVE_CONTRACT_SOURCE_BUILTIN

    def test_non_object_root_rejected_when_required(self, tmp_path: Path):
        path = tmp_path / "array.json"
        path.write_text("[1, 2, 3]", encoding="utf-8")

        contract = resolve_contract(str(path), required=True)
        assert contract is not None
        assert contract.source == KIMI_NATIVE_CONTRACT_SOURCE_BUILTIN

    def test_wrong_base_url_falls_back_to_builtin(self, tmp_path: Path):
        payload = _make_payload(base_url="https://api.moonshot.ai/v1")
        path = _write_contract(tmp_path, payload)

        contract = resolve_contract(str(path), required=True)
        assert contract is not None
        assert contract.source == KIMI_NATIVE_CONTRACT_SOURCE_BUILTIN

    def test_empty_client_name_falls_back_to_builtin(self, tmp_path: Path):
        payload = _make_payload(client_name="  ")
        path = _write_contract(tmp_path, payload)

        contract = resolve_contract(str(path), required=True)
        assert contract is not None
        assert contract.source == KIMI_NATIVE_CONTRACT_SOURCE_BUILTIN


# ---------------------------------------------------------------------------
# Atomic replacement hot read
# ---------------------------------------------------------------------------


def _resolve_contract_snapshot(path: str) -> dict[str, str]:
    """Independent-process helper: resolve a descriptor and return key fields.

    Runs in a fresh interpreter so the read cannot share parent-process
    module state, caches, or open file handles with the writer.
    """
    from litellm.secret_managers.kimi_native_contract import resolve_contract

    contract = resolve_contract(path, required=True)
    assert contract is not None
    return {
        "client_version": contract.client_version,
        "user_agent": contract.user_agent,
        "x_msh_version": contract.x_msh_version,
        "x_msh_device_id": contract.x_msh_device_id,
        "digest": contract.digest,
    }


class TestAtomicReplacement:
    def test_hot_read_after_atomic_replace(self, tmp_path: Path):
        payload_v1 = _make_payload(
            client_version="1.0.0",
            user_agent="kimi-code-cli/1.0.0",
            x_msh_version="1.0.0",
            x_msh_device_id="11111111-1111-4111-8111-111111111111",
        )
        path = _write_contract(tmp_path, payload_v1)

        c1 = resolve_contract(str(path), required=True)
        assert c1 is not None
        assert c1.client_version == "1.0.0"
        assert c1.user_agent == "kimi-code-cli/1.0.0"
        assert c1.x_msh_version == "1.0.0"
        assert c1.x_msh_device_id == "11111111-1111-4111-8111-111111111111"

        # Atomic replacement: write to temp then rename
        payload_v2 = _make_payload(
            client_version="2.0.0",
            user_agent="kimi-code-cli/2.0.0",
            x_msh_version="2.0.0",
            x_msh_device_id="22222222-2222-4222-8222-222222222222",
        )
        tmp_file = tmp_path / "contract.json.tmp"
        tmp_file.write_text(json.dumps(payload_v2), encoding="utf-8")
        os.replace(str(tmp_file), str(path))

        c2 = resolve_contract(str(path), required=True)
        assert c2 is not None
        assert c2.client_version == "2.0.0"
        assert c2.user_agent == "kimi-code-cli/2.0.0"
        assert c2.x_msh_version == "2.0.0"
        assert c2.x_msh_device_id == "22222222-2222-4222-8222-222222222222"
        # Coherent version/UA/X-Msh values flip together after the replace.
        assert c2.user_agent == f"kimi-code-cli/{c2.client_version}"
        assert c2.x_msh_version == c2.client_version
        assert c2.digest != c1.digest

    def test_atomic_replace_observed_by_independent_readers(self, tmp_path: Path):
        """Separate processes only ever observe full v1 or full v2 descriptors.

        Proves multi-worker restart-free activation: an atomic os.replace of a
        coherent version/UA/X-Msh descriptor is never seen as a partial mix of
        the old and new contracts by independent readers.
        """
        payload_v1 = _make_payload(
            client_version="1.0.0",
            user_agent="kimi-code-cli/1.0.0",
            x_msh_version="1.0.0",
            x_msh_device_id="aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa",
        )
        payload_v2 = _make_payload(
            client_version="2.0.0",
            user_agent="kimi-code-cli/2.0.0",
            x_msh_version="2.0.0",
            x_msh_device_id="bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb",
        )
        path = _write_contract(tmp_path, payload_v1)
        path_str = str(path)

        expected_v1 = {
            "client_version": "1.0.0",
            "user_agent": "kimi-code-cli/1.0.0",
            "x_msh_version": "1.0.0",
            "x_msh_device_id": "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa",
            "digest": payload_v1["digest"],
        }
        expected_v2 = {
            "client_version": "2.0.0",
            "user_agent": "kimi-code-cli/2.0.0",
            "x_msh_version": "2.0.0",
            "x_msh_device_id": "bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb",
            "digest": payload_v2["digest"],
        }

        # Prefer processes; fall back to threads only if the platform cannot
        # spawn workers (still proves independent resolve_contract calls).
        try:
            ctx = multiprocessing.get_context("spawn")
            executor = ProcessPoolExecutor(max_workers=4, mp_context=ctx)
        except (ValueError, OSError):  # pragma: no cover - platform fallback
            from concurrent.futures import ThreadPoolExecutor

            executor = ThreadPoolExecutor(max_workers=4)

        snapshots: list[dict[str, str]] = []
        try:
            # Baseline independent readers see the full v1 descriptor.
            futures = [
                executor.submit(_resolve_contract_snapshot, path_str)
                for _ in range(4)
            ]
            for future in as_completed(futures):
                snapshots.append(future.result(timeout=30))

            for snap in snapshots:
                assert snap == expected_v1

            # Atomic publication of the full coherent v2 descriptor.
            tmp_file = tmp_path / "contract.json.hot"
            tmp_file.write_text(json.dumps(payload_v2), encoding="utf-8")
            os.replace(str(tmp_file), path_str)

            # Fresh independent readers only observe full v2 (no mix of v1/v2).
            post_futures = [
                executor.submit(_resolve_contract_snapshot, path_str)
                for _ in range(8)
            ]
            post_snapshots = [
                future.result(timeout=30) for future in as_completed(post_futures)
            ]
        finally:
            executor.shutdown(wait=True, cancel_futures=True)

        assert post_snapshots
        for snap in post_snapshots:
            assert snap == expected_v2
            # Coherence: never a partial cross of v1 version with v2 UA/X-Msh.
            assert snap["user_agent"] == f"kimi-code-cli/{snap['client_version']}"
            assert snap["x_msh_version"] == snap["client_version"]
            assert snap["digest"] == expected_v2["digest"]
            assert snap != expected_v1


# ---------------------------------------------------------------------------
# URL joins
# ---------------------------------------------------------------------------


class TestUrlJoins:
    @pytest.mark.parametrize(
        ("usage", "endpoint_path"),
        [
            ("models", "models"),
            ("usages", "usages"),
            ("chat_completions", "chat/completions"),
        ],
    )
    def test_supported_url_from_contract(
        self,
        tmp_path: Path,
        usage: str,
        endpoint_path: str,
    ):
        payload = _make_payload()
        path = _write_contract(tmp_path, payload)
        contract = resolve_contract(str(path), required=True)

        url = resolve_endpoint_url(contract, usage)
        assert url == f"https://api.kimi.com/coding/v1/{endpoint_path}"

    @pytest.mark.parametrize(
        ("usage", "endpoint_path"),
        [
            ("models", "models"),
            ("usages", "usages"),
            ("chat_completions", "chat/completions"),
        ],
    )
    def test_supported_url_fallback_without_contract(
        self,
        usage: str,
        endpoint_path: str,
    ):
        url = resolve_endpoint_url(None, usage)
        assert url == f"https://api.kimi.com/coding/v1/{endpoint_path}"

    @pytest.mark.parametrize(
        "usage",
        ["", "model", "usage", "embeddings", "chat/completions", "MODELS"],
    )
    def test_unknown_usage_raises(self, usage: str):
        with pytest.raises(ValueError, match="unknown contract usage"):
            resolve_endpoint_url(None, usage)

    @pytest.mark.parametrize(
        "usage",
        ["models", "usages", "chat_completions"],
    )
    def test_no_generic_moonshot_base(self, usage: str):
        """The fallback base must never be api.moonshot.ai."""
        url = resolve_endpoint_url(None, usage)
        assert "moonshot" not in url
        assert url.startswith(KIMI_NATIVE_BASE_URL)


# ---------------------------------------------------------------------------
# Caller spoof isolation / header builder
# ---------------------------------------------------------------------------


class TestHeaderBuilder:
    def test_emits_only_expected_headers(self, tmp_path: Path):
        payload = _make_payload(user_agent="kimi-code-cli/0.29.1")
        path = _write_contract(tmp_path, payload)
        contract = resolve_contract(str(path), required=True)

        headers = build_outbound_headers(
            contract, "my-token", json_body=True
        )

        assert headers == {
            "User-Agent": "kimi-code-cli/0.29.1",
            "Authorization": "Bearer my-token",
            "Content-Type": "application/json",
            "X-Msh-Platform": "kimi_code_cli",
            "X-Msh-Version": "0.29.1",
            "X-Msh-Device-Name": "aawm-service-node",
            "X-Msh-Device-Model": "aawm-managed",
            "X-Msh-Os-Version": "linux-6.x",
            "X-Msh-Device-Id": "0d3f8a2e-7b14-4c6a-9e5f-a1b2c3d4e5f6",
        }

    def test_no_content_type_without_json_body(self, tmp_path: Path):
        payload = _make_payload()
        path = _write_contract(tmp_path, payload)
        contract = resolve_contract(str(path), required=True)

        headers = build_outbound_headers(contract, "tok", json_body=False)
        assert "Content-Type" not in headers
        assert "Accept" not in headers

    def test_accept_json_emits_accept_header(self, tmp_path: Path):
        payload = _make_payload()
        path = _write_contract(tmp_path, payload)
        contract = resolve_contract(str(path), required=True)

        headers = build_outbound_headers(
            contract, "tok", json_body=False, accept_json=True
        )
        assert headers["Accept"] == "application/json"
        assert "Content-Type" not in headers

    def test_accept_json_without_contract(self):
        headers = build_outbound_headers(
            None, "tok", json_body=False, accept_json=True
        )
        assert headers["Accept"] == "application/json"

    def test_no_authorization_without_token(self, tmp_path: Path):
        payload = _make_payload()
        path = _write_contract(tmp_path, payload)
        contract = resolve_contract(str(path), required=True)

        headers = build_outbound_headers(contract, None, json_body=True)
        assert "Authorization" not in headers

    def test_fallback_user_agent_without_contract(self):
        headers = build_outbound_headers(
            None, "tok", json_body=True, fallback_user_agent="litellm/1.0"
        )
        assert headers["User-Agent"] == "litellm/1.0"
        # Without a contract, no X-Msh headers are emitted.
        assert not any(k.lower().startswith("x-msh") for k in headers)

    def test_empty_token_rejected(self):
        with pytest.raises(KimiNativeContractError, match="non-empty"):
            build_outbound_headers(None, "  ", json_body=True)

    def test_x_msh_headers_from_descriptor_are_sanitized(self, tmp_path: Path):
        """X-Msh headers come exclusively from the descriptor, not callers."""
        payload = _make_payload()
        path = _write_contract(tmp_path, payload)
        contract = resolve_contract(str(path), required=True)

        headers = build_outbound_headers(contract, "tok", json_body=True)
        # Exactly the six descriptor-controlled X-Msh headers.
        x_msh = {k: v for k, v in headers.items() if k.lower().startswith("x-msh")}
        assert x_msh == {
            "X-Msh-Platform": "kimi_code_cli",
            "X-Msh-Version": "0.29.1",
            "X-Msh-Device-Name": "aawm-service-node",
            "X-Msh-Device-Model": "aawm-managed",
            "X-Msh-Os-Version": "linux-6.x",
            "X-Msh-Device-Id": "0d3f8a2e-7b14-4c6a-9e5f-a1b2c3d4e5f6",
        }
        # No session, forwarded, or api-key headers.
        for key in headers:
            lower = key.lower()
            assert "session" not in lower
            assert "api-key" not in lower
            assert "forwarded" not in lower

    def test_no_x_msh_without_contract(self):
        """Without a descriptor, no X-Msh headers are emitted."""
        headers = build_outbound_headers(None, "tok", json_body=True)
        assert not any(k.lower().startswith("x-msh") for k in headers)


# ---------------------------------------------------------------------------
# Deployment gate
# ---------------------------------------------------------------------------


class TestDeploymentGate:
    def test_no_path_not_required_returns_none(self, monkeypatch):
        monkeypatch.delenv(KIMI_NATIVE_CONTRACT_PATH_ENV, raising=False)
        monkeypatch.delenv(KIMI_NATIVE_CONTRACT_REQUIRED_ENV, raising=False)

        assert resolve_contract() is None

    def test_no_path_required_resolves_builtin(self, monkeypatch):
        """MS-035: required=true with no configured path still resolves the
        conservative built-in identity; absence never fails the route."""
        monkeypatch.delenv(KIMI_NATIVE_CONTRACT_PATH_ENV, raising=False)
        monkeypatch.setenv(KIMI_NATIVE_CONTRACT_REQUIRED_ENV, "true")

        contract = resolve_contract()

        assert contract is not None
        assert contract.source == KIMI_NATIVE_CONTRACT_SOURCE_BUILTIN

    def test_env_path_used_when_no_explicit_path(
        self, tmp_path: Path, monkeypatch
    ):
        payload = _make_payload()
        path = _write_contract(tmp_path, payload)
        monkeypatch.setenv(KIMI_NATIVE_CONTRACT_PATH_ENV, str(path))
        monkeypatch.delenv(KIMI_NATIVE_CONTRACT_REQUIRED_ENV, raising=False)

        contract = resolve_contract()
        assert contract is not None
        assert contract.user_agent == "kimi-code-cli/0.29.1"

    def test_required_env_values(self, tmp_path: Path, monkeypatch):
        payload = _make_payload()
        path = _write_contract(tmp_path, payload)

        for val in ("1", "true", "True", "yes"):
            monkeypatch.setenv(KIMI_NATIVE_CONTRACT_REQUIRED_ENV, val)
            contract = resolve_contract(str(path))
            assert contract is not None

    def test_invalid_contract_not_required_returns_none(self, tmp_path: Path):
        path = tmp_path / "bad.json"
        path.write_text("not json", encoding="utf-8")

        assert resolve_contract(str(path), required=False) is None


# ---------------------------------------------------------------------------
# ISO-8601 timestamp parsing
# ---------------------------------------------------------------------------


class TestTimestamps:
    def test_iso8601_with_z(self, tmp_path: Path):
        now = time.time()
        payload = _make_payload(
            issued_at="2020-01-01T00:00:00Z",
            expires_at="2099-01-01T00:00:00Z",
        )
        path = _write_contract(tmp_path, payload)
        contract = resolve_contract(str(path), required=True, now=now)
        assert contract is not None

    def test_iso8601_with_offset(self, tmp_path: Path):
        now = time.time()
        payload = _make_payload(
            issued_at="2020-01-01T01:00:00+01:00",
            expires_at="2099-01-01T01:00:00+01:00",
        )
        path = _write_contract(tmp_path, payload)
        contract = resolve_contract(str(path), required=True, now=now)
        assert contract is not None

    def test_epoch_milliseconds(self, tmp_path: Path):
        now = time.time()
        payload = _make_payload(
            issued_at=(now - 60) * 1000,
            expires_at=(now + 3600) * 1000,
        )
        path = _write_contract(tmp_path, payload)
        contract = resolve_contract(str(path), required=True, now=now)
        assert contract is not None


# ---------------------------------------------------------------------------
# X-Msh descriptor field validation
# ---------------------------------------------------------------------------


class TestXMshFieldValidation:
    @pytest.mark.parametrize(
        "field",
        [
            "x_msh_platform",
            "x_msh_version",
            "x_msh_device_name",
            "x_msh_device_model",
            "x_msh_os_version",
            "x_msh_device_id",
        ],
    )
    def test_empty_x_msh_field_falls_back_to_builtin(self, tmp_path: Path, field: str):
        payload = _make_payload(**{field: ""})
        path = _write_contract(tmp_path, payload)

        contract = resolve_contract(str(path), required=True)
        assert contract is not None
        assert contract.source == KIMI_NATIVE_CONTRACT_SOURCE_BUILTIN

    @pytest.mark.parametrize(
        "field",
        [
            "x_msh_platform",
            "x_msh_version",
            "x_msh_device_name",
            "x_msh_device_model",
            "x_msh_os_version",
            "x_msh_device_id",
        ],
    )
    def test_non_ascii_x_msh_field_falls_back_to_builtin(self, tmp_path: Path, field: str):
        payload = _make_payload(**{field: "caf\u00e9-\u00fc"})
        path = _write_contract(tmp_path, payload)

        contract = resolve_contract(str(path), required=True)
        assert contract is not None
        assert contract.source == KIMI_NATIVE_CONTRACT_SOURCE_BUILTIN

    def test_client_coherence(self, tmp_path: Path):
        """client_name, client_version, and user_agent must be coherent."""
        payload = _make_payload()
        path = _write_contract(tmp_path, payload)
        contract = resolve_contract(str(path), required=True)

        assert contract.client_name == "kimi-code"
        assert contract.client_version == "0.29.1"
        assert contract.user_agent == f"kimi-code-cli/{contract.client_version}"
        assert contract.x_msh_version == contract.client_version

    def test_coherence_works_for_future_versions(self, tmp_path: Path):
        """Coherence is dynamic, not hardcoded to 0.29.1."""
        payload = _make_payload(
            client_version="1.99.0",
            user_agent="kimi-code-cli/1.99.0",
            x_msh_version="1.99.0",
        )
        path = _write_contract(tmp_path, payload)
        contract = resolve_contract(str(path), required=True)

        assert contract.client_version == "1.99.0"
        assert contract.user_agent == "kimi-code-cli/1.99.0"
        assert contract.x_msh_version == "1.99.0"

    def test_wrong_client_name_falls_back_to_builtin(self, tmp_path: Path):
        payload = _make_payload(client_name="kimi-code-fork")
        path = _write_contract(tmp_path, payload)

        contract = resolve_contract(str(path), required=True)
        assert contract is not None
        assert contract.source == KIMI_NATIVE_CONTRACT_SOURCE_BUILTIN

    def test_incoherent_user_agent_falls_back_to_builtin(self, tmp_path: Path):
        payload = _make_payload(user_agent="kimi-code/0.29.1")
        path = _write_contract(tmp_path, payload)

        contract = resolve_contract(str(path), required=True)
        assert contract is not None
        assert contract.source == KIMI_NATIVE_CONTRACT_SOURCE_BUILTIN

    def test_user_agent_version_mismatch_falls_back_to_builtin(self, tmp_path: Path):
        payload = _make_payload(user_agent="kimi-code-cli/0.28.0")
        path = _write_contract(tmp_path, payload)

        contract = resolve_contract(str(path), required=True)
        assert contract is not None
        assert contract.source == KIMI_NATIVE_CONTRACT_SOURCE_BUILTIN

    def test_wrong_x_msh_platform_falls_back_to_builtin(self, tmp_path: Path):
        payload = _make_payload(x_msh_platform="kimi_desktop")
        path = _write_contract(tmp_path, payload)

        contract = resolve_contract(str(path), required=True)
        assert contract is not None
        assert contract.source == KIMI_NATIVE_CONTRACT_SOURCE_BUILTIN

    def test_x_msh_version_mismatch_falls_back_to_builtin(self, tmp_path: Path):
        payload = _make_payload(x_msh_version="0.28.0")
        path = _write_contract(tmp_path, payload)

        contract = resolve_contract(str(path), required=True)
        assert contract is not None
        assert contract.source == KIMI_NATIVE_CONTRACT_SOURCE_BUILTIN

    def test_non_uuid_device_id_falls_back_to_builtin(self, tmp_path: Path):
        payload = _make_payload(x_msh_device_id="aawm-litellm-gateway")
        path = _write_contract(tmp_path, payload)

        contract = resolve_contract(str(path), required=True)
        assert contract is not None
        assert contract.source == KIMI_NATIVE_CONTRACT_SOURCE_BUILTIN

    @pytest.mark.parametrize(
        "bad_id",
        [
            "0D3F8A2E-7B14-4C6A-9E5F-A1B2C3D4E5F6",  # uppercase
            "{0d3f8a2e-7b14-4c6a-9e5f-a1b2c3d4e5f6}",  # braced
            "0d3f8a2e7b144c6a9e5fa1b2c3d4e5f6",  # no hyphens
            "urn:uuid:0d3f8a2e-7b14-4c6a-9e5f-a1b2c3d4e5f6",  # urn prefix
        ],
    )
    def test_noncanonical_uuid_device_id_falls_back_to_builtin(
        self, tmp_path: Path, bad_id: str
    ):
        payload = _make_payload(x_msh_device_id=bad_id)
        path = _write_contract(tmp_path, payload)

        contract = resolve_contract(str(path), required=True)
        assert contract is not None
        assert contract.source == KIMI_NATIVE_CONTRACT_SOURCE_BUILTIN

    def test_canonical_lowercase_uuid_device_id_accepted(self, tmp_path: Path):
        payload = _make_payload(
            x_msh_device_id="0d3f8a2e-7b14-4c6a-9e5f-a1b2c3d4e5f6"
        )
        path = _write_contract(tmp_path, payload)
        contract = resolve_contract(str(path), required=True)

        assert contract.x_msh_device_id == "0d3f8a2e-7b14-4c6a-9e5f-a1b2c3d4e5f6"
