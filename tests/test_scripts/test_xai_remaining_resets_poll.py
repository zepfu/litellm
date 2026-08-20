"""Tests for the XAI GetRemainingResets OIDC poll (XAI-005/006/007)."""

from __future__ import annotations

import gzip
import hashlib
import importlib.util
import os
import shutil
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "scripts"
LOOP_PATH = SCRIPTS_DIR / "run_provider_status_observations_loop.py"

# Prefer this worktree over any installed LiteLLM package so the sidecar
# stdlib cursor helpers resolve during import.
sys.path.insert(0, str(REPO_ROOT))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


def _load_loop_module():
    spec = importlib.util.spec_from_file_location(
        "run_provider_status_observations_loop",
        LOOP_PATH,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


loop = _load_loop_module()


PROVIDER = "xai"
CREDIT_FAMILY = "xai_usage_limit_reset"
CREDIT_TYPE = "usage_reset_token"
SOURCE = "xai_grok_web_remaining_resets"
PARSER_VERSION = "xai_grok_web_remaining_resets_v1"

# Ceiling RESETS_ENDPOINT
DEFAULT_RESETS_URL = "https://grok.com/prod_mc_billing.ConsumerUiSvc/GetRemainingResets"


def _encode_varint(value: int) -> bytes:
    if value < 0:
        raise ValueError("varint must be non-negative")
    out = bytearray()
    while value > 0x7F:
        out.append((value & 0x7F) | 0x80)
        value >>= 7
    out.append(value)
    return bytes(out)


def _encode_key(field_number: int, wire_type: int) -> bytes:
    return _encode_varint((field_number << 3) | wire_type)


def _encode_len_delimited(field_number: int, payload: bytes) -> bytes:
    return _encode_key(field_number, 2) + _encode_varint(len(payload)) + payload


def _encode_varint_field(field_number: int, value: int) -> bytes:
    return _encode_key(field_number, 0) + _encode_varint(value)


def _encode_timestamp(seconds: int) -> bytes:
    return _encode_varint_field(1, seconds)


def _encode_reset_token(
    token_id: str,
    *,
    validity_end: int,
    validity_start: int | None = None,
) -> bytes:
    body = _encode_len_delimited(10, token_id.encode("utf-8"))
    if validity_start is not None:
        body += _encode_len_delimited(20, _encode_timestamp(validity_start))
    body += _encode_len_delimited(30, _encode_timestamp(validity_end))
    return body


def _grpc_web_frame(payload: bytes, *, flags: int = 0) -> bytes:
    return bytes([flags]) + len(payload).to_bytes(4, "big") + payload


def _empty_valid_frame() -> bytes:
    return bytes([0, 0, 0, 0, 0])


def _response_with_one_unexpired_token(
    token_id: str = "reset-token-abc-123",
    *,
    validity_end: int | None = None,
    validity_start: int | None = None,
) -> bytes:
    now = int(time.time())
    end = validity_end if validity_end is not None else now + 7 * 24 * 3600
    start = validity_start if validity_start is not None else now - 3600
    token = _encode_reset_token(token_id, validity_end=end, validity_start=start)
    message = _encode_len_delimited(10, token)
    return _grpc_web_frame(message)


def _hashed_credit_identity(token_id: str) -> str:
    return loop.derive_xai_reset_credit_identity(token_id)


class ParseEmptyValidFrameTests(unittest.TestCase):
    def test_empty_valid_frame_is_known_zero(self) -> None:
        parsed = loop.parse_xai_remaining_resets_grpc_web(_empty_valid_frame())
        self.assertEqual(parsed.status, "ok")
        self.assertTrue(parsed.known_zero)
        self.assertFalse(parsed.tokens_field_present)
        self.assertEqual(parsed.unexpired_count, 0)
        self.assertEqual(list(parsed.tokens), [])
        self.assertFalse(parsed.unknown)
        self.assertIsNone(parsed.auth_error)


class ParseOneUnexpiredTokenTests(unittest.TestCase):
    def test_one_unexpired_token_count_and_hashed_identity(self) -> None:
        raw_token_id = "reset-token-abc-123"
        body = _response_with_one_unexpired_token(raw_token_id)
        parsed = loop.parse_xai_remaining_resets_grpc_web(body)
        self.assertEqual(parsed.status, "ok")
        self.assertTrue(parsed.tokens_field_present)
        self.assertEqual(parsed.unexpired_count, 1)
        self.assertEqual(len(parsed.tokens), 1)
        self.assertEqual(parsed.tokens[0].token_id, raw_token_id)
        self.assertGreater(parsed.tokens[0].validity_end, int(time.time()))

        observations = loop.build_xai_reset_token_observations(parsed)
        self.assertEqual(len(observations), 1)
        observation = observations[0]
        identity = observation["credit_identity"]
        self.assertNotEqual(identity, raw_token_id)
        self.assertNotIn(raw_token_id, identity)
        self.assertEqual(identity, _hashed_credit_identity(raw_token_id))
        self.assertEqual(observation["provider"], PROVIDER)
        self.assertEqual(observation["credit_family"], CREDIT_FAMILY)
        self.assertEqual(observation["credit_type"], CREDIT_TYPE)
        self.assertEqual(observation["source"], SOURCE)
        self.assertEqual(observation["parser_version"], PARSER_VERSION)

        again = loop.build_xai_reset_token_observations(
            loop.parse_xai_remaining_resets_grpc_web(body)
        )
        self.assertEqual(again[0]["credit_identity"], identity)


class MissingTokensFieldTests(unittest.TestCase):
    def test_missing_tokens_field_with_nonempty_payload_is_unknown(self) -> None:
        # Field 1 string "hello" — nonempty proto without field 10 tokens.
        payload = _encode_len_delimited(1, b"hello")
        body = _grpc_web_frame(payload)
        parsed = loop.parse_xai_remaining_resets_grpc_web(body)
        self.assertTrue(parsed.unknown)
        self.assertFalse(parsed.known_zero)
        self.assertFalse(parsed.tokens_field_present)
        self.assertNotEqual(parsed.status, "ok")
        self.assertFalse(loop.should_persist_xai_reset_token_observations(parsed))


class GrpcStatusAuthTests(unittest.TestCase):
    def test_grpc_status_16_is_reauthentication_required(self) -> None:
        trailer_payload = b"grpc-status:16\r\ngrpc-message:unauthenticated\r\n"
        body = _grpc_web_frame(b"", flags=0) + _grpc_web_frame(
            trailer_payload, flags=0x80
        )
        parsed = loop.parse_xai_remaining_resets_grpc_web(body)
        self.assertEqual(parsed.grpc_status, 16)
        self.assertTrue(parsed.auth_error)
        self.assertEqual(parsed.auth_error, "reauthentication_required")
        self.assertTrue(parsed.last_good_state_retained)
        self.assertFalse(loop.should_persist_xai_reset_token_observations(parsed))


class HttpAuthErrorTests(unittest.TestCase):
    def test_http_401_is_reauthentication_required(self) -> None:
        classified = loop.classify_xai_remaining_resets_http_error(401, b"")
        self.assertTrue(classified.auth_error)
        self.assertEqual(classified.auth_error, "reauthentication_required")
        self.assertTrue(classified.last_good_state_retained)
        self.assertFalse(loop.should_persist_xai_reset_token_observations(classified))

    def test_http_403_is_reauthentication_required(self) -> None:
        classified = loop.classify_xai_remaining_resets_http_error(403, b"")
        self.assertTrue(classified.auth_error)
        self.assertEqual(classified.auth_error, "reauthentication_required")
        self.assertTrue(classified.last_good_state_retained)
        self.assertFalse(loop.should_persist_xai_reset_token_observations(classified))


class TruncatedAndUnframedBodyTests(unittest.TestCase):
    def test_truncated_proto_is_unknown(self) -> None:
        # Frame claims 20 bytes but only 3 follow.
        body = bytes([0, 0, 0, 0, 20]) + b"abc"
        parsed = loop.parse_xai_remaining_resets_grpc_web(body)
        self.assertTrue(parsed.unknown)
        self.assertFalse(parsed.known_zero)
        self.assertFalse(loop.should_persist_xai_reset_token_observations(parsed))

    def test_unframed_body_is_unknown(self) -> None:
        parsed = loop.parse_xai_remaining_resets_grpc_web(b"not-a-grpc-web-frame")
        self.assertTrue(parsed.unknown)
        self.assertFalse(parsed.known_zero)
        self.assertFalse(loop.should_persist_xai_reset_token_observations(parsed))

    def test_truncated_inner_proto_is_unknown(self) -> None:
        # Valid frame wrapping a truncated length-delimited field.
        truncated_message = _encode_key(10, 2) + _encode_varint(50) + b"short"
        body = _grpc_web_frame(truncated_message)
        parsed = loop.parse_xai_remaining_resets_grpc_web(body)
        self.assertTrue(parsed.unknown)
        self.assertFalse(loop.should_persist_xai_reset_token_observations(parsed))


class SourceMustNotContainRedeemResetTests(unittest.TestCase):
    def test_loop_source_does_not_contain_redeem_reset(self) -> None:
        source = LOOP_PATH.read_text(encoding="utf-8")
        self.assertNotIn("RedeemReset", source)


def _build_dockerfile_shaped_tree(root: Path) -> Path:
    image_root = root / "app"
    scripts_dir = image_root / "scripts"
    secret_dir = image_root / "litellm" / "secret_managers"
    cursor_dir = image_root / "litellm" / "llms" / "cursor_agent"
    scripts_dir.mkdir(parents=True)
    secret_dir.mkdir(parents=True)
    cursor_dir.mkdir(parents=True)

    for name in (
        "record_provider_status_observations.py",
        "grok_oidc_refresh.py",
        "codex_oauth_refresh.py",
        "xai_oauth_refresh.py",
        "kimi_oauth_refresh.py",
        "run_provider_status_observations_loop.py",
    ):
        shutil.copy2(REPO_ROOT / "scripts" / name, scripts_dir / name)
    for name in (
        "credential_error_sanitizer.py",
        "credential_file_lock.py",
        "credential_file_metadata.py",
        "credential_file_write.py",
        "grok_oidc_auth_path.py",
        "codex_oauth_inventory.py",
        "kimi_native_contract.py",
        "grok_native_version_contract.py",
    ):
        shutil.copy2(
            REPO_ROOT / "litellm" / "secret_managers" / name,
            secret_dir / name,
        )
    for name in ("constants.py", "dashboard.py", "usage.py"):
        shutil.copy2(
            REPO_ROOT / "litellm" / "llms" / "cursor_agent" / name,
            cursor_dir / name,
        )
    for init_path in (
        scripts_dir / "__init__.py",
        image_root / "litellm" / "__init__.py",
        secret_dir / "__init__.py",
        image_root / "litellm" / "llms" / "__init__.py",
        cursor_dir / "__init__.py",
    ):
        init_path.write_text("", encoding="utf-8")
    return image_root


class PackagingImportTests(unittest.TestCase):
    def test_dockerfile_shaped_tree_imports_with_reset_poll_enabled_and_missing_auth(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            image_root = _build_dockerfile_shaped_tree(tmp_path)
            dest = image_root / "scripts" / "run_provider_status_observations_loop.py"
            missing_auth = tmp_path / "missing-grok-oidc-auth.json"

            env = os.environ.copy()
            env["AAWM_XAI_RESET_POLL_ENABLED"] = "1"
            env["AAWM_GROK_OIDC_AUTH_FILE"] = str(missing_auth)
            env["PYTHONPATH"] = str(image_root)
            env.pop("AAWM_GROK_WEB_AUTH_FILE", None)
            env.pop("AAWM_GROK_WEB_AUTH_COOKIE", None)

            spec = importlib.util.spec_from_file_location(
                "packaged_xai_reset_loop", dest
            )
            assert spec is not None
            assert spec.loader is not None
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            with patch.dict(os.environ, env, clear=False):
                sys_path_prefix = list(sys.path)
                sys.path.insert(0, str(image_root))
                sys.path.insert(0, str(image_root / "scripts"))
                try:
                    spec.loader.exec_module(module)
                except ModuleNotFoundError as exc:  # pragma: no cover
                    self.fail(
                        f"Dockerfile-shaped import raised ModuleNotFoundError: {exc}"
                    )
                finally:
                    sys.path[:] = sys_path_prefix

                self.assertFalse(hasattr(module, "httpx"))
                self.assertNotIn("httpx", getattr(module, "__dict__", {}))
                imported_names = {
                    getattr(value, "__name__", "")
                    for value in vars(module).values()
                }
                self.assertNotIn("httpx", imported_names)

                config = module.ProviderStatusLoopConfig.from_env()
                self.assertTrue(config.xai_reset_poll_enabled)
                auth_path = Path(config.grok_oidc_auth_file)
                self.assertFalse(auth_path.exists())
                classified = module.classify_xai_remaining_resets_missing_auth(
                    config.grok_oidc_auth_file
                )
                self.assertTrue(classified.auth_error)
                self.assertEqual(classified.auth_error, "reauthentication_required")
                self.assertTrue(classified.last_good_state_retained)
                self.assertFalse(
                    module.should_persist_xai_reset_token_observations(classified)
                )


class LifecycleTests(unittest.TestCase):
    def test_visible_then_missing_before_expiry_is_used(self) -> None:
        now = int(time.time())
        token_id = "lifecycle-token-before-expiry"
        previous = [
            {
                "credit_identity": _hashed_credit_identity(token_id),
                "provider": PROVIDER,
                "credit_family": CREDIT_FAMILY,
                "credit_type": CREDIT_TYPE,
                "raw_credit_id": token_id,
                "expires_at": now + 3600,
                "status": "available",
            }
        ]
        current_tokens: list = []
        synthesized = loop.synthesize_xai_reset_token_lifecycle_observations(
            previous_observations=previous,
            current_tokens=current_tokens,
            now_ts=now,
        )
        statuses = {row["credit_identity"]: row["status"] for row in synthesized}
        self.assertEqual(statuses[_hashed_credit_identity(token_id)], "used")

    def test_missing_after_expiry_is_expired(self) -> None:
        now = int(time.time())
        token_id = "lifecycle-token-after-expiry"
        previous = [
            {
                "credit_identity": _hashed_credit_identity(token_id),
                "provider": PROVIDER,
                "credit_family": CREDIT_FAMILY,
                "credit_type": CREDIT_TYPE,
                "raw_credit_id": token_id,
                "expires_at": now - 60,
                "status": "available",
            }
        ]
        synthesized = loop.synthesize_xai_reset_token_lifecycle_observations(
            previous_observations=previous,
            current_tokens=[],
            now_ts=now,
        )
        statuses = {row["credit_identity"]: row["status"] for row in synthesized}
        self.assertEqual(statuses[_hashed_credit_identity(token_id)], "expired")


class FailedPollDoesNotSynthesizeEmptyInventoryTests(unittest.TestCase):
    def test_unknown_poll_does_not_synthesize_empty_inventory(self) -> None:
        parsed = loop.parse_xai_remaining_resets_grpc_web(b"not-a-frame")
        self.assertTrue(parsed.unknown)
        self.assertFalse(loop.should_persist_xai_reset_token_observations(parsed))
        observations = loop.build_xai_reset_token_observations(parsed)
        self.assertEqual(observations, [])

    def test_auth_failure_does_not_synthesize_empty_inventory(self) -> None:
        classified = loop.classify_xai_remaining_resets_http_error(401, b"")
        self.assertFalse(loop.should_persist_xai_reset_token_observations(classified))
        observations = loop.build_xai_reset_token_observations(classified)
        self.assertEqual(observations, [])


class HashedIdentityFlagTests(unittest.TestCase):
    def test_hashed_identity_uses_hash_provider_credit_id_true(self) -> None:
        raw_token_id = "hash-flag-token"
        parsed = loop.parse_xai_remaining_resets_grpc_web(
            _response_with_one_unexpired_token(raw_token_id)
        )
        with patch.object(
            loop.probes,
            "derive_provider_credit_identity",
            wraps=loop.probes.derive_provider_credit_identity,
        ) as mocked:
            observations = loop.build_xai_reset_token_observations(parsed)
        self.assertEqual(len(observations), 1)
        self.assertTrue(mocked.called)
        for call in mocked.call_args_list:
            kwargs = call.kwargs
            if "hash_provider_credit_id" in kwargs:
                self.assertTrue(kwargs["hash_provider_credit_id"])
            else:
                args = call.args
                # Positional fallback: last bool-like argument should be True.
                self.assertIn(True, args)


class WeeklyBillingIndependenceTests(unittest.TestCase):
    def test_weekly_billing_poll_still_independently_scheduled(self) -> None:
        source = LOOP_PATH.read_text(encoding="utf-8")
        self.assertIn("poll_grok_billing", source)
        self.assertIn("poll_xai_remaining_resets", source)
        self.assertIn("cli-chat-proxy.grok.com/v1/billing", source)

        config = SimpleNamespace(
            grok_billing_poll_enabled=True,
            grok_billing_poll_interval_seconds=3600,
            xai_reset_poll_enabled=True,
            xai_reset_poll_interval_seconds=3600,
        )
        tasks = loop.iter_scheduled_provider_poll_tasks(config)
        names = [task[0] if isinstance(task, tuple) else getattr(task, "name", "") for task in tasks]
        joined = " ".join(str(name) for name in names)
        self.assertTrue(
            any("grok" in str(name).lower() and "bill" in str(name).lower() for name in names)
            or "billing" in joined.lower(),
            f"weekly billing task missing from schedule: {names}",
        )
        self.assertTrue(
            any("reset" in str(name).lower() for name in names),
            f"xai remaining-resets task missing from schedule: {names}",
        )

    def test_resets_failure_does_not_disable_billing(self) -> None:
        billing_ran = {"count": 0}

        def fake_billing(*_args, **_kwargs):
            billing_ran["count"] += 1
            return {"ok": True}

        def fake_resets(*_args, **_kwargs):
            raise RuntimeError("resets poll exploded")

        result = loop.run_independent_grok_polls(
            poll_grok_billing=fake_billing,
            poll_xai_remaining_resets=fake_resets,
        )
        self.assertEqual(billing_ran["count"], 1)
        self.assertTrue(result["billing_ok"])
        self.assertFalse(result["resets_ok"])


class GzipFrameAndExpiredTokenTests(unittest.TestCase):
    def test_gzip_flag_is_decoded(self) -> None:
        raw_token_id = "gzip-token"
        inner = _encode_len_delimited(
            10,
            _encode_reset_token(
                raw_token_id,
                validity_end=int(time.time()) + 3600,
                validity_start=int(time.time()) - 10,
            ),
        )
        body = _grpc_web_frame(gzip.compress(inner), flags=0x01)
        parsed = loop.parse_xai_remaining_resets_grpc_web(body)
        self.assertEqual(parsed.status, "ok")
        self.assertEqual(parsed.unexpired_count, 1)

    def test_expired_token_is_not_counted_or_persisted(self) -> None:
        raw_token_id = "already-expired"
        now = int(time.time())
        token = _encode_reset_token(
            raw_token_id,
            validity_end=now - 10,
            validity_start=now - 1000,
        )
        body = _grpc_web_frame(_encode_len_delimited(10, token))
        parsed = loop.parse_xai_remaining_resets_grpc_web(body)
        self.assertTrue(parsed.tokens_field_present)
        self.assertEqual(parsed.unexpired_count, 0)
        observations = loop.build_xai_reset_token_observations(parsed)
        self.assertEqual(observations, [])

    def test_empty_token_id_is_ignored(self) -> None:
        now = int(time.time())
        token = _encode_reset_token(
            "",
            validity_end=now + 3600,
            validity_start=now - 10,
        )
        body = _grpc_web_frame(_encode_len_delimited(10, token))
        parsed = loop.parse_xai_remaining_resets_grpc_web(body)
        self.assertEqual(parsed.unexpired_count, 0)
        observations = loop.build_xai_reset_token_observations(parsed)
        self.assertEqual(observations, [])


class ConfigDefaultsTests(unittest.TestCase):
    def test_argparse_defaults_are_off(self) -> None:
        keys = (
            "AAWM_XAI_RESET_POLL_ENABLED",
            "AAWM_XAI_RESET_POLL_INTERVAL_SECONDS",
            "AAWM_XAI_RESET_POLL_HTTP_TIMEOUT_SECONDS",
            "AAWM_XAI_RESET_POLL_URL",
        )
        saved = {key: os.environ.get(key) for key in keys}
        try:
            for key in keys:
                os.environ.pop(key, None)
            config = loop.ProviderStatusLoopConfig.from_env()
            self.assertFalse(config.xai_reset_poll_enabled)
            parser = loop.build_arg_parser()
            args = parser.parse_args([])
            self.assertFalse(bool(int(getattr(args, "xai_reset_poll_enabled", 0))))
            self.assertEqual(int(args.xai_reset_poll_interval_seconds), 3600)
            self.assertEqual(int(args.xai_reset_poll_http_timeout_seconds), 30)
            self.assertEqual(args.xai_reset_poll_url, DEFAULT_RESETS_URL)
        finally:
            for key, value in saved.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

    def test_known_zero_is_persistable(self) -> None:
        parsed = loop.parse_xai_remaining_resets_grpc_web(_empty_valid_frame())
        self.assertTrue(parsed.known_zero)
        self.assertTrue(loop.should_persist_xai_reset_token_observations(parsed))


if __name__ == "__main__":
    unittest.main()
