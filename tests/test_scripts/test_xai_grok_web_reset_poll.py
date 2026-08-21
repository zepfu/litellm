"""XAI-005/006/007: Grok GetRemainingResets poller, parser, and credit lifecycle."""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import inspect
import os
import re
import sys
import time
import urllib.error
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
LOOP_PATH = REPO_ROOT / "scripts" / "run_provider_status_observations_loop.py"
COMPOSE_PATH = REPO_ROOT / "docker-compose.dev.yml"
DOCS_PATH = REPO_ROOT / "docs" / "aawm-provider-status-observations.md"

DEFAULT_RESETS_URL = "https://grok.com/prod_mc_billing.ConsumerUiSvc/GetRemainingResets"
EMPTY_GRPC_WEB_FRAME = b"\x00\x00\x00\x00\x00"


def _varint(value: int) -> bytes:
    if value < 0:
        raise ValueError("varint must be non-negative")
    parts: List[int] = []
    remaining = value
    while True:
        bits = remaining & 0x7F
        remaining >>= 7
        if remaining:
            parts.append(bits | 0x80)
        else:
            parts.append(bits)
            break
    return bytes(parts)


def _key(field: int, wire: int) -> bytes:
    return _varint((field << 3) | wire)


def _ld(field: int, payload: bytes) -> bytes:
    return _key(field, 2) + _varint(len(payload)) + payload


def _timestamp(seconds: int) -> bytes:
    return _key(1, 0) + _varint(seconds)


def encode_reset_token(
    token_id: str,
    validity_start: Optional[int] = None,
    validity_end: Optional[int] = None,
) -> bytes:
    parts = [_ld(10, token_id.encode("utf-8"))]
    if validity_start is not None:
        parts.append(_ld(20, _timestamp(validity_start)))
    if validity_end is not None:
        parts.append(_ld(30, _timestamp(validity_end)))
    return b"".join(parts)


def encode_remaining_resets_message(tokens: List[bytes]) -> bytes:
    return b"".join(_ld(10, token) for token in tokens)


def grpc_web_data_frame(payload: bytes) -> bytes:
    return b"\x00" + len(payload).to_bytes(4, "big") + payload


def grpc_web_trailer_frame(status: int, message: str = "") -> bytes:
    payload = f"grpc-status: {status}\r\ngrpc-message: {message}\r\n".encode("utf-8")
    return b"\x80" + len(payload).to_bytes(4, "big") + payload


def grpc_web_body(
    payload: bytes,
    status: int = 0,
    message: str = "",
) -> bytes:
    return grpc_web_data_frame(payload) + grpc_web_trailer_frame(status, message)


def _load_loop_module():
    name = "run_provider_status_observations_loop"
    existing = sys.modules.get(name)
    if existing is not None and getattr(existing, "__file__", None) == str(LOOP_PATH):
        return existing
    spec = importlib.util.spec_from_file_location(name, LOOP_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def loop_mod():
    scripts_dir = str(REPO_ROOT / "scripts")
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    return _load_loop_module()


def _parse_dockerfile_copy_sources(dockerfile: Path) -> List[Path]:
    sources: List[Path] = []
    for raw_line in dockerfile.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line.startswith("COPY ") or line.startswith("COPY --from"):
            continue
        parts = line.split()
        args = [part for part in parts[1:] if not part.startswith("--")]
        if len(args) < 2:
            continue
        for src in args[:-1]:
            if src in {"."}:
                continue
            path = (REPO_ROOT / src).resolve()
            if path.exists():
                sources.append(path)
    return sources


def _dockerfile_shaped_tree(tmp_path: Path) -> Path:
    dockerfiles = [
        path
        for path in [REPO_ROOT / "Dockerfile", *REPO_ROOT.glob("Dockerfile*"), *REPO_ROOT.glob("docker/Dockerfile*")]
        if path.is_file()
        and "run_provider_status_observations_loop.py" in path.read_text(encoding="utf-8", errors="ignore")
    ]
    copied = False
    for dockerfile in dockerfiles:
        for source in _parse_dockerfile_copy_sources(dockerfile):
            rel = source.relative_to(REPO_ROOT)
            dest = tmp_path / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            if source.is_dir():
                for child in source.rglob("*"):
                    if child.is_file():
                        child_dest = dest / child.relative_to(source)
                        child_dest.parent.mkdir(parents=True, exist_ok=True)
                        child_dest.write_bytes(child.read_bytes())
            else:
                dest.write_bytes(source.read_bytes())
            copied = True
    if not copied:
        dest = tmp_path / "scripts" / "run_provider_status_observations_loop.py"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(LOOP_PATH.read_bytes())
    return tmp_path


def test_loop_source_has_no_redeem_reset_or_grok_web_auth_file():
    source = LOOP_PATH.read_text(encoding="utf-8")
    assert "RedeemReset" not in source
    assert "AAWM_GROK_WEB_AUTH" not in source
    assert "AAWM_GROK_WEB_COOKIE" not in source
    assert "AAWM_GROK_WEB_AUTH_FILE" not in source


def test_default_enabled_is_false(loop_mod, monkeypatch):
    monkeypatch.delenv("AAWM_XAI_RESET_POLL_ENABLED", raising=False)
    assert loop_mod.xai_reset_poll_enabled() is False
    monkeypatch.setenv("AAWM_XAI_RESET_POLL_ENABLED", "0")
    assert loop_mod.xai_reset_poll_enabled() is False
    monkeypatch.setenv("AAWM_XAI_RESET_POLL_ENABLED", "false")
    assert loop_mod.xai_reset_poll_enabled() is False
    monkeypatch.setenv("AAWM_XAI_RESET_POLL_ENABLED", "1")
    assert loop_mod.xai_reset_poll_enabled() is True


def test_compose_defaults_enabled_off_and_reuses_oidc_auth_file():
    text = COMPOSE_PATH.read_text(encoding="utf-8")
    assert "AAWM_XAI_RESET_POLL_ENABLED" in text
    assert "${AAWM_XAI_RESET_POLL_ENABLED:-0}" in text
    assert "AAWM_XAI_RESET_POLL_INTERVAL_SECONDS" in text
    assert "AAWM_XAI_RESET_POLL_HTTP_TIMEOUT_SECONDS" in text
    assert "AAWM_GROK_OIDC_AUTH_FILE" in text
    assert "AAWM_GROK_WEB_AUTH" not in text
    assert "AAWM_GROK_WEB_COOKIE" not in text
    assert "AAWM_GROK_WEB_AUTH_FILE" not in text
    assert not re.search(r"grok[-_]?web[-_]?cookie", text, re.I)
    assert "AAWM_GROK_WEB_AUTH" not in text


def test_docs_cover_oidc_reuse_and_default_disabled():
    text = DOCS_PATH.read_text(encoding="utf-8")
    assert "GetRemainingResets" in text
    assert "AAWM_XAI_RESET_POLL_ENABLED" in text
    assert "AAWM_GROK_OIDC_AUTH_FILE" in text
    assert "RedeemReset" not in text or "does not call" in text.lower() or "never" in text.lower()


def test_split_grpc_web_data_and_trailer(loop_mod):
    payload = encode_remaining_resets_message(
        [encode_reset_token("tok-1", 10, 20)]
    )
    body = grpc_web_body(payload, status=0)
    frames = loop_mod.split_xai_grpc_web_frames(body)
    assert frames.data_payloads == [payload]
    assert frames.grpc_status == 0
    assert frames.truncated is False
    assert frames.leftover is False


def test_split_grpc_web_empty_frame_is_five_zero_bytes(loop_mod):
    body = EMPTY_GRPC_WEB_FRAME + grpc_web_trailer_frame(0)
    frames = loop_mod.split_xai_grpc_web_frames(body)
    assert frames.data_payloads == [b""]
    assert frames.grpc_status == 0


def test_split_grpc_web_truncated_and_leftover(loop_mod):
    truncated = b"\x00\x00\x00\x00\x0a\x01\x02"
    frames = loop_mod.split_xai_grpc_web_frames(truncated)
    assert frames.truncated is True

    leftover = grpc_web_body(b"", status=0) + b"\x00"
    frames = loop_mod.split_xai_grpc_web_frames(leftover)
    assert frames.leftover is True or frames.truncated is True


def test_proto_walk_token_fields(loop_mod):
    now = int(time.time())
    token = encode_reset_token("abc", now - 10, now + 3600)
    message = encode_remaining_resets_message([token])
    fields = loop_mod.walk_xai_protobuf_fields(message)
    assert any(field == 10 and wire == 2 for field, wire, _value in fields)
    inner = loop_mod.walk_xai_protobuf_fields(token)
    numbers = {field for field, _wire, _value in inner}
    assert {10, 20, 30}.issubset(numbers)


def test_proto_walk_rejects_truncated_and_leftover(loop_mod):
    with pytest.raises(ValueError):
        loop_mod.walk_xai_protobuf_fields(b"\x12\x05\x01")
    with pytest.raises(ValueError):
        loop_mod.walk_xai_protobuf_fields(b"\x08\x01\x82")


def test_known_zero_empty_data_frame_grpc_status_0(loop_mod):
    parsed = loop_mod.parse_xai_remaining_resets_grpc_web(
        grpc_web_body(b"", status=0)
    )
    assert parsed.outcome == "zero"
    assert parsed.grpc_status == 0
    assert parsed.tokens == ()
    assert parsed.last_good_state_retained is False


def test_known_zero_from_header_status_and_empty_body(loop_mod):
    parsed = loop_mod.parse_xai_remaining_resets_grpc_web(
        EMPTY_GRPC_WEB_FRAME,
        headers={"grpc-status": "0"},
    )
    assert parsed.outcome == "zero"


def test_missing_tokens_is_unknown_not_zero(loop_mod):
    other_field = _ld(1, b"hello")
    parsed = loop_mod.parse_xai_remaining_resets_grpc_web(grpc_web_body(other_field, status=0))
    assert parsed.outcome == "unknown"
    assert parsed.tokens == ()
    assert parsed.last_good_state_retained is True


def test_truncated_and_leftover_are_unknown(loop_mod):
    truncated = loop_mod.parse_xai_remaining_resets_grpc_web(b"\x00\x00\x00\x00\x04\x01")
    assert truncated.outcome == "unknown"
    leftover = loop_mod.parse_xai_remaining_resets_grpc_web(
        grpc_web_body(b"", status=0) + b"\xff"
    )
    assert leftover.outcome == "unknown"


def test_grpc_status_nonzero_unknown_except_16_auth(loop_mod):
    now = int(time.time()) + 120
    payload = encode_remaining_resets_message(
        [encode_reset_token("tok", now - 1, now)]
    )
    unknown = loop_mod.parse_xai_remaining_resets_grpc_web(
        grpc_web_body(payload, status=2)
    )
    assert unknown.outcome == "unknown"
    assert unknown.last_good_state_retained is True
    assert unknown.grpc_status == 2

    auth = loop_mod.parse_xai_remaining_resets_grpc_web(
        grpc_web_body(payload, status=16)
    )
    assert auth.outcome == "auth"
    assert auth.grpc_status == 16
    assert auth.last_good_state_retained is True
    assert auth.reason in {"reauthentication_required", "auth", "unauthenticated"}


def test_parse_valid_tokens_ok(loop_mod):
    now = int(time.time())
    payload = encode_remaining_resets_message(
        [
            encode_reset_token("keep-me", now - 5, now + 60),
            encode_reset_token("", now - 5, now + 60),
            encode_reset_token("expired", now - 50, now - 1),
        ]
    )
    parsed = loop_mod.parse_xai_remaining_resets_grpc_web(grpc_web_body(payload, status=0))
    assert parsed.outcome == "ok"
    ids = [token.token_id for token in parsed.tokens]
    assert "keep-me" in ids
    assert "expired" in ids


def test_request_is_bearer_only_empty_frame_and_ceiling_headers(loop_mod):
    url, headers, body = loop_mod.build_xai_reset_poll_request("oidc-token-value")
    assert url == DEFAULT_RESETS_URL
    assert body == EMPTY_GRPC_WEB_FRAME
    normalized = {key.lower(): value for key, value in headers.items()}
    assert normalized["origin"] == "https://grok.com"
    assert normalized["referer"] == "https://grok.com/?_s=usage"
    assert normalized["accept"] == "*/*"
    assert normalized["content-type"] == "application/grpc-web+proto"
    assert normalized["x-grpc-web"] == "1"
    assert normalized["x-user-agent"] == "connect-es/2.1.1"
    assert normalized["authorization"] == "Bearer oidc-token-value"
    assert normalized["user-agent"] == "aawm-provider-status-observations"
    assert "cookie" not in normalized
    assert "Ceiling" not in normalized["user-agent"]


def test_attempts_and_backoff_match_grok_billing_contract(loop_mod):
    assert loop_mod.XAI_RESET_POLL_ATTEMPTS == 3
    assert float(loop_mod.XAI_RESET_POLL_BACKOFF_SECONDS) == 0.5
    grok_attempts = getattr(loop_mod, "GROK_BILLING_POLL_ATTEMPTS", None) or getattr(
        loop_mod, "_GROK_BILLING_POLL_ATTEMPTS", None
    )
    grok_backoff = getattr(loop_mod, "GROK_BILLING_POLL_BACKOFF_SECONDS", None) or getattr(
        loop_mod, "_GROK_BILLING_POLL_BACKOFF_SECONDS", None
    )
    if grok_attempts is not None:
        assert loop_mod.XAI_RESET_POLL_ATTEMPTS == grok_attempts
    if grok_backoff is not None:
        assert float(loop_mod.XAI_RESET_POLL_BACKOFF_SECONDS) == float(grok_backoff)


def test_fetch_retries_then_raises(loop_mod, monkeypatch):
    calls = {"n": 0}

    def boom(*_args, **_kwargs):
        calls["n"] += 1
        raise urllib.error.URLError("nope")

    monkeypatch.setattr(loop_mod.urllib.request, "urlopen", boom)
    monkeypatch.setattr(loop_mod.time, "sleep", lambda _seconds: None)
    with pytest.raises(urllib.error.URLError):
        loop_mod.fetch_xai_remaining_resets(
            access_token="abc",
            url=DEFAULT_RESETS_URL,
            timeout=1,
        )
    assert calls["n"] == 3


def test_credit_identity_is_hashed_and_account_is_hashed(loop_mod):
    token_id = "tok_live_secret_reset"
    user_id = "user-raw-identity-123"
    identity = loop_mod.derive_xai_reset_credit_identity(token_id)
    assert identity != token_id
    assert token_id not in identity
    source = inspect.getsource(loop_mod.derive_xai_reset_credit_identity)
    assert "derive_provider_credit_identity" in source
    assert "hash_provider_credit_id=True" in source.replace(" ", "") or "hash_provider_credit_id=True" in source

    account_source = inspect.getsource(loop_mod._xai_account_identity_hash)
    assert "account_identity_hash" in account_source
    hashed = loop_mod._xai_account_identity_hash(user_id)
    assert hashed != user_id
    assert user_id not in str(hashed)


def test_persist_plan_constants_and_dedupe(loop_mod):
    now = int(time.time())
    parsed = loop_mod.parse_xai_remaining_resets_grpc_web(
        grpc_web_body(
            encode_remaining_resets_message(
                [
                    encode_reset_token("same", now - 1, now + 99),
                    encode_reset_token("same", now - 1, now + 99),
                ]
            ),
            status=0,
        )
    )
    writes = loop_mod.plan_xai_reset_credit_writes(
        parse_result=parsed,
        previous_rows=[],
        now=now,
        user_id="user-1",
        account_hasher=lambda user: "acct-" + hashlib.sha256(user.encode()).hexdigest()[:16],
    )
    available = [row for row in writes if row["status"] == "available"]
    assert len(available) == 1
    row = available[0]
    assert row["provider"] == "xai"
    assert row["credit_family"] == "xai_usage_limit_reset"
    assert row["credit_type"] == "usage_reset_token"
    assert row["source"] == "xai_grok_web_remaining_resets"
    assert row["parser_version"] == "xai_grok_web_remaining_resets_v1"
    assert row["available_count"] == 1
    assert row.get("redeem_status") is None
    assert row.get("redeem_at") is None
    assert "same" not in str(row["credit_identity"])
    assert "user-1" not in str(row["account_hash"])
    assert row["account_hash"] != "user-1"


def test_persist_skips_empty_id_and_expired_tokens(loop_mod):
    now = int(time.time())
    parsed = loop_mod.parse_xai_remaining_resets_grpc_web(
        grpc_web_body(
            encode_remaining_resets_message(
                [
                    encode_reset_token("", now - 1, now + 99),
                    encode_reset_token("old", now - 50, now - 1),
                    encode_reset_token("keep", now - 1, now + 99),
                ]
            ),
            status=0,
        )
    )
    writes = loop_mod.plan_xai_reset_credit_writes(
        parse_result=parsed,
        previous_rows=[],
        now=now,
        user_id="u",
        account_hasher=lambda user: "h-" + user[::-1],
    )
    available_ids = [row["credit_identity"] for row in writes if row["status"] == "available"]
    assert len(available_ids) == 1
    assert loop_mod.derive_xai_reset_credit_identity("keep") in available_ids


def test_missing_before_expiry_used_and_at_or_after_expiry_expired(loop_mod):
    now = int(time.time())
    keep_id = loop_mod.derive_xai_reset_credit_identity("keep")
    used_id = "prev-used"
    expired_id = "prev-expired"
    parsed = loop_mod.parse_xai_remaining_resets_grpc_web(
        grpc_web_body(
            encode_remaining_resets_message(
                [encode_reset_token("keep", now - 1, now + 500)]
            ),
            status=0,
        )
    )
    writes = loop_mod.plan_xai_reset_credit_writes(
        parse_result=parsed,
        previous_rows=[
            {"credit_identity": keep_id, "status": "available", "expires_at": now + 500},
            {"credit_identity": used_id, "status": "available", "expires_at": now + 40},
            {"credit_identity": expired_id, "status": "available", "expires_at": now - 1},
        ],
        now=now,
        user_id="u",
        account_hasher=lambda user: "h",
    )
    by_id = {row["credit_identity"]: row["status"] for row in writes}
    assert by_id[used_id] == "used"
    assert by_id[expired_id] == "expired"
    assert by_id[keep_id] == "available"


def test_known_zero_synthesizes_used_and_expired(loop_mod):
    now = int(time.time())
    parsed = loop_mod.parse_xai_remaining_resets_grpc_web(grpc_web_body(b"", status=0))
    writes = loop_mod.plan_xai_reset_credit_writes(
        parse_result=parsed,
        previous_rows=[
            {"credit_identity": "future", "status": "available", "expires_at": now + 10},
            {"credit_identity": "past", "status": "available", "expires_at": now},
            {"credit_identity": "already-used", "status": "used", "expires_at": now + 10},
        ],
        now=now,
        user_id="u",
        account_hasher=lambda user: "h",
    )
    by_id = {row["credit_identity"]: row["status"] for row in writes}
    assert by_id["future"] == "used"
    assert by_id["past"] == "expired"
    assert "already-used" not in by_id


def test_failed_unknown_and_auth_skip_synthesis(loop_mod):
    now = int(time.time())
    previous = [
        {"credit_identity": "keep-me", "status": "available", "expires_at": now + 99}
    ]
    for body, headers in (
        (grpc_web_body(b"", status=7), None),
        (grpc_web_body(b"", status=16), None),
        (b"\x00\x00\x00\x00\x03\x01", None),
        (encode_remaining_resets_message([encode_reset_token("x", 1, now + 9)]), {"grpc-status": "2"}),
    ):
        parsed = loop_mod.parse_xai_remaining_resets_grpc_web(body, headers=headers)
        assert parsed.outcome in {"unknown", "auth"}
        writes = loop_mod.plan_xai_reset_credit_writes(
            parse_result=parsed,
            previous_rows=previous,
            now=now,
            user_id="u",
            account_hasher=lambda user: "h",
        )
        assert writes == []


def test_grpc_status_16_does_not_call_xai_oauth(loop_mod, monkeypatch):
    oauth_calls: List[str] = []

    def mark_oauth(*_args, **_kwargs):
        oauth_calls.append("called")
        raise AssertionError("xAI OAuth must not run on grpc-status 16")

    for name in dir(loop_mod):
        lower = name.lower()
        if "oauth" in lower or "redeem" in lower:
            target = getattr(loop_mod, name)
            if callable(target):
                monkeypatch.setattr(loop_mod, name, mark_oauth)

    monkeypatch.setattr(loop_mod, "xai_reset_poll_enabled", lambda: True)
    monkeypatch.setattr(
        loop_mod,
        "_load_grok_billing_auth_context",
        lambda: {"access_token": "oidc", "user_id": "user-1"},
    )
    monkeypatch.setattr(
        loop_mod,
        "fetch_xai_remaining_resets",
        lambda **_kwargs: (200, {"grpc-status": "16"}, grpc_web_body(b"", status=16)),
    )
    commits: List[Any] = []
    monkeypatch.setattr(loop_mod, "_commit_xai_reset_credit_writes", lambda writes: commits.append(list(writes)))
    loop_mod._run_xai_reset_poll_once()
    assert commits == []
    assert oauth_calls == []
    source = inspect.getsource(loop_mod._run_xai_reset_poll_once)
    assert "RedeemReset" not in source
    assert "oauth" not in source.lower()


def test_reset_task_failure_does_not_raise(loop_mod, monkeypatch):
    monkeypatch.setattr(loop_mod, "xai_reset_poll_enabled", lambda: True)
    monkeypatch.setattr(
        loop_mod,
        "_run_xai_reset_poll_once",
        lambda: (_ for _ in ()).throw(RuntimeError("reset failed")),
    )

    async def _run():
        await loop_mod._run_xai_reset_poll_task()

    import asyncio

    asyncio.run(_run())
    grok = inspect.getsource(loop_mod._run_grok_billing_poll_task)
    assert "xai_reset_poll" not in grok
    assert "RedeemReset" not in grok


def test_scheduler_registers_xai_reset_poll_next_to_grok_billing():
    source = LOOP_PATH.read_text(encoding="utf-8")
    assert '(_run_xai_reset_poll_task, "xai_reset_poll")' in source
    grok_idx = source.find('(_run_grok_billing_poll_task, "grok_billing_poll")')
    xai_idx = source.find('(_run_xai_reset_poll_task, "xai_reset_poll")')
    assert grok_idx != -1
    assert xai_idx != -1


def test_telemetry_event_name_and_last_good_state_flag(loop_mod):
    source = inspect.getsource(loop_mod._emit_xai_reset_poll_telemetry)
    assert "xai_reset_poll" in source
    assert "last_good_state_retained" in source
    payload_source = LOOP_PATH.read_text(encoding="utf-8")
    emit_fn = ast.parse(inspect.getsource(loop_mod._emit_xai_reset_poll_telemetry))
    dump = ast.dump(emit_fn)
    assert "Authorization" not in dump
    assert "access_token" not in payload_source[payload_source.find("def _emit_xai_reset_poll_telemetry") : payload_source.find("def _emit_xai_reset_poll_telemetry") + 1500]


def test_auth_loader_reuses_grok_oidc_not_new_cookie_file(loop_mod):
    source = inspect.getsource(loop_mod._run_xai_reset_poll_once)
    assert "_load_grok_billing_auth_context" in source
    assert "AAWM_GROK_WEB" not in source
    loop_source = LOOP_PATH.read_text(encoding="utf-8")
    assert "AAWM_GROK_OIDC_AUTH_FILE" in loop_source


def test_packaging_import_enabled_without_auth_file(tmp_path, monkeypatch):
    monkeypatch.setenv("AAWM_XAI_RESET_POLL_ENABLED", "1")
    monkeypatch.delenv("AAWM_GROK_OIDC_AUTH_FILE", raising=False)
    tree = _dockerfile_shaped_tree(tmp_path)
    loop_copy = next(tree.rglob("run_provider_status_observations_loop.py"))
    scripts_dir = str(loop_copy.parent)
    monkeypatch.syspath_prepend(scripts_dir)
    spec = importlib.util.spec_from_file_location(
        "sidecar_xai_reset_poll_packaging_loop",
        loop_copy,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    assert module.xai_reset_poll_enabled() is True
    assert "RedeemReset" not in loop_copy.read_text(encoding="utf-8")


def test_packaging_sidecar_test_still_covers_loop_module():
    packaging = REPO_ROOT / "tests" / "test_scripts" / "test_cursor_agent_sidecar_packaging.py"
    text = packaging.read_text(encoding="utf-8")
    assert "run_provider_status_observations_loop.py" in text
    assert "AAWM_XAI_RESET_POLL_ENABLED" in text


def _xai_reset_poll_config(loop_mod, **overrides):
    kwargs = {
        "apply": True,
        "dsn": "postgresql://fake-local/aawm_tristore_rr109",
        "environment": "dev",
        "interval_seconds": 300.0,
        "timeout": 2.0,
        "ping_count": 1,
        "ping_timeout": 2,
        "skip_icmp": False,
        "once": True,
        "setup_schema": False,
        "db_lock_timeout_ms": 1000,
        "db_statement_timeout_ms": 5000,
        "xai_reset_poll_enabled": True,
        "xai_reset_poll_interval_seconds": 3600.0,
        "xai_reset_poll_http_timeout_seconds": 1.0,
        "xai_reset_poll_url": DEFAULT_RESETS_URL,
        "xai_reset_poll_max_attempts": 5,
        "grok_oidc_auth_file": "/tmp/rr109-grok-oidc-auth.json",
    }
    kwargs.update(overrides)
    return loop_mod.ProviderStatusLoopConfig(**kwargs)


def test_previous_credit_state_load_failure_does_not_commit_complete_lifecycle(
    loop_mod,
    monkeypatch,
):
    now = int(time.time())
    commits: List[Any] = []

    monkeypatch.setattr(
        loop_mod,
        "_load_grok_billing_auth_context",
        lambda *_args, **_kwargs: {"access_token": "oidc", "user_id": "user-1"},
    )
    monkeypatch.setattr(
        loop_mod,
        "fetch_xai_remaining_resets",
        lambda **_kwargs: (
            200,
            {},
            grpc_web_body(
                encode_remaining_resets_message(
                    [encode_reset_token("keep", now - 1, now + 500)]
                ),
                status=0,
            ),
        ),
    )
    monkeypatch.setattr(
        loop_mod.probes,
        "load_provider_credit_current_rows",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("forced previous-state load failure")
        ),
    )
    monkeypatch.setattr(
        loop_mod,
        "_commit_xai_reset_credit_writes",
        lambda writes, config=None, **_kwargs: commits.append(list(writes)) or len(writes),
    )

    result = loop_mod._run_xai_reset_poll_once(_xai_reset_poll_config(loop_mod))

    assert commits == []
    assert result is not None
    assert result.get("persisted") is not True
    assert result.get("observation_count", 0) == 0
    assert result.get("last_good_state_retained") is True or result.get("outcome") != "ok"


def test_non_default_xai_reset_poll_max_attempts_is_parsed_from_cli_and_env(
    loop_mod,
    monkeypatch,
):
    monkeypatch.delenv("AAWM_XAI_RESET_POLL_MAX_ATTEMPTS", raising=False)
    help_text = loop_mod._build_parser().format_help()
    assert "--xai-reset-poll-max-attempts" in help_text
    assert "AAWM_XAI_RESET_POLL_MAX_ATTEMPTS" in help_text

    cli_config = loop_mod.parse_config(["--xai-reset-poll-max-attempts", "5"])
    assert cli_config.xai_reset_poll_max_attempts == 5

    monkeypatch.setenv("AAWM_XAI_RESET_POLL_MAX_ATTEMPTS", "7")
    env_config = loop_mod.parse_config([])
    assert env_config.xai_reset_poll_max_attempts == 7


def test_non_default_xai_reset_poll_max_attempts_observes_exactly_that_many_fetches(
    loop_mod,
    monkeypatch,
):
    calls = {"n": 0}

    def boom(*_args, **_kwargs):
        calls["n"] += 1
        raise urllib.error.URLError("nope")

    monkeypatch.setattr(loop_mod.urllib.request, "urlopen", boom)
    monkeypatch.setattr(loop_mod.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(
        loop_mod,
        "_load_grok_billing_auth_context",
        lambda *_args, **_kwargs: {"access_token": "oidc", "user_id": "user-1"},
    )

    config = _xai_reset_poll_config(loop_mod, apply=False, xai_reset_poll_max_attempts=5)
    with pytest.raises(urllib.error.URLError):
        loop_mod._run_xai_reset_poll_once(config)
    assert calls["n"] == 5
