"""Packaging contract for the stdlib conversation_init module in the sidecar image."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DOCKERFILE_PATH = (
    _REPO_ROOT / "docker" / "Dockerfile.provider_status_observations"
)
_LOOP_SCRIPT = "run_provider_status_observations_loop.py"

_SCRIPT_FILES = (
    "record_provider_status_observations.py",
    "grok_oidc_refresh.py",
    "codex_oauth_refresh.py",
    "xai_oauth_refresh.py",
    "kimi_oauth_refresh.py",
    "nous_oauth_refresh.py",
    "cursor_agent_auth_refresh.py",
    _LOOP_SCRIPT,
)

_SECRET_MANAGER_FILES = (
    "credential_error_sanitizer.py",
    "credential_file_lock.py",
    "credential_file_metadata.py",
    "credential_file_write.py",
    "grok_oidc_auth_path.py",
    "codex_oauth_inventory.py",
    "kimi_native_contract.py",
    "grok_native_version_contract.py",
    "xai_oauth_credentials.py",
)

_CURSOR_FILES = (
    "constants.py",
    "dashboard.py",
    "connect.py",
    "usage.py",
)

_CHATGPT_FILES = ("conversation_init.py",)
_USAGE_CAPTURE_FILES = (
    "__init__.py",
    "models.py",
    "privacy.py",
    "timeutil.py",
    "reconstruct.py",
    "pg_ledger.py",
    "native_history_ingest.py",
)


def _dockerfile() -> str:
    return _DOCKERFILE_PATH.read_text(encoding="utf-8")


def _build_sidecar_layout(tmp_path: Path) -> Path:
    image_root = tmp_path / "app"
    scripts_dir = image_root / "scripts"
    secret_dir = image_root / "litellm" / "secret_managers"
    cursor_dir = image_root / "litellm" / "llms" / "cursor_agent"
    chatgpt_dir = image_root / "litellm" / "llms" / "chatgpt"
    usage_dir = scripts_dir / "chatgpt_chat_usage_capture"
    scripts_dir.mkdir(parents=True)
    secret_dir.mkdir(parents=True)
    cursor_dir.mkdir(parents=True)
    chatgpt_dir.mkdir(parents=True)
    usage_dir.mkdir(parents=True)

    for name in _SCRIPT_FILES:
        shutil.copy2(_REPO_ROOT / "scripts" / name, scripts_dir / name)
    shutil.copy2(
        _REPO_ROOT / "scripts" / "apply_chatgpt_usage_ledger_2026_09_08.sql",
        scripts_dir / "apply_chatgpt_usage_ledger_2026_09_08.sql",
    )
    for name in _USAGE_CAPTURE_FILES:
        shutil.copy2(
            _REPO_ROOT / "scripts" / "chatgpt_chat_usage_capture" / name,
            usage_dir / name,
        )
    for name in _SECRET_MANAGER_FILES:
        shutil.copy2(
            _REPO_ROOT / "litellm" / "secret_managers" / name,
            secret_dir / name,
        )
    for name in _CURSOR_FILES:
        shutil.copy2(
            _REPO_ROOT / "litellm" / "llms" / "cursor_agent" / name,
            cursor_dir / name,
        )
    for name in _CHATGPT_FILES:
        shutil.copy2(
            _REPO_ROOT / "litellm" / "llms" / "chatgpt" / name,
            chatgpt_dir / name,
        )

    for init_path in (
        scripts_dir / "__init__.py",
        image_root / "litellm" / "__init__.py",
        secret_dir / "__init__.py",
        image_root / "litellm" / "llms" / "__init__.py",
        cursor_dir / "__init__.py",
        chatgpt_dir / "__init__.py",
        usage_dir / "__init__.py",
    ):
        init_path.write_text("", encoding="utf-8")
    return image_root


def test_dockerfile_copies_conversation_init() -> None:
    dockerfile = _dockerfile()

    assert (
        "COPY litellm/llms/chatgpt/conversation_init.py "
        "/app/litellm/llms/chatgpt/conversation_init.py"
    ) in dockerfile


def test_dockerfile_copies_history_usage_ledger() -> None:
    dockerfile = _dockerfile()

    assert (
        "COPY scripts/apply_chatgpt_usage_ledger_2026_09_08.sql "
        "/app/scripts/apply_chatgpt_usage_ledger_2026_09_08.sql"
    ) in dockerfile
    assert (
        "COPY scripts/chatgpt_chat_usage_capture/pg_ledger.py "
        "/app/scripts/chatgpt_chat_usage_capture/pg_ledger.py"
    ) in dockerfile
    assert (
        "COPY scripts/chatgpt_chat_usage_capture/native_history_ingest.py "
        "/app/scripts/chatgpt_chat_usage_capture/native_history_ingest.py"
    ) in dockerfile
    assert "authenticator.py" not in dockerfile


def test_dockerfile_touches_chatgpt_package_init() -> None:
    dockerfile = _dockerfile()

    assert "/app/litellm/llms/chatgpt/__init__.py" in dockerfile


def test_dockerfile_does_not_ship_chatgpt_authenticator() -> None:
    dockerfile = _dockerfile()

    assert "authenticator.py" not in dockerfile
    assert "common_utils.py" not in dockerfile
    assert "httpx" not in dockerfile


def test_loop_imports_conversation_init_from_stdlib() -> None:
    loop_source = (
        _REPO_ROOT / "scripts" / _LOOP_SCRIPT
    ).read_text(encoding="utf-8")

    assert "from litellm.llms.chatgpt.conversation_init import" in loop_source


def test_gold_lock_imports_loop_from_copied_sidecar_layout(tmp_path: Path) -> None:
    image_root = _build_sidecar_layout(tmp_path)
    loop_path = image_root / "scripts" / _LOOP_SCRIPT
    helper = """
import importlib.util
import sys
from pathlib import Path

loop_path = Path(sys.argv[1])
spec = importlib.util.spec_from_file_location(
    "run_provider_status_observations_loop",
    loop_path,
)
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
assert "litellm.llms.chatgpt.common_utils" not in sys.modules
assert "httpx" not in sys.modules
from litellm.llms.chatgpt.conversation_init import (
    CHATGPT_CONVERSATION_INIT_DEFAULT_URL,
    conversation_init_request_contract,
)
contract = conversation_init_request_contract()
assert contract["method"] == "POST"
assert contract["body_omitted"] is True
"""
    env = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}
    env["PYTHONPATH"] = str(image_root)
    result = subprocess.run(
        [sys.executable, "-c", helper, str(loop_path)],
        cwd=str(image_root),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr + result.stdout


def test_bound_account_history_cleanup_does_not_skip_conversation_init(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from litellm.llms.chatgpt.conversation_init import (
        OracleBrowserCleanupError,
        hash_chatgpt_conversation_init_canonical_account_id,
    )
    from litellm.secret_managers.codex_oauth_inventory import (
        CodexOAuthCredentialRecord,
        CodexOAuthCredentialSnapshot,
    )
    from scripts import run_provider_status_observations_loop as loop

    account_id = "acct-test-bound"
    account_hash = hash_chatgpt_conversation_init_canonical_account_id(account_id)
    assert account_hash is not None
    record = CodexOAuthCredentialRecord(
        label="account1",
        auth_path=tmp_path / "auth.json",
        lock_path=tmp_path / "auth.lock",
        priority=1,
        weight=1.0,
        enabled=True,
        models=("*",),
        expected_account_hash=account_hash,
        declaration_order=0,
    )
    binding = loop.ChatGPTConversationInitAccountBinding(
        cdp_endpoint="http://127.0.0.1:9222",
        page_target_id="page-target-1",
    )
    config = loop.ProviderStatusLoopConfig(
        apply=False,
        dsn=None,
        environment="test",
        interval_seconds=1.0,
        timeout=1.0,
        ping_count=1,
        ping_timeout=1,
        skip_icmp=True,
        once=True,
        setup_schema=False,
        db_lock_timeout_ms=1000,
        db_statement_timeout_ms=1000,
        chatgpt_conversation_init_source_path=str(tmp_path / "conversation-init.json"),
    )
    state = loop.SidecarTaskState()
    sessions: list[int] = []

    class _Transport:
        def fetch(self, request):
            del request
            return {
                "status_code": 200,
                "payload": {
                    "type": "conversation_init",
                    "default_model_slug": "gpt-6-pro",
                    "intended_default_model_slug": "gpt-6-pro",
                    "model_limits": [],
                    "limits_progress": [],
                    "blocked_features": [],
                    "atlas_mode_enabled": False,
                    "banner_info": {},
                    "user": {"id": "user-test-0001"},
                    "account_id": account_id,
                },
                "native_capture": {
                    "account_hash": account_hash,
                    "identity_source": "native_request_header",
                    "selector_evidence": "request_and_extra_info",
                    "request_response_correlated": True,
                    "request_method": "POST",
                    "request_body_omitted": True,
                    "browser_challenge": False,
                },
            }

    @contextmanager
    def fake_binding(*_args, **_kwargs):
        session = len(sessions)
        sessions.append(session)
        yield loop.ChatGPTConversationInitResolvedBinding(
            cdp_endpoint="http://127.0.0.1:9222",
            page_target_id="page-target-1",
            lifecycle_capability=object(),
        )
        if session == 0:
            raise OracleBrowserCleanupError(
                "Oracle browser owner process cleanup remains unproven."
            )

    monkeypatch.setattr(
        loop,
        "load_codex_oauth_credential",
        lambda rec: CodexOAuthCredentialSnapshot(
            record=rec,
            account_hash=account_hash,
            expires_at=None,
            access_token="token",
            account_id=account_id,
        ),
    )
    monkeypatch.setattr(loop, "_chatgpt_oracle_browser_binding", fake_binding)
    monkeypatch.setattr(
        loop,
        "observe_native_chatgpt_history_from_oracle_browser",
        lambda **_kwargs: (_ for _ in ()).throw(
            OracleBrowserCleanupError(
                "Oracle browser history observer is unavailable (Error at cdp_connect)."
            )
        ),
    )
    monkeypatch.setattr(
        "litellm.llms.chatgpt.conversation_init.build_oracle_browser_conversation_init_transport",
        lambda **_kwargs: _Transport(),
    )

    payloads, coverage = loop._collect_bound_chatgpt_conversation_init_account(
        config,
        record,
        binding,
        observed_at=datetime.now(timezone.utc),
        state=state,
    )
    assert sessions == [0, 1]
    assert coverage["history_usage"]["cleanup_unproven"] is True
    assert coverage["history_usage"]["capture_status"] == "capture_failed"
    assert "cdp_connect" in str(coverage["history_usage"]["failure_reason"])
    assert coverage["history_usage"]["cleanup_error_class"] == "OracleBrowserCleanupError"
    assert coverage["history_usage"]["error_class"] == "OracleBrowserCleanupError"
    assert coverage["capture_status"] == "parsed"
    assert coverage["fresh_capture"] is True
    assert coverage["account_identity_verified"] is True
    assert payloads
    serialized = str(coverage) + str(payloads)
    assert "cookie" not in serialized.lower()
    assert "authorization" not in serialized.lower()
    assert "Bearer" not in serialized
    assert account_id not in serialized
