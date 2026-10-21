"""OPENAI041 stock-client acceptance harness (quota a1 -> a2 seed case)."""

from __future__ import annotations

import asyncio
import base64
import json
import os
import socket
import subprocess
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
import uvicorn
from fastapi import HTTPException

from litellm.proxy.pass_through_endpoints.aawm_alias_routing import (
    codex_oauth,
    cooldown_state,
    state,
    durable,
)
from litellm.proxy.pass_through_endpoints import pass_through_endpoints as pte
from litellm.proxy.proxy_server import app as proxy_app
from litellm.secret_managers.codex_oauth_inventory import (
    codex_oauth_account_identity_hash,
)


MODEL = "gpt-5.6-sol"
STOCK_CODEX = "/usr/bin/codex"
MASTER_KEY = "sk-openai041-stock-test"
ACCOUNT1 = "openai041-acct-1"
ACCOUNT2 = "openai041-acct-2"


def _jwt(account_id: str) -> str:
    payload = {
        "exp": 9999999999,
        "https://api.openai.com/auth": {"chatgpt_account_id": account_id},
    }
    encoded = base64.urlsafe_b64encode(json.dumps(payload).encode()).decode()
    return f"header.{encoded.rstrip('=')}.signature"


def _write_credential(path: Path, account_id: str) -> str:
    access_token = _jwt(account_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "OPENAI_API_KEY": None,
                "tokens": {
                    "access_token": access_token,
                    "account_id": account_id,
                    "expires_at": 9999999999,
                },
            }
        ),
        encoding="utf-8",
    )
    path.chmod(0o600)
    return access_token


def _inventory(codex_home: Path) -> str:
    records = []
    for label, account_id in (
        ("account1", ACCOUNT1),
        ("account2", ACCOUNT2),
    ):
        records.append(
            {
                "label": label,
                "auth_path": str(codex_home / f"{label}-inventory-auth.json"),
                "lock_path": str(codex_home / f"{label}.lock"),
                "priority": 10 if label == "account1" else 20,
                "weight": 1.0,
                "enabled": True,
                "models": [MODEL],
                "expected_account_hash": codex_oauth_account_identity_hash(
                    account_id
                ),
            }
        )
    return json.dumps(
        {
            "schema_version": 1,
            "routing": {
                "credential_affinity": "interchangeable",
                "strategy": "priority",
            },
            "accounts": records,
        }
    )


class _FakeRedisClient:
    def __init__(self, parent: "_FakeRedisCache") -> None:
        self._parent = parent

    async def get(self, name: str) -> Any:
        return self._parent.data.get(name)

    async def set(
        self, name: str, value: Any, nx: bool = False, ex: Any = None
    ) -> bool:
        if nx and name in self._parent.data:
            return False
        self._parent.data[name] = (
            value if isinstance(value, bytes) else str(value).encode()
        )
        return True

    async def delete(self, *names: str) -> int:
        return sum(self._parent.data.pop(name, None) is not None for name in names)

    async def persist(self, name: str) -> bool:
        self._parent.ttl.pop(name, None)
        return True

    async def eval(self, script: str, _numkeys: int, *args: Any) -> Any:
        key = args[0]
        raw = self._parent.data.get(key)
        current = json.loads(raw) if raw else None
        if "session_owner_retire_owned" in script:
            if current is None:
                return [0, "missing"]
            if current.get("state") != "owned" or current.get("owner") != args[1]:
                return [3, json.dumps(current)]
            self._parent.data.pop(key, None)
            return [1, json.dumps(current)]
        if "session_owner_xai_migration" in script:
            if current is not None:
                return [2, json.dumps(current)]
            payload = json.loads(args[1])
            self._parent.data[key] = json.dumps(payload).encode()
            return [1, json.dumps(payload)]
        if "reservation_token" in script and "owned" in script:
            token = args[1]
            if current is None:
                return [0, "missing"]
            if current.get("state") == "owned":
                return [2, json.dumps(current)]
            if current.get("state") != "reserved" or current.get(
                "reservation_token"
            ) != token:
                return [0, json.dumps(current)]
            payload = json.loads(args[2])
            payload["reserved_at_epoch"] = current.get("reserved_at_epoch")
            self._parent.data[key] = json.dumps(payload).encode()
            return [1, json.dumps(payload)]
        if "DEL" in script and "reserved" in script:
            if (
                current
                and current.get("state") == "reserved"
                and current.get("reservation_token") == args[1]
            ):
                self._parent.data.pop(key, None)
                return 1
            return 0
        if "reservation_token" in script:
            if (
                current
                and current.get("state") == "reserved"
                and current.get("reservation_token") == args[1]
            ):
                payload = json.loads(args[2])
                self._parent.data[key] = json.dumps(payload).encode()
                return 1
            return 0
        raise AssertionError(f"unexpected Lua script: {script[:80]}")


class _FakeRedisCache:
    def __init__(self) -> None:
        self.namespace = "openai041"
        self.data: dict[str, bytes] = {}
        self.ttl: dict[str, Any] = {}
        self.client = _FakeRedisClient(self)

    def check_and_fix_namespace(self, key: str) -> str:
        return f"{self.namespace}:{key}"

    def init_async_client(self) -> _FakeRedisClient:
        return self.client

    def _get_cache_logic(self, cached_response: Any) -> Any:
        value = cached_response
        if value is None:
            return None
        if isinstance(value, bytes):
            value = value.decode()
        return json.loads(value) if isinstance(value, str) else value

    async def async_set_cache(self, key: str, value: Any, **kwargs: Any) -> bool:
        return await self.client.set(
            self.check_and_fix_namespace(key),
            json.dumps(value).encode(),
            nx=bool(kwargs.get("nx")),
            ex=kwargs.get("ttl"),
        )

    async def async_get_cache(self, key: str, **_kwargs: Any) -> Any:
        return self._get_cache_logic(self.data.get(self.check_and_fix_namespace(key)))


class _FakeDualCache:
    def __init__(self, redis_cache: _FakeRedisCache) -> None:
        self.redis_cache = redis_cache

    async def async_get_cache(self, key: str, **kwargs: Any) -> Any:
        return await self.redis_cache.async_get_cache(key, **kwargs)


    async def async_set_cache(self, key: str, value: Any, **kwargs: Any) -> Any:
        return await self.redis_cache.async_set_cache(key, value, **kwargs)


class _RawSseStream(httpx.AsyncByteStream):
    def __init__(self, body: bytes) -> None:
        self._body = body

    async def __aiter__(self):
        yield self._body

    async def aclose(self) -> None:
        return None


def _sse_response(content: str, request: httpx.Request) -> httpx.Response:
    item_id = "msg_openai041"
    response = {
        "id": "resp_success",
        "created_at": 1700000000,
        "model": MODEL,
        "output": [
            {
                "id": item_id,
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [
                    {
                        "type": "output_text",
                        "text": content,
                        "annotations": [],
                    }
                ],
            }
        ],
        "status": "completed",
        "usage": {
            "input_tokens": 1,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens": 2,
            "output_tokens_details": {"reasoning_tokens": 0},
            "total_tokens": 3,
        },
    }
    body = (
        f'event: response.created\ndata: {json.dumps({"type": "response.created", "response": {**response, "output": []}})}\n\n'
        f'event: response.output_item.added\ndata: {json.dumps({"type": "response.output_item.added", "output_index": 0, "item": {**response["output"][0], "status": "in_progress", "content": []}})}\n\n'
        f'event: response.content_part.added\ndata: {json.dumps({"type": "response.content_part.added", "item_id": item_id, "output_index": 0, "content_index": 0, "part": {"type": "output_text", "text": ""}})}\n\n'
        f'event: response.output_text.delta\ndata: {json.dumps({"type": "response.output_text.delta", "item_id": item_id, "output_index": 0, "content_index": 0, "delta": content})}\n\n'
        f'event: response.output_item.done\ndata: {json.dumps({"type": "response.output_item.done", "output_index": 0, "item": response["output"][0]})}\n\n'
        f'event: response.completed\ndata: {json.dumps({"type": "response.completed", "response": response})}\n\n'
        "data: [DONE]\n\n"
    ).encode()
    return httpx.Response(
        200,
        headers={"content-type": "text/event-stream"},
        stream=_RawSseStream(body),
        request=request,
    )


def _mapped_inventory_loader(codex_home: Path):
    real_loader = codex_oauth.load_codex_oauth_inventory
    auth_paths = {
        "account1": codex_home / "auth.json",
        "account2": codex_home / "account2-auth.json",
    }

    def _load():
        inventory = real_loader()
        records = []
        for record in inventory.records:
            fields = dict(record.__dict__)
            fields["auth_path"] = auth_paths[record.label]
            fields["lock_path"] = codex_home / f"{record.label}.lock"
            records.append(type(record)(**fields))
        object.__setattr__(inventory, "records", tuple(records))
        return inventory

    return _load


def _configure_runtime(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    codex_home = tmp_path / "codex-home"
    codex_home.mkdir(parents=True)
    _write_credential(codex_home / "auth.json", ACCOUNT1)
    _write_credential(codex_home / "account2-auth.json", ACCOUNT2)
    mapped_inventory = _mapped_inventory_loader(codex_home)
    monkeypatch.setenv("CODEX_HOME", str(codex_home))
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("LITELLM_CODEX_OAUTH_INVENTORY", _inventory(codex_home))
    monkeypatch.setenv("LITELLM_MASTER_KEY", MASTER_KEY)
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(codex_oauth, "load_codex_oauth_inventory", mapped_inventory)
    monkeypatch.setattr(
        "litellm.secret_managers.codex_oauth_inventory.load_codex_oauth_inventory",
        mapped_inventory,
    )
    return codex_home


@pytest.fixture
def proxy_harness(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> tuple[str, Path, list[httpx.Request], list[str]]:
    codex_home = _configure_runtime(monkeypatch, tmp_path)
    redis_cache = _FakeRedisCache()
    monkeypatch.setattr(durable, "_dual_cache_override", lambda: _FakeDualCache(redis_cache))
    real_publish = cooldown_state._publish_codex_cooldown_memory

    def _publish_debug(*args: Any, **kwargs: Any) -> None:
        keys = kwargs.get("keys") or (args[0] if args else ())
        print("\nPUBLISH_KEYS", keys)
        real_publish(*args, **kwargs)
        family = state.alias_routing_state.codex
        print(
            "FAMILY_KEYS",
            {
                key: max(0.0, value - time.monotonic())
                for key, value in family.cooldown_until_monotonic_by_key.items()
            },
        )

    monkeypatch.setattr(cooldown_state, "_publish_codex_cooldown_memory", _publish_debug)
    requests: list[httpx.Request] = []
    seen_accounts: list[str] = []

    async def _dispatch(upstream_request: httpx.Request) -> httpx.Response:
        requests.append(upstream_request)
        account = upstream_request.headers["ChatGPT-Account-Id"]
        seen_accounts.append(account)
        print(
            "\nREQUEST_BODY "
            + json.dumps(json.loads(upstream_request.content.decode()), indent=2)
        )
        if account == ACCOUNT2:
            return _sse_response("OPENAI041 stock response", upstream_request)
        if len(seen_accounts) == 2:
            exc = HTTPException(
                status_code=429,
                detail={
                    "error": {"code": "usage_limit_reached"},
                    "quota": {"resets_in_seconds": 900},
                    "failover_disposition": "usage_limit_reached",
                },
                headers={"Retry-After": "900"},
            )
            exc.attempted_provider_call = True
            setattr(exc, "_aawm_provider_returned", True)
            raise exc
        return _sse_response("OPENAI041 stock response", upstream_request)

    client = httpx.AsyncClient(
        transport=httpx.MockTransport(_dispatch), timeout=10.0
    )
    monkeypatch.setattr(
        pte,
        "get_async_httpx_client",
        lambda *_args, **_kwargs: SimpleNamespace(client=client),
    )

    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    host, port = sock.getsockname()
    server = uvicorn.Server(
        uvicorn.Config(proxy_app, host=host, port=port, log_level="warning")
    )
    thread = threading.Thread(
        target=lambda: asyncio.run(server.serve(sockets=[sock])), daemon=True
    )
    thread.start()
    deadline = time.monotonic() + 15
    while not server.started and time.monotonic() < deadline and thread.is_alive():
        time.sleep(0.05)
    if not server.started:
        server.should_exit = True
        thread.join(timeout=2)
        raise RuntimeError("stock acceptance proxy failed to start")

    try:
        yield f"http://{host}:{port}", codex_home, requests, seen_accounts
    finally:
        server.should_exit = True
        thread.join(timeout=5)
        sock.close()


def _codex_env(codex_home: Path) -> dict[str, str]:
    env = os.environ.copy()
    env["HOME"] = str(codex_home.parent)
    env["CODEX_HOME"] = str(codex_home)
    env["LITELLM_MASTER_KEY"] = MASTER_KEY
    return env


def _base_overrides(base_url: str) -> list[str]:
    provider = "model_providers.litellm-recipe"
    return [
        "-c",
        'model_provider="litellm-recipe"',
        "-c",
        f'{provider}.name="LiteLLMRecipe"',
        "-c",
        f'{provider}.base_url="{base_url}/openai_passthrough"',
        "-c",
        f'{provider}.requires_openai_auth=true',
        "-c",
        f'{provider}.wire_api="responses"',
        "-c",
        f'{provider}.http_headers={{"Authorization" = "Bearer {MASTER_KEY}"}}',
        "-c",
        'sandbox_mode="read-only"',
        "-c",
        'approval_policy="never"',
        "-c",
        f'model="{MODEL}"',
    ]


def _run_codex(
    args: list[str], codex_home: Path
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        env=_codex_env(codex_home),
        cwd=str(codex_home.parent),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=30,
        check=False,
    )


def _codex_exec(codex_home: Path, base_url: str, prompt: str) -> subprocess.CompletedProcess[str]:
    return _run_codex(
        [
            STOCK_CODEX,
            "exec",
            "--json",
            "--skip-git-repo-check",
            "--ignore-user-config",
            *_base_overrides(base_url),
            prompt,
        ],
        codex_home,
    )


def _codex_resume(codex_home: Path, base_url: str, prompt: str) -> subprocess.CompletedProcess[str]:
    return _run_codex(
        [
            STOCK_CODEX,
            "exec",
            "resume",
            "--last",
            "--json",
            "--skip-git-repo-check",
            "--ignore-user-config",
            *_base_overrides(base_url),
            prompt,
        ],
        codex_home,
    )


def test_stock_codex_quota_fails_over_and_resume_pins_replacement(
    proxy_harness: tuple[str, Path, list[httpx.Request], list[str]],
) -> None:
    base_url, codex_home, _requests, seen_accounts = proxy_harness
    assert codex_home.is_dir()
    first = _codex_exec(codex_home, base_url, "Say OPENAI041 first")
    assert first.returncode == 0, (first.stdout, first.stderr)
    trigger = _codex_exec(codex_home, base_url, "Say OPENAI041 failover")
    print("\nTRIGGER_STDOUT\n" + trigger.stdout)
    print("\nTRIGGER_STDERR\n" + trigger.stderr)
    assert trigger.returncode == 0, (trigger.stdout, trigger.stderr)
    assert seen_accounts == [ACCOUNT1, ACCOUNT1, ACCOUNT2]

    resume = _codex_resume(codex_home, base_url, "Say OPENAI041 again")
    assert resume.returncode == 0, (resume.stdout, resume.stderr)
    assert seen_accounts[-1] == ACCOUNT2
