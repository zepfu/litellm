"""OPENAI041 stock-client acceptance harness for credential failover affinity."""

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
from copy import deepcopy
from typing import Any
from types import SimpleNamespace

import httpx
import pytest
import uvicorn

from litellm.proxy.pass_through_endpoints.aawm_alias_routing import (
    alias_routing_state,
    codex_oauth,
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


def _failure_plan(failure_kind: str) -> list[str]:
    return ["success", failure_kind, "success", "success"]


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


def _inventory(
    codex_home: Path,
    source_account: str,
    fallback_account: str,
) -> str:
    return json.dumps(
        {
            "schema_version": 1,
            "routing": {
                "credential_affinity": "interchangeable",
                "strategy": "priority",
            },
            "accounts": [
                {
                    "label": "account1",
                    "auth_path": str(codex_home / "auth.json"),
                    "lock_path": str(codex_home / "account1.lock"),
                    "priority": 10,
                    "weight": 1.0,
                    "enabled": True,
                    "models": [MODEL],
                    "expected_account_hash": codex_oauth_account_identity_hash(source_account),
                },
                {
                    "label": "account2",
                    "auth_path": str(codex_home / "account2-auth.json"),
                    "lock_path": str(codex_home / "account2.lock"),
                    "priority": 20,
                    "weight": 1.0,
                    "enabled": True,
                    "models": [MODEL],
                    "expected_account_hash": codex_oauth_account_identity_hash(fallback_account),
                },
            ],
        }
    )


class _FakeRedisClient:
    def __init__(self, parent: "_FakeRedisCache") -> None:
        self._parent = parent

    async def get(self, name: str) -> Any:
        return self._parent.data.get(name)

    async def set(self, name: str, value: Any, nx: bool = False, ex: Any = None) -> bool:
        if nx and name in self._parent.data:
            return False
        self._parent.data[name] = value if isinstance(value, bytes) else str(value).encode()
        return True

    async def delete(self, *names: str) -> int:
        return sum(self._parent.data.pop(name, None) is not None for name in names)

    async def persist(self, name: str) -> bool:
        self._parent.ttl.pop(name, None)
        return True

    @staticmethod
    def _lua_attributes_equal(left: Any, right: Any) -> bool:
        if not isinstance(left, dict) or not isinstance(right, dict):
            return False

        def normalize(value: Any) -> Any:
            return None if value is None else value

        left_count = len(left)
        right_count = len(right)
        if left_count != right_count:
            return False
        return all(str(normalize(left.get(key))) == str(normalize(right.get(key))) for key in left)

    def _portable_failover_rebind(self, args: tuple[Any, ...]) -> list[Any]:
        key = args[0]
        raw = self._parent.data.get(key)
        if raw is None:
            return [0, "missing"]
        current = json.loads(raw)
        if not isinstance(current, dict) or not isinstance(current.get("attributes"), dict):
            return [-1, "malformed"]
        expected_source = json.loads(args[2])
        expected_destination = json.loads(args[4])
        destination_owner = args[3]
        if (
            current.get("state") == "owned"
            and current.get("owner") == destination_owner
            and self._lua_attributes_equal(
                current["attributes"],
                expected_destination["attributes"],
            )
        ):
            return [2, raw]
        if current.get("state") != "owned":
            return [3, raw]
        old_ordinal = float(current.get("terminal_completion_ordinal_epoch") or 0)
        new_ordinal = float(args[5])
        if new_ordinal <= old_ordinal:
            return [0, raw]
        if current.get("owner") != args[1] or not self._lua_attributes_equal(
            current["attributes"],
            expected_source["attributes"],
        ):
            return [0, raw]
        payload = dict(expected_destination)
        if current.get("owned_at_epoch") is not None:
            payload["owned_at_epoch"] = current["owned_at_epoch"]
        if current.get("reserved_at_epoch") is not None:
            payload["reserved_at_epoch"] = current["reserved_at_epoch"]
        if new_ordinal > 0:
            payload["terminal_completion_ordinal_epoch"] = new_ordinal
        encoded = json.dumps(payload).encode()
        self._parent.data[key] = encoded
        self._parent.ttl.pop(key, None)
        return [1, encoded]

    def _mutable_owner_refresh(self, args: tuple[Any, ...]) -> list[Any]:
        key = args[0]
        raw = self._parent.data.get(key)
        if raw is None:
            return [0, "missing"]
        current = json.loads(raw)
        if not isinstance(current, dict):
            return [-1, "malformed"]
        if current.get("state") != "owned":
            return [0, json.dumps(current)]
        if args[1] and current.get("owner") != args[1]:
            return [0, json.dumps(current)]
        ordinal = float(args[3])
        if ordinal <= float(current.get("terminal_completion_ordinal_epoch") or 0):
            return [0, json.dumps(current)]
        updates = json.loads(args[2])
        expected = json.loads(args[5])
        mutable_keys = json.loads(args[6])
        if not isinstance(updates, dict) or not isinstance(expected, dict) or not isinstance(mutable_keys, list):
            return [-1, "malformed"]
        if any(key not in mutable_keys for key in updates):
            return [-1, "mutable_contract_violation"]
        if not self._lua_attributes_equal(current.get("attributes", {}), expected):
            return [0, json.dumps(current)]
        payload = deepcopy(current)
        attributes = payload.setdefault("attributes", {})
        for key in mutable_keys:
            if updates.get(key) is not None:
                attributes[key] = updates[key]
            else:
                attributes.pop(key, None)
        payload["terminal_completion_ordinal_epoch"] = ordinal
        payload["last_renewed_at_epoch"] = float(args[4]) or ordinal
        encoded = json.dumps(payload).encode()
        self._parent.data[key] = encoded
        self._parent.ttl.pop(key, None)
        return [1, encoded]

    async def eval(self, script: str, _numkeys: int, *args: Any) -> Any:
        key = args[0]
        raw = self._parent.data.get(key)
        current = json.loads(raw) if raw else None
        if "mutable_contract_violation" in script:
            return self._mutable_owner_refresh(args)
        if "local function attributes_equal(left, right)" in script:
            return self._portable_failover_rebind(args)
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
            if current.get("state") != "reserved" or current.get("reservation_token") != token:
                return [0, json.dumps(current)]
            payload = json.loads(args[2])
            payload["reserved_at_epoch"] = current.get("reserved_at_epoch")
            self._parent.data[key] = json.dumps(payload).encode()
            return [1, json.dumps(payload)]
        if "DEL" in script and "reserved" in script:
            if current and current.get("state") == "reserved" and current.get("reservation_token") == args[1]:
                self._parent.data.pop(key, None)
                return 1
            return 0
        if "reservation_token" in script:
            if current and current.get("state") == "reserved" and current.get("reservation_token") == args[1]:
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


def _assert_stock_contract(
    request: httpx.Request,
    *,
    expected_markers: list[str],
) -> None:
    body = json.loads(request.content.decode())
    assert body.get("previous_response_id", None) is None
    messages = body["input"]
    assert all(item.get("type") == "message" for item in messages)
    markers = [
        item["content"][0]["text"]
        for item in messages
        if item.get("role") in {"user", "assistant"} and "OPENAI041" in item["content"][0].get("text", "")
    ]
    assert markers == expected_markers
    tools = body["tools"]
    namespace = next((tool for tool in tools if tool.get("name") == "functions"), None)
    assert namespace is not None
    assert namespace["type"] == "namespace"
    assert namespace["description"] == ""
    assert [child["type"] for child in namespace["tools"]] == [
        "custom",
        "function",
        "function",
    ]
    assert [child["name"] for child in namespace["tools"]] == [
        "exec",
        "wait",
        "request_user_input",
    ]


def _sse_response(
    content: str,
    request: httpx.Request,
    *,
    response_id: str = "resp_success",
) -> httpx.Response:
    item_id = f"msg_{response_id}"
    response = {
        "id": response_id,
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
        f'data: {json.dumps({"type": "response.created", "sequence_number": 1, "response": {**response, "output": []}})}\n\n'
        f'data: {json.dumps({"type": "response.output_item.added", "sequence_number": 2, "output_index": 0, "item": {**response["output"][0], "status": "in_progress", "content": []}})}\n\n'
        f'data: {json.dumps({"type": "response.content_part.added", "sequence_number": 3, "item_id": item_id, "output_index": 0, "content_index": 0, "part": {"type": "output_text", "text": ""}})}\n\n'
        f'data: {json.dumps({"type": "response.output_text.delta", "sequence_number": 4, "item_id": item_id, "output_index": 0, "content_index": 0, "delta": content})}\n\n'
        f'data: {json.dumps({"type": "response.output_item.done", "sequence_number": 5, "output_index": 0, "item": response["output"][0]})}\n\n'
        f'data: {json.dumps({"type": "response.completed", "sequence_number": 6, "response": response})}\n\n'
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


def _configure_runtime(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    source_account: str,
    fallback_account: str,
) -> Path:
    codex_home = tmp_path / "codex-home"
    codex_home.mkdir(parents=True)
    _write_credential(codex_home / "auth.json", source_account)
    _write_credential(codex_home / "account2-auth.json", fallback_account)
    mapped_inventory = _mapped_inventory_loader(codex_home)
    monkeypatch.setenv("CODEX_HOME", str(codex_home))
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv(
        "LITELLM_CODEX_OAUTH_INVENTORY",
        _inventory(codex_home, source_account, fallback_account),
    )
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
    request: pytest.FixtureRequest,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> tuple[str, Path, _FakeRedisCache, list[httpx.Request], list[str], str, str, str]:
    reverse, failure_kind = request.param
    source_account, fallback_account = (ACCOUNT2, ACCOUNT1) if reverse else (ACCOUNT1, ACCOUNT2)
    codex_home = _configure_runtime(
        monkeypatch,
        tmp_path,
        source_account=source_account,
        fallback_account=fallback_account,
    )
    redis_cache = _FakeRedisCache()
    monkeypatch.setattr(durable, "_dual_cache_override", lambda: _FakeDualCache(redis_cache))
    requests: list[httpx.Request] = []
    seen_accounts: list[str] = []

    async def _dispatch(upstream_request: httpx.Request) -> httpx.Response:
        requests.append(upstream_request)
        account = upstream_request.headers["ChatGPT-Account-Id"]
        seen_accounts.append(account)
        # Request 1 warms the stock session. Request 2 exhausts the preferred
        # account. Request 3 proves the selected replacement succeeds, and
        # request 4 proves that replacement remains pinned on resume.
        failure_mode = _failure_plan(failure_kind)[min(len(requests) - 1, len(_failure_plan(failure_kind)) - 1)]
        if failure_mode == "success":
            return _sse_response(
                f"OPENAI041 stock response-{len(seen_accounts)}",
                upstream_request,
                response_id=f"resp_{len(seen_accounts)}",
            )
        if failure_mode == "quota":
            return httpx.Response(
                status_code=429,
                content=json.dumps(
                    {
                        "error": {
                            "type": "usage_limit_reached",
                            "code": "usage_limit_reached",
                            "message": "Quota exhausted.",
                            "plan_type": "team",
                            "resets_in_seconds": 900,
                        }
                    }
                ).encode(),
                headers={
                    "content-type": "application/json",
                    "retry-after": "900",
                },
                request=upstream_request,
            )
        if failure_mode == "capacity":
            return httpx.Response(
                status_code=200,
                headers={"content-type": "text/event-stream"},
                content=b"".join(
                    [
                        b'data: {"type":"response.created","sequence_number":1,"response":{"id":"resp_capacity","created_at":1700000000,"output":[],"status":"in_progress"}}\n\n',
                        (
                            b'data: {"type":"response.failed","sequence_number":2,'
                            b'"response":{"id":"resp_capacity","status":"failed",'
                            b'"output":[],"error":{"code":"server_is_overloaded","type":"server_error","message":"The selected model is at capacity."}}}\n\n'
                        ),
                        b"data: [DONE]\n\n",
                    ]
                ),
                request=upstream_request,
            )
        raise AssertionError(f"unexpected request ordinal {len(requests)}")

    client = httpx.AsyncClient(transport=httpx.MockTransport(_dispatch), timeout=10.0)
    monkeypatch.setattr(
        pte,
        "get_async_httpx_client",
        lambda *_args, **_kwargs: SimpleNamespace(client=client),
    )

    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    host, port = sock.getsockname()
    server = uvicorn.Server(uvicorn.Config(proxy_app, host=host, port=port, log_level="warning"))
    thread = threading.Thread(target=lambda: asyncio.run(server.serve(sockets=[sock])), daemon=True)
    thread.start()
    deadline = time.monotonic() + 15
    while not server.started and time.monotonic() < deadline and thread.is_alive():
        time.sleep(0.05)
    if not server.started:
        server.should_exit = True
        thread.join(timeout=2)
        raise RuntimeError("stock acceptance proxy failed to start")

    try:
        yield (
            f"http://{host}:{port}",
            codex_home,
            redis_cache,
            requests,
            seen_accounts,
            source_account,
            fallback_account,
            failure_kind,
        )
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
        f"{provider}.requires_openai_auth=true",
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


def _run_codex(args: list[str], codex_home: Path) -> subprocess.CompletedProcess[str]:
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


@pytest.mark.parametrize(
    "proxy_harness",
    [(False, "quota"), (True, "quota"), (False, "capacity"), (True, "capacity")],
    indirect=True,
    ids=["quota-forward", "quota-reverse", "capacity-forward", "capacity-reverse"],
)
def test_stock_codex_failure_affinity(
    proxy_harness: tuple[
        str,
        Path,
        _FakeRedisCache,
        list[httpx.Request],
        list[str],
        str,
        str,
        str,
    ],
) -> None:
    (
        base_url,
        codex_home,
        redis_cache,
        requests,
        seen_accounts,
        source_account,
        fallback_account,
        failure_kind,
    ) = proxy_harness
    assert codex_home.is_dir()
    alias_routing_state.reset_for_tests()
    try:
        _run_stock_affinity_case(
            base_url,
            codex_home,
            redis_cache,
            requests,
            seen_accounts,
            source_account,
            fallback_account,
            failure_kind,
        )
    finally:
        alias_routing_state.reset_for_tests()


def _run_stock_affinity_case(
    base_url: str,
    codex_home: Path,
    redis_cache: _FakeRedisCache,
    requests: list[httpx.Request],
    seen_accounts: list[str],
    source_account: str,
    fallback_account: str,
    failure_kind: str = "quota",
) -> None:
    prompts = [
        "Say OPENAI041 first",
        "Say OPENAI041 failover",
        "Say OPENAI041 again",
        "Say OPENAI041 pinned",
    ]
    executions = [_codex_exec(codex_home, base_url, prompts[0])]
    executions.extend(_codex_resume(codex_home, base_url, prompt) for prompt in prompts[1:])
    for execution in executions:
        assert execution.returncode == 0, (execution.stdout, execution.stderr)

    for successful_execution in (executions[0], executions[2], executions[3]):
        assert successful_execution.stdout.count('"type":"turn.started"') == 1
        assert successful_execution.stdout.count('"type":"item.completed"') == 1
        assert successful_execution.stdout.count('"type":"turn.completed"') == 1
        assert successful_execution.stdout.count('"text":"OPENAI041 stock response-') == 1

    trigger, resume = executions[1], executions[-1]
    expected_attempts = [
        source_account,
        source_account,
        fallback_account,
        fallback_account,
        fallback_account,
    ]
    invocation_for_attempt = (0, 1, 1, 2, 3)
    marker_count_for_attempt = (1, 3, 3, 5, 7)
    response_numbers = (1, 3, 4)
    expected_owner = fallback_account
    assert seen_accounts == expected_attempts
    assert len(requests) == len(expected_attempts)
    for index, request in enumerate(requests, start=1):
        invocation_index = invocation_for_attempt[index - 1]
        assert invocation_index < len(executions)
        body = json.loads(request.content.decode())
        _assert_stock_contract(
            request,
            expected_markers=_expected_conversation_markers(
                prompts,
                marker_count_for_attempt[index - 1],
                response_numbers,
            ),
        )
        _assert_once_each(
            json.dumps(body["input"]),
            ["OPENAI041 first", "OPENAI041 failover", "OPENAI041 again", "OPENAI041 pinned"][: invocation_index + 1],
        )
        assert body["model"] == MODEL

    thread_id = json.loads(next(line for line in trigger.stdout.splitlines() if '"type":"thread.started"' in line))[
        "thread_id"
    ]
    resume_thread_id = json.loads(
        next(line for line in resume.stdout.splitlines() if '"type":"thread.started"' in line)
    )["thread_id"]
    assert resume_thread_id == thread_id

    owner_keys = [key for key in redis_cache.data if ":session_owner:" in key]
    assert len(owner_keys) == 1
    owner_record = json.loads(redis_cache.data[owner_keys[0]])
    assert owner_record["state"] == "owned"
    assert owner_record["attributes"]["account_hash"] == codex_oauth_account_identity_hash(expected_owner)


def _expected_conversation_markers(
    prompts: list[str],
    retained_message_count: int,
    response_numbers: tuple[int, ...] = (1, 3, 4),
) -> list[str]:
    expected: list[str] = []
    for index in range((retained_message_count + 1) // 2):
        expected.append(prompts[index])
        if 2 * index + 1 < retained_message_count:
            response_number = response_numbers[index]
            expected.append(f"OPENAI041 stock response-{response_number}")
    return expected


def _assert_once_each(haystack: str, needles: list[str]) -> None:
    for needle in needles:
        assert haystack.count(needle) == 1, (needle, haystack.count(needle))
