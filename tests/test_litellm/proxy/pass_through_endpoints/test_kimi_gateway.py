import asyncio
import json
import logging
import os
import time
from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi import FastAPI

from litellm._logging import (
    AawmRouteAccessLogReplacementFilter,
    clear_aawm_route_access_log_replacements,
    verbose_proxy_logger,
)
from litellm.proxy.aawm_route_logging import (
    clear_aawm_route_rollups,
    flush_aawm_route_rollups,
)
from litellm.proxy.pass_through_endpoints import kimi_gateway


class _RawAsyncByteStream(httpx.AsyncByteStream):
    def __init__(self, chunks: tuple[bytes, ...]) -> None:
        self._chunks = chunks

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for chunk in self._chunks:
            yield chunk

    async def aclose(self) -> None:
        return None


class _FailingRawAsyncByteStream(httpx.AsyncByteStream):
    async def __aiter__(self) -> AsyncIterator[bytes]:
        yield b'data: {"choices":[{"delta":{"content":"partial"}}]}\n\n'
        raise httpx.ReadError("Bearer upstream-stream-secret")

    async def aclose(self) -> None:
        return None


def _gateway_app() -> FastAPI:
    app = FastAPI()
    app.include_router(kimi_gateway.router)
    return app


def _request(
    app: FastAPI,
    method: str,
    path: str,
    *,
    client_host: str = "127.0.0.1",
    content: bytes = b"",
    headers: dict[str, str] | None = None,
) -> httpx.Response:
    async def _send() -> httpx.Response:
        transport = httpx.ASGITransport(
            app=app,
            client=(client_host, 49152),
        )
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            return await client.request(
                method,
                path,
                content=content,
                headers=_bearer_headers() if headers is None else headers,
            )

    return asyncio.run(_send())


def _write_credential(path: Path, token: str, expires_at: object | None = None) -> None:
    path.write_text(
        json.dumps(
            {
                "access_token": token,
                "expires_at": (expires_at if expires_at is not None else time.time() + 3600),
            }
        ),
        encoding="utf-8",
    )


def _bearer_headers(token: str = "gateway-access-token") -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def _build_uvicorn_access_record(
    *,
    method: str,
    full_path: str,
    status_code: int,
    client_addr: str = "172.18.0.1:49152",
) -> logging.LogRecord:
    return logging.LogRecord(
        name="uvicorn.access",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg='%s - "%s %s HTTP/%s" %d',
        args=(client_addr, method, full_path, "1.1", status_code),
        exc_info=None,
    )


@pytest.fixture
def credential_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    credential_directory = tmp_path / "credentials"
    credential_directory.mkdir()
    path = credential_directory / "kimi-code.json"
    _write_credential(path, "gateway-access-token")
    monkeypatch.setenv("LITELLM_KIMI_OAUTH_AUTH_FILE", str(path))
    return path


@pytest.fixture(autouse=True)
def clear_gateway_route_log_state() -> Iterator[None]:
    clear_aawm_route_access_log_replacements()
    clear_aawm_route_rollups()
    yield
    clear_aawm_route_access_log_replacements()
    clear_aawm_route_rollups()


def _set_upstream(
    monkeypatch: pytest.MonkeyPatch,
    handler: Any,
) -> list[httpx.Request]:
    requests: list[httpx.Request] = []

    def _recording_handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return handler(request)

    monkeypatch.setattr(
        kimi_gateway,
        "_get_upstream_transport",
        lambda: httpx.MockTransport(_recording_handler),
    )
    return requests


def test_kimi_gateway_accepts_current_bearer_from_docker_bridge_like_peer(
    monkeypatch: pytest.MonkeyPatch,
    credential_path: Path,
) -> None:
    _set_upstream(
        monkeypatch,
        lambda request: httpx.Response(
            200,
            json={"data": []},
            request=request,
        ),
    )

    response = _request(
        _gateway_app(),
        "GET",
        "/kimi/v1/models",
        client_host="172.18.0.1",
        headers=_bearer_headers(),
    )

    assert response.status_code == 200


@pytest.mark.parametrize(
    "headers",
    [
        {},
        _bearer_headers("wrong-access-token"),
        {"Authorization": "Basic gateway-access-token"},
        {"Authorization": "Bearer"},
        {"Authorization": "Bearer  gateway-access-token"},
    ],
)
def test_kimi_gateway_rejects_missing_wrong_or_malformed_bearers(
    credential_path: Path,
    headers: dict[str, str],
) -> None:
    response = _request(
        _gateway_app(),
        "GET",
        "/kimi/v1/models",
        client_host="172.18.0.1",
        headers=headers,
    )

    assert response.status_code == 401
    assert response.json()["detail"] == "Kimi Code gateway authorization is invalid."
    assert "wrong-access-token" not in response.text
    assert "gateway-access-token" not in response.text


def test_kimi_gateway_auth_failure_uses_structured_log_and_failed_route_rollup(
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
    credential_path: Path,
) -> None:
    clear_aawm_route_access_log_replacements()
    clear_aawm_route_rollups()
    monkeypatch.setenv("AAWM_ROUTE_ROLLUP_INTERVAL_SECONDS", "60")
    caplog.set_level(logging.WARNING, logger=verbose_proxy_logger.name)

    response = _request(
        _gateway_app(),
        "GET",
        "/kimi/v1/usages",
        client_host="172.18.0.1",
        headers={"User-Agent": "curl/8.0"},
    )

    assert response.status_code == 401
    error_records = [
        record
        for record in caplog.records
        if record.name == verbose_proxy_logger.name
        and "Kimi Code gateway surfaced handled client/provider error"
        in record.getMessage()
    ]
    assert len(error_records) == 1
    assert error_records[0].levelno == logging.WARNING
    assert (
        error_records[0].getMessage()
        == "Kimi Code gateway surfaced handled client/provider error status=401 "
        "error=Kimi Code gateway authorization is invalid."
    )
    assert getattr(error_records[0], "source") == "kimi_code_gateway"
    assert getattr(error_records[0], "failure_kind") == (
        "gateway_authentication_rejected"
    )

    rendered_rollup = "\n".join(flush_aawm_route_rollups(force=True))
    assert "/kimi/v1/usages" in rendered_rollup
    assert "kimi_code/__managed_account__ - Turns: 0" in rendered_rollup
    assert "[Kimi Code gateway authorization is invalid.] [Failed]" in rendered_rollup

    access_filter = AawmRouteAccessLogReplacementFilter()
    access_record = _build_uvicorn_access_record(
        method="GET",
        full_path="/kimi/v1/usages",
        status_code=401,
    )
    assert access_filter.filter(access_record) is False
    assert access_filter.filter(access_record) is True
    clear_aawm_route_rollups()


def test_kimi_gateway_ignores_forwarded_headers_for_authorization_and_strips_them_upstream(
    monkeypatch: pytest.MonkeyPatch,
    credential_path: Path,
) -> None:
    requests = _set_upstream(
        monkeypatch,
        lambda request: httpx.Response(200, json={"data": []}, request=request),
    )
    app = _gateway_app()

    rejected_response = _request(
        app,
        "GET",
        "/kimi/v1/models",
        client_host="172.18.0.1",
        headers={
            "Authorization": "Bearer wrong-access-token",
            "Forwarded": "for=127.0.0.1",
            "X-Forwarded-For": "127.0.0.1",
        },
    )
    accepted_response = _request(
        app,
        "GET",
        "/kimi/v1/models",
        client_host="172.18.0.1",
        headers={
            **_bearer_headers(),
            "Forwarded": "for=10.0.0.1",
            "X-Forwarded-For": "10.0.0.1",
        },
    )

    assert rejected_response.status_code == 401
    assert accepted_response.status_code == 200
    assert len(requests) == 1
    assert "forwarded" not in requests[0].headers
    assert "x-forwarded-for" not in requests[0].headers


def test_kimi_gateway_uses_current_credential_and_strips_inbound_identity_headers(
    monkeypatch: pytest.MonkeyPatch,
    credential_path: Path,
) -> None:
    requests = _set_upstream(
        monkeypatch,
        lambda request: httpx.Response(200, json={"data": []}, request=request),
    )

    response = _request(
        _gateway_app(),
        "GET",
        "/kimi/v1/models",
        headers={
            **_bearer_headers(),
            "User-Agent": "caller-controlled-agent",
            "X-Msh-Device-Id": "caller-device-id",
            "X-Custom-Header": "retained",
        },
    )
    replacement_path = credential_path.with_name("kimi-code.replacement.json")
    _write_credential(replacement_path, "replacement-access-token")
    os.replace(replacement_path, credential_path)
    stale_response = _request(
        _gateway_app(),
        "GET",
        "/kimi/v1/models",
        headers=_bearer_headers(),
    )
    replacement_response = _request(
        _gateway_app(),
        "GET",
        "/kimi/v1/models",
        headers=_bearer_headers("replacement-access-token"),
    )

    assert response.status_code == 200
    assert stale_response.status_code == 401
    assert replacement_response.status_code == 200
    assert requests[0].headers["authorization"] == "Bearer gateway-access-token"
    assert requests[1].headers["authorization"] == "Bearer replacement-access-token"
    assert requests[0].headers["user-agent"].startswith("litellm/")
    assert "x-msh-device-id" not in requests[0].headers
    assert requests[0].headers["x-custom-header"] == "retained"


@pytest.mark.parametrize(
    ("credential_contents", "expected_detail"),
    [
        (None, "credential file is missing"),
        (
            {
                "access_token": "malformed-credential-secret",
                "expires_at": "not-an-expiry",
            },
            "credential expiry is missing or malformed",
        ),
        (
            {
                "access_token": "expired-credential-secret",
                "expires_at": 1,
            },
            "access token is expired",
        ),
    ],
)
def test_kimi_gateway_fails_closed_without_disclosing_credential_tokens(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    credential_contents: dict[str, object] | None,
    expected_detail: str,
) -> None:
    credential_path = tmp_path / "kimi-code.json"
    monkeypatch.setenv("LITELLM_KIMI_OAUTH_AUTH_FILE", str(credential_path))
    if credential_contents is not None:
        credential_path.write_text(json.dumps(credential_contents), encoding="utf-8")

    response = _request(_gateway_app(), "GET", "/kimi/v1/models")

    assert response.status_code == 401
    assert expected_detail in response.json()["detail"]
    assert "credential-secret" not in response.text


def test_kimi_gateway_preserves_models_usages_and_nonstream_chat_responses(
    monkeypatch: pytest.MonkeyPatch,
    credential_path: Path,
) -> None:
    upstream_responses = {
        "/coding/v1/models": (207, b'{"data":[{"id":"k3"}]}'),
        "/coding/v1/usages": (206, b'{"usage":{"remaining":99}}'),
        "/coding/v1/chat/completions": (201, b'{"id":"chatcmpl-kimi","choices":[]}'),
    }

    def _handler(request: httpx.Request) -> httpx.Response:
        status_code, content = upstream_responses[request.url.path]
        return httpx.Response(
            status_code,
            content=content,
            headers={
                "content-type": "application/json",
                "x-trace-id": "safe-trace",
                "retry-after": "3",
            },
            request=request,
        )

    requests = _set_upstream(monkeypatch, _handler)
    app = _gateway_app()
    models_response = _request(app, "GET", "/kimi/v1/models")
    usages_response = _request(app, "GET", "/kimi/v1/usages")
    chat_body = b'{"model":"k3","messages":[{"role":"user","content":"ping"}]}'
    chat_response = _request(
        app,
        "POST",
        "/kimi/v1/chat/completions",
        content=chat_body,
        headers={**_bearer_headers(), "content-type": "application/json"},
    )

    assert models_response.status_code == 207
    assert models_response.content == upstream_responses["/coding/v1/models"][1]
    assert usages_response.status_code == 206
    assert usages_response.content == upstream_responses["/coding/v1/usages"][1]
    assert chat_response.status_code == 201
    assert chat_response.content == upstream_responses["/coding/v1/chat/completions"][1]
    assert chat_response.headers["x-trace-id"] == "safe-trace"
    assert chat_response.headers["retry-after"] == "3"
    assert requests[2].content == chat_body


def test_kimi_gateway_success_records_route_rollup_turn(
    monkeypatch: pytest.MonkeyPatch,
    credential_path: Path,
) -> None:
    clear_aawm_route_access_log_replacements()
    clear_aawm_route_rollups()
    monkeypatch.setenv("AAWM_ROUTE_ROLLUP_INTERVAL_SECONDS", "60")
    _set_upstream(
        monkeypatch,
        lambda request: httpx.Response(
            200,
            json={"data": []},
            request=request,
        ),
    )

    response = _request(
        _gateway_app(),
        "GET",
        "/kimi/v1/models",
        client_host="172.18.0.1",
        headers={**_bearer_headers(), "User-Agent": "curl/8.0"},
    )

    assert response.status_code == 200
    rendered_rollup = "\n".join(flush_aawm_route_rollups(force=True))
    assert "/kimi/v1/models" in rendered_rollup
    assert "kimi_code/__managed_account__ - Turns: 1" in rendered_rollup

    access_filter = AawmRouteAccessLogReplacementFilter()
    assert (
        access_filter.filter(
            _build_uvicorn_access_record(
                method="GET",
                full_path="/kimi/v1/models",
                status_code=200,
            )
        )
        is False
    )
    clear_aawm_route_rollups()


def test_kimi_gateway_upstream_failure_logs_exact_source_error_and_failed_rollup(
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
    credential_path: Path,
) -> None:
    clear_aawm_route_access_log_replacements()
    clear_aawm_route_rollups()
    monkeypatch.setenv("AAWM_ROUTE_ROLLUP_INTERVAL_SECONDS", "60")
    caplog.set_level(logging.WARNING, logger=verbose_proxy_logger.name)
    upstream_body = b'{"error":{"message":"Kimi quota exhausted.","code":"quota"}}'
    _set_upstream(
        monkeypatch,
        lambda request: httpx.Response(
            429,
            content=upstream_body,
            headers={"content-type": "application/json"},
            request=request,
        ),
    )

    response = _request(
        _gateway_app(),
        "POST",
        "/kimi/v1/chat/completions",
        client_host="172.18.0.1",
        content=b'{"model":"k3-high","messages":[]}',
        headers={
            **_bearer_headers(),
            "content-type": "application/json",
            "User-Agent": "codex-cli/0.144.6",
        },
    )

    assert response.status_code == 429
    assert response.content == upstream_body
    error_records = [
        record
        for record in caplog.records
        if record.name == verbose_proxy_logger.name
        and "Kimi Code gateway surfaced handled client/provider error"
        in record.getMessage()
    ]
    assert len(error_records) == 1
    assert error_records[0].getMessage().endswith(
        "status=429 error=Kimi quota exhausted."
    )
    assert '{"error":' not in error_records[0].getMessage()
    assert getattr(error_records[0], "failure_kind") == "gateway_upstream_non_success"

    rendered_rollup = "\n".join(flush_aawm_route_rollups(force=True))
    assert "/kimi/v1/chat/completions" in rendered_rollup
    assert "kimi_code/k3-high - Turns: 0" in rendered_rollup
    assert "[Kimi quota exhausted.] [Failed]" in rendered_rollup

    access_filter = AawmRouteAccessLogReplacementFilter()
    assert (
        access_filter.filter(
            _build_uvicorn_access_record(
                method="POST",
                full_path="/kimi/v1/chat/completions",
                status_code=429,
            )
        )
        is False
    )
    clear_aawm_route_rollups()


def test_kimi_gateway_preserves_raw_sse_bytes_order_done_and_usage(
    monkeypatch: pytest.MonkeyPatch,
    credential_path: Path,
) -> None:
    raw_sse = (
        b'data: {"choices":[{"delta":{"reasoning_content":"first"}}]}\n\n'
        b'data: {"choices":[{"delta":{"content":"second"}}]}\n\n'
        b'data: {"choices":[],"usage":{"total_tokens":7}}\n\n'
        b"data: [DONE]\n\n"
    )
    _set_upstream(
        monkeypatch,
        lambda request: httpx.Response(
            200,
            stream=_RawAsyncByteStream((raw_sse,)),
            headers={"content-type": "text/event-stream", "x-trace-id": "safe-trace"},
            request=request,
        ),
    )

    response = _request(
        _gateway_app(),
        "POST",
        "/kimi/v1/chat/completions",
        content=b'{"model":"k3","stream":true,"messages":[]}',
        headers={**_bearer_headers(), "content-type": "application/json"},
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.content == raw_sse
    assert response.content.index(b"reasoning_content") < response.content.index(b'"content":"second"')
    assert response.content.index(b'"total_tokens":7') < response.content.index(b"[DONE]")


def test_kimi_gateway_stream_read_failure_is_logged_without_asgi_exception(
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
    credential_path: Path,
) -> None:
    monkeypatch.setenv("AAWM_ROUTE_ROLLUP_INTERVAL_SECONDS", "60")
    caplog.set_level(logging.ERROR, logger=verbose_proxy_logger.name)
    _set_upstream(
        monkeypatch,
        lambda request: httpx.Response(
            200,
            stream=_FailingRawAsyncByteStream(),
            headers={"content-type": "text/event-stream", "x-trace-id": "trace-123"},
            request=request,
        ),
    )

    response = _request(
        _gateway_app(),
        "POST",
        "/kimi/v1/chat/completions",
        client_host="172.18.0.1",
        content=b'{"model":"k3-high","stream":true,"messages":[]}',
        headers={
            **_bearer_headers(),
            "content-type": "application/json",
            "User-Agent": "codex-cli/0.144.6",
        },
    )

    assert response.status_code == 200
    assert b'"content":"partial"' in response.content
    error_records = [
        record
        for record in caplog.records
        if record.name == verbose_proxy_logger.name
        and getattr(record, "failure_kind", None)
        == "gateway_upstream_stream_failed"
    ]
    assert len(error_records) == 1
    assert error_records[0].levelno == logging.ERROR
    assert error_records[0].getMessage().endswith(
        "status=502 error=Kimi Code gateway upstream response stream failed."
    )
    assert getattr(error_records[0], "trace_id") == "trace-123"
    assert "upstream-stream-secret" not in error_records[0].getMessage()
    assert not error_records[0].exc_info

    rendered_rollup = "\n".join(flush_aawm_route_rollups(force=True))
    assert "kimi_code/k3-high - Turns: 0" in rendered_rollup
    assert "[Kimi Code gateway upstream response stream failed.] [Failed]" in (
        rendered_rollup
    )
    assert "upstream-stream-secret" not in rendered_rollup


def test_kimi_gateway_uses_kimi_code_base_url_not_kimi_cli_base_url(
    monkeypatch: pytest.MonkeyPatch,
    credential_path: Path,
) -> None:
    monkeypatch.setenv("KIMI_CODE_BASE_URL", "https://kimi-gateway.test/custom/v1/")
    monkeypatch.setenv("KIMI_CLI_BASE_URL", "https://must-not-be-used.test/ignored")
    requests = _set_upstream(
        monkeypatch,
        lambda request: httpx.Response(200, json={"data": []}, request=request),
    )

    response = _request(_gateway_app(), "GET", "/kimi/v1/models")

    assert response.status_code == 200
    assert str(requests[0].url) == "https://kimi-gateway.test/custom/v1/models"


def test_kimi_gateway_rejects_invalid_base_url_without_an_upstream_request(
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
    credential_path: Path,
) -> None:
    clear_aawm_route_access_log_replacements()
    clear_aawm_route_rollups()
    monkeypatch.setenv("AAWM_ROUTE_ROLLUP_INTERVAL_SECONDS", "60")
    caplog.set_level(logging.ERROR, logger=verbose_proxy_logger.name)
    monkeypatch.setenv("KIMI_CODE_BASE_URL", "http://insecure.example.test/coding/v1")
    requests = _set_upstream(
        monkeypatch,
        lambda request: httpx.Response(200, json={"data": []}, request=request),
    )

    response = _request(_gateway_app(), "GET", "/kimi/v1/models")

    assert response.status_code == 503
    assert requests == []
    assert any(
        record.name == verbose_proxy_logger.name
        and record.levelno == logging.ERROR
        and record.getMessage()
        == "Kimi Code gateway surfaced handled client/provider error status=503 "
        "error=Kimi Code gateway upstream is unavailable."
        and getattr(record, "failure_kind", None)
        == "gateway_configuration_unavailable"
        for record in caplog.records
    )
    rendered_rollup = "\n".join(flush_aawm_route_rollups(force=True))
    assert "[Kimi Code gateway upstream is unavailable.] [Failed]" in rendered_rollup
    clear_aawm_route_rollups()


def test_kimi_gateway_transport_failure_uses_handled_502_log_and_rollup(
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
    credential_path: Path,
) -> None:
    clear_aawm_route_access_log_replacements()
    clear_aawm_route_rollups()
    monkeypatch.setenv("AAWM_ROUTE_ROLLUP_INTERVAL_SECONDS", "60")
    caplog.set_level(logging.ERROR, logger=verbose_proxy_logger.name)

    def _raise_connect_error(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("provider connection token=secret", request=request)

    _set_upstream(monkeypatch, _raise_connect_error)

    response = _request(
        _gateway_app(),
        "GET",
        "/kimi/v1/models",
        headers={**_bearer_headers(), "User-Agent": "curl/8.0"},
    )

    assert response.status_code == 502
    assert response.json()["detail"] == "Kimi Code gateway upstream request failed."
    assert any(
        record.name == verbose_proxy_logger.name
        and record.levelno == logging.ERROR
        and record.getMessage()
        == "Kimi Code gateway surfaced handled client/provider error status=502 "
        "error=Kimi Code gateway upstream request failed."
        and getattr(record, "failure_kind", None)
        == "gateway_upstream_request_failed"
        and not record.exc_info
        for record in caplog.records
    )
    rendered_rollup = "\n".join(flush_aawm_route_rollups(force=True))
    assert "[Kimi Code gateway upstream request failed.] [Failed]" in rendered_rollup
    assert "provider connection" not in rendered_rollup
    clear_aawm_route_rollups()


def test_kimi_gateway_has_no_oauth_or_token_proxy_routes() -> None:
    app = _gateway_app()
    registered_paths = {route.path for route in app.routes}

    assert registered_paths == {
        "/docs",
        "/docs/oauth2-redirect",
        "/kimi/v1/chat/completions",
        "/kimi/v1/models",
        "/kimi/v1/usages",
        "/openapi.json",
        "/redoc",
    }
    response = _request(app, "POST", "/kimi/v1/oauth/token")
    assert response.status_code == 404
