import asyncio
import json
import logging
from collections.abc import AsyncIterator, Iterator
from typing import Any

import httpx
import pytest
from fastapi import FastAPI, HTTPException

from litellm._logging import (
    AawmRouteAccessLogReplacementFilter,
    clear_aawm_route_access_log_replacements,
    verbose_proxy_logger,
)
from litellm.proxy.aawm_route_logging import (
    clear_aawm_route_rollups,
    flush_aawm_route_rollups,
)
from litellm.proxy.pass_through_endpoints import muse_code_gateway
from litellm.proxy.response_api_endpoints import endpoints as responses_endpoints


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
        yield b"event: response.created\ndata: {\"type\":\"response.created\"}\n\n"
        raise httpx.ReadError("Bearer upstream-stream-secret")

    async def aclose(self) -> None:
        return None


def _app() -> FastAPI:
    app = FastAPI()
    app.include_router(muse_code_gateway.router)
    app.include_router(responses_endpoints.router)
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
                headers={} if headers is None else headers,
            )

    return asyncio.run(_send())


def _bearer_headers(token: str = "meta-oauth-secret-token") -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def _muse_identity_headers(token: str = "meta-oauth-secret-token") -> dict[str, str]:
    return {
        **_bearer_headers(token),
        "x-client-id": "tbh:exec",
        "user-agent": "muse-build/1.1.1 (non-interactive; linux-x86_64; build abc)",
        "x-tbh-session-id": "11111111-1111-1111-1111-111111111111",
        "x-meta-ai-gateway-session-id": "11111111-1111-1111-1111-111111111111",
        "traceparent": "00-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa-bbbbbbbbbbbbbbbb-01",
        "content-type": "application/json",
        "accept": "text/event-stream",
    }


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


def _set_upstream(
    monkeypatch: pytest.MonkeyPatch,
    handler: Any,
) -> list[httpx.Request]:
    requests: list[httpx.Request] = []

    def _recording_handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return handler(request)

    monkeypatch.setattr(
        muse_code_gateway,
        "_get_upstream_transport",
        lambda: httpx.MockTransport(_recording_handler),
    )
    return requests


def _muse_responses_payload(*, stream: bool = True) -> dict[str, Any]:
    return {
        "model": "muse-spark-1.3-contributor",
        "input": [
            {
                "type": "message",
                "role": "user",
                "content": "Reply with the single word pong.",
            }
        ],
        "stream": stream,
        "store": False,
        "previous_response_id": "resp_meta_native_not_litellm",
        "prompt_cache_key": "tbh:main:11111111-1111-1111-1111-111111111111",
        "tools": [
            {
                "type": "namespace",
                "name": "muse",
                "description": "Muse Code tool set.",
                "tools": [
                    {
                        "type": "function",
                        "name": "bash",
                        "strict": True,
                        "parameters": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {},
                            "required": [],
                        },
                    }
                ],
            }
        ],
    }


def _muse_responses_body(*, stream: bool = True) -> bytes:
    return json.dumps(_muse_responses_payload(stream=stream)).encode("utf-8")


@pytest.fixture(autouse=True)
def enable_muse_facade(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AAWM_MUSE_CODE_FACADE_ENABLED", "1")
    monkeypatch.setenv("AAWM_ROUTE_ROLLUP_INTERVAL_SECONDS", "60")


@pytest.fixture(autouse=True)
def clear_gateway_route_log_state() -> Iterator[None]:
    clear_aawm_route_access_log_replacements()
    clear_aawm_route_rollups()
    yield
    clear_aawm_route_access_log_replacements()
    clear_aawm_route_rollups()


def test_muse_catalog_forwards_bearer_to_meta_and_returns_upstream_body(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    catalog = {
        "object": "list",
        "data": [
            {
                "id": "muse-spark-1.3-contributor",
                "object": "model",
                "created": 1788320726,
                "owned_by": "meta",
            }
        ],
    }
    requests = _set_upstream(
        monkeypatch,
        lambda request: httpx.Response(
            200,
            json=catalog,
            headers={"content-type": "application/json"},
            request=request,
        ),
    )

    response = _request(
        _app(),
        "GET",
        "/muse-code/models",
        headers={
            **_bearer_headers(),
            "x-client-id": "tbh:exec",
            "user-agent": "muse-build/1.1.1 (non-interactive; linux-x86_64; build abc)",
            "accept": "*/*",
        },
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/json")
    assert response.json() == catalog
    assert len(requests) == 1
    assert str(requests[0].url) == "https://api.meta.ai/muse-code/models"
    assert requests[0].method == "GET"
    assert requests[0].headers["authorization"] == "Bearer meta-oauth-secret-token"
    assert requests[0].headers["x-client-id"] == "tbh:exec"
    assert "meta-oauth-secret-token" not in response.text


def test_muse_catalog_is_not_a_locally_synthesized_list(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    catalog = {
        "object": "list",
        "data": [
            {
                "id": "muse-spark-from-meta",
                "object": "model",
                "owned_by": "meta",
            }
        ],
    }
    _set_upstream(
        monkeypatch,
        lambda request: httpx.Response(200, json=catalog, request=request),
    )

    response = _request(
        _app(),
        "GET",
        "/muse-code/models",
        headers=_bearer_headers(),
    )

    ids = [row["id"] for row in response.json()["data"]]
    assert ids == ["muse-spark-from-meta"]
    assert "muse-spark-1.3-contributor" not in ids


@pytest.mark.parametrize(
    "headers",
    [
        {},
        {"Authorization": "Basic meta-oauth-secret-token"},
        {"Authorization": "Bearer"},
        {"Authorization": "Bearer  meta-oauth-secret-token"},
        {"Authorization": "Bearer meta-oauth-secret-token extra"},
    ],
)
def test_muse_catalog_rejects_missing_or_malformed_bearers(
    headers: dict[str, str],
) -> None:
    response = _request(
        _app(),
        "GET",
        "/muse-code/models",
        headers=headers,
    )

    assert response.status_code == 401
    assert response.json()["detail"] == "Muse Code gateway authorization is invalid."
    assert "meta-oauth-secret-token" not in response.text


def test_muse_responses_forwards_unmodified_body_and_identity_headers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    body = _muse_responses_body(stream=False)
    requests = _set_upstream(
        monkeypatch,
        lambda request: httpx.Response(
            200,
            json={"id": "resp_meta", "model": "muse-spark-1.3-contributor"},
            headers={"content-type": "application/json", "x-request-id": "meta-rid"},
            request=request,
        ),
    )

    response = _request(
        _app(),
        "POST",
        "/responses",
        content=body,
        headers=_muse_identity_headers(),
    )

    assert response.status_code == 200
    assert response.json()["model"] == "muse-spark-1.3-contributor"
    assert len(requests) == 1
    assert str(requests[0].url) == "https://api.meta.ai/v1/responses"
    assert requests[0].method == "POST"
    assert requests[0].headers["authorization"] == "Bearer meta-oauth-secret-token"
    assert requests[0].headers["x-client-id"] == "tbh:exec"
    assert (
        requests[0].headers["x-tbh-session-id"]
        == "11111111-1111-1111-1111-111111111111"
    )
    assert (
        requests[0].headers["x-meta-ai-gateway-session-id"]
        == "11111111-1111-1111-1111-111111111111"
    )
    assert (
        requests[0].headers["traceparent"]
        == "00-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa-bbbbbbbbbbbbbbbb-01"
    )
    forwarded = json.loads(requests[0].content)
    original = json.loads(body)
    assert forwarded == original
    assert forwarded["tools"][0]["type"] == "namespace"
    assert forwarded["previous_response_id"] == "resp_meta_native_not_litellm"
    assert forwarded["model"] == "muse-spark-1.3-contributor"
    assert "meta-oauth-secret-token" not in response.text


def test_muse_responses_preserves_raw_sse_bytes_including_done(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_sse = (
        b"event: response.created\n"
        b'data: {"type":"response.created","sequence_number":0}\n\n'
        b"event: response.in_progress\n"
        b'data: {"type":"response.in_progress","sequence_number":1}\n\n'
        b"event: response.completed\n"
        b'data: {"type":"response.completed","sequence_number":2}\n\n'
        b"data: [DONE]\n\n"
    )
    _set_upstream(
        monkeypatch,
        lambda request: httpx.Response(
            200,
            stream=_RawAsyncByteStream((raw_sse,)),
            headers={"content-type": "text/event-stream", "x-request-id": "safe-trace"},
            request=request,
        ),
    )

    response = _request(
        _app(),
        "POST",
        "/responses",
        content=_muse_responses_body(),
        headers=_muse_identity_headers(),
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.content == raw_sse
    assert response.content.index(b"response.created") < response.content.index(
        b"response.in_progress"
    )
    assert response.content.index(b"response.completed") < response.content.index(
        b"[DONE]"
    )


def test_muse_responses_stream_read_failure_is_logged_without_asgi_exception(
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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
        _app(),
        "POST",
        "/responses",
        content=_muse_responses_body(),
        headers=_muse_identity_headers(),
    )

    assert response.status_code == 200
    assert b"response.created" in response.content
    error_records = [
        record
        for record in caplog.records
        if record.name == verbose_proxy_logger.name
        and getattr(record, "failure_kind", None)
        == "gateway_upstream_stream_failed"
    ]
    assert len(error_records) == 1
    assert "upstream-stream-secret" not in error_records[0].getMessage()
    assert "meta-oauth-secret-token" not in error_records[0].getMessage()
    assert not error_records[0].exc_info
    rendered_rollup = "\n".join(flush_aawm_route_rollups(force=True))
    assert "upstream-stream-secret" not in rendered_rollup
    assert "meta-oauth-secret-token" not in rendered_rollup


def test_muse_responses_missing_bearer_fails_closed_without_credentials(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.WARNING, logger=verbose_proxy_logger.name)
    response = _request(
        _app(),
        "POST",
        "/responses",
        content=_muse_responses_body(),
        headers={
            "x-client-id": "tbh:exec",
            "user-agent": "muse-build/1.1.1",
            "content-type": "application/json",
        },
    )

    assert response.status_code == 401
    assert "meta-oauth-secret-token" not in response.text
    error_records = [
        record
        for record in caplog.records
        if record.name == verbose_proxy_logger.name
        and "Muse Code gateway surfaced handled client/provider error"
        in record.getMessage()
    ]
    assert len(error_records) == 1
    assert getattr(error_records[0], "route_family") == "muse_code"
    assert getattr(error_records[0], "failure_kind") == (
        "gateway_authentication_rejected"
    )
    combined = "\n".join(record.getMessage() for record in caplog.records)
    assert "meta-oauth-secret-token" not in combined
    assert "Authorization" not in combined


def test_non_muse_responses_is_not_forced_to_meta(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    auth_calls: list[str] = []

    async def _fake_auth(*args: Any, **kwargs: Any) -> None:
        auth_calls.append("user_api_key_auth")
        raise HTTPException(status_code=401, detail="invalid api key")

    monkeypatch.setattr(muse_code_gateway, "user_api_key_auth", _fake_auth)
    requests = _set_upstream(
        monkeypatch,
        lambda request: httpx.Response(200, json={"data": []}, request=request),
    )

    response = _request(
        _app(),
        "POST",
        "/responses",
        content=b'{"model":"gpt-4o","input":"hi"}',
        headers={
            "Authorization": "Bearer sk-litellm-virtual-key",
            "content-type": "application/json",
        },
    )

    assert response.status_code == 401
    assert requests == []
    assert auth_calls == ["user_api_key_auth"]


def test_muse_catalog_success_stamps_route_family_without_authorization(
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    caplog.set_level(logging.WARNING, logger=verbose_proxy_logger.name)
    _set_upstream(
        monkeypatch,
        lambda request: httpx.Response(200, json={"object": "list", "data": []}, request=request),
    )

    response = _request(
        _app(),
        "GET",
        "/muse-code/models",
        client_host="172.18.0.1",
        headers=_bearer_headers(),
    )

    assert response.status_code == 200
    rendered_rollup = "\n".join(flush_aawm_route_rollups(force=True))
    assert "/muse-code/models" in rendered_rollup
    assert "muse_code/catalog" in rendered_rollup
    assert "meta-oauth-secret-token" not in rendered_rollup
    assert "Authorization" not in rendered_rollup
    access_filter = AawmRouteAccessLogReplacementFilter()
    assert (
        access_filter.filter(
            _build_uvicorn_access_record(
                method="GET",
                full_path="/muse-code/models",
                status_code=200,
            )
        )
        is False
    )


def test_muse_responses_failure_stamps_route_family_without_authorization(
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    caplog.set_level(logging.WARNING, logger=verbose_proxy_logger.name)
    upstream_body = (
        b'{"error":{"message":"Authentication Error","type":"authentication_error",'
        b'"code":"invalid_api_key"}}'
    )
    _set_upstream(
        monkeypatch,
        lambda request: httpx.Response(
            401,
            content=upstream_body,
            headers={"content-type": "application/json"},
            request=request,
        ),
    )

    response = _request(
        _app(),
        "POST",
        "/responses",
        content=_muse_responses_body(),
        headers=_muse_identity_headers(),
    )

    assert response.status_code == 401
    assert response.content == upstream_body
    error_records = [
        record
        for record in caplog.records
        if record.name == verbose_proxy_logger.name
        and "Muse Code gateway surfaced handled client/provider error"
        in record.getMessage()
    ]
    assert len(error_records) == 1
    assert getattr(error_records[0], "route_family") == "muse_code"
    assert getattr(error_records[0], "failure_kind") == "gateway_upstream_non_success"
    assert "meta-oauth-secret-token" not in error_records[0].getMessage()
    rendered_rollup = "\n".join(flush_aawm_route_rollups(force=True))
    assert "muse-spark-1.3-contributor" in rendered_rollup
    assert "meta-oauth-secret-token" not in rendered_rollup


def test_muse_responses_rollup_shows_request_reasoning_effort(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _muse_responses_payload(stream=False)
    payload["reasoning"] = {"effort": "high"}
    _set_upstream(
        monkeypatch,
        lambda request: httpx.Response(
            200,
            json={"id": "resp_meta", "model": "muse-spark-1.3-contributor"},
            headers={"content-type": "application/json"},
            request=request,
        ),
    )

    response = _request(
        _app(),
        "POST",
        "/responses",
        content=json.dumps(payload).encode("utf-8"),
        headers=_muse_identity_headers(),
    )

    assert response.status_code == 200
    rendered_rollup = "\n".join(flush_aawm_route_rollups(force=True))
    assert "muse-spark-1.3-contributor:high" in rendered_rollup
    assert "muse-spark-1.3-contributor:none" not in rendered_rollup


def test_muse_responses_rollup_shows_muse_tui_max_as_request_ultra(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _muse_responses_payload(stream=False)
    payload["reasoning"] = {"effort": "ultra"}
    _set_upstream(
        monkeypatch,
        lambda request: httpx.Response(
            200,
            json={"id": "resp_meta", "model": "muse-spark-1.3-contributor"},
            headers={"content-type": "application/json"},
            request=request,
        ),
    )

    response = _request(
        _app(),
        "POST",
        "/responses",
        content=json.dumps(payload).encode("utf-8"),
        headers=_muse_identity_headers(),
    )

    assert response.status_code == 200
    rendered_rollup = "\n".join(flush_aawm_route_rollups(force=True))
    assert "muse-spark-1.3-contributor:ultra" in rendered_rollup
    assert "muse-spark-1.3-contributor:none" not in rendered_rollup


def test_disabled_facade_catalog_handler_is_not_found(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("AAWM_MUSE_CODE_FACADE_ENABLED", raising=False)
    requests = _set_upstream(
        monkeypatch,
        lambda request: httpx.Response(200, json={"data": []}, request=request),
    )

    response = _request(_app(), "GET", "/muse-code/models", headers=_bearer_headers())

    assert response.status_code == 404
    assert response.json() == {"detail": "Not Found"}
    assert requests == []
