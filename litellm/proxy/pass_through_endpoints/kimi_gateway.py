"""Trusted-local raw gateway for the Kimi Code CLI."""

from __future__ import annotations

import hmac
import json
import os
from collections.abc import AsyncIterator
from typing import Optional
from urllib.parse import SplitResult, urlsplit, urlunsplit

import httpx
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import Response, StreamingResponse

from litellm.llms.kimi_code.chat.transformation import (
    KIMI_CODE_API_BASE,
    KimiCodeAuthenticationError,
    KimiCodeChatConfig,
)


KIMI_CODE_BASE_URL_ENV = "KIMI_CODE_BASE_URL"
KIMI_CODE_GATEWAY_PREFIX = "/kimi/v1"
KIMI_CODE_GATEWAY_TIMEOUT_SECONDS = 120.0

_KIMI_CODE_GATEWAY_UPSTREAM_SUFFIXES = {
    "models": "/models",
    "usages": "/usages",
    "chat_completions": "/chat/completions",
}
_KIMI_CODE_GATEWAY_REQUEST_HOP_BY_HOP_HEADERS = frozenset(
    {
        "connection",
        "content-length",
        "host",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailer",
        "transfer-encoding",
        "upgrade",
    }
)
_KIMI_CODE_GATEWAY_REQUEST_STRIPPED_HEADERS = frozenset(
    {
        "authorization",
        "forwarded",
        "user-agent",
    }
)
_KIMI_CODE_GATEWAY_RESPONSE_HEADERS = frozenset(
    {
        "content-type",
        "retry-after",
        "x-request-id",
        "x-trace-id",
    }
)

router = APIRouter(prefix=KIMI_CODE_GATEWAY_PREFIX, tags=["kimi code gateway"])


def _require_current_bearer(request: Request, access_token: str) -> None:
    """Require the current Kimi Code credential as an exactly formatted bearer."""

    authorization = request.headers.get("authorization")
    scheme, separator, bearer = authorization.partition(" ") if authorization is not None else ("", "", "")
    if (
        scheme != "Bearer"
        or not separator
        or not bearer
        or bearer != bearer.strip()
        or " " in bearer
        or not hmac.compare_digest(bearer.encode("utf-8"), access_token.encode("utf-8"))
    ):
        raise HTTPException(
            status_code=401,
            detail="Kimi Code gateway authorization is invalid.",
        )


def _get_kimi_code_base_url() -> str:
    configured_base_url = os.getenv(KIMI_CODE_BASE_URL_ENV, KIMI_CODE_API_BASE)
    try:
        parsed_base_url = urlsplit(configured_base_url)
    except (TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=503,
            detail="Kimi Code gateway upstream is unavailable.",
        ) from exc

    if (
        parsed_base_url.scheme != "https"
        or not parsed_base_url.netloc
        or parsed_base_url.username is not None
        or parsed_base_url.password is not None
        or parsed_base_url.query
        or parsed_base_url.fragment
    ):
        raise HTTPException(
            status_code=503,
            detail="Kimi Code gateway upstream is unavailable.",
        )

    return urlunsplit(
        SplitResult(
            scheme=parsed_base_url.scheme,
            netloc=parsed_base_url.netloc,
            path=parsed_base_url.path.rstrip("/"),
            query="",
            fragment="",
        )
    )


def _get_upstream_url(endpoint: str) -> str:
    return "{}{}".format(
        _get_kimi_code_base_url(),
        _KIMI_CODE_GATEWAY_UPSTREAM_SUFFIXES[endpoint],
    )


def _get_upstream_transport() -> Optional[httpx.AsyncBaseTransport]:
    """Test seam. Production requests use the default HTTPX transport."""

    return None


def _get_upstream_headers(request: Request, access_token: str) -> dict[str, str]:
    headers = {
        name: value
        for name, value in request.headers.items()
        if name.lower() not in _KIMI_CODE_GATEWAY_REQUEST_HOP_BY_HOP_HEADERS
        and name.lower() not in _KIMI_CODE_GATEWAY_REQUEST_STRIPPED_HEADERS
        and not name.lower().startswith("x-forwarded-")
        and not name.lower().startswith("x-msh-")
    }
    headers["Authorization"] = f"Bearer {access_token}"
    headers["User-Agent"] = KimiCodeChatConfig._user_agent()
    return headers


def _get_response_headers(headers: httpx.Headers) -> dict[str, str]:
    return {
        name: value
        for name, value in headers.items()
        if name.lower() in _KIMI_CODE_GATEWAY_RESPONSE_HEADERS or name.lower().startswith("x-ratelimit-")
    }


def _request_requires_streaming_response(request_body: bytes) -> bool:
    try:
        body = json.loads(request_body)
    except (TypeError, ValueError):
        return False
    return isinstance(body, dict) and body.get("stream") is True


async def _stream_response(response: httpx.Response, client: httpx.AsyncClient) -> AsyncIterator[bytes]:
    try:
        async for chunk in response.aiter_raw():
            yield chunk
    finally:
        await response.aclose()
        await client.aclose()


async def _proxy_kimi_code_request(request: Request, endpoint: str) -> Response:
    try:
        access_token = KimiCodeChatConfig._get_access_token()
    except KimiCodeAuthenticationError as exc:
        raise HTTPException(status_code=401, detail=exc.message) from exc

    _require_current_bearer(request, access_token)

    request_body = await request.body()
    client = httpx.AsyncClient(
        timeout=KIMI_CODE_GATEWAY_TIMEOUT_SECONDS,
        transport=_get_upstream_transport(),
    )
    try:
        upstream_request = client.build_request(
            method=request.method,
            url=_get_upstream_url(endpoint),
            headers=_get_upstream_headers(request, access_token),
            content=request_body,
        )
        upstream_response = await client.send(upstream_request, stream=True)
    except httpx.HTTPError as exc:
        await client.aclose()
        raise HTTPException(
            status_code=502,
            detail="Kimi Code gateway upstream request failed.",
        ) from exc

    response_headers = _get_response_headers(upstream_response.headers)
    is_sse_response = "text/event-stream" in upstream_response.headers.get("content-type", "").lower()
    if is_sse_response or _request_requires_streaming_response(request_body):
        return StreamingResponse(
            _stream_response(upstream_response, client),
            status_code=upstream_response.status_code,
            headers=response_headers,
            media_type=None,
        )

    try:
        response_body = await upstream_response.aread()
    finally:
        await upstream_response.aclose()
        await client.aclose()
    return Response(
        content=response_body,
        status_code=upstream_response.status_code,
        headers=response_headers,
        media_type=None,
    )


@router.get("/models")
async def get_kimi_code_models(request: Request) -> Response:
    return await _proxy_kimi_code_request(request, "models")


@router.get("/usages")
async def get_kimi_code_usages(request: Request) -> Response:
    return await _proxy_kimi_code_request(request, "usages")


@router.post("/chat/completions")
async def post_kimi_code_chat_completions(request: Request) -> Response:
    return await _proxy_kimi_code_request(request, "chat_completions")
