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

from litellm._logging import _redact_string, verbose_proxy_logger
from litellm.llms.kimi_code.chat.transformation import (
    KIMI_CODE_API_BASE,
    KimiCodeAuthenticationError,
    KimiCodeChatConfig,
)
from litellm.proxy.aawm_route_logging import (
    emit_aawm_route_access_log,
    record_aawm_route_rollup_failure,
    record_aawm_route_rollup_turn,
    register_aawm_route_rollup_access_log_replacement,
)


KIMI_CODE_BASE_URL_ENV = "KIMI_CODE_BASE_URL"
KIMI_CODE_GATEWAY_PREFIX = "/kimi/v1"
KIMI_CODE_GATEWAY_TIMEOUT_SECONDS = 120.0
_KIMI_CODE_GATEWAY_ERROR_SUMMARY_MAX_CHARS = 500
_KIMI_CODE_GATEWAY_MANAGED_ACCOUNT_MODEL = "kimi_code/__managed_account__"


class _GatewayRequestPayload(dict[str, object]):
    pass


class _GatewayRouteKwargs(dict[str, object]):
    pass


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


def _get_gateway_route_model(endpoint: str, request_body: bytes) -> str:
    if endpoint != "chat_completions":
        return _KIMI_CODE_GATEWAY_MANAGED_ACCOUNT_MODEL
    try:
        body = json.loads(request_body)
    except (TypeError, ValueError):
        return "kimi_code/unknown"
    model = body.get("model") if isinstance(body, dict) else None
    if not isinstance(model, str) or not model.strip():
        return "kimi_code/unknown"
    normalized_model = " ".join(model.split())
    if "/" in normalized_model:
        return normalized_model
    return f"kimi_code/{normalized_model}"


def _get_gateway_logging_target(endpoint: str) -> str:
    return "{}{}".format(
        KIMI_CODE_API_BASE,
        _KIMI_CODE_GATEWAY_UPSTREAM_SUFFIXES[endpoint],
    )


def _extract_gateway_error_value(payload: object) -> Optional[str]:
    if isinstance(payload, str):
        return payload
    if not isinstance(payload, dict):
        return None
    error = payload.get("error")
    if isinstance(error, str):
        return error
    if isinstance(error, dict):
        for key in ("message", "detail", "error"):
            value = _extract_gateway_error_value(error.get(key))
            if value:
                return value
    for key in ("detail", "message"):
        value = _extract_gateway_error_value(payload.get(key))
        if value:
            return value
    return None


def _sanitize_gateway_error_summary(
    detail: object,
    *,
    status_code: int,
) -> str:
    if isinstance(detail, bytes):
        detail_text = detail.decode("utf-8", errors="replace")
    elif isinstance(detail, str):
        detail_text = detail
    else:
        detail_text = ""

    parsed_detail: object = detail
    if detail_text:
        try:
            parsed_detail = json.loads(detail_text)
        except (TypeError, ValueError):
            parsed_detail = detail_text

    summary = _extract_gateway_error_value(parsed_detail)
    if summary is None and detail_text:
        summary = detail_text
    normalized_summary = " ".join(str(summary or "").split())
    redacted_summary = _redact_string(normalized_summary)
    if not redacted_summary:
        return f"HTTP {status_code} request rejected"
    if len(redacted_summary) > _KIMI_CODE_GATEWAY_ERROR_SUMMARY_MAX_CHARS:
        return (
            redacted_summary[: _KIMI_CODE_GATEWAY_ERROR_SUMMARY_MAX_CHARS - 3]
            + "..."
        )
    return redacted_summary


def _build_gateway_route_state(
    *,
    request: Request,
    endpoint: str,
    request_body: bytes,
) -> tuple[_GatewayRequestPayload, _GatewayRouteKwargs]:
    model = _get_gateway_route_model(endpoint, request_body)
    request_payload = _GatewayRequestPayload(model=model)
    kwargs = _GatewayRouteKwargs(
        litellm_params={
            "metadata": {
                "custom_llm_provider": "kimi_code",
                "route_family": "kimi_code_gateway",
            }
        }
    )
    return request_payload, kwargs


def _emit_gateway_route_context(
    *,
    request: Request,
    target: str,
    request_payload: _GatewayRequestPayload,
    kwargs: _GatewayRouteKwargs,
) -> None:
    emit_aawm_route_access_log(
        request=request,
        target=target,
        request_body=request_payload,
        kwargs=kwargs,
        route_type="KIMI",
    )


def _log_gateway_failure(
    *,
    request: Request,
    target: str,
    request_payload: _GatewayRequestPayload,
    kwargs: _GatewayRouteKwargs,
    status_code: int,
    detail: object,
    failure_kind: str,
    trace_id: Optional[str] = None,
) -> str:
    _emit_gateway_route_context(
        request=request,
        target=target,
        request_payload=request_payload,
        kwargs=kwargs,
    )
    summary = _sanitize_gateway_error_summary(
        detail,
        status_code=status_code,
    )
    record_aawm_route_rollup_failure(
        kwargs,
        message=summary,
    )
    log_fn = (
        verbose_proxy_logger.warning
        if status_code < 500
        else verbose_proxy_logger.error
    )
    log_fn(
        "Kimi Code gateway surfaced handled client/provider error status=%s error=%s",
        status_code,
        summary,
        extra={
            "source": "kimi_code_gateway",
            "container": os.getenv("HOSTNAME"),
            "endpoint": request.url.path,
            "upstream_url": target,
            "provider": "kimi_code",
            "model": request_payload["model"],
            "model_alias": None,
            "route_family": "kimi_code_gateway",
            "status_code": status_code,
            "trace_id": trace_id,
            "litellm_call_id": request.headers.get("x-litellm-call-id"),
            "failure_kind": failure_kind,
        },
        exc_info=False,
    )
    return summary


async def _stream_response(
    response: httpx.Response,
    client: httpx.AsyncClient,
    *,
    request: Request,
    target: str,
    request_payload: _GatewayRequestPayload,
    route_kwargs: _GatewayRouteKwargs,
) -> AsyncIterator[bytes]:
    try:
        async for chunk in response.aiter_raw():
            yield chunk
    except httpx.HTTPError:
        _log_gateway_failure(
            request=request,
            target=target,
            request_payload=request_payload,
            kwargs=route_kwargs,
            status_code=502,
            detail="Kimi Code gateway upstream response stream failed.",
            failure_kind="gateway_upstream_stream_failed",
            trace_id=response.headers.get("x-trace-id")
            or response.headers.get("x-request-id"),
        )
    else:
        record_aawm_route_rollup_turn(route_kwargs)
    finally:
        await response.aclose()
        await client.aclose()


async def _proxy_kimi_code_request(request: Request, endpoint: str) -> Response:
    register_aawm_route_rollup_access_log_replacement(request)
    request_body = await request.body()
    request_payload, route_kwargs = _build_gateway_route_state(
        request=request,
        endpoint=endpoint,
        request_body=request_body,
    )
    logging_target = _get_gateway_logging_target(endpoint)

    try:
        access_token = KimiCodeChatConfig._get_access_token()
    except KimiCodeAuthenticationError as exc:
        _log_gateway_failure(
            request=request,
            target=logging_target,
            request_payload=request_payload,
            kwargs=route_kwargs,
            status_code=401,
            detail=exc.message,
            failure_kind="gateway_authentication_unavailable",
        )
        raise HTTPException(status_code=401, detail=exc.message) from exc

    try:
        _require_current_bearer(request, access_token)
    except HTTPException as exc:
        _log_gateway_failure(
            request=request,
            target=logging_target,
            request_payload=request_payload,
            kwargs=route_kwargs,
            status_code=exc.status_code,
            detail=exc.detail,
            failure_kind="gateway_authentication_rejected",
        )
        raise

    try:
        upstream_url = _get_upstream_url(endpoint)
    except HTTPException as exc:
        _log_gateway_failure(
            request=request,
            target=logging_target,
            request_payload=request_payload,
            kwargs=route_kwargs,
            status_code=exc.status_code,
            detail=exc.detail,
            failure_kind="gateway_configuration_unavailable",
        )
        raise

    _emit_gateway_route_context(
        request=request,
        target=upstream_url,
        request_payload=request_payload,
        kwargs=route_kwargs,
    )
    client = httpx.AsyncClient(
        timeout=KIMI_CODE_GATEWAY_TIMEOUT_SECONDS,
        transport=_get_upstream_transport(),
    )
    try:
        upstream_request = client.build_request(
            method=request.method,
            url=upstream_url,
            headers=_get_upstream_headers(request, access_token),
            content=request_body,
        )
        upstream_response = await client.send(upstream_request, stream=True)
    except httpx.HTTPError as exc:
        await client.aclose()
        detail = "Kimi Code gateway upstream request failed."
        _log_gateway_failure(
            request=request,
            target=upstream_url,
            request_payload=request_payload,
            kwargs=route_kwargs,
            status_code=502,
            detail=detail,
            failure_kind="gateway_upstream_request_failed",
        )
        raise HTTPException(
            status_code=502,
            detail=detail,
        ) from exc

    response_headers = _get_response_headers(upstream_response.headers)
    is_sse_response = "text/event-stream" in upstream_response.headers.get("content-type", "").lower()
    if (
        200 <= upstream_response.status_code < 300
        and (is_sse_response or _request_requires_streaming_response(request_body))
    ):
        return StreamingResponse(
            _stream_response(
                upstream_response,
                client,
                request=request,
                target=upstream_url,
                request_payload=request_payload,
                route_kwargs=route_kwargs,
            ),
            status_code=upstream_response.status_code,
            headers=response_headers,
            media_type=None,
        )

    try:
        response_body = await upstream_response.aread()
    except httpx.HTTPError as exc:
        detail = "Kimi Code gateway upstream response read failed."
        _log_gateway_failure(
            request=request,
            target=upstream_url,
            request_payload=request_payload,
            kwargs=route_kwargs,
            status_code=502,
            detail=detail,
            failure_kind="gateway_upstream_response_read_failed",
            trace_id=upstream_response.headers.get("x-trace-id")
            or upstream_response.headers.get("x-request-id"),
        )
        raise HTTPException(status_code=502, detail=detail) from exc
    finally:
        await upstream_response.aclose()
        await client.aclose()

    if not 200 <= upstream_response.status_code < 300:
        _log_gateway_failure(
            request=request,
            target=upstream_url,
            request_payload=request_payload,
            kwargs=route_kwargs,
            status_code=upstream_response.status_code,
            detail=response_body,
            failure_kind="gateway_upstream_non_success",
            trace_id=upstream_response.headers.get("x-trace-id")
            or upstream_response.headers.get("x-request-id"),
        )
    else:
        record_aawm_route_rollup_turn(route_kwargs)
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
