"""Alpha-gated Muse Code Meta pass-through.

When ``AAWM_MUSE_CODE_FACADE_ENABLED`` is truthy (``1`` / ``true`` /
``yes`` / ``on``):

- ``GET /muse-code/models`` forwards to ``https://api.meta.ai/muse-code/models``
- Muse-shaped ``POST /responses`` forwards to ``https://api.meta.ai/v1/responses``

The client's Meta OAuth/API ``Authorization: Bearer`` is forwarded. Catalog
and call bodies are not rewritten onto LiteLLM aliases. Observability is a
side channel (``route_family=muse_code``) and never persists credentials.

When the facade is disabled, ``GET /muse-code/models`` is unregistered and
``POST /responses`` stays on the stock LiteLLM handler.
"""

from __future__ import annotations

import json
import os
from collections.abc import AsyncIterator
from typing import Any, Optional

import fastapi
import httpx
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import Response, StreamingResponse

from litellm._logging import _redact_string, verbose_proxy_logger
from litellm.proxy.aawm_route_logging import (
    emit_aawm_route_access_log,
    record_aawm_route_rollup_failure,
    record_aawm_route_rollup_turn,
    register_aawm_route_rollup_access_log_replacement,
)
from litellm.proxy.auth.user_api_key_auth import (
    UserAPIKeyAuth,
    anthropic_api_key_header as _anthropic_api_key_header,
    api_key_header as _api_key_header,
    azure_api_key_header as _azure_api_key_header,
    azure_apim_header as _azure_apim_header,
    custom_litellm_key_header as _custom_litellm_key_header,
    google_ai_studio_api_key_header as _google_ai_studio_api_key_header,
    user_api_key_auth,
)

AAWM_MUSE_CODE_FACADE_ENABLED_ENV = "AAWM_MUSE_CODE_FACADE_ENABLED"
AAWM_MUSE_CODE_MODEL_IDS_ENV = "AAWM_MUSE_CODE_MODEL_IDS"
MUSE_CODE_GATEWAY_PREFIX = "/muse-code"
MUSE_CODE_ROUTE_FAMILY = "muse_code"
MUSE_CODE_CLIENT_ID_PREFIX = "tbh:"
MUSE_CODE_USER_AGENT_PREFIX = "muse-build"
MUSE_CODE_MODEL_ID_PREFIX = "muse-"
MUSE_CODE_CATALOG_UPSTREAM_URL = "https://api.meta.ai/muse-code/models"
MUSE_CODE_RESPONSES_UPSTREAM_URL = "https://api.meta.ai/v1/responses"
MUSE_CODE_GATEWAY_TIMEOUT_SECONDS = 120.0
_MUSE_CODE_GATEWAY_ERROR_SUMMARY_MAX_CHARS = 500
_MUSE_CODE_TRUTHY_VALUES = frozenset({"1", "true", "yes", "on"})
_MUSE_CODE_SESSION_HEADER_NAMES: tuple[str, ...] = (
    "x-meta-ai-gateway-session-id",
    "x-tbh-session-id",
)
_MUSE_CODE_FORWARDED_REQUEST_HEADERS: frozenset[str] = frozenset(
    {
        "authorization",
        "content-type",
        "accept",
        "user-agent",
        "x-client-id",
        "x-tbh-session-id",
        "x-meta-ai-gateway-session-id",
        "traceparent",
        "x-request-id",
        "x-api-version",
    }
)
_MUSE_CODE_RESPONSE_HEADERS: frozenset[str] = frozenset(
    {
        "content-type",
        "retry-after",
        "x-request-id",
        "x-trace-id",
        "x-fb-request-id",
        "x-fb-trace-id",
        "x-route",
    }
)
DEFAULT_MUSE_CODE_MODEL_IDS: tuple[str, ...] = (
    "muse-spark-1.3-contributor",
    "muse-spark-1.3",
    "muse-spark-1.2-contributor",
    "muse-spark-1.2",
)

router = APIRouter(prefix=MUSE_CODE_GATEWAY_PREFIX, tags=["muse code gateway"])


class _GatewayRequestPayload(dict[str, object]):
    pass


class _GatewayRouteKwargs(dict[str, object]):
    pass


def is_muse_code_facade_enabled() -> bool:
    """Return True only when the alpha Muse Code facade env flag is truthy."""

    raw = os.getenv(AAWM_MUSE_CODE_FACADE_ENABLED_ENV)
    if raw is None:
        return False
    return raw.strip().lower() in _MUSE_CODE_TRUTHY_VALUES


def _header_value(request: Request, name: str) -> str:
    value = request.headers.get(name) or request.headers.get(name.lower())
    if not isinstance(value, str):
        return ""
    return value.strip()


def _muse_code_user_agent(request: Request) -> str:
    return _header_value(request, "user-agent")


def _muse_code_client_id(request: Request) -> str:
    return _header_value(request, "x-client-id")


def _muse_code_session_id(request: Request) -> str:
    for name in _MUSE_CODE_SESSION_HEADER_NAMES:
        value = _header_value(request, name)
        if value:
            return value
    return ""


def muse_code_session_id_from_request(request: Request) -> str:
    """Session UUID from Muse gateway headers, never from body ``session_id``."""

    return _muse_code_session_id(request)


def configured_muse_code_model_ids() -> tuple[str, ...]:
    """Hint list of Muse catalog ids. Not used to synthesize a catalog."""

    raw = os.getenv(AAWM_MUSE_CODE_MODEL_IDS_ENV)
    if raw is None or not raw.strip():
        return DEFAULT_MUSE_CODE_MODEL_IDS

    seen: set[str] = set()
    ordered: list[str] = []
    for part in raw.split(","):
        model_id = part.strip()
        if not model_id or model_id in seen:
            continue
        seen.add(model_id)
        ordered.append(model_id)
    if not ordered:
        return DEFAULT_MUSE_CODE_MODEL_IDS
    return tuple(ordered)


def is_known_muse_code_model_id(model_id: Any) -> bool:
    if not isinstance(model_id, str) or not model_id.strip():
        return False
    cleaned = model_id.strip()
    if cleaned in configured_muse_code_model_ids():
        return True
    return cleaned.startswith(MUSE_CODE_MODEL_ID_PREFIX)


def is_muse_code_responses_request(
    request: Request,
    data: Optional[dict[str, Any]] = None,
) -> bool:
    """True when the facade is on and this ``POST /responses`` is Muse-shaped.

    Detection is facade-gated. Non-Muse / facade-off traffic keeps the stock
    Responses handler. Shape is ``x-client-id`` ``tbh:…``, User-Agent
    ``muse-build…``, or a Muse catalog model id.
    """

    if not is_muse_code_facade_enabled():
        return False

    client_id = _muse_code_client_id(request)
    if client_id.startswith(MUSE_CODE_CLIENT_ID_PREFIX):
        return True

    user_agent = _muse_code_user_agent(request)
    if user_agent.lower().startswith(MUSE_CODE_USER_AGENT_PREFIX):
        return True

    model_id = None
    if isinstance(data, dict):
        model_id = data.get("model")
    return is_known_muse_code_model_id(model_id)


def _get_upstream_transport() -> Optional[httpx.AsyncBaseTransport]:
    """Test seam. Production requests use the default HTTPX transport."""

    return None


def _extract_inbound_bearer(request: Request) -> Optional[str]:
    authorization = request.headers.get("authorization") or request.headers.get(
        "Authorization"
    )
    if not isinstance(authorization, str):
        return None
    scheme, separator, bearer = authorization.partition(" ")
    if (
        scheme != "Bearer"
        or not separator
        or not bearer
        or bearer != bearer.strip()
        or " " in bearer
    ):
        return None
    return bearer


def _require_inbound_bearer(request: Request) -> str:
    bearer = _extract_inbound_bearer(request)
    if bearer is None:
        raise HTTPException(
            status_code=401,
            detail="Muse Code gateway authorization is invalid.",
        )
    return bearer


def _get_upstream_headers(request: Request) -> dict[str, str]:
    headers: dict[str, str] = {}
    for name, value in request.headers.items():
        lowered = name.lower()
        if lowered in _MUSE_CODE_FORWARDED_REQUEST_HEADERS and value:
            headers[lowered] = value
    return headers


def _get_response_headers(headers: httpx.Headers) -> dict[str, str]:
    return {
        name: value
        for name, value in headers.items()
        if name.lower() in _MUSE_CODE_RESPONSE_HEADERS
        or name.lower().startswith("x-ratelimit-")
    }


def _request_requires_streaming_response(request_body: bytes) -> bool:
    try:
        body = json.loads(request_body)
    except (TypeError, ValueError):
        return False
    return isinstance(body, dict) and body.get("stream") is True


def _get_gateway_route_model(request_body: bytes, *, catalog: bool) -> str:
    if catalog:
        return "muse_code/catalog"
    try:
        body = json.loads(request_body)
    except (TypeError, ValueError):
        return "muse_code/unknown"
    model = body.get("model") if isinstance(body, dict) else None
    if not isinstance(model, str) or not model.strip():
        return "muse_code/unknown"
    return " ".join(model.split())


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
    if len(redacted_summary) > _MUSE_CODE_GATEWAY_ERROR_SUMMARY_MAX_CHARS:
        return (
            redacted_summary[: _MUSE_CODE_GATEWAY_ERROR_SUMMARY_MAX_CHARS - 3]
            + "..."
        )
    return redacted_summary


def _build_gateway_route_state(
    *,
    request_body: bytes,
    catalog: bool,
    session_id: str,
) -> tuple[_GatewayRequestPayload, _GatewayRouteKwargs]:
    model = _get_gateway_route_model(request_body, catalog=catalog)
    request_payload = _GatewayRequestPayload(model=model)
    metadata: dict[str, object] = {
        "custom_llm_provider": "muse_code",
        "route_family": MUSE_CODE_ROUTE_FAMILY,
        "passthrough_route_family": MUSE_CODE_ROUTE_FAMILY,
        "tags": [
            f"route:{MUSE_CODE_ROUTE_FAMILY}",
            "muse-code",
        ],
    }
    if session_id:
        metadata["session_id"] = session_id
        metadata["muse_code_session_id"] = session_id
    kwargs = _GatewayRouteKwargs(litellm_params={"metadata": metadata})
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
        route_type="MUSE",
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
    session_id: str = "",
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
    extra = {
        "source": "muse_code_gateway",
        "container": os.getenv("HOSTNAME"),
        "endpoint": request.url.path,
        "upstream_url": target,
        "provider": "muse_code",
        "model": request_payload["model"],
        "model_alias": None,
        "route_family": MUSE_CODE_ROUTE_FAMILY,
        "status_code": status_code,
        "trace_id": trace_id,
        "litellm_call_id": request.headers.get("x-litellm-call-id"),
        "failure_kind": failure_kind,
    }
    if session_id:
        extra["session_id"] = session_id
    log_fn(
        "Muse Code gateway surfaced handled client/provider error status=%s error=%s",
        status_code,
        summary,
        extra=extra,
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
    session_id: str,
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
            detail="Muse Code gateway upstream response stream failed.",
            failure_kind="gateway_upstream_stream_failed",
            session_id=session_id,
            trace_id=response.headers.get("x-trace-id")
            or response.headers.get("x-request-id")
            or response.headers.get("x-fb-trace-id"),
        )
    else:
        record_aawm_route_rollup_turn(route_kwargs)
    finally:
        await response.aclose()
        await client.aclose()


async def _proxy_muse_code_request(
    request: Request,
    *,
    upstream_url: str,
    catalog: bool,
) -> Response:
    register_aawm_route_rollup_access_log_replacement(request)
    request_body = await request.body()
    session_id = _muse_code_session_id(request)
    request_payload, route_kwargs = _build_gateway_route_state(
        request_body=request_body,
        catalog=catalog,
        session_id=session_id,
    )

    try:
        _require_inbound_bearer(request)
    except HTTPException as exc:
        _log_gateway_failure(
            request=request,
            target=upstream_url,
            request_payload=request_payload,
            kwargs=route_kwargs,
            status_code=exc.status_code,
            detail=exc.detail,
            failure_kind="gateway_authentication_rejected",
            session_id=session_id,
        )
        raise

    _emit_gateway_route_context(
        request=request,
        target=upstream_url,
        request_payload=request_payload,
        kwargs=route_kwargs,
    )
    client = httpx.AsyncClient(
        timeout=MUSE_CODE_GATEWAY_TIMEOUT_SECONDS,
        transport=_get_upstream_transport(),
    )
    try:
        upstream_request = client.build_request(
            method=request.method,
            url=upstream_url,
            headers=_get_upstream_headers(request),
            content=request_body,
        )
        upstream_response = await client.send(upstream_request, stream=True)
    except httpx.HTTPError as exc:
        await client.aclose()
        detail = "Muse Code gateway upstream request failed."
        _log_gateway_failure(
            request=request,
            target=upstream_url,
            request_payload=request_payload,
            kwargs=route_kwargs,
            status_code=502,
            detail=detail,
            failure_kind="gateway_upstream_request_failed",
            session_id=session_id,
        )
        raise HTTPException(
            status_code=502,
            detail=detail,
        ) from exc

    response_headers = _get_response_headers(upstream_response.headers)
    is_sse_response = "text/event-stream" in upstream_response.headers.get(
        "content-type", ""
    ).lower()
    if 200 <= upstream_response.status_code < 300 and (
        is_sse_response or _request_requires_streaming_response(request_body)
    ):
        return StreamingResponse(
            _stream_response(
                upstream_response,
                client,
                request=request,
                target=upstream_url,
                request_payload=request_payload,
                route_kwargs=route_kwargs,
                session_id=session_id,
            ),
            status_code=upstream_response.status_code,
            headers=response_headers,
            media_type=None,
        )

    try:
        response_body = await upstream_response.aread()
    except httpx.HTTPError as exc:
        detail = "Muse Code gateway upstream response read failed."
        _log_gateway_failure(
            request=request,
            target=upstream_url,
            request_payload=request_payload,
            kwargs=route_kwargs,
            status_code=502,
            detail=detail,
            failure_kind="gateway_upstream_response_read_failed",
            session_id=session_id,
            trace_id=upstream_response.headers.get("x-trace-id")
            or upstream_response.headers.get("x-request-id")
            or upstream_response.headers.get("x-fb-trace-id"),
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
            session_id=session_id,
            trace_id=upstream_response.headers.get("x-trace-id")
            or upstream_response.headers.get("x-request-id")
            or upstream_response.headers.get("x-fb-trace-id"),
        )
    else:
        record_aawm_route_rollup_turn(route_kwargs)
    return Response(
        content=response_body,
        status_code=upstream_response.status_code,
        headers=response_headers,
        media_type=None,
    )


async def maybe_proxy_muse_code_responses(request: Request) -> Optional[Response]:
    """Forward Muse-shaped ``POST /responses`` to Meta, or return None.

    Callers must invoke this *before* LiteLLM virtual-key auth so Muse's Meta
    Bearer is not treated as a LiteLLM key.
    """

    if not is_muse_code_facade_enabled():
        return None

    cached_body: Optional[dict[str, Any]] = None
    try:
        from litellm.proxy.common_utils.http_parsing_utils import (
            _safe_get_request_parsed_body,
        )

        parsed = _safe_get_request_parsed_body(request=request)
        if isinstance(parsed, dict):
            cached_body = parsed
    except Exception:
        cached_body = None

    if cached_body is None:
        raw_body = await request.body()
        if raw_body:
            try:
                parsed_body = json.loads(raw_body)
            except (TypeError, ValueError):
                parsed_body = None
            cached_body = parsed_body if isinstance(parsed_body, dict) else None
        else:
            cached_body = {}

    if not is_muse_code_responses_request(request, cached_body):
        return None

    return await _proxy_muse_code_request(
        request,
        upstream_url=MUSE_CODE_RESPONSES_UPSTREAM_URL,
        catalog=False,
    )


async def muse_code_or_user_api_key_auth(
    request: Request,
    api_key: str = fastapi.Security(_api_key_header),
    azure_api_key_header: str = fastapi.Security(_azure_api_key_header),
    anthropic_api_key_header: Optional[str] = fastapi.Security(
        _anthropic_api_key_header
    ),
    google_ai_studio_api_key_header: Optional[str] = fastapi.Security(
        _google_ai_studio_api_key_header
    ),
    azure_apim_header: Optional[str] = fastapi.Security(_azure_apim_header),
    custom_litellm_key_header: Optional[str] = fastapi.Security(
        _custom_litellm_key_header
    ),
) -> UserAPIKeyAuth:
    """Skip LiteLLM virtual-key auth for Muse-shaped Responses; otherwise auth."""

    muse_response = await maybe_proxy_muse_code_responses(request)
    if muse_response is not None:
        request.scope["muse_code_passthrough_response"] = muse_response
        return UserAPIKeyAuth()
    return await user_api_key_auth(
        request=request,
        api_key=api_key,
        azure_api_key_header=azure_api_key_header,
        anthropic_api_key_header=anthropic_api_key_header,
        google_ai_studio_api_key_header=google_ai_studio_api_key_header,
        azure_apim_header=azure_apim_header,
        custom_litellm_key_header=custom_litellm_key_header,
    )


@router.get("/models")
async def get_muse_code_models(request: Request) -> Response:
    if not is_muse_code_facade_enabled():
        raise HTTPException(status_code=404, detail="Not Found")
    return await _proxy_muse_code_request(
        request,
        upstream_url=MUSE_CODE_CATALOG_UPSTREAM_URL,
        catalog=True,
    )
