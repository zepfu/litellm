"""Alpha-gated Muse Code catalog facade.

Serves ``GET /muse-code/models`` in the schema Muse probes at
``{base-url}/muse-code/models``. The router is inert unless
``AAWM_MUSE_CODE_FACADE_ENABLED`` is truthy (``1`` / ``true`` / ``yes`` / ``on``).
When disabled, the path 404s like an unregistered route.

Inbound auth is LiteLLM virtual-key ``user_api_key_auth``. Prefer
``Authorization`` bearer; also accept ``x-litellm-api-key`` (dual-key facades).
Do not hmac-compare to a Meta token.
"""

from __future__ import annotations

import copy
import os
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse

from litellm._logging import _redact_string, verbose_proxy_logger
from litellm.proxy._types import ProxyException, UserAPIKeyAuth
from litellm.proxy.aawm_route_logging import (
    emit_aawm_route_access_log,
    record_aawm_route_rollup_failure,
    record_aawm_route_rollup_turn,
    register_aawm_route_rollup_access_log_replacement,
)
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth

AAWM_MUSE_CODE_FACADE_ENABLED_ENV = "AAWM_MUSE_CODE_FACADE_ENABLED"
AAWM_MUSE_CODE_MODEL_IDS_ENV = "AAWM_MUSE_CODE_MODEL_IDS"
MUSE_CODE_GATEWAY_PREFIX = "/muse-code"
MUSE_CODE_ROUTE_FAMILY = "muse_code"
MUSE_CODE_CATALOG_TARGET = "/muse-code/models"
_MUSE_CODE_GATEWAY_ERROR_SUMMARY_MAX_CHARS = 500
_MUSE_CODE_TRUTHY_VALUES = frozenset({"1", "true", "yes", "on"})

DEFAULT_MUSE_CODE_MODEL_IDS: tuple[str, ...] = (
    "muse-spark-1.3-contributor",
    "muse-spark-1.3",
    "muse-spark-1.2-contributor",
    "muse-spark-1.2",
)

# Identity mapping for later call routing. Unknown Muse ids stay themselves;
# they are never coerced onto basic/work/OpenAI aliases.
MUSE_CODE_LITELLM_ALIASES: dict[str, str] = {
    model_id: model_id for model_id in DEFAULT_MUSE_CODE_MODEL_IDS
}

# Sanitized MUSE-001 capture of GET /muse-code/models capability metadata.
_MUSE_CODE_CATALOG_BY_ID: dict[str, dict[str, Any]] = {
    "muse-spark-1.3-contributor": {
        "id": "muse-spark-1.3-contributor",
        "object": "model",
        "created": 1788320726,
        "owned_by": "meta",
        "metadata": {
            "muse-code": {
                "name": "muse-spark-1.3-contributor",
                "family": "avocado",
                "release_date": "2026-09-02",
                "is_hidden": False,
                "attachment": True,
                "reasoning": True,
                "temperature": False,
                "tool_call": True,
                "modalities": {
                    "input": ["text", "image"],
                    "output": ["text"],
                },
                "limit": {"context": 1007997, "output": 128000},
                "options": {
                    "reasoningEffort": "high",
                    "forceReasoning": True,
                    "include": ["reasoning.encrypted_content"],
                    "temperature": 0.9,
                    "top_p": 0.9,
                },
                "description": (
                    "Your content, including inter-session messages, may be used for product improvement."
                ),
                "variants": {
                    "minimal": {"reasoningEffort": "minimal"},
                    "low": {"reasoningEffort": "low"},
                    "medium": {"reasoningEffort": "medium"},
                    "high": {"reasoningEffort": "high"},
                    "xhigh": {
                        "reasoningEffort": "xhigh",
                        "selector": {
                            "description": "Use this for deepest analysis and complex fixes."
                        },
                    },
                    "max": {"reasoningEffort": "max"},
                },
            }
        },
    },
    "muse-spark-1.3": {
        "id": "muse-spark-1.3",
        "object": "model",
        "created": 1787927249,
        "owned_by": "meta",
        "metadata": {
            "muse-code": {
                "name": "muse-spark-1.3",
                "family": "avocado",
                "release_date": "2026-09-02",
                "is_hidden": False,
                "attachment": True,
                "reasoning": True,
                "temperature": False,
                "tool_call": True,
                "modalities": {
                    "input": ["text", "image"],
                    "output": ["text"],
                },
                "limit": {"context": 1007997, "output": 128000},
                "options": {
                    "reasoningEffort": "high",
                    "forceReasoning": True,
                    "include": ["reasoning.encrypted_content"],
                    "temperature": 0.9,
                    "top_p": 0.9,
                },
                "variants": {
                    "minimal": {"reasoningEffort": "minimal"},
                    "low": {"reasoningEffort": "low"},
                    "medium": {"reasoningEffort": "medium"},
                    "high": {"reasoningEffort": "high"},
                    "xhigh": {"reasoningEffort": "xhigh"},
                    "max": {"reasoningEffort": "max"},
                },
                "description": None,
            }
        },
    },
    "muse-spark-1.2-contributor": {
        "id": "muse-spark-1.2-contributor",
        "object": "model",
        "created": 1785432904,
        "owned_by": "meta",
        "metadata": {
            "muse-code": {
                "name": "muse-spark-1.2-contributor",
                "family": "avocado",
                "release_date": "2026-08-05",
                "is_hidden": False,
                "attachment": True,
                "reasoning": True,
                "temperature": False,
                "tool_call": True,
                "modalities": {
                    "input": ["text", "image"],
                    "output": ["text"],
                },
                "limit": {"context": 1007997, "output": 128000},
                "options": {
                    "reasoningEffort": "high",
                    "forceReasoning": True,
                    "include": ["reasoning.encrypted_content"],
                    "temperature": 0.9,
                    "top_p": 0.9,
                },
                "description": (
                    "Your content, including inter-session messages, may be used for product improvement."
                ),
                "variants": {
                    "minimal": {"reasoningEffort": "minimal"},
                    "low": {"reasoningEffort": "low"},
                    "medium": {"reasoningEffort": "medium"},
                    "high": {"reasoningEffort": "high"},
                    "xhigh": {
                        "reasoningEffort": "xhigh",
                        "selector": {
                            "description": "Use this for deepest analysis and complex fixes."
                        },
                    },
                },
            }
        },
    },
    "muse-spark-1.2": {
        "id": "muse-spark-1.2",
        "object": "model",
        "created": 1785432700,
        "owned_by": "meta",
        "metadata": {
            "muse-code": {
                "name": "muse-spark-1.2",
                "family": "avocado",
                "release_date": "2026-08-05",
                "is_hidden": False,
                "attachment": True,
                "reasoning": True,
                "temperature": False,
                "tool_call": True,
                "modalities": {
                    "input": ["text", "image"],
                    "output": ["text"],
                },
                "limit": {"context": 1007997, "output": 128000},
                "options": {
                    "reasoningEffort": "high",
                    "forceReasoning": True,
                    "include": ["reasoning.encrypted_content"],
                    "temperature": 0.9,
                    "top_p": 0.9,
                },
                "variants": {
                    "minimal": {"reasoningEffort": "minimal"},
                    "low": {"reasoningEffort": "low"},
                    "medium": {"reasoningEffort": "medium"},
                    "high": {"reasoningEffort": "high"},
                    "xhigh": {
                        "reasoningEffort": "xhigh",
                        "selector": {
                            "description": "Use this for deepest analysis and complex fixes."
                        },
                    },
                },
                "description": None,
            }
        },
    },
}

router = APIRouter(prefix=MUSE_CODE_GATEWAY_PREFIX, tags=["muse code gateway"])


def is_muse_code_facade_enabled() -> bool:
    """Return True only when the alpha Muse Code facade env flag is truthy."""

    raw = os.getenv(AAWM_MUSE_CODE_FACADE_ENABLED_ENV)
    if raw is None:
        return False
    return raw.strip().lower() in _MUSE_CODE_TRUTHY_VALUES


def _muse_code_virtual_key(request: Request) -> str:
    """LiteLLM virtual key from Authorization or x-litellm-api-key.

    Dual-key facades send the virtual key in ``x-litellm-api-key``; Muse's
    native probe uses ``Authorization``. Prefer the explicit LiteLLM header
    when both are present. Do not treat either value as a Meta token.
    """

    header_key = request.headers.get("x-litellm-api-key")
    authorization = request.headers.get("Authorization") or request.headers.get(
        "authorization"
    )
    raw = header_key or authorization or ""
    cleaned = raw.strip() if isinstance(raw, str) else ""
    if not cleaned:
        return ""
    if cleaned.lower().startswith("bearer "):
        return cleaned
    return f"Bearer {cleaned}"


def resolve_muse_code_litellm_alias(muse_model_id: str) -> str:
    """Return the LiteLLM alias for a Muse catalog id.

    Captured spark ids map to themselves. Unknown ids are not rewritten onto
    basic/work/OpenAI aliases.
    """

    return MUSE_CODE_LITELLM_ALIASES.get(muse_model_id, muse_model_id)


def _configured_muse_code_model_ids() -> tuple[str, ...]:
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


def _catalog_entry_for_id(model_id: str) -> dict[str, Any]:
    captured = _MUSE_CODE_CATALOG_BY_ID.get(model_id)
    if captured is not None:
        return copy.deepcopy(captured)
    return {
        "id": model_id,
        "object": "model",
        "created": 0,
        "owned_by": "meta",
        "metadata": {
            "muse-code": {
                "name": model_id,
            }
        },
    }


def _build_muse_code_catalog() -> dict[str, Any]:
    return {
        "object": "list",
        "data": [
            _catalog_entry_for_id(model_id)
            for model_id in _configured_muse_code_model_ids()
        ],
    }


def _build_route_state() -> tuple[dict[str, object], dict[str, object]]:
    request_payload: dict[str, object] = {"model": "muse-code/catalog"}
    kwargs: dict[str, object] = {
        "litellm_params": {
            "metadata": {
                "custom_llm_provider": "muse_code",
                "route_family": MUSE_CODE_ROUTE_FAMILY,
            }
        }
    }
    return request_payload, kwargs


def _emit_route_context(
    *,
    request: Request,
    request_payload: dict[str, object],
    kwargs: dict[str, object],
) -> None:
    emit_aawm_route_access_log(
        request=request,
        target=MUSE_CODE_CATALOG_TARGET,
        request_body=request_payload,
        kwargs=kwargs,
        route_type="MUSE",
    )


def _sanitize_error_summary(detail: object, *, status_code: int) -> str:
    if isinstance(detail, bytes):
        detail_text = detail.decode("utf-8", errors="replace")
    elif isinstance(detail, str):
        detail_text = detail
    else:
        detail_text = ""
    normalized_summary = " ".join(detail_text.split())
    redacted_summary = _redact_string(normalized_summary)
    if not redacted_summary:
        return f"HTTP {status_code} request rejected"
    if len(redacted_summary) > _MUSE_CODE_GATEWAY_ERROR_SUMMARY_MAX_CHARS:
        return redacted_summary[: _MUSE_CODE_GATEWAY_ERROR_SUMMARY_MAX_CHARS - 3] + "..."
    return redacted_summary


def _proxy_exception_status_code(exc: ProxyException) -> int:
    try:
        return int(exc.code) if exc.code is not None else 401
    except (TypeError, ValueError):
        return 401


def _log_catalog_failure(
    *,
    request: Request,
    request_payload: dict[str, object],
    kwargs: dict[str, object],
    status_code: int,
    detail: object,
    failure_kind: str,
) -> None:
    _emit_route_context(
        request=request,
        request_payload=request_payload,
        kwargs=kwargs,
    )
    summary = _sanitize_error_summary(detail, status_code=status_code)
    record_aawm_route_rollup_failure(kwargs, message=summary)
    log_fn = (
        verbose_proxy_logger.warning
        if status_code < 500
        else verbose_proxy_logger.error
    )
    log_fn(
        "Muse Code catalog surfaced handled client/provider error status=%s error=%s",
        status_code,
        summary,
        extra={
            "source": "muse_code_gateway",
            "container": os.getenv("HOSTNAME"),
            "endpoint": request.url.path,
            "provider": "muse_code",
            "model": request_payload["model"],
            "model_alias": None,
            "route_family": MUSE_CODE_ROUTE_FAMILY,
            "status_code": status_code,
            "litellm_call_id": request.headers.get("x-litellm-call-id"),
            "failure_kind": failure_kind,
        },
        exc_info=False,
    )


def _raise_disabled_not_found() -> None:
    raise HTTPException(status_code=404, detail="Not Found")


async def _authenticate_muse_code_catalog(request: Request) -> UserAPIKeyAuth:
    if not is_muse_code_facade_enabled():
        _raise_disabled_not_found()

    register_aawm_route_rollup_access_log_replacement(request)
    request_payload, route_kwargs = _build_route_state()
    try:
        return await user_api_key_auth(
            request=request,
            api_key=_muse_code_virtual_key(request),
        )
    except HTTPException as exc:
        _log_catalog_failure(
            request=request,
            request_payload=request_payload,
            kwargs=route_kwargs,
            status_code=exc.status_code,
            detail=exc.detail,
            failure_kind="muse_code_authentication_rejected",
        )
        raise
    except ProxyException as exc:
        _log_catalog_failure(
            request=request,
            request_payload=request_payload,
            kwargs=route_kwargs,
            status_code=_proxy_exception_status_code(exc),
            detail=exc.message,
            failure_kind="muse_code_authentication_rejected",
        )
        raise


@router.get("/models")
async def get_muse_code_models(
    request: Request,
    _user_api_key_dict: UserAPIKeyAuth = Depends(_authenticate_muse_code_catalog),
) -> JSONResponse:
    request_payload, route_kwargs = _build_route_state()
    _emit_route_context(
        request=request,
        request_payload=request_payload,
        kwargs=route_kwargs,
    )
    record_aawm_route_rollup_turn(route_kwargs)
    return JSONResponse(content=_build_muse_code_catalog())
