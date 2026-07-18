"""Anthropic known failure classifiers for pass-through (RR-056 #3)."""

from __future__ import annotations

from typing import Any, Optional

import httpx
from fastapi import status

from litellm.proxy.pass_through_endpoints.provider_failure_classifiers.common import (
    _ANTHROPIC_CONTEXT_OVERFLOW_MARKERS,
    _ANTHROPIC_INVALID_AUTHENTICATION_MARKER,
    _ANTHROPIC_MODEL_NOT_FOUND_PREFIX,
    _coerce_upstream_error_payload,
    _extract_passthrough_exception_detail,
    _is_anthropic_passthrough_target,
)


def _is_known_anthropic_context_overflow_response(
    *,
    status_code: Optional[int],
    payload: dict[str, Any],
) -> bool:
    if status_code != status.HTTP_400_BAD_REQUEST:
        return False

    error = payload.get("error")
    if not isinstance(error, dict):
        return False

    error_type = str(error.get("type") or "").strip().lower()
    message = str(error.get("message") or "").strip().lower()
    if error_type != "invalid_request_error":
        return False

    return any(marker in message for marker in _ANTHROPIC_CONTEXT_OVERFLOW_MARKERS)


def _get_known_anthropic_passthrough_failure_kind(
    *,
    url: Optional[httpx.URL],
    custom_llm_provider: Optional[str],
    status_code: Optional[int],
    exc: Exception,
) -> Optional[str]:
    if status_code not in {
        status.HTTP_400_BAD_REQUEST,
        status.HTTP_401_UNAUTHORIZED,
        status.HTTP_404_NOT_FOUND,
    }:
        return None

    if not _is_anthropic_passthrough_target(
        url=url,
        custom_llm_provider=custom_llm_provider,
    ):
        return None

    detail = _extract_passthrough_exception_detail(exc)
    if detail is None:
        return None

    payload = _coerce_upstream_error_payload(detail)
    if not isinstance(payload, dict):
        return None
    error = payload.get("error")
    if not isinstance(error, dict):
        return None

    error_type = str(error.get("type") or "").strip().lower()
    message = str(error.get("message") or "").strip().lower()
    if _is_known_anthropic_context_overflow_response(
        status_code=status_code,
        payload=payload,
    ):
        return "anthropic_context_overflow"
    if (
        status_code == status.HTTP_401_UNAUTHORIZED
        and error_type == "authentication_error"
        and _ANTHROPIC_INVALID_AUTHENTICATION_MARKER in message
    ):
        return "anthropic_client_authentication_error"
    if (
        status_code == status.HTTP_404_NOT_FOUND
        and error_type == "not_found_error"
        and message.startswith(_ANTHROPIC_MODEL_NOT_FOUND_PREFIX)
    ):
        return "anthropic_model_not_found"
    return None
