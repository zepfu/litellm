"""OpenAI API known failure classifiers for pass-through requests."""

from __future__ import annotations

from typing import Optional

import httpx
from fastapi import status

from litellm.proxy.pass_through_endpoints.provider_failure_classifiers.common import (
    _coerce_upstream_error_payload,
    _extract_passthrough_exception_detail,
)

_OPENAI_MODEL_NOT_FOUND_ERROR_CODE = "model_not_found"
_OPENAI_MODEL_NOT_FOUND_ERROR_TYPE = "invalid_request_error"


def _is_openai_passthrough_target(
    *,
    url: Optional[httpx.URL],
    custom_llm_provider: Optional[str],
) -> bool:
    provider = str(custom_llm_provider or "").strip().lower()
    hostname = str(getattr(url, "host", "") or "").lower() if url is not None else ""
    return provider == "openai" or hostname == "api.openai.com"


def _get_openai_model_not_found_error_summary(exc: Exception) -> Optional[str]:
    detail = _extract_passthrough_exception_detail(exc)
    payload = _coerce_upstream_error_payload(detail)
    if not isinstance(payload, dict):
        return None
    error = payload.get("error")
    if not isinstance(error, dict):
        return None
    if str(error.get("code") or "").strip() != _OPENAI_MODEL_NOT_FOUND_ERROR_CODE:
        return None
    if str(error.get("type") or "").strip() != _OPENAI_MODEL_NOT_FOUND_ERROR_TYPE:
        return None
    message = " ".join(str(error.get("message") or "").split())
    if not message:
        return "requested model does not exist"
    return message[:512]


def _is_known_openai_model_not_found_response(
    *,
    url: Optional[httpx.URL],
    custom_llm_provider: Optional[str],
    status_code: Optional[int],
    exc: Exception,
) -> bool:
    if status_code not in {
        status.HTTP_400_BAD_REQUEST,
        status.HTTP_404_NOT_FOUND,
    }:
        return False
    if not _is_openai_passthrough_target(
        url=url,
        custom_llm_provider=custom_llm_provider,
    ):
        return False
    return _get_openai_model_not_found_error_summary(exc) is not None


def _get_openai_model_not_found_failure_kind() -> str:
    return "openai_model_not_found"
