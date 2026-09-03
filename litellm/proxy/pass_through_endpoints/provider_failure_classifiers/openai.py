"""OpenAI API known failure classifiers for pass-through requests."""

from __future__ import annotations

import re
from typing import Optional

import httpx
from fastapi import status

from litellm.proxy.pass_through_endpoints.provider_failure_classifiers.common import (
    _OPENAI_INVALID_ENCRYPTED_CONTENT_ERROR_CODE,
    _coerce_upstream_error_payload,
    _extract_passthrough_exception_detail,
)

_OPENAI_MODEL_NOT_FOUND_ERROR_CODE = "model_not_found"
_OPENAI_MODEL_NOT_FOUND_ERROR_TYPE = "invalid_request_error"
_OPENAI_MODEL_NOT_FOUND_MESSAGE_RE = re.compile(
    r"^The requested model '([^']+)' does not exist\.$"
)


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


def _get_openai_model_not_found_model(exc: Exception) -> Optional[str]:
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
    match = _OPENAI_MODEL_NOT_FOUND_MESSAGE_RE.fullmatch(message)
    return match.group(1).strip() if match is not None else None


def _is_known_openai_model_not_found_response(
    *,
    url: Optional[httpx.URL],
    custom_llm_provider: Optional[str],
    status_code: Optional[int],
    exc: Exception,
    expected_model: Optional[str] = None,
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
    normalized_expected_model = str(expected_model or "").strip()
    if not normalized_expected_model:
        return False
    return _get_openai_model_not_found_model(exc) == normalized_expected_model


def _get_openai_model_not_found_failure_kind() -> str:
    return "openai_model_not_found"


def _extract_openai_error_dict(exc: Exception) -> Optional[dict]:
    payload = getattr(exc, "error_payload", None)
    if isinstance(payload, dict):
        nested = payload.get("error")
        if isinstance(nested, dict):
            return nested
        if payload.get("code") or payload.get("message"):
            return payload
    detail = _extract_passthrough_exception_detail(exc)
    coerced = _coerce_upstream_error_payload(detail)
    if not isinstance(coerced, dict):
        response = getattr(exc, "response", None)
        content = getattr(response, "content", None) if response is not None else None
        coerced = _coerce_upstream_error_payload(content)
    if not isinstance(coerced, dict):
        return None
    error = coerced.get("error")
    return error if isinstance(error, dict) else coerced if coerced.get("code") else None


def _openai_error_is_invalid_encrypted_content(exc: Exception) -> bool:
    """True when structured error.code is invalid_encrypted_content."""
    error = _extract_openai_error_dict(exc)
    return (
        isinstance(error, dict)
        and str(error.get("code") or "").strip()
        == _OPENAI_INVALID_ENCRYPTED_CONTENT_ERROR_CODE
    )


def _is_openai_api_key_responses_target(
    *,
    url: Optional[httpx.URL],
    custom_llm_provider: Optional[str],
) -> bool:
    if str(custom_llm_provider or "").strip().lower() != "openai":
        return False
    if url is None:
        return False
    hostname = str(getattr(url, "host", "") or "").lower()
    if hostname != "api.openai.com":
        return False
    path = str(getattr(url, "path", "") or "").lower()
    return path == "/v1/responses" or path.startswith("/v1/responses/")


def _is_known_openai_invalid_encrypted_content_response(
    *,
    url: Optional[httpx.URL],
    custom_llm_provider: Optional[str],
    status_code: Optional[int],
    exc: Exception,
) -> bool:
    """True for OpenAI API-key Responses decrypt failures, including 502 wrappers."""
    if status_code not in {
        status.HTTP_400_BAD_REQUEST,
        status.HTTP_502_BAD_GATEWAY,
    }:
        return False
    if not _is_openai_api_key_responses_target(
        url=url,
        custom_llm_provider=custom_llm_provider,
    ):
        return False
    return _openai_error_is_invalid_encrypted_content(exc)


def _get_openai_invalid_encrypted_content_failure_kind() -> str:
    return "openai_invalid_encrypted_content"


def _get_openai_invalid_encrypted_content_error_summary(exc: Exception) -> Optional[str]:
    error = _extract_openai_error_dict(exc)
    if not isinstance(error, dict):
        return None
    message = " ".join(str(error.get("message") or "").split())
    if message:
        return message[:512]
    return "encrypted content could not be decrypted"
