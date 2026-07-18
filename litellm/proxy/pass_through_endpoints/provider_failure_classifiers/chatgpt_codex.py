"""ChatGPT Codex known failure classifiers for pass-through (RR-056 #3)."""

from __future__ import annotations

from typing import Optional
from urllib.parse import urlparse

import httpx
from fastapi import status

from litellm.proxy.pass_through_endpoints.provider_failure_classifiers.common import (
    _CHATGPT_CODEX_BLOCK_PAGE_MARKERS,
    _CHATGPT_CODEX_INVALID_ENCRYPTED_CONTENT_ERROR_CODE,
    _CHATGPT_CODEX_MODEL_NOT_SUPPORTED_FOR_ACCOUNT_MARKERS,
    _coerce_upstream_error_payload,
    _extract_passthrough_exception_detail,
)


def _is_known_chatgpt_codex_block_page_response(
    *,
    url: Optional[httpx.URL],
    status_code: Optional[int],
    exc: Exception,
) -> bool:
    if status_code != status.HTTP_403_FORBIDDEN:
        return False

    parsed_url = urlparse(str(url or ""))
    if (
        str(parsed_url.hostname or "").lower() != "chatgpt.com"
        or "/backend-api/codex/" not in str(parsed_url.path or "").lower()
    ):
        return False

    detail = _extract_passthrough_exception_detail(exc)
    if not detail:
        return False

    normalized_detail = str(detail).lower()
    return any(
        marker in normalized_detail for marker in _CHATGPT_CODEX_BLOCK_PAGE_MARKERS
    )


def _is_known_chatgpt_codex_invalid_encrypted_content_response(
    *,
    url: Optional[httpx.URL],
    status_code: Optional[int],
    exc: Exception,
) -> bool:
    if status_code != status.HTTP_400_BAD_REQUEST:
        return False

    parsed_url = urlparse(str(url or ""))
    if (
        str(parsed_url.hostname or "").lower() != "chatgpt.com"
        or "/backend-api/codex/" not in str(parsed_url.path or "").lower()
    ):
        return False

    detail = _extract_passthrough_exception_detail(exc)
    if detail is None:
        return False

    payload = _coerce_upstream_error_payload(detail)
    if not isinstance(payload, dict):
        return False

    error = payload.get("error")
    if not isinstance(error, dict):
        return False

    return (
        str(error.get("code") or "").strip()
        == _CHATGPT_CODEX_INVALID_ENCRYPTED_CONTENT_ERROR_CODE
    )


def _is_known_chatgpt_codex_model_not_supported_for_account_response(
    *,
    url: Optional[httpx.URL],
    status_code: Optional[int],
    exc: Exception,
) -> bool:
    if status_code != status.HTTP_400_BAD_REQUEST:
        return False

    parsed_url = urlparse(str(url or ""))
    if (
        str(parsed_url.hostname or "").lower() != "chatgpt.com"
        or "/backend-api/codex/" not in str(parsed_url.path or "").lower()
    ):
        return False

    detail = _extract_passthrough_exception_detail(exc)
    if detail is None:
        return False

    payload = _coerce_upstream_error_payload(detail)
    if isinstance(payload, dict):
        detail_text = str(payload.get("detail") or "")
    else:
        detail_text = str(detail)

    normalized_detail = detail_text.lower()
    if (
        _CHATGPT_CODEX_MODEL_NOT_SUPPORTED_FOR_ACCOUNT_MARKERS[0]
        not in normalized_detail
    ):
        return False
    return (
        _CHATGPT_CODEX_MODEL_NOT_SUPPORTED_FOR_ACCOUNT_MARKERS[1] in normalized_detail
        or "is not supported" in normalized_detail
    )


def _get_passthrough_chatgpt_codex_model_not_supported_failure_kind() -> str:
    return "openai_chatgpt_codex_model_not_supported_for_account"
